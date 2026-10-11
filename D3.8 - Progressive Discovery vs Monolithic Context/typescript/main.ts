/**
 * D3.8 - Progressive Discovery vs Monolithic Context (TypeScript port of python/main.py)
 *
 * 60 tools, 20 docs, 5 skills; one scripted 6-request session replayed under four strategies:
 *   monolithic : every tool schema + every document in every request (prefix cached)
 *   swap       : small core + search_tools; the client REBUILDS `tools` per request
 *   append     : small STABLE core (search_tools, call_tool, load_doc); discovered schemas are appended
 *                to the conversation, never swapped into `tools`
 *   skills     : Agent-Skills-style metadata in the system prompt; load_skill reveals SKILL.md on demand
 * Measures round trips, prompt tokens per call, cache writes/reads, cost (cached / uncached) and
 * tool-selection pressure. Server-side tool search (tool_search_tool_bm25 + defer_loading) is live-only.
 */

import type Anthropic from "@anthropic-ai/sdk";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MODEL_SONNET, PRICING, banner, costUsd, getClient, isLive, section, textBlock, textOf, toolUseBlock } from "../../shared/typescript/claudeClient.ts";
import type { Json, MockRequest, ResponderResult } from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
const MODEL = MODEL_SONNET; // high-volume enterprise assistant tier
const readJson = (file: string): Json => JSON.parse(readFileSync(path.join(DATA, file), "utf8")) as Json;

interface CatalogTool { domain: string; name: string; description: string; input_schema: Anthropic.Tool.InputSchema }
const CATALOG = readJson("tool_catalog.json").tools as CatalogTool[];
const API_TOOLS: Anthropic.Tool[] = CATALOG.map((t) => ({ name: t.name, description: t.description, input_schema: t.input_schema }));
const TOOL_BY_NAME = new Map(API_TOOLS.map((t) => [t.name, t] as [string, Anthropic.Tool]));
interface Doc { id: string; title: string; summary: string; body: string }
const DOCS = readJson("knowledge_docs.json").docs as Doc[];
const DOC_BY_ID = new Map(DOCS.map((d) => [d.id, d] as [string, Doc]));
interface SessionRequest {
  id: string; user: string; skill: string; search?: string; tool?: string; args?: Json; result?: Json;
  distractor?: string; distractor_result?: Json; doc?: string; answer: string;
}
const REQUESTS = readJson("session.json").requests as SessionRequest[];

const compact = (obj: unknown): string => JSON.stringify(obj);
/** Order-insensitive JSON for equality checks (Python dict == semantics). */
function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value as Json).sort().map((k) => `${JSON.stringify(k)}:${stableJson((value as Json)[k])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
const fmt = (n: number): string => n.toLocaleString("en-US");

// =============================================================================
// 1. Skills: progressive disclosure in three levels (metadata -> SKILL.md -> bundled files)
// =============================================================================
interface Skill { name: string; description: string; tools: string[]; docs: string[]; body: string; files: Map<string, string> }

function loadSkills(): Skill[] {
  const root = path.join(DATA, "skills");
  return readdirSync(root).sort().map((dir) => {
    const raw = readFileSync(path.join(root, dir, "SKILL.md"), "utf8").replace(/\r\n/g, "\n");
    const first = raw.indexOf("---\n");
    const second = raw.indexOf("---\n", first + 4);
    const front = raw.slice(first + 4, second);
    const body = raw.slice(second + 4);
    const meta: Record<string, string> = {};
    for (const line of front.trim().split("\n")) {
      const i = line.indexOf(":");
      meta[(i >= 0 ? line.slice(0, i) : line).trim()] = i >= 0 ? line.slice(i + 1).trim() : "";
    }
    const files = new Map<string, string>();
    for (const f of readdirSync(path.join(root, dir)).sort()) if (f !== "SKILL.md") files.set(f, readFileSync(path.join(root, dir, f), "utf8"));
    return {
      name: meta.name!, description: meta.description!, tools: meta.tools!.split(",").map((t) => t.trim()),
      docs: meta.docs!.split(",").map((x) => x.trim()), body: body.trim(), files,
    };
  });
}

const SKILLS = loadSkills();
const SKILL_BY_NAME = new Map(SKILLS.map((s) => [s.name, s] as [string, Skill]));

// =============================================================================
// 2. Local tool search (BM25 over name + description) - same tokenizer/formula as Python
// =============================================================================
const STOPWORDS = new Set(
  "a an and are as at be by can do does for from has have how i if in is it its of on or the their this to use when with you your".split(" "),
);

function tokenize(text: string): string[] {
  const out: string[] = [];
  for (let t of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (STOPWORDS.has(t)) continue;
    if (t.length > 3 && t.endsWith("s") && !t.endsWith("ss")) t = t.slice(0, -1);
    out.push(t);
  }
  return out;
}

class ToolSearch {
  readonly k1: number;
  readonly b: number;
  readonly docs = new Map<string, string[]>();
  readonly df = new Map<string, number>();
  readonly avgdl: number;

  constructor(tools: Anthropic.Tool[], k1 = 1.2, b = 0.75) {
    this.k1 = k1;
    this.b = b;
    let total = 0;
    for (const t of tools) {
      const toks = tokenize(t.name + " " + (t.description ?? ""));
      this.docs.set(t.name, toks);
      total += toks.length;
      for (const tok of new Set(toks)) this.df.set(tok, (this.df.get(tok) ?? 0) + 1);
    }
    this.avgdl = total / this.docs.size;
  }

  scores(query: string): [string, number][] {
    const terms = [...new Set(tokenize(query))];
    const n = this.docs.size;
    const out: [string, number][] = [];
    for (const [name, toks] of this.docs) {
      let s = 0.0;
      for (const t of terms) {
        const f = toks.filter((x) => x === t).length;
        if (f === 0) continue;
        const df = this.df.get(t)!;
        const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
        s += (idf * f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * toks.length) / this.avgdl));
      }
      if (s > 0) out.push([name, s]);
    }
    return out.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  }

  search(query: string, k = 3): string[] {
    return this.scores(query).slice(0, k).map(([name]) => name);
  }
}

const SEARCH = new ToolSearch(API_TOOLS);

// =============================================================================
// 3. Prompts and meta-tools for each strategy
// =============================================================================
const BASE_SYSTEM = `You are Northwind's enterprise assistant. You help employees with HR, payroll, expenses, procurement, IT, facilities, sales (CRM), legal and analytics requests by calling internal tools and reading internal policy documents.

Operating rules:
- Act only through tools. Never invent balances, booking IDs, request numbers, approvals or policy terms; if a tool or document does not give you the fact, say so and explain how the employee can get it.
- Prefer the most specific tool for the job. Several tools look alike (for example HR balances vs payroll accruals); read descriptions carefully and pick the one that answers the exact question asked.
- Resolve relative dates against today's date, which is 2026-10-11, and state absolute dates in answers.
- Before any action that changes state (bookings, requests, resets), make sure all required inputs are known. If something essential is missing, ask one short clarifying question instead of guessing.
- Policy answers must cite the document ID they come from, for example (DOC-04).
- Keep answers to one to three sentences unless the employee asks for detail. Plain business English, no internal jargon, no emojis.
- Never reveal secrets, credentials or other employees' personal data beyond what the requester is entitled to see; for payroll and HR data, only the employee, their manager or HR may see the details.
- Security incidents (lost devices, phishing) must be reported to the Security Operations Center within one hour; tell the employee this whenever it is relevant.
- If a request falls outside these domains, say so politely and suggest the right team.
- When a tool returns an error, read it, correct your input or choose a different approach; do not repeat the identical failing call.
- Money: state currency and amounts exactly as tools return them; do not round totals that drive approvals.
- People: refer to employees by full name as given; never guess an employee ID or username.
- Approvals: explain who must approve and why (threshold, policy) whenever you create a request that needs approval.
- Accessibility: if the employee mentions an accessibility need, offer the accessible option first (rooms, parking, formats).
- Escalation: for legal, medical, harassment or safety concerns, do not try to resolve them yourself; point to HR, Legal or Security.
- Audit: every state-changing action you take is logged with the employee's identity; never perform actions on behalf of someone else unless the request clearly comes from their manager or an administrator.`;

type Mode = "monolithic" | "swap" | "append" | "skills";

function systemFor(mode: Mode): string {
  if (mode === "monolithic") {
    const docs = DOCS.map((d) => `[${d.id}] ${d.title}\n${d.body}`).join("\n\n");
    return `${BASE_SYSTEM}\n\n# Knowledge base (all documents)\n${docs}`;
  }
  if (mode === "skills") {
    const meta = SKILLS.map((s) => `- ${s.name}: ${s.description}`).join("\n");
    return `${BASE_SYSTEM}\n\n# Skills (call load_skill to get full instructions before acting)\n${meta}`;
  }
  const index = DOCS.map((d) => `${d.id}  ${d.title} - ${d.summary}`).join("\n");
  return `${BASE_SYSTEM}\n\n# Document index (call load_doc for the full text)\n${index}`;
}

const SEARCH_TOOLS: Anthropic.Tool = {
  name: "search_tools",
  description: "Search the catalog of 60 enterprise tools by keywords; returns the best matching tools. " +
    "Call this before using any capability you do not already have.",
  input_schema: {
    type: "object",
    properties: { query: { type: "string", description: "What you need to do, in keywords" }, k: { type: "integer", description: "Maximum results (default 3)" } },
    required: ["query"],
  },
};
const LOAD_DOC: Anthropic.Tool = {
  name: "load_doc",
  description: "Load the full text of a knowledge document by ID from the document index. Use for policy questions.",
  input_schema: { type: "object", properties: { doc_id: { type: "string" } }, required: ["doc_id"] },
};
const CALL_TOOL: Anthropic.Tool = {
  name: "call_tool",
  description: "Invoke a catalog tool whose schema was disclosed earlier in this conversation (by search_tools " +
    "or load_skill). Pass its name and arguments matching that schema.",
  input_schema: { type: "object", properties: { name: { type: "string" }, arguments: { type: "object" } }, required: ["name", "arguments"] },
};
const LOAD_SKILL: Anthropic.Tool = {
  name: "load_skill",
  description: "Load a skill's full instructions (SKILL.md) plus the schemas of the tools it uses. Call it " +
    "before acting on a request that matches a skill description.",
  input_schema: { type: "object", properties: { name: { type: "string", enum: SKILLS.map((s) => s.name) } }, required: ["name"] },
};
const CORE: Record<Mode, Anthropic.Tool[]> = {
  monolithic: API_TOOLS,
  swap: [SEARCH_TOOLS, LOAD_DOC],
  append: [SEARCH_TOOLS, CALL_TOOL, LOAD_DOC],
  skills: [LOAD_SKILL, CALL_TOOL, LOAD_DOC],
};
const STRATEGY_LABEL: Record<Mode, string> = { monolithic: "monolithic", swap: "progressive-swap", append: "progressive-append", skills: "skills-style" };

// =============================================================================
// 4. Fake enterprise backends + the harness that runs one strategy over the session
// =============================================================================
function backend(name: string, args: Json): [string, boolean] {
  const schema = TOOL_BY_NAME.get(name)!.input_schema;
  const missing = (schema.required ?? []).filter((r) => !(r in args));
  if (missing.length) return [`missing required argument(s) for ${name}: ${missing.join(", ")}`, true];
  for (const r of REQUESTS) {
    if (r.tool === name && stableJson(r.args) === stableJson(args)) return [compact(r.result), false];
    if (r.distractor === name && stableJson(r.args) === stableJson(args)) return [compact(r.distractor_result), false];
  }
  return [compact({ ok: true, tool: name }), false];
}

interface CallStat { request: string; prompt: number; cacheWrite: number; cacheRead: number; output: number; cost: number; visibleTools: number }
const cacheEvent = (c: CallStat): string => (c.cacheWrite ? "W" : c.cacheRead ? "R" : "-");

class Runner {
  readonly client: Anthropic;
  readonly mode: Mode;
  readonly messages: Anthropic.MessageParam[] = [];
  readonly disclosed: string[] = []; // append/skills: schemas living in the conversation
  requestTools: string[] = []; // swap: tools rebuilt for the current request
  readonly calls: CallStat[] = [];
  readonly skillsLoaded: string[] = [];
  readonly docsLoaded: string[] = [];

  constructor(client: Anthropic, mode: Mode) {
    this.client = client;
    this.mode = mode;
  }

  tools(): Anthropic.Tool[] {
    if (this.mode === "swap") return [...CORE.swap, ...this.requestTools.map((n) => TOOL_BY_NAME.get(n)!)];
    return CORE[this.mode];
  }

  execute(name: string, args: Json): [string, boolean] {
    if (name === "search_tools") {
      const hits = SEARCH.search(String(args.query ?? ""), Number(args.k ?? 3));
      if (this.mode === "swap") { // schemas go into `tools` (prefix changes!), result lists names only
        for (const h of hits) if (!this.requestTools.includes(h)) this.requestTools.push(h);
        return [compact(hits.map((h) => ({ name: h, description: TOOL_BY_NAME.get(h)!.description }))), false];
      }
      for (const h of hits) if (!this.disclosed.includes(h)) this.disclosed.push(h); // schemas appended to conversation
      return [compact(hits.map((h) => TOOL_BY_NAME.get(h)!)), false];
    }
    if (name === "load_doc") {
      const doc = DOC_BY_ID.get(String(args.doc_id ?? ""));
      if (!doc) return [`unknown document ${args.doc_id}`, true];
      this.docsLoaded.push(doc.id);
      return [`[${doc.id}] ${doc.title}\n${doc.body}`, false];
    }
    if (name === "load_skill") {
      const skill = SKILL_BY_NAME.get(String(args.name ?? ""));
      if (!skill) return [`unknown skill ${args.name}`, true];
      this.skillsLoaded.push(skill.name);
      for (const t of skill.tools) if (!this.disclosed.includes(t)) this.disclosed.push(t);
      const files = skill.files.size ? [...skill.files.keys()].join(", ") : "none";
      return [`${skill.body}\n\nBundled files (read on demand): ${files}\nTool schemas: ${compact(skill.tools.map((t) => TOOL_BY_NAME.get(t)!))}`, false];
    }
    if (name === "call_tool") {
      const target = String(args.name ?? "");
      if (!this.disclosed.includes(target)) return [`tool '${target}' has not been disclosed yet - call search_tools or load_skill first`, true];
      return backend(target, (args.arguments ?? {}) as Json);
    }
    if (!this.tools().some((t) => t.name === name)) return [`tool '${name}' is not available in this request`, true];
    return backend(name, args);
  }

  async runRequest(req: SessionRequest): Promise<string> {
    this.messages.push({ role: "user", content: req.user });
    if (this.mode === "swap") this.requestTools = []; // rebuild the tool set from scratch for every request
    for (let i = 0; i < 8; i++) {
      const resp = await this.client.messages.create({
        model: MODEL, max_tokens: 1024,
        system: [{ type: "text", text: systemFor(this.mode), cache_control: { type: "ephemeral" } }],
        tools: this.tools(), messages: this.messages,
      });
      const u = resp.usage;
      const visible = this.tools().length + (this.mode === "append" || this.mode === "skills" ? this.disclosed.length : 0);
      this.calls.push({
        request: req.id, prompt: u.input_tokens + (u.cache_creation_input_tokens ?? 0) + (u.cache_read_input_tokens ?? 0),
        cacheWrite: u.cache_creation_input_tokens ?? 0, cacheRead: u.cache_read_input_tokens ?? 0,
        output: u.output_tokens, cost: costUsd(resp.model, u), visibleTools: visible,
      });
      this.messages.push({ role: "assistant", content: resp.content });
      const calls = resp.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use");
      if (resp.stop_reason !== "tool_use" || calls.length === 0) return textOf(resp).trim();
      const results: Anthropic.ToolResultBlockParam[] = calls.map((c) => {
        const [out, isError] = this.execute(c.name, c.input as Json);
        return { type: "tool_result", tool_use_id: c.id, content: out, ...(isError ? { is_error: true } : {}) };
      });
      this.messages.push({ role: "user", content: results });
    }
    return "(max turns)";
  }
}

// =============================================================================
// 5. Offline responder: plays the assistant for every strategy
// =============================================================================
type Call = [string, Json, string, boolean];

function conversation(req: MockRequest): [SessionRequest, Call[], Call[]] {
  const uses = new Map<string, [string, Json]>();
  const allCalls: Call[] = [];
  let turnStart = 0;
  let current = "";
  for (const m of req.messages) {
    if (m.role === "user" && typeof m.content === "string") {
      current = m.content;
      turnStart = allCalls.length;
      continue;
    }
    for (const b of Array.isArray(m.content) ? (m.content as Json[]) : []) {
      if (b.type === "tool_use") uses.set(b.id, [b.name, b.input]);
      else if (b.type === "tool_result" && uses.has(b.tool_use_id)) {
        const [name, inp] = uses.get(b.tool_use_id)!;
        const text = typeof b.content === "string" ? b.content : ((b.content ?? []) as Json[]).map((x) => String(x.text ?? "")).join("");
        allCalls.push([name, inp, text, Boolean(b.is_error)]);
      }
    }
  }
  return [REQUESTS.find((r) => r.user === current)!, allCalls.slice(turnStart), allCalls];
}

/** Language-neutral output-token estimate so the Python and TS ports report identical usage. */
const reply = (blocks: Json[]): ResponderResult => ({ content: blocks, usage: { output_tokens: Math.max(1, Math.floor(compact(blocks).length / 4)) } });

function answerOf(entry: SessionRequest): ResponderResult {
  const result = entry.result ?? {};
  return reply([textBlock(entry.answer.replace(/\{(\w+)\}/g, (_, k: string) => String(result[k])))]);
}

function responder(req: MockRequest): ResponderResult {
  const [entry, turn, allc] = conversation(req);
  const names = req.toolNames;
  const turnNames = turn.map((c) => c[0]);
  const { tool, doc } = entry;
  const calledTarget = turn.some((c) => c[0] === "call_tool" && c[1].name === tool) || (tool !== undefined && turnNames.includes(tool));
  const loadedDocs = allc.filter((c) => c[0] === "load_doc").map((c) => c[1].doc_id);
  const mode: Mode = names.includes("load_skill") ? "skills" : names.includes("call_tool") ? "append" : names.includes("search_tools") ? "swap" : "monolithic";

  if (mode === "monolithic") {
    if (tool && !calledTarget) {
      const distractor = entry.distractor;
      if (distractor && !turnNames.includes(distractor)) return reply([toolUseBlock(distractor, entry.args!)]); // simulated look-alike pick
      return reply([toolUseBlock(tool, entry.args!)]);
    }
    return answerOf(entry); // docs are already in the system prompt
  }

  if (mode === "swap") {
    if (tool && !names.includes(tool)) {
      const blocks = [toolUseBlock("search_tools", { query: entry.search })];
      if (doc && !turnNames.includes("load_doc")) blocks.push(toolUseBlock("load_doc", { doc_id: doc }));
      return reply(blocks);
    }
    if (tool && !calledTarget) return reply([toolUseBlock(tool, entry.args!)]);
    if (!tool && doc && !turnNames.includes("load_doc")) return reply([toolUseBlock("load_doc", { doc_id: doc })]);
    return answerOf(entry);
  }

  const disclosed: string[] = [];
  for (const c of allc) {
    if (c[0] === "search_tools") disclosed.push(...(JSON.parse(c[2]) as Json[]).map((t) => String(t.name)));
    else if (c[0] === "load_skill" && !c[3]) disclosed.push(...SKILL_BY_NAME.get(String(c[1].name))!.tools);
  }
  const blocks: Json[] = [];
  if (mode === "skills" && !allc.some((c) => c[0] === "load_skill" && c[1].name === entry.skill)) {
    return reply([toolUseBlock("load_skill", { name: entry.skill })]);
  }
  if (mode === "append" && tool && !disclosed.includes(tool)) {
    blocks.push(toolUseBlock("search_tools", { query: entry.search }));
    if (doc && !loadedDocs.includes(doc)) blocks.push(toolUseBlock("load_doc", { doc_id: doc }));
    return reply(blocks);
  }
  if (doc && !loadedDocs.includes(doc)) blocks.push(toolUseBlock("load_doc", { doc_id: doc }));
  if (tool && !calledTarget) blocks.push(toolUseBlock("call_tool", { name: tool, arguments: entry.args }));
  return blocks.length ? reply(blocks) : answerOf(entry);
}

// =============================================================================
// Demo
// =============================================================================
const TOOL_SEARCH_SNIPPET = `await client.messages.create({
  model: "claude-sonnet-5-5", max_tokens: 1024,
  tools: [
    { type: "tool_search_tool_bm25_20251119", name: "tool_search_tool_bm25" },
    LOAD_DOC,                                              // small always-loaded core
    ...CATALOG.map((t) => ({ ...t, defer_loading: true })),  // 60 tools known, NOT loaded
  ],
  messages: [{ role: "user", content: "Reset VPN access for user jdoe." }],
});`;

async function countTokens(client: Anthropic, system?: string, tools?: Anthropic.Tool[]): Promise<number> {
  const res = await client.messages.countTokens({
    model: MODEL, messages: [{ role: "user", content: "x" }], ...(system ? { system } : {}), ...(tools ? { tools } : {}),
  });
  return res.input_tokens;
}

banner("D3.8 - Progressive Discovery vs Monolithic Context");
const client = getClient(responder);

section("1. What could go into context");
const base = await countTokens(client);
const sizes: [string, number][] = [
  ["60 tool schemas", (await countTokens(client, undefined, API_TOOLS)) - base],
  ["20 documents, full text", (await countTokens(client, DOCS.map((d) => `[${d.id}] ${d.title}\n${d.body}`).join("\n\n"))) - base],
  ["document index (id + title + summary)", (await countTokens(client, DOCS.map((d) => `${d.id}  ${d.title} - ${d.summary}`).join("\n"))) - base],
  ["5 skill metadata lines", (await countTokens(client, SKILLS.map((s) => `- ${s.name}: ${s.description}`).join("\n"))) - base],
  ["5 SKILL.md bodies", (await countTokens(client, SKILLS.map((s) => s.body).join("\n\n"))) - base],
  ["core meta-tools (search_tools, call_tool, load_doc)", (await countTokens(client, undefined, CORE.append)) - base],
  ["base system prompt", (await countTokens(client, BASE_SYSTEM)) - base],
];
for (const [label, n] of sizes) console.log(`  ${label.padEnd(52)} ~${fmt(n).padStart(6)} tokens  (count_tokens)`);
const domains = [...new Set(CATALOG.map((t) => t.domain))].sort();
console.log(`  ${API_TOOLS.length} tools across ${domains.length} domains: ${domains.join(", ")}`);

section("2. Scripted session x 4 strategies (" + MODEL + ")");
console.log("Same 6 requests, same conversation, four ways of building context. Cells: API calls / prompt tokens.");
const runners = new Map<Mode, Runner>();
const finals = new Map<Mode, string[]>();
for (const mode of ["monolithic", "swap", "append", "skills"] as Mode[]) {
  const r = new Runner(client, mode);
  const answers: string[] = [];
  for (const q of REQUESTS) answers.push(await r.runRequest(q));
  finals.set(mode, answers);
  runners.set(mode, r);
}
console.log(`  ${"req".padEnd(5)}` + [...runners.keys()].map((m) => STRATEGY_LABEL[m].padStart(22)).join(""));
for (const q of REQUESTS) {
  const cells = [...runners.values()].map((r) => {
    const cs = r.calls.filter((c) => c.request === q.id);
    let tok = 0;
    for (const c of cs) tok += c.prompt;
    return `${cs.length} / ${fmt(tok)}`;
  });
  console.log(`  ${q.id.padEnd(5)}` + cells.map((c) => c.padStart(22)).join(""));
}
const same = [...finals.values()].every((a) => JSON.stringify(a) === JSON.stringify(finals.get("monolithic")));
console.log(`  answers identical across strategies: ${same ? "yes" : "no"}`);
console.log(`  e.g. R4 -> ${finals.get("append")![3]}`);

section("3. Totals: tokens, cost, round trips");
const p = PRICING[MODEL]!;
console.log(
  `  ${"strategy".padEnd(20)}${"calls".padStart(6)}${"prompt tok".padStart(12)}${"avg/call".padStart(10)}${"cached".padStart(8)}${"writes".padStart(8)}` +
    `${"cost".padStart(10)}${"no-cache cost".padStart(15)}`,
);
for (const [mode, r] of runners) {
  let prompt = 0, writes = 0, reads = 0, out = 0, cost = 0.0;
  for (const c of r.calls) {
    prompt += c.prompt;
    writes += c.cacheWrite;
    reads += c.cacheRead;
    out += c.output;
    cost += c.cost;
  }
  const noCache = (prompt * p.input + out * p.output) / 1e6;
  console.log(
    `  ${STRATEGY_LABEL[mode].padEnd(20)}${String(r.calls.length).padStart(6)}${fmt(prompt).padStart(12)}${fmt(Math.floor(prompt / r.calls.length)).padStart(10)}` +
      `${((100 * reads) / prompt).toFixed(0).padStart(7)}%${fmt(writes).padStart(8)}${("$" + cost.toFixed(4)).padStart(10)}${("$" + noCache.toFixed(4)).padStart(15)}`,
  );
}
const mono = runners.get("monolithic")!;
let monoPrompt = 0;
for (const c of mono.calls) monoPrompt += c.prompt;
const projected = Math.floor(monoPrompt / mono.calls.length) + 4 * sizes[0]![1];
console.log(`  Scale check: with 5x the tools (~300, e.g. several MCP servers) monolithic grows to ~${fmt(projected)} tokens`);
console.log("  per call; progressive per-call context stays roughly flat because only matches are disclosed.");
console.log("  Prompt caching makes the monolithic prefix cheap to RE-read, but it still occupies the context");
console.log("  window on every call (crowding out conversation and retrieved data), raises time-to-first-token,");
console.log("  and every cache miss (TTL expiry, any prefix edit) re-bills the whole thing.");
console.log("  (Offline mock: only the tools+system prefix is cached; with message caching the conversation");
console.log("  tail would be cheaper too - the per-call context size is the robust signal here.)");

section("4. Cache behaviour per call (W = cache write, R = cache read)");
for (const [mode, r] of runners) {
  const groups = REQUESTS.map((q) => r.calls.filter((c) => c.request === q.id).map(cacheEvent).join(" "));
  console.log(`  ${STRATEGY_LABEL[mode].padEnd(20)}` + groups.join(" | "));
}
console.log("  swap rebuilds `tools` per request: tools render first, so every new tool set is a new prefix");
console.log("  (a cache WRITE) - the cache only helps when an identical set happens to recur.");
console.log("  append keeps `tools` + system byte-identical and adds discovered schemas to the conversation,");
console.log("  so after the first call every request reads the cached prefix. Server-side tool search works");
console.log("  the same way: discovered definitions are appended inline, not swapped into the prefix.");

section("5. Tool-selection pressure");
console.log("  Tool definitions visible when the model chooses (max per request; append/skills count disclosed schemas):");
console.log(`  ${"req".padEnd(5)}` + [...runners.keys()].map((m) => STRATEGY_LABEL[m].padStart(20)).join("") + "   search_tools top-3");
for (const q of REQUESTS) {
  const vis = [...runners.values()].map((r) => Math.max(...r.calls.filter((c) => c.request === q.id).map((c) => c.visibleTools)));
  const top = q.search ? SEARCH.search(q.search).join(", ") : "(docs only)";
  console.log(`  ${q.id.padEnd(5)}` + vis.map((v) => String(v).padStart(20)).join("") + `   ${top}`);
}
console.log("  Look-alikes such as hr_get_pto_balance vs payroll_get_leave_accrual ('leave days for an employee')");
console.log("  compete only when both are in view. In the monolithic run R1 picked the payroll tool first and");
console.log("  paid an extra round trip (simulated, but a typical large-catalog error - measure it with evals).");
console.log("  Note the flip side: append/skills accumulate disclosed schemas in history (trim per D2.4).");

section("6. Skills: three levels of progressive disclosure");
const sk = runners.get("skills")!;
const [metaTokens, bodyTokens] = [sizes[3]![1], sizes[4]![1]];
console.log(`  level 1  metadata for ${SKILLS.length} skills, always in the system prompt : ~${fmt(metaTokens)} tokens`);
console.log(
  `  level 2  SKILL.md bodies loaded on demand              : ${sk.skillsLoaded.length} loads ` +
    `(${[...new Set(sk.skillsLoaded)].join(", ")}) of ~${fmt(bodyTokens)} tokens total`,
);
const never = SKILLS.flatMap((s) => [...s.files.keys()].map((f) => `${s.name}/${f}`));
console.log(`  level 3  bundled files, read only when needed           : ${never.join(", ")} -> never loaded (0 tokens)`);
console.log("  On the Claude API, Agent Skills run in the code-execution container (see D2.5); the disclosure");
console.log("  pattern is the same: descriptions decide WHEN a skill loads, so write them like tool descriptions.");

section("7. Server-side tool search (live-only)");
console.log("The production version of progressive discovery: declare all tools, defer most, let Claude search.");
for (const line of TOOL_SEARCH_SNIPPET.split("\n")) console.log("    " + line);
if (isLive()) {
  const resp = await client.messages.create({
    model: MODEL, max_tokens: 1024,
    tools: [{ type: "tool_search_tool_bm25_20251119", name: "tool_search_tool_bm25" }, LOAD_DOC, ...API_TOOLS.map((t) => ({ ...t, defer_loading: true }))],
    messages: [{ role: "user", content: "Reset VPN access for user jdoe." }],
  });
  console.log(`  live: stop_reason=${resp.stop_reason}, blocks=${JSON.stringify(resp.content.map((b) => b.type))}, input_tokens=${resp.usage.input_tokens}`);
} else {
  console.log("  [live-only] the mock cannot run server tools. Never defer ALL tools (keep a small core loaded);");
  console.log("  discovered definitions are appended inline, so the cached prefix survives. Regex variant:");
  console.log("  tool_search_tool_regex_20251119. Rule of thumb: worth it once schemas exceed roughly 10K tokens.");
}

section("8. Recommendation");
const recs: [string, string][] = [
  ["Few tools (< ~15), small stable docs, one domain", "Monolithic + prompt caching"],
  ["Latency-critical, 1-2 calls per request", "Monolithic (or preloaded core set)"],
  ["Dozens-hundreds of tools / many MCP servers", "Server-side tool search, defer_loading"],
  ["Large per-task instructions and playbooks", "Skills (metadata -> SKILL.md -> files)"],
  ["Large or changing knowledge base", "load_doc / RAG on demand (D3.5)"],
  ["Dynamic tool sets mid-session", "Append-only (tool search / tool_addition),"],
  ["", "never swap the tools array"],
];
for (const [situation, choice] of recs) console.log(`  ${situation.padEnd(50)} ${choice}`);
