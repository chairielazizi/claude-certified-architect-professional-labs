/**
 * D2.4 - Context Window Optimization and Token Management
 * =======================================================
 *
 * An IT-helpdesk copilot turn that naively stuffs everything into the prompt
 * (12 KB articles with export boilerplate, a 22-message history, a verbose
 * telemetry dump) is brought under a context budget step by step:
 *
 *   1. Pre-flight: client.messages.countTokens on the naive request + per-part breakdown.
 *   2. Context budget allocator (water-filling with spill-over) across
 *      retrieved docs / history / tool results, after reserving system + output.
 *   3. Lossless-ish cleanup: strip doc boilerplate, trim the verbose tool result.
 *   4. Relevance filtering: BM25 ranks KB articles; keep the relevant ones that fit.
 *   5. History strategies compared on tokens AND fact recall:
 *      sliding window | summarize older turns (Haiku) | pinned facts + window.
 *   6. Assemble, verify with countTokens, and call the model: naive vs
 *      sliding-window vs optimized answers (lost facts show up in the answer).
 *   7. Server-side alternatives (live-only request code): context editing
 *      (clear_tool_uses_20250919) and compaction (compact_20260112) vs memory.
 */

import type Anthropic from "@anthropic-ai/sdk";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  MODEL_HAIKU,
  MODEL_OPUS,
  MODEL_SONNET,
  PRICING,
  type MockRequest,
  banner,
  costUsd,
  getClient,
  isLive,
  section,
  textOf,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
const readJson = <T>(name: string): T => JSON.parse(readFileSync(path.join(DATA, name), "utf-8")) as T;

interface Doc { id: string; title: string; body: string }
interface Turn { role: "user" | "assistant"; content: string }
interface Conversation { system: string; critical_facts: string[]; turns: Turn[]; question: string }
interface TelemetryEvent { [k: string]: string | number; ts: string; level: string; source: string; message: string }
interface Telemetry { tool: string; device: string; collected_at: string; agent_version: string; events: TelemetryEvent[] }

const KB = readJson<{ docs: Doc[] }>("kb_docs.json").docs;
const CONVO = readJson<Conversation>("conversation.json");
const TELEMETRY = readJson<Telemetry>("telemetry_tool_result.json");

const MODEL = MODEL_SONNET;
const CONTEXT_BUDGET = 2600; // tokens we allow per request for this route (cost/latency/quality target)
const OUTPUT_RESERVE = 700; // = max_tokens; reserve it up front so the answer is never squeezed out
const SAFETY_MARGIN = 0.9; // local estimates pack to 90% of a bucket; countTokens verifies the final request

type Msg = Anthropic.MessageParam;

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------
/**
 * countTokens is authoritative (model-specific tokenizer); a local chars/4
 * estimate is only for inner packing loops. Never use tiktoken for Claude.
 */
class TokenCounter {
  readonly client: Anthropic;
  calls = 0;
  constructor(client: Anthropic) {
    this.client = client;
  }
  async request(messages: Msg[], system?: string): Promise<number> {
    this.calls += 1;
    const r = await this.client.messages.countTokens({ model: MODEL, messages, ...(system ? { system } : {}) });
    return r.input_tokens;
  }
  /** Tokens a piece of text adds to a request (minus the fixed message overhead). Baseline is "." - empty text is rejected. */
  async text(text: string): Promise<number> {
    return (await this.request([{ role: "user", content: text }])) - (await this.request([{ role: "user", content: "." }]));
  }
}

const est = (text: string): number => Math.floor(text.length / 4) + 1;

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderDocs(docs: Doc[]): string {
  const inner = docs.map((d) => `<doc id="${d.id}" title="${d.title}">\n${d.body}\n</doc>`).join("\n");
  return `<kb>\n${inner}\n</kb>`;
}

function renderTelemetry(obj: unknown, pretty: boolean): string {
  const body = pretty ? JSON.stringify(obj, null, 2) : JSON.stringify(obj);
  return `<telemetry>\n${body}\n</telemetry>`;
}

function finalTurn(question: string, docsBlock = "", telemetryBlock = "", pinned = ""): Msg {
  const parts = [pinned, docsBlock, telemetryBlock].filter(Boolean);
  parts.push(`<question>${question}</question>`);
  return { role: "user", content: parts.join("\n\n") };
}

// ---------------------------------------------------------------------------
// 2. Budget allocator: weighted water-filling with spill-over
// ---------------------------------------------------------------------------
interface Bucket { name: string; demand: number; weight: number }

/**
 * Give each bucket a weight-proportional share, capped at its demand; whatever a
 * satisfied bucket doesn't need spills over to the others (repeat until stable).
 */
function allocate(available: number, buckets: Bucket[]): Record<string, number> {
  const alloc: Record<string, number> = Object.fromEntries(buckets.map((b) => [b.name, 0]));
  let active = buckets.filter((b) => b.demand > 0);
  let remaining = available;
  while (active.length && remaining > 0) {
    const totalW = active.reduce((s, b) => s + b.weight, 0);
    const satisfied = active.filter((b) => alloc[b.name]! + Math.floor((remaining * b.weight) / totalW) >= b.demand);
    if (!satisfied.length) {
      // nobody can be fully served: split what's left by weight and stop
      for (const b of active) alloc[b.name]! += Math.floor((remaining * b.weight) / totalW);
      break;
    }
    for (const b of satisfied) {
      remaining -= b.demand - alloc[b.name]!;
      alloc[b.name] = b.demand;
    }
    active = active.filter((b) => !satisfied.includes(b));
  }
  return alloc;
}

// ---------------------------------------------------------------------------
// 3. Cleanup: boilerplate + tool-result trimming
// ---------------------------------------------------------------------------
const BOILERPLATE = [/^ORBIT IT KNOWLEDGE BASE.*$/, /^Document owner:.*$/, /^-{4,}$/, /^This document is uncontrolled.*$/, /^\(c\) \d{4} Orbit.*$/];

function stripBoilerplate(body: string): string {
  return body.split("\n").filter((line) => !BOILERPLATE.some((p) => p.test(line))).join("\n").trim();
}

/**
 * Project to the fields the model needs, keep only relevant/severe events, cap the
 * count, and say what was dropped so the model knows the data is partial.
 */
function trimToolResult(result: Telemetry, keepSources: Set<string>, minLevel: string, fields: string[], maxItems: number) {
  const rank: Record<string, number> = { Information: 0, Warning: 1, Error: 2 };
  const events = result.events;
  let kept = events.filter((e) => keepSources.has(e.source) || rank[e.level]! >= rank[minLevel]!);
  kept = kept.slice(-maxItems); // most recent events matter most
  const errors: Record<string, number> = {};
  for (const e of events) if (e.level === "Error") errors[e.message] = (errors[e.message] ?? 0) + 1;
  return {
    device: result.device,
    collected_at: result.collected_at,
    summary: { total_events: events.length, returned: kept.length, errors_by_message: errors },
    events: kept.map((e) => Object.fromEntries(fields.map((k) => [k, e[k]]))),
  };
}

// ---------------------------------------------------------------------------
// 4. Relevance: BM25 over KB articles
// ---------------------------------------------------------------------------
const STOPWORDS = new Set(
  ("a an and are as at be but by can do does for from has have i if in is it its me my of on or so " +
    "that the this to was what when with you your should try next").split(" "),
);

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+(?:\.[0-9]+)*/g) ?? []).filter((w) => !STOPWORDS.has(w) && w.length > 1);
}

function countOf(items: string[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const it of items) m.set(it, (m.get(it) ?? 0) + 1);
  return m;
}

function bm25(query: string, texts: string[], k1 = 1.5, b = 0.75): number[] {
  const toks = texts.map(tokenize);
  const n = toks.length;
  const avgdl = toks.reduce((s, t) => s + t.length, 0) / n;
  const df = countOf(toks.flatMap((t) => [...new Set(t)]));
  const qTerms = [...new Set(tokenize(query))].sort();
  return toks.map((t) => {
    const tf = countOf(t);
    let s = 0;
    for (const w of qTerms) {
      const f = tf.get(w);
      if (f) {
        const d = df.get(w)!;
        const idf = Math.log((n - d + 0.5) / (d + 0.5) + 1);
        s += (idf * f * (k1 + 1)) / (f + k1 * (1 - b + (b * t.length) / avgdl));
      }
    }
    return s;
  });
}

// ---------------------------------------------------------------------------
// 5. History strategies
// ---------------------------------------------------------------------------
const FACT_PATTERNS: [string, RegExp][] = [
  ["asset_tag", /\bORB-\d{5}\b/],
  ["ticket", /\bINC-\d{5}\b/],
  ["os_version", /\bWindows 11 \d{2}H\d\b/],
  ["vpn_client", /\bGlobalProtect \d+\.\d+\.\d+\b/],
  ["already_tried", /\balready tried ([^.,]+)/],
  ["deadline", /\b((?:fly|flying|travel(?:ling)?) to [A-Z][a-z]+[^.,]*)/],
];

/**
 * Deterministic fact pinning: the latest match per fact wins. Cheap, auditable,
 * and immune to a summarizer deciding a detail is unimportant.
 */
function extractFacts(turns: Turn[]): Record<string, string> {
  const facts: Record<string, string> = {};
  for (const t of turns) {
    for (const [name, re] of FACT_PATTERNS) {
      const m = t.content.match(re);
      if (m) facts[name] = m.length > 1 && m[1] !== undefined ? m[1] : m[0];
    }
  }
  return facts;
}

const pinnedBlock = (facts: Record<string, string>): string =>
  "<pinned_facts>\n" + Object.entries(facts).map(([k, v]) => `- ${k}: ${v}`).join("\n") + "\n</pinned_facts>";

function slidingWindow(turns: Turn[], budget: number): Turn[] {
  const kept: Turn[] = [];
  let used = 0;
  for (const t of [...turns].reverse()) {
    const cost = est(t.content) + 8; // + per-message overhead
    if (used + cost > budget * SAFETY_MARGIN) break;
    kept.unshift(t);
    used += cost;
  }
  while (kept.length && kept[0]!.role !== "user") kept.shift(); // a conversation must start with a user turn
  return kept;
}

function recall(text: string, facts: string[]): string[] {
  const low = text.toLowerCase();
  return facts.filter((f) => low.includes(f.toLowerCase()));
}

const SUMMARIZER_SYSTEM =
  "You are a conversation summarizer for an IT helpdesk. Summarize the transcript in <transcript> into a " +
  "compact running summary (max 80 words). Preserve identifiers verbatim: asset tags, ticket numbers, versions.";

async function summarizeOlder(client: Anthropic, turns: Turn[]): Promise<[string, Anthropic.Message]> {
  const transcript = turns.map((t) => `${t.role}: ${t.content}`).join("\n");
  const msg = await client.messages.create({
    model: MODEL_HAIKU, max_tokens: 300, system: SUMMARIZER_SYSTEM, output_config: { effort: "low" },
    messages: [{ role: "user", content: `<transcript>\n${transcript}\n</transcript>` }],
  });
  return [textOf(msg), msg];
}

// ---------------------------------------------------------------------------
// Offline responder
// ---------------------------------------------------------------------------
const SUMMARY_TEXT =
  "Dana (Finance), laptop ORB-55812 (ThinkPad X1 Carbon Gen 11). Outlook shared calendar fixed via repair " +
  "(KB-106). VPN drops about every 10 minutes on home and hotel Wi-Fi, also during Teams calls while plugged in; " +
  "ticket INC-48213 open since yesterday with no callback. Windows 11 24H2, GlobalProtect 6.2.1. Expense tool " +
  "slowness is a side effect of the VPN drops.";

function responder(req: MockRequest): string {
  if (req.systemText.includes("conversation summarizer")) return SUMMARY_TEXT; // note: it silently dropped "already tried reinstalling"
  const text = req.allText;
  const has = (s: string): boolean => text.includes(s);
  const device = has("ORB-55812") && has("24H2") ? "ORB-55812 on Windows 11 24H2" : "your laptop";
  const version = has("GlobalProtect 6.2.1") ? "GlobalProtect 6.2.1 to 6.2.4+" : "GlobalProtect to 6.2.4+";
  const steps: string[] = [];
  if (has("KB-101")) {
    steps.push(
      `1. Upgrade ${version} from Company Portal (KB-101). ${device} matches the known ` +
        "24H2 bug, and telemetry shows ROUTE_TABLE_RESET right after each DHCP renewal.",
    );
  } else steps.push("1. Check your VPN client version against the approved list.");
  steps.push(
    has("INC-48213")
      ? "2. If 6.2.4 is not listed yet, IT can push it via Intune in about 15 minutes; I have added this to INC-48213."
      : "2. If 6.2.4 is not listed yet, IT can push it via Intune; I will open a new ticket for that.",
  );
  steps.push(
    has("reinstall")
      ? "3. Skip another same-version reinstall - it cannot fix this bug (KB-110)."
      : "3. Try reinstalling GlobalProtect from Company Portal (KB-110).",
  );
  if (has("KB-104")) steps.push("4. MTU (KB-104) is unlikely to be the main cause because you also drop on home Wi-Fi; 6.2.4 handles MTU anyway.");
  const trip = has("Singapore") ? " before your Singapore trip on Monday" : "";
  steps.push(`5. Re-test on home Wi-Fi for 30 minutes after the upgrade${trip}.`);
  return steps.join("\n");
}

// ---------------------------------------------------------------------------
// 7. Server-side context management (live-only)
// ---------------------------------------------------------------------------
const CONTEXT_EDITING_SNIPPET = `\
const resp = await client.beta.messages.create({
  betas: ["context-management-2025-06-27"],
  model: "claude-opus-5-5", max_tokens: 1024, tools, messages,
  context_management: { edits: [{
    type: "clear_tool_uses_20250919",
    trigger: { type: "input_tokens", value: 30000 },       // start clearing above this
    keep: { type: "tool_uses", value: 3 },                 // keep the 3 most recent tool results
    clear_at_least: { type: "input_tokens", value: 5000 }, // clear in big batches (cache-friendly)
    exclude_tools: ["get_user_profile"],                   // never clear these
  }] },
});
console.log(resp.context_management); // which edits were applied, tokens cleared`;

const COMPACTION_SNIPPET = `\
const resp = await client.beta.messages.create({
  betas: ["compact-2026-01-12"],
  model: "claude-opus-5-5", max_tokens: 16000, messages,
  context_management: { edits: [{
    type: "compact_20260112",
    trigger: { type: "input_tokens", value: 150000 },      // default trigger
    instructions: "Preserve asset tags, ticket ids, versions and deadlines verbatim.",
  }] },
});
messages.push({ role: "assistant", content: resp.content }); // keep the compaction block!`;

async function runServerSide(client: Anthropic): Promise<void> {
  const short: Anthropic.Beta.BetaMessageParam[] = [{ role: "user", content: "My VPN drops every ten minutes on Windows 11 24H2. Ideas?" }];
  const tool: Anthropic.Beta.BetaTool = {
    name: "get_device_telemetry",
    description: "Fetch recent device event logs.",
    input_schema: { type: "object", properties: { device: { type: "string" } }, required: ["device"] },
  };
  const r1 = await client.beta.messages.create({
    betas: ["context-management-2025-06-27"], model: MODEL_OPUS, max_tokens: 1024, tools: [tool], messages: short,
    context_management: {
      edits: [{
        type: "clear_tool_uses_20250919",
        trigger: { type: "input_tokens", value: 30000 },
        keep: { type: "tool_uses", value: 3 },
        clear_at_least: { type: "input_tokens", value: 5000 },
        exclude_tools: ["get_user_profile"],
      }],
    },
  });
  console.log(`  live context editing -> stop_reason=${r1.stop_reason} context_management=${JSON.stringify(r1.context_management)}`);
  const r2 = await client.beta.messages.create({
    betas: ["compact-2026-01-12"], model: MODEL_OPUS, max_tokens: 4000, messages: short,
    context_management: {
      edits: [{
        type: "compact_20260112",
        trigger: { type: "input_tokens", value: 150000 },
        instructions: "Preserve asset tags, ticket ids, versions and deadlines verbatim.",
      }],
    },
  });
  console.log(`  live compaction      -> stop_reason=${r2.stop_reason} block types=${JSON.stringify(r2.content.map((b) => b.type))}`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
function table(headers: string[], rows: string[][]): void {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i]!.length)));
  console.log("  " + headers.map((h, i) => h.padEnd(widths[i]!)).join("  "));
  console.log("  " + widths.map((w) => "-".repeat(w)).join("  "));
  for (const r of rows) console.log("  " + r.map((c, i) => c.padEnd(widths[i]!)).join("  "));
}
const num = (x: number, d = 0): string => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

banner("D2.4 - Context Window Optimization and Token Management");
const client = getClient(responder);
const tc = new TokenCounter(client);
const system = CONVO.system;
const turns = CONVO.turns;
const question = CONVO.question;
const critical = CONVO.critical_facts;

// ------------------------------------------------------------------ 1
section("1. Pre-flight: count_tokens on the naive request");
const rawDocs = renderDocs(KB);
const rawTel = renderTelemetry(TELEMETRY, true);
const naiveMsgs: Msg[] = [...turns, finalTurn(question, rawDocs, rawTel)];
const naiveTotal = await tc.request(naiveMsgs, system);
const parts: Record<string, number> = {
  "system prompt": await tc.text(system),
  "KB articles (12, raw)": await tc.text(rawDocs),
  "history (22 msgs)": (await tc.request([...turns, finalTurn(question)])) - (await tc.request([finalTurn(question)])),
  "telemetry (raw JSON)": await tc.text(rawTel),
  question: await tc.text(question),
};
for (const [name, n] of Object.entries(parts)) {
  console.log(`  ${name.padEnd(24)} ${num(n).padStart(6)} tok  ${"#".repeat(Math.max(1, Math.floor(n / 100)))}`);
}
const inputBudget = CONTEXT_BUDGET - OUTPUT_RESERVE;
console.log(
  `  ${"TOTAL (count_tokens)".padEnd(24)} ${num(naiveTotal).padStart(6)} tok   budget ${num(CONTEXT_BUDGET)} = ${num(inputBudget)} input + ${OUTPUT_RESERVE} output reserve`,
);
console.log(
  `  -> ${naiveTotal > inputBudget ? "OVER" : "within"} budget by ${num(Math.abs(naiveTotal - inputBudget))} tokens. ` +
    "A 1M window would accept it, but every token costs money, latency and attention.",
);

// ------------------------------------------------------------------ 2
section("2. Budget allocator (raw demands)");
const fixed = parts["system prompt"]! + parts["question"]! + 30; // + framing tags
const available = inputBudget - fixed;
const weights = { docs: 0.5, history: 0.3, tools: 0.2 };
{
  const rawBuckets: Bucket[] = [
    { name: "docs", demand: parts["KB articles (12, raw)"]!, weight: weights.docs },
    { name: "history", demand: parts["history (22 msgs)"]!, weight: weights.history },
    { name: "tools", demand: parts["telemetry (raw JSON)"]!, weight: weights.tools },
  ];
  const allocRaw = allocate(available, rawBuckets);
  console.log(`  available for variable context = ${num(inputBudget)} - fixed ${fixed} (system+question+framing) = ${num(available)}`);
  table(
    ["bucket", "weight", "demand", "allocated", "fits?"],
    rawBuckets.map((b) => [b.name, b.weight.toFixed(1), num(b.demand), num(allocRaw[b.name]!), allocRaw[b.name]! >= b.demand ? "yes" : "NO"]),
  );
  console.log("  Everything is squeezed: clean the inputs before deciding what to drop.");
}

// ------------------------------------------------------------------ 3
section("3. Cleanup: strip boilerplate, trim the verbose tool result");
const cleanKb: Doc[] = KB.map((d) => ({ ...d, body: stripBoilerplate(d.body) }));
const cleanDocsTok = await tc.text(renderDocs(cleanKb));
console.log(`  KB boilerplate stripped     : ${num(parts["KB articles (12, raw)"]!).padStart(5)} -> ${num(cleanDocsTok).padStart(5)} tok`);
const keepFields = ["ts", "source", "level", "message"];
const trimmed = trimToolResult(TELEMETRY, new Set(["vpn"]), "Warning", keepFields, 6);
const telBlock = renderTelemetry(trimmed, false);
const telTok = await tc.text(telBlock);
console.log(
  `  telemetry trimmed           : ${num(parts["telemetry (raw JSON)"]!).padStart(5)} -> ${num(telTok).padStart(5)} tok ` +
    `(${trimmed.summary.returned}/${trimmed.summary.total_events} events, ` +
    `${keepFields.length} of ${Object.keys(TELEMETRY.events[0]!).length} fields, compact JSON + summary)`,
);
const buckets: Bucket[] = [
  { name: "docs", demand: cleanDocsTok, weight: weights.docs },
  { name: "history", demand: parts["history (22 msgs)"]!, weight: weights.history },
  { name: "tools", demand: telTok, weight: weights.tools },
];
const alloc = allocate(available, buckets);
console.log("  Re-allocation on cleaned demands (tools' unused share spills over to docs/history):");
table(
  ["bucket", "demand", "allocated", "fits?"],
  buckets.map((b) => [b.name, num(b.demand), num(alloc[b.name]!), alloc[b.name]! >= b.demand ? "yes" : "NO"]),
);

// ------------------------------------------------------------------ 4
section("4. Relevance filtering: BM25 rank, keep what is relevant AND fits");
const facts = extractFacts(turns);
const query = question + " " + Object.values(facts).join(" ");
const scores = bm25(query, cleanKb.map((d) => d.title + "\n" + d.body));
const ranked = cleanKb.map((d, i) => ({ d, s: scores[i]! })).sort((a, b) => b.s - a.s || (a.d.id < b.d.id ? -1 : 1));
const top = ranked[0]!.s;
const keptDocs: Doc[] = [];
{
  let used = 0;
  const rows: string[][] = [];
  for (const { d, s } of ranked) {
    const cost = est(renderDocs([d]));
    let verdict: string;
    if (s < 0.35 * top) verdict = "drop (low relevance)";
    else if (used + cost > alloc["docs"]! * SAFETY_MARGIN) verdict = "drop (over budget)";
    else {
      keptDocs.push(d);
      used += cost;
      verdict = "KEEP";
    }
    rows.push([d.id, s.toFixed(2).padStart(5), `~${cost}`, verdict, d.title.slice(0, 52)]);
  }
  table(["doc", "bm25", "tok", "decision", "title"], rows);
}
const docsBlock = renderDocs(keptDocs);
const docsTok = await tc.text(docsBlock);
console.log(
  `  KB context: ${num(parts["KB articles (12, raw)"]!)} raw -> ${num(cleanDocsTok)} clean -> ${num(docsTok)} tok (${keptDocs.length} of ${KB.length} articles)`,
);

// ------------------------------------------------------------------ 5
section("5. History strategies: tokens vs fact recall");
const budgetH = alloc["history"]!;
const qOnly = await tc.request([finalTurn(question)]);
const fullTok = parts["history (22 msgs)"]!;
const window = slidingWindow(turns, budgetH);
const keepLast = 4;
const [summary, smsg] = await summarizeOlder(client, turns.slice(0, -keepLast));
const summaryMsgs: Msg[] = [{ role: "user", content: `<conversation_summary>\n${summary}\n</conversation_summary>` }, ...turns.slice(-keepLast)];
const pin = pinnedBlock(facts);
const pinWindow = slidingWindow(turns, budgetH - est(pin) - 8);
const strategies: [string, Msg[], string, number][] = [
  ["full history", [...turns], "", 0],
  ["sliding window", window, "", 0],
  ["summary + last 4", summaryMsgs, "", costUsd(MODEL_HAIKU, smsg.usage)],
  ["pinned facts + window", pinWindow, pin, 0],
];
const results: Record<string, [number, string[]]> = {};
{
  const rows: string[][] = [];
  for (const [name, msgs, pinned, extra] of strategies) {
    const tok = name === "full history" ? fullTok : (await tc.request([...msgs, finalTurn(question, "", "", pinned)])) - qOnly;
    const contextText = pinned + "\n" + msgs.map((m) => String(m.content)).join("\n");
    const kept = recall(contextText, critical);
    const lost = critical.filter((f) => !kept.includes(f));
    results[name] = [tok, lost];
    rows.push([name, num(tok), tok <= budgetH ? "yes" : "NO", `${kept.length}/${critical.length}`, lost.join(", ") || "-", extra ? `$${extra.toFixed(6)}` : "-"]);
  }
  table(["strategy", "tokens", `fits ${budgetH}?`, "facts", "lost", "extra call"], rows);
}
console.log("  pinned facts:");
for (const line of pin.split("\n").slice(1, -1)) console.log("    " + line);
const viable = Object.entries(results).filter(([, [t]]) => t <= budgetH);
const best = viable.reduce((a, b) => (b[1][1].length < a[1][1].length || (b[1][1].length === a[1][1].length && b[1][0] < a[1][0]) ? b : a));
console.log(`  -> choose '${best[0]}': highest recall within budget. The summarizer silently dropped a detail;`);
console.log("     sliding windows forget the oldest turns - often where identifiers were given.");

// ------------------------------------------------------------------ 6
section("6. Assemble, verify with count_tokens, call the model");
{
  const chosen = strategies.find(([name]) => name === best[0])!;
  const optimized: Msg[] = [...chosen[1], finalTurn(question, docsBlock, telBlock, chosen[2])];
  const optTotal = await tc.request(optimized, system);
  console.log(`  optimized request: ${num(optTotal)} tok (budget ${num(inputBudget)}) -> ${optTotal <= inputBudget ? "OK" : "STILL OVER"}`);
  const variants: [string, Msg[]][] = [
    ["naive (everything)", naiveMsgs],
    ["sliding window only", [...window, finalTurn(question, docsBlock, telBlock)]],
    ["optimized", optimized],
  ];
  const rows: string[][] = [];
  const answers: Record<string, string> = {};
  for (const [name, msgs] of variants) {
    const m = await client.messages.create({ model: MODEL, max_tokens: OUTPUT_RESERVE, system, messages: msgs, output_config: { effort: "medium" } });
    if (m.stop_reason === "max_tokens") console.log(`  ${name}: answer truncated - the output reserve was too small`);
    answers[name] = textOf(m);
    rows.push([name, num(m.usage.input_tokens), String(m.usage.output_tokens), `$${costUsd(MODEL, m.usage).toFixed(5)}`, `${recall(textOf(m), critical).length}/${critical.length}`]);
  }
  table(["variant", "input tok", "output tok", "cost", "critical facts in answer"], rows);
  for (const name of ["sliding window only", "optimized"]) {
    console.log(`  answer (${name}):`);
    for (const line of answers[name]!.split("\n")) console.log("    | " + line);
  }
  const inAnswer = recall(answers["sliding window only"]!, critical);
  const missing = critical.filter((f) => !inAnswer.includes(f));
  console.log(`  Window-only answer is missing ${missing.join(", ") || "nothing"} -> it offers to open a DUPLICATE ticket.`);
  const perReq = ((naiveTotal - optTotal) * PRICING[MODEL]!.input) / 1e6;
  console.log(
    `  Saved ${num(naiveTotal - optTotal)} input tok/request (${Math.round((1 - optTotal / naiveTotal) * 100)}%) = ` +
      `$${num(perReq * 100_000, 2)} per 100k requests on Sonnet 5.5 - with a better answer than the window-only variant.`,
  );
  console.log(`  (${tc.calls} count_tokens calls: free to call, but rate-limited - estimate locally, verify once per request.)`);
}

// ------------------------------------------------------------------ 7
section("7. Server-side context management (live-only)");
console.log("  Context editing - clears stale tool results/thinking in place, no summary:");
for (const line of CONTEXT_EDITING_SNIPPET.split("\n")) console.log("    " + line);
console.log("  Compaction - summarizes earlier context into a compaction block near the limit:");
for (const line of COMPACTION_SNIPPET.split("\n")) console.log("    " + line);
table(["mechanism", "what happens", "best for", "watch out"], [
  ["client-side (this lab)", "you trim/filter/summarize/pin", "every request; full control", "you own the logic + tests"],
  ["context editing", "server clears old tool_use/results", "long tool loops, stale results", "cache invalidates at edit point"],
  ["compaction", "server summarizes older turns", "chats nearing the window", "pass compaction block back"],
  ["memory tool", "model reads/writes /memories files", "facts across SESSIONS", "you implement storage"],
]);
if (isLive()) {
  await runServerSide(client);
} else {
  console.log("  [live-only] Set ANTHROPIC_API_KEY to send both requests (betas context-management-2025-06-27,");
  console.log("  compact-2026-01-12); the offline mock has no server-side context management.");
}

section("Takeaways");
console.log("  * Count before you send (count_tokens) and reserve output tokens up front.");
console.log("  * Budget by bucket; let unused share spill over; clean inputs before cutting them.");
console.log("  * Retrieve less but better: relevance-filter docs, trim tool results to needed fields.");
console.log("  * Long histories: pin critical facts + window or summary; measure recall, not just tokens.");
console.log("  * Server-side: context editing clears, compaction summarizes, memory persists across sessions.");
