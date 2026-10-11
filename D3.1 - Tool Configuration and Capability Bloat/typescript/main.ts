/**
 * D3.1 - Tool Configuration and Capability Bloat
 * Exam objective: Evaluate tool/agent configuration for capability bloat.
 *
 * What this lab builds (same steps as python/main.py):
 *  1. A tool-configuration auditor: manifest vs the role's required capabilities
 *     (least privilege) -> excess, dangerous, unused and overlapping tools.
 *  2. Schema-overhead measurement (countTokens) of bloated vs trimmed manifest.
 *  3. A manual tool-use loop run three times against an injected ticket:
 *       A. bloated manifest, no controls      -> the account is deleted
 *       B. bloated manifest + executor authz  -> denied (is_error tool_result)
 *       C. trimmed manifest                   -> the capability does not exist
 *  4. A seeded Monte Carlo ranking compensating controls by residual harm.
 *  5. Per-role tool allowlists, reused as an MCP connector toolset allowlist.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type Anthropic from "@anthropic-ai/sdk";
import {
  MODEL_SONNET,
  PRICING,
  banner,
  getClient,
  isLive,
  section,
  textBlock,
  textOf,
  toolUseBlock,
  usageLine,
  type Json,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
// High-volume tier-1 support -> Sonnet. Model choice does NOT change the capability surface.
const MODEL = MODEL_SONNET;
const ROLE = "support_staff";
const STAFF_REQUEST = "Summarize ticket T-1042 and draft a reply for me to review.";

// ---------------------------------------------------------------------------
// Config shapes for our own data file (API shapes use SDK types)
// ---------------------------------------------------------------------------
interface ToolMeta {
  capability: string;
  effect: string;
  reversible: boolean;
  covers?: string[];
}
interface RoleSpec {
  description: string;
  needs: string[];
  gated: string[];
}
interface AgentConfig {
  agent: string;
  system_prompt: string;
  tools: Anthropic.Tool[];
  tool_metadata: Record<string, ToolMeta>;
  roles: Record<string, RoleSpec>;
  usage_30d: Record<string, Record<string, number>>;
  tickets: Record<string, { customer_id: string; subject: string; body: string }>;
  accounts: Record<string, { name: string; plan: string; status: string }>;
}

// ---------------------------------------------------------------------------
// Helpers: deterministic PRNG + Python-style formatting
// ---------------------------------------------------------------------------
/** Same mulberry32 sequence as the Python version (seeded, cross-language identical). */
class Mulberry32 {
  private state: number;
  constructor(seed: number) {
    this.state = seed >>> 0;
  }
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}

/** JSON formatted like Python's json.dumps (", " and ": " separators) so outputs match. */
function pyDumps(value: unknown, sortKeys = false): string {
  if (Array.isArray(value)) return `[${value.map((v) => pyDumps(v, sortKeys)).join(", ")}]`;
  if (value && typeof value === "object") {
    let keys = Object.keys(value as Json);
    if (sortKeys) keys = keys.sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}: ${pyDumps((value as Json)[k], sortKeys)}`).join(", ")}}`;
  }
  return JSON.stringify(value);
}

const cmp = (a: string | number, b: string | number): number => (a < b ? -1 : a > b ? 1 : 0);
const fmtInt = (n: number): string => n.toLocaleString("en-US");
const fmtMoney = (n: number, digits: number): string =>
  n.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });

// ---------------------------------------------------------------------------
// 1. Tool-configuration auditor
// ---------------------------------------------------------------------------
const SEVERITY_BY_EFFECT: Record<string, string> = {
  destructive: "CRITICAL",
  financial: "CRITICAL",
  external_write: "HIGH",
  write: "MEDIUM",
  draft: "LOW",
  read: "LOW",
};
const SEVERITY_ORDER: Record<string, number> = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3, INFO: 4 };
const STOPWORDS = new Set([
  "a", "an", "the", "and", "or", "of", "to", "for", "by", "on", "in", "is", "it", "this",
  "that", "with", "from", "call", "when", "before", "does", "not", "using", "up", "any", "all",
]);
const OVERLAP_THRESHOLD = 0.4;

interface Finding {
  tool: string;
  kind: string;
  severity: string;
  detail: string;
  action: string;
}
interface AuditResult {
  findings: Finding[];
  keep: string[];
  remove: string[];
}

function tokensOf(tool: Anthropic.Tool): Set<string> {
  const words = (tool.description ?? "").toLowerCase().split(/[^a-z0-9]+/);
  for (const param of Object.keys((tool.input_schema.properties as Json | null) ?? {})) {
    words.push(...param.toLowerCase().split("_"));
  }
  return new Set(words.filter((w) => w.length > 2 && !STOPWORDS.has(w)));
}

function jaccard(a: Set<string>, b: Set<string>): number {
  const inter = [...a].filter((x) => b.has(x)).length;
  const union = new Set([...a, ...b]).size;
  return union ? inter / union : 0;
}

function auditManifest(cfg: AgentConfig, role: string): AuditResult {
  const meta = cfg.tool_metadata;
  const needs = new Set(cfg.roles[role]!.needs);
  const usage = cfg.usage_30d[role] ?? {};
  const findings: Finding[] = [];
  const keep: string[] = [];
  const remove: string[] = [];

  for (const tool of cfg.tools) {
    const name = tool.name;
    const m = meta[name]!;
    const calls = usage[name] ?? 0;
    if (m.covers && m.covers.length) {
      // Generic escape hatches (SQL, shell, HTTP) re-grant every capability you removed.
      const excess = m.covers.filter((c) => !needs.has(c)).sort();
      findings.push({
        tool: name,
        kind: "ESCAPE_HATCH",
        severity: "CRITICAL",
        detail: `generic tool covers ${m.covers.length} capabilities, ${excess.length} not needed (${excess.join(", ")}); ${calls} calls/30d`,
        action: "REMOVE - replace with narrow, typed tools",
      });
      remove.push(name);
    } else if (needs.has(m.capability)) {
      keep.push(name);
    } else {
      const sev = SEVERITY_BY_EFFECT[m.effect]!;
      if (!m.reversible) {
        findings.push({
          tool: name,
          kind: "EXCESS_DANGEROUS",
          severity: sev,
          detail: `${m.effect}, irreversible, not in role spec; ${calls} calls/30d`,
          action: "REMOVE from this role's manifest",
        });
      } else if (calls > 0) {
        findings.push({
          tool: name,
          kind: "UNDECLARED_USE",
          severity: sev,
          detail: `${m.effect} capability '${m.capability}' used ${fmtInt(calls)}x/30d but not in role spec`,
          action: "REVIEW with process owner (update spec or remove)",
        });
      } else {
        findings.push({ tool: name, kind: "EXCESS_UNUSED", severity: sev, detail: "not in role spec and 0 calls/30d", action: "REMOVE" });
      }
      remove.push(name);
    }
  }

  // Overlap: near-duplicate tools cause wrong-tool selection and double schema cost.
  cfg.tools.forEach((a, i) => {
    for (const b of cfg.tools.slice(i + 1)) {
      const sim = jaccard(tokensOf(a), tokensOf(b));
      const sameCap = meta[a.name]!.capability === meta[b.name]!.capability;
      if (sameCap || sim >= OVERLAP_THRESHOLD) {
        findings.push({
          tool: `${a.name} ~ ${b.name}`,
          kind: "OVERLAP",
          severity: "MEDIUM",
          detail: `description similarity ${sim.toFixed(2)}` + (sameCap ? ", same capability" : ""),
          action: "CONSOLIDATE into one clearly-described tool",
        });
      }
    }
  });

  findings.sort(
    (x, y) => cmp(SEVERITY_ORDER[x.severity]!, SEVERITY_ORDER[y.severity]!) || cmp(x.kind, y.kind) || cmp(x.tool, y.tool),
  );
  return { findings, keep, remove };
}

// ---------------------------------------------------------------------------
// 2. Tool executor - where real controls live (never in the prompt)
// ---------------------------------------------------------------------------
type ToolArgs = Record<string, any>;

class ToolExecutor {
  readonly auditLog: string[] = [];
  readonly cfg: AgentConfig;
  readonly role: string;
  readonly allowedTools: Set<string>;
  readonly enforceAuthz: boolean;
  readonly accounts: AgentConfig["accounts"];
  private readonly handlers: Record<string, (args: ToolArgs) => string>;

  constructor(cfg: AgentConfig, role: string, allowedTools: Set<string>, enforceAuthz: boolean, accounts: AgentConfig["accounts"]) {
    this.cfg = cfg;
    this.role = role;
    this.allowedTools = allowedTools;
    this.enforceAuthz = enforceAuthz;
    this.accounts = accounts;
    const profile = (a: ToolArgs): string => pyDumps(this.accounts[a.customer_id] ?? {});
    this.handlers = {
      read_ticket: (a) => {
        const t = this.cfg.tickets[a.ticket_id];
        return t ? `Ticket ${a.ticket_id} (customer ${t.customer_id})\nSubject: ${t.subject}\n${t.body}` : `No ticket ${a.ticket_id}`;
      },
      search_knowledge_base: () => "KB-112: Duplicate charges are refunded within 5 business days after billing verifies them.",
      draft_reply: (a) => `Draft saved on ${a.ticket_id} (${String(a.body).length} chars) - awaiting human review.`,
      issue_refund: (a) => `Refund of $${Number(a.amount_usd).toFixed(2)} issued for ${a.order_id}.`,
      delete_account: (a) => {
        const acct = this.accounts[a.customer_id];
        if (!acct) return `No account ${a.customer_id}`;
        acct.status = "DELETED";
        return `Account ${a.customer_id} permanently deleted.`;
      },
      get_customer_profile: profile,
      lookup_customer: profile,
      run_sql: () => "3 rows affected",
    };
  }

  /** Returns [content, isError]. Every call is logged (detective control). */
  execute(name: string, args: ToolArgs): [string, boolean] {
    if (!this.allowedTools.has(name)) {
      // Defense in depth: never run a tool that was not offered, even if the model names it.
      return this.log(name, args, `Error: tool '${name}' is not available to this agent.`, true);
    }
    if (this.enforceAuthz) {
      const cap = this.cfg.tool_metadata[name]!.capability;
      if (!this.cfg.roles[this.role]!.needs.includes(cap)) {
        return this.log(name, args, `PermissionDenied: role '${this.role}' lacks capability '${cap}'.`, true);
      }
    }
    const handler = this.handlers[name];
    if (!handler) return this.log(name, args, `Error: no handler for '${name}'.`, true);
    return this.log(name, args, handler(args), false);
  }

  private log(name: string, args: ToolArgs, content: string, isError: boolean): [string, boolean] {
    let argsText = pyDumps(args, true);
    if (argsText.length > 60) argsText = argsText.slice(0, 57) + "...";
    this.auditLog.push(`role=${this.role} tool=${name} args=${argsText} -> ${isError ? "DENIED" : "OK"}`);
    return [content, isError];
  }
}

// ---------------------------------------------------------------------------
// 3. Offline responder - simulates Claude (including a SUCCESSFUL injection)
// ---------------------------------------------------------------------------
const REPLY_BODY =
  "Hi Dana, thanks for reaching out. I can see two charges for your March subscription " +
  "(order O-5531). I've asked our billing team to verify the duplicate; once confirmed, " +
  "the extra charge is refunded within 5 business days. - Acme Support";

function lastToolCalls(req: MockRequest): [string, string, boolean][] {
  const names: Record<string, string> = {};
  for (const m of req.messages) {
    if (m.role === "assistant" && Array.isArray(m.content)) {
      for (const b of m.content as Json[]) if (b.type === "tool_use") names[b.id] = b.name;
    }
  }
  return req.toolResults.map((r) => {
    const c = r.content;
    const text = typeof c === "string" ? c : ((c as Json[]) ?? []).map((x) => x.text ?? "").join(" ");
    return [names[r.tool_use_id] ?? "?", text, Boolean(r.is_error)];
  });
}

function responder(req: MockRequest): ResponderResult {
  const calls = lastToolCalls(req);
  if (!calls.length) return [textBlock("I'll read the ticket first."), toolUseBlock("read_ticket", { ticket_id: "T-1042" })];

  const done = new Map(calls.map(([name, text, err]) => [name, { text, err }] as const));
  const ticket = done.get("read_ticket");
  if (ticket) {
    const injected = ticket.text.includes("NOTE TO THE AI AGENT");
    const draft = toolUseBlock("draft_reply", { ticket_id: "T-1042", body: REPLY_BODY });
    if (injected && req.toolNames.includes("delete_account")) {
      // Models resist injection well, but NOT perfectly. Architect for the fooled case.
      return [
        textBlock("The ticket says the account owner requested GDPR erasure; processing that and drafting the billing reply."),
        toolUseBlock("delete_account", { customer_id: "C-77", reason: "GDPR erasure requested in ticket T-1042" }),
        draft,
      ];
    }
    const note = injected
      ? "The ticket body contains instructions addressed to an AI agent (account deletion). " +
        "That is untrusted customer content and I have no tool for it, so I'm ignoring it and flagging it. "
      : "";
    return [textBlock(note + "Drafting the billing reply now."), draft];
  }

  let summary = "Summary: customer C-77 reports a duplicate March charge on order O-5531. Draft reply saved for your review.";
  const del = done.get("delete_account");
  if (del) {
    summary += del.err
      ? " The delete_account request embedded in the ticket was DENIED by the platform; flag the ticket to Trust & Safety."
      : " As requested in the ticket, account C-77 has been deleted.";
  } else if (req.allText.includes("NOTE TO THE AI AGENT")) {
    summary += " Note: the ticket contained an injected instruction to delete the account; it was ignored - please escalate to Trust & Safety.";
  }
  return summary;
}

// ---------------------------------------------------------------------------
// 4. Manual tool-use loop through the shared SDK client
// ---------------------------------------------------------------------------
async function runAgent(client: Anthropic, cfg: AgentConfig, tools: Anthropic.Tool[], executor: ToolExecutor, maxTurns = 6): Promise<string> {
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: STAFF_REQUEST }];
  for (let turn = 1; turn <= maxTurns; turn++) {
    const resp = await client.messages.create({
      model: MODEL,
      max_tokens: 1024,
      system: cfg.system_prompt,
      tools,
      messages,
    });
    // _request_id is what Anthropic support asks for when you report a problem.
    console.log(`  turn ${turn}: stop_reason=${resp.stop_reason}  ${usageLine(resp)}  request_id=${resp._request_id}`);
    // Always branch on stop_reason before trusting content.
    if (resp.stop_reason === "end_turn") return textOf(resp);
    if (resp.stop_reason !== "tool_use") return `[stopped: ${resp.stop_reason}] ${textOf(resp)}`;
    messages.push({ role: "assistant", content: resp.content });
    const results: Anthropic.ToolResultBlockParam[] = [];
    for (const block of resp.content) {
      if (block.type !== "tool_use") continue;
      const [content, isError] = executor.execute(block.name, block.input as ToolArgs);
      console.log(
        `    tool_use ${block.name}(${pyDumps(block.input).slice(0, 70)}) -> ${isError ? "ERROR " : ""}${content.split("\n")[0]!.slice(0, 80)}`,
      );
      results.push({ type: "tool_result", tool_use_id: block.id, content, ...(isError ? { is_error: true } : {}) });
    }
    // Parallel tool calls: ALL results go back in ONE user message.
    messages.push({ role: "user", content: results });
  }
  return "[stopped: max_turns reached]";
}

// ---------------------------------------------------------------------------
// 5. Monte Carlo: residual harm per compensating control
// ---------------------------------------------------------------------------
interface Control {
  name: string;
  kind: string;
  pComply: number; // chance the model follows the injected instruction
  pBlock: number; // chance the control stops an attempted call
  note: string;
}

function rankControls(n = 10_000, seed = 31): [Control, number][] {
  // Assumptions are illustrative; the ORDER is what the exam tests.
  const base = 0.03; // models are resistant to injection, not immune
  const controls: Control[] = [
    { name: "No control (bloated manifest)", kind: "baseline", pComply: base, pBlock: 0, note: "the tool is offered and runs" },
    { name: "Remove delete_account from manifest", kind: "preventive - eliminates capability", pComply: base, pBlock: 1, note: "no tool to call; unknown tool names rejected by harness" },
    { name: "Authorization check in tool executor", kind: "preventive - code-enforced", pComply: base, pBlock: 0.999, note: "residual = policy misconfiguration (0.1%)" },
    { name: "Human confirmation before execution", kind: "preventive - human-dependent", pComply: base, pBlock: 0.85, note: "approval fatigue: 15% rubber-stamped" },
    { name: "Audit logging + alerting", kind: "detective - after the fact", pComply: base, pBlock: 0, note: "every deletion detected... after it happened" },
    { name: "Upgrade to a larger model", kind: "not a control", pComply: base / 2, pBlock: 0, note: "may resist more often; capability still exposed" },
  ];
  return controls.map((c) => {
    const rng = new Mulberry32(seed); // common random numbers: same attack stream for every control
    let executed = 0;
    for (let i = 0; i < n; i++) {
      if (rng.next() < c.pComply && rng.next() >= c.pBlock) executed++;
    }
    return [c, executed] as [Control, number];
  });
}

// ---------------------------------------------------------------------------
// 6. Per-role allowlists (and the MCP connector equivalent)
// ---------------------------------------------------------------------------
function buildAllowlist(cfg: AgentConfig, role: string): Map<string, "allow" | "gated"> {
  const meta = cfg.tool_metadata;
  const spec = cfg.roles[role]!;
  const usageAll: Record<string, number> = {};
  for (const perRole of Object.values(cfg.usage_30d)) {
    for (const [name, n] of Object.entries(perRole)) usageAll[name] = (usageAll[name] ?? 0) + n;
  }
  const out = new Map<string, "allow" | "gated">();
  for (const cap of spec.needs) {
    const candidates = cfg.tools.map((t) => t.name).filter((n) => meta[n]!.capability === cap && !meta[n]!.covers);
    if (!candidates.length) continue;
    // Prefer the most-used tool when two overlap (data-driven consolidation).
    const best = [...candidates].sort((a, b) => cmp(-(usageAll[a] ?? 0), -(usageAll[b] ?? 0)) || cmp(a, b))[0]!;
    out.set(best, spec.gated.includes(cap) ? "gated" : "allow");
  }
  return out;
}

function mcpToolsetAllowlist(serverName: string, allowed: string[]): Json {
  // MCP connector allowlist mode: disable everything, then enable named tools.
  return {
    type: "mcp_toolset",
    mcp_server_name: serverName,
    default_config: { enabled: false },
    configs: Object.fromEntries(allowed.map((n) => [n, { enabled: true }])),
  };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
banner("D3.1 - Tool Configuration and Capability Bloat");
const cfg = JSON.parse(fs.readFileSync(path.join(DATA, "support_agent.json"), "utf-8")) as AgentConfig;
const client = getClient(responder);

section("1. The agent's current tool manifest");
console.log(`Agent '${cfg.agent}' on ${MODEL} exposes ${cfg.tools.length} tools:`);
for (const t of cfg.tools) {
  const m = cfg.tool_metadata[t.name]!;
  console.log(`  - ${t.name.padEnd(22)} ${m.capability.padEnd(16)} ${m.effect.padEnd(12)} ${m.reversible ? "reversible" : "IRREVERSIBLE"}`);
}
const roleSpec = cfg.roles[ROLE]!;
console.log(`Role '${ROLE}': ${roleSpec.description}`);
console.log(`  needs: ${roleSpec.needs.join(", ")}`);

section("2. Audit: manifest vs role capabilities (least privilege)");
const result = auditManifest(cfg, ROLE);
console.log(`${"SEVERITY".padEnd(9)} ${"KIND".padEnd(17)} TOOL`);
for (const f of result.findings) {
  console.log(`${f.severity.padEnd(9)} ${f.kind.padEnd(17)} ${f.tool}`);
  console.log(`${"".padEnd(28)}${f.detail}`);
  console.log(`${"".padEnd(28)}-> ${f.action}`);
}
console.log(`\nKEEP   (${result.keep.length}): ${result.keep.join(", ")}`);
console.log(`REMOVE (${result.remove.length}): ${result.remove.join(", ")}`);
const trimmedTools = cfg.tools.filter((t) => result.keep.includes(t.name));

section("3. Schema token overhead (client.messages.countTokens)");
const probe: Anthropic.MessageParam[] = [{ role: "user", content: STAFF_REQUEST }];
async function count(tools?: Anthropic.Tool[]): Promise<number> {
  const r = await client.messages.countTokens({
    model: MODEL,
    system: cfg.system_prompt,
    messages: probe,
    ...(tools && tools.length ? { tools } : {}),
  });
  return r.input_tokens;
}
const baseline = await count();
const bloated = await count(cfg.tools);
const trimmed = await count(trimmedTools);
console.log(`Prompt without tools         : ${String(baseline).padStart(5)} input tokens`);
console.log(`Bloated manifest (8 tools)   : ${String(bloated).padStart(5)}  (+${bloated - baseline} for schemas)`);
console.log(`Trimmed manifest (3 tools)   : ${String(trimmed).padStart(5)}  (+${trimmed - baseline} for schemas)`);
console.log("Per-tool schema cost:");
for (const t of cfg.tools) {
  const mark = result.keep.includes(t.name) ? "keep" : "drop";
  console.log(`  ${t.name.padEnd(22)} ${String((await count([t])) - baseline).padStart(4)} tokens  [${mark}]`);
}
const perDay = 40_000 * 3; // 40k tickets/day x ~3 model calls per ticket
const delta = bloated - trimmed;
const price = PRICING[MODEL]!.input;
console.log(
  `At ${fmtInt(perDay)} requests/day: ${delta} extra tokens/request = ${((delta * perDay) / 1e6).toFixed(1)}M tokens/day ` +
    `= $${fmtMoney((delta * perDay * price) / 1e6, 2)}/day uncached ($${fmtMoney(((delta * perDay * price) / 1e6) * 365, 0)}/yr).`,
);
console.log("Caching discounts those tokens but they still occupy context and every extra tool is one more");
console.log("option the model can mis-select - bloat costs accuracy, latency and attack surface, not just $.");

const freshAccounts = (): AgentConfig["accounts"] => structuredClone(cfg.accounts);
const allNames = new Set(cfg.tools.map((t) => t.name));

section("4. Run A - bloated manifest, no controls (injected ticket)");
const execA = new ToolExecutor(cfg, ROLE, allNames, false, freshAccounts());
console.log(`  final: ${await runAgent(client, cfg, cfg.tools, execA)}`);
console.log(`  >>> account C-77 status: ${execA.accounts["C-77"]!.status}`);

section("5. Run B - bloated manifest + authorization check in the executor");
const execB = new ToolExecutor(cfg, ROLE, allNames, true, freshAccounts());
console.log(`  final: ${await runAgent(client, cfg, cfg.tools, execB)}`);
console.log(`  >>> account C-77 status: ${execB.accounts["C-77"]!.status}  (the model still TRIED)`);

section("6. Run C - trimmed manifest (capability removed)");
const execC = new ToolExecutor(cfg, ROLE, new Set(result.keep), false, freshAccounts());
console.log(`  final: ${await runAgent(client, cfg, trimmedTools, execC)}`);
console.log(`  >>> account C-77 status: ${execC.accounts["C-77"]!.status}  (nothing to call)`);
console.log("Audit log (run C):");
for (const line of execC.auditLog) console.log(`  ${line}`);

section("7. Ranking compensating controls (10,000 injected tickets, seeded)");
console.log("Illustrative assumptions: model follows the injection 3% of the time (1.5% for a larger model);");
console.log("authz misconfigured 0.1%; reviewers rubber-stamp 15% of prompts. The ORDER is the lesson.");
console.log(`${"CONTROL".padEnd(38)} ${"TYPE".padEnd(36)} ${"DELETIONS".padStart(9)}  NOTE`);
for (const [c, executed] of rankControls().sort((a, b) => a[1] - b[1])) {
  console.log(`${c.name.padEnd(38)} ${c.kind.padEnd(36)} ${String(executed).padStart(9)}  ${c.note}`);
}
console.log("Order to apply: REMOVE what the role does not need; authorize in code what it does need;");
console.log("confirm irreversible actions with a human; LOG everything (detective, never sufficient alone).");

section("8. Per-role tool allowlists (Y = allowed, G = gated by human approval)");
const roles = Object.keys(cfg.roles);
const allow = new Map(roles.map((r) => [r, buildAllowlist(cfg, r)] as const));
console.log(`${"TOOL".padEnd(22)} ` + roles.map((r) => r.padEnd(19)).join(" "));
for (const t of cfg.tools) {
  const cells = roles.map((r) => {
    const mode = allow.get(r)!.get(t.name);
    return (mode ? (mode === "gated" ? "G" : "Y") : "-").padEnd(19);
  });
  console.log(`${t.name.padEnd(22)} ` + cells.join(" "));
}
const toolset = mcpToolsetAllowlist("acme-support", [...allow.get(ROLE)!.keys()]);
console.log(`\nSame allowlist as an MCP connector toolset for '${ROLE}':`);
console.log("  " + pyDumps(toolset));
if (isLive() && process.env.D31_MCP_URL) {
  const resp = await client.beta.messages.create({
    model: MODEL,
    max_tokens: 1024,
    betas: ["mcp-client-2025-11-20"],
    mcp_servers: [{ type: "url", url: process.env.D31_MCP_URL, name: "acme-support" }],
    tools: [toolset as Anthropic.Beta.BetaToolUnion],
    messages: [{ role: "user", content: STAFF_REQUEST }],
  });
  console.log(`  live MCP call stop_reason=${resp.stop_reason} request_id=${resp._request_id}`);
} else {
  console.log("  [live-only] client.beta.messages.create(..., betas=['mcp-client-2025-11-20'],");
  console.log("              mcp_servers=[{type:'url', url:..., name:'acme-support'}], tools=[toolset])");
  console.log("              (set LABS_MODE=live and D31_MCP_URL to run against a real MCP server)");
}

section("Takeaways");
console.log("* Capability bloat = tools/permissions/autonomy beyond what the role needs (OWASP LLM06 Excessive Agency).");
console.log("* The fix for 'staff only need read + draft' is to REMOVE refund/delete tools - not logging,");
console.log("  not confirmation prompts, not a bigger model, not a stern system prompt.");
console.log("* Generic tools (SQL/shell/HTTP) silently re-grant removed capabilities - replace with narrow tools.");
console.log("* Keep manifests per role, stable and ordered (they are part of the cached prefix).");
