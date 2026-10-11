/**
 * D1.4 - Multi-Agent Systems and Orchestration
 *
 * Hierarchical research system (orchestrator + parallel subagents), plus the other topologies:
 *
 *   orchestrator (Opus) --plan (structured output)--> subagents (Sonnet/Haiku), each with an
 *   ISOLATED context and its own tool loop --condensed JSON--> blackboard + orchestrator synthesis
 *
 * Also shown: sequential vs parallel wall-clock, blackboard conflicts, token/cost multiplier vs a
 * single agent, failure isolation (chaos test), a sequential pipeline, a peer handoff, and a
 * deterministic "should this be multi-agent at all?" check.
 */
import type Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  LATENCY_SCALE,
  MODEL_HAIKU,
  MODEL_OPUS,
  MODEL_SONNET,
  banner,
  costUsd,
  elapsedMs,
  estimateTokens,
  getClient,
  isLive,
  section,
  textBlock,
  textOf,
  toolUseBlock,
  type MockRequest,
  type Responder,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
const CORPUS: { id: string; topic: string; text: string }[] = JSON.parse(readFileSync(path.join(DATA, "corpus.json"), "utf-8")).snippets;
const TASK =
  "Should we launch an AI-first customer-support offering for EU mid-market retailers next year? " +
  "Cover market, competitors, unit economics and regulation, and give a recommendation.";
const MODEL_MAP = { sonnet: MODEL_SONNET, haiku: MODEL_HAIKU };
const SUBAGENT_DEADLINE_MS = isLive() ? 120_000 : 120_000 * LATENCY_SCALE; // wall-clock budget per subagent
const cj = (obj: unknown): string => JSON.stringify(obj);

// ---------------------------------------------------------------------------
// Ledger: tokens and dollars per phase, so multipliers are computed, not asserted
// ---------------------------------------------------------------------------
const LEDGER = new Map<string, { calls: number; tokens: number; cost: number }>();
const row = (phase: string) => LEDGER.get(phase) ?? { calls: 0, tokens: 0, cost: 0 };
function record(phase: string, msg: Anthropic.Message): void {
  const r = row(phase);
  r.calls += 1;
  r.tokens += msg.usage.input_tokens + msg.usage.output_tokens;
  r.cost += costUsd(msg.model, msg.usage);
  LEDGER.set(phase, r);
}

// ---------------------------------------------------------------------------
// Contracts between orchestrator and subagents
// ---------------------------------------------------------------------------
const Subtask = z.object({
  id: z.string(),
  topic: z.enum(["market", "competitors", "cost", "regulatory"]),
  objective: z.string(),
  model: z.enum(["sonnet", "haiku"]),
  boundaries: z.string(),
  critical: z.boolean(),
});
type Subtask = z.infer<typeof Subtask>;
const Plan = z.object({ subtasks: z.array(Subtask) });
type Plan = z.infer<typeof Plan>;

const Fact = z.object({ key: z.string(), value: z.string(), confidence: z.number() });
type Fact = z.infer<typeof Fact>;
const SubagentResult = z.object({
  topic: z.string(),
  findings: z.array(z.string()),
  facts: z.array(Fact),
  confidence: z.number(),
  open_questions: z.array(z.string()),
});
type SubagentResult = z.infer<typeof SubagentResult>;

const RESULT_SCHEMA = {
  type: "object",
  properties: {
    topic: { type: "string" },
    findings: { type: "array", items: { type: "string" } },
    facts: {
      type: "array",
      items: {
        type: "object",
        properties: { key: { type: "string" }, value: { type: "string" }, confidence: { type: "number" } },
        required: ["key", "value", "confidence"],
        additionalProperties: false,
      },
    },
    confidence: { type: "number" },
    open_questions: { type: "array", items: { type: "string" } },
  },
  required: ["topic", "findings", "facts", "confidence", "open_questions"],
  additionalProperties: false,
};

const SEARCH_TOOL: Anthropic.Tool = {
  name: "search_corpus",
  description: "Search the research corpus. Call this before answering; returns the top 3 snippets.",
  strict: true,
  input_schema: { type: "object", properties: { query: { type: "string" } }, required: ["query"], additionalProperties: false },
};

const words = (s: string): Set<string> => new Set(s.toLowerCase().match(/[a-z0-9]+/g) ?? []);

function searchCorpus(query: string, k = 3): { id: string; text: string }[] {
  const q = words(query);
  const scored = CORPUS.map((s) => {
    const w = words(s.text);
    return { score: [...q].filter((t) => w.has(t)).length, s };
  });
  scored.sort((a, b) => b.score - a.score || a.s.id.localeCompare(b.s.id));
  return scored.slice(0, k).filter((x) => x.score > 0).map(({ s }) => ({ id: s.id, text: s.text }));
}

interface Outcome {
  subtask: Subtask;
  result: SubagentResult;
  calls: number;
  consumedTokens: number; // everything the subagent read and wrote in ITS context
  returnedTokens: number; // what the orchestrator has to read
  sources: string[];
}

class SubagentError extends Error {}

/** One subagent = its own system prompt, its own message history, its own tool loop. */
async function runSubagent(client: Anthropic, st: Subtask, phase: string): Promise<Outcome> {
  const system =
    `You are a research subagent. Your topic: ${st.topic}.\nObjective: ${st.objective}\n` +
    `Boundaries: ${st.boundaries}\nUse search_corpus, then return ONLY the JSON result ` +
    "(<= 3 findings, facts with confidence, open questions). Be concise: the orchestrator's " +
    "context is the scarce resource.";
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: `Brief ${st.id}: ${st.objective}` }];
  let calls = 0;
  let consumed = 0;
  const sources: string[] = [];
  for (let i = 0; i < 4; i++) {
    // subagent step limit
    const msg = await client.messages.create({
      model: MODEL_MAP[st.model],
      max_tokens: 1024,
      system,
      tools: [SEARCH_TOOL],
      output_config: { format: { type: "json_schema", schema: RESULT_SCHEMA } },
      messages,
    });
    record(phase, msg);
    calls += 1;
    consumed += msg.usage.input_tokens + msg.usage.output_tokens;
    if (msg.stop_reason === "tool_use") {
      messages.push({ role: "assistant", content: msg.content });
      const results: Anthropic.ToolResultBlockParam[] = [];
      for (const block of msg.content) {
        if (block.type !== "tool_use") continue;
        const hits = searchCorpus((block.input as { query: string }).query);
        sources.push(...hits.map((h) => h.id));
        results.push({ type: "tool_result", tool_use_id: block.id, content: cj(hits) });
      }
      messages.push({ role: "user", content: results });
      continue;
    }
    if (msg.stop_reason !== "end_turn") throw new SubagentError(`${st.id}: stop_reason=${msg.stop_reason}`);
    const result = SubagentResult.parse(JSON.parse(textOf(msg)));
    return { subtask: st, result, calls, consumedTokens: consumed, returnedTokens: estimateTokens(cj(result)), sources };
  }
  throw new SubagentError(`${st.id}: step limit reached`);
}

function withDeadline<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label}: deadline exceeded`)), ms);
  });
  return Promise.race([p, deadline]).finally(() => clearTimeout(timer));
}

async function runAllParallel(client: Anthropic, plan: Plan, phase: string): Promise<(Outcome | Error)[]> {
  // allSettled: one failing subagent must not reject the whole batch (failure isolation)
  const settled = await Promise.allSettled(plan.subtasks.map((st) => withDeadline(runSubagent(client, st, phase), SUBAGENT_DEADLINE_MS, st.id)));
  return settled.map((s) => (s.status === "fulfilled" ? s.value : (s.reason as Error)));
}

// ---------------------------------------------------------------------------
// Blackboard: shared state the harness maintains; subagents never see each other's context
// ---------------------------------------------------------------------------
interface Entry {
  key: string;
  value: string;
  author: string;
  confidence: number;
}

class Blackboard {
  entries: Entry[] = [];
  post(author: string, facts: Fact[]): void {
    this.entries.push(...facts.map((f) => ({ key: f.key, value: f.value, author, confidence: f.confidence })));
  }
  conflicts(): Map<string, Entry[]> {
    const byKey = new Map<string, Entry[]>();
    for (const e of this.entries) byKey.set(e.key, [...(byKey.get(e.key) ?? []), e]);
    return new Map([...byKey].filter(([, v]) => new Set(v.map((e) => e.value)).size > 1));
  }
  resolved(): Map<string, Entry> {
    const best = new Map<string, Entry>();
    for (const e of this.entries) {
      // policy: highest confidence wins, first poster on ties
      const cur = best.get(e.key);
      if (!cur || e.confidence > cur.confidence) best.set(e.key, e);
    }
    return best;
  }
}

async function synthesize(client: Anthropic, outcomes: Outcome[], board: Blackboard, missing: string[], phase: string): Promise<string> {
  const payload = {
    results: outcomes.map((o) => o.result),
    blackboard: Object.fromEntries([...board.resolved()].map(([k, e]) => [k, { value: e.value, by: e.author, confidence: e.confidence }])),
    conflicts: Object.fromEntries([...board.conflicts()].map(([k, v]) => [k, v.map((e) => `${e.value} (${e.author}, ${e.confidence})`)])),
  };
  const note = missing.length ? `\nMISSING: ${missing.join(", ")}` : "";
  const msg = await client.messages.create({
    model: MODEL_OPUS,
    max_tokens: 1500,
    system:
      "You are the lead research orchestrator. Synthesise subagent results into a recommendation. " +
      "Surface conflicts and missing inputs explicitly; never fill gaps with guesses.",
    messages: [{ role: "user", content: `Task: ${TASK}\n\nSubagent results:\n${cj(payload)}${note}` }],
  });
  record(phase, msg);
  return textOf(msg);
}

// ---------------------------------------------------------------------------
// Topology chooser: when NOT to use multi-agent
// ---------------------------------------------------------------------------
interface TaskProfile {
  name: string;
  breadth?: number; // independent threads of work
  tightlyCoupled?: boolean; // do workers need each other's full context?
  exceedsOneContext?: boolean;
  highValue?: boolean; // can it pay for many times the tokens?
  latencySensitive?: boolean;
  fixedStages?: boolean;
  specialistHandoff?: boolean;
}

function recommendTopology(t: TaskProfile): [string, string] {
  if (t.specialistHandoff) return ["peer handoff", "conversation ownership moves to a specialist agent"];
  if (t.latencySensitive && !t.exceedsOneContext) return ["single agent", "planning + synthesis round trips would blow the latency budget"];
  if (t.tightlyCoupled) return ["single agent", "workers would need shared context; coordination costs and conflicting edits"];
  if (t.fixedStages) return ["sequential pipeline", "fixed stage order; each stage's output is the next stage's input"];
  if ((t.breadth ?? 1) >= 3 && t.highValue) return ["hierarchical multi-agent", "broad, parallelisable, valuable enough for the token multiplier"];
  if (t.exceedsOneContext) return ["hierarchical multi-agent", "context isolation: subagents compress what the lead must read"];
  return ["single agent", "not enough independent breadth to pay for extra tokens"];
}

// ---------------------------------------------------------------------------
// Offline responder
// ---------------------------------------------------------------------------
const PLAN_JSON = {
  subtasks: [
    { id: "s1", topic: "market", objective: "Size the EU mid-market retail segment and current AI-support adoption", model: "sonnet", boundaries: "No competitor analysis, no pricing", critical: false },
    { id: "s2", topic: "competitors", objective: "List direct competitors and their pricing model", model: "haiku", boundaries: "No market sizing beyond what sources state", critical: false },
    { id: "s3", topic: "cost", objective: "Estimate unit economics per ticket: AI vs human, incl. overheads", model: "sonnet", boundaries: "Use pilot data; no revenue forecast", critical: true },
    { id: "s4", topic: "regulatory", objective: "Identify EU AI Act and GDPR obligations for an AI support assistant", model: "sonnet", boundaries: "Obligations only; flag items for legal review", critical: true },
  ],
};
const QUERIES: Record<string, string> = {
  market: "EU mid-market retailers AI customer support adoption",
  competitors: "competitors vendors pricing addressable retailers",
  cost: "ticket cost human ai model usage integration",
  regulatory: "eu ai act gdpr dpia obligations chatbots data residency",
};
const RESULTS: Record<string, SubagentResult> = {
  market: { topic: "market", findings: ["~120k EU mid-market retailers (survey)", "AI-support adoption 18% in 2026, doubled since 2024", "24/7 and multilingual coverage drive adoption"],
    facts: [{ key: "eu_midmarket_retailers", value: "120000", confidence: 0.7 }, { key: "ai_support_adoption_pct", value: "18", confidence: 0.65 }],
    confidence: 0.7, open_questions: ["Survey method for the 120k figure"] },
  competitors: { topic: "competitors", findings: ["4 AI-first vendors target this segment", "Pricing per resolved conversation: $0.60-$1.20", "Analyst puts addressable base at ~95k"],
    facts: [{ key: "eu_midmarket_retailers", value: "95000", confidence: 0.5 }, { key: "direct_competitors", value: "4", confidence: 0.8 }],
    confidence: 0.65, open_questions: ["Competitor resolution rates"] },
  cost: { topic: "cost", findings: ["AI ticket ~$0.04 model cost with caching", "Human ticket ~$4.10 fully loaded", "Add 20-30% integration/QA in year one"],
    facts: [{ key: "ai_cost_per_ticket_usd", value: "0.04", confidence: 0.8 }, { key: "human_cost_per_ticket_usd", value: "4.10", confidence: 0.75 }],
    confidence: 0.75, open_questions: ["Share of tickets AI can fully resolve"] },
  regulatory: { topic: "regulatory", findings: ["AI Act transparency: disclose AI interaction", "GDPR DPIA likely required", "EU data processing often required"],
    facts: [{ key: "ai_act_obligation", value: "transparency (disclose AI)", confidence: 0.7 }, { key: "dpia_required", value: "likely", confidence: 0.8 }],
    confidence: 0.7, open_questions: ["Legal review of sector-specific rules"] },
};
const FULL_SYNTHESIS =
  "GO for a 2027 pilot, conditional on a DPIA. Market: 95k-120k EU mid-market retailers (sources " +
  "conflict; 120k used at 0.7 confidence - validate before the business case); adoption 18% and " +
  "doubling. Economics: ~$0.04 AI vs ~$4.10 human per ticket, plus 20-30% integration/QA in year one. " +
  "Competition: 4 AI-first vendors at $0.60-$1.20 per resolved conversation. Regulation: AI Act " +
  "transparency duties, DPIA likely, plan EU data processing.";
const PARTIAL_SYNTHESIS =
  "PARTIAL - NOT DECISION-READY. Regulatory analysis is missing (subagent failed), and it is a " +
  "critical input. Market (95k-120k retailers, 18% adoption) and economics (~$0.04 vs ~$4.10 per " +
  "ticket) look attractive. Re-run the regulatory subagent before any go/no-go.";
const BASELINE_ANSWER =
  "Recommendation: GO. About 120,000 EU mid-market retailers, 18% already use AI support, AI tickets " +
  "cost ~$0.04 vs $4.10 for humans, four competitors exist, and chatbots must disclose AI use.";

function responder(req: MockRequest): ResponderResult {
  const s = req.systemText;
  const u = req.lastUserText;
  if (s.includes("You plan research")) return JSON.stringify(PLAN_JSON);
  if (s.includes("research subagent")) {
    const topic = /Your topic: (\w+)/.exec(s)![1]!;
    if (!req.toolResults.length) return [textBlock("Searching the corpus."), toolUseBlock("search_corpus", { query: QUERIES[topic] })];
    return JSON.stringify(RESULTS[topic]);
  }
  if (s.includes("lead research orchestrator")) return u.includes("MISSING:") ? PARTIAL_SYNTHESIS : FULL_SYNTHESIS;
  if (s.includes("single research analyst")) return BASELINE_ANSWER;
  if (s.includes("extract commercial terms"))
    return "- 3-year term, $180k/year\n- 99.5% uptime SLA\n- Data stored in US-East\n- Price uplift capped at 7%/year\n- 90-day termination for convenience";
  if (s.includes("risk analyst"))
    return "1) US-East storage conflicts with EU data-residency needs (high). 2) 99.5% SLA allows ~44 h downtime/year (medium). 3) 7%/yr uplift = +14.5% by year 3 (medium).";
  if (s.includes("memo writer"))
    return "Decision: negotiate before signing. Require EU data residency and a 99.9% SLA; cap uplift at 3%. Termination terms are acceptable.";
  if (s.includes("triage agent"))
    return [
      textBlock("This is a billing issue; handing off."),
      toolUseBlock("transfer_to_billing_agent", { summary: "Upgraded to Team plan yesterday; charged twice; wants the duplicate refunded." }),
    ];
  if (s.includes("billing agent"))
    return "Thanks for your patience - I can see the duplicate Team-plan charge from yesterday. I have refunded it; it will appear on your statement within 5 business days.";
  return "OK";
}

/** Chaos testing: inject an upstream failure into matching requests (SDK raises APIConnectionError). */
function withFault(inner: Responder, predicate: (req: MockRequest) => boolean): Responder {
  return (req) => {
    if (predicate(req)) throw new Error("injected fault: upstream unavailable");
    return inner(req);
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
banner("D1.4 - Multi-Agent Systems and Orchestration");
const client = getClient(responder);

section("1. Orchestrator plans the work (Opus, structured output)");
console.log(`  task: ${TASK}`);
const planMsg = await client.messages.parse({
  model: MODEL_OPUS,
  max_tokens: 2000,
  output_config: { format: zodOutputFormat(Plan) },
  system:
    "You plan research for a team of subagents. For each subtask give an objective, the cheapest " +
    "adequate model, explicit boundaries (to avoid duplicated work) and whether it is critical.",
  messages: [{ role: "user", content: TASK }],
});
record("orchestrator: plan", planMsg);
const plan = planMsg.parsed_output;
if (!plan) {
  console.log("  no valid plan - abort");
  process.exit(0);
}
for (const st of plan.subtasks) {
  console.log(`  ${st.id} ${st.topic.padEnd(12)} model=${st.model.padEnd(6)} critical=${(st.critical ? "yes" : "no ").padEnd(3)} | ${st.objective}`);
  console.log(`     boundaries: ${st.boundaries}`);
}

section("2. Subagents: isolated context, own tool loop, condensed result");
let t0 = performance.now();
for (const st of plan.subtasks) await runSubagent(client, st, "timing baseline (sequential)"); // baseline: one after another
const seqMs = elapsedMs(t0);
t0 = performance.now();
const raw = await runAllParallel(client, plan, "subagents (parallel)");
const parMs = elapsedMs(t0);
const outcomes = raw.filter((o): o is Outcome => !(o instanceof Error));
for (const o of outcomes) {
  const ratio = o.consumedTokens / Math.max(1, o.returnedTokens);
  console.log(
    `  ${o.subtask.id} ${o.subtask.topic.padEnd(12)} ${MODEL_MAP[o.subtask.model].padEnd(18)} calls=${o.calls} ` +
      `sources=${o.sources.join(",")}  consumed=${o.consumedTokens} tok -> returned=${o.returnedTokens} tok (${ratio.toFixed(1)}x compression)`,
  );
}
console.log(
  `  wall-clock: sequential ~${Math.round(seqMs / 10) * 10} ms vs parallel ~${Math.round(parMs / 10) * 10} ms ` +
    `-> ${(seqMs / parMs).toFixed(1)}x faster (latency = slowest subagent, not the sum)`,
);

section("3. Shared state: blackboard (harness-mediated)");
const board = new Blackboard();
for (const o of outcomes) board.post(o.subtask.topic, o.result.facts);
for (const e of board.entries) console.log(`  ${e.author.padEnd(12)} posted ${e.key.padEnd(26)} = ${e.value.padEnd(28)} conf=${e.confidence.toFixed(2)}`);
for (const [key, entries] of board.conflicts()) {
  const winner = board.resolved().get(key)!;
  console.log(
    `  CONFLICT on ${key}: ` + entries.map((e) => `${e.value} (${e.author})`).join(" vs ") +
      ` -> keep ${winner.value} (higher confidence), surface the range in the synthesis`,
  );
}

section("4. Orchestrator synthesis (Opus reads condensed results only)");
console.log(`  ${await synthesize(client, outcomes, board, [], "orchestrator: synthesis")}`);

section("5. Token / cost multiplier vs a single agent");
const corpusText = CORPUS.map((s) => `[${s.id}] ${s.text}`).join("\n");
const single = await client.messages.create({
  model: MODEL_OPUS,
  max_tokens: 1500,
  system: "You are a single research analyst. Answer from the sources.",
  messages: [{ role: "user", content: `Task: ${TASK}\n\nSources:\n${corpusText}` }],
});
record("single agent baseline", single);
const multiPhases = ["orchestrator: plan", "subagents (parallel)", "orchestrator: synthesis"];
const multiTokens = multiPhases.reduce((s, p) => s + row(p).tokens, 0);
const multiCost = multiPhases.reduce((s, p) => s + row(p).cost, 0);
const singleRow = row("single agent baseline");
console.log(`  ${"phase".padEnd(32)}${"calls".padStart(6)}${"tokens".padStart(8)}${"cost $".padStart(11)}`);
for (const p of [...multiPhases, "single agent baseline"]) {
  const r = row(p);
  console.log(`  ${p.padEnd(32)}${String(r.calls).padStart(6)}${String(r.tokens).padStart(8)}${r.cost.toFixed(5).padStart(11)}`);
}
console.log(
  `  multi-agent total: ${multiTokens} tokens, $${multiCost.toFixed(5)}  ->  ` +
    `${(multiTokens / singleRow.tokens).toFixed(1)}x tokens, ${(multiCost / singleRow.cost).toFixed(1)}x cost of the single agent`,
);
console.log("  (cheaper subagent models soften the cost multiplier; the single agent also missed the 95k/120k conflict)");

section("6. Failure isolation (chaos test: regulatory subagent outage)");
const chaos = getClient(withFault(responder, (r) => r.systemText.includes("Your topic: regulatory")));
const chaosRaw = await runAllParallel(chaos, plan, "chaos run");
const ok: Outcome[] = [];
const failed: Subtask[] = [];
plan.subtasks.forEach((st, i) => {
  const o = chaosRaw[i]!;
  if (o instanceof Error) {
    failed.push(st);
    console.log(`  ${st.id} ${st.topic.padEnd(12)} FAILED: ${o.constructor.name} (siblings unaffected)`);
  } else {
    ok.push(o);
    console.log(`  ${st.id} ${st.topic.padEnd(12)} ok`);
  }
});
const criticalMissing = failed.filter((st) => st.critical).map((st) => st.topic);
const status = criticalMissing.length ? "PARTIAL (critical input missing)" : failed.length ? "DEGRADED" : "COMPLETE";
console.log(`  orchestrator policy -> status ${status}; options: retry subagent, re-plan, or report with gaps`);
const board2 = new Blackboard();
for (const o of ok) board2.post(o.subtask.topic, o.result.facts);
console.log(`  ${await synthesize(chaos, ok, board2, failed.map((st) => st.topic), "chaos run")}`);

section("7. Other topologies: sequential pipeline and peer handoff");
const proposal =
  "Vendor X proposes a 3-year contract at $180k/year for an AI support platform, 99.5% uptime SLA, " +
  "data stored in US-East, price uplift capped at 7%/year, 90-day termination for convenience.";
console.log("  pipeline: extractor (Haiku) -> risk analyst (Sonnet) -> memo writer (Sonnet)");
let stageInput = proposal;
let totalMs = 0;
const stages: [string, string, string][] = [
  ["extractor", MODEL_HAIKU, "You extract commercial terms as bullets."],
  ["risk analyst", MODEL_SONNET, "You are a risk analyst. Rank the top 3 risks."],
  ["memo writer", MODEL_SONNET, "You are a memo writer. Write a 2-line decision memo."],
];
for (const [name, model, system] of stages) {
  const ts = performance.now();
  const m = await client.messages.create({ model, max_tokens: 600, system, messages: [{ role: "user", content: stageInput }] });
  totalMs += elapsedMs(ts);
  record("pipeline", m);
  stageInput = textOf(m); // each stage sees ONLY the previous stage's output
  const lines = stageInput.split("\n");
  console.log(`    [${name}] ${lines[0]}${lines.length > 1 ? " ..." : ""}`);
}
console.log(`    latency is the SUM of stages: ~${Math.round(totalMs / 10) * 10} ms; a bad early stage poisons later ones`);

const userMsg = "I upgraded to the Team plan yesterday and was charged twice.";
const summarySchema: Anthropic.Tool.InputSchema = { type: "object", properties: { summary: { type: "string" } }, required: ["summary"], additionalProperties: false };
const transferTools: Anthropic.Tool[] = [
  { name: "transfer_to_billing_agent", description: "Hand the conversation to the billing specialist.", strict: true, input_schema: summarySchema },
  { name: "transfer_to_technical_agent", description: "Hand the conversation to the technical specialist.", strict: true, input_schema: summarySchema },
];
const triage = await client.messages.create({
  model: MODEL_HAIKU,
  max_tokens: 400,
  tools: transferTools,
  system: "You are the triage agent. Transfer to the right specialist with a one-line summary.",
  messages: [{ role: "user", content: userMsg }],
});
record("handoff", triage);
const handoff = triage.content.find((b): b is Anthropic.ToolUseBlock => b.type === "tool_use");
if (triage.stop_reason === "tool_use" && handoff) {
  const target = handoff.name.replace(/^transfer_to_/, "").replace(/_agent$/, "");
  const note = (handoff.input as { summary: string }).summary;
  console.log(
    `  peer handoff: triage (Haiku) -> ${target} (Sonnet); context passed = handoff note ` +
      `(~${estimateTokens(note)} tok), not the whole transcript`,
  );
  const reply = await client.messages.create({
    model: MODEL_SONNET,
    max_tokens: 600,
    system: `You are the ${target} agent. You now own this conversation end to end.`,
    messages: [{ role: "user", content: `${userMsg}\n\n<handoff from="triage">${note}</handoff>` }],
  });
  record("handoff", reply);
  console.log(`    ${target} agent: ${textOf(reply)}`);
}

section("8. When NOT to use multi-agent");
const profiles: TaskProfile[] = [
  { name: "Due diligence across market, competitors, cost, regulation", breadth: 4, highValue: true },
  { name: "Refactor a tightly coupled billing module", breadth: 3, tightlyCoupled: true, highValue: true },
  { name: "Live chat answer within 2 seconds", latencySensitive: true },
  { name: "Invoices: extract -> validate -> post to ERP", fixedStages: true },
  { name: "Support chat that may need billing or tech specialists", specialistHandoff: true },
  { name: "Summarise one 20-page report" },
];
for (const t of profiles) {
  const [topo, why] = recommendTopology(t);
  console.log(`  ${t.name.padEnd(58)} -> ${topo.padEnd(24)} (${why})`);
}

section("Summary");
console.log("  Multi-agent buys breadth, parallel speed and context isolation at a token multiplier.");
console.log("  Use it for broad, parallelisable, high-value work; give every subagent an objective, boundaries");
console.log("  and an output format; isolate failures; and keep tightly coupled or latency-critical work single-agent.");
