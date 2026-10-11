/**
 * D2.1 - Model Selection and Trade-offs
 * =====================================
 *
 * A model-selection engine an architect could actually run against a portfolio:
 *
 *   1. Model catalog: tier, latency profile, context, pricing, default effort.
 *   2. Models API lookup (live-only) - discover limits/capabilities at runtime.
 *   3. Workload profiles -> every (model x effort) configuration is checked against
 *      hard constraints (capability, latency SLA, context) and priced at volume
 *      (tokens x PRICING, cache reads, Batch discount, Haiku long-context tier).
 *   4. Portfolio decision table vs a "one model for everything" baseline.
 *   5. "Most capable model at lower effort" vs "cheaper model at max effort",
 *      judged on cost per SOLVED task, not cost per token.
 *   6. Routing cascade: Haiku triages difficulty -> route to Haiku/Sonnet/Opus.
 *   7. Same prompt through each model with the shared client -> measured latency + cost.
 *
 * Everything non-LLM here (constraint filter, cost model, router) is real code.
 * The heuristic numbers (tiers, effort multipliers) are ILLUSTRATIVE - the method
 * is the lesson; your own eval sweep supplies the real numbers.
 */

import type Anthropic from "@anthropic-ai/sdk";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  MODEL_FABLE,
  MODEL_HAIKU,
  MODEL_OPUS,
  MODEL_SONNET,
  PRICING,
  type MockRequest,
  type Price,
  banner,
  costUsd,
  elapsedMs,
  getClient,
  isLive,
  section,
  textOf,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");

// ---------------------------------------------------------------------------
// 1. Catalog. Context/max-output/default-effort are from the Oct 2026 model
//    table; tier + latency are ILLUSTRATIVE (latency mirrors the offline mock).
// ---------------------------------------------------------------------------
type Effort = "low" | "medium" | "high" | "xhigh" | "max";
const EFFORTS: Effort[] = ["low", "medium", "high", "xhigh", "max"];

// Thinking tokens are billed as output tokens, so effort mostly moves OUTPUT cost
// and latency. Multipliers and capability bonuses are illustrative assumptions.
const EFFORT_OUTPUT_MULT: Record<Effort, number> = { low: 0.5, medium: 1.0, high: 1.6, xhigh: 2.4, max: 3.5 };
const EFFORT_CAPABILITY_BONUS: Record<Effort, number> = { low: -0.5, medium: 0.0, high: 0.4, xhigh: 0.7, max: 0.9 };

const BATCH_DISCOUNT = 0.5; // Message Batches API: 50% off input + output, async (<= 24h)
const HAIKU_LONG_CONTEXT_THRESHOLD = 100_000; // Haiku 5.5 prompt > 100K tokens -> higher price tier
const HAIKU_LONG_CONTEXT_PRICE: Price = { input: 0.5, output: 2.5, cacheWrite: 0.625, cacheRead: 0.05 };

interface ModelProfile {
  id: string;
  name: string;
  tier: number; // 1-5 capability scale used by workload.capability_need
  ttftMs: number;
  msPerOutputToken: number;
  contextWindow: number;
  maxOutput: number;
  defaultEffort: Effort;
  bestFor: string;
}

const CATALOG: ModelProfile[] = [
  { id: MODEL_HAIKU, name: "Haiku 5.5", tier: 2, ttftMs: 250, msPerOutputToken: 3, contextWindow: 1_000_000, maxOutput: 128_000, defaultEffort: "medium", bestFor: "routing, classification, extraction, sub-agents" },
  { id: MODEL_SONNET, name: "Sonnet 5.5", tier: 3, ttftMs: 500, msPerOutputToken: 7, contextWindow: 1_000_000, maxOutput: 128_000, defaultEffort: "high", bestFor: "high-volume production, near-frontier coding" },
  { id: MODEL_OPUS, name: "Opus 5.5", tier: 4, ttftMs: 900, msPerOutputToken: 12, contextWindow: 1_000_000, maxOutput: 128_000, defaultEffort: "medium", bestFor: "default: complex reasoning, agents, coding" },
  { id: MODEL_FABLE, name: "Fable 5.1", tier: 5, ttftMs: 1500, msPerOutputToken: 18, contextWindow: 1_000_000, maxOutput: 128_000, defaultEffort: "high", bestFor: "hardest reasoning, long-horizon agents" },
];
const BY_ID: Record<string, ModelProfile> = Object.fromEntries(CATALOG.map((m) => [m.id, m]));

interface Workload {
  id: string;
  description: string;
  requests_per_month: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  cacheable_input_fraction: number;
  context_need_tokens: number;
  capability_need: number;
  latency_sla_ms: number | null;
  batchable: boolean;
  agent_turns: number;
}

interface Evaluation {
  model: ModelProfile;
  effort: Effort;
  effectiveCapability: number;
  latencyMs: number;
  costPerRequest: number;
  monthlyUsd: number;
  longContextTier: boolean;
  rejected: string[];
}

const feasibleEv = (ev: Evaluation): boolean => ev.rejected.length === 0;
const headroom = (ev: Evaluation, wl: Workload): number => ev.effectiveCapability - wl.capability_need;
const label = (ev: Evaluation): string => `${ev.model.name} @ ${ev.effort}`;

// ---------------------------------------------------------------------------
// Formatting helpers (kept identical in the Python twin)
// ---------------------------------------------------------------------------
const num = (x: number, digits = 0): string =>
  x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const pct = (x: number): string => `${Math.round(x * 100)}%`;

function usd(x: number): string {
  if (x >= 100) return `$${num(x, 0)}`;
  if (x >= 1) return `$${num(x, 2)}`;
  if (x >= 0.01) return `$${x.toFixed(4)}`;
  return `$${x.toFixed(6)}`;
}

function table(headers: string[], rows: string[][]): void {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i]!.length)));
  console.log("  " + headers.map((h, i) => h.padEnd(widths[i]!)).join("  "));
  console.log("  " + widths.map((w) => "-".repeat(w)).join("  "));
  for (const r of rows) console.log("  " + r.map((c, i) => c.padEnd(widths[i]!)).join("  "));
}

// ---------------------------------------------------------------------------
// 3. The selection engine: constraints first, then price at volume
// ---------------------------------------------------------------------------
/** Per-1M-token prices for one call. Haiku 5.5 has a long-context price tier. */
function unitPrices(modelId: string, promptTokens: number): Price {
  if (modelId === MODEL_HAIKU && promptTokens > HAIKU_LONG_CONTEXT_THRESHOLD) return { ...HAIKU_LONG_CONTEXT_PRICE };
  return { ...PRICING[modelId]! };
}

function costPerRequest(model: ModelProfile, wl: Workload, effort: Effort): number {
  const p = unitPrices(model.id, wl.context_need_tokens);
  // Steady state: the cacheable prefix is read from cache (write cost amortized away
  // at volume); the rest is billed at the base input price.
  const cached = wl.avg_input_tokens * wl.cacheable_input_fraction;
  const uncached = wl.avg_input_tokens - cached;
  const out = wl.avg_output_tokens * EFFORT_OUTPUT_MULT[effort];
  let cost = (uncached * p.input + cached * p.cacheRead + out * p.output) / 1_000_000;
  if (wl.batchable) cost *= 1 - BATCH_DISCOUNT; // nobody is waiting -> Batches API
  return cost;
}

function latencyMs(model: ModelProfile, wl: Workload, effort: Effort): number {
  // Agent tasks pay time-to-first-token once per turn plus generation time.
  const out = wl.avg_output_tokens * EFFORT_OUTPUT_MULT[effort];
  return wl.agent_turns * model.ttftMs + out * model.msPerOutputToken;
}

function evaluate(model: ModelProfile, wl: Workload, effort: Effort): Evaluation {
  const effCap = model.tier + EFFORT_CAPABILITY_BONUS[effort];
  const lat = latencyMs(model, wl, effort);
  const cpr = costPerRequest(model, wl, effort);
  const ev: Evaluation = {
    model,
    effort,
    effectiveCapability: effCap,
    latencyMs: lat,
    costPerRequest: cpr,
    monthlyUsd: cpr * wl.requests_per_month,
    longContextTier: model.id === MODEL_HAIKU && wl.context_need_tokens > HAIKU_LONG_CONTEXT_THRESHOLD,
    rejected: [],
  };
  // Hard constraints are filters, not weights: a cheap model that misses the
  // quality bar or the SLA is not "a bit worse", it is unusable.
  if (effCap + 1e-9 < wl.capability_need) ev.rejected.push("capability");
  if (wl.latency_sla_ms !== null && lat > wl.latency_sla_ms) ev.rejected.push("latency");
  if (wl.context_need_tokens > model.contextWindow) ev.rejected.push("context");
  if ((wl.avg_output_tokens * EFFORT_OUTPUT_MULT[effort]) / Math.max(1, wl.agent_turns) > model.maxOutput) ev.rejected.push("max_output");
  return ev;
}

/** Return [feasible configs sorted cheapest-first, all configs]. */
function select(wl: Workload): [Evaluation[], Evaluation[]] {
  const all = CATALOG.flatMap((m) => EFFORTS.map((e) => evaluate(m, wl, e)));
  const feasible = all.filter(feasibleEv);
  // Cheapest first; on a tie prefer more capability headroom.
  const r6 = (x: number): number => Math.round(x * 1e6) / 1e6;
  feasible.sort((a, b) => r6(a.monthlyUsd) - r6(b.monthlyUsd) || headroom(b, wl) - headroom(a, wl));
  return [feasible, all];
}

interface CascadeRequest {
  id: string;
  text: string;
  expected: string;
  offline_mock: { difficulty: string; confidence: number; reason: string; answer: string };
}
interface WorkloadFile {
  workloads: Workload[];
  research_agent_eval: {
    _note: string;
    workload_id: string;
    human_fallback_cost_usd: number;
    candidates: { model: string; effort: Effort; pass_rate: number }[];
  };
  cascade_requests: CascadeRequest[];
}

// ---------------------------------------------------------------------------
// 6. Routing cascade
// ---------------------------------------------------------------------------
const DIFFICULTY_ORDER = ["simple", "moderate", "complex"] as const;
type Difficulty = (typeof DIFFICULTY_ORDER)[number];
const ROUTES: Record<Difficulty, [string, Effort]> = {
  simple: [MODEL_HAIKU, "low"],
  moderate: [MODEL_SONNET, "medium"],
  complex: [MODEL_OPUS, "high"],
};
const CONFIDENCE_FLOOR = 0.75; // below this, escalate one tier: misrouting UP costs cents, DOWN costs quality

const TRIAGE_SYSTEM =
  "You are a difficulty triage classifier for a customer-operations assistant.\n" +
  "Classify the request inside <request> tags:\n" +
  "- simple: one lookup or one well-specified action, no judgment\n" +
  "- moderate: summarize/draft/compare with a few constraints\n" +
  "- complex: multi-step diagnosis, planning, or reasoning across many facts\n" +
  "Return JSON only. confidence is your probability (0-1) that the label is right.";
const TRIAGE_SCHEMA = {
  type: "object",
  properties: {
    difficulty: { type: "string", enum: [...DIFFICULTY_ORDER] },
    confidence: { type: "number" },
    reason: { type: "string" },
  },
  required: ["difficulty", "confidence", "reason"],
  additionalProperties: false,
};
const ANSWER_SYSTEM = "You are a customer operations assistant for an online retailer. Answer concisely and accurately.";
const PROBE_SYSTEM = "Latency probe. Answer in exactly three short bullet points.";
const PROBE_PROMPT = "Why do payment APIs need idempotency keys?";
const PROBE_ANSWER =
  "- Retries after timeouts are safe: the same key never charges twice.\n" +
  "- The server can return the original result instead of re-executing.\n" +
  "- Clients can retry aggressively without coordinating state.";

/** Offline stand-in for Claude: canned but realistic replies keyed on the request. */
function makeResponder(cascade: CascadeRequest[]) {
  const find = (text: string): CascadeRequest | undefined => cascade.find((r) => text.includes(r.text));
  return (req: MockRequest): string => {
    if (req.systemText.includes("difficulty triage")) {
      const r = find(req.lastUserText);
      const mock = r ? r.offline_mock : { difficulty: "complex", confidence: 0.5, reason: "unknown" };
      return JSON.stringify({ difficulty: mock.difficulty, confidence: mock.confidence, reason: mock.reason });
    }
    if (req.systemText.includes("Latency probe")) return PROBE_ANSWER;
    const r = find(req.lastUserText);
    return r ? r.offline_mock.answer : "I can help with that.";
  };
}

interface Triage {
  difficulty: Difficulty;
  confidence: number;
  reason: string;
  msg: Anthropic.Message;
}

async function triage(client: Anthropic, text: string): Promise<Triage> {
  const msg = await client.messages.create({
    model: MODEL_HAIKU,
    max_tokens: 300,
    system: TRIAGE_SYSTEM,
    // Structured outputs guarantee parseable JSON; low effort keeps the router fast.
    output_config: { effort: "low", format: { type: "json_schema", schema: TRIAGE_SCHEMA } },
    messages: [{ role: "user", content: `<request>${text}</request>` }],
  });
  if (msg.stop_reason !== "end_turn") {
    // refusal / max_tokens -> fail safe to the strongest route
    return { difficulty: "complex", confidence: 0, reason: `triage stop_reason=${msg.stop_reason}`, msg };
  }
  const data = JSON.parse(textOf(msg)) as Omit<Triage, "msg">;
  return { ...data, msg };
}

function route(t: Triage): [Difficulty, string, Effort, boolean] {
  let level: Difficulty = t.difficulty;
  let escalated = false;
  if (t.confidence < CONFIDENCE_FLOOR && level !== "complex") {
    level = DIFFICULTY_ORDER[DIFFICULTY_ORDER.indexOf(level) + 1]!;
    escalated = true;
  }
  const [model, effort] = ROUTES[level];
  return [level, model, effort, escalated];
}

async function answer(client: Anthropic, model: string, effort: Effort, text: string): Promise<[Anthropic.Message, number]> {
  const t0 = performance.now();
  const msg = await client.messages.create({
    model,
    max_tokens: 2048,
    system: ANSWER_SYSTEM,
    output_config: { effort },
    messages: [{ role: "user", content: text }],
  });
  return [msg, elapsedMs(t0)];
}

// ---------------------------------------------------------------------------
// 2. Models API (live-only)
// ---------------------------------------------------------------------------
const MODELS_API_SNIPPET = `\
for await (const m of client.models.list()) {        // auto-paginates
  const caps = m.capabilities;                       // typed ModelCapabilities | null
  console.log(m.id, m.max_input_tokens, m.max_tokens,
    caps?.effort.max.supported,
    caps?.thinking.types.adaptive.supported,
    caps?.batch.supported,
    caps?.context_management.compact_20260112?.supported);
}
const info = await client.models.retrieve("claude-opus-5-5");
`;

async function modelsApiLookup(client: Anthropic): Promise<void> {
  const prefixes = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable-5"];
  for await (const m of client.models.list()) {
    if (!prefixes.some((p) => m.id.startsWith(p))) continue;
    const caps = m.capabilities;
    if (!caps) {
      console.log(`  ${m.id.padEnd(20)} ctx=${m.max_input_tokens} max_out=${m.max_tokens} (no capability data)`);
      continue;
    }
    console.log(
      `  ${m.id.padEnd(20)} ctx=${m.max_input_tokens} max_out=${m.max_tokens} ` +
        `effort.max=${caps.effort.max.supported} adaptive=${caps.thinking.types.adaptive.supported} ` +
        `batch=${caps.batch.supported} structured=${caps.structured_outputs.supported}`,
    );
  }
  const info = await client.models.retrieve(MODEL_OPUS);
  console.log(`  retrieve(${MODEL_OPUS}) -> display_name='${info.display_name}' max_input_tokens=${info.max_input_tokens}`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
banner("D2.1 - Model Selection and Trade-offs");
const raw = JSON.parse(readFileSync(path.join(DATA, "workloads.json"), "utf-8")) as WorkloadFile;
const workloads = raw.workloads;
const cascade = raw.cascade_requests;
const client = getClient(makeResponder(cascade));

// ------------------------------------------------------------------ 1
section("1. Model catalog (Oct 2026 prices; tier/latency illustrative)");
table(
  ["model", "id", "tier", "in/out per 1M", "cache rd", "TTFT", "ctx", "def.effort", "best for"],
  CATALOG.map((m) => {
    const p = PRICING[m.id]!;
    return [
      m.name, m.id, m.tier.toFixed(0), `$${p.input.toFixed(2)}/$${p.output.toFixed(2)}`, `$${p.cacheRead.toFixed(2)}`,
      `${m.ttftMs.toFixed(0)} ms`, `${Math.floor(m.contextWindow / 1000)}K`, m.defaultEffort, m.bestFor,
    ];
  }),
);
console.log("  Batch API: -50% on input+output. Haiku 5.5 prompts >100K tokens bill at $0.50/$2.50.");
console.log("  Effort is a cost/depth dial on ONE model: thinking tokens bill as output tokens.");

// ------------------------------------------------------------------ 2
section("2. Models API lookup - capabilities at runtime");
console.log("  Request code (@anthropic-ai/sdk):");
for (const line of MODELS_API_SNIPPET.trimEnd().split("\n")) console.log("    " + line);
if (isLive()) {
  await modelsApiLookup(client);
} else {
  console.log("  [live-only] The offline mock does not serve /v1/models; the catalog above is the");
  console.log("  cached table. In production, gate features on capabilities (e.g. effort.max,");
  console.log("  context_management.compact_20260112) instead of hard-coding model names.");
}

// ------------------------------------------------------------------ 3
section("3. Workload profiles -> constraint filter -> priced at volume");
const decisions: [Workload, Evaluation][] = [];
for (const wl of workloads) {
  const [feasible, all] = select(wl);
  const reasons: Record<string, number> = {};
  for (const ev of all) for (const r of ev.rejected) reasons[r] = (reasons[r] ?? 0) + 1;
  const sla = wl.latency_sla_ms !== null ? `${num(wl.latency_sla_ms)} ms` : "none";
  console.log(`\n  [${wl.id}] ${wl.description}`);
  console.log(
    `    volume=${num(wl.requests_per_month)}/mo  in=${num(wl.avg_input_tokens)} out=${num(wl.avg_output_tokens)} tok  ` +
      `cacheable=${pct(wl.cacheable_input_fraction)}  need=${wl.capability_need.toFixed(1)}  SLA=${sla}  ` +
      `batch=${wl.batchable ? "yes" : "no"}  turns=${wl.agent_turns}`,
  );
  const rej = Object.keys(reasons)
    .sort()
    .map((k) => `${k} ${reasons[k]}`)
    .join(", ");
  console.log(`    feasible configs: ${feasible.length} of ${all.length}  (rejected: ${rej || "none"})`);
  if (feasible.length === 0) {
    console.log("    NO FEASIBLE CONFIG - relax the SLA, decompose the task, or raise the budget.");
    continue;
  }
  const best = feasible[0]!;
  decisions.push([wl, best]);
  const hr = headroom(best, wl);
  console.log(
    `    RECOMMEND  ${label(best).padEnd(20)} ${usd(best.monthlyUsd)}/mo  ~${num(best.latencyMs)} ms  ` +
      `headroom ${hr >= 0 ? "+" : ""}${hr.toFixed(1)}`,
  );
  const alt = feasible.slice(1).find((ev) => ev.model.id !== best.model.id);
  if (alt) {
    console.log(
      `    runner-up  ${label(alt).padEnd(20)} ${usd(alt.monthlyUsd)}/mo  ~${num(alt.latencyMs)} ms  ` +
        `(+${pct(alt.monthlyUsd / best.monthlyUsd - 1)} cost)`,
    );
  }
  if (best.longContextTier) console.log("    note: Haiku long-context price tier applied (prompt >100K tokens) - still cheapest here.");
  if (wl.batchable) console.log("    note: no interactive SLA -> Message Batches API halves the bill.");
  // The interesting case: a MORE capable model at LOWER effort beats a cheaper
  // model that needs max effort to reach the bar.
  const lowerTier = feasible.filter((ev) => ev.model.tier < best.model.tier);
  if (lowerTier.length) {
    const cheapestLower = lowerTier.reduce((a, b) => (b.monthlyUsd < a.monthlyUsd ? b : a));
    console.log(
      `    note: ${label(best)} beats ${label(cheapestLower)} by ` +
        `${usd(cheapestLower.monthlyUsd - best.monthlyUsd)}/mo - the stronger model needs fewer thinking tokens.`,
    );
  }
}

// ------------------------------------------------------------------ 4
section("4. Portfolio decision table vs 'Opus 5.5 @ medium for everything'");
{
  const rows: string[][] = [];
  let totalRec = 0;
  let totalBase = 0;
  let baseFailures = 0;
  for (const [wl, best] of decisions) {
    const base = evaluate(BY_ID[MODEL_OPUS]!, wl, "medium");
    totalRec += best.monthlyUsd;
    totalBase += base.monthlyUsd;
    baseFailures += feasibleEv(base) ? 0 : 1;
    const baseNote = feasibleEv(base) ? "ok" : "FAILS " + base.rejected.join("+");
    rows.push([wl.id, best.model.name, best.effort, wl.batchable ? "yes" : "no", usd(best.monthlyUsd), usd(base.monthlyUsd), baseNote]);
  }
  table(["workload", "model", "effort", "batch", "$/mo (rec)", "$/mo (Opus med)", "Opus med vs constraints"], rows);
  console.log(`  Portfolio: ${usd(totalRec)}/mo right-sized vs ${usd(totalBase)}/mo single-model (${pct(1 - totalRec / totalBase)} lower).`);
  console.log(`  The single-model baseline also violates ${baseFailures} of ${decisions.length} workloads' constraints:`);
  console.log("  too slow for the real-time SLA, not capable enough for the hardest agents.");
}

// ------------------------------------------------------------------ 5
section("5. Capable model @ low effort vs cheaper model @ max effort");
{
  const evCfg = raw.research_agent_eval;
  const wl = workloads.find((w) => w.id === evCfg.workload_id)!;
  const fallback = evCfg.human_fallback_cost_usd;
  console.log(`  Workload: ${wl.id}. ${evCfg._note}`);
  const scored: { label: string; attempt: number; pass: number; perSolved: number; expected: number }[] = [];
  for (const c of evCfg.candidates) {
    const m = BY_ID[c.model]!;
    const attempt = costPerRequest(m, wl, c.effort);
    scored.push({
      label: `${m.name} @ ${c.effort}`,
      attempt,
      pass: c.pass_rate,
      perSolved: attempt / c.pass_rate,
      expected: attempt + (1 - c.pass_rate) * fallback,
    });
  }
  table(
    ["config", "$/attempt", "pass rate", "$/solved task", `$/task incl $${fallback.toFixed(0)} human fallback`],
    scored.map((s) => [s.label, usd(s.attempt), pct(s.pass), usd(s.perSolved), usd(s.expected)]),
  );
  const minBy = (k: "attempt" | "perSolved" | "expected") => scored.reduce((a, b) => (b[k] < a[k] ? b : a));
  console.log(`  Cheapest per attempt : ${minBy("attempt").label}`);
  console.log(`  Cheapest per SOLVED  : ${minBy("perSolved").label}`);
  console.log(`  Cheapest incl. errors: ${minBy("expected").label}`);
  console.log("  Lesson: per-token price lists do not rank models. Price the solved task (and the cost");
  console.log("  of a failure) on your own eval; sweep effort before switching models.");
}

// ------------------------------------------------------------------ 6
section("6. Routing cascade: Haiku triages difficulty -> route");
{
  console.log(
    `  Routes: simple->Haiku 5.5@low, moderate->Sonnet 5.5@medium, complex->Opus 5.5@high; confidence < ${CONFIDENCE_FLOOR} escalates one tier.`,
  );
  const rows: string[][] = [];
  let routedCost = 0;
  let baseCost = 0;
  let routedLat = 0;
  let baseLat = 0;
  let correct = 0;
  for (const r of cascade) {
    const t = await triage(client, r.text);
    const [, model, effort, escalated] = route(t);
    const triCost = costUsd(MODEL_HAIKU, t.msg.usage);
    const [msg, lat] = await answer(client, model, effort, r.text);
    if (msg.stop_reason !== "end_turn") console.log(`  ${r.id}: stop_reason=${msg.stop_reason} - would retry on a stronger route`);
    const c = triCost + costUsd(model, msg.usage);
    const [bmsg, blat] = await answer(client, MODEL_OPUS, "medium", r.text);
    const b = costUsd(MODEL_OPUS, bmsg.usage);
    routedCost += c;
    baseCost += b;
    routedLat += lat;
    baseLat += blat;
    if (t.difficulty === r.expected) correct += 1;
    rows.push([
      r.id, r.expected, `${t.difficulty} (${t.confidence.toFixed(2)})`,
      `${BY_ID[model]!.name}@${effort}${escalated ? " ^" : ""}`, usd(c), `${num(lat)} ms`, usd(b),
    ]);
  }
  table(["req", "expected", "triage (conf)", "routed to", "$ routed", "latency", "$ all-Opus"], rows);
  const n = cascade.length;
  console.log(`  Triage accuracy ${correct}/${n}  ('^' = escalated for low confidence - safe direction)`);
  console.log(
    `  Routed total ${usd(routedCost)} (incl. triage) vs all-Opus ${usd(baseCost)} -> ` +
      `${pct(1 - routedCost / baseCost)} cheaper; mean latency ${num(routedLat / n)} ms vs ${num(baseLat / n)} ms`,
  );
  console.log("  Caveat: route BETWEEN conversations. Switching model inside one conversation");
  console.log("  cold-starts the prompt cache (caches are model-scoped) - spawn a sub-agent instead.");
}

// ------------------------------------------------------------------ 7
section("7. Same prompt, every model - measured latency and cost");
{
  // Warm-up call: the first request pays client/connection setup; exclude it from timings.
  await client.messages.create({ model: MODEL_HAIKU, max_tokens: 64, messages: [{ role: "user", content: "ping" }] });
  const rows: string[][] = [];
  for (const m of CATALOG) {
    const t0 = performance.now();
    const msg = await client.messages.create({
      model: m.id,
      max_tokens: 1024,
      system: PROBE_SYSTEM,
      output_config: { effort: "low" },
      messages: [{ role: "user", content: PROBE_PROMPT }],
    });
    const ms = elapsedMs(t0);
    const c = costUsd(m.id, msg.usage);
    rows.push([m.name, `${num(ms)} ms`, String(msg.usage.input_tokens), String(msg.usage.output_tokens), usd(c), usd(c * 1_000_000)]);
  }
  table(["model", "latency", "in tok", "out tok", "$/request", "$ per 1M requests"], rows);
  const note = isLive() ? "real measurements" : "offline latency is the shared mock's simulated profile";
  console.log(`  (${note}; output text is identical, so the spread is pure model price/speed.)`);
}

section("Takeaways");
console.log("  * Filter on hard constraints (quality bar, SLA, context) BEFORE comparing price.");
console.log("  * Price at volume: tokens x price x requests, plus cache reads and Batch discounts.");
console.log("  * Effort is the first dial; a stronger model at lower effort can be the cheapest config.");
console.log("  * Cascades (cheap triage -> right-sized model) cut cost when most traffic is simple.");
