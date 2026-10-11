/**
 * D3.4 - Observability and Monitoring at Scale
 * Exam objective: Analyze observability challenges and select monitoring strategies at scale.
 *
 * Same steps as python/main.py: a seeded day of LLM traffic (10,000 requests, 8 tenants,
 * 4 features, 3 models, 2,000 users) with four injected incidents
 *   INC1 Sonnet latency spike (10:00-11:00)      INC2 initech agent loop / cost runaway (15:00)
 *   INC3 umbrella/extract refusal rise (16:00)   INC4 summarize quality drop after a deploy (20:00)
 * and a comparison of monitoring strategies: histograms vs averages, bounded cardinality +
 * exemplars, head vs tail trace sampling, W3C trace context + loop detection, per-tenant
 * cost attribution, SLO burn-rate vs static alerts, and sampled LLM-as-judge quality
 * monitoring (real calls through the shared SDK client).
 * Scale and strategy selection; D4.6 covers instrumenting a single app.
 */

import { createHash } from "node:crypto";
import type Anthropic from "@anthropic-ai/sdk";
import {
  MODEL_HAIKU,
  MODEL_OPUS,
  MODEL_SONNET,
  PRICING,
  banner,
  costUsd,
  estimateTokens,
  getClient,
  isLive,
  section,
  textBlock,
  textOf,
  type Json,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const SEED = 3404;
const N_REQUESTS = 10_000;
const MINUTES = 1440;
const N_USERS = 2000;
const TENANTS: [string, number][] = [
  ["acme", 0.24], ["globex", 0.18], ["initech", 0.14], ["umbrella", 0.12],
  ["hooli", 0.1], ["stark", 0.09], ["wayne", 0.07], ["wonka", 0.06],
];
const FEATURES: [string, number][] = [["chat", 0.45], ["summarize", 0.25], ["extract", 0.18], ["agent", 0.12]];
const FEATURE_MODEL: Record<string, string> = { chat: MODEL_SONNET, summarize: MODEL_SONNET, extract: MODEL_HAIKU, agent: MODEL_OPUS };
const TOKENS: Record<string, [number, number]> = { chat: [1200, 300], summarize: [4000, 250], extract: [2500, 150], agent: [6000, 600] };
const LAT: Record<string, [number, number]> = { [MODEL_HAIKU]: [250, 3], [MODEL_SONNET]: [500, 7], [MODEL_OPUS]: [900, 12] }; // ttft ms, ms/token
const SLO_LATENCY_MS: Record<string, number> = { chat: 8000, summarize: 15000, extract: 5000, agent: 600000 }; // agents run async
const TOOL_MS = 200;
const CONTEXT_GROWTH = 800; // tokens appended to the agent context per looping step
const ERRORS = new Set(["overloaded_529", "api_error_500", "rate_limit_429"]);
const SHORT: Record<string, string> = { [MODEL_HAIKU]: "haiku", [MODEL_SONNET]: "sonnet", [MODEL_OPUS]: "opus" };

// ---------------------------------------------------------------------------
// Deterministic helpers (identical to the Python version)
// ---------------------------------------------------------------------------
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

const sha = (text: string): string => createHash("sha256").update(text).digest("hex");
/** Deterministic [0,1) from an id: the basis of consistent (trace-id) sampling. */
const hash01 = (key: string, salt: string): number => parseInt(sha(salt + key).slice(0, 8), 16) / 4294967296;

function weighted(u: number, items: [string, number][]): string {
  let cum = 0;
  for (const [name, w] of items) {
    cum += w;
    if (u < cum) return name;
  }
  return items[items.length - 1]![0];
}

const hhmm = (m: number): string => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
const fmtInt = (n: number): string => Math.round(n).toLocaleString("en-US");
const f0 = (x: number, w: number): string => x.toFixed(0).padStart(w);
/** JSON like Python's json.dumps (", " / ": ") so judge outputs (and token counts) match. */
function pyDumps(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(pyDumps).join(", ")}]`;
  if (value && typeof value === "object")
    return `{${Object.entries(value as Json).map(([k, v]) => `${JSON.stringify(k)}: ${pyDumps(v)}`).join(", ")}}`;
  return JSON.stringify(value);
}

function pctExact(values: number[], p: number): number {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.max(0, Math.ceil((p / 100) * s.length) - 1)]!;
}

function wilson(passes: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [0, 1];
  const p = passes / n;
  const denom = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

// ---------------------------------------------------------------------------
// 1. Traffic simulation with injected incidents
// ---------------------------------------------------------------------------
interface Req {
  idx: number;
  minute: number;
  traceId: string;
  tenant: string;
  feature: string;
  model: string;
  user: string;
  steps: number;
  inputTokens: number;
  outputTokens: number;
  latencyMs: number;
  status: string;
  cost: number;
  good: boolean; // hidden ground truth for answer quality
  loop: boolean; // ground truth: INC2 agent loop
}
const isSlow = (r: Req): boolean => r.latencyMs > SLO_LATENCY_MS[r.feature]!;
// SLO "bad event": server error or too slow (refusals are valid responses)
const isBad = (r: Req): boolean => ERRORS.has(r.status) || isSlow(r);

const diurnal = (m: number): number => 0.3 + 0.7 * Math.max(0, 1 - Math.abs(m - 840) / 600);

function simulate(rng: Mulberry32): Req[] {
  const reqs: Req[] = [];
  for (let idx = 0; idx < N_REQUESTS; idx++) {
    let m: number;
    for (;;) {
      // rejection sampling against a diurnal curve (peak 14:00)
      m = Math.floor(rng.next() * MINUTES);
      if (rng.next() < diurnal(m)) break;
    }
    const tenant = weighted(rng.next(), TENANTS);
    const feature = weighted(rng.next(), FEATURES);
    const model = FEATURE_MODEL[feature]!;
    const user = `u${String(Math.floor(rng.next() * N_USERS)).padStart(4, "0")}`;
    const [baseIn, baseOut] = TOKENS[feature]!;
    const inTok = Math.floor(baseIn * (0.6 + 0.8 * rng.next()));
    const outTok = Math.floor(baseOut * (0.5 + rng.next()));
    const uSteps = rng.next();
    const loop = feature === "agent" && tenant === "initech" && m >= 900; // INC2
    const steps = feature !== "agent" ? 1 : loop ? 25 + Math.floor(uSteps * 20) : 3 + Math.floor(uSteps * 5);
    const [ttft, decode] = LAT[model]!;
    let perCall = (ttft + decode * outTok) * (0.85 + 0.3 * rng.next());
    const uTail = rng.next();
    const uTail2 = rng.next();
    const uSpike = rng.next();
    if (uTail < 0.01) perCall *= 2 + 2 * uTail2; // heavy tail
    let latency = perCall * steps + TOOL_MS * (steps - 1);
    if (model === MODEL_SONNET && m >= 600 && m < 660) latency *= 3 + 2 * uSpike; // INC1
    const totalIn = steps * inTok + (loop ? (CONTEXT_GROWTH * steps * (steps + 1)) / 2 : 0);
    const totalOut = steps * outTok;
    const cost = (totalIn * PRICING[model]!.input + totalOut * PRICING[model]!.output) / 1e6;
    let pRefusal = 0.005;
    if (tenant === "umbrella" && feature === "extract" && m >= 960) pRefusal = 0.005 + 0.495 * Math.min(1, (m - 960) / 60); // INC3 (bad deploy)
    const u = rng.next();
    let status: string;
    if (u < 0.001) status = "overloaded_529";
    else if (u < 0.0015) status = "api_error_500";
    else if (u < 0.002) status = "rate_limit_429";
    else if (u < 0.002 + pRefusal) status = "refusal";
    else if (u < 0.005 + pRefusal) status = "max_tokens";
    else status = "ok";
    const pGood = feature === "summarize" && m >= 1200 ? 0.45 : 0.93; // INC4
    const good = rng.next() < pGood;
    reqs.push({
      idx, minute: m, traceId: sha(`trace-${idx}`).slice(0, 32), tenant, feature, model, user, steps,
      inputTokens: totalIn, outputTokens: totalOut, latencyMs: latency, status, cost, good, loop,
    });
  }
  return reqs;
}

// ---------------------------------------------------------------------------
// 2. Histogram aggregation
// ---------------------------------------------------------------------------
const BUCKETS = [100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000];

function histogram(values: number[]): number[] {
  const counts = new Array<number>(BUCKETS.length + 1).fill(0);
  for (const v of values) {
    let i = 0;
    while (i < BUCKETS.length && v > BUCKETS[i]!) i++;
    counts[i]!++;
  }
  return counts;
}

/** Percentile from bucket counts with linear interpolation (how metric backends do it). */
function pctHist(counts: number[], p: number): number {
  const rank = (p / 100) * counts.reduce((a, b) => a + b, 0);
  let cum = 0;
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i]!;
    if (c && cum + c >= rank) {
      const lower = i > 0 ? BUCKETS[i - 1]! : 0;
      if (i === BUCKETS.length) return lower;
      return lower + ((BUCKETS[i]! - lower) * (rank - cum)) / c;
    }
    cum += c;
  }
  return BUCKETS[BUCKETS.length - 1]!;
}

// ---------------------------------------------------------------------------
// 5. Trace model (W3C trace context)
// ---------------------------------------------------------------------------
interface Span {
  spanId: string;
  parentId: string | null;
  name: string;
  startMs: number;
  durMs: number;
  attrs: Record<string, string | number>;
}

function buildAgentTrace(r: Req): Span[] {
  const spans: Span[] = [];
  const add = (name: string, parent: string | null, start: number, dur: number, attrs: Record<string, string | number>): string => {
    const spanId = sha(`span-${r.idx}-${spans.length}`).slice(0, 16);
    spans.push({ spanId, parentId: parent, name, startMs: start, durMs: dur, attrs });
    return spanId;
  };
  const root = add("POST /v1/agent", null, 0, r.latencyMs, { tenant: r.tenant, feature: r.feature });
  const orch = add("agent.orchestrator", root, 2, r.latencyMs - 4, { "gen_ai.request.model": r.model });
  const perCall = (r.latencyMs - TOOL_MS * (r.steps - 1)) / r.steps;
  const tools = ["search_orders{order_id=O-77}", "get_shipment{order_id=O-77}", "draft_reply{ticket=T-9}"];
  let t = 4;
  for (let i = 0; i < r.steps; i++) {
    add(`llm.call ${SHORT[r.model]}`, orch, t, perCall, { step: i + 1, "anthropic.request_id": "req_" + sha(`req-${r.idx}-${i}`).slice(0, 12) });
    t += perCall;
    if (i < r.steps - 1) {
      add(`tool.${r.loop ? tools[0] : tools[i % tools.length]}`, orch, t, TOOL_MS, {});
      t += TOOL_MS;
    }
  }
  return spans;
}

// ---------------------------------------------------------------------------
// 7. Alerting: static threshold vs multi-window burn rate
// ---------------------------------------------------------------------------
const SLO_TARGET = 0.995;
const BUDGET = 1 - SLO_TARGET;

/** Collapse per-evaluation booleans into [startMinute, endMinute] alert episodes. */
function episodes(firing: [number, boolean][]): [number, number][] {
  const out: [number, number][] = [];
  let start: number | null = null;
  let last = 0;
  for (const [minute, on] of firing) {
    if (on && start === null) start = minute;
    if (!on && start !== null) {
      out.push([start, last]);
      start = null;
    }
    last = minute;
  }
  if (start !== null) out.push([start, last]);
  return out;
}

// ---------------------------------------------------------------------------
// 8. LLM-as-judge (real calls through the shared client)
// ---------------------------------------------------------------------------
const JUDGE_SYSTEM =
  "You are a strict faithfulness judge. Compare the SUMMARY with the SOURCE. " +
  "A summary is faithful only if every number and claim in it is supported by the source.";
const JUDGE_SCHEMA = {
  type: "object",
  properties: {
    faithful: { type: "boolean" },
    score: { type: "integer", enum: [1, 2, 3, 4, 5] },
    issue: { type: "string" },
  },
  required: ["faithful", "score", "issue"],
  additionalProperties: false,
};

function makeDoc(r: Req): [string, string] {
  const h = Buffer.from(sha(`doc-${r.idx}`), "hex");
  const [q, growth, revenue, churn] = [1 + (h[0]! % 4), 5 + (h[1]! % 20), 10 + (h[2]! % 90), 1 + (h[3]! % 9)];
  const source = `In Q${q}, revenue grew ${growth}% to $${revenue}M while churn fell to ${churn}%. The board approved a buyback.`;
  const shownGrowth = r.good ? growth : growth + 7; // an unfaithful summary hallucinates a number
  return [source, `Q${q}: revenue up ${shownGrowth}% to $${revenue}M; churn down to ${churn}%.`];
}

function judgeResponder(req: MockRequest): ResponderResult {
  const text = req.lastUserText;
  const src = text.match(/<source>(.*?)<\/source>/s)?.[1] ?? "";
  const summ = text.match(/<summary>(.*?)<\/summary>/s)?.[1] ?? "";
  const nums = (s: string) => new Set(s.match(/\d+(?:\.\d+)?/g) ?? []);
  const srcNums = nums(src);
  const unsupported = [...nums(summ)].filter((x) => !srcNums.has(x)).sort((a, b) => Number(a) - Number(b));
  const verdict = unsupported.length
    ? { faithful: false, score: 2, issue: `summary states ${unsupported.join(", ")} which the source does not support` }
    : { faithful: true, score: 5, issue: "none" };
  const body = pyDumps(verdict);
  return { content: [textBlock(body)], usage: { output_tokens: estimateTokens(body) } };
}

/** Bounded concurrency helper. */
function limiter(n: number) {
  let active = 0;
  const queue: (() => void)[] = [];
  return async <T>(fn: () => Promise<T>): Promise<T> => {
    while (active >= n) await new Promise<void>((r) => queue.push(r));
    active++;
    try {
      return await fn();
    } finally {
      active--;
      queue.shift()?.();
    }
  };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
banner("D3.4 - Observability and Monitoring at Scale");

section("1. Simulated day of traffic (seeded) with 4 injected incidents");
const reqs = simulate(new Mulberry32(SEED));
const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);
const totalCost = sum(reqs.map((r) => r.cost));
const nErr = reqs.filter((r) => ERRORS.has(r.status)).length;
const distinct = (f: (r: Req) => string): number => new Set(reqs.map(f)).size;
console.log(
  `${fmtInt(reqs.length)} requests | ${distinct((r) => r.tenant)} tenants | ${distinct((r) => r.feature)} features | ` +
    `${distinct((r) => r.model)} models | ${fmtInt(distinct((r) => r.user))} distinct users`,
);
console.log(
  `LLM calls (agent steps included): ${fmtInt(sum(reqs.map((r) => r.steps)))} | tokens in/out: ` +
    `${(sum(reqs.map((r) => r.inputTokens)) / 1e6).toFixed(1)}M / ${(sum(reqs.map((r) => r.outputTokens)) / 1e6).toFixed(2)}M | ` +
    `cost $${totalCost.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} | server errors ${nErr} | ` +
    `refusals ${reqs.filter((r) => r.status === "refusal").length}`,
);
console.log("Incidents: INC1 Sonnet latency 10:00-11:00 | INC2 initech agent loop from 15:00 |");
console.log("           INC3 umbrella/extract refusals from 16:00 | INC4 summarize quality drop from 20:00");

section("2. Aggregation: averages hide tails; histograms give mergeable percentiles");
console.log(`${"MODEL".padEnd(7)} ${"N".padStart(5)} ${"MEAN".padStart(7)} | ${"P50 exact/hist".padStart(15)} | ${"P95 exact/hist".padStart(15)} | ${"P99 exact/hist".padStart(17)}`);
for (const model of [MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS]) {
  const lat = reqs.filter((r) => r.model === model).map((r) => r.latencyMs);
  const h = histogram(lat);
  const cells = [50, 95, 99].map((p) => `${f0(pctExact(lat, p), 7)}/${pctHist(h, p).toFixed(0).padEnd(7)}`);
  console.log(`${SHORT[model]!.padEnd(7)} ${String(lat.length).padStart(5)} ${f0(sum(lat) / lat.length, 7)} | ${cells[0]!.padStart(15)} | ${cells[1]!.padStart(15)} | ${cells[2]!.padStart(17)}`);
}
console.log(`A histogram is ${BUCKETS.length + 1} counters per series regardless of volume, and histograms from many`);
console.log("hosts/minutes can be summed before computing percentiles - averages and per-host p95s cannot.");

section("3. Cardinality: what may be a metric label");
const base = new Set(reqs.map((r) => `${r.tenant}|${r.feature}|${r.model}|${r.status}`));
const withUser = new Set(reqs.map((r) => `${r.tenant}|${r.feature}|${r.model}|${r.status}|${r.user}`));
const nStatuses = new Set(reqs.map((r) => r.status)).size;
console.log(`labels {tenant, feature, model, status}        -> ${fmtInt(base.size).padStart(6)} series observed (max ${fmtInt(8 * 4 * 3 * nStatuses)})`);
console.log(`... + user_id                                -> ${fmtInt(withUser.size).padStart(6)} series observed today`);
console.log(`... + user_id at 200k monthly users          -> up to ${fmtInt(8 * 4 * 3 * nStatuses * 200_000)} series`);
console.log("Rule: metric labels must be bounded (tenant tier, feature, model, status). Put user_id, trace_id and");
console.log("request_id in logs/traces, and link metrics to traces with EXEMPLARS:");
const worst = new Map<number, Req>();
for (const r of reqs.filter((x) => x.model === MODEL_SONNET && x.minute >= 600 && x.minute < 660)) {
  const i = histogram([r.latencyMs]).indexOf(1);
  if (!worst.has(i) || r.latencyMs > worst.get(i)!.latencyMs) worst.set(i, r);
}
const top = Math.max(...worst.keys());
const ex = worst.get(top)!;
console.log(
  `  sonnet 10:00-11:00 slowest bucket (${fmtInt(BUCKETS[top - 1]!)}-${fmtInt(BUCKETS[top]!)} ms] exemplar: trace_id=${ex.traceId}` +
    ` (${fmtInt(ex.latencyMs)} ms, tenant=${ex.tenant}, user=${ex.user})`,
);

section("4. Trace sampling: head-based vs tail-based");
const cats: [string, (r: Req) => boolean][] = [
  ["server errors", (r) => ERRORS.has(r.status)],
  ["refusals", (r) => r.status === "refusal"],
  ["SLO-slow", isSlow],
  ["agent-loop (INC2)", (r) => r.loop],
  ["cost > $1", (r) => r.cost > 1.0],
];
const head = new Set(reqs.filter((r) => hash01(r.traceId, "head") < 0.05).map((r) => r.idx));
// Decided AFTER the trace completes (needs a buffering collector): keep every
// error/refusal/slow/expensive/long trace + a 2% baseline of normal traffic.
const tailKeep = (r: Req): boolean =>
  ERRORS.has(r.status) || r.status === "refusal" || isSlow(r) || r.cost > 1.0 || r.steps > 15 || hash01(r.traceId, "tail") < 0.02;
const tail = new Set(reqs.filter(tailKeep).map((r) => r.idx));
console.log(`${"POLICY".padEnd(24)} ${"KEPT".padStart(6)} ${"VOLUME".padStart(7)} | ` + cats.map(([c]) => c.padStart(17)).join(" | "));
for (const [name, kept] of [["head 5% (at trace start)", head], ["tail (after completion)", tail]] as const) {
  const cov = cats.map(([, fn]) => {
    const matching = reqs.filter(fn);
    const got = matching.filter((r) => kept.has(r.idx)).length;
    return matching.length ? `${String(got).padStart(4)}/${String(matching.length).padEnd(4)} ${f0((got / matching.length) * 100, 5)}%` : "n/a";
  });
  console.log(`${name.padEnd(24)} ${String(kept.size).padStart(6)} ${((kept.size / reqs.length) * 100).toFixed(1).padStart(6)}% | ` + cov.map((c) => c.padStart(17)).join(" | "));
}
console.log("Head sampling is cheap and stateless but keeps ~5% of the traces you need most. Tail sampling keeps");
console.log("all of them at similar volume, at the price of buffering whole traces before deciding.");

section("5. Distributed trace correlation across agent steps");
const loopReq = reqs.find((r) => r.loop)!;
const spans = buildAgentTrace(loopReq);
const byId = new Map(spans.map((s) => [s.spanId, s] as const));
const depth = (s: Span): number => (s.parentId === null ? 0 : 1 + depth(byId.get(s.parentId)!));
console.log(`trace_id=${loopReq.traceId} tenant=${loopReq.tenant} started ${hhmm(loopReq.minute)}, ${spans.length} spans:`);
for (const s of spans.slice(0, 7)) {
  const extra = "anthropic.request_id" in s.attrs ? ` request_id=${s.attrs["anthropic.request_id"]}` : "";
  console.log(`  ${"  ".repeat(depth(s))}- ${s.name.padEnd(34)} span=${s.spanId} +${fmtInt(s.startMs).padStart(9)}ms ${fmtInt(s.durMs).padStart(9)}ms${extra}`);
}
console.log(`  ${"  ".repeat(2)}... ${spans.length - 7} more spans`);
console.log(`Propagate to tools/subagents as a W3C header: traceparent: 00-${loopReq.traceId}-${spans[2]!.spanId}-01`);
const toolCounts = new Map<string, number>();
for (const s of spans) if (s.name.startsWith("tool.")) toolCounts.set(s.name, (toolCounts.get(s.name) ?? 0) + 1);
const [loopTool, loopN] = [...toolCounts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? 1 : a[0] > b[0] ? -1 : 0))[0]!;
console.log(`Loop detector: ${loopTool} called ${loopN}x with identical args in one trace (steps=${loopReq.steps}) -> LOOP SUSPECTED`);

section("6. Cost attribution per tenant (from usage tokens)");
const hourly = new Map<string, number[]>(TENANTS.map(([t]) => [t, new Array<number>(24).fill(0)]));
for (const r of reqs) hourly.get(r.tenant)![Math.floor(r.minute / 60)]! += r.cost;
console.log(
  `${"TENANT".padEnd(9)} ${"REQS".padStart(5)} ${"COST".padStart(9)} ${"SHARE".padStart(6)} ${"$/HR 00-15".padStart(10)} ${"$/HR 15-24".padStart(10)} ${"AGENT P99 STEPS".padStart(16)}  ANOMALY`,
);
const anomalyFirst = new Map<string, number>();
for (const [t] of TENANTS) {
  const rs = reqs.filter((r) => r.tenant === t);
  const cost = sum(rs.map((r) => r.cost));
  const hrs = hourly.get(t)!;
  const before = sum(hrs.slice(0, 15)) / 15;
  const after = sum(hrs.slice(15)) / 9;
  const steps = rs.filter((r) => r.feature === "agent").map((r) => r.steps);
  let flag = "";
  for (let h = 6; h < 24; h++) {
    const window = hrs.slice(h - 6, h).sort((a, b) => a - b);
    const baseline = (window[2]! + window[3]!) / 2; // trailing 6h median
    if (hrs[h]! > 3 * baseline && hrs[h]! > 2.0) {
      anomalyFirst.set(t, h);
      flag = `cost ${(hrs[h]! / baseline).toFixed(1)}x 6h median at ${String(h).padStart(2, "0")}:00`;
      break;
    }
  }
  console.log(
    `${t.padEnd(9)} ${String(rs.length).padStart(5)} ${cost.toFixed(2).padStart(9)} ${((cost / totalCost) * 100).toFixed(1).padStart(5)}% ` +
      `${before.toFixed(2).padStart(10)} ${after.toFixed(2).padStart(10)} ${f0(steps.length ? pctExact(steps, 99) : 0, 16)}  ${flag}`,
  );
}
console.log("Guardrails: per-tenant budgets + anomaly alerts, a max_steps cap per agent run, per-tenant rate limits,");
console.log("and (for hard isolation) separate workspaces/keys whose spend limits and usage reports attribute cost natively.");

section("7. Alerting: static threshold vs SLO burn rate (SLO 99.5% good events)");
const tot = new Array<number>(MINUTES).fill(0);
const bad = new Array<number>(MINUTES).fill(0);
for (const r of reqs) {
  tot[r.minute]!++;
  if (isBad(r)) bad[r.minute]!++;
}
const ptot = new Array<number>(MINUTES + 1).fill(0);
const pbad = new Array<number>(MINUTES + 1).fill(0);
for (let m = 0; m < MINUTES; m++) {
  ptot[m + 1] = ptot[m]! + tot[m]!;
  pbad[m + 1] = pbad[m]! + bad[m]!;
}
const rate = (end: number, window: number): number => {
  const lo = Math.max(0, end + 1 - window);
  const n = ptot[end + 1]! - ptot[lo]!;
  return n ? (pbad[end + 1]! - pbad[lo]!) / n : 0;
};
const evals: number[] = [];
for (let m = 60; m < MINUTES; m += 5) evals.push(m);
const staticRule: [number, boolean][] = evals.map((m) => [m, rate(m, 5) > 0.02]);
const pageRule: [number, boolean][] = evals.map((m) => [m, rate(m, 60) / BUDGET > 14.4 && rate(m, 5) / BUDGET > 14.4]);
const ticketRule: [number, boolean][] = evals.filter((m) => m >= 360).map((m) => [m, rate(m, 360) / BUDGET > 6 && rate(m, 30) / BUDGET > 6]); // full 6h window
const dayBad = sum(bad) / sum(tot);
console.log(`Day: ${sum(bad)} bad events / ${fmtInt(sum(tot))} = ${(dayBad * 100).toFixed(2)}% (budget 0.50%, burn ${(dayBad / BUDGET).toFixed(1)}x)`);
for (const [label, rule] of [
  ["static: bad rate > 2% over 5m", staticRule],
  ["PAGE: burn > 14.4x over 1h AND 5m", pageRule],
  ["TICKET: burn > 6x over 6h AND 30m", ticketRule],
] as const) {
  const eps = episodes(rule);
  const shown = eps.slice(0, 4).map(([a, b]) => `${hhmm(a)}-${hhmm(b)}`).join(", ") + (eps.length > 4 ? " ..." : "");
  console.log(`  ${label.padEnd(36)} ${String(rule.filter(([, on]) => on).length).padStart(3)} firing evals, ${String(eps.length).padStart(3)} episodes: ${shown || "-"}`);
}
console.log("The static rule fires all day on random blips (alert fatigue) - the real incident is buried.");
console.log("Burn-rate alerts fire on budget consumption speed: one page for INC1, nothing for noise.");
console.log("Note: INC2-INC4 never breach the availability SLO - cost, refusal and quality need their own monitors.");

// Refusal-rate monitor: per-segment ratio alert with a minimum volume and a confidence bound.
const segKey = (t: string, f: string) => `${t}|${f}`;
const segN = new Map<string, number[]>();
const segK = new Map<string, number[]>();
for (const [t] of TENANTS) for (const [f] of FEATURES) {
  segN.set(segKey(t, f), new Array<number>(24).fill(0));
  segK.set(segKey(t, f), new Array<number>(24).fill(0));
}
for (const r of reqs) {
  const h = Math.floor(r.minute / 60);
  segN.get(segKey(r.tenant, r.feature))![h]!++;
  if (r.status === "refusal") segK.get(segKey(r.tenant, r.feature))![h]!++;
}
let refusalAlert: [string, string, number, number, number] | null = null;
for (let endH = 3; endH <= 24; endH++) {
  // evaluate hourly over a trailing 3h window
  for (const [key, ns] of segN) {
    const n = sum(ns.slice(endH - 3, endH));
    const k = sum(segK.get(key)!.slice(endH - 3, endH));
    if (refusalAlert === null && n >= 20 && wilson(k, n)[0] > 0.05) {
      const [t, f] = key.split("|") as [string, string];
      refusalAlert = [t, f, endH * 60, k, n];
    }
  }
}
const beforeReqs = reqs.filter((r) => r.minute < 960);
const afterReqs = reqs.filter((r) => r.minute >= 960);
const rateBefore = beforeReqs.filter((r) => r.status === "refusal").length / beforeReqs.length;
const rateAfter = afterReqs.filter((r) => r.status === "refusal").length / afterReqs.length;
console.log(
  `Refusal rate across ALL traffic: ${(rateBefore * 100).toFixed(1)}% before 16:00 vs ${(rateAfter * 100).toFixed(1)}% after` +
    " - easy to miss on a global dashboard.",
);
if (refusalAlert) {
  const [t, f, end, k, n] = refusalAlert;
  console.log(
    `Segment monitor (tenant x feature, 3h window, n>=20, Wilson lower bound > 5%): ${t}/${f} at ${hhmm(end)}` +
      ` - ${k}/${n} refused (${((k / n) * 100).toFixed(0)}%)`,
  );
}

section("8. Quality: sampled LLM-as-judge (real calls via the shared client)");
const sample: Req[] = [];
for (let hour = 0; hour < 24; hour++) {
  const inHour = reqs
    .filter((r) => r.feature === "summarize" && Math.floor(r.minute / 60) === hour)
    .sort((a, b) => hash01(a.traceId, "judge") - hash01(b.traceId, "judge"));
  sample.push(...inHour.slice(0, 6).sort((a, b) => a.idx - b.idx)); // stratified: 6 per hour
}
const client = getClient(judgeResponder);
const limit = limiter(8);
type Verdict = [Req, boolean, string, number];
async function judge(r: Req): Promise<Verdict> {
  const [source, summary] = makeDoc(r);
  const resp = await limit(() =>
    client.messages.create({
      model: MODEL_SONNET,
      max_tokens: 300,
      system: JUDGE_SYSTEM,
      output_config: { effort: "low", format: { type: "json_schema", schema: JUDGE_SCHEMA } },
      messages: [{ role: "user", content: `<source>${source}</source>\n<summary>${summary}</summary>` }],
    }),
  );
  if (resp.stop_reason !== "end_turn") return [r, false, `judge stopped: ${resp.stop_reason}`, 0];
  const verdict = JSON.parse(textOf(resp)) as { faithful: boolean; score: number; issue: string };
  return [r, Boolean(verdict.faithful), verdict.issue, costUsd(MODEL_SONNET, resp.usage)];
}
const verdicts = await Promise.all(sample.map(judge));
const judgeCost = sum(verdicts.map((v) => v[3]));
console.log(
  `Judged ${verdicts.length} of ${fmtInt(reqs.filter((r) => r.feature === "summarize").length)} summarize responses ` +
    `(6/hour, stratified) for $${judgeCost.toFixed(4)}. Example verdicts:`,
);
for (const [r, ok, issue] of [...verdicts.filter((v) => v[0].minute >= 1200).slice(0, 2), verdicts[0]!]) {
  console.log(`  ${hhmm(r.minute)} trace=${r.traceId.slice(0, 12)}.. faithful=${String(ok).padEnd(5)} issue=${issue}`);
}
console.log(`${"WINDOW".padEnd(12)} ${"N".padStart(3)} ${"PASS".padStart(5)} ${"RATE".padStart(6)}  ${"95% WILSON CI".padEnd(15)} STATUS`);
let judgeAlert: number | null = null;
for (let w = 0; w < 24; w += 2) {
  const vs = verdicts.filter((v) => Math.floor(v[0].minute / 60) >= w && Math.floor(v[0].minute / 60) < w + 2);
  const k = vs.filter((v) => v[1]).length;
  const [lo, hi] = wilson(k, vs.length);
  const alert = hi < 0.85; // even the optimistic bound is below the 85% quality target
  if (alert && judgeAlert === null) judgeAlert = w;
  const pad2 = (x: number) => String(x).padStart(2, "0");
  console.log(
    `${pad2(w)}:00-${pad2(w + 2)}:00  ${String(vs.length).padStart(3)} ${String(k).padStart(5)} ${f0(vs.length ? (k / vs.length) * 100 : 0, 5)}%  ` +
      `[${f0(lo * 100, 3)}%, ${f0(hi * 100, 3)}%]      ${alert ? "ALERT" : "ok"}`,
  );
}
const batch: Anthropic.Messages.BatchCreateParams.Request[] = sample.map((r) => {
  const [source, summary] = makeDoc(r);
  return {
    custom_id: `judge-${r.idx}`,
    params: {
      model: MODEL_SONNET,
      max_tokens: 300,
      system: JUDGE_SYSTEM,
      output_config: { effort: "low", format: { type: "json_schema", schema: JUDGE_SCHEMA } },
      messages: [{ role: "user", content: `<source>${source}</source>\n<summary>${summary}</summary>` }],
    },
  };
});
if (isLive()) {
  const mb = await client.messages.batches.create({ requests: batch });
  console.log(`Submitted nightly judge batch ${mb.id} (${batch.length} requests, 50% price).`);
} else {
  console.log(`[live-only] client.messages.batches.create(requests=<${batch.length} judge requests>) - the Batches API`);
  console.log("            halves judge cost; results arrive asynchronously (within 24h), fine for trend monitoring.");
}

section("Incident detection matrix (time to detect)");
const staticEps = episodes(staticRule);
const burnEps = [...episodes(pageRule), ...episodes(ticketRule)].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
const staticCell = (start: number, end: number): string => {
  const k = staticEps.filter(([a, b]) => a <= end && b >= start).length;
  return k ? `${k} of ${staticEps.length} episodes overlap` : "silent";
};
const burnCell = (start: number, end: number): string => {
  const first = burnEps.find(([a]) => a >= start && a <= end);
  return first ? `${first[0] - start} min` : "missed";
};
const initechAnomaly = anomalyFirst.get("initech");
const rows: [string, number, number, string, string][] = [
  ["INC1 Sonnet latency spike", 600, 660, "-", "-"],
  ["INC2 initech agent loop", 900, MINUTES, initechAnomaly !== undefined ? `${initechAnomaly * 60 + 60 - 900} min (cost anomaly)` : "missed", "-"],
  ["INC3 umbrella refusals", 960, MINUTES, refusalAlert ? `${refusalAlert[2] - 960} min (refusal monitor)` : "missed", "-"],
  ["INC4 summarize quality", 1200, MINUTES, "missed", judgeAlert !== null ? `<= ${(judgeAlert + 2) * 60 - 1200} min (judge window)` : "missed"],
];
console.log(`${"INCIDENT".padEnd(27)} ${"STATIC THRESHOLD".padEnd(27)} ${"BURN-RATE".padEnd(10)} ${"COST/REFUSAL MONITOR".padEnd(30)} LLM JUDGE`);
for (const [name, start, end, c, j] of rows) {
  console.log(`${name.padEnd(27)} ${staticCell(start, end).padEnd(27)} ${burnCell(start, end).padEnd(10)} ${c.padEnd(30)} ${j}`);
}
console.log("A static threshold that fires on noise all day is not a detector even when it 'fires' during an incident.");

section("Monitoring strategy recommendations");
const table: [string, string, string][] = [
  ["Tail latency at volume", "Histograms per model/feature; p95/p99 SLIs", "usage + latency per call; TTFT when streaming"],
  ["Alert fatigue", "Multi-window SLO burn-rate alerts", "page on fast burn, ticket on slow burn"],
  ["High-cardinality dims", "Bounded labels; user/trace ids in logs + exemplars", "never user_id as a metric label"],
  ["Trace storage cost", "Tail-based sampling at the collector", "keep 100% errors/slow/costly/refusals + 2%"],
  ["Multi-step agents", "W3C traceparent through tools/subagents", "span per LLM call/tool; log request_id"],
  ["Runaway cost", "Per-tenant attribution, budgets, anomalies", "usage tokens x price; max_steps; workspace limits"],
  ["Refusals / truncation", "stop_reason ratio monitors (min volume)", "refusal, max_tokens, pause_turn per tenant"],
  ["Silent quality drift", "Stratified LLM-as-judge, calibrated", "Batches API for cost; compare pre/post deploy"],
  ["Rate limits / overload", "429/529 + retry rates per workspace", "anthropic-ratelimit-* headers, retry-after"],
];
console.log(`${"CHALLENGE".padEnd(24)} ${"STRATEGY".padEnd(51)} SIGNAL / HOOK`);
for (const [c, s, h] of table) console.log(`${c.padEnd(24)} ${s.padEnd(51)} ${h}`);
