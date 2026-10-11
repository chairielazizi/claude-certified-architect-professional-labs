/**
 * D3.3 - Accuracy vs Latency Trade-offs
 * Exam objective: Evaluate accuracy-latency trade-offs and justify configuration decisions.
 *
 * Same steps as python/main.py:
 *  1. Sweep model (Haiku/Sonnet/Opus) x effort (low/medium/high) x retrieval top-k (3/8)
 *     over a labeled eval set; every request is streamed, giving accuracy (code grader),
 *     total latency, TTFT and cost. Streaming on/off changes PERCEIVED latency only.
 *  2. SLA-driven selection (cheapest passing config), Pareto frontier, decision record.
 *  3. Latency levers that do not cost accuracy: streaming, parallel tool calls,
 *     prompt caching, prompt trimming.
 *
 * Latency source: LIVE -> measured (elapsedMs, first text delta). OFFLINE -> a transparent
 * latency model fed by each response's real `usage`, so decisions are deterministic.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type Anthropic from "@anthropic-ai/sdk";
import {
  LATENCY_SCALE,
  MODEL_HAIKU,
  MODEL_OPUS,
  MODEL_SONNET,
  PRICING,
  banner,
  costUsd,
  elapsedMs,
  estimateTokens,
  getClient,
  isLive,
  section,
  sleep,
  textBlock,
  textOf,
  toolUseBlock,
  type Json,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
const CORPUS = JSON.parse(fs.readFileSync(path.join(DATA, "corpus.json"), "utf-8")) as Record<string, string>;
interface EvalItem {
  id: string;
  category: string;
  required_skill: number;
  needs: string[];
  question: string;
  gold: string;
  wrong: string;
  distractor_wrong?: string;
  rationale: string;
}
const EVAL = JSON.parse(fs.readFileSync(path.join(DATA, "eval_set.json"), "utf-8")) as EvalItem[];

type Effort = "low" | "medium" | "high";
const MODELS = [MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS];
const EFFORTS: Effort[] = ["low", "medium", "high"];
const TOP_KS = [3, 8];
const SHORT: Record<string, string> = { [MODEL_HAIKU]: "haiku", [MODEL_SONNET]: "sonnet", [MODEL_OPUS]: "opus" };
const SYSTEM =
  "You answer employee travel-policy questions using ONLY the policy excerpts provided. " +
  "Reply in exactly two lines: 'ANSWER: <short answer>' then 'REASON: <one sentence citing excerpt ids>'.";

/** [base TTFT ms, prefill ms per uncached input token, decode ms per output token] */
const LATENCY_MODEL: Record<string, [number, number, number]> = {
  [MODEL_HAIKU]: [250, 0.04, 3],
  [MODEL_SONNET]: [500, 0.08, 7],
  [MODEL_OPUS]: [900, 0.12, 12],
};
const CACHED_PREFILL_FACTOR = 0.1; // cache reads skip most prefill work
const cmp = (a: string | number, b: string | number): number => (a < b ? -1 : a > b ? 1 : 0);

// ---------------------------------------------------------------------------
// Retrieval: a small BM25 index (top-k is an accuracy AND a cost/latency knob)
// ---------------------------------------------------------------------------
const STOP = new Set(
  ("a an the and or of to for in on at is are be by with my i do does what which who how much many must " +
    "can under per each all any from than after was had have has it this that as").split(" "),
);

function tokenize(text: string): string[] {
  const out: string[] = [];
  for (let w of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (STOP.has(w)) continue;
    if (w.length > 3 && w.endsWith("s")) w = w.slice(0, -1);
    out.push(w);
  }
  return out;
}

class BM25 {
  private readonly docs: Map<string, string[]>;
  private readonly df = new Map<string, number>();
  private readonly avgdl: number;
  private readonly k1: number;
  private readonly b: number;
  constructor(docs: Record<string, string>, k1 = 1.2, b = 0.75) {
    this.k1 = k1;
    this.b = b;
    this.docs = new Map(Object.entries(docs).map(([d, t]) => [d, tokenize(t)] as const));
    let total = 0;
    for (const toks of this.docs.values()) {
      total += toks.length;
      for (const w of new Set(toks)) this.df.set(w, (this.df.get(w) ?? 0) + 1);
    }
    this.avgdl = total / this.docs.size;
  }

  search(query: string, k: number): string[] {
    const n = this.docs.size;
    const q = [...new Set(tokenize(query))].sort();
    const scored: { s: number; id: string }[] = [];
    for (const [id, toks] of this.docs) {
      let s = 0;
      for (const w of q) {
        const f = toks.filter((t) => t === w).length;
        if (f) {
          const df = this.df.get(w)!;
          const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
          s += (idf * f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * toks.length) / this.avgdl));
        }
      }
      scored.push({ s, id });
    }
    scored.sort((x, y) => y.s - x.s || cmp(x.id, y.id));
    return scored.slice(0, k).map((x) => x.id);
  }
}

function buildPrompt(contextIds: string[], question: string): string {
  return `Policy excerpts:\n${contextIds.map((d) => `[${d}] ${CORPUS[d]}`).join("\n")}\n\nQuestion: ${question}`;
}

// ---------------------------------------------------------------------------
// Latency model, grader, percentiles
// ---------------------------------------------------------------------------
function modelledLatency(model: string, usage: Anthropic.Usage): [number, number] {
  const [base, prefill, decode] = LATENCY_MODEL[model]!;
  const thinking = usage.output_tokens_details?.thinking_tokens ?? 0;
  const uncached = usage.input_tokens + (usage.cache_creation_input_tokens ?? 0);
  const cached = usage.cache_read_input_tokens ?? 0;
  const ttft = base + prefill * uncached + prefill * CACHED_PREFILL_FACTOR * cached + decode * thinking;
  return [ttft, ttft + decode * (usage.output_tokens - thinking)];
}

const normalize = (text: string): string =>
  text.toLowerCase().replace(/,/g, "").replace(/\s+/g, " ").replace(/^[ .]+|[ .]+$/g, "");

function grade(answerText: string, gold: string): boolean {
  const m = answerText.match(/ANSWER:\s*(.+)/);
  return Boolean(m) && normalize(m![1]!) === normalize(gold);
}

/** Nearest-rank percentile (with n=10, p95 is the slowest request). */
function percentile(values: number[], p: number): number {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.max(0, Math.ceil((p / 100) * s.length) - 1)]!;
}

// ---------------------------------------------------------------------------
// Offline responder: capability grows with model tier and effort
// ---------------------------------------------------------------------------
const CAPABILITY: Record<string, number> = { [MODEL_HAIKU]: 1, [MODEL_SONNET]: 2, [MODEL_OPUS]: 3 };
const EFFORT_BONUS: Record<string, number> = { low: 0, medium: 0.5, high: 1 };
const THINK_PER_SKILL: Record<string, number> = { low: 10, medium: 40, high: 90 }; // adaptive thinking scales with difficulty

function simulateAnswer(model: string, effort: string, item: EvalItem, context: string[]): [string, string, number] {
  const skill = (CAPABILITY[model] ?? 3) + (EFFORT_BONUS[effort] ?? 0.5);
  const missing = item.needs.filter((d) => !context.includes(d));
  let answer: string;
  let reason: string;
  if (model === MODEL_HAIKU && context.includes("D04") && item.needs.includes("D03")) {
    // Smaller models are likelier to be misled by a superseded document in a big context.
    [answer, reason] = [item.distractor_wrong ?? item.wrong, "D04 lists the applicable rate."];
  } else if (missing.length) {
    [answer, reason] = [item.wrong, "The excerpts do not fully settle this; best reading of the text."];
  } else if (skill < item.required_skill) {
    [answer, reason] = [item.wrong, "Based on a direct reading of the excerpts."];
  } else {
    [answer, reason] = [item.gold, item.rationale];
  }
  const thinking = effort === "low" && item.required_skill <= 1 ? 0 : Math.trunc((THINK_PER_SKILL[effort] ?? 40) * item.required_skill);
  return [answer, reason, thinking];
}

function sweepResponder(req: MockRequest): ResponderResult {
  const user = req.lastUserText;
  const idx = user.indexOf("Question: ");
  const question = (idx >= 0 ? user.slice(idx + "Question: ".length) : user).trim();
  const item = EVAL.find((it) => it.question === question);
  if (!item) return "ANSWER: unknown\nREASON: no matching policy.";
  const context = [...user.matchAll(/^\[(D\d\d)\]/gm)].map((m) => m[1]!);
  const effort = String(req.body.output_config?.effort ?? "medium");
  const [answer, reason, thinking] = simulateAnswer(req.model, effort, item, context);
  const text = `ANSWER: ${answer}\nREASON: ${reason}`;
  const visible = estimateTokens(text);
  return { content: [textBlock(text)], usage: { output_tokens: visible + thinking, output_tokens_details: { thinking_tokens: thinking } } };
}

// ---------------------------------------------------------------------------
// The sweep
// ---------------------------------------------------------------------------
interface Config {
  model: string;
  effort: Effort;
  k: number;
}
const label = (c: Config): string => `${SHORT[c.model]}/${c.effort}/k${c.k}`;
interface CallResult {
  itemId: string;
  correct: boolean;
  ttftMs: number;
  totalMs: number;
  cost: number;
}
interface Cell {
  cfg: Config;
  acc: number;
  p50: number;
  p95: number;
  p95Ttft: number;
  costPer1k: number;
  failures: string[];
}

/** Bounded concurrency (respects rate limits live; keeps the offline run fast). */
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

async function runOne(client: Anthropic, limit: ReturnType<typeof limiter>, bm25: BM25, cfg: Config, item: EvalItem): Promise<CallResult> {
  const context = bm25.search(item.question, cfg.k);
  const [msg, first, wall] = await limit(async () => {
    const t0 = performance.now();
    let firstText: number | null = null;
    const stream = client.messages.stream({
      model: cfg.model,
      max_tokens: 4096,
      thinking: { type: "adaptive" },
      output_config: { effort: cfg.effort },
      system: SYSTEM,
      messages: [{ role: "user", content: buildPrompt(context, item.question) }],
    });
    stream.on("text", () => {
      if (firstText === null) firstText = elapsedMs(t0); // time to first visible token
    });
    const final = await stream.finalMessage();
    return [final, firstText as number | null, elapsedMs(t0)] as const;
  });
  const [ttft, total] = isLive() ? [first ?? wall, wall] : modelledLatency(cfg.model, msg.usage);
  const ok = msg.stop_reason === "end_turn" && grade(textOf(msg), item.gold);
  return { itemId: item.id, correct: ok, ttftMs: ttft, totalMs: total, cost: costUsd(cfg.model, msg.usage) };
}

async function runSweep(client: Anthropic, bm25: BM25): Promise<Cell[]> {
  const limit = limiter(isLive() ? 4 : 16);
  const configs: Config[] = MODELS.flatMap((m) => EFFORTS.flatMap((e) => TOP_KS.map((k) => ({ model: m, effort: e, k }))));
  const results = await Promise.all(configs.flatMap((c) => EVAL.map((it) => runOne(client, limit, bm25, c, it))));
  const n = EVAL.length;
  return configs.map((cfg, i) => {
    const rs = results.slice(i * n, (i + 1) * n);
    return {
      cfg,
      acc: rs.filter((r) => r.correct).length / n,
      p50: percentile(rs.map((r) => r.totalMs), 50),
      p95: percentile(rs.map((r) => r.totalMs), 95),
      p95Ttft: percentile(rs.map((r) => r.ttftMs), 95),
      costPer1k: (rs.reduce((s, r) => s + r.cost, 0) / n) * 1000,
      failures: rs.filter((r) => !r.correct).map((r) => r.itemId),
    };
  });
}

// ---------------------------------------------------------------------------
// SLA selection + Pareto frontier
// ---------------------------------------------------------------------------
interface Scenario {
  name: string;
  gates: string;
  minAcc: number;
  maxMs: number | null;
  metric: "total" | "ttft"; // total = stream off / machine consumer; ttft = stream on / human reader
}
const SCENARIOS: Scenario[] = [
  { name: "S1 Internal API (needs full JSON)", gates: "p95 total <= 2000 ms, acc >= 90%", minAcc: 0.9, maxMs: 2000, metric: "total" },
  { name: "S2 Chat UI, streaming on", gates: "p95 TTFT <= 2800 ms, acc >= 95%", minAcc: 0.95, maxMs: 2800, metric: "ttft" },
  { name: "S2b Same chat UI, streaming off", gates: "p95 total <= 2800 ms, acc >= 95%", minAcc: 0.95, maxMs: 2800, metric: "total" },
  { name: "S3 Nightly bulk re-check", gates: "no latency gate, acc >= 70%", minAcc: 0.7, maxMs: null, metric: "total" },
  { name: "S4 High-volume triage", gates: "p95 total <= 1000 ms, acc >= 50%", minAcc: 0.5, maxMs: 1000, metric: "total" },
];

function pick(cells: Cell[], sc: Scenario): Cell | null {
  const latency = (c: Cell) => (sc.metric === "ttft" ? c.p95Ttft : c.p95);
  const ok = cells.filter((c) => c.acc >= sc.minAcc - 1e-9 && (sc.maxMs === null || latency(c) <= sc.maxMs));
  // Cheapest that meets every gate; ties -> more accurate, then faster.
  ok.sort((a, b) => a.costPer1k - b.costPer1k || b.acc - a.acc || latency(a) - latency(b));
  return ok[0] ?? null;
}

function pareto(cells: Cell[]): Cell[] {
  const dominates = (a: Cell, b: Cell) =>
    a.acc >= b.acc && a.p95 <= b.p95 && a.costPer1k <= b.costPer1k && (a.acc > b.acc || a.p95 < b.p95 || a.costPer1k < b.costPer1k);
  return cells.filter((c) => !cells.some((o) => o !== c && dominates(o, c))).sort((a, b) => a.costPer1k - b.costPer1k || b.acc - a.acc);
}

// ---------------------------------------------------------------------------
// Latency levers that do not cost accuracy
// ---------------------------------------------------------------------------
const CITIES = ["London", "Tokyo", "Chicago"];
const CITY_CAPS: Record<string, string> = { London: "GBP 260", Tokyo: "JPY 32000", Chicago: "USD 280" };
const TOOL_MS = 300; // simulated downstream API latency per tool call
const CAP_TOOL: Anthropic.Tool = {
  name: "get_city_cap",
  description: "Return the current hotel nightly cap for one city. Call once per city needed.",
  strict: true,
  input_schema: { type: "object", properties: { city: { type: "string" } }, required: ["city"], additionalProperties: false },
};

function toolsResponder(req: MockRequest): ResponderResult {
  const requested = req.messages
    .filter((m) => m.role === "assistant" && Array.isArray(m.content))
    .flatMap((m) => (m.content as Json[]).filter((b) => b.type === "tool_use").map((b) => String(b.input.city)));
  const remaining = CITIES.filter((c) => !requested.includes(c));
  if (!remaining.length) return "London GBP 260, Tokyo JPY 32000, Chicago USD 280 per night (2026 policy).";
  const oneAtATime = Boolean(req.body.tool_choice?.disable_parallel_tool_use);
  const batch = oneAtATime ? remaining.slice(0, 1) : remaining;
  return {
    content: batch.map((c) => toolUseBlock("get_city_cap", { city: c })),
    usage: { output_tokens: 25 * batch.length }, // ~25 output tokens per tool call
  };
}

async function getCityCap(city: string): Promise<string> {
  await sleep(TOOL_MS * (isLive() ? 1 : LATENCY_SCALE));
  return CITY_CAPS[city] ?? "unknown";
}

/** Returns [model round trips, model ms, tool ms, total ms]. */
async function toolAgent(client: Anthropic, parallel: boolean): Promise<[number, number, number, number]> {
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: "Compare the hotel caps for London, Tokyo and Chicago." }];
  const choice: Anthropic.ToolChoice = parallel ? { type: "auto" } : { type: "auto", disable_parallel_tool_use: true };
  let trips = 0;
  let modelMs = 0;
  let toolMs = 0;
  const t0 = performance.now();
  for (;;) {
    const tCall = performance.now();
    const resp = await client.messages.create({ model: MODEL_SONNET, max_tokens: 1024, tools: [CAP_TOOL], tool_choice: choice, messages });
    trips++;
    modelMs += isLive() ? elapsedMs(tCall) : modelledLatency(MODEL_SONNET, resp.usage)[1];
    if (resp.stop_reason !== "tool_use") break;
    const calls = resp.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use");
    // Independent tool calls run concurrently; results return in ONE user message.
    const outputs = await Promise.all(calls.map((b) => getCityCap(String((b.input as Json).city))));
    toolMs += TOOL_MS; // concurrent batch costs one tool latency, not calls.length
    messages.push({ role: "assistant", content: resp.content });
    messages.push({ role: "user", content: calls.map((b, i) => ({ type: "tool_result" as const, tool_use_id: b.id, content: outputs[i]! })) });
  }
  const total = isLive() ? elapsedMs(t0) : modelMs + toolMs;
  return [trips, modelMs, toolMs, total];
}

function handbook(): string {
  const examples: string[] = [];
  for (let i = 1; i <= 30; i++) {
    const cents = i * 17 * 67;
    examples.push(`Example ${i}: ${i * 17} miles of personal car use is reimbursed at USD ${Math.floor(cents / 100)}.${String(cents % 100).padStart(2, "0")}.`);
  }
  const policy = Object.entries(CORPUS).map(([d, t]) => `[${d}] ${t}`).join("\n");
  return `Travel policy handbook (static, versioned).\n${policy}\nWorked mileage examples:\n` + examples.join("\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
const f0 = (x: number, w: number) => x.toFixed(0).padStart(w);
function fmtCell(c: Cell): string {
  return (
    `${SHORT[c.cfg.model]!.padEnd(7)}${c.cfg.effort.padEnd(8)}${String(c.cfg.k).padStart(2)}  ${f0(c.acc * 100, 4)}%  ${f0(c.p50, 6)}  ${f0(c.p95, 6)}` +
    `  ${f0(c.p95Ttft, 8)}  ${c.costPer1k.toFixed(3).padStart(7)}  ${c.failures.join(",") || "-"}`
  );
}

banner("D3.3 - Accuracy vs Latency Trade-offs");
const bm25 = new BM25(CORPUS);
const src = isLive() ? "measured wall clock" : "latency model fed by real usage (deterministic)";

section("1. Eval set and retrieval (BM25 top-k)");
console.log(`${EVAL.length} labeled questions, ${Object.keys(CORPUS).length} policy excerpts. Needed docs vs retrieved:`);
for (const it of EVAL) {
  const k3 = bm25.search(it.question, 3);
  const k8 = bm25.search(it.question, 8);
  const miss3 = it.needs.filter((d) => !k3.includes(d));
  const flag = miss3.length ? "k=3 MISSES " + miss3.join(",") : "k=3 ok";
  const distractor = k8.includes("D04") && !k3.includes("D04") ? "  (superseded D04 in k=8)" : "";
  console.log(`  ${it.id} ${it.category.padEnd(12)} needs ${it.needs.join(",").padEnd(12)} ${flag}${distractor}`);
}

section("2. Sweep: model x effort x top-k (180 streamed calls)");
console.log(`Latency source: ${src}`);
const client = getClient(sweepResponder);
const cells = await runSweep(client, bm25); // 180 requests, bounded concurrency
console.log(`${"MODEL".padEnd(7)}${"EFFORT".padEnd(8)}${"K".padStart(2)}  ${"ACC".padStart(5)}  ${"P50ms".padStart(6)}  ${"P95ms".padStart(6)}  ${"P95-TTFT".padStart(8)}  ${"$/1k".padStart(7)}  WRONG`);
for (const c of cells) console.log(fmtCell(c));
console.log("Streaming on/off: same accuracy and cost per row; perceived latency = P95-TTFT (on) or P95ms (off).");

section("3. SLA-driven selection (cheapest config that passes every gate)");
const winners = new Map<string, Cell | null>();
for (const sc of SCENARIOS) {
  const w = pick(cells, sc);
  winners.set(sc.name, w);
  console.log(`${sc.name.padEnd(36)} [${sc.gates}]`);
  if (w) {
    const lat = sc.metric === "ttft" ? w.p95Ttft : w.p95;
    console.log(`${"".padEnd(4)}-> ${label(w.cfg).padEnd(18)} acc ${(w.acc * 100).toFixed(0)}%  p95 ${lat.toFixed(0)} ms  $${w.costPer1k.toFixed(3)}/1k`);
  } else {
    console.log(`${"".padEnd(4)}-> NO configuration qualifies`);
  }
}

section("4. Pareto frontier (accuracy up, p95 down, cost down)");
for (const c of pareto(cells)) {
  console.log(`  ${label(c.cfg).padEnd(18)} acc ${f0(c.acc * 100, 3)}%  p95 ${f0(c.p95, 5)} ms  $${c.costPer1k.toFixed(3)}/1k`);
}
console.log("Everything off the frontier is dominated: another config is at least as good on all three axes.");

section("5. Latency levers that do not cost accuracy");
// (a) streaming
const s1 = winners.get(SCENARIOS[0]!.name);
if (!s1) throw new Error("S1 must have a winner");
const hard = EVAL[EVAL.length - 1]!;
{
  const t0 = performance.now();
  let firstChunk = "";
  let firstMs: number | null = null;
  const stream = client.messages.stream({
    model: s1.cfg.model,
    max_tokens: 4096,
    thinking: { type: "adaptive" },
    output_config: { effort: s1.cfg.effort },
    system: SYSTEM,
    messages: [{ role: "user", content: buildPrompt(bm25.search(hard.question, s1.cfg.k), hard.question) }],
  });
  stream.on("text", (delta) => {
    if (firstMs === null) {
      firstMs = elapsedMs(t0);
      firstChunk = delta;
    }
  });
  const final = await stream.finalMessage();
  const [ttft, total] = isLive() ? [(firstMs as number | null) ?? 0, elapsedMs(t0)] : modelledLatency(s1.cfg.model, final.usage);
  console.log(
    `(a) Streaming [${label(s1.cfg)}, ${hard.id}]: first text '${firstChunk.replace(/\n/g, " ").trim()}' at ${ttft.toFixed(0)} ms;` +
      ` complete at ${total.toFixed(0)} ms -> user sees progress ${(total - ttft).toFixed(0)} ms sooner (more on long answers).`,
  );
}

// (b) parallel tool calls
const tclient = getClient(toolsResponder);
const seq = await toolAgent(tclient, false);
const par = await toolAgent(tclient, true);
console.log("(b) Parallel tool calls (3 independent city lookups, 300 ms each):");
console.log(`    disable_parallel_tool_use=True : ${seq[0]} model round trips, model ${seq[1].toFixed(0)} ms + tools ${seq[2].toFixed(0)} ms = ${seq[3].toFixed(0)} ms`);
console.log(`    parallel (default)             : ${par[0]} model round trips, model ${par[1].toFixed(0)} ms + tools ${par[2].toFixed(0)} ms = ${par[3].toFixed(0)} ms`);

// (c) prompt caching
const cclient = getClient(() => "ANSWER: GBP 260\nREASON: D03 sets the current London cap.");
const systemBlocks: Anthropic.TextBlockParam[] = [{ type: "text", text: handbook(), cache_control: { type: "ephemeral" } }];
console.log("(c) Prompt caching of a static handbook prefix (same model, 2 sequential requests):");
for (const [i, q] of [EVAL[2]!.question, EVAL[3]!.question].entries()) {
  const r = await cclient.messages.create({ model: MODEL_SONNET, max_tokens: 512, system: systemBlocks, messages: [{ role: "user", content: q }] });
  const [t] = modelledLatency(MODEL_SONNET, r.usage);
  console.log(
    `    request ${i + 1}: cache_write=${r.usage.cache_creation_input_tokens} cache_read=${r.usage.cache_read_input_tokens}` +
      ` uncached=${r.usage.input_tokens}  TTFT ~${t.toFixed(0)} ms  cost $${costUsd(MODEL_SONNET, r.usage).toFixed(5)}`,
  );
}

// (d) prompt trimming
const verboseSystem =
  SYSTEM + " You are a friendly, professional, helpful assistant. Always be polite. " +
  "For completeness, here is the entire travel policy:\n" + Object.values(CORPUS).join("\n");
const leanPrompt = buildPrompt(bm25.search(hard.question, 8), hard.question);
const verboseN = (await client.messages.countTokens({ model: MODEL_OPUS, system: verboseSystem, messages: [{ role: "user", content: leanPrompt }] })).input_tokens;
const leanN = (await client.messages.countTokens({ model: MODEL_OPUS, system: SYSTEM, messages: [{ role: "user", content: leanPrompt }] })).input_tokens;
const saved = verboseN - leanN;
console.log(
  `(d) Prompt trimming: whole-policy system prompt ${verboseN} tokens vs retrieved-only ${leanN} tokens` +
    ` -> ${saved} fewer tokens, ~${(saved * LATENCY_MODEL[MODEL_OPUS]![1]).toFixed(0)} ms less prefill on Opus,` +
    ` $${(saved * PRICING[MODEL_OPUS]!.input).toLocaleString("en-US", { maximumFractionDigits: 0 })} less per 1M requests.`,
);

section("6. Decision record");
const s2 = winners.get(SCENARIOS[1]!.name);
if (!s2) throw new Error("S2 must have a winner");
const runnerUp = cells.filter((c) => c.acc >= 0.9 && c !== s1).sort((a, b) => a.p95 - b.p95)[0]!;
console.log("ADR-D33-01  Travel-policy assistant: production configuration");
console.log("Context   : two consumers - an internal API (needs the full answer) and an employee chat UI.");
console.log(`Decision  : API -> ${label(s1.cfg)} (acc ${(s1.acc * 100).toFixed(0)}%, p95 ${s1.p95.toFixed(0)} ms, $${s1.costPer1k.toFixed(3)}/1k).`);
console.log(`            Chat UI -> ${label(s2.cfg)} + streaming (acc ${(s2.acc * 100).toFixed(0)}%, p95 TTFT ${s2.p95Ttft.toFixed(0)} ms).`);
console.log(`Rationale : thinking tokens dominate latency; a larger model at LOW effort beat ${label(runnerUp.cfg)}`);
console.log(`            (acc ${(runnerUp.acc * 100).toFixed(0)}%, p95 ${runnerUp.p95.toFixed(0)} ms) on latency at equal accuracy. k=8 is required for`);
console.log("            multi-hop questions; streaming lets the UI afford medium effort for the remaining exception case.");
console.log(`Accepted  : API config misses ${s1.failures.join(",")} (exception + date math). Mitigation: validator flags`);
console.log("            low-confidence answers and re-runs only those at higher effort, asynchronously.");
console.log("Revisit if: weekly eval accuracy < 90%, p95 drifts > 10%, traffic mix shifts to multi-hop, or a new model ships.");

section("Takeaways");
console.log("* Decide against explicit SLAs: accuracy floor + latency metric (total vs TTFT) + cost - then pick the cheapest pass.");
console.log("* Sweep effort before switching models; for small models, more context (top-k) can hurt via distractors.");
console.log("* Streaming fixes PERCEIVED latency for humans only; machines waiting on full JSON gain nothing.");
console.log("* Parallel tools, caching and trimming cut latency without touching accuracy - apply them first.");
