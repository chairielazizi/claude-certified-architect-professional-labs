/**
 * D4.5 - Token, Latency and Cost-Performance Optimization
 *
 * Start from a measured baseline workload (12 refund questions answered over a large
 * static policy manual plus a verbose tool result) and apply levers IN ORDER, measuring
 * each one: cost per COMPLETED task (retries and escalations included), p95 latency and
 * quality.
 *
 *   Free wins first:  1 prompt caching  2 trim input tokens  3 right-size output  4 Batches API
 *   Trade-offs last:  5 effort tuning   6 model routing cascade   7 parallelism (latency)
 *
 * Run from the repo root:
 *   node "D4.5 - Token, Latency and Cost-Performance Optimization/typescript/main.ts"
 */

import type Anthropic from "@anthropic-ai/sdk";
import {
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
  textBlock,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const P95_SLO_MS = 6000;
const MAX_ATTEMPTS = 2;

// ---------------------------------------------------------------------------
// Deterministic helpers (same algorithms as the Python version)
// ---------------------------------------------------------------------------
function fnv1a32(s: string): number {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(s)) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const fmt = (x: number, d = 2): string => (Math.floor(x * 10 ** d + 0.5) / 10 ** d).toFixed(d);
const signedPct = (x: number): string => (x >= 0 ? "+" : "") + fmt(100 * x, 1) + "%";
function total(xs: Iterable<number>): number {
  let s = 0;
  for (const x of xs) s += x;
  return s;
}
function percentile(values: number[], p: number): number {
  const s = [...values].sort((x, y) => x - y);
  const rank = (p / 100) * (s.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo]! + (s[hi]! - s[lo]!) * (rank - lo);
}

// ---------------------------------------------------------------------------
// Workload: a big static manual (cacheable) + per-task verbose tool results
// ---------------------------------------------------------------------------
const RULES: [string, string][] = [
  ["R-1", "Unopened items are refunded at the full item price times quantity."],
  ["R-2", "Opened items carry a 10 percent restocking fee: refund 90 percent of price times quantity."],
  ["R-3", "If an order-level promotion applied, first reduce the item price by the promotion percentage, then apply R-1 or R-2."],
];
const FILLER_TOPICS = ["warranty claims", "gift cards", "loyalty points", "price matching", "pre-orders", "backorders", "bulk orders",
  "international duties", "carrier claims", "lost parcels", "address changes", "split shipments", "subscriptions",
  "trade-ins", "repairs", "extended warranty", "student discounts", "tax exemption", "invoices", "chargebacks",
  "fraud review", "account merges", "data requests", "accessibility", "corporate accounts", "reseller orders",
  "events", "rentals", "refurbished items", "bundles", "custom engraving", "personalization", "digital goods",
  "software licenses", "installation", "recycling", "hazardous goods", "oversized items", "white glove", "pickup points"];
const MANUAL =
  "ACME RETURNS AND REFUNDS MANUAL (v12)\n\n" + RULES.map(([sid, txt]) => `Section ${sid}: ${txt}`).join("\n") + "\n\n" +
  FILLER_TOPICS.map(
    (t, i) =>
      `Section G-${String(i + 1).padStart(2, "0")} (${t}): Requests about ${t} are handled by the customer care team. Agents verify the order, record the ` +
      `request in the case system, and follow the ${t} checklist. Escalate unusual ${t} requests to a team lead within one business day.`,
  ).join("\n");
const ITEM_NAMES = ["desk lamp", "office chair", "monitor arm", "keyboard", "webcam", "headset", "dock", "mouse"];
const LEVEL: Record<string, number> = { easy: 1, medium: 2, hard: 3 };
const EVENTS = ["created", "paid", "picked", "packed", "shipped", "in transit", "out for delivery", "delivered", "survey sent",
  "return requested", "label issued", "carrier scan", "received", "inspected"];

interface Item {
  sku: string;
  name: string;
  unit_price: number;
  qty: number;
  opened: boolean;
}
interface OrderRecord {
  order_id: string;
  customer: { name: string; tier: string; since: string };
  items: Item[];
  order_promo_pct: number;
  shipping: { method: string; cost: number; address: string };
  payment: { method: string; last4: string; captured: boolean };
  history: { ts: string; event: string; actor: string; note: string }[];
}
interface Task {
  id: string;
  order_id: string;
  difficulty: "easy" | "medium" | "hard";
  record: OrderRecord;
  return_sku: string;
  expected_amount: number;
  expected_rule: string;
}

function makeTask(i: number): Task {
  const difficulty = i % 6 === 5 ? "hard" : i % 6 === 4 ? "medium" : "easy";
  const items: Item[] = [0, 1, 2].map((k) => ({
    sku: `SKU-${100 + i * 3 + k}`,
    name: ITEM_NAMES[(i + k) % ITEM_NAMES.length]!,
    unit_price: 15 + ((i * 37 + k * 11) % 80) + 0.5,
    qty: 1 + ((i + k) % 3),
    opened: difficulty !== "easy" && k === i % 3,
  }));
  const promo = difficulty === "hard" ? 20 : 0;
  const history = Array.from({ length: 14 }, (_, e) => ({
    ts: `2026-09-${String(1 + e).padStart(2, "0")}T${String(8 + (e % 9)).padStart(2, "0")}:15:00Z`,
    event: EVENTS[e]!,
    actor: e % 2 ? "system" : "warehouse-bot",
    note: `Automated status update number ${e + 1} for order ORD-5${String(i).padStart(4, "0")}.`,
  }));
  const record: OrderRecord = {
    order_id: `ORD-5${String(i).padStart(4, "0")}`,
    customer: { name: `Customer ${i}`, tier: "standard", since: "2021-03-04" },
    items,
    order_promo_pct: promo,
    shipping: { method: "ground", cost: 7.5, address: "redacted" },
    payment: { method: "card", last4: "4242", captured: true },
    history,
  };
  const ret = items[i % 3]!;
  const price = ret.unit_price * (1 - promo / 100);
  const refund = price * ret.qty * (ret.opened ? 0.9 : 1.0);
  const rule = promo ? "R-3" : ret.opened ? "R-2" : "R-1";
  return { id: `T${String(i + 1).padStart(2, "0")}`, order_id: record.order_id, difficulty, record, return_sku: ret.sku, expected_amount: Number(fmt(refund, 2)), expected_rule: rule };
}

const TASKS = Array.from({ length: 12 }, (_, i) => makeTask(i));
const GET_ORDER_TOOL: Anthropic.Tool = {
  name: "get_order",
  description: "Fetch an order record by id.",
  input_schema: { type: "object", properties: { order_id: { type: "string" } }, required: ["order_id"] },
};

/** Project the tool result to the fields the task needs (lever 2). */
function trimmedRecord(task: Task): Record<string, unknown> {
  const item = task.record.items.find((it) => it.sku === task.return_sku)!;
  return {
    order_id: task.order_id,
    return_item: { sku: item.sku, unit_price: item.unit_price, qty: item.qty, opened: item.opened },
    order_promo_pct: task.record.order_promo_pct,
  };
}

// ---------------------------------------------------------------------------
// Offline responder: quality depends on model x effort vs task difficulty
// ---------------------------------------------------------------------------
const CAPABILITY: Record<string, number> = {
  [`${MODEL_HAIKU}|low`]: 1, [`${MODEL_HAIKU}|medium`]: 1, [`${MODEL_SONNET}|low`]: 1, [`${MODEL_SONNET}|medium`]: 2,
  [`${MODEL_SONNET}|high`]: 2, [`${MODEL_OPUS}|low`]: 2, [`${MODEL_OPUS}|medium`]: 3, [`${MODEL_OPUS}|high`]: 3,
};
const THINKING: Record<string, number> = { low: 40, medium: 250, high: 700 }; // adaptive-thinking tokens - billed as output

function responder(req: MockRequest): ResponderResult {
  const m = /ORD-5\d{4}/.exec(req.allText);
  const task = TASKS.find((t) => m !== null && t.order_id === m[0])!;
  const effort = String(req.body.output_config?.effort ?? "high");
  const ok = (CAPABILITY[`${req.model}|${effort}`] ?? 1) >= LEVEL[task.difficulty]!;
  let amount: number;
  let rule: string;
  if (ok) {
    amount = task.expected_amount;
    rule = task.expected_rule;
  } else {
    // typical slip: skip the last rule in the chain
    const item = task.record.items.find((it) => it.sku === task.return_sku)!;
    amount = Number(fmt(item.unit_price * item.qty * (task.difficulty === "hard" ? 0.9 : 1.0), 2));
    rule = task.difficulty === "hard" ? "R-2" : "R-1";
  }
  let text = `Refund: ${fmt(amount, 2)} dollars (policy ${rule}).`;
  if (!req.allText.includes("at most two sentences")) {
    text +=
      " To explain in detail: the manual describes how refunds are computed from the item price, the quantity returned, " +
      "whether the packaging was opened and whether an order-level promotion applied. I checked the order history, the " +
      "payment record and the shipping record, confirmed the item was delivered and inspected, and then applied the " +
      "relevant sections in order. Shipping costs are not refunded for this case, and the refund goes back to the " +
      "original card ending in 4242 within five to seven business days after inspection.";
  }
  const think = THINKING[effort]!;
  return { content: [textBlock(text)], usage: { output_tokens: estimateTokens(text) + think, output_tokens_details: { thinking_tokens: think } } };
}

// ---------------------------------------------------------------------------
// Configurations and measurement
// ---------------------------------------------------------------------------
type Effort = "low" | "medium" | "high";
interface Config {
  name: string;
  lever: string;
  kind: "baseline" | "free" | "trade-off";
  model: string;
  effort: Effort;
  cache: boolean;
  trim: boolean;
  concise: boolean;
  maxTokens: number;
  cascade: [string, Effort][];
}
const cfg = (c: Partial<Config> & Pick<Config, "name" | "lever" | "kind">): Config => ({
  model: MODEL_OPUS, effort: "high", cache: false, trim: false, concise: false, maxTokens: 4096, cascade: [], ...c,
});

interface Attempt {
  model: string;
  cost: number;
  latencyMs: number;
  ok: boolean;
  inputTokens: number;
  cacheRead: number;
  cacheWrite: number;
  outputTokens: number;
  thinking: number;
}
interface TaskRun {
  taskId: string;
  attempts: Attempt[];
}
const completedRun = (r: TaskRun): boolean => r.attempts[r.attempts.length - 1]!.ok;
const runLatency = (r: TaskRun): number => total(r.attempts.map((a) => a.latencyMs)); // retries/escalations add user-visible latency

interface Result {
  cfg: Config;
  runs: TaskRun[];
}
const attemptsOf = (r: Result): Attempt[] => r.runs.flatMap((x) => x.attempts);
const completedOf = (r: Result): number => r.runs.filter(completedRun).length;
const costPerCompleted = (r: Result): number => total(attemptsOf(r).map((a) => a.cost)) / Math.max(1, completedOf(r));
const p95Of = (r: Result): number => percentile(r.runs.map(runLatency), 95);

const LAT: Record<string, [number, number, number]> = { [MODEL_HAIKU]: [250, 3, 0.01], [MODEL_SONNET]: [500, 7, 0.02], [MODEL_OPUS]: [900, 12, 0.04] }; // ttft, ms/out tok, ms/in tok

/** Illustrative latency model: TTFT + prefill (cache reads ~10% of the cost of fresh input) + generation. */
function modeledLatency(key: string, model: string, usage: Anthropic.Usage): number {
  const [ttft, perOut, perIn] = LAT[model]!;
  const rng = mulberry32(fnv1a32(key));
  const prefill = perIn * (usage.input_tokens + (usage.cache_creation_input_tokens ?? 0) + 0.1 * (usage.cache_read_input_tokens ?? 0));
  return ttft * (0.9 + 0.2 * rng()) + prefill + perOut * usage.output_tokens;
}

function buildRequest(c: Config, task: Task, model: string, effort: Effort): Anthropic.MessageCreateParamsNonStreaming {
  const style = c.concise
    ? "Answer in at most two sentences: the refund amount and the policy section id."
    : "Explain your reasoning in detail, including relevant background from the manual.";
  const record = c.trim ? JSON.stringify(trimmedRecord(task)) : JSON.stringify(task.record, null, 2);
  return {
    model,
    max_tokens: c.maxTokens,
    system: c.cache ? [{ type: "text", text: MANUAL, cache_control: { type: "ephemeral" } }] : MANUAL,
    tools: [GET_ORDER_TOOL],
    output_config: { effort },
    messages: [
      { role: "user", content: `Order ${task.order_id}: the customer returns ${task.return_sku}. How much is refunded? ${style}` },
      { role: "assistant", content: [{ type: "tool_use", id: `toolu_${task.id}`, name: "get_order", input: { order_id: task.order_id } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: `toolu_${task.id}`, content: record }] },
    ],
  };
}

function validate(task: Task, text: string): boolean {
  const m = /Refund: (-?\d+(?:\.\d+)?) dollars \(policy (R-\d)\)/.exec(text);
  return m !== null && Math.abs(Number(m[1]) - task.expected_amount) < 0.01 && m[2] === task.expected_rule;
}

async function call(client: Anthropic, c: Config, task: Task, model: string, effort: Effort, key: string): Promise<Attempt> {
  const t0 = performance.now();
  const msg = await client.messages.create(buildRequest(c, task, model, effort));
  const wall = elapsedMs(t0);
  const text = msg.content.map((b) => (b.type === "text" ? b.text : "")).join("");
  const u = msg.usage;
  const latencyMs = isLive() ? wall : modeledLatency(key, msg.model, u); // offline: deterministic model, live: wall-clock
  return {
    model,
    cost: costUsd(msg.model, u),
    latencyMs,
    ok: validate(task, text),
    inputTokens: u.input_tokens,
    cacheRead: u.cache_read_input_tokens ?? 0,
    cacheWrite: u.cache_creation_input_tokens ?? 0,
    outputTokens: u.output_tokens,
    thinking: u.output_tokens_details?.thinking_tokens ?? 0,
  };
}

async function runTask(client: Anthropic, c: Config, task: Task): Promise<TaskRun> {
  const attempts: Attempt[] = [];
  if (c.cascade.length) {
    // escalate to the next tier only when the cheap answer fails validation
    for (const [model, effort] of c.cascade) {
      attempts.push(await call(client, c, task, model, effort, `${c.name}:${task.id}:${model}`));
      if (attempts[attempts.length - 1]!.ok) break;
    }
  } else {
    for (let n = 0; n < MAX_ATTEMPTS; n++) {
      // a failed validation is retried - and billed
      attempts.push(await call(client, c, task, c.model, c.effort, `${c.name}:${task.id}:${n}`));
      if (attempts[attempts.length - 1]!.ok) break;
    }
  }
  return { taskId: task.id, attempts };
}

async function runConfig(c: Config): Promise<Result> {
  const client = getClient(responder); // fresh client = cold cache, so every config pays its own cache writes
  const runs: TaskRun[] = [];
  for (const t of TASKS) runs.push(await runTask(client, c, t)); // sequential, so later requests can read the cache
  return { cfg: c, runs };
}

/** Makespan of running tasks in order on N workers (greedy: next task -> first free worker). */
function schedule(latencies: number[], workers: number): number {
  const free = new Array<number>(workers).fill(0);
  for (const lat of latencies) {
    let i = 0;
    for (let j = 1; j < free.length; j++) if (free[j]! < free[i]!) i = j;
    free[i]! += lat;
  }
  return Math.max(...free);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main(): Promise<void> {
  banner("D4.5 - Token, Latency and Cost-Performance Optimization");
  const diffs = ["easy", "medium", "hard"].map((d) => `${d}=${TASKS.filter((t) => t.difficulty === d).length}`).join(", ");
  console.log(`Workload: ${TASKS.length} refund tasks (${diffs}); static manual ~${estimateTokens(MANUAL)} tokens;`);
  console.log(
    `verbose tool result ~${estimateTokens(JSON.stringify(TASKS[0]!.record, null, 2))} tokens vs trimmed ~` +
      `${estimateTokens(JSON.stringify(trimmedRecord(TASKS[0]!)))} tokens. p95 SLO ${fmt(P95_SLO_MS, 0)} ms.`,
  );
  console.log("Offline latency is modeled (TTFT + prefill + generation); costs come from real usage objects x PRICING.");

  const base = cfg({ name: "baseline", lever: "0. baseline: Opus/high, raw JSON, verbose", kind: "baseline" });
  const c1 = cfg({ name: "cache", lever: "1. prompt caching on the static prefix", kind: "free", cache: true });
  const c2 = cfg({ name: "trim", lever: "2. trim the tool result to needed fields", kind: "free", cache: true, trim: true });
  const lean = { cache: true, trim: true, concise: true, maxTokens: 1024 };
  const c3 = cfg({ name: "output", lever: "3. concise output + right-sized max_tokens", kind: "free", ...lean });
  const effortMed = cfg({ name: "opus-medium", lever: "5a. effort high -> medium", kind: "trade-off", ...lean, effort: "medium" });
  const effortLow = cfg({ name: "opus-low", lever: "5b. effort -> low", kind: "trade-off", ...lean, effort: "low" });
  const sonnet = cfg({ name: "sonnet-medium", lever: "6a. all traffic on Sonnet 5.5 (medium)", kind: "trade-off", model: MODEL_SONNET, ...lean, effort: "medium" });
  const cascade = cfg({
    name: "cascade", lever: "6b. cascade Haiku -> Sonnet -> Opus", kind: "trade-off", ...lean,
    cascade: [[MODEL_HAIKU, "low"], [MODEL_SONNET, "medium"], [MODEL_OPUS, "medium"]],
  });
  const configs = [base, c1, c2, c3, effortMed, effortLow, sonnet, cascade];
  const results = new Map((await Promise.all(configs.map(runConfig))).map((r) => [r.cfg.name, r]));
  const R = (name: string): Result => results.get(name)!;

  // -- profile ------------------------------------------------------------
  section("0. Baseline token profile (where the money goes)");
  const b = R("baseline");
  const a0 = attemptsOf(b)[0]!;
  const price = PRICING[MODEL_OPUS]!;
  const inCost = (a0.inputTokens * price.input) / 1e6;
  const thinkCost = (a0.thinking * price.output) / 1e6;
  const visCost = ((a0.outputTokens - a0.thinking) * price.output) / 1e6;
  console.log(
    `  per request: input ${a0.inputTokens} tok ($${fmt(inCost, 5)}), visible output ${a0.outputTokens - a0.thinking} tok ($${fmt(visCost, 5)}), ` +
      `thinking ${a0.thinking} tok ($${fmt(thinkCost, 5)})`,
  );
  console.log("  -> the static manual is re-billed on every call (caching), the tool JSON is mostly irrelevant (trim),");
  console.log("     and verbose answers + high-effort thinking dominate output spend (concise output, then effort).");

  // -- free wins ----------------------------------------------------------
  section("1-4. Free wins, applied in order (quality must hold)");
  const baseCost = costPerCompleted(b);
  const row = (label: string, kind: string, r: Result | null, prev: number, cost = 0, p95: string | null = null): number => {
    const c = r ? costPerCompleted(r) : cost;
    const lat = p95 ?? (r ? fmt(p95Of(r), 0) : "-");
    const comp = r ? completedOf(r) : completedOf(R("output"));
    const held = comp >= completedOf(b) ? "yes" : "NO";
    console.log(
      `  ${label.padEnd(46)}${kind.padEnd(10)}${("$" + fmt(c, 5)).padStart(10)}${signedPct(c / prev - 1).padStart(9)}${signedPct(c / baseCost - 1).padStart(9)}` +
        `${lat.padStart(9)}${`${comp}/${TASKS.length}`.padStart(7)}${held.padStart(6)}`,
    );
    return c;
  };
  console.log(`  ${"step".padEnd(46)}${"type".padEnd(10)}${"$/task".padStart(10)}${"d_step".padStart(9)}${"d_base".padStart(9)}${"p95 ms".padStart(9)}${"qual".padStart(7)}${"held".padStart(6)}`);
  let prev = row(base.lever, "baseline", b, baseCost);
  for (const c of [c1, c2, c3]) prev = row(c.lever, "free", R(c.name), prev);
  const afterFree = prev;
  row("4. Batches API for non-urgent work (50% off)", "free", null, afterFree, afterFree * 0.5, "async");
  const r1 = attemptsOf(R("cache"));
  console.log(
    `\n  caching detail: request 1 wrote ${r1[0]!.cacheWrite} tokens (1.25x), requests 2-12 read ${r1[1]!.cacheRead} tokens each ` +
      `(Opus 5.5 cache read $${fmt(price.cacheRead, 2)}/MTok vs $${fmt(price.input, 2)} input)`,
  );
  console.log("  The first cached request costs MORE than uncached; measure from the second request on (warm cache).");

  section("4. Batches API request (live-only in this lab)");
  console.log("  client.messages.batches.create(requests=[");
  console.log('      {"custom_id": task_id, "params": {model, max_tokens, system (with cache_control), tools, messages}}, ...])');
  console.log("  -> poll client.messages.batches.retrieve(id) until processing_status == 'ended', then iterate");
  console.log("     client.messages.batches.results(id); results arrive in ANY order - join on custom_id.");
  if (isLive()) {
    const batch = await getClient(responder).messages.batches.create({
      requests: TASKS.slice(0, 2).map((t) => ({ custom_id: t.id, params: buildRequest(c3, t, MODEL_OPUS, "high") })),
    });
    console.log(`  LIVE: submitted batch ${batch.id} status=${batch.processing_status} (not polling in this demo)`);
  } else {
    console.log("  OFFLINE: the mock has no /v1/messages/batches endpoint - cost shown above is projected (50% of step 3).");
  }
  console.log("  Use for evals, backfills and nightly jobs - never for a user waiting on a response (up to 24 h).");

  // -- trade-offs ---------------------------------------------------------
  section("5-6. Trade-offs: effort and model routing (measure, don't assume)");
  console.log(`  ${"configuration".padEnd(40)}${"$/task".padStart(10)}${"p95 ms".padStart(9)}${"qual".padStart(7)}${"calls".padStart(7)}${"cache-write $".padStart(15)}  verdict`);
  const candidates: [string, Result][] = [
    ["opus-high (step 3)", R("output")], ["opus-medium", R("opus-medium")], ["opus-low", R("opus-low")],
    ["sonnet-medium", R("sonnet-medium")], ["cascade H->S->O", R("cascade")],
  ];
  const eligible: [number, string][] = [];
  for (const [label, r] of candidates) {
    const cw = total(attemptsOf(r).map((a) => (a.cacheWrite * PRICING[a.model]!.cacheWrite) / 1e6));
    const qualityOk = completedOf(r) === TASKS.length;
    const latencyOk = p95Of(r) <= P95_SLO_MS;
    const verdict = qualityOk && latencyOk ? "OK" : !qualityOk ? "quality NOT held" : "p95 over SLO";
    if (qualityOk && latencyOk) eligible.push([costPerCompleted(r), label]);
    console.log(
      `  ${label.padEnd(40)}${("$" + fmt(costPerCompleted(r), 5)).padStart(10)}${fmt(p95Of(r), 0).padStart(9)}${`${completedOf(r)}/${TASKS.length}`.padStart(7)}` +
        `${String(attemptsOf(r).length).padStart(7)}${("$" + fmt(cw, 5)).padStart(15)}  ${verdict}`,
    );
  }
  eligible.sort((x, y) => x[0] - y[0] || (x[1] < y[1] ? -1 : x[1] > y[1] ? 1 : 0));
  const [chosenCost, chosen] = eligible[0]!;
  const casc = R("cascade");
  const tiers = [MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS].map((m) => `${m}=${attemptsOf(casc).filter((a) => a.model === m).length}`);
  console.log(`\n  cascade calls by tier: ${tiers.join(", ")} - each tier wrote its OWN cache entry (caches are model-scoped)`);
  console.log("  Opus at medium effort holds quality; low effort and Sonnet fail the hard tasks (and their retries are billed).");
  console.log("  The cascade is cheaper per task but stacks latency on escalated tasks - measure the most capable model at");
  console.log("  lower effort BEFORE building a cascade; build one only if the traffic mix and latency budget justify it.");
  console.log(`  CHOSEN: ${chosen} at $${fmt(chosenCost, 5)}/task (${signedPct(chosenCost / baseCost - 1)} vs baseline), quality held, p95 within SLO`);

  // -- parallelism --------------------------------------------------------
  section("7. Parallelism for wall-clock latency");
  const chosenCfg = chosen === "opus-medium" ? effortMed : chosen.startsWith("opus-high") ? c3 : cascade;
  const client = getClient(responder);
  const warm = await runTask(client, chosenCfg, TASKS[0]!); // warm the cache with ONE request first...
  const queue = TASKS.slice(1);
  const rest: TaskRun[] = new Array(queue.length);
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < queue.length) {
      const i = next++;
      rest[i] = await runTask(client, chosenCfg, queue[i]!);
    }
  };
  await Promise.all([worker(), worker(), worker(), worker()]); // ...then fan out the rest, 4 at a time
  const lats = [runLatency(warm), ...rest.map(runLatency)];
  const sequential = total(lats);
  const parallel = runLatency(warm) + schedule(lats.slice(1), 4);
  console.log(
    `  12 independent tasks on ${chosen}: sequential ${fmt(sequential / 1000, 1)} s vs warm-up + 4-way parallel ${fmt(parallel / 1000, 1)} s ` +
      `(${fmt(sequential / parallel, 1)}x faster wall-clock)`,
  );
  console.log(`  per-request p95 is unchanged (${fmt(percentile(lats, 95), 0)} ms) and so is cost - parallelism buys throughput, not cheaper calls.`);
  console.log(`  fan-out cache reads after warm-up: ${rest.filter((r) => r.attempts[0]!.cacheRead > 0).length}/11 (a cache entry is readable only once the`);
  console.log("  first response has started, so N simultaneous cold requests would all pay full price).");

  // -- summary ------------------------------------------------------------
  section("Waterfall summary: cost per completed task");
  const steps: [string, number][] = [
    ["baseline", baseCost], ["+ caching", costPerCompleted(R("cache"))], ["+ trim input", costPerCompleted(R("trim"))],
    ["+ concise output", costPerCompleted(R("output"))], [`+ ${chosen}`, chosenCost],
  ];
  const width = 40;
  for (const [label, c] of steps) console.log(`  ${label.padEnd(20)}${("$" + fmt(c, 5)).padStart(10)}  ${"#".repeat(Math.max(1, Math.floor((width * c) / baseCost + 0.5)))}`);
  console.log(`  (batch re-runs of the final config would cost ~$${fmt(chosenCost * 0.5, 5)}/task)`);
  console.log("  Order matters: free wins first (they never cost quality), then trade-offs validated against the eval.");
}

await main();
