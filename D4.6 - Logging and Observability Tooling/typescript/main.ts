/**
 * D4.6 - Logging and Observability Tooling
 *
 * Instruments a small tool-using Claude agent with:
 *   - structured JSON logs per LLM call and tool call (request_id, trace/span ids, model,
 *     prompt version, tokens incl. cache, cost, latency, stop_reason, tool names, error type)
 *     with PII redaction BEFORE anything is written
 *   - OpenTelemetry-style spans: agent.session -> agent.step -> llm.call / tool.call (JSON export)
 *   - metrics: counters, histograms (p50/p95 from buckets vs exact), cost/hour gauge, text dashboard
 *   - alert rules: error rate, p95 latency, cost/hour, refusal spike, truncation rate, cache hit ratio
 *   - typed SDK error handling and logging (429 + retry-after, 529 overloaded, connection errors)
 *   - the SDK's own debug logging switch (ANTHROPIC_LOG=debug)
 *
 * Outputs go to <lab>/labs_output/ (gitignored).
 *
 * Run from the repo root:
 *   node "D4.6 - Logging and Observability Tooling/typescript/main.ts"
 */

import Anthropic from "@anthropic-ai/sdk";
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  MODEL_SONNET,
  banner,
  costUsd,
  elapsedMs,
  estimateTokens,
  getClient,
  isLive,
  section,
  sleep,
  textBlock,
  toolUseBlock,
  type Json,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const OUT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "labs_output");
const MODEL = MODEL_SONNET;
const MAX_ATTEMPTS = 3;
const MAX_STEPS = 4;

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
const pct = (x: number): string => fmt(100 * x, 1) + "%";
function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((x, y) => x - y);
  const rank = (p / 100) * (s.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo]! + (s[hi]! - s[lo]!) * (rank - lo);
}
const sha256 = (s: string): string => createHash("sha256").update(s).digest("hex");

// ---------------------------------------------------------------------------
// 1. PII redaction - applied to every string BEFORE it reaches a log sink
// ---------------------------------------------------------------------------
const REDACTIONS: [string, RegExp][] = [
  ["[EMAIL]", /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g],
  ["[SSN]", /\b\d{3}-\d{2}-\d{4}\b/g],
  ["[CARD]", /\b(?:\d[ -]?){13,16}\b/g],
  ["[PHONE]", /\b\d{3}[-. ]\d{3}[-. ]\d{4}\b/g],
];
function redact(text: string): string {
  for (const [token, pattern] of REDACTIONS) text = text.replace(pattern, token);
  return text;
}
function preview(text: string, n = 80): string {
  const t = redact(text.replace(/\n/g, " "));
  return t.length <= n ? t : t.slice(0, n - 3) + "...";
}

// ---------------------------------------------------------------------------
// 2. Clock + ids (simulated offline so logs are reproducible; real time live)
// ---------------------------------------------------------------------------
class Clock {
  ms: number;
  constructor(startIso: string) {
    this.ms = Date.parse(startIso);
  }
  now(): number {
    return isLive() ? Date.now() : this.ms;
  }
  advance(ms: number): void {
    if (!isLive()) this.ms += Math.floor(ms + 0.5);
  }
  set(ms: number): void {
    if (!isLive()) this.ms = ms;
  }
  async sleep(seconds: number): Promise<void> {
    if (isLive()) await sleep(Math.min(seconds, 2) * 1000); // keep the live demo short
    else this.advance(seconds * 1000);
  }
}
const iso = (ms: number): string => new Date(ms).toISOString();

const START_ISO = "2026-10-11T09:00:00Z";
const CLOCK = new Clock(START_ISO);
const START_MS = CLOCK.ms;
const ID_RNG = mulberry32(20261011);
const hexId = (chunks: number): string =>
  Array.from({ length: chunks }, () => Math.floor(ID_RNG() * 4294967296).toString(16).padStart(8, "0")).join("");

// ---------------------------------------------------------------------------
// 3. Minimal OpenTelemetry-style tracer
// ---------------------------------------------------------------------------
interface Span {
  traceId: string;
  spanId: string;
  parentSpanId: string | null;
  name: string;
  startMs: number;
  endMs: number;
  status: string;
  attributes: Json;
}
const durationMs = (s: Span): number => s.endMs - s.startMs;

class Tracer {
  spans: Span[] = [];
  start(name: string, parent: Span | null, attrs: Json = {}): Span {
    const span: Span = {
      traceId: parent ? parent.traceId : hexId(4),
      spanId: hexId(2),
      parentSpanId: parent ? parent.spanId : null,
      name,
      startMs: CLOCK.now(),
      endMs: 0,
      status: "UNSET",
      attributes: { ...attrs },
    };
    this.spans.push(span);
    return span;
  }
  end(span: Span, status = "OK", attrs: Json = {}): void {
    Object.assign(span.attributes, attrs);
    span.status = status;
    span.endMs = CLOCK.now();
  }
  export(): Json[] {
    return this.spans.map((s) => ({
      trace_id: s.traceId, span_id: s.spanId, parent_span_id: s.parentSpanId, name: s.name,
      start_time: iso(s.startMs), end_time: iso(s.endMs), duration_ms: durationMs(s), status: s.status, attributes: s.attributes,
    }));
  }
}

// ---------------------------------------------------------------------------
// 4. Structured logger + metrics registry
// ---------------------------------------------------------------------------
class JsonLogger {
  records: Json[] = [];
  log(level: string, event: string, span: Span | null, fields: Json): Json {
    const rec: Json = { ts: iso(CLOCK.now()), level, event, trace_id: span ? span.traceId : null, span_id: span ? span.spanId : null };
    for (const [k, v] of Object.entries(fields)) rec[k] = typeof v === "string" ? redact(v) : v; // defence in depth: redact every string field
    this.records.push(rec);
    return rec;
  }
}

const BUCKETS = [300, 600, 900, 1200, 2400, 4800, Infinity];

class Metrics {
  counters = new Map<string, number>();
  hist = new Map<string, number[]>();
  sums = new Map<string, number>();
  inc(name: string, n = 1): void {
    this.counters.set(name, (this.counters.get(name) ?? 0) + n);
  }
  observe(name: string, value: number): void {
    if (!this.hist.has(name)) this.hist.set(name, []);
    this.hist.get(name)!.push(value);
  }
  add(name: string, value: number): void {
    this.sums.set(name, (this.sums.get(name) ?? 0) + value);
  }
  c(name: string): number {
    return this.counters.get(name) ?? 0;
  }
  s(name: string): number {
    return this.sums.get(name) ?? 0;
  }
  bucketCounts(name: string): number[] {
    const vals = this.hist.get(name) ?? [];
    return BUCKETS.map((le) => vals.filter((v) => v <= le).length); // cumulative, Prometheus-style
  }
  /** Estimate a quantile from cumulative buckets with linear interpolation (like PromQL histogram_quantile). */
  histogramQuantile(name: string, q: number): number {
    const cum = this.bucketCounts(name);
    const n = cum[cum.length - 1]!;
    if (n === 0) return 0;
    const rank = q * n;
    let prevLe = 0;
    let prevC = 0;
    for (let i = 0; i < BUCKETS.length; i++) {
      const le = BUCKETS[i]!;
      const cnt = cum[i]!;
      if (cnt >= rank) {
        if (!Number.isFinite(le)) return prevLe;
        return prevLe + (le - prevLe) * (cnt > prevC ? (rank - prevC) / (cnt - prevC) : 0);
      }
      prevLe = le;
      prevC = cnt;
    }
    return prevLe;
  }
}

const LOG = new JsonLogger();
const TRACER = new Tracer();
const METRICS = new Metrics();

// ---------------------------------------------------------------------------
// 5. The agent under observation: tools, sessions, offline responder
// ---------------------------------------------------------------------------
const ORDERS: Record<string, Json> = {
  "ORD-1001": { status: "in_transit", eta: "2026-10-14" },
  "ORD-1002": { status: "refunded", refund_amount: 49.9, refund_date: "2026-10-09" },
  "ORD-1003": { status: "in_transit", eta: "2026-10-16" },
  "ORD-1004": { status: "delivered", delivered: "2026-10-01", return_window_until: "2026-10-31" },
  "ORD-1005": { status: "processing" },
  "ORD-1008": { status: "in_transit", eta: "2026-10-15" },
  "ORD-1009": { status: "in_transit", eta: "2026-10-13" },
};
const CUSTOMERS: Record<string, Json> = {
  "jane.doe@example.com": { name: "Jane Doe", email: "jane.doe@example.com", phone: "555-123-4567", address: "12 Elm Street", tier: "gold" },
};
const TOOLS: Anthropic.Tool[] = [
  {
    name: "get_order",
    description: "Look up an order's status by order id.",
    input_schema: { type: "object", properties: { order_id: { type: "string" } }, required: ["order_id"] },
  },
  {
    name: "get_customer",
    description: "Look up a customer profile by email address.",
    input_schema: { type: "object", properties: { email: { type: "string" } }, required: ["email"] },
  },
];
const TOOL_LATENCY: Record<string, number> = { get_order: 120, get_customer: 80 };
const PROMPT_VERSION = "support-agent@v7";
const SYSTEM_TEXT =
  "You are Acme's order-support agent. Use get_order for order questions and get_customer only when the user " +
  "gives an email address. Answer in one or two sentences. Never repeat personal data back to the user. " +
  Array.from(
    { length: 40 },
    (_, i) => `Guideline ${i + 1}: follow the escalation matrix for case type ${i + 1}, confirm identity before account changes, and log the outcome in the case system.`,
  ).join(" ");
const SYSTEM: Anthropic.TextBlockParam[] = [{ type: "text", text: SYSTEM_TEXT, cache_control: { type: "ephemeral" } }];
const PROMPT_HASH = sha256(SYSTEM_TEXT).slice(0, 8);

interface Session {
  sid: string;
  user: string;
  text: string;
  maxTokens: number;
}
const SESSIONS: Session[] = [
  { sid: "S1", user: "u1@example.com", text: "Where is my order ORD-1001?", maxTokens: 1024 },
  { sid: "S2", user: "jane.doe@example.com", text: "Hi, I'm Jane Doe (jane.doe@example.com, 555-123-4567). Has my refund for ORD-1002 been processed?", maxTokens: 1024 },
  { sid: "S3", user: "u3@example.com", text: "Can you check the delivery date for ORD-1003?", maxTokens: 1024 },
  { sid: "S4", user: "u4@example.com", text: "Is ORD-1004 still eligible for a return?", maxTokens: 1024 },
  { sid: "S5", user: "u5@example.com", text: "Track ORD-1005 please.", maxTokens: 1024 },
  { sid: "S6", user: "u6@example.com", text: "[RED-TEAM PROMPT agent-01 | policy area: credential theft | full text held in the red-team store]", maxTokens: 1024 },
  { sid: "S7", user: "u7@example.com", text: "Summarize every interaction on my account for ORD-1007 in full detail.", maxTokens: 60 }, // misconfigured route
  { sid: "S8", user: "u8@example.com", text: "Which arrives first, ORD-1008 or ORD-1009?", maxTokens: 1024 },
];

function runTool(name: string, args: Json): [string, boolean] {
  if (name === "get_order") {
    const rec = ORDERS[String(args.order_id)];
    return rec ? [JSON.stringify({ order_id: args.order_id, ...rec }), true] : [JSON.stringify({ error: "order not found" }), false];
  }
  if (name === "get_customer") {
    const cust = CUSTOMERS[String(args.email)];
    return cust ? [JSON.stringify(cust), true] : [JSON.stringify({ error: "customer not found" }), false];
  }
  return [JSON.stringify({ error: `unknown tool ${name}` }), false];
}

function finalAnswer(results: Json[]): string {
  const data = results.map((r) => JSON.parse(String(r.content)) as Json);
  const orders = data.filter((d) => "order_id" in d);
  if (orders.length === 2 && orders.every((o) => "eta" in o)) {
    let first = orders[0]!;
    for (const o of orders) if (o.eta < first.eta) first = o;
    return `${first.order_id} arrives first, on ${first.eta}.`;
  }
  const o = orders[0]!;
  if (o.status === "refunded") return `Yes - your refund of ${fmt(o.refund_amount, 2)} dollars for ${o.order_id} was processed on ${o.refund_date}.`;
  if (o.status === "delivered") return `Yes - ${o.order_id} was delivered on ${o.delivered} and can be returned until ${o.return_window_until}.`;
  return `${o.order_id} is in transit and should arrive on ${o.eta}.`;
}

function responder(req: MockRequest): ResponderResult {
  if (req.toolResults.length) {
    const text = finalAnswer(req.toolResults);
    return { content: [textBlock(text)], usage: { output_tokens: estimateTokens(text) } };
  }
  const user = req.lastUserText;
  if (user.includes("RED-TEAM")) {
    return {
      content: [], stop_reason: "refusal", usage: { output_tokens: 0 },
      stop_details: { type: "refusal", category: "cyber", explanation: "The request could enable cyber harm." },
    };
  }
  if (user.includes("Summarize every interaction")) {
    return "Here is a complete history of your account. " +
      Array.from({ length: 29 }, (_, i) => `On day ${i + 1} you contacted support about delivery, billing and settings, and the agent resolved the case.`).join(" ");
  }
  if (user.includes("Track ORD-1005")) throw new Error("connection reset by peer"); // surfaces in the SDK as APIConnectionError
  const blocks: Json[] = [];
  const email = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/.exec(user);
  if (email) blocks.push(toolUseBlock("get_customer", { email: email[0] }, "toolu_customer"));
  for (const m of user.matchAll(/ORD-\d{4}/g)) blocks.push(toolUseBlock("get_order", { order_id: m[0] }, `toolu_${m[0].replace("-", "")}`));
  const outTokens = blocks.reduce((s, b) => s + estimateTokens(b.name + JSON.stringify(b.input)) + 10, 0);
  return { content: blocks, usage: { output_tokens: outTokens } };
}

// ---------------------------------------------------------------------------
// 6. Fault injection with REAL SDK exception types (offline demo of error paths)
// ---------------------------------------------------------------------------
function statusError(status: number, errType: string, retryAfter: string | null, reqId: string): InstanceType<typeof Anthropic.APIError> {
  const headers = new Headers({ "request-id": reqId });
  if (retryAfter) headers.set("retry-after", retryAfter);
  const body = { type: "error", error: { type: errType, message: `injected ${errType}` } };
  return Anthropic.APIError.generate(status, body, `injected ${errType}`, headers); // 429 -> RateLimitError, 529 -> InternalServerError
}
const FAULTS = new Map<string, () => Error>([
  ["S3|1|0", () => statusError(429, "rate_limit_error", "2", "req_fault_s3_0")],
  ["S4|1|0", () => statusError(529, "overloaded_error", null, "req_fault_s4_0")],
  ["S4|1|1", () => statusError(529, "overloaded_error", null, "req_fault_s4_1")],
]);
const FAILED_ATTEMPT_MS: Record<string, number> = { rate_limit_error: 40, overloaded_error: 300, connection_error: 1500, api_error: 200 };
const LAT_PROFILE: [number, number] = [500, 7]; // Sonnet: TTFT ms, ms per output token

/** One logical LLM call: attempts with typed error handling, backoff, one log record + span per attempt. */
async function llmCall(client: Anthropic, sess: Session, step: number, messages: Anthropic.MessageParam[], parent: Span, sessionId: string): Promise<Anthropic.Message | null> {
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const span = TRACER.start("llm.call", parent, { "gen_ai.system": "anthropic", "gen_ai.request.model": MODEL, attempt });
    const base = { session_id: sessionId, prompt_version: PROMPT_VERSION, prompt_hash: PROMPT_HASH, model: MODEL, step, attempt };
    let err: string;
    let status: number | null;
    let retryAfter: number | null;
    let reqId: string | null;
    let retryable: boolean;
    try {
      const fault = FAULTS.get(`${sess.sid}|${step}|${attempt}`);
      if (fault) throw fault();
      const t0 = performance.now();
      const msg = await client.messages.create({ model: MODEL, max_tokens: sess.maxTokens, system: SYSTEM, tools: TOOLS, messages });
      const wall = elapsedMs(t0);
      const rng = mulberry32(fnv1a32(`${sess.sid}:${step}:${attempt}`));
      const latency = isLive() ? wall : (LAT_PROFILE[0] * (0.85 + 0.3 * rng()) + LAT_PROFILE[1] * msg.usage.output_tokens) * (rng() < 0.15 ? 2.5 : 1.0); // seeded slow tail
      CLOCK.advance(latency);
      const u = msg.usage;
      const cost = costUsd(msg.model, u);
      const toolNames = msg.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use").map((b) => b.name);
      LOG.log(msg.stop_reason === "max_tokens" || msg.stop_reason === "refusal" ? "WARN" : "INFO", "llm.call", span, {
        ...base, request_id: msg._request_id, input_tokens: u.input_tokens, output_tokens: u.output_tokens,
        cache_read_tokens: u.cache_read_input_tokens ?? 0, cache_write_tokens: u.cache_creation_input_tokens ?? 0,
        cost_usd: Math.floor(cost * 1e8 + 0.5) / 1e8, latency_ms: Math.floor(latency + 0.5), stop_reason: msg.stop_reason,
        tool_names: toolNames, error_type: null, user_message_preview: preview(sess.text),
      });
      METRICS.inc("llm_requests_total{outcome=ok}");
      METRICS.inc(`llm_stop_reason_total{reason=${msg.stop_reason}}`);
      METRICS.observe("llm_latency_ms", latency);
      const adds: [string, number][] = [["tokens_input", u.input_tokens], ["tokens_output", u.output_tokens], ["cost_usd", cost],
        ["tokens_cache_read", u.cache_read_input_tokens ?? 0], ["tokens_cache_write", u.cache_creation_input_tokens ?? 0]];
      for (const [k, v] of adds) METRICS.add(k, v);
      TRACER.end(span, "OK", {
        "gen_ai.response.id": msg.id, "gen_ai.usage.input_tokens": u.input_tokens,
        "gen_ai.usage.output_tokens": u.output_tokens, "gen_ai.response.finish_reasons": [msg.stop_reason],
      });
      return msg;
    } catch (e) {
      // Most specific first. In the TS SDK APIConnectionError extends APIError, so check it before APIError.
      if (e instanceof Anthropic.RateLimitError) {
        [err, status, retryAfter, reqId, retryable] = ["rate_limit_error", 429, Number(e.headers.get("retry-after") ?? "1"), e.requestID ?? null, true];
      } else if (e instanceof Anthropic.APIConnectionError) {
        [err, status, retryAfter, reqId, retryable] = ["connection_error", null, null, null, true];
      } else if (e instanceof Anthropic.APIError) {
        [err, status, retryAfter, reqId, retryable] = [e.type ?? "api_error", e.status ?? null, null, e.requestID ?? null, (e.status ?? 0) >= 500];
      } else {
        throw e;
      }
    }
    CLOCK.advance(FAILED_ATTEMPT_MS[err] ?? 200);
    const last = attempt === MAX_ATTEMPTS - 1 || !retryable;
    const rng = mulberry32(fnv1a32(`backoff:${sess.sid}:${step}:${attempt}`));
    const delay = retryAfter !== null ? retryAfter : Math.min(8, 2 ** attempt) * (1 + 0.25 * rng());
    LOG.log("ERROR", "llm.call", span, {
      ...base, request_id: reqId, error_type: err, status_code: status, retry_after_s: retryAfter, retryable,
      next_retry_in_s: last ? null : Math.floor(delay * 1000 + 0.5) / 1000, user_message_preview: preview(sess.text),
    });
    METRICS.inc(`llm_requests_total{outcome=error,error_type=${err}}`);
    TRACER.end(span, "ERROR", { "error.type": err, "http.status_code": status });
    if (last) return null;
    await CLOCK.sleep(delay);
  }
  return null;
}

interface Outcome {
  sid: string;
  status: string;
  stop: string;
  root: Span;
}

async function runSession(client: Anthropic, sess: Session): Promise<Outcome> {
  const sessionId = sha256(`acme-salt:${sess.user}`).slice(0, 12); // pseudonymous, never the raw email
  const root = TRACER.start("agent.session", null, { "session.id": sessionId, "prompt.version": PROMPT_VERSION });
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: sess.text }];
  let status = "OK";
  let stop = "";
  for (let step = 1; step <= MAX_STEPS; step++) {
    const stepSpan = TRACER.start("agent.step", root, { step });
    const msg = await llmCall(client, sess, step, messages, stepSpan, sessionId);
    if (msg === null) {
      TRACER.end(stepSpan, "ERROR");
      [status, stop] = ["ERROR", "error"];
      break;
    }
    stop = msg.stop_reason ?? "";
    if (msg.stop_reason !== "tool_use") {
      TRACER.end(stepSpan, "OK");
      if (stop === "max_tokens" || stop === "refusal") status = "DEGRADED";
      break;
    }
    messages.push({ role: "assistant", content: msg.content });
    const results: Anthropic.ToolResultBlockParam[] = [];
    for (const block of msg.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use")) {
      const input = block.input as Json;
      const tspan = TRACER.start("tool.call", stepSpan, { "tool.name": block.name, "tool.args": redact(JSON.stringify(input)) });
      const [result, ok] = runTool(block.name, input);
      const rng = mulberry32(fnv1a32(`tool:${sess.sid}:${block.id}`));
      const tlat = TOOL_LATENCY[block.name]! * (0.8 + 0.4 * rng());
      CLOCK.advance(tlat);
      LOG.log(ok ? "INFO" : "WARN", "tool.call", tspan, {
        session_id: sessionId, tool_name: block.name, tool_args: JSON.stringify(input), result_preview: preview(result, 60),
        latency_ms: Math.floor(tlat + 0.5), ok,
      });
      METRICS.inc(`tool_calls_total{tool=${block.name},ok=${ok}}`);
      METRICS.observe(`tool_latency_ms{tool=${block.name}}`, tlat);
      TRACER.end(tspan, ok ? "OK" : "ERROR");
      results.push({ type: "tool_result", tool_use_id: block.id, content: result });
    }
    messages.push({ role: "user", content: results });
    TRACER.end(stepSpan, "OK");
  }
  TRACER.end(root, status === "ERROR" ? "ERROR" : "OK", { "session.status": status });
  METRICS.inc(`sessions_total{status=${status}}`);
  return { sid: sess.sid, status, stop, root };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
/** Stable, language-neutral rendering of a log record for the console. */
function render(rec: Json): string {
  const parts = Object.entries(rec).map(([k, v]) => {
    let s: string;
    if (v === null || v === undefined) s = "null";
    else if (typeof v === "boolean") s = String(v);
    else if (typeof v === "number") s = Number.isInteger(v) ? String(v) : fmt(v, 6);
    else if (Array.isArray(v)) s = "[" + v.map((x) => `"${x}"`).join(",") + "]";
    else s = `"${v}"`;
    return `"${k}":${s}`;
  });
  return "{" + parts.join(", ") + "}";
}

async function main(): Promise<void> {
  banner("D4.6 - Logging and Observability Tooling");
  console.log(`Agent: order support on ${MODEL}, prompt ${PROMPT_VERSION} (hash ${PROMPT_HASH}), ${SESSIONS.length} sessions, tools: get_order, get_customer`);

  section("1. PII redaction before logging");
  const sample = "I'm jane.doe@example.com, call 555-123-4567, card 4111 1111 1111 1111, SSN 123-45-6789";
  console.log(`  raw     : ${sample}`);
  console.log(`  redacted: ${redact(sample)}`);
  console.log("  Redact at the logger (every string field), not at each call site - one forgotten call site leaks PII.");

  section("2. Run the instrumented agent");
  // SDK auto-retries are disabled so every attempt is visible to OUR handler and logged with its error type.
  const client = getClient(responder).withOptions({ maxRetries: 0 });
  const outcomes: Outcome[] = [];
  for (const [i, sess] of SESSIONS.entries()) {
    CLOCK.set(START_MS + i * 300_000); // simulated traffic: a session every 5 minutes (no-op live)
    outcomes.push(await runSession(client, sess));
  }
  console.log(`  ${"session".padEnd(9)}${"status".padEnd(10)}${"final stop".padEnd(12)}${"llm attempts".padStart(13)}${"tool calls".padStart(11)}${"duration ms".padStart(13)}`);
  for (const o of outcomes) {
    const llm = TRACER.spans.filter((s) => s.traceId === o.root.traceId && s.name === "llm.call").length;
    const tools = TRACER.spans.filter((s) => s.traceId === o.root.traceId && s.name === "tool.call").length;
    console.log(`  ${o.sid.padEnd(9)}${o.status.padEnd(10)}${o.stop.padEnd(12)}${String(llm).padStart(13)}${String(tools).padStart(11)}${String(durationMs(o.root)).padStart(13)}`);
  }

  section("3. Structured JSON log records (one per LLM / tool call)");
  const okRec = LOG.records.find(
    (r) => r.event === "llm.call" && r.tool_names?.length && r.level === "INFO" && r.session_id && String(r.user_message_preview).includes("[EMAIL]"),
  )!;
  console.log("  " + render({ ...okRec, request_id: String(okRec.request_id).slice(0, 12) + "..." }));
  const errRec = LOG.records.find((r) => r.error_type === "rate_limit_error")!;
  console.log("  " + render({ ...errRec, request_id: String(errRec.request_id) }));
  console.log("  Fields: who (pseudonymous session_id), what (model, prompt_version/hash), cost (tokens incl. cache, $),");
  console.log("  speed (latency_ms), outcome (stop_reason, error_type), correlation (trace_id/span_id, request_id for support).");

  section("4. Trace for session S2 (agent.session -> agent.step -> llm.call / tool.call)");
  const s2 = outcomes.find((o) => o.sid === "S2")!.root;
  const depth = new Map<string, number>([[s2.spanId, 0]]);
  for (const s of TRACER.spans) {
    if (s.traceId !== s2.traceId) continue;
    const d = s.parentSpanId ? depth.get(s.parentSpanId)! + 1 : 0;
    depth.set(s.spanId, d);
    let attrs = "";
    if (s.name === "llm.call") {
      attrs = ` in=${s.attributes["gen_ai.usage.input_tokens"]} out=${s.attributes["gen_ai.usage.output_tokens"]} finish=${(s.attributes["gen_ai.response.finish_reasons"] ?? ["-"])[0]}`;
    } else if (s.name === "tool.call") attrs = ` ${s.attributes["tool.name"]} args=${s.attributes["tool.args"]}`;
    else if (s.name === "agent.step") attrs = ` #${s.attributes.step}`;
    console.log(`  ${"  ".repeat(d)}${s.name}${attrs}  [${durationMs(s)} ms, ${s.status}]  span=${s.spanId}`);
  }
  console.log(`  trace_id=${s2.traceId} - the same id is on every log record of the session, so logs and traces join.`);

  section("5. Metrics dashboard");
  const okCalls = METRICS.c("llm_requests_total{outcome=ok}");
  const errCounts: [string, number][] = [...METRICS.counters]
    .filter(([k]) => k.includes("outcome=error"))
    .map(([k, v]): [string, number] => [k.split("error_type=")[1]!.replace(/\}$/, ""), v])
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const errTotal = errCounts.reduce((s, [, v]) => s + v, 0);
  const attempts = okCalls + errTotal;
  const sessOk = METRICS.c("sessions_total{status=OK}");
  const sessDeg = METRICS.c("sessions_total{status=DEGRADED}");
  const sessErr = METRICS.c("sessions_total{status=ERROR}");
  const lat = METRICS.hist.get("llm_latency_ms")!;
  const stops = new Map(
    [...METRICS.counters].filter(([k]) => k.startsWith("llm_stop_reason_total")).map(([k, v]): [string, number] => [k.split("reason=")[1]!.replace(/\}$/, ""), v]),
  );
  const windowH = isLive()
    ? Math.max(1e-6, (TRACER.spans[TRACER.spans.length - 1]!.endMs - TRACER.spans[0]!.startMs) / 3_600_000)
    : (CLOCK.now() - START_MS) / 3_600_000;
  const cost = METRICS.s("cost_usd");
  const cacheRatio = METRICS.s("tokens_cache_read") / (METRICS.s("tokens_cache_read") + METRICS.s("tokens_input"));
  console.log(`  LLM attempts   : ${attempts} total, ${okCalls} ok, ${errTotal} errors (${pct(errTotal / attempts)}) ` + errCounts.map(([k, v]) => `${k}=${v}`).join(" "));
  console.log(`  sessions       : OK=${sessOk} DEGRADED=${sessDeg} ERROR=${sessErr}`);
  console.log("  stop reasons   : " + [...stops].sort((a, b) => (a[0] < b[0] ? -1 : 1)).map(([k, v]) => `${k}=${v}`).join(" "));
  console.log(
    `  LLM latency ms : exact p50=${fmt(percentile(lat, 50), 0)} p95=${fmt(percentile(lat, 95), 0)} | from buckets p50~${fmt(METRICS.histogramQuantile("llm_latency_ms", 0.5), 0)} ` +
      `p95~${fmt(METRICS.histogramQuantile("llm_latency_ms", 0.95), 0)} (bucket estimates are coarse)`,
  );
  const cum = METRICS.bucketCounts("llm_latency_ms");
  let prev = 0;
  BUCKETS.forEach((le, i) => {
    const label = Number.isFinite(le) ? fmt(le, 0) : "+Inf";
    const n = cum[i]! - prev;
    console.log(`    le ${label.padStart(5)} | ${"#".repeat(n).padEnd(12)} ${n}`);
    prev = cum[i]!;
  });
  console.log(
    `  tokens         : input=${METRICS.s("tokens_input")} output=${METRICS.s("tokens_output")} ` +
      `cache_read=${METRICS.s("tokens_cache_read")} cache_write=${METRICS.s("tokens_cache_write")} (cache hit ratio ${pct(cacheRatio)})`,
  );
  console.log(`  cost           : $${fmt(cost, 6)} over ${fmt(windowH * 60, 1)} min -> $${fmt(cost / windowH, 4)}/hour`);
  const toolNames = [...new Set([...METRICS.counters.keys()].filter((k) => k.startsWith("tool_calls_total")).map((k) => k.split("tool=")[1]!.split(",")[0]!))].sort();
  console.log(
    "  tools          : " +
      toolNames
        .map((t) => {
          const calls = [...METRICS.counters].filter(([k]) => k.startsWith(`tool_calls_total{tool=${t},`)).reduce((s, [, v]) => s + v, 0);
          return `${t} calls=${calls} p95=${fmt(percentile(METRICS.hist.get(`tool_latency_ms{tool=${t}}`)!, 95), 0)}ms`;
        })
        .join("  "),
  );

  section("6. Alert rules");
  const refusals = stops.get("refusal") ?? 0;
  const truncations = stops.get("max_tokens") ?? 0;
  const rules: [string, number, number, ">" | "<", string, (v: number) => string][] = [
    ["LLM error rate (attempts)", errTotal / attempts, 0.05, ">", "page", pct],
    ["session failure rate", sessErr / SESSIONS.length, 0.05, ">", "page", pct],
    ["LLM p95 latency (ms)", percentile(lat, 95), 2000, ">", "ticket", (v) => fmt(v, 0)],
    ["cost per hour (USD)", cost / windowH, 5.0, ">", "ticket", (v) => "$" + fmt(v, 4)],
    ["refusal rate", refusals / okCalls, 0.05, ">", "ticket", pct],
    ["max_tokens truncation rate", truncations / okCalls, 0.02, ">", "ticket", pct],
    ["cache hit ratio", cacheRatio, 0.5, "<", "ticket", pct],
  ];
  console.log(`  ${"rule".padEnd(30)}${"value".padStart(10)}  ${"threshold".padEnd(12)}${"severity".padEnd(9)}state`);
  for (const [name, value, thr, op, sev, f] of rules) {
    const firing = op === ">" ? value > thr : value < thr;
    console.log(`  ${name.padEnd(30)}${f(value).padStart(10)}  ${(op + " " + f(thr)).padEnd(12)}${sev.padEnd(9)}${firing ? "FIRING" : "ok"}`);
  }
  console.log("  Alert on symptoms users feel (errors, latency, truncation) and on silent cost regressions (cache hit ratio).");

  section("7. Typed error handling (what each error means operationally)");
  for (const rec of LOG.records.filter((r) => r.error_type)) {
    const retry = rec.next_retry_in_s !== null && rec.next_retry_in_s !== undefined ? `retry in ${fmt(rec.next_retry_in_s, 2)} s` : "give up";
    console.log(
      `  ${rec.ts} session=${rec.session_id} step=${rec.step} attempt=${rec.attempt} ` +
        `${String(rec.error_type).padEnd(17)} status=${String(rec.status_code ?? "-").padEnd(4)} ` +
        `request_id=${String(rec.request_id || "-").padEnd(15)} -> ${retry}`,
    );
  }
  console.log("  429 rate_limit_error: honour retry-after | 529 overloaded / 5xx: exponential backoff + jitter |");
  console.log("  connection errors: retry, then fail the session cleanly | 4xx (400/401/403/404): never retry - fix the request.");

  section("8. SDK debug logging (ANTHROPIC_LOG=debug)");
  const captured: string[] = [];
  const capture = (...args: unknown[]): void => {
    captured.push(args.map(String).join(" "));
  };
  // Same effect as starting the process with ANTHROPIC_LOG=debug, but routed into our own logger.
  const debugClient = getClient(responder, { logLevel: "debug", logger: { debug: capture, info: capture, warn: capture, error: capture } });
  await debugClient.messages.create({ model: MODEL, max_tokens: 64, messages: [{ role: "user", content: "Where is my order ORD-1001?" }] });
  console.log(`  SDK debug lines captured for one request: ${captured.length ? "yes" : "no"}`);
  console.log("  Enable with ANTHROPIC_LOG=debug (or info) for local debugging; debug output includes request details and can");
  console.log("  contain prompt text - keep it off in production or route it through the same redaction pipeline.");

  section("9. Export");
  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(path.join(OUT_DIR, "llm_calls.jsonl"), LOG.records.map((r) => JSON.stringify(r)).join("\n") + "\n", "utf-8");
  writeFileSync(path.join(OUT_DIR, "spans.json"), JSON.stringify(TRACER.export(), null, 2), "utf-8");
  const histograms: Json = {};
  for (const k of METRICS.hist.keys()) {
    histograms[k] = { buckets_le: BUCKETS.map((b) => (Number.isFinite(b) ? fmt(b, 0) : "+Inf")), cumulative: METRICS.bucketCounts(k) };
  }
  writeFileSync(
    path.join(OUT_DIR, "metrics.json"),
    JSON.stringify({ counters: Object.fromEntries(METRICS.counters), sums: Object.fromEntries(METRICS.sums), histograms }, null, 2),
    "utf-8",
  );
  const leaked = LOG.records.filter((r) => Object.values(r).some((v) => typeof v === "string" && (v.includes("jane.doe@example.com") || v.includes("555-123-4567"))));
  console.log(
    `  labs_output/llm_calls.jsonl : ${LOG.records.length} records ` +
      `(${LOG.records.filter((r) => r.event === "llm.call").length} llm.call, ${LOG.records.filter((r) => r.event === "tool.call").length} tool.call)`,
  );
  console.log(`  labs_output/spans.json      : ${TRACER.spans.length} spans in ${new Set(TRACER.spans.map((s) => s.traceId)).size} traces`);
  console.log(`  labs_output/metrics.json    : ${METRICS.counters.size} counters, ${METRICS.hist.size} histograms`);
  console.log(`  PII check on written logs   : ${leaked.length} records contain raw email/phone`);
  console.log("  In production, ship these to your log store / OTel collector; this lab is the instrumentation, D3.4 covers");
  console.log("  the at-scale strategy (sampling, cardinality, retention, centralised dashboards).");
}

await main();
