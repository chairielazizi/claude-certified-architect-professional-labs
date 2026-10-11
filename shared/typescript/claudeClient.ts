/**
 * Shared Claude client used by every TypeScript lab in this repository.
 *
 * Every lab talks to Claude through the *real* Anthropic SDK (`@anthropic-ai/sdk`)
 * with real Messages API request shapes. When no credentials are configured, the
 * SDK is given a mock `fetch` that answers in the Messages API wire format, so:
 *
 *   - every demo runs offline, deterministically, and for free;
 *   - the code you read is exactly the code that runs against the live API;
 *   - set ANTHROPIC_API_KEY (or LABS_MODE=live) and the same lab goes live.
 *
 * Mode selection (env var LABS_MODE):
 *   live     -> always call the real API
 *   offline  -> always use the mock fetch
 *   (unset)  -> live if ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is set, else offline
 *
 * Offline simulation mirrors shared/python/claude_client.py: token estimates,
 * prompt-caching simulation, per-model latency, max_tokens truncation, SSE
 * streaming for text/tool_use blocks, and /v1/messages/count_tokens.
 *
 * Usage from a lab (folder layout: <root>/<lab>/typescript/main.ts):
 *
 *   import { getClient, MODEL_OPUS, textOf } from "../../shared/typescript/claudeClient.ts";
 *   const client = getClient((req) => `canned answer for: ${req.lastUserText}`);
 *   const msg = await client.messages.create({ model: MODEL_OPUS, max_tokens: 1024,
 *     messages: [{ role: "user", content: "Hi" }] });
 *   console.log(textOf(msg));
 */

import Anthropic from "@anthropic-ai/sdk";
import { createHash, randomUUID } from "node:crypto";

// ---------------------------------------------------------------------------
// Models & pricing (cached 2026-10; always confirm on the live pricing page)
// ---------------------------------------------------------------------------
export const MODEL_FABLE = "claude-fable-5-1"; // most capable widely released model
export const MODEL_OPUS = "claude-opus-5-5"; // default: frontier intelligence, agents, coding
export const MODEL_SONNET = "claude-sonnet-5-5"; // balanced speed / capability, high volume
export const MODEL_HAIKU = "claude-haiku-5-5"; // fastest / cheapest: routing, classification
export const DEFAULT_MODEL = MODEL_OPUS;

export interface Price {
  input: number;
  output: number;
  cacheWrite: number;
  cacheRead: number;
}

/** USD per 1M tokens. cacheWrite = 5-minute TTL write (~1.25x input). */
export const PRICING: Record<string, Price> = {
  [MODEL_FABLE]: { input: 10.0, output: 50.0, cacheWrite: 12.5, cacheRead: 0.25 },
  [MODEL_OPUS]: { input: 4.0, output: 20.0, cacheWrite: 5.0, cacheRead: 0.2 },
  [MODEL_SONNET]: { input: 2.0, output: 10.0, cacheWrite: 2.5, cacheRead: 0.2 },
  // Haiku 5.5 cache pricing assumed at the usual ratios (1.25x write, 0.1x read).
  [MODEL_HAIKU]: { input: 0.1, output: 0.5, cacheWrite: 0.125, cacheRead: 0.01 },
};

/** Simulated latency for offline mode: [time-to-first-token ms, ms per output token]. */
const LATENCY_PROFILE_MS: Record<string, [number, number]> = {
  [MODEL_HAIKU]: [250, 3],
  [MODEL_SONNET]: [500, 7],
  [MODEL_OPUS]: [900, 12],
  [MODEL_FABLE]: [1500, 18],
};
/** Offline sleeps are scaled down so demos stay fast; elapsedMs() scales them back up. */
export const LATENCY_SCALE = Number(process.env.LABS_LATENCY_SCALE ?? "0.1");
/** Simplified minimum cacheable prefix for the offline cache simulation. */
export const MIN_CACHEABLE_TOKENS = 1024;

// ---------------------------------------------------------------------------
// Mode helpers
// ---------------------------------------------------------------------------
export function isLive(): boolean {
  const mode = (process.env.LABS_MODE ?? "").trim().toLowerCase();
  if (mode === "live") return true;
  if (mode === "offline") return false;
  return Boolean(process.env.ANTHROPIC_API_KEY || process.env.ANTHROPIC_AUTH_TOKEN);
}

export function modeLabel(): string {
  return isLive()
    ? "LIVE (real Claude API)"
    : "OFFLINE (mock transport - set ANTHROPIC_API_KEY to go live)";
}

export function banner(title: string): void {
  const line = "=".repeat(Math.max(60, title.length + 4));
  console.log(line);
  console.log(`  ${title}`);
  console.log(`  mode: ${modeLabel()}`);
  console.log(line);
}

export function section(title: string): void {
  console.log(`\n--- ${title} ${"-".repeat(Math.max(3, 56 - title.length))}`);
}

/** Milliseconds since t0 (from performance.now()); offline values are scaled back up. */
export function elapsedMs(t0: number): number {
  const wall = performance.now() - t0;
  return isLive() ? wall : wall / LATENCY_SCALE;
}

export const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Token / cost helpers
// ---------------------------------------------------------------------------
/** Rough heuristic (~4 chars/token). Use client.messages.countTokens for real counts. */
export function estimateTokens(text: string): number {
  return text ? Math.max(1, Math.floor(text.length / 4)) : 0;
}

/** Dollar cost of one response from its usage object. */
export function costUsd(model: string, usage: Anthropic.Usage): number {
  const p = PRICING[model] ?? PRICING[DEFAULT_MODEL]!;
  return (
    ((usage.input_tokens ?? 0) * p.input +
      (usage.output_tokens ?? 0) * p.output +
      (usage.cache_creation_input_tokens ?? 0) * p.cacheWrite +
      (usage.cache_read_input_tokens ?? 0) * p.cacheRead) /
    1_000_000
  );
}

/** Concatenate all text blocks of a Message. */
export function textOf(message: Anthropic.Message): string {
  return message.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
}

export function usageLine(message: Anthropic.Message): string {
  const u = message.usage;
  return (
    `in=${u.input_tokens} out=${u.output_tokens} ` +
    `cache_write=${u.cache_creation_input_tokens ?? 0} cache_read=${u.cache_read_input_tokens ?? 0} ` +
    `cost=$${costUsd(message.model, u).toFixed(6)}`
  );
}

// ---------------------------------------------------------------------------
// Offline mock: request view + responder contract
// ---------------------------------------------------------------------------
/** Wire-format JSON, as sent to / returned by the API (loosely typed on purpose). */
export type Json = Record<string, any>;

function blockText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content as Json[]) {
    if (b?.type === "text") parts.push(String(b.text ?? ""));
    else if (b?.type === "tool_result") parts.push(blockText(b.content ?? ""));
  }
  return parts.join("\n");
}

/** Convenience view over a Messages API request body, handed to responders. */
export class MockRequest {
  readonly body: Json;
  /** How many requests this client has served before this one. */
  readonly callIndex: number;

  constructor(body: Json, callIndex = 0) {
    this.body = body;
    this.callIndex = callIndex;
  }
  get model(): string {
    return this.body.model ?? "";
  }
  get messages(): Json[] {
    return this.body.messages ?? [];
  }
  get systemText(): string {
    return blockText(this.body.system ?? "");
  }
  get tools(): Json[] {
    return this.body.tools ?? [];
  }
  get toolNames(): string[] {
    return this.tools.map((t) => String(t.name ?? ""));
  }
  get lastUserText(): string {
    for (const m of [...this.messages].reverse()) {
      if (m.role === "user") return blockText(m.content);
    }
    return "";
  }
  /** tool_result blocks in the most recent user turn (empty if none). */
  get toolResults(): Json[] {
    for (const m of [...this.messages].reverse()) {
      if (m.role === "user") {
        return Array.isArray(m.content) ? m.content.filter((b: Json) => b?.type === "tool_result") : [];
      }
    }
    return [];
  }
  /** System + every message's text - handy for keyword-based canned replies. */
  get allText(): string {
    return [this.systemText, ...this.messages.map((m) => blockText(m.content))].join("\n");
  }
}

/**
 * A responder returns one of:
 *   string                -> a single text block, stop_reason "end_turn"
 *   Json[]                -> content blocks (use textBlock / toolUseBlock helpers);
 *                            stop_reason is "tool_use" if any tool_use block is present
 *   { content, ... }      -> full control: { content, stop_reason?, usage?, stop_details? }
 */
export type ResponderResult =
  | string
  | Json[]
  | { content: Json[]; stop_reason?: string; usage?: Json; stop_details?: Json };
export type Responder = (req: MockRequest) => ResponderResult | Promise<ResponderResult>;

export function textBlock(text: string): Json {
  return { type: "text", text };
}

export function toolUseBlock(name: string, input: Json, id?: string): Json {
  return { type: "tool_use", id: id ?? `toolu_${randomUUID().replace(/-/g, "").slice(0, 20)}`, name, input };
}

export function defaultResponder(req: MockRequest): ResponderResult {
  let snippet = req.lastUserText.trim().replace(/\n/g, " ");
  if (snippet.length > 140) snippet = snippet.slice(0, 137) + "...";
  return `[offline mock reply from ${req.model}] You asked: "${snippet}"`;
}

// ---------------------------------------------------------------------------
// Offline mock: transport implementation
// ---------------------------------------------------------------------------
interface MockState {
  responder: Responder;
  calls: number;
  cache: Map<string, number>; // prefix hash -> expiry (epoch ms)
}

/** Stable JSON (sorted keys) so the cache simulation behaves like a prefix hash. */
function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(", ")}]`;
  if (value && typeof value === "object") {
    const entries = Object.keys(value as Json)
      .sort()
      .map((k) => `${JSON.stringify(k)}: ${stableStringify((value as Json)[k])}`);
    return `{${entries.join(", ")}}`;
  }
  return JSON.stringify(value);
}

function stripCacheControl(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripCacheControl);
  if (value && typeof value === "object") {
    const out: Json = {};
    for (const [k, v] of Object.entries(value as Json)) if (k !== "cache_control") out[k] = stripCacheControl(v);
    return out;
  }
  return value;
}

/** Prompt render order is tools -> system -> messages; flatten into cacheable segments. */
function segments(body: Json): Json[] {
  const segs: Json[] = [...(body.tools ?? [])];
  if (typeof body.system === "string" && body.system) segs.push({ type: "text", text: body.system });
  else if (Array.isArray(body.system)) segs.push(...body.system);
  for (const m of body.messages ?? []) {
    if (typeof m.content === "string") segs.push({ role: m.role, type: "text", text: m.content });
    else for (const b of m.content ?? []) segs.push({ role: m.role, ...b });
  }
  return segs;
}

function usageFor(body: Json, state: MockState): Json {
  const segs = segments(body);
  const total = estimateTokens(stableStringify(stripCacheControl(segs)));
  let breakpoint = -1;
  segs.forEach((s, i) => {
    if (s?.cache_control) breakpoint = i;
  });
  if (body.cache_control && segs.length) breakpoint = segs.length - 1; // top-level automatic caching
  const usage: Json = { input_tokens: total, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 };
  if (breakpoint < 0) return usage;
  const prefix = stableStringify(stripCacheControl(segs.slice(0, breakpoint + 1)));
  const prefixTokens = estimateTokens(prefix);
  if (prefixTokens < MIN_CACHEABLE_TOKENS) return usage; // silently not cached - like the real API
  const key = createHash("sha256").update(String(body.model ?? "") + prefix).digest("hex");
  const now = Date.now();
  if ((state.cache.get(key) ?? 0) > now) usage.cache_read_input_tokens = prefixTokens;
  else usage.cache_creation_input_tokens = prefixTokens;
  const hasHourTtl = JSON.stringify(segs.slice(0, breakpoint + 1)).includes('"ttl":"1h"');
  state.cache.set(key, now + (hasHourTtl ? 3_600_000 : 300_000));
  usage.input_tokens = Math.max(0, total - prefixTokens);
  return usage;
}

function normalize(result: ResponderResult): { content: Json[]; stop_reason?: string; usage?: Json; stop_details?: Json } {
  if (typeof result === "string") return { content: [textBlock(result)] };
  if (Array.isArray(result)) return { content: result };
  if (result && Array.isArray(result.content)) return { ...result };
  throw new TypeError("Responder must return a string, an array of blocks, or { content: [...] }");
}

async function buildMessage(body: Json, state: MockState): Promise<{ message: Json; latencyMs: number }> {
  const req = new MockRequest(body, state.calls);
  state.calls += 1;
  const out = normalize(await state.responder(req));
  const content = out.content;
  let stopReason = out.stop_reason ?? (content.some((b) => b.type === "tool_use") ? "tool_use" : "end_turn");
  let outTokens = estimateTokens(JSON.stringify(content));
  const maxTokens = Number(body.max_tokens ?? 4096);
  if (outTokens > maxTokens && stopReason === "end_turn") {
    for (const b of content) if (b.type === "text") b.text = String(b.text).slice(0, maxTokens * 4);
    outTokens = maxTokens;
    stopReason = "max_tokens";
  }
  const usage = { ...usageFor(body, state), output_tokens: outTokens, ...(out.usage ?? {}) };
  const message: Json = {
    id: `msg_offline_${randomUUID().replace(/-/g, "").slice(0, 16)}`,
    type: "message",
    role: "assistant",
    model: body.model ?? DEFAULT_MODEL,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage,
  };
  if (out.stop_details) message.stop_details = out.stop_details;
  const [ttft, perTok] = LATENCY_PROFILE_MS[message.model] ?? LATENCY_PROFILE_MS[DEFAULT_MODEL]!;
  return { message, latencyMs: (ttft + perTok * outTokens) * LATENCY_SCALE };
}

/** Render a complete message as a Messages API SSE stream. */
function toSSE(message: Json): string {
  const events: [string, Json][] = [];
  events.push([
    "message_start",
    { type: "message_start", message: { ...message, content: [], stop_reason: null, usage: { ...message.usage, output_tokens: 1 } } },
  ]);
  (message.content as Json[]).forEach((block, i) => {
    if (block.type === "text") {
      events.push(["content_block_start", { type: "content_block_start", index: i, content_block: { type: "text", text: "" } }]);
      const text = String(block.text);
      for (let j = 0; j < text.length; j += 24)
        events.push(["content_block_delta", { type: "content_block_delta", index: i, delta: { type: "text_delta", text: text.slice(j, j + 24) } }]);
    } else if (block.type === "tool_use") {
      events.push([
        "content_block_start",
        { type: "content_block_start", index: i, content_block: { type: "tool_use", id: block.id, name: block.name, input: {} } },
      ]);
      const raw = JSON.stringify(block.input);
      for (let j = 0; j < raw.length; j += 32)
        events.push(["content_block_delta", { type: "content_block_delta", index: i, delta: { type: "input_json_delta", partial_json: raw.slice(j, j + 32) } }]);
    } else {
      events.push(["content_block_start", { type: "content_block_start", index: i, content_block: block }]);
    }
    events.push(["content_block_stop", { type: "content_block_stop", index: i }]);
  });
  events.push([
    "message_delta",
    { type: "message_delta", delta: { stop_reason: message.stop_reason, stop_sequence: null }, usage: { output_tokens: message.usage.output_tokens } },
  ]);
  events.push(["message_stop", { type: "message_stop" }]);
  return events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join("");
}

function makeMockFetch(state: MockState) {
  return async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = new URL(typeof input === "string" ? input : input instanceof URL ? input.href : input.url);
    const body: Json = typeof init?.body === "string" ? JSON.parse(init.body) : {};
    const json = (status: number, data: unknown, headers: Record<string, string> = {}) =>
      new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json", ...headers } });

    if (url.pathname.endsWith("/messages/count_tokens")) {
      return json(200, { input_tokens: estimateTokens(stableStringify(stripCacheControl(segments(body)))) });
    }
    if (url.pathname.endsWith("/messages")) {
      const { message, latencyMs } = await buildMessage(body, state);
      await sleep(latencyMs);
      const headers = { "request-id": `req_offline_${randomUUID().replace(/-/g, "").slice(0, 12)}` };
      if (body.stream) {
        return new Response(toSSE(message), { status: 200, headers: { "content-type": "text/event-stream", ...headers } });
      }
      return json(200, message, headers);
    }
    return json(404, {
      type: "error",
      error: { type: "not_found_error", message: `Offline mock does not implement ${url.pathname}. Run with LABS_MODE=live.` },
    });
  };
}

// ---------------------------------------------------------------------------
// Public factory
// ---------------------------------------------------------------------------
/**
 * Return an Anthropic client: real when live, mock-fetch backed when offline.
 * `responder` is only consulted offline. Extra options go to `new Anthropic()`.
 */
export function getClient(responder?: Responder, options: ConstructorParameters<typeof Anthropic>[0] = {}): Anthropic {
  if (isLive()) return new Anthropic(options);
  const state: MockState = { responder: responder ?? defaultResponder, calls: 0, cache: new Map() };
  return new Anthropic({ ...options, apiKey: "offline-mock-key", maxRetries: 0, fetch: makeMockFetch(state) });
}
