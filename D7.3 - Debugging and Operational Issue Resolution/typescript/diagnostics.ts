/**
 * Operational debugging checklist as code (D7.3) - twin of python/diagnostics.py.
 *
 *   lintRequest()      - catch the 400s before they happen (removed params on current models, prefill,
 *                        forced tool_choice, broken tool_use/tool_result pairing, bad model ids)
 *   diagnoseResponse() - interpret stop_reason (max_tokens truncation, refusal, unfinished tool loop)
 *   cacheCheck()       - flag cache_read_input_tokens == 0 on repeated calls (silent cache invalidators)
 */

import type Anthropic from "@anthropic-ai/sdk";
import { MODEL_FABLE, MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET } from "../../shared/typescript/claudeClient.ts";

export type Json = Record<string, any>;

const KNOWN_MODELS = new Set([MODEL_FABLE, MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU]);
const NO_SAMPLING_PARAMS = new Set([MODEL_FABLE, MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU]); // temperature/top_p/top_k -> 400
const NO_FORCED_TOOL_CHOICE = new Set([MODEL_FABLE, MODEL_OPUS, MODEL_SONNET]); // tool_choice any/tool -> 400
const NO_THINKING_DISABLED = new Set([MODEL_FABLE, MODEL_OPUS, MODEL_SONNET]); // thinking.type=disabled -> 400

const blocks = (content: unknown): Json[] => (Array.isArray(content) ? (content.filter((b) => b && typeof b === "object") as Json[]) : []);

/** Return [code, problem, fix] tuples for request shapes the API would reject. */
export function lintRequest(params: Json): [string, string, string][] {
  const issues: [string, string, string][] = [];
  const model = String(params.model ?? "");
  const messages: Json[] = params.messages ?? [];

  if (!KNOWN_MODELS.has(model)) {
    const hint = model.replace(/(\d)\.(\d)/g, "$1-$2");
    issues.push(["R009", `unknown model id "${model}" -> 404 not_found_error`, `use an exact id, e.g. "${hint}"`]);
  }
  for (const p of ["temperature", "top_p", "top_k"]) {
    if (p in params && NO_SAMPLING_PARAMS.has(model)) {
      issues.push(["R001", `${p} is not accepted on ${model} -> 400`, "delete it; steer with the prompt and output_config.effort"]);
    }
  }
  const thinking: Json = params.thinking ?? {};
  if (thinking.type === "enabled" || "budget_tokens" in thinking) {
    issues.push(["R002", "thinking budget_tokens is removed on current models -> 400", 'use thinking={"type": "adaptive"} and output_config={"effort": ...}']);
  }
  if (thinking.type === "disabled" && NO_THINKING_DISABLED.has(model)) {
    const fix = model === MODEL_SONNET ? 'Sonnet 5.5: use {"type": "between_tools"} at effort <= high' : "omit thinking and lower output_config.effort";
    issues.push(["R003", `thinking disabled is not supported on ${model} -> 400`, fix]);
  }
  if (["any", "tool"].includes((params.tool_choice ?? {}).type) && NO_FORCED_TOOL_CHOICE.has(model)) {
    issues.push(["R005", `forced tool_choice is not supported on ${model} -> 400`, "use auto + an instruction (strict: true tools) or structured outputs"]);
  }
  if (!Number.isInteger(params.max_tokens) || params.max_tokens <= 0) {
    issues.push(["R008", "max_tokens missing or not a positive integer -> 400", "set max_tokens explicitly"]);
  }
  if (messages.length && messages[0]!.role !== "user") {
    issues.push(["R006", "first message must have role user -> 400", "start the conversation with a user turn"]);
  }
  if (messages.length && messages[messages.length - 1]!.role === "assistant") {
    issues.push(["R004", "conversation ends with an assistant turn (prefill) -> 400 on current models", "end with a user turn; use structured outputs to control the format"]);
  }

  // tool_use / tool_result pairing: every tool_use id must be answered in the NEXT user message,
  // tool_result blocks first, and every tool_result must reference a tool_use from the turn before.
  messages.forEach((m, i) => {
    if (m.role !== "assistant") return;
    const ids = blocks(m.content).filter((b) => b.type === "tool_use").map((b) => String(b.id));
    if (!ids.length) return;
    const nxt = messages[i + 1];
    const nextBlocks = nxt && nxt.role === "user" ? blocks(nxt.content) : [];
    const results = nextBlocks.filter((b) => b.type === "tool_result").map((b) => String(b.tool_use_id));
    for (const tid of ids) {
      if (!results.includes(tid)) {
        issues.push(["R007", `tool_use ${tid} has no matching tool_result in the next user message -> 400`, "send one tool_result per tool_use id, in the very next user turn"]);
      }
    }
    for (const rid of results) {
      if (!ids.includes(rid)) issues.push(["R007", `tool_result references unknown tool_use_id ${rid} -> 400`, "copy the id from the tool_use block"]);
    }
    const firstResult = nextBlocks.findIndex((b) => b.type === "tool_result");
    if (firstResult > 0 && nextBlocks.slice(0, firstResult).some((b) => b.type !== "tool_result")) {
      issues.push(["R007", "text placed before tool_result blocks in the user message -> 400", "put tool_result blocks first, text after"]);
    }
  });
  return issues;
}

export function diagnoseResponse(msg: Anthropic.Message): string {
  const reason = msg.stop_reason;
  if (reason === "max_tokens") return "TRUNCATED: hit max_tokens - raise max_tokens or stream; never parse or act on partial output";
  if (reason === "refusal") return "REFUSAL: inspect stop_details.category; rephrase or route - do not blind-retry";
  if (reason === "tool_use") return "TOOL_USE: the loop is not finished - run the tools and send tool_result blocks";
  if (reason === "pause_turn") return "PAUSE_TURN: server tool paused - resend the conversation to continue";
  if (reason === "end_turn") {
    const hasText = msg.content.some((b) => b.type === "text" && b.text);
    return hasText ? "OK: end_turn" : "SUSPICIOUS: end_turn with no text";
  }
  return `UNEXPECTED stop_reason=${reason}`;
}

export function cacheCheck(usages: Anthropic.Usage[]): string {
  const reads = usages.map((u) => u.cache_read_input_tokens ?? 0);
  const writes = usages.map((u) => u.cache_creation_input_tokens ?? 0);
  if (usages.length > 1 && reads.every((r) => r === 0)) {
    if (writes.every((w) => w > 0)) {
      return "FAIL: every call WRITES the cache and none READS it - the prefix changes per request " +
        "(timestamp/UUID in system prompt, unsorted JSON, tool list or model changes). You pay 1.25x for nothing.";
    }
    return "FAIL: no cache activity - prefix below the minimum cacheable length or no cache_control breakpoint";
  }
  return `OK: cache reads on ${reads.filter((r) => r > 0).length}/${usages.length} calls`;
}

export const CHECKLIST: [string, string, string][] = [
  ["Output cut off / invalid JSON", "stop_reason == max_tokens?", "raise max_tokens, stream, or split the task"],
  ["Agent loop stops early", "stop_reason tool_use / pause_turn handled?", "execute tools, append tool_result, continue"],
  ["400 tool_result errors", "ids paired, results first, next user turn?", "fix the history builder (append-only)"],
  ["400 after model upgrade", "temperature/top_p/top_k, budget_tokens, prefill, forced tool_choice?", "remove; use adaptive thinking + effort"],
  ["Costs jumped, cache_read=0", "volatile data early in prompt? tools/model changed?", "freeze the prefix, move volatile data last"],
  ["429 rate_limit_error", "which limit (RPM/ITPM/OTPM)? retry-after honored?", "throttle, honor retry-after, restore caching"],
  ["529 overloaded_error", "status page? our retries amplifying?", "jittered backoff, circuit breaker, fallback model"],
  ["404 not_found_error", "exact model id (claude-opus-5-5)? enabled for org?", "fix the id; GET /v1/models to list"],
  ["Can't reproduce a failure", "request_id logged? ANTHROPIC_LOG=debug?", "log request_id on every call and error"],
];
