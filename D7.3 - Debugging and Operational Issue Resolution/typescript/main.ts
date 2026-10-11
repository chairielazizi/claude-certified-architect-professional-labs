/**
 * D7.3 - Debugging and Operational Issue Resolution (TypeScript twin of python/main.py)
 *
 * Incident: after deploy v2.14.0 the summarizer service hits Claude API 429s and 529s.
 *   1. deterministic log analysis: parse, cluster error signatures, timeline, deploy correlation
 *   2. triage agent: manual tool-use loop (search_logs, get_metrics, get_recent_deploys, get_runbook)
 *      -> structured incident report, then grounding checks against the deterministic facts
 *   3. resilient client: typed SDK errors most-specific-first, retry-after, jittered backoff,
 *      fallback model, circuit breaker, request_id logging (errors simulated - see resilience.ts)
 *   4. operational debugging checklist as code: max_tokens truncation, cache_read=0, request linter
 *   5. the incident report
 */

import Anthropic from "@anthropic-ai/sdk";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET, banner, costUsd, getClient, section, textOf, toolUseBlock,
} from "../../shared/typescript/claudeClient.ts";
import type { Json, MockRequest, ResponderResult } from "../../shared/typescript/claudeClient.ts";
import { CHECKLIST, cacheCheck, diagnoseResponse, lintRequest } from "./diagnostics.ts";
import { bar, clusterErrors, correlate, epoch, mean, parseLog } from "./loganalysis.ts";
import type { Facts, LogRecord } from "./loganalysis.ts";
import { AllModelsFailedError, FaultInjector, ResilientClaude, VirtualClock, mulberry32 } from "./resilience.ts";
import type { CreateParams, Fault } from "./resilience.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(HERE, "..", "data");
const OUT = path.join(HERE, "..", "labs_output");
const NOW = "2026-10-08T14:15:00Z"; // frozen "current time" so the demo is deterministic
const MAX_TURNS = 8;

const readData = (name: string): string => readFileSync(path.join(DATA, name), "utf8").replace(/\r\n/g, "\n");
const loadJson = (name: string): any => JSON.parse(readData(name));

function wrap(text: string, width = 96, indent = "    "): string[] {
  const lines: string[] = [];
  let cur = "";
  for (const word of text.split(/\s+/).filter(Boolean)) {
    if (cur && cur.length + 1 + word.length > width) {
      lines.push(cur);
      cur = word;
    } else {
      cur = cur ? `${cur} ${word}` : word;
    }
  }
  if (cur) lines.push(cur);
  return lines.map((l) => indent + l);
}

// ---------------------------------------------------------------------------
// Incident report schema: JSON schema for the API, Zod for validation
// ---------------------------------------------------------------------------
const IncidentReportSchema = z.object({
  title: z.string(),
  severity: z.enum(["SEV1", "SEV2", "SEV3"]),
  impact: z.string(),
  trigger: z.string(),
  root_cause: z.string(),
  contributing_factors: z.array(z.string()),
  evidence: z.array(z.string()),
  timeline: z.array(z.object({ time: z.string(), event: z.string() })),
  immediate_actions: z.array(z.string()),
  follow_up_actions: z.array(z.string()),
  runbook_id: z.string(),
  runbook_url: z.string(),
  confidence: z.number(),
});
type IncidentReport = z.infer<typeof IncidentReportSchema>;

// ---------------------------------------------------------------------------
// Triage tools: real implementations over the incident data
// ---------------------------------------------------------------------------
const METRIC_NAMES = ["rps", "p95_latency_ms", "error_rate", "claude_429", "claude_529", "cache_hit_ratio", "input_tokens_per_min", "in_flight"];

const TOOLS: Anthropic.Tool[] = [
  {
    name: "search_logs",
    description:
      "Search application log lines (case-insensitive substring). Returns the total number of matches and " +
      "the first `limit` lines. since/until are HH:MM in UTC, or empty strings for no bound.",
    strict: true,
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        level: { type: "string", enum: ["ANY", "INFO", "WARN", "ERROR"] },
        since: { type: "string" },
        until: { type: "string" },
        limit: { type: "integer" },
      },
      required: ["query", "level", "since", "until", "limit"],
      additionalProperties: false,
    },
  },
  {
    name: "get_metrics",
    description:
      "Per-minute metrics for the summarizer service between start and end (HH:MM, UTC). " +
      "Returns the points plus max and mean before/after the latest deploy.",
    strict: true,
    input_schema: {
      type: "object",
      properties: { metric: { type: "string", enum: METRIC_NAMES }, start: { type: "string" }, end: { type: "string" } },
      required: ["metric", "start", "end"],
      additionalProperties: false,
    },
  },
  {
    name: "get_recent_deploys",
    description: "Deploys that finished within the last `hours` hours. Empty service string means all services.",
    strict: true,
    input_schema: {
      type: "object",
      properties: { service: { type: "string" }, hours: { type: "integer" } },
      required: ["service", "hours"],
      additionalProperties: false,
    },
  },
  {
    name: "get_runbook",
    description: "Find operational runbooks relevant to a topic (keywords such as error codes or symptoms).",
    strict: true,
    input_schema: {
      type: "object",
      properties: { topic: { type: "string" } },
      required: ["topic"],
      additionalProperties: false,
    },
  },
];

class TriageTools {
  records: LogRecord[];
  metrics: Json;
  deploys: Json[];
  runbooks: Json[];

  constructor(records: LogRecord[], metrics: Json, deploys: Json[], runbooks: Json[]) {
    this.records = records;
    this.metrics = metrics;
    this.deploys = deploys;
    this.runbooks = runbooks;
  }

  searchLogs(a: Json): Json {
    const q = String(a.query).toLowerCase();
    const hits = this.records.filter(
      (r) => r.raw.toLowerCase().includes(q) && (a.level === "ANY" || r.level === a.level) &&
        (!a.since || r.time.slice(0, 5) >= a.since) && (!a.until || r.time.slice(0, 5) <= a.until),
    );
    const limit = Math.max(1, Math.min(Number(a.limit), 20)); // cap output: tool results cost context tokens
    return { matches: hits.length, returned: Math.min(limit, hits.length), lines: hits.slice(0, limit).map((r) => r.raw) };
  }

  getMetrics(a: Json): Json {
    const metric = String(a.metric);
    const pts = (this.metrics.points as Json[]).filter((p) => a.start <= p.minute && p.minute <= a.end);
    const before = pts.filter((p) => p.minute < "13:59").map((p) => p[metric] as number);
    const after = pts.filter((p) => p.minute >= "14:00").map((p) => p[metric] as number);
    return {
      metric,
      points: pts.map((p) => [p.minute, p[metric]]),
      max: pts.length ? Math.max(...pts.map((p) => p[metric] as number)) : 0,
      mean_before_deploy: mean(before).toFixed(3),
      mean_after_deploy: mean(after).toFixed(3),
    };
  }

  getRecentDeploys(a: Json): Json {
    const now = epoch(NOW);
    const found = this.deploys.filter((d) => now - epoch(d.finished) <= Number(a.hours) * 3600 && (!a.service || d.service === a.service));
    return { now: NOW, deploys: found };
  }

  getRunbook(a: Json): Json {
    const t = String(a.topic).toLowerCase();
    const scored = this.runbooks.map((rb): [number, Json] => [(rb.keywords as string[]).filter((k) => t.includes(k.toLowerCase())).length, rb]);
    const ranked = scored.filter((s) => s[0] > 0).sort((x, y) => y[0] - x[0]).slice(0, 2);
    return { results: ranked.map(([score, rb]) => ({ id: rb.id, title: rb.title, url: rb.url, steps: rb.steps, score })) };
  }

  dispatch(name: string, args: Json): Json {
    switch (name) {
      case "search_logs": return this.searchLogs(args);
      case "get_metrics": return this.getMetrics(args);
      case "get_recent_deploys": return this.getRecentDeploys(args);
      case "get_runbook": return this.getRunbook(args);
      default: throw new Error(`unknown tool ${name}`);
    }
  }
}

function summarizeToolResult(name: string, r: Json): string {
  if (name === "search_logs") return `matches=${r.matches} returned=${r.returned}`;
  if (name === "get_metrics") return `${r.metric}: max=${r.max} mean_before=${r.mean_before_deploy} mean_after=${r.mean_after_deploy}`;
  if (name === "get_recent_deploys") return "deploys=" + (r.deploys as Json[]).map((d) => `${d.service}:${d.version}@${String(d.finished).slice(11, 16)}`).join(", ");
  if (name === "get_runbook") return "results=" + (r.results as Json[]).map((x) => `${x.id}(score ${x.score})`).join(", ");
  return "ok";
}

// ---------------------------------------------------------------------------
// Offline responder
// ---------------------------------------------------------------------------
const CANNED_REPORT = loadJson(path.join("offline", "incident_report.json"));
const LONG_POSTMORTEM = (
  "INC-2291 postmortem summary. At 13:59:40 UTC deploy v2.14.0 of the summarizer service completed. The release " +
  "changed three things at once: the prompt template, the worker concurrency and the retry policy. Within two minutes " +
  "the Claude API began returning 429 rate_limit_error responses because uncached input tokens per minute exceeded " +
  "the organisation limit. The new retry policy retried every 500 ms regardless of retry-after, which multiplied " +
  "traffic and later amplified 529 overloaded_error responses. Customer impact lasted until the rollback. "
).repeat(2);

function triageResponder(req: MockRequest): ResponderResult {
  const turn = req.messages.filter((m) => m.role === "assistant").length;
  if (turn === 0) {
    // broad first: what changed, and what do the golden signals say?
    return [
      toolUseBlock("get_recent_deploys", { service: "summarizer", hours: 24 }),
      toolUseBlock("get_metrics", { metric: "claude_429", start: "13:50", end: "14:14" }),
      toolUseBlock("get_metrics", { metric: "cache_hit_ratio", start: "13:50", end: "14:14" }),
    ];
  }
  if (turn === 1) {
    // then narrow: confirm the hypothesis in the logs
    return [
      toolUseBlock("search_logs", { query: "status=429", level: "ERROR", since: "13:55", until: "14:15", limit: 3 }),
      toolUseBlock("search_logs", { query: "retrying", level: "WARN", since: "", until: "", limit: 3 }),
      toolUseBlock("get_metrics", { metric: "input_tokens_per_min", start: "13:50", end: "14:14" }),
    ];
  }
  if (turn === 2) {
    return [
      toolUseBlock("search_logs", { query: "status=529", level: "ERROR", since: "", until: "", limit: 3 }),
      toolUseBlock("get_runbook", { topic: "429 rate limit retry-after concurrency" }),
    ];
  }
  return JSON.stringify(CANNED_REPORT);
}

function responder(req: MockRequest): ResponderResult {
  if (req.tools.length) return triageResponder(req);
  if (req.lastUserText.includes("detailed postmortem")) return LONG_POSTMORTEM;
  return "Q3 saw two incidents, both rate-limit related; caching and throttling fixes cut error rates by 90%.";
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------
function stepLogAnalysis(records: LogRecord[], unparsed: number, metrics: Json, deploys: Json[]): Facts {
  section("1. Deterministic log analysis (ground truth before any model call)");
  const levels = ["INFO", "WARN", "ERROR"].map((lvl) => `${lvl}=${records.filter((r) => r.level === lvl).length}`);
  console.log(`  parsed ${records.length} lines (${unparsed} unparsed): ${levels.join(", ")}`);
  console.log("\n  error/warning clusters (volatile fields like request_id, latency, attempt removed):");
  for (const c of clusterErrors(records)) {
    console.log(`    ${String(c.count).padStart(3)}x ${c.level.padEnd(5)} ${c.first}-${c.last}  ${c.signature.slice(0, 92)}`);
  }

  const facts = correlate(records, metrics, deploys);
  console.log("\n  per-minute timeline from metrics.json  (429 bar | 529 bar | error rate | cache hit | in-flight)");
  const pts = metrics.points as Json[];
  const max429 = Math.max(...pts.map((p) => p.claude_429));
  const max529 = Math.max(...pts.map((p) => p.claude_529));
  const deployMinute = facts.deployFinished.slice(0, 5);
  for (const p of pts) {
    const marker = p.minute === deployMinute ? `  <- deploy ${facts.deployVersion} finished ${facts.deployFinished}` : "";
    console.log(
      `    ${p.minute}  ${bar(p.claude_429, max429, 20)} ${String(p.claude_429).padStart(3)} | ${bar(p.claude_529, max529, 10)} ${String(p.claude_529).padStart(2)} | ` +
        `err ${(p.error_rate * 100).toFixed(1).padStart(5)}% | cache ${(p.cache_hit_ratio * 100).toFixed(0).padStart(3)}% | ${String(p.in_flight).padStart(2)}${marker}`,
    );
  }

  console.log("\n  correlation with the deploy:");
  const rows: [string, string][] = [
    ["suspect deploy", `${facts.deployVersion} (finished ${facts.deployFinished}); web-v5.2.0 ruled out (other service, 3h earlier)`],
    ["first 429 / first 529", `${facts.first429} (${facts.secondsDeployToFirst429} s after deploy) / ${facts.first529}`],
    ["cache hit ratio (logs)", `${facts.cacheRatioBefore.toFixed(2)} -> ${facts.cacheRatioAfter.toFixed(2)}`],
    ["uncached input tokens per call", `${facts.uncachedInputPerCallBefore.toFixed(0)} -> ${facts.uncachedInputPerCallAfter.toFixed(0)}`],
    ["input tokens/min vs limit", `peak ${facts.itpmPeak} vs ${facts.itpmLimit} (${((facts.itpmPeak / facts.itpmLimit) * 100).toFixed(0)}% of limit)`],
    ["in-flight requests", `${facts.inFlightBefore.toFixed(0)} -> ${facts.inFlightAfter.toFixed(0)}`],
    ["error rate (metrics)", `${(facts.errorRateBefore * 100).toFixed(2)}% -> ${(facts.errorRateAfter * 100).toFixed(1)}%`],
    ["p95 latency (metrics)", `${facts.p95BeforeMs.toFixed(0)} ms -> ${facts.p95AfterMs.toFixed(0)} ms`],
    ["retries ignoring retry-after", `${facts.retriesIgnoringRetryAfter} (retry-after up to ${facts.maxRetryAfterS} s, retried after 500 ms)`],
  ];
  for (const [k, v] of rows) console.log(`    ${k.padEnd(32)} ${v}`);
  return facts;
}

interface AgentStats {
  turns: number;
  toolCalls: number;
  input: number;
  output: number;
  cost: number;
}

async function stepTriage(client: Anthropic, tools: TriageTools, schema: Json): Promise<[IncidentReport | null, AgentStats]> {
  section("2. Triage agent: manual tool-use loop (Opus 5.5, structured final answer)");
  const system =
    "You are an SRE incident triage agent for the summarizer service. Investigate with the tools before " +
    "concluding: check recent deploys, metrics and logs, then find the matching runbook. Every claim must be " +
    "backed by tool output with concrete times and numbers. When you are done, answer with the incident report.";
  const messages: Anthropic.MessageParam[] = [{
    role: "user",
    content: "PagerDuty: SummarizerErrorRateHigh fired at 14:08 UTC (error_rate 0.31). Find the root cause, the " +
      `remediation and the runbook. Current time ${NOW}.`,
  }];
  const stats: AgentStats = { turns: 0, toolCalls: 0, input: 0, output: 0, cost: 0 };
  for (let turn = 1; turn <= MAX_TURNS; turn++) {
    const resp = await client.messages.create({
      model: MODEL_OPUS,
      max_tokens: 8000,
      thinking: { type: "adaptive" },
      output_config: { effort: "high", format: { type: "json_schema", schema } },
      system,
      tools: TOOLS,
      messages,
    });
    stats.turns = turn;
    stats.input += resp.usage.input_tokens;
    stats.output += resp.usage.output_tokens;
    stats.cost += costUsd(resp.model, resp.usage);
    if (resp.stop_reason === "tool_use") {
      const toolUses = resp.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use");
      messages.push({ role: "assistant", content: resp.content });
      const results: Anthropic.ToolResultBlockParam[] = [];
      for (const tu of toolUses) {
        stats.toolCalls++;
        try {
          const out = tools.dispatch(tu.name, tu.input as Json);
          results.push({ type: "tool_result", tool_use_id: tu.id, content: JSON.stringify(out) });
          console.log(`  turn ${turn}  ${tu.name.padEnd(18)} ${JSON.stringify(tu.input).slice(0, 62).padEnd(62)} => ${summarizeToolResult(tu.name, out)}`);
        } catch (e) {
          // a failing tool becomes an is_error result the model can react to
          results.push({ type: "tool_result", tool_use_id: tu.id, content: `tool error: ${String(e)}`, is_error: true });
          console.log(`  turn ${turn}  ${tu.name.padEnd(18)} ERROR ${String(e)}`);
        }
      }
      messages.push({ role: "user", content: results }); // all results for this turn in ONE user message
      continue;
    }
    console.log(`  turn ${turn}  final answer: stop_reason=${resp.stop_reason} -> ${diagnoseResponse(resp)}`);
    if (resp.stop_reason !== "end_turn") return [null, stats];
    const report = IncidentReportSchema.parse(JSON.parse(textOf(resp)));
    console.log(`  agent: ${stats.turns} turns, ${stats.toolCalls} tool calls, in=${stats.input} out=${stats.output} cost=$${stats.cost.toFixed(5)}`);
    return [report, stats];
  }
  console.log(`  stopped after MAX_TURNS=${MAX_TURNS} without an answer`);
  return [null, stats];
}

/** Trust, but verify: every checkable claim is compared with the deterministic facts. */
function groundingChecks(report: IncidentReport, facts: Facts, records: LogRecord[], deploys: Json[], runbooks: Json[]): [string, boolean, string][] {
  const versions = new Set(deploys.map((d) => d.version));
  const m = /v\d+\.\d+\.\d+/.exec(report.trigger);
  const logTimes = new Set(records.map((r) => r.time));
  const times = report.timeline.map((t) => t.time);
  const rb = runbooks.find((r) => r.id === report.runbook_id);
  const evidence = report.evidence.join(" ");
  const delay = `${facts.secondsDeployToFirst429} s`;
  return [
    ["trigger names a real deploy", Boolean(m && versions.has(m[0])), m ? m[0] : "none"],
    ["runbook id exists and URL matches", Boolean(rb && rb.url === report.runbook_url), report.runbook_id],
    ["timeline times exist in the log", times.every((t) => logTimes.has(t)), `${times.filter((t) => logTimes.has(t)).length}/${times.length}`],
    ["timeline is chronological", times.every((t, i) => i === 0 || times[i - 1]! <= t), `${times.length} entries`],
    ["evidence cites the first 429 time", evidence.includes(facts.first429), facts.first429],
    ["evidence cites deploy -> 429 delay", evidence.includes(delay), delay],
    ["confidence within [0, 1]", report.confidence >= 0 && report.confidence <= 1, report.confidence.toFixed(2)],
  ];
}

async function stepResilience(client: Anthropic): Promise<void> {
  section("3. Resilient client: typed errors, retry-after, jitter, fallback, breaker");
  console.log("  Errors are SIMULATED by FaultInjector using the SDK's own exception classes (the offline mock only");
  console.log("  returns 200/404). Sleeps use a virtual clock. Primary model: Opus 5.5, fallback: Sonnet 5.5.");
  const injector = new FaultInjector();
  const clock = new VirtualClock();
  const rc = new ResilientClaude(client, [MODEL_OPUS, MODEL_SONNET], injector, clock, mulberry32(7));
  const params: CreateParams = {
    max_tokens: 600,
    system: "You are the summarization service. Summarise the document in one sentence.",
    messages: [{ role: "user", content: "Document: Q3 reliability review (two incidents, both rate-limit related)." }],
  };
  const f429 = (ra: number): Fault => ({ kind: "status", status: 429, errorType: "rate_limit_error", message: "input tokens per minute rate limit exceeded", retryAfter: ra });
  const f529: Fault = { kind: "status", status: 529, errorType: "overloaded_error", message: "Overloaded" };
  const scenarios: [string, Record<string, (Fault | null)[]>, number][] = [
    ["S1 429 + retry-after 3s: wait exactly what the API asks", { [MODEL_OPUS]: [f429(3), null] }, 0],
    ["S2 529 twice: exponential backoff with full jitter", { [MODEL_OPUS]: [f529, f529, null] }, 0],
    ["S3 sustained 529: breaker opens after 3 failures -> fallback", { [MODEL_OPUS]: [f529, f529, f529] }, 0],
    ["S4 breaker OPEN: fail fast, no request sent to Opus", {}, 5],
    ["S5 cooldown elapsed: half-open probe closes the breaker", {}, 30],
    ["S6 400 invalid_request_error: never retry", { [MODEL_OPUS]: [{ kind: "status", status: 400, errorType: "invalid_request_error", message: "temperature: not supported for this model" }] }, 0],
    ["S7 429 + retry-after 120s exceeds the 30s budget -> fallback", { [MODEL_OPUS]: [f429(120)] }, 0],
    ["S8 network timeout: retry with backoff", { [MODEL_OPUS]: [{ kind: "timeout" }, null] }, 0],
  ];
  for (const [name, plan, advance] of scenarios) {
    console.log(`\n  ${name}`);
    if (advance) {
      clock.sleep(advance);
      console.log(`    (virtual clock advanced ${advance.toFixed(0)}s)`);
    }
    for (const [model, faults] of Object.entries(plan)) injector.script(model, faults);
    let outcome: string;
    try {
      const msg = await rc.create(params);
      outcome = `OK, served by ${msg.model}`;
    } catch (e) {
      if (e instanceof Anthropic.BadRequestError) outcome = `raised BadRequestError to the caller (request_id=${e.requestID})`;
      else if (e instanceof AllModelsFailedError) outcome = "raised AllModelsFailedError";
      else throw e;
    }
    const states = [...rc.breakers.entries()].map(([m, b]) => `${m.split("-")[1]}=${b.state}`).join(", ");
    console.log(`    => ${outcome} | breakers: ${states}`);
  }
  const sdkTries = 1 + 2;
  const callerTries = 3;
  console.log(`\n  wrapper attempts: ${rc.sent} (${injector.injected} simulated failures) across ${scenarios.length} scenarios; virtual time elapsed ${clock.now.toFixed(2)}s`);
  console.log(`  retry-storm math: SDK default (${sdkTries} tries) x wrapper (${rc.maxAttempts}) x caller (${callerTries}) = ` +
    `${sdkTries * rc.maxAttempts * callerTries} requests per user action -> keep ONE retry layer`);
}

function staticPrompt(): string {
  const rules: string[] = [];
  for (let i = 1; i <= 40; i++) {
    const n = String(i).padStart(2, "0");
    rules.push(`Rule ${n}: keep summaries factual, cite the source section, avoid speculation, and use plain language suitable for an operations audience (style guide item ${i}).`);
  }
  return "You are the summarization service for operations reports.\n" + rules.join("\n");
}

async function stepChecklist(client: Anthropic): Promise<void> {
  section("4. Operational debugging checklist (as code)");
  console.log("  4a) max_tokens truncation");
  for (const maxTokens of [60, 1024]) {
    const msg = await client.messages.create({
      model: MODEL_HAIKU,
      max_tokens: maxTokens,
      system: "You are the summarization service.",
      messages: [{ role: "user", content: "Write a detailed postmortem summary of INC-2291." }],
    });
    console.log(`    max_tokens=${String(maxTokens).padEnd(5)} stop_reason=${String(msg.stop_reason).padEnd(10)} out=${String(msg.usage.output_tokens).padEnd(4)} -> ${diagnoseResponse(msg)}`);
  }

  console.log("\n  4b) cache_read_input_tokens == 0 investigation (the v2.14.0 prompt template bug)");
  const stat = staticPrompt();
  const doc = "Document: weekly operations report for the summarizer service.";
  const stamps = ["2026-10-08T14:03:11Z", "2026-10-08T14:03:12Z"];
  const groups: [string, [Anthropic.TextBlockParam[], string][]][] = [
    ["v7 timestamp first in system", stamps.map((ts) => [[{ type: "text", text: `Request time: ${ts}\n${stat}`, cache_control: { type: "ephemeral" } }], doc])],
    ["v6 static system, time in user", stamps.map((ts) => [[{ type: "text", text: stat, cache_control: { type: "ephemeral" } }], `Request time: ${ts}\n${doc}`])],
  ];
  for (const [label, calls] of groups) {
    const usages: Anthropic.Usage[] = [];
    let cost = 0;
    for (const [system, user] of calls) {
      const msg = await client.messages.create({ model: MODEL_SONNET, max_tokens: 300, system, messages: [{ role: "user", content: user }] });
      usages.push(msg.usage);
      cost += costUsd(msg.model, msg.usage);
      const u = msg.usage;
      console.log(`    ${label.padEnd(31)} in=${String(u.input_tokens).padEnd(4)} cache_write=${String(u.cache_creation_input_tokens ?? 0).padEnd(5)} cache_read=${u.cache_read_input_tokens ?? 0}`);
    }
    console.log(`    -> ${cacheCheck(usages)}  (2 calls: $${cost.toFixed(5)})`);
  }

  console.log("\n  4c) request linter: catch the 400s before sending");
  const user = { role: "user", content: "Summarise the incident." };
  const tool = { name: "get_runbook", description: "Find runbooks", input_schema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] } };
  const badRequests: [string, Json][] = [
    ["temperature on Opus 5.5", { model: MODEL_OPUS, max_tokens: 1024, temperature: 0.2, messages: [user] }],
    ["budget_tokens thinking", { model: MODEL_OPUS, max_tokens: 16000, thinking: { type: "enabled", budget_tokens: 8000 }, messages: [user] }],
    ["thinking disabled on Sonnet 5.5", { model: MODEL_SONNET, max_tokens: 1024, thinking: { type: "disabled" }, messages: [user] }],
    ["assistant prefill", { model: MODEL_SONNET, max_tokens: 512, messages: [user, { role: "assistant", content: "{" }] }],
    ["forced tool_choice on Sonnet 5.5", { model: MODEL_SONNET, max_tokens: 512, tools: [tool], tool_choice: { type: "any" }, messages: [user] }],
    ["broken tool_use/tool_result pairing", {
      model: MODEL_OPUS, max_tokens: 1024, tools: [tool], messages: [
        user,
        { role: "assistant", content: [{ type: "tool_use", id: "toolu_01", name: "get_runbook", input: { topic: "429" } }] },
        { role: "user", content: [{ type: "text", text: "here you go" }, { type: "tool_result", tool_use_id: "toolu_02", content: "[]" }] },
      ],
    }],
    ["dotted model id", { model: "claude-opus-5.5", max_tokens: 1024, messages: [user] }],
    ["correct request", { model: MODEL_OPUS, max_tokens: 4096, thinking: { type: "adaptive" }, output_config: { effort: "medium" }, messages: [user] }],
  ];
  for (const [label, params] of badRequests) {
    const issues = lintRequest(params);
    if (!issues.length) console.log(`    ${label.padEnd(36)} OK - no issues`);
    for (const [code, problem, fix] of issues) {
      console.log(`    ${label.padEnd(36)} ${code} ${problem}`);
      console.log(`    ${"".padEnd(36)}      fix: ${fix}`);
    }
  }

  console.log("\n  4d) checklist: symptom -> check -> fix");
  for (const [symptom, check, fix] of CHECKLIST) console.log(`    ${symptom.padEnd(30)} | ${check.padEnd(62)} | ${fix}`);
}

function renderReport(report: IncidentReport, checks: [string, boolean, string][], stats: AgentStats): string[] {
  const out = [
    `INCIDENT REPORT - ${report.title}`,
    `Severity: ${report.severity}   Confidence: ${report.confidence.toFixed(2)}   Runbook: ${report.runbook_id} ${report.runbook_url}`,
  ];
  for (const [label, text] of [["Impact", report.impact], ["Trigger", report.trigger], ["Root cause", report.root_cause]] as const) {
    out.push(`${label}:`, ...wrap(text));
  }
  for (const [label, items] of [["Contributing factors", report.contributing_factors], ["Evidence", report.evidence]] as const) {
    out.push(`${label}:`);
    for (const item of items) {
      const lines = wrap(item, 96, "      ");
      out.push("    - " + lines[0]!.trim(), ...lines.slice(1));
    }
  }
  out.push("Timeline (UTC):", ...report.timeline.map((t) => `    ${t.time}  ${t.event}`));
  for (const [label, items] of [["Immediate actions", report.immediate_actions], ["Follow-up actions", report.follow_up_actions]] as const) {
    out.push(`${label}:`);
    items.forEach((item, i) => {
      const lines = wrap(item, 96, "       ");
      out.push(`    ${i + 1}. ` + lines[0]!.trim(), ...lines.slice(1));
    });
  }
  const passed = checks.filter(([, ok]) => ok).length;
  out.push(`Verification: ${passed}/${checks.length} grounding checks passed; agent used ${stats.turns} turns, ${stats.toolCalls} tool calls, $${stats.cost.toFixed(5)}`);
  return out;
}

banner("D7.3 - Debugging and Operational Issue Resolution");
const [records, unparsed] = parseLog(readData("app.log.txt"));
const metrics = loadJson("metrics.json") as Json;
const deploys = loadJson("deploys.json") as Json[];
const runbooks = loadJson("runbooks.json") as Json[];
const schema = loadJson("incident_report.schema.json") as Json;
const client = getClient(responder);

const facts = stepLogAnalysis(records, unparsed, metrics, deploys);
const [report, stats] = await stepTriage(client, new TriageTools(records, metrics, deploys, runbooks), schema);
let checks: [string, boolean, string][] = [];
if (report) {
  checks = groundingChecks(report, facts, records, deploys, runbooks);
  console.log("\n  grounding checks (agent claims vs deterministic facts):");
  for (const [name, ok, detail] of checks) console.log(`    [${ok ? "PASS" : "FAIL"}] ${name.padEnd(36)} ${detail}`);
}
await stepResilience(client);
await stepChecklist(client);

section("5. Incident report");
if (!report) {
  console.log("  no report produced");
} else {
  const lines = renderReport(report, checks, stats);
  for (const line of lines) console.log(`  ${line}`);
  mkdirSync(OUT, { recursive: true });
  writeFileSync(path.join(OUT, "incident_report.json"), JSON.stringify(report, null, 2), "utf8");
  writeFileSync(path.join(OUT, "incident_report.md"), "```\n" + lines.join("\n") + "\n```\n", "utf8");
  console.log("\n  wrote labs_output/incident_report.json and labs_output/incident_report.md");
}
