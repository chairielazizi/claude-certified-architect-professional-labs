/**
 * D1.2 - End-to-End Architecture and Feedback Loops
 *
 * A support-ticket pipeline with explicit, typed stage contracts:
 *
 *   INPUT       validate -> normalise -> PII scrub -> dedupe            (deterministic code)
 *   PROCESSING  Claude classify + extract (structured outputs)          (LLM, with retry policy)
 *   OUTPUT      schema + business validation -> route -> deliver        (deterministic code)
 *   FEEDBACK    human corrections -> metrics -> few-shot + eval set -> prompt v2 -> gated re-run
 *
 * Every stage can fail independently; failures are retried when transient and parked in a
 * dead-letter queue (DLQ) when not, so one bad ticket never blocks the batch.
 */
import Anthropic from "@anthropic-ai/sdk";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  LATENCY_SCALE,
  MODEL_HAIKU,
  banner,
  getClient,
  isLive,
  section,
  sleep,
  textOf,
  type MockRequest,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");

// ---------------------------------------------------------------------------
// Stage contracts. Each stage accepts exactly one type and emits exactly one type,
// so stages can be tested, scaled and replaced independently.
// ---------------------------------------------------------------------------
const CATEGORIES = ["billing", "access", "bug", "how_to", "privacy", "account", "sales"] as const;
const PRIORITIES = ["P1", "P2", "P3", "P4"] as const;

interface RawTicket {
  // INPUT contract (as received from email/chat/web)
  id: string;
  channel: string;
  customer_email: string;
  subject: string;
  body: string;
}

interface NormalizedTicket {
  // INPUT -> PROCESSING contract
  id: string;
  channel: string;
  text: string; // scrubbed: safe to send to the model
  piiFound: string[];
  fingerprint: string;
}

// PROCESSING -> OUTPUT contract (validated in code)
const Classification = z.object({
  category: z.enum(CATEGORIES),
  priority: z.enum(PRIORITIES),
  product_area: z.string(),
  summary: z.string(),
  key_facts: z.array(z.string()),
  // Numeric ranges are NOT enforceable by structured outputs, so the range check lives here.
  confidence: z.number().min(0).max(1),
});
type Classification = z.infer<typeof Classification>;

interface RoutedTicket {
  // OUTPUT contract (what downstream systems receive)
  ticketId: string;
  classification: Classification;
  queue: string;
  slaHours: number;
  status: string;
}

class DeadLetter {
  ticketId: string;
  stage: string;
  reason: string;
  attempts: number;
  payload: unknown;
  constructor(ticketId: string, stage: string, reason: string, attempts: number, payload: unknown = null) {
    this.ticketId = ticketId;
    this.stage = stage;
    this.reason = reason;
    this.attempts = attempts;
    this.payload = payload;
  }
}

// The JSON schema sent as output_config.format. Hand-written on purpose: it IS the contract.
const CLASSIFICATION_SCHEMA = {
  type: "object",
  properties: {
    category: { type: "string", enum: [...CATEGORIES] },
    priority: { type: "string", enum: [...PRIORITIES] },
    product_area: { type: "string" },
    summary: { type: "string" },
    key_facts: { type: "array", items: { type: "string" } },
    confidence: { type: "number" },
  },
  required: ["category", "priority", "product_area", "summary", "key_facts", "confidence"],
  additionalProperties: false,
};

const ALLOWED_CHANNELS = new Set(["email", "chat", "web"]);
const QUEUES: Record<Classification["category"], string> = {
  billing: "billing-ops",
  access: "identity-team",
  bug: "engineering-triage",
  how_to: "tier1-support",
  privacy: "privacy-office",
  account: "tier1-support",
  sales: "sales-desk",
};
const SLA_HOURS: Record<Classification["priority"], number> = { P1: 1, P2: 4, P3: 24, P4: 72 };
const CONFIDENCE_FLOOR = 0.7;
const MAX_ATTEMPTS = 3;
const INITIAL_MAX_TOKENS = 200; // sized from typical output length; outliers trigger an adaptive retry

// ---------------------------------------------------------------------------
// INPUT stage (deterministic)
// ---------------------------------------------------------------------------
const CARD_RE = /\b(?:\d[ -]?){12,18}\d\b/g;
const EMAIL_RE = /[\w.+-]+@[\w-]+(?:\.[\w-]+)+/g;
const PHONE_RE = /\+?\d{1,3}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b/g;
const QUOTED_REPLY_RE = /\n\s*On .{0,120}wrote:.*/s;
const has = (re: RegExp, s: string): boolean => new RegExp(re.source, re.flags.replace("g", "")).test(s);
const collapse = (s: string): string => s.split(/\s+/).filter(Boolean).join(" ");

function luhnOk(digits: string): boolean {
  let total = 0;
  const parity = digits.length % 2;
  for (let i = 0; i < digits.length; i++) {
    let d = Number(digits[i]);
    if (i % 2 === parity) d = d * 2 > 9 ? d * 2 - 9 : d * 2;
    total += d;
  }
  return total % 10 === 0;
}

function validateRaw(t: RawTicket): string[] {
  const errors: string[] = [];
  if (!t.body.trim()) errors.push("empty body");
  if (t.body.length > 20_000) errors.push("body exceeds 20k chars");
  if (!t.customer_email.trim()) errors.push("missing customer_email (cannot reply)");
  if (!ALLOWED_CHANNELS.has(t.channel.trim().toLowerCase())) errors.push(`unknown channel '${t.channel}'`);
  return errors;
}

function scrubPii(input: string): [string, string[]] {
  const found: string[] = [];
  let text = input.replace(CARD_RE, (m) => {
    const digits = m.replace(/\D/g, "");
    if (digits.length >= 13 && digits.length <= 19 && luhnOk(digits)) {
      // Luhn avoids redacting order numbers
      found.push("CARD");
      return "[CARD]";
    }
    return m;
  });
  if (has(EMAIL_RE, text)) {
    found.push("EMAIL");
    text = text.replace(EMAIL_RE, "[EMAIL]");
  }
  if (has(PHONE_RE, text)) {
    found.push("PHONE");
    text = text.replace(PHONE_RE, "[PHONE]");
  }
  return [text, found];
}

function normalize(t: RawTicket): NormalizedTicket {
  let body = t.body.replace(QUOTED_REPLY_RE, ""); // drop quoted email history
  body = body
    .trim()
    .split(/\r?\n/)
    .map((line) => collapse(line))
    .join("\n");
  const [text, pii] = scrubPii(`Subject: ${t.subject.trim()}\n\n${body}`);
  // Fingerprint on whitespace/case-insensitive text so re-sent tickets collapse to one.
  const fingerprint = createHash("sha256").update(collapse(text.toLowerCase())).digest("hex").slice(0, 12);
  return { id: t.id, channel: t.channel.trim().toLowerCase(), text, piiFound: pii, fingerprint };
}

// ---------------------------------------------------------------------------
// Prompt registry - prompts are versioned artefacts, not string literals in code paths
// ---------------------------------------------------------------------------
const BASE_SYSTEM = `You classify customer-support tickets for a B2B SaaS company.
Categories: billing (charges, invoices, refunds, payments), access (login, SSO, permissions),
bug (product defects, errors), how_to (usage questions), privacy (data-protection requests),
account (profile, users, plan settings without a charge problem), sales (pricing, discounts, quotes).
Priority: P1 = many users blocked or security incident; P2 = single customer blocked, money at risk or a
legal deadline; P3 = degraded but workable; P4 = question or request.
Return the JSON object only. confidence is your probability (0-1) that category AND priority are right.`;

interface Example {
  text: string;
  labels: { category: string; priority: string };
  why: string;
}
interface PromptVersion {
  version: string;
  system: string;
  examples: Example[];
}

function buildPrompt(version: string, examples: Example[]): PromptVersion {
  if (!examples.length) return { version, system: BASE_SYSTEM, examples };
  const blocks = examples.map(
    (ex) =>
      "<example>\n" +
      `<ticket>${ex.text}</ticket>\n` +
      `<labels>{"category": "${ex.labels.category}", "priority": "${ex.labels.priority}"}</labels>\n` +
      `<why>${ex.why}</why>\n` +
      "</example>",
  );
  const system = BASE_SYSTEM + "\n\nLearn from these reviewed tickets:\n<examples>\n" + blocks.join("\n") + "\n</examples>";
  return { version, system, examples };
}

// ---------------------------------------------------------------------------
// PROCESSING stage (Claude) with a retry policy and DLQ hand-off
// ---------------------------------------------------------------------------
function isTransient(err: unknown): boolean {
  if (err instanceof Anthropic.APIConnectionError) return true; // includes connection timeouts
  if (err instanceof Anthropic.APIError && typeof err.status === "number")
    return [408, 409, 429].includes(err.status) || err.status >= 500;
  return false;
}

interface ProcessStats {
  retries: number;
}

async function processTicket(
  client: Anthropic,
  t: NormalizedTicket,
  prompt: PromptVersion,
  stats: ProcessStats,
  verbose = true,
): Promise<Classification | DeadLetter> {
  let maxTokens = INITIAL_MAX_TOKENS;
  let lastError = "unknown";
  const note = (msg: string) => {
    if (verbose) console.log(`    ${t.id} ${msg}`);
  };

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    if (attempt > 1) stats.retries += 1;
    let msg: Anthropic.Message;
    try {
      msg = await client.messages.create({
        model: MODEL_HAIKU,
        max_tokens: maxTokens,
        system: prompt.system,
        output_config: { effort: "low", format: { type: "json_schema", schema: CLASSIFICATION_SCHEMA } },
        messages: [{ role: "user", content: `Ticket ${t.id} (channel: ${t.channel})\n\n${t.text}` }],
      });
    } catch (err) {
      if (!(err instanceof Anthropic.APIError)) throw err;
      if (!isTransient(err)) return new DeadLetter(t.id, "processing", `non-retryable ${err.constructor.name}`, attempt, t);
      lastError = err.constructor.name;
      const backoffMs = 500 * 2 ** (attempt - 1); // add jitter in production
      note(`attempt ${attempt}: ${lastError} (transient) -> backoff ${backoffMs} ms`);
      await sleep(backoffMs * (isLive() ? 1 : LATENCY_SCALE * 0.1));
      continue;
    }
    // Always inspect stop_reason before trusting content.
    if (msg.stop_reason === "max_tokens") {
      lastError = `truncated at max_tokens=${maxTokens}`;
      note(`attempt ${attempt}: stop_reason=max_tokens (${maxTokens}) -> retry with ${maxTokens * 2}`);
      maxTokens *= 2;
      continue;
    }
    if (msg.stop_reason === "refusal") return new DeadLetter(t.id, "processing", "model refusal - needs human handling", attempt, t);
    let parsed: ReturnType<typeof Classification.safeParse>;
    try {
      parsed = Classification.safeParse(JSON.parse(textOf(msg)));
    } catch {
      lastError = "contract violation: invalid JSON";
      note(`attempt ${attempt}: ${lastError} -> retry`);
      continue;
    }
    if (parsed.success) return parsed.data;
    lastError = `contract violation: ${parsed.error.issues[0]?.message}`;
    note(`attempt ${attempt}: ${lastError} -> retry`);
  }
  return new DeadLetter(t.id, "processing", `retries exhausted (${lastError})`, MAX_ATTEMPTS, t);
}

// ---------------------------------------------------------------------------
// OUTPUT stage (deterministic): business validation -> routing -> idempotent delivery
// ---------------------------------------------------------------------------
class Outbox {
  delivered = new Map<string, RoutedTicket>();
  deliver(r: RoutedTicket): string {
    if (this.delivered.has(r.ticketId)) return "duplicate-suppressed"; // idempotency key = ticket id
    this.delivered.set(r.ticketId, r);
    return "delivered";
  }
}

function route(t: NormalizedTicket, c: Classification, outbox: Outbox): RoutedTicket | DeadLetter {
  const leaks = (
    [
      ["EMAIL", EMAIL_RE],
      ["PHONE", PHONE_RE],
    ] as const
  )
    .filter(([, rx]) => has(rx, c.summary))
    .map(([name]) => name);
  if (leaks.length) return new DeadLetter(t.id, "output", `PII in model summary: ${leaks.join(", ")}`, 1, c);
  const queue = c.confidence < CONFIDENCE_FLOOR ? "human-review" : QUEUES[c.category];
  const routed: RoutedTicket = { ticketId: t.id, classification: c, queue, slaHours: SLA_HOURS[c.priority], status: "pending" };
  routed.status = outbox.deliver(routed);
  return routed;
}

// ---------------------------------------------------------------------------
// FEEDBACK: corrections, metrics, promotion
// ---------------------------------------------------------------------------
interface Gold {
  category: string;
  priority: string;
  review_note?: string;
}
interface Correction {
  ticketId: string;
  field: "category" | "priority";
  predicted: string;
  corrected: string;
  note: string;
}
interface Metrics {
  n: number;
  categoryAcc: number;
  priorityAcc: number;
  exactMatch: number;
  lowConfRate: number;
  confWhenRight: number;
  confWhenWrong: number;
}

function computeMetrics(preds: Map<string, Classification>, gold: Map<string, Gold>): Metrics {
  const ids = [...preds.keys()].filter((i) => gold.has(i));
  const cat = ids.map((i) => preds.get(i)!.category === gold.get(i)!.category);
  const pri = ids.map((i) => preds.get(i)!.priority === gold.get(i)!.priority);
  const exact = cat.map((c, k) => c && pri[k]!);
  const right = ids.filter((_, k) => exact[k]).map((i) => preds.get(i)!.confidence);
  const wrong = ids.filter((_, k) => !exact[k]).map((i) => preds.get(i)!.confidence);
  const n = ids.length;
  const avg = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
  const count = (xs: boolean[]) => xs.filter(Boolean).length;
  return {
    n,
    categoryAcc: count(cat) / n,
    priorityAcc: count(pri) / n,
    exactMatch: count(exact) / n,
    lowConfRate: ids.filter((i) => preds.get(i)!.confidence < CONFIDENCE_FLOOR).length / n,
    confWhenRight: avg(right),
    confWhenWrong: avg(wrong),
  };
}

const pct = (x: number, width = 0): string => `${(x * 100).toFixed(1)}%`.padStart(width);

function printMetrics(label: string, m: Metrics): void {
  const wrong = m.exactMatch < 1 ? m.confWhenWrong.toFixed(2) : "n/a";
  console.log(
    `  ${label.padEnd(4)} n=${m.n}  category=${pct(m.categoryAcc, 6)}  priority=${pct(m.priorityAcc, 6)}  ` +
      `exact=${pct(m.exactMatch, 6)}  low-conf=${pct(m.lowConfRate, 5)}  ` +
      `conf(right)=${m.confWhenRight.toFixed(2)}  conf(wrong)=${wrong}`,
  );
}

// ---------------------------------------------------------------------------
// Offline responder - canned classifications, plus simulated failures
// ---------------------------------------------------------------------------
const MOCK: { outputs: Record<string, Record<string, unknown>>; v1_mistakes: Record<string, Record<string, unknown>> } =
  JSON.parse(readFileSync(path.join(DATA, "mock_classifications.json"), "utf-8"));

/** Stateful so failures are deterministic: T-1009 blips once; T-1010 hits a 3-attempt outage. */
function makeTicketResponder() {
  const attempts = new Map<string, number>();
  return (req: MockRequest): string => {
    const tid = /Ticket (T-\d+)/.exec(req.lastUserText)?.[1] ?? "";
    const n = (attempts.get(tid) ?? 0) + 1;
    attempts.set(tid, n);
    if (tid === "T-1009" && n === 1) throw new Error("simulated network blip"); // SDK surfaces APIConnectionError
    if (tid === "T-1010" && n <= 3) throw new Error("simulated upstream outage");
    const out = { ...MOCK.outputs[tid] };
    if (!req.systemText.includes("<examples>") && MOCK.v1_mistakes[tid]) Object.assign(out, MOCK.v1_mistakes[tid]); // zero-shot errors
    return JSON.stringify(out);
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
banner("D1.2 - End-to-End Architecture and Feedback Loops");
const client = getClient(makeTicketResponder());
const rawRows: (RawTicket & { gold: Gold | null })[] = JSON.parse(readFileSync(path.join(DATA, "tickets.json"), "utf-8"));
const raw: RawTicket[] = rawRows.map(({ id, channel, customer_email, subject, body }) => ({ id, channel, customer_email, subject, body }));
const gold = new Map<string, Gold>(rawRows.filter((r) => r.gold).map((r) => [r.id, r.gold!]));
const dlq: DeadLetter[] = [];

section("0. Stage contracts");
console.log("  RawTicket -> [INPUT] -> NormalizedTicket -> [PROCESSING] -> Classification -> [OUTPUT] -> RoutedTicket");
console.log("  Any stage may emit DeadLetter(ticket_id, stage, reason, attempts) instead.");
console.log(`  Classification schema fields: ${Object.keys(CLASSIFICATION_SCHEMA.properties).join(", ")}`);

section("1. INPUT: validate -> normalise -> PII scrub -> dedupe");
const seen = new Map<string, string>();
const normalized: NormalizedTicket[] = [];
let duplicates = 0;
for (const t of raw) {
  const errors = validateRaw(t);
  if (errors.length) {
    dlq.push(new DeadLetter(t.id, "input", errors.join("; "), 1, t));
    console.log(`  ${t.id} REJECTED -> DLQ (${errors.join("; ")})`);
    continue;
  }
  const n = normalize(t);
  if (seen.has(n.fingerprint)) {
    console.log(`  ${t.id} DUPLICATE of ${seen.get(n.fingerprint)} (fingerprint ${n.fingerprint}) -> dropped`);
    duplicates += 1;
    continue;
  }
  seen.set(n.fingerprint, t.id);
  normalized.push(n);
  const pii = n.piiFound.join(",") || "none";
  console.log(`  ${t.id} ok  channel=${n.channel.padEnd(5)} pii=${pii.padEnd(6)} fp=${n.fingerprint}  "${n.text.split("\n")[0]!.slice(0, 44)}"`);
}
console.log(`  -> ${normalized.length} tickets to process; customer_email never leaves the input stage (data minimisation)`);
const sample = normalized.flatMap((n) => n.text.split("\n")).find((line) => line.includes("[CARD]")) ?? "";
console.log(`  scrubbed line sent to Claude: "${sample.slice(0, 100)}"`);

section("2. PROCESSING: Claude classify + extract (prompt v1, Haiku, structured outputs)");
const promptV1 = buildPrompt("v1", []);
const stats: ProcessStats = { retries: 0 };
const preds = new Map<string, Classification>();
const byId = new Map(normalized.map((n) => [n.id, n]));
for (const n of normalized) {
  const result = await processTicket(client, n, promptV1, stats);
  if (result instanceof DeadLetter) {
    dlq.push(result);
    console.log(`  ${n.id} -> DLQ (${result.reason})`);
  } else {
    preds.set(n.id, result);
    console.log(`  ${n.id} -> ${result.category.padEnd(8)} ${result.priority}  conf=${result.confidence.toFixed(2)}  ${result.product_area}`);
  }
}
console.log(`  retries used: ${stats.retries}  (SDK retries are disabled offline so the pipeline policy is visible)`);

section("3. OUTPUT: validate -> route -> deliver");
const outbox = new Outbox();
for (const [tid, c] of preds) {
  const r = route(byId.get(tid)!, c, outbox);
  if (r instanceof DeadLetter) {
    dlq.push(r);
    console.log(`  ${tid} -> DLQ (${r.reason})`);
    continue;
  }
  const flag = r.queue === "human-review" ? "  <- below confidence floor" : "";
  console.log(
    `  ${tid} ${c.category.padEnd(8)} ${c.priority} -> queue=${r.queue.padEnd(19)} SLA=${String(r.slaHours).padStart(2)}h  ${r.status}${flag}`,
  );
}
const again = route(byId.get("T-1001")!, preds.get("T-1001")!, outbox);
console.log(`  re-delivery of T-1001 (e.g. after a worker crash) -> ${again instanceof DeadLetter ? "?" : again.status}`);

section("4. Dead-letter queue: park, inspect, replay");
for (const d of dlq) console.log(`  [${d.stage.padEnd(10)}] ${d.ticketId}: ${d.reason} (attempts=${d.attempts})`);
console.log("  ...upstream dependency recovered; replaying processing-stage letters:");
for (const d of dlq.filter((x) => x.stage === "processing")) {
  const result = await processTicket(client, byId.get(d.ticketId)!, promptV1, stats);
  if (!(result instanceof DeadLetter)) {
    dlq.splice(dlq.indexOf(d), 1);
    preds.set(d.ticketId, result);
    const r = route(byId.get(d.ticketId)!, result, outbox);
    const where = r instanceof DeadLetter ? "queue=? " : `queue=${r.queue} ${r.status}`;
    console.log(`  ${d.ticketId} replayed -> ${result.category} ${result.priority} -> ${where}`);
  }
}
console.log(`  DLQ now holds ${dlq.length} item(s): ${dlq.map((d) => d.ticketId).join(", ")} (input errors need a human, not a retry)`);

section("5. FEEDBACK: human review captures corrections");
const corrections: Correction[] = [];
for (const [tid, c] of preds) {
  const g = gold.get(tid)!;
  for (const fld of ["category", "priority"] as const) {
    if (c[fld] !== g[fld]) corrections.push({ ticketId: tid, field: fld, predicted: c[fld], corrected: g[fld], note: g.review_note ?? "" });
  }
}
for (const corr of corrections)
  console.log(`  ${corr.ticketId} ${corr.field}: ${corr.predicted} -> ${corr.corrected}   note: ${corr.note}`);
console.log(`  ${corrections.length} correction(s) on ${preds.size} reviewed tickets`);

section("6. Metrics for prompt v1 (production batch)");
const mProd = computeMetrics(preds, gold);
printMetrics("v1", mProd);
console.log(
  `  human override rate = ${pct(1 - mProd.exactMatch)}; DLQ rate = ${dlq.length}/${raw.length - duplicates} ` +
    `unique tickets; pipeline retries = ${stats.retries}`,
);
console.log("  note: wrong answers still carried high self-reported confidence -> never gate on confidence alone");

section("7. Promote corrections -> few-shot examples + eval set; bump prompt version");
const correctedIds = [...new Set(corrections.map((c) => c.ticketId))].sort();
const examples: Example[] = correctedIds.map((i) => ({
  text: byId.get(i)!.text,
  labels: { category: gold.get(i)!.category, priority: gold.get(i)!.priority },
  why: gold.get(i)!.review_note ?? "",
}));
const evalSet = new Map<string, Gold>([...preds.keys()].map((tid) => [tid, gold.get(tid)!])); // every verified ticket = regression case
const promptV2 = buildPrompt("v2", examples);
console.log(`  few-shot examples added: ${correctedIds.join(", ")}`);
console.log(`  eval set size: ${evalSet.size} human-labelled tickets`);
console.log(
  `  prompt ${promptV1.version} -> ${promptV2.version}: system prompt ${promptV1.system.length} -> ` +
    `${promptV2.system.length} chars (examples block appended; base instructions unchanged)`,
);

section("8. Offline eval on the eval set: v1 vs v2 + promotion gate");
const results = new Map<string, Map<string, Classification>>();
for (const prompt of [promptV1, promptV2]) {
  const quiet: ProcessStats = { retries: 0 };
  const out = new Map<string, Classification>();
  for (const tid of evalSet.keys()) {
    const r = await processTicket(client, byId.get(tid)!, prompt, quiet, false); // eval runs never deliver
    if (!(r instanceof DeadLetter)) out.set(tid, r);
  }
  results.set(prompt.version, out);
}
const r1 = results.get("v1")!;
const r2 = results.get("v2")!;
const m1 = computeMetrics(r1, evalSet);
const m2 = computeMetrics(r2, evalSet);
printMetrics("v1", m1);
printMetrics("v2", m2);
const isRight = (c: Classification | undefined, g: Gold) => !!c && c.category === g.category && c.priority === g.priority;
const regressions = [...evalSet.keys()].filter((tid) => r1.has(tid) && r2.has(tid) && isRight(r1.get(tid), evalSet.get(tid)!) && !isRight(r2.get(tid), evalSet.get(tid)!));
const promote = m2.exactMatch >= m1.exactMatch && regressions.length === 0;
console.log(`  gate: exact-match ${pct(m1.exactMatch)} -> ${pct(m2.exactMatch)}, regressions=${regressions.join(", ") || "none"}`);
console.log(`  decision: ${promote ? "PROMOTE v2 to production" : "KEEP v1 (gate failed)"}`);

section("Summary");
console.log("  input -> processing -> output -> feedback is a LOOP: every human correction becomes");
console.log("  a few-shot example and a regression test, and a new prompt version ships only through the eval gate.");
