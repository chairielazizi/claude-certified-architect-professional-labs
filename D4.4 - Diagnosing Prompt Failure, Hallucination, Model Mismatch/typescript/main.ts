/**
 * D4.4 - Diagnosing Prompt Failure, Hallucination, Model Mismatch
 *
 * A diagnostic harness that walks SYMPTOM -> HYPOTHESES -> PROBES -> EVIDENCE -> ROOT CAUSE
 * -> FIX -> VERIFY for six broken Claude applications:
 *
 *   1. ambiguous prompt -> inconsistent output format
 *   2. hallucination from missing grounding
 *   3. stale RAG index after a document refresh   (exam sample question 3)
 *   4. model mismatch on multi-step reasoning
 *   5. truncation (stop_reason = max_tokens)
 *   6. context overload / lost-in-the-middle
 *
 * Run from the repo root:
 *   node "D4.4 - Diagnosing Prompt Failure, Hallucination, Model Mismatch/typescript/main.ts"
 */

import type Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { createHash } from "node:crypto";
import { z } from "zod";
import {
  MODEL_HAIKU,
  MODEL_OPUS,
  MODEL_SONNET,
  banner,
  costUsd,
  estimateTokens,
  getClient,
  section,
  textBlock,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

// ---------------------------------------------------------------------------
// Small deterministic helpers (same algorithms as the Python version)
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
function total(xs: Iterable<number>): number {
  let s = 0;
  for (const x of xs) s += x;
  return s;
}
const reply = (text: string): ResponderResult => ({ content: [textBlock(text)], usage: { output_tokens: estimateTokens(text) } });

const STOP = new Set(["a", "an", "the", "is", "are", "of", "to", "and", "or", "in", "on", "for", "it", "its", "does", "do", "how",
  "what", "can", "be", "by", "with", "into", "many", "much", "long", "which", "this", "that", "next"]);

const contentWords = (s: string): Set<string> =>
  new Set((s.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => !STOP.has(w) && (w.length > 2 || /^\d+$/.test(w))));
const overlap = (a: Set<string>, b: Set<string>): number => [...a].filter((w) => b.has(w)).length;
const textOf = (msg: Anthropic.Message): string => msg.content.map((b) => (b.type === "text" ? b.text : "")).join("");

// ---------------------------------------------------------------------------
// Diagnostic report helper
// ---------------------------------------------------------------------------
const SUMMARY: [string, string, string][] = [];

class Report {
  readonly title: string;
  constructor(n: number, title: string) {
    this.title = title;
    section(`Scenario ${n}: ${title}`);
  }
  line(label: string, text: string): void {
    console.log(`  ${label.padEnd(11)}: ${text}`);
  }
  hypotheses(items: string[]): void {
    this.line("HYPOTHESES", items.map((h, i) => `${i + 1}) ${h}`).join("; "));
  }
  probe(name: string, evidence: string): void {
    this.line("PROBE", name);
    this.line("  evidence", evidence);
  }
  conclude(category: string, short: string, root: string, fix: string, verify: string): void {
    this.line("ROOT CAUSE", root);
    this.line("FIX", fix);
    this.line("VERIFY", verify);
    SUMMARY.push([this.title, category, short]);
  }
}

// ---------------------------------------------------------------------------
// Scenario data
// ---------------------------------------------------------------------------
const TICKET = "The mobile app crashes every time I upload a photo larger than 10 MB on Android 15. Please fix this soon.";
const FORMAT_VARIANTS = [
  "The customer reports that the mobile app crashes when uploading photos larger than 10 MB on Android; they want a fix soon.",
  "- Issue: app crashes on large photo upload\n- Product: mobile app (Android)\n- Urgency: medium",
  '{"issue": "App crashes when uploading photos over 10 MB", "product": "mobile app", "urgency": "medium"}',
  "Customer says the Android app crashes on photo uploads over 10 MB and asks for a quick fix.",
  "1. Problem: crash on photo upload\n2. Platform: Android 15\n3. Priority: medium",
  "- Crash when uploading large photos\n- Android 15, mobile app\n- Needs fix soon",
  '```json\n{"issue": "Crash on large photo upload", "product": "mobile app", "urgency": "medium"}\n```',
  '{"issue": "Photo upload crash above 10 MB", "product": "mobile app", "urgency": "medium"}',
];
const STRUCTURED_SUMMARY = { issue: "App crashes when uploading photos larger than 10 MB on Android 15", product: "mobile app", urgency: "medium" };

const PRODUCT_DOCS: Record<string, string> = {
  "x200-warranty": "The X200 router has a 2-year limited warranty period that covers hardware defects.",
  "x200-security": "The X200 supports WPA3 and WPA2 encryption on its 2.4 GHz and 5 GHz bands.",
  "x200-setup": "Set up the X200 with the mobile app by scanning the QR code on the base of the router.",
};
const FABRICATED_WARRANTY = [
  "The X200 comes with a 3-year warranty.",
  "The X200 includes a 2-year warranty.",
  "The X200 is covered by a 5-year warranty.",
  "The X200 has a 3-year warranty.",
  "The X200 has a 4-year warranty on all parts.",
];

const POLICY_V1: Record<string, string> = {
  refunds: "Refunds are accepted within 30 days of delivery for unused items.",
  shipping: "Standard shipping costs 5 dollars and takes 3 to 5 business days.",
  "returns-label": "Return labels are emailed within one business day of approval.",
};
const POLICY_V2: Record<string, string> = {
  refunds: "Refunds are accepted within 14 days of delivery for unused items.",
  shipping: "Standard shipping costs 7 dollars and takes 3 to 5 business days.",
  "returns-label": "Return labels are emailed within one business day of approval.",
};

// Multi-step proration tasks: list of [days, monthly price] segments in a 30-day cycle, minus a credit.
interface BillingTask {
  id: string;
  kind: "simple" | "complex";
  text: string;
  answer: number;
  segments: [number, number][];
  credit: number;
}
const BILLING_TASKS: BillingTask[] = [
  { id: "S1", kind: "simple", text: "The Pro plan costs 30 dollars per month. What is the monthly price of the Pro plan?", answer: 30, segments: [], credit: 0 },
  { id: "S2", kind: "simple", text: "A plan costs 50 dollars for a 30-day cycle. What does one full cycle cost?", answer: 50, segments: [], credit: 0 },
  { id: "S3", kind: "simple", text: "A customer has a 5 dollar credit and a 20 dollar invoice. How much do they owe?", answer: 15, segments: [], credit: 0 },
  ...(
    [
      ["C1", [[10, 30], [10, 50], [10, 30]], 0],
      ["C2", [[15, 20], [15, 40]], 5],
      ["C3", [[5, 50], [25, 30]], 0],
      ["C4", [[12, 45], [18, 15]], 2.5],
      ["C5", [[7, 60], [8, 30], [15, 90]], 0],
    ] as [string, [number, number][], number][]
  ).map(([id, segments, credit]): BillingTask => {
    const parts = segments.map(([d, p]) => `the ${p} dollar/month plan for ${d} days`).join(", then ");
    const creditText = credit ? ` A ${fmt(credit, 2)} dollar goodwill credit applies.` : "";
    return {
      id,
      kind: "complex",
      segments,
      credit,
      text: `In a 30-day billing cycle the customer was on ${parts}. Prorate by day.${creditText} What is the net charge for the cycle?`,
      // The grader computes the truth with real code - never trust a model to grade arithmetic.
      answer: total(segments.map(([d, p]) => (d * p) / 30)) - credit,
    };
  }),
];
// Which tasks each tier gets right in this simulation (complex multi-step is where small models slip).
const CAPABILITY: Record<string, Set<string>> = {
  [MODEL_HAIKU]: new Set(["S1", "S2", "S3", "C2"]),
  [MODEL_SONNET]: new Set(["S1", "S2", "S3", "C1", "C2", "C3", "C4"]),
  [MODEL_OPUS]: new Set(["S1", "S2", "S3", "C1", "C2", "C3", "C4", "C5"]),
};

const INCIDENTS = [
  { id: "INC-1", title: "Login latency spike", events: 2 },
  { id: "INC-2", title: "EU region outage", events: 9 },
  { id: "INC-3", title: "Webhook retries", events: 3 },
  { id: "INC-4", title: "Billing double charge", events: 8 },
  { id: "INC-5", title: "Search index lag", events: 2 },
  { id: "INC-6", title: "Certificate expiry", events: 7 },
];

const TOPICS = [
  "travel booking", "expense receipts", "laptop refresh", "badge access", "parking", "meeting rooms", "printing",
  "visitor registration", "software requests", "password rotation", "remote work equipment", "mobile phones",
  "conference attendance", "training budget", "mentoring", "performance reviews", "office moves", "desk booking",
  "shipping samples", "vendor onboarding", "purchase orders", "contract signatures", "data retention", "email signatures",
  "social media", "press inquiries", "gifts and hospitality", "conflicts of interest", "open source use", "code reviews",
  "incident reporting", "on-call rotation", "change management", "backup restore", "cloud accounts", "encryption keys",
  "customer data access", "call recording", "accessibility", "first aid", "fire drills", "pets in the office",
  "bicycle storage", "kitchen use", "lost and found", "recycling", "lockers", "after-hours access",
  "translation services", "brand assets", "slide templates", "intranet posts", "team events", "charity matching",
  "referral bonuses", "relocation", "name badges", "open-plan noise", "plant care",
];
const NEEDLE =
  "Policy P-NEEDLE (vacation carry-over): Employees may carry over up to 7 unused vacation days into the next year. " +
  "Carrying over more than 7 days requires director approval.";
const VACATION_Q = "How many unused vacation days can an employee carry over into the next year?";

const fillerDoc = (i: number, topic: string): string =>
  `Policy P-${String(i).padStart(3, "0")} (${topic}): Requests about ${topic} go through the internal service portal. ` +
  `The owning team reviews ${topic} requests within five business days. ` +
  `Exceptions to the ${topic} rules require written approval from a department head. ` +
  `Questions about ${topic} go to the operations mailbox and records are kept for audit purposes.`;

// ---------------------------------------------------------------------------
// Offline responder: one "model" behaving like each broken app would
// ---------------------------------------------------------------------------
const COUNTERS = new Map<string, number>();
function bump(key: string): number {
  const n = COUNTERS.get(key) ?? 0;
  COUNTERS.set(key, n + 1);
  return n;
}

/** Faithful extraction: the context sentence (and its doc id) that overlaps the question most. */
function bestSentence(question: string, context: string): [string, string, number] {
  const q = contentWords(question);
  let best: [string, string, number] = ["", "", 0];
  for (const m of context.matchAll(/<doc id="([^"]+)"[^>]*>\n([\s\S]*?)\n<\/doc>/g)) {
    for (const sent of m[2]!.split(/(?<=\.)\s+/)) {
      const score = overlap(q, contentWords(sent));
      if (score > best[2]) best = [sent.trim(), m[1]!, score];
    }
  }
  return best;
}

function responder(req: MockRequest): ResponderResult {
  const system = req.systemText;
  const user = req.lastUserText;
  if (system.startsWith("Summarize")) {
    if (req.body.output_config?.format) return reply(JSON.stringify(STRUCTURED_SUMMARY));
    return reply(FORMAT_VARIANTS[bump("fmt") % FORMAT_VARIANTS.length]!);
  }
  if (system.startsWith("You are a helpful product expert")) {
    if (!user.includes("<doc")) {
      // no grounding: confident, unstable inventions
      if (user.includes("6 GHz")) return reply("Yes, the X200 supports 6 GHz Wi-Fi 6E for the fastest speeds.");
      return reply(FABRICATED_WARRANTY[bump("warranty") % FABRICATED_WARRANTY.length]!);
    }
    const [sent, doc, score] = bestSentence(user.split("Question:").pop()!, user);
    if (score < 3) return reply("I don't know based on the provided documents."); // weak overlap = the documents do not answer it
    return reply(`${sent} [${doc}]`);
  }
  if (system.startsWith("You answer questions about Acme policies")) {
    const [sent, doc] = bestSentence(user.split("Question:").pop()!, user);
    return reply(`${sent} [${doc}]`);
  }
  if (system.startsWith("You are a billing calculator")) {
    const task = BILLING_TASKS.find((t) => user.includes(t.text))!;
    // classic slip: charge the last plan for the full month and forget the proration
    const value = CAPABILITY[req.model]?.has(task.id) ? task.answer : task.segments[task.segments.length - 1]![1] - task.credit;
    return reply(`Net charge: ${fmt(value, 2)} dollars`);
  }
  if (system.startsWith("Write an incident report")) {
    const inc = INCIDENTS.find((i) => user.includes(i.id))!;
    const report = {
      title: inc.title,
      impact: `Customers affected by ${inc.title.toLowerCase()}`,
      timeline: Array.from({ length: inc.events }, (_, k) => `T+${k * 5}m: responders took step ${k + 1} to mitigate ${inc.title.toLowerCase()} and confirmed the result`),
      root_cause: "Under investigation",
      actions: ["Add alerting", "Write a runbook"],
    };
    const text = JSON.stringify(report);
    if (estimateTokens(text) > Number(req.body.max_tokens ?? 4096)) return text; // let the API truncate it: stop_reason becomes "max_tokens"
    return reply(text);
  }
  if (system.startsWith("You are the HR policy assistant")) {
    const docs = [...user.matchAll(/<doc id="([^"]+)">/g)].map((m) => m[1]!);
    if (!docs.includes("P-NEEDLE")) return reply("I could not find a carry-over limit in the documents.");
    const pos = docs.indexOf("P-NEEDLE");
    const n = docs.length;
    const rel = n > 1 ? pos / (n - 1) : 0;
    const queryFirst = user.indexOf("Question:") < user.indexOf("<doc");
    const missed = n > 20 && ((queryFirst && rel > 0.2 && rel < 0.8) || (!queryFirst && rel > 0.4 && rel < 0.6));
    return reply(missed ? "I could not find a carry-over limit in the documents." : "Employees may carry over up to 7 unused vacation days [P-NEEDLE].");
  }
  return reply("OK");
}

// ---------------------------------------------------------------------------
// Probes (the reusable diagnostic toolkit)
// ---------------------------------------------------------------------------
function classifyFormat(text: string): string {
  const t = text.trim();
  try {
    const obj: unknown = JSON.parse(t);
    return typeof obj === "object" && obj !== null && !Array.isArray(obj) ? "json" : "other-json";
  } catch {
    // not JSON - fall through
  }
  if (t.startsWith("```")) return "fenced-json";
  const lines = t.split(/\r?\n/);
  if (lines.length && lines.every((l) => l.startsWith("- "))) return "bullets";
  if (lines.length && lines.every((l) => /^\d+\. /.test(l))) return "numbered";
  return "prose";
}

function coefficientOfVariation(xs: number[]): number {
  const m = total(xs) / xs.length;
  const v = total(xs.map((x) => (x - m) ** 2)) / xs.length;
  return m ? Math.sqrt(v) / m : 0;
}

/** Share of answer content words that appear in the supplied context (0 if no context). */
function groundedFraction(answer: string, context: string): number {
  const a = contentWords(answer.replace(/\[[^\]]+\]/g, ""));
  if (a.size === 0) return 1;
  return overlap(a, contentWords(context)) / a.size;
}

function retrieve(question: string, docs: [string, string][], k: number): [string, string][] {
  const q = contentWords(question);
  const scored = docs.map((d): [[string, string], number] => [d, overlap(q, contentWords(d[1]))]);
  scored.sort((x, y) => y[1] - x[1] || (x[0][0] < y[0][0] ? -1 : x[0][0] > y[0][0] ? 1 : 0));
  return scored.slice(0, k).map((s) => s[0]);
}

function docBlock(items: [string, string][], version?: Record<string, number>): string {
  return items.map(([id, text]) => `<doc id="${id}"${version ? ` version="${version[id]}"` : ""}>\n${text}\n</doc>`).join("\n");
}

const sha = (text: string): string => createHash("sha256").update(text).digest("hex").slice(0, 12);

const LATENCY_PROFILE: Record<string, [number, number]> = { [MODEL_HAIKU]: [250, 3], [MODEL_SONNET]: [500, 7], [MODEL_OPUS]: [900, 12] };
function modeledLatency(key: string, model: string, outTokens: number): number {
  const rng = mulberry32(fnv1a32(key));
  const [ttft, per] = LATENCY_PROFILE[model]!;
  return ttft * (0.85 + 0.3 * rng()) + per * outTokens;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------
async function scenarioFormat(client: Anthropic): Promise<void> {
  const r = new Report(1, "Ambiguous prompt -> inconsistent output format");
  r.line("SYMPTOM", "the downstream parser rejects ~half of the ticket summaries; nothing changed in the code");
  r.hypotheses(["prompt does not specify an output contract", "model regression", "parser bug"]);
  const outs: string[] = [];
  for (let i = 0; i < 8; i++) {
    outs.push(textOf(await client.messages.create({ model: MODEL_SONNET, max_tokens: 512, system: "Summarize the support ticket.", messages: [{ role: "user", content: TICKET }] })));
  }
  const formats = outs.map(classifyFormat);
  const counts = new Map<string, number>();
  for (const f of formats) counts.set(f, (counts.get(f) ?? 0) + 1);
  const parseOk = formats.filter((f) => f === "json").length;
  r.probe(
    "same input sampled 8x (no temperature knob - variance is inherent, so sample it)",
    `${counts.size} distinct formats (` + [...counts].map(([k, v]) => `${k}=${v}`).join(", ") + `); strict-JSON parse success ${parseOk}/8; ` +
      `length CV=${fmt(coefficientOfVariation(outs.map((o) => o.length)), 2)}`,
  );
  r.probe("parser on a known-good JSON sample", "parses fine -> parser is not the problem");

  const TicketSummary = z.object({ issue: z.string(), product: z.string(), urgency: z.enum(["low", "medium", "high"]) });
  const fixed: boolean[] = [];
  for (let i = 0; i < 8; i++) {
    const msg = await client.messages.parse({
      model: MODEL_SONNET,
      max_tokens: 512,
      system: "Summarize the support ticket as JSON with keys issue, product, urgency.",
      messages: [{ role: "user", content: TICKET }],
      output_config: { format: zodOutputFormat(TicketSummary) },
    });
    fixed.push(msg.parsed_output !== null);
  }
  r.conclude(
    "prompt",
    "no output contract",
    "underspecified prompt: no output contract, so each sample picks its own format",
    "state the format explicitly AND enforce it with structured outputs (output_config.format / messages.parse)",
    `8 samples after fix: ${fixed.filter(Boolean).length}/8 schema-valid, 1 format`,
  );
}

async function scenarioGrounding(client: Anthropic): Promise<void> {
  const r = new Report(2, "Hallucination from missing grounding");
  r.line("SYMPTOM", "the product bot states warranty terms and features with full confidence; some are wrong");
  r.hypotheses(["no source documents in the prompt", "outdated documents", "model ignores provided context"]);
  const sysPrompt = "You are a helpful product expert for the X200 router.";
  const q1 = "How long is the warranty period of the X200?";
  const q2 = "Does the X200 support 6 GHz Wi-Fi?";
  const ask = async (system: string, content: string): Promise<string> =>
    textOf(await client.messages.create({ model: MODEL_SONNET, max_tokens: 256, system, messages: [{ role: "user", content }] }));
  const samples: string[] = [];
  for (let i = 0; i < 5; i++) samples.push(await ask(sysPrompt, q1));
  const years = samples.map((s) => /(\d+)-year/.exec(s)?.[1]).filter((y): y is string => y !== undefined);
  r.probe("self-consistency: ask the same factual question 5x", `warranty answers ${years.join("/")} years -> ${new Set(years).size} different 'facts' = invention, not knowledge`);
  const unans = await ask(sysPrompt, q2);
  r.probe(
    "groundedness of answers vs supplied context",
    `context supplied: none -> groundedness ${pct(groundedFraction(samples[0]!, ""))}; unanswerable question answered confidently: '${unans}'`,
  );
  const groundedSys = sysPrompt + " Answer ONLY from the documents. Cite the doc id in brackets. If the answer is not in the documents, say you don't know.";
  const fixed: string[] = [];
  for (const q of [q1, q1, q1, q2]) {
    const ctx = docBlock(retrieve(q, Object.entries(PRODUCT_DOCS), 2));
    fixed.push(await ask(groundedSys, `${ctx}\n\nQuestion: ${q}`));
  }
  const ctxAll = Object.values(PRODUCT_DOCS).join(" ");
  r.conclude(
    "grounding",
    "no source documents",
    "the model was asked for facts it was never given; it filled the gap fluently",
    "retrieve source docs into the prompt, require citations, and give explicit permission to say 'I don't know'",
    `warranty answer x3: '${fixed[0]}' (consistent: ${new Set(fixed.slice(0, 3)).size === 1 ? "yes" : "no"}, ` +
      `groundedness ${pct(groundedFraction(fixed[0]!, ctxAll))}); unanswerable -> '${fixed[3]}'`,
  );
}

async function scenarioStaleIndex(client: Anthropic): Promise<void> {
  const r = new Report(3, "Stale RAG index after a document refresh");
  r.line("SYMPTOM", "after Monday's policy refresh the assistant gives confident WRONG refund answers;");
  r.line("", "latency, error rate and model are unchanged");
  r.hypotheses(["retrieval serves stale chunks (index not rebuilt)", "model or prompt changed", "infra degradation"]);
  const sysPrompt = "You answer questions about Acme policies using ONLY the provided context. Cite the doc id.";
  type Chunk = { text: string; version: number; hash: string };
  // built last week
  const index: Record<string, Chunk> = Object.fromEntries(Object.entries(POLICY_V1).map(([id, text]) => [id, { text, version: 1, hash: sha(text) }]));
  const source: Record<string, Chunk> = Object.fromEntries(
    Object.entries(POLICY_V2).map(([id, text]) => [id, { text, version: POLICY_V1[id] !== text ? 2 : 1, hash: sha(text) }]),
  );
  const question = "Within how many days of delivery are refunds accepted?";
  const asText = (store: Record<string, Chunk>): [string, string][] => Object.entries(store).map(([k, v]) => [k, v.text]);

  const ask = async (store: Record<string, Chunk>, key: string): Promise<[string, number]> => {
    const hits = retrieve(question, asText(store), 2);
    const ctx = docBlock(hits, Object.fromEntries(hits.map(([k]) => [k, store[k]!.version])));
    const msg = await client.messages.create({ model: MODEL_SONNET, max_tokens: 256, system: sysPrompt, messages: [{ role: "user", content: `${ctx}\n\nQuestion: ${question}` }] });
    return [textOf(msg), modeledLatency(key, msg.model, msg.usage.output_tokens)];
  };
  const before: [string, number][] = [];
  for (let i = 0; i < 3; i++) before.push(await ask(index, `before-${i}`)); // last week: index == source
  const after: [string, number][] = [];
  for (let i = 0; i < 3; i++) after.push(await ask(index, `after-${i}`)); // today: source changed, index did not
  r.probe("config diff (model id, prompt version) week-over-week", `model ${MODEL_SONNET} both weeks; prompt hash unchanged -> not a model/prompt change`);
  const p50b = before.map((x) => x[1]).sort((a, b) => a - b)[1]!;
  const p50a = after.map((x) => x[1]).sort((a, b) => a - b)[1]!;
  r.probe("latency and errors before vs after refresh", `p50 ${fmt(p50b, 0)} ms vs ${fmt(p50a, 0)} ms, errors 0 vs 0 -> infra and model are healthy`);
  const ans = after[0]![0];
  r.probe(
    "generation check: is the answer faithful to what was retrieved?",
    `answer '${ans}' is ${pct(groundedFraction(ans, Object.values(index).map((v) => v.text).join(" ")))} grounded in the retrieved chunks ` +
      "-> the model is NOT hallucinating",
  );
  const retrieved = retrieve(question, asText(index), 2);
  const stale = retrieved.filter(([k]) => index[k]!.hash !== source[k]!.hash);
  const detail = retrieved.map(([k]) => `${k}: index v${index[k]!.version} ${index[k]!.hash} vs source v${source[k]!.version} ${source[k]!.hash}`).join(", ");
  r.probe("retrieval check: compare retrieved chunk hashes/versions with the source of truth", `${stale.length}/${retrieved.length} retrieved chunks are stale (${detail})`);
  const reindexed: Record<string, Chunk> = Object.fromEntries(Object.entries(source).map(([k, v]) => [k, { ...v }])); // incremental re-embed of changed docs
  const [fixed] = await ask(reindexed, "fixed");
  r.conclude(
    "data pipeline",
    "index not rebuilt after refresh",
    "the refresh updated the source store but the re-indexing job did not run; the model faithfully answered from stale chunks",
    "re-index changed docs (content-hash diff), alert on ingest/index job failures, monitor index-vs-source version drift, add regression cases for updated docs",
    `after re-index: '${fixed}'`,
  );
}

async function scenarioModelMismatch(client: Anthropic): Promise<void> {
  const r = new Report(4, "Model mismatch on multi-step reasoning");
  r.line("SYMPTOM", "after a cost-cutting switch to a smaller model, prorated refund totals are often wrong;");
  r.line("", "simple billing questions are still fine");
  r.hypotheses(["task exceeds the smaller model's reasoning capability", "prompt regression", "bad test data"]);
  const sysPrompt = "You are a billing calculator. Show only the final line 'Net charge: <amount> dollars'.";
  const rows: [string, number, number, number, number][] = [];
  const taskCost = new Map<string, number>();
  for (const model of [MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS]) {
    const correct = { simple: 0, complex: 0 };
    let cost = 0;
    for (const task of BILLING_TASKS) {
      const msg = await client.messages.create({
        model,
        max_tokens: 1024,
        system: sysPrompt,
        output_config: { effort: "medium" },
        messages: [{ role: "user", content: task.text }],
      });
      const m = /(-?\d+(?:\.\d+)?)/.exec(textOf(msg));
      const ok = m !== null && Math.abs(Number(m[1]) - task.answer) < 0.01;
      correct[task.kind] += ok ? 1 : 0;
      taskCost.set(`${model}|${task.id}`, costUsd(msg.model, msg.usage));
      cost += taskCost.get(`${model}|${task.id}`)!;
    }
    rows.push([model, correct.simple, correct.complex, cost, correct.simple + correct.complex]);
  }
  for (const [model, s, c, cost, nOk] of rows) {
    r.probe(`same 8 tasks on ${model}`, `simple ${s}/3, multi-step ${c}/5, total cost $${fmt(cost, 6)}, cost per correct answer $${fmt(cost / nOk, 6)}`);
  }
  const routed = total(BILLING_TASKS.map((t) => taskCost.get(`${t.kind === "simple" ? MODEL_HAIKU : MODEL_OPUS}|${t.id}`)!));
  r.conclude(
    "model",
    "small model on multi-step math",
    "capability mismatch: the same prompt is correct on larger models, so the prompt is not the problem",
    "route by complexity (small model for lookups, larger model for multi-step math) or use the larger model at lower " +
      "effort; compute arithmetic in code/tools; always re-run the eval before downgrading a model",
    `routed (simple->Haiku, multi-step->Opus): 8/8 correct for $${fmt(routed, 6)} vs $${fmt(rows[2]![3], 6)} all-Opus; ` +
      "a wrong refund costs far more than the tokens saved",
  );
}

async function scenarioTruncation(client: Anthropic): Promise<void> {
  const r = new Report(5, "Truncation (stop_reason = max_tokens)");
  r.line("SYMPTOM", "some incident reports are cut off mid-sentence and fail JSON parsing; short incidents are fine");
  r.hypotheses(["max_tokens too low for long outputs", "model produces invalid JSON", "network cut the stream"]);
  const sysPrompt = "Write an incident report as compact JSON with keys title, impact, timeline, root_cause, actions.";

  const run = async (maxTokens: number): Promise<[string, string, number, boolean][]> => {
    const out: [string, string, number, boolean][] = [];
    for (const inc of INCIDENTS) {
      const msg = await client.messages.create({
        model: MODEL_SONNET,
        max_tokens: maxTokens,
        system: sysPrompt,
        messages: [{ role: "user", content: `Incident ${inc.id}: ${inc.title}, ${inc.events} timeline events.` }],
      });
      let valid: boolean;
      try {
        JSON.parse(textOf(msg));
        valid = true;
      } catch {
        valid = false;
      }
      out.push([inc.id, msg.stop_reason ?? "", msg.usage.output_tokens, valid]);
    }
    return out;
  };

  const before = await run(150);
  const hist = new Map<string, number>();
  for (const [, stop] of before) hist.set(stop, (hist.get(stop) ?? 0) + 1);
  r.probe("stop_reason histogram (max_tokens=150)", [...hist].map(([k, v]) => `${k}=${v}`).join(", "));
  r.probe("correlate stop_reason with parse failures", before.map(([i, stop, out, ok]) => `${i}: ${stop} out=${out} json=${ok ? "ok" : "INVALID"}`).join("; "));
  const after = await run(2048);
  r.conclude(
    "config",
    "max_tokens below output length",
    "max_tokens (copied from a chat feature) is below the length of long reports; every invalid JSON has stop_reason=max_tokens and output_tokens == max_tokens",
    "raise max_tokens as a backstop, ask for concise output, stream long generations, and treat stop_reason " +
      "'max_tokens' as a failure (never parse it; retry with a higher cap)",
    `max_tokens=2048: ${after.filter((x) => x[1] === "end_turn").length}/6 end_turn, ${after.filter((x) => x[3]).length}/6 valid JSON`,
  );
}

async function scenarioContext(client: Anthropic): Promise<void> {
  const r = new Report(6, "Context overload / lost-in-the-middle");
  r.line("SYMPTOM", "as the policy KB grew, the HR assistant says 'not found' for facts that exist; cost per call rising");
  r.hypotheses(["too much context: relevant fact buried mid-prompt", "fact missing from KB", "retrieval bug"]);
  const sysPrompt = "You are the HR policy assistant. Answer from the documents and cite the policy id.";
  const budget = 4000;

  const build = (pos: number, queryFirst: boolean, retrieval: boolean): string => {
    let docs: [string, string][] = TOPICS.map((t, i): [string, string] => [`P-${String(i).padStart(3, "0")}`, fillerDoc(i, t)]);
    docs.splice(pos, 0, ["P-NEEDLE", NEEDLE]);
    if (retrieval) docs = retrieve(VACATION_Q, docs, 3);
    const body = docs.map(([d, t]) => `<doc id="${d}">\n${t}\n</doc>`).join("\n");
    return queryFirst ? `Question: ${VACATION_Q}\n\n${body}` : `${body}\n\nQuestion: ${VACATION_Q}`;
  };
  const found = async (content: string): Promise<boolean> => {
    const msg = await client.messages.create({ model: MODEL_SONNET, max_tokens: 256, system: sysPrompt, messages: [{ role: "user", content }] });
    return textOf(msg).includes("7");
  };

  const nDocs = TOPICS.length + 1;
  const full = await client.messages.countTokens({ model: MODEL_SONNET, system: sysPrompt, messages: [{ role: "user", content: build(30, true, false) }] });
  r.probe("count_tokens on the production request", `${nDocs} docs -> ${full.input_tokens} input tokens vs route budget ${budget} -> ${full.input_tokens > budget ? "OVER budget" : "within budget"}`);
  const positions = [0, 15, 30, 45, nDocs - 1];
  const sweepRun = async (queryFirst: boolean, retrieval: boolean): Promise<boolean[]> => {
    const out: boolean[] = [];
    for (const p of positions) out.push(await found(build(p, queryFirst, retrieval)));
    return out;
  };
  const show = (res: boolean[]): string => positions.map((p, i) => `pos ${p}: ${res[i] ? "found" : "MISSED"}`).join("  ");
  r.probe("needle position sweep (fact moved through the context, question first)", show(await sweepRun(true, false)));
  r.probe("same sweep with documents first, question last", show(await sweepRun(false, false)));
  const sweepRag = await sweepRun(false, true);
  const small = await client.messages.countTokens({ model: MODEL_SONNET, system: sysPrompt, messages: [{ role: "user", content: build(30, false, true) }] });
  r.conclude(
    "context",
    "fact buried in oversized context",
    "context stuffing: the fact is present but buried mid-context (U-shaped recall), with the question placed before 60 documents",
    "retrieve the top-k relevant docs instead of everything, put long documents first and the question last, and gate requests with count_tokens",
    `top-3 retrieval + question last: found at ${sweepRag.filter(Boolean).length}/5 positions; ${small.input_tokens} tokens ` +
      `(${pct(1 - small.input_tokens / full.input_tokens)} fewer)`,
  );
}

async function main(): Promise<void> {
  banner("D4.4 - Diagnosing Prompt Failure, Hallucination, Model Mismatch");
  section("Diagnostic decision tree (symptom -> first probe -> likely root cause)");
  const tree: [string, string, string][] = [
    ["output format varies between runs", "sample N times, measure variance", "underspecified prompt"],
    ["confident facts with no source", "self-consistency + groundedness", "missing grounding"],
    ["wrong after content change, latency OK", "retrieved chunk version vs source", "stale index / ingestion"],
    ["fails only on complex tasks", "same inputs across model tiers", "model mismatch"],
    ["cut off / invalid JSON on long outputs", "stop_reason histogram", "max_tokens truncation"],
    ["misses facts that are in the KB", "count_tokens + position sweep", "context overload"],
  ];
  console.log(`  ${"SYMPTOM".padEnd(40)}${"FIRST PROBE".padEnd(36)}LIKELY ROOT CAUSE`);
  for (const [s, p, c] of tree) console.log(`  ${s.padEnd(40)}${p.padEnd(36)}${c}`);
  console.log("  Rule: change ONE thing at a time, and probe the cheapest layer that could explain the symptom first");
  console.log("  (config diff, stop_reason, latency/errors) before blaming the model.");
  const client = getClient(responder);
  await scenarioFormat(client);
  await scenarioGrounding(client);
  await scenarioStaleIndex(client);
  await scenarioModelMismatch(client);
  await scenarioTruncation(client);
  await scenarioContext(client);
  section("Diagnosis summary");
  console.log(`  ${"scenario".padEnd(48)}${"layer".padEnd(15)}root cause`);
  for (const [title, cat, root] of SUMMARY) console.log(`  ${title.padEnd(48)}${cat.padEnd(15)}${root}`);
}

await main();
