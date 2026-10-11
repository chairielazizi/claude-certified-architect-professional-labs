/**
 * D4.3 - A/B Testing and Iterative Improvement
 *
 * 1. Offline A/B: two prompt variants on the SAME eval set -> paired analysis
 *    (McNemar exact test + paired bootstrap CI).
 * 2. Online A/B simulation: deterministic hash bucketing, sample-ratio-mismatch check,
 *    power / minimum sample size, two-proportion z-test with 95% CI, guardrail metrics.
 * 3. A/A test: why peeking at p-values inflates false positives.
 * 4. Iteration loop (hill-climb): Claude proposes prompt edits from TRAIN failures ->
 *    evaluate on VALIDATION -> keep or revert -> final report on held-out TEST.
 *
 * Run from the repo root:
 *   node "D4.3 - A-B Testing and Iterative Improvement/typescript/main.ts"
 */

import type Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  MODEL_HAIKU,
  MODEL_OPUS,
  banner,
  costUsd,
  estimateTokens,
  getClient,
  section,
  textBlock,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
const APP_MODEL = MODEL_HAIKU; // ticket router: classification at volume
const ANALYZER_MODEL = MODEL_OPUS; // reads failures and proposes edits (a stronger model than the one under test)
const CONCURRENCY = 16;

// ---------------------------------------------------------------------------
// Deterministic helpers (identical algorithms in the Python version)
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

const hash01 = (s: string): number => mulberry32(fnv1a32(s))();
const fmt = (x: number, d = 2): string => (Math.floor(x * 10 ** d + 0.5) / 10 ** d).toFixed(d);
const pct = (x: number): string => fmt(100 * x, 1) + "%";
const signedPct = (x: number): string => (x >= 0 ? "+" : "") + fmt(100 * x, 1) + "%";
const fmtP = (p: number): string => (p < 0.0001 ? "<0.0001" : fmt(p, 4));
function total(xs: Iterable<number>): number {
  let s = 0;
  for (const x of xs) s += x;
  return s;
}
const mean = (xs: number[]): number => (xs.length ? total(xs) / xs.length : 0);

function percentile(values: number[], p: number): number {
  const s = [...values].sort((x, y) => x - y);
  const rank = (p / 100) * (s.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo]! + (s[hi]! - s[lo]!) * (rank - lo);
}

/** Minimal async semaphore (like asyncio.Semaphore). */
class Semaphore {
  private available: number;
  private waiters: (() => void)[] = [];
  constructor(n: number) {
    this.available = n;
  }
  async run<T>(fn: () => Promise<T>): Promise<T> {
    if (this.available > 0) this.available -= 1;
    else await new Promise<void>((resolve) => this.waiters.push(resolve));
    try {
      return await fn();
    } finally {
      const next = this.waiters.shift();
      if (next) next();
      else this.available += 1;
    }
  }
}

// ---------------------------------------------------------------------------
// Statistics (implemented from scratch)
// ---------------------------------------------------------------------------
/** Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7). */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return sign * y;
}
const normCdf = (z: number): number => 0.5 * (1 + erf(z / Math.sqrt(2)));

/** Acklam's rational approximation of the inverse normal CDF. */
function normPpf(p: number): number {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  const hi = 1 - 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }
  if (p > hi) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) / (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
}

/** b = A right & B wrong, c = A wrong & B right. Returns [exact two-sided p, chi2-with-continuity p]. */
function mcnemar(b: number, c: number): [number, number] {
  const n = b + c;
  if (n === 0) return [1, 1];
  let term = 0.5 ** n; // P(X = 0) for X ~ Binomial(n, 0.5), then iterate
  let tail = 0;
  for (let i = 0; i <= Math.min(b, c); i++) {
    tail += term;
    term = (term * (n - i)) / (i + 1);
  }
  const exact = Math.min(1, 2 * tail);
  const chi2 = (Math.abs(b - c) - 1) ** 2 / n;
  return [exact, 1 - erf(Math.sqrt(chi2 / 2))];
}

/** 95% percentile CI for mean(b) - mean(a), resampling CASES (keeps the pairing). */
function pairedBootstrap(a: number[], b: number[], iters: number, seed: number): [number, number] {
  const rng = mulberry32(seed);
  const n = a.length;
  const diffs: number[] = [];
  for (let it = 0; it < iters; it++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      const i = Math.floor(rng() * n);
      s += b[i]! - a[i]!;
    }
    diffs.push(s / n);
  }
  return [percentile(diffs, 2.5), percentile(diffs, 97.5)];
}

/** Returns [diff p2-p1, z, two-sided p, 95% CI half-width]. Pooled SE for the test, unpooled for the CI. */
function twoPropZtest(x1: number, n1: number, x2: number, n2: number): [number, number, number, number] {
  const p1 = x1 / n1;
  const p2 = x2 / n2;
  const pool = (x1 + x2) / (n1 + n2);
  const sePool = Math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2));
  const z = sePool > 0 ? (p2 - p1) / sePool : 0;
  const p = 2 * (1 - normCdf(Math.abs(z)));
  const half = 1.959964 * Math.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2);
  return [p2 - p1, z, p, half];
}

function sampleSizePerArm(p1: number, mde: number, alpha = 0.05, power = 0.8): number {
  const p2 = p1 + mde;
  const pbar = (p1 + p2) / 2;
  const za = normPpf(1 - alpha / 2);
  const zb = normPpf(power);
  const n = (za * Math.sqrt(2 * pbar * (1 - pbar)) + zb * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / mde ** 2;
  return Math.ceil(n);
}

// ---------------------------------------------------------------------------
// Dataset, prompts, offline responder
// ---------------------------------------------------------------------------
interface Ticket {
  id: string;
  label: string;
  difficulty: "easy" | "hard";
  text: string;
}
const SPEC = JSON.parse(readFileSync(path.join(DATA, "ticket_templates.json"), "utf-8")) as {
  labels: string[];
  templates: { label: string; difficulty: "easy" | "hard"; text: string; fillers: string[] }[];
};
const LABELS = SPEC.labels;
const TICKETS: Ticket[] = [];
SPEC.templates.forEach((tpl, ti) =>
  tpl.fillers.forEach((filler, fi) =>
    TICKETS.push({ id: `${tpl.label.slice(0, 4)}-${String(ti).padStart(2, "0")}-${fi}`, label: tpl.label, difficulty: tpl.difficulty, text: tpl.text.replace("{x}", filler) }),
  ),
);
const BY_TEXT = new Map(TICKETS.map((t) => [t.text, t]));

// Stable stratified split: within each label, order by hash and deal round-robin.
const SPLIT = new Map<string, string>();
for (const lab of LABELS) {
  const group = TICKETS.filter((t) => t.label === lab).sort((x, y) => fnv1a32("split:" + x.id) - fnv1a32("split:" + y.id));
  group.forEach((t, i) => SPLIT.set(t.id, ["train", "validation", "test"][i % 3]!));
}

const ROLE = "You route customer support tickets to exactly one team: billing, shipping, technical, account, sales.";
const FORMAT = "Reply in the format:\nLabel: <team>\nReason: <one sentence>";
const DEFS =
  "Label definitions:\n- billing: charges, invoices, refunds, payment methods, fees (including delivery surcharges).\n" +
  "- shipping: physical delivery status, addresses, couriers.\n- technical: errors, crashes, outages, broken pages (even the billing page).\n" +
  "- account: sign-in identity, security, members, ownership, profile data.\n- sales: pricing questions, discounts, demos, contracts, renewals.\n" +
  "Tie-break: route by the action the customer needs, not by the words they use.";
const PROMPT_A = ROLE + "\n" + FORMAT;
const PROMPT_B = PROMPT_A + "\n" + DEFS;

// Offline "model": P(correct) depends on difficulty and on which prompt blocks are present.
const BASE_P = { easy: 0.75, hard: 0.38 };
const EFFECTS: Record<string, [number, number]> = { defs: [0.045, 0.2], fewshot: [0.03, 0.1] };
const MARKERS: Record<string, string> = {
  defs: "Label definitions:", fewshot: "Examples:", label_only: "Do not add a reason",
  overfit: "Known tickets:", fluff: "Think carefully",
};

const featuresOf = (system: string): string[] => Object.keys(MARKERS).filter((f) => system.includes(MARKERS[f]!)).sort();

function mockLabel(ticket: Ticket, system: string): string {
  const feats = featuresOf(system);
  if (feats.includes("overfit") && system.includes(ticket.text)) return ticket.label; // memorised: right on train, no help elsewhere
  const hard = ticket.difficulty === "hard";
  const quality = feats.filter((f) => f in EFFECTS); // only blocks that change the reasoning perturb outcomes
  const p = BASE_P[ticket.difficulty] + total(quality.map((f) => EFFECTS[f]![hard ? 1 : 0]));
  // 70% shared per-case difficulty (paired designs exploit this) + 30% variant-specific noise
  const u = 0.7 * hash01("case:" + ticket.id) + 0.3 * hash01("var:" + ticket.id + ":" + quality.join(","));
  if (u < p) return ticket.label;
  const gold = LABELS.indexOf(ticket.label);
  return LABELS[(gold + 1 + (fnv1a32("wrong:" + ticket.id) % 4)) % 5]!;
}

function analyzerReply(prompt: string): string {
  const failing = [...prompt.matchAll(/^- \[[^\]]+\] (.+?) \| predicted=\w+ \| gold=(\w+)$/gm)].map((m) => [m[1]!, m[2]!]);
  const known = failing.slice(0, 4).map(([t, g]) => `'${t}' -> ${g}`).join("; ");
  return JSON.stringify({
    candidates: [
      { name: "patch the failing tickets", rationale: "Hard-code the exact train tickets that failed.", block: "Known tickets: " + known },
      {
        name: "few-shot examples",
        rationale: "Hard tickets use words from another team; show three boundary examples.",
        block: "Examples:\nTicket: I see a handling fee on my statement. -> billing\nTicket: The invoices page shows error 504. -> technical\nTicket: We want to renegotiate our renewal. -> sales",
      },
      {
        name: "label-only output",
        rationale: "Reasons are never read downstream; dropping them cuts output tokens and latency.",
        block: "Output override: respond with 'Label: <team>' only. Do not add a reason.",
      },
      { name: "accuracy reminder", rationale: "Remind the model to be careful.", block: "Think carefully and be accurate." },
    ],
  });
}

function responder(req: MockRequest): ResponderResult {
  if (req.systemText.startsWith("You improve prompts")) {
    const text = analyzerReply(req.lastUserText);
    return { content: [textBlock(text)], usage: { output_tokens: estimateTokens(text) } };
  }
  const m = /Ticket: (.+)$/m.exec(req.lastUserText);
  const ticket = m ? BY_TEXT.get(m[1]!.trim()) : undefined;
  if (!ticket) return "Label: technical\nReason: unclear ticket.";
  const label = mockLabel(ticket, req.systemText);
  const feats = featuresOf(req.systemText);
  let text: string;
  if (feats.includes("label_only")) text = `Label: ${label}`;
  else if (feats.includes("defs")) {
    text =
      `Label: ${label}\nReason: Per the label definitions, the customer needs an action that the ${label} team owns; ` +
      "the ticket wording may mention other areas, such as fees, apps or delivery, but the requested action decides the route.";
  } else text = `Label: ${label}\nReason: The customer describes an issue that matches the ${label} team based on the main request.`;
  return { content: [textBlock(text)], usage: { output_tokens: estimateTokens(text) } };
}

const CandidateList = z.object({
  candidates: z.array(z.object({ name: z.string(), rationale: z.string(), block: z.string() })),
});

// ---------------------------------------------------------------------------
// Running a prompt variant over a set of tickets
// ---------------------------------------------------------------------------
interface Outcome {
  ticketId: string;
  correct: number;
  outputTokens: number;
  cost: number;
}

async function evaluate(client: Anthropic, sem: Semaphore, system: string, tickets: Ticket[]): Promise<Outcome[]> {
  return Promise.all(
    tickets.map(async (t) => {
      const msg = await sem.run(() =>
        client.messages.create({
          model: APP_MODEL,
          max_tokens: 256,
          system,
          output_config: { effort: "low" },
          messages: [{ role: "user", content: "Ticket: " + t.text }],
        }),
      );
      const text = msg.content.map((b) => (b.type === "text" ? b.text : "")).join("").toLowerCase();
      const m = /label:\s*([a-z]+)/.exec(text);
      return { ticketId: t.id, correct: m && m[1] === t.label ? 1 : 0, outputTokens: msg.usage.output_tokens, cost: costUsd(msg.model, msg.usage) };
    }),
  );
}
const acc = (outs: Outcome[]): number => outs.reduce((s, o) => s + o.correct, 0) / outs.length;
const sumCorrect = (outs: Outcome[]): number => outs.reduce((s, o) => s + o.correct, 0);

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main(): Promise<void> {
  banner("D4.3 - A/B Testing and Iterative Improvement");
  const client = getClient(responder);
  const sem = new Semaphore(CONCURRENCY);
  const train = TICKETS.filter((t) => SPLIT.get(t.id) === "train");
  const val = TICKETS.filter((t) => SPLIT.get(t.id) === "validation");
  const test = TICKETS.filter((t) => SPLIT.get(t.id) === "test");

  // -- 1 ------------------------------------------------------------------
  section("1. Setup: dataset, splits, variants");
  console.log(
    `  ${TICKETS.length} routing tickets (${TICKETS.filter((t) => t.difficulty === "hard").length} hard), ` +
      `split stratified by label: train=${train.length} validation=${val.length} test=${test.length}`,
  );
  console.log(`  App: ${APP_MODEL} (effort=low). Variant A = role + format. Variant B = A + label definitions & tie-break rule.`);
  console.log(`  Prompt size: A ~${estimateTokens(PROMPT_A)} tokens, B ~${estimateTokens(PROMPT_B)} tokens`);

  // -- 2 ------------------------------------------------------------------
  section("2. Offline A/B on the same 120 cases (paired)");
  const outA = await evaluate(client, sem, PROMPT_A, TICKETS);
  const outB = await evaluate(client, sem, PROMPT_B, TICKETS);
  for (const diff of ["easy", "hard"]) {
    const idx = TICKETS.map((t, i) => (t.difficulty === diff ? i : -1)).filter((i) => i >= 0);
    console.log(`  ${diff.padEnd(5)} n=${String(idx.length).padEnd(3)} A=${pct(mean(idx.map((i) => outA[i]!.correct)))}  B=${pct(mean(idx.map((i) => outB[i]!.correct)))}`);
  }
  console.log(`  all   n=${String(TICKETS.length).padEnd(3)} A=${pct(acc(outA))}  B=${pct(acc(outB))}  diff=${signedPct(acc(outB) - acc(outA))}`);
  const bOnlyA = outA.filter((x, i) => x.correct && !outB[i]!.correct).length;
  const cOnlyB = outA.filter((x, i) => !x.correct && outB[i]!.correct).length;
  const both = outA.filter((x, i) => x.correct && outB[i]!.correct).length;
  console.log(`  paired table: both right=${both}, A only=${bOnlyA}, B only=${cOnlyB}, both wrong=${TICKETS.length - both - bOnlyA - cOnlyB}`);
  const [pExact, pChi] = mcnemar(bOnlyA, cOnlyB);
  let [lo, hi] = pairedBootstrap(outA.map((o) => o.correct), outB.map((o) => o.correct), 2000, 7);
  console.log(`  McNemar (only discordant pairs matter): exact p=${fmtP(pExact)}, chi2-corrected p=${fmtP(pChi)}`);
  console.log(`  paired bootstrap 95% CI for B-A: [${signedPct(lo)}, ${signedPct(hi)}] (2000 resamples of cases)`);
  const tokA = mean(outA.map((o) => o.outputTokens));
  const tokB = mean(outB.map((o) => o.outputTokens));
  const costA = mean(outA.map((o) => o.cost));
  const costB = mean(outB.map((o) => o.cost));
  console.log(`  output tokens/ticket A=${fmt(tokA, 1)} B=${fmt(tokB, 1)}; cost/ticket A=$${fmt(costA, 7)} B=$${fmt(costB, 7)}`);
  console.log(
    "  Decision: B is better offline" + (pExact < 0.05 ? " (significant at 5%)." : " but not significant - collect more cases.") +
      " Next: does it hold for real users?",
  );

  // -- 3 ------------------------------------------------------------------
  section("3. Online A/B design: bucketing, sample-ratio check, power");
  const experiment = "ticket-router-defs-v1";
  const nUsers = 8000;
  // Deterministic: same user -> same arm on every request and every server; salt by experiment
  // so different experiments get independent assignments.
  const bucket = (user: string): string =>
    parseInt(createHash("sha256").update(`${experiment}:${user}`).digest("hex").slice(0, 8), 16) % 100 < 50 ? "A" : "B";
  const users = Array.from({ length: nUsers }, (_, i) => `u${String(i + 1).padStart(5, "0")}`);
  const armOf = new Map(users.map((u) => [u, bucket(u)]));
  const nA = users.filter((u) => armOf.get(u) === "A").length;
  const nB = nUsers - nA;
  const chi2 = ((nA - nUsers / 2) ** 2 + (nB - nUsers / 2) ** 2) / (nUsers / 2);
  const srmP = 1 - erf(Math.sqrt(chi2 / 2));
  console.log(`  sha256('${experiment}:<user>') % 100 < 50 -> A.  A=${nA} users, B=${nB} users`);
  console.log(`  sample-ratio-mismatch check: chi2=${fmt(chi2, 3)} p=${fmtP(srmP)} -> ${srmP < 0.001 ? "SRM! stop and debug assignment" : "OK (no SRM)"}`);
  const baseRate = 0.7;
  const mde = 0.04;
  const need = sampleSizePerArm(baseRate, mde);
  console.log(`  power analysis: baseline success ${pct(baseRate)}, minimum detectable effect ${signedPct(mde)} abs, alpha=0.05, power=0.80`);
  console.log(`  -> need ${need} users per arm; we have ~${Math.min(nA, nB)} -> ${Math.min(nA, nB) >= need ? "adequately powered" : "UNDERPOWERED"}`);

  // Measure each arm's real token/cost profile with a few calls through the shared client.
  const profA = await evaluate(client, sem, PROMPT_A, train.slice(0, 10));
  const profB = await evaluate(client, sem, PROMPT_B, train.slice(0, 10));
  const profile: Record<string, [number, number]> = {
    A: [mean(profA.map((o) => o.outputTokens)), mean(profA.map((o) => o.cost))],
    B: [mean(profB.map((o) => o.outputTokens)), mean(profB.map((o) => o.cost))],
  };
  console.log(
    `  measured per-call profile (10 calls/arm): A ${fmt(profile.A![0], 1)} out tok $${fmt(profile.A![1], 7)}, ` +
      `B ${fmt(profile.B![0], 1)} out tok $${fmt(profile.B![1], 7)}`,
  );

  // -- 4 ------------------------------------------------------------------
  section("4. Online results (simulated users) and guardrails");
  const trueSuccess: Record<string, number> = { A: 0.7, B: 0.75 }; // the online lift is smaller than offline - typical
  type Stats = Record<"success" | "thumbs" | "safety" | "latency" | "cost", number[]>;
  const newStats = (): Stats => ({ success: [], thumbs: [], safety: [], latency: [], cost: [] });
  const stats: Record<string, Stats> = { A: newStats(), B: newStats() };
  for (const u of users) {
    const arm = armOf.get(u)!;
    const rng = mulberry32(fnv1a32("user:" + u));
    const success = rng() < trueSuccess[arm]! ? 1 : 0;
    const thumbs = rng() < (success ? 0.55 : 0.1) ? 1 : 0;
    const safety = rng() < 0.003 ? 1 : 0;
    const [outTok, callCost] = profile[arm]!;
    // latency: Haiku profile (250 ms TTFT + 3 ms/token), +-30% jitter, 5% slow tail
    const lat = (250 + 3 * outTok) * (0.7 + 0.6 * rng()) * (rng() < 0.05 ? 2 : 1);
    const cost = callCost * (0.8 + 0.4 * rng());
    const s = stats[arm]!;
    s.success.push(success);
    s.thumbs.push(thumbs);
    s.safety.push(safety);
    s.latency.push(lat);
    s.cost.push(cost);
  }
  const rate = (arm: string, k: keyof Stats): number => mean(stats[arm]![k]);
  console.log(`  ${"metric".padEnd(26)}${"A".padStart(12)}${"B".padStart(12)}${"B vs A".padStart(12)}`);
  const rows: [string, keyof Stats, (v: number) => string][] = [
    ["task success rate", "success", pct],
    ["thumbs-up rate", "thumbs", pct],
    ["safety incident rate", "safety", (v) => fmt(100 * v, 2) + "%"],
  ];
  for (const [name, k, f] of rows) {
    console.log(`  ${name.padEnd(26)}${f(rate("A", k)).padStart(12)}${f(rate("B", k)).padStart(12)}${signedPct(rate("B", k) - rate("A", k)).padStart(12)}`);
  }
  const p95: Record<string, number> = { A: percentile(stats.A!.latency, 95), B: percentile(stats.B!.latency, 95) };
  const cps: Record<string, number> = {
    A: total(stats.A!.cost) / total(stats.A!.success),
    B: total(stats.B!.cost) / total(stats.B!.success),
  };
  console.log(`  ${"p95 latency (ms)".padEnd(26)}${fmt(p95.A!, 0).padStart(12)}${fmt(p95.B!, 0).padStart(12)}${signedPct(p95.B! / p95.A! - 1).padStart(12)}`);
  console.log(`  ${"cost per successful task".padEnd(26)}${("$" + fmt(cps.A!, 7)).padStart(12)}${("$" + fmt(cps.B!, 7)).padStart(12)}${signedPct(cps.B! / cps.A! - 1).padStart(12)}`);
  const [diff, z, p, half] = twoPropZtest(total(stats.A!.success), nA, total(stats.B!.success), nB);
  console.log(`\n  primary metric (task success): diff=${signedPct(diff)}  z=${fmt(z, 2)}  p=${fmtP(p)}  95% CI [${signedPct(diff - half)}, ${signedPct(diff + half)}]`);
  const [d2, z2, p2, h2] = twoPropZtest(total(stats.A!.thumbs), nA, total(stats.B!.thumbs), nB);
  console.log(`  secondary (thumbs-up):         diff=${signedPct(d2)}  z=${fmt(z2, 2)}  p=${fmtP(p2)}  95% CI [${signedPct(d2 - h2)}, ${signedPct(d2 + h2)}]`);
  const [, zs] = twoPropZtest(total(stats.A!.safety), nA, total(stats.B!.safety), nB);
  const safetyPOneSided = 1 - normCdf(zs);
  const guardrails: [string, boolean, string][] = [
    ["p95 latency regression <= +10%", p95.B! / p95.A! - 1 <= 0.1, signedPct(p95.B! / p95.A! - 1)],
    ["cost per success regression <= +15%", cps.B! / cps.A! - 1 <= 0.15, signedPct(cps.B! / cps.A! - 1)],
    ["safety incidents not significantly higher", safetyPOneSided >= 0.05, `one-sided p=${fmtP(safetyPOneSided)}`],
  ];
  console.log("  guardrails (must not regress, even if the primary metric wins):");
  for (const [name, ok, observed] of guardrails) console.log(`    [${ok ? "OK  " : "FAIL"}] ${name.padEnd(42)} observed ${observed}`);
  const primaryWin = p < 0.05 && diff > 0;
  const allOk = guardrails.every(([, ok]) => ok);
  const decision = primaryWin && allOk ? "SHIP B" : primaryWin ? "DO NOT SHIP - primary metric wins but guardrails fail; iterate" : "DO NOT SHIP - no significant win";
  console.log(`  DECISION: ${decision}`);

  // -- 5 ------------------------------------------------------------------
  section("5. A/A tests: fixed horizon vs peeking");
  const experiments = 200;
  const perArm = 1000;
  const looks = 10;
  let fpFixed = 0;
  let fpPeek = 0;
  for (let e = 0; e < experiments; e++) {
    const rng = mulberry32(1000 + e);
    let xa = 0;
    let xb = 0;
    let peeked = false;
    for (let i = 1; i <= perArm; i++) {
      xa += rng() < 0.7 ? 1 : 0;
      xb += rng() < 0.7 ? 1 : 0;
      if (i % Math.floor(perArm / looks) === 0) {
        const pv = twoPropZtest(xa, i, xb, i)[2];
        if (pv < 0.05) peeked = true;
        if (i === perArm && pv < 0.05) fpFixed += 1;
      }
    }
    fpPeek += peeked ? 1 : 0;
  }
  console.log(`  ${experiments} simulated A/A experiments (no real difference), ${perArm} users per arm`);
  console.log(`  false-positive rate, test once at the planned sample size: ${pct(fpFixed / experiments)} (expected ~5%, +/-3% at 200 sims)`);
  console.log(`  false-positive rate, stop at the first of ${looks} peeks with p<0.05: ${pct(fpPeek / experiments)}`);
  console.log("  Fix: pre-register the sample size and test once, or use a sequential method (alpha spending / SPRT).");

  // -- 6 ------------------------------------------------------------------
  section("6. Iteration loop: propose -> evaluate on validation -> keep or revert");
  let current = PROMPT_B;
  let curTrain = await evaluate(client, sem, current, train);
  let curVal = await evaluate(client, sem, current, val);
  const failing = train.filter((_, i) => !curTrain[i]!.correct);
  const analysisPrompt =
    "Current prompt:\n" + current + "\n\nFailing TRAIN tickets:\n" +
    failing.map((t) => `- [${t.id}] ${t.text} | predicted=wrong | gold=${t.label}`).join("\n") +
    "\n\nPropose independent candidate edits as prompt blocks.";
  const proposal = await client.messages.parse({
    model: ANALYZER_MODEL,
    max_tokens: 2048,
    system: "You improve prompts for a ticket router. Only TRAIN failures are shown to you.",
    messages: [{ role: "user", content: analysisPrompt }],
    output_config: { format: zodOutputFormat(CandidateList) },
  });
  const cands = proposal.parsed_output?.candidates ?? [];
  console.log(`  start: variant B  train=${pct(acc(curTrain))} validation=${pct(acc(curVal))} out_tok=${fmt(mean(curVal.map((o) => o.outputTokens)), 1)}`);
  console.log(`  analyzer (${ANALYZER_MODEL}) saw ${failing.length} train failures and proposed ${cands.length} edits`);
  console.log("  keep rule: validation +2 cases or more with output tokens <= 1.2x, OR validation not worse with output tokens <= 0.7x");
  console.log(`  ${"#".padEnd(3)}${"candidate".padEnd(27)}${"train".padStart(8)}${"valid".padStart(8)}${"d_val".padStart(7)}${"out_tok".padStart(9)}  decision`);
  for (const [idx, cand] of cands.entries()) {
    const trial = current + "\n" + cand.block;
    const tr = await evaluate(client, sem, trial, train);
    const va = await evaluate(client, sem, trial, val);
    const dVal = sumCorrect(va) - sumCorrect(curVal);
    const tokRatio = mean(va.map((o) => o.outputTokens)) / mean(curVal.map((o) => o.outputTokens));
    const keep = (dVal >= 2 && tokRatio <= 1.2) || (dVal >= 0 && tokRatio <= 0.7);
    let why: string;
    if (keep) why = dVal >= 2 ? "KEEP - quality gain" : "KEEP - same quality, far fewer tokens";
    else if (acc(tr) - acc(curTrain) >= 0.05 && dVal < 2) why = "REVERT - train up, validation flat: overfit";
    else why = "REVERT - no measurable gain";
    const dStr = (dVal >= 0 ? "+" : "") + String(dVal);
    console.log(
      `  ${String(idx + 1).padEnd(3)}${cand.name.padEnd(27)}${pct(acc(tr)).padStart(8)}${pct(acc(va)).padStart(8)}${dStr.padStart(7)}${fmt(mean(va.map((o) => o.outputTokens)), 1).padStart(9)}  ${why}`,
    );
    if (keep) [current, curTrain, curVal] = [trial, tr, va];
  }
  console.log("  (n=40 per split: a 2-case threshold is illustrative; real loops need more cases or reps to clear noise.)");

  // -- 7 ------------------------------------------------------------------
  section("7. Final report on the held-out TEST split (never used for decisions)");
  const finalVariants: [string, string][] = [["A (baseline)", PROMPT_A], ["B (defs)", PROMPT_B], ["C (hill-climbed)", current]];
  const testOut = new Map<string, Outcome[]>();
  for (const [name, prompt] of finalVariants) testOut.set(name, await evaluate(client, sem, prompt, test));
  console.log(`  ${"variant".padEnd(18)}${"test acc".padStart(10)}${"out_tok".padStart(9)}${"cost/ticket".padStart(13)}${"latency*".padStart(10)}`);
  for (const [name] of finalVariants) {
    const o = testOut.get(name)!;
    const lat = 250 + 3 * mean(o.map((x) => x.outputTokens));
    console.log(
      `  ${name.padEnd(18)}${pct(acc(o)).padStart(10)}${fmt(mean(o.map((x) => x.outputTokens)), 1).padStart(9)}${("$" + fmt(mean(o.map((x) => x.cost)), 7)).padStart(13)}${fmt(lat, 0).padStart(10)}`,
    );
  }
  const aC = testOut.get("A (baseline)")!.map((x) => x.correct);
  const cC = testOut.get("C (hill-climbed)")!.map((x) => x.correct);
  [lo, hi] = pairedBootstrap(aC, cC, 2000, 11);
  const bb = aC.filter((x, i) => x && !cC[i]).length;
  const cc = aC.filter((x, i) => !x && cC[i]).length;
  const [pe] = mcnemar(bb, cc);
  console.log("  * modeled latency = 250 ms + 3 ms per output token (Haiku profile); use measured wall-clock in live runs");
  console.log(`  C vs A on test: ${signedPct(mean(cC) - mean(aC))}, paired 95% CI [${signedPct(lo)}, ${signedPct(hi)}], McNemar exact p=${fmtP(pe)}`);
  console.log("  Recommendation: run an online A/B of C vs A - C keeps B's quality gain without B's latency and cost regressions.");
}

await main();
