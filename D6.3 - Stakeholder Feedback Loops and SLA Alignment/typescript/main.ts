/**
 * D6.3 - Stakeholder Feedback Loops and SLA Alignment
 *
 * Service: the Claims Correspondence Copilot (Harbor Mutual, fictional), one month after go-live.
 *
 * What this lab builds:
 *   (a) an SLI / SLO / SLA catalog for a Claude-powered service: availability, latency,
 *       eval-based quality (LLM judge on a sample, reported with a confidence interval),
 *       cost per completed task and rejection/escalation rate
 *   (b) a seeded simulation of a month of request logs -> SLIs, error budget, multiwindow
 *       burn-rate alerts, SLA breach detection and service credits (with SLA exclusions)
 *   (c) a stakeholder feedback loop: Claude (Haiku, structured output) classifies feedback,
 *       code corroborates themes against telemetry, maps them to backlog priorities and
 *       closes the loop with "you said, we did" release notes
 *   (d) expectation alignment: a monthly service review with SLO attainment, top issues,
 *       what changed, what's next, known limitations (measured) and what the system will NOT do
 *
 * Principle: numbers come from telemetry and code; Claude classifies and writes the narrative.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { MODEL_HAIKU, MODEL_OPUS, PRICING, banner, getClient, section, usageLine } from "../../shared/typescript/claudeClient.ts";
import type { MockRequest } from "../../shared/typescript/claudeClient.ts";

const LAB = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const DATA = path.join(LAB, "data");
const OUT = path.join(LAB, "labs_output");

const THEMES = [
  "availability", "latency", "verbosity_tone", "factual_error", "missing_context", "policy_risk",
  "formatting", "feature_request", "praise",
] as const;

const FeedbackClassification = z.object({
  items: z.array(
    z.object({
      id: z.string(),
      theme: z.enum(THEMES),
      severity: z.number().int().min(1).max(4).describe("1 = cosmetic ... 4 = compliance risk or customer harm"),
      sentiment: z.enum(["positive", "neutral", "negative"]),
      component: z.enum(["prompt", "retrieval", "integration", "ui", "model", "process"]),
      summary: z.string(),
    }),
  ),
});
type FeedbackLabel = z.infer<typeof FeedbackClassification>["items"][number];

const ServiceReviewNarrative = z.object({
  executive_summary: z.string(),
  highlights: z.array(z.string()),
  concerns: z.array(z.string()),
  expectation_notes: z.array(z.string()),
  next_month_focus: z.array(z.string()),
});
type ServiceReviewNarrative = z.infer<typeof ServiceReviewNarrative>;

const CLASSIFY_SYSTEM = `You triage user feedback for a Claude-powered letter-drafting service.
For each item return: one theme (availability, latency, verbosity_tone, factual_error, missing_context,
policy_risk, formatting, feature_request, praise), severity 1-4 (1 cosmetic, 4 compliance risk or customer harm),
sentiment, the component most likely responsible, and a one-line neutral summary. Return every id exactly once.`;

const NARRATIVE_SYSTEM = `You write the narrative sections of a monthly service review for business stakeholders.
Be candid: lead with what missed its target and why, then what was done about it.
Use ONLY numbers that appear in the FACTS block. Describe LLM quality probabilistically
(sample-based estimate with a confidence interval), never as a guarantee.`;

function responder(req: MockRequest): string {
  const mock = path.join(DATA, "mock_responses");
  if (req.systemText.includes("triage user feedback")) return fs.readFileSync(path.join(mock, "feedback_classification.json"), "utf-8");
  if (req.systemText.includes("monthly service review")) return fs.readFileSync(path.join(mock, "service_review_narrative.json"), "utf-8");
  return "{}";
}

// ---------------------------------------------------------------------------
// Deterministic helpers (identical in Python)
// ---------------------------------------------------------------------------
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function nearestRank(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  const k = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.max(1, k) - 1]!;
}

/** Wilson score interval: honest uncertainty for a sampled pass rate (works for small n and p near 1). */
function wilson(k: number, n: number, z = 1.96): [number, number, number] {
  if (n === 0) return [0, 0, 1];
  const p = k / n;
  const denom = 1 + (z * z) / n;
  const center = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [p, Math.max(0, center - half), Math.min(1, center + half)];
}

const pct = (x: number, d = 1): string => `${(x * 100).toFixed(d)}%`;
const commas = (s: string): string => s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
const intc = (n: number): string => commas(String(n));
function money(x: number, d = 2): string {
  const [i, f] = x.toFixed(d).split(".");
  return "$" + commas(i!) + (f !== undefined ? "." + f : "");
}
const g = (v: number): string => String(Number(v.toPrecision(6)));
const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const lineCount = (s: string): number => s.split("\n").length - (s.endsWith("\n") ? 1 : 0);
const listOrNone = (xs: string[]): string => (xs.length ? `[${xs.map((x) => `'${x}'`).join(", ")}]` : "none");

function spark(values: number[], lo: number, hi: number): string {
  const levels = " .:-=+*#";
  let out = "";
  for (const v of values) {
    const idx = hi > lo ? Math.trunc(((v - lo) / (hi - lo)) * (levels.length - 1)) : 0;
    out += levels[Math.max(0, Math.min(levels.length - 1, idx))];
  }
  return out;
}

const NUMBER_PATTERNS = [/\$\s?(\d[\d,]*(?:\.\d+)?)/gi, /(\d+(?:\.\d+)?)\s?%/gi, /(\d+(?:\.\d+)?)\s?pp\b/gi, /(\d+(?:\.\d+)?)\s?(?:s|sec|seconds)\b/gi];

function ungrounded(text: string, facts: number[]): string[] {
  const bad: string[] = [];
  for (const pat of NUMBER_PATTERNS) {
    for (const m of text.matchAll(pat)) {
      const v = Number(m[1]!.replace(/,/g, ""));
      if (!facts.some((f) => Math.abs(v - f) <= Math.max(0.05, 0.005 * Math.abs(f)))) bad.push(m[1]!);
    }
  }
  return bad;
}

// ---------------------------------------------------------------------------
// Data shapes
// ---------------------------------------------------------------------------
interface Slo {
  id: string; name: string; kind: string; sli: string; window: string;
  target?: number; target_max_usd?: number; target_max?: number; threshold_s?: number; min_samples?: number;
  sla: { target?: number; credits?: { below: number; credit_pct: number }[]; metric?: string; max_s?: number; credit_pct?: number } | null;
}
interface Catalog {
  service: string; provider: string; customer: string;
  period: { label: string; month: string; days: number; start_weekday: number };
  business_hours: { open: number; close: number; tz: string };
  slos: Slo[];
  sla_terms: { monthly_fee_usd: number; credit_cap_pct: number; exclusions: string[] };
  error_budget_policy: { id: string; trigger: string; action: string }[];
  burn_alert_guard: { min_requests: number; min_failures: number };
  burn_alerts: { name: string; severity: string; long_h: number; short_h: number; threshold: number }[];
  judge: { model: string; sample_rate: number; tokens_per_judgement: { input: number; output: number }; calibration: string };
}
interface Behaviour { actions: Record<string, number>; judge_pass: number; factual_error: number }
interface Simulation {
  seed: number; model: string; weekday_requests: number; weekend_requests: number; jitter: number;
  baseline: Behaviour & {
    p_client_error: number; p_server_error: number; p_timeout: number; p_throttled: number;
    latency_median_s: number; latency_sigma: number; input_tokens: number; output_tokens: number;
    cache_read_tokens: number; cache_write_tokens: number;
  };
  incident: { id: string; day: number; start_hour: number; hours: number; error_rate: number; cause: string; sla_excluded: boolean };
  prompt_change: Behaviour & { version: string; start: [number, number]; end: [number, number]; latency_mult: number; output_mult: number };
}
interface FeedbackItem { id: string; day: number; channel: string; submitter: string; role: string; text: string }
interface Release { version: string; day: number; type: string; summary: string; addresses_themes: string[] }
interface Releases {
  releases: Release[];
  theme_backlog: Record<string, { id: string; title: string; effort: number }>;
  risk_override_themes: string[];
}
interface Limitations { will_not: string[]; known_limitations: { id: string; title: string; detail: string; metric: string | null }[] }

interface Req {
  day: number; hour: number; status: string; cause: string; latencyS: number; outputTokens: number; cost: number;
  action: string; judged: boolean; score: number; factual: boolean; v14: boolean;
}
interface Stats {
  total: number; valid: number; good: number; bad: number; availability: number; ok: number; fastShare: number;
  p50: number; p95: number; p99: number; judged: number; passed: number; qPoint: number; qLow: number; qHigh: number;
  factual: number; spend: number; completed: number; costPerLetter: number; escalationRate: number; meanOutTokens: number;
}
type Episode = [string, string, string, string, number, number];
type ScoreRow = [string, string, string, string, string];
interface ThemeRow { theme: string; n: number; sev: number; corr: number; hasSignal: boolean; themeCorroborated: boolean; open: FeedbackLabel[]; submitters: number }
interface BacklogItem { theme: string; id: string; title: string; effort: number; score: number; override: boolean; items: string[]; priority: string }

// ---------------------------------------------------------------------------
// Simulation: one record per drafting request
// ---------------------------------------------------------------------------
function simulate(sim: Simulation, cat: Catalog): Req[] {
  const rng = mulberry32(sim.seed);
  const b = sim.baseline;
  const inc = sim.incident;
  const pc = sim.prompt_change;
  const openH = cat.business_hours.open;
  const closeH = cat.business_hours.close;
  const price = PRICING[sim.model]!;
  const startKey = pc.start[0] * 100 + pc.start[1];
  const endKey = pc.end[0] * 100 + pc.end[1];
  const sampleRate = cat.judge.sample_rate;
  const reqs: Req[] = [];
  for (let day = 1; day <= cat.period.days; day++) {
    const weekday = (cat.period.start_weekday + day - 1) % 7; // 0 = Monday
    const base = weekday >= 5 ? sim.weekend_requests : sim.weekday_requests;
    const n = base + Math.floor(rng() * (2 * sim.jitter + 1)) - sim.jitter;
    for (let i = 0; i < n; i++) {
      const u: number[] = [];
      for (let k = 0; k < 23; k++) u.push(rng()); // fixed draw count per request keeps both languages in lockstep
      const hour = openH + Math.floor(u[0]! * (closeH - openH));
      const key = day * 100 + hour;
      const v14 = startKey <= key && key < endKey;
      const inIncident = day === inc.day && inc.start_hour <= hour && hour < inc.start_hour + inc.hours;
      const p: Behaviour = v14 ? pc : b;
      let status = "ok";
      let cause = "service";
      if (u[1]! < b.p_client_error) status = "client_error";
      else if (inIncident && u[2]! < inc.error_rate) {
        status = "timeout";
        cause = "dependency";
      } else if (u[3]! < b.p_server_error) status = "server_error";
      else if (u[3]! < b.p_server_error + b.p_timeout) status = "timeout";
      else if (u[3]! < b.p_server_error + b.p_timeout + b.p_throttled) status = "throttled";
      if (status !== "ok") {
        reqs.push({ day, hour, status, cause, latencyS: 0, outputTokens: 0, cost: 0, action: "", judged: false, score: 0, factual: false, v14 });
        continue;
      }
      // Approx. log-normal latency using only + and * (Irwin-Hall normal, 2nd-order exp),
      // so Python and TypeScript produce bit-identical floats.
      let zsum = 0;
      for (let k = 4; k < 16; k++) zsum += u[k]!;
      const z = zsum - 6.0;
      const x = b.latency_sigma * z;
      const latency = b.latency_median_s * (1 + x + (x * x) / 2) * (v14 ? pc.latency_mult : 1.0);
      const inTok = Math.floor(b.input_tokens * (0.8 + 0.4 * u[16]!) + 0.5);
      const outTok = Math.floor(b.output_tokens * (0.8 + 0.4 * u[17]!) * (v14 ? pc.output_mult : 1.0) + 0.5);
      const cost =
        (inTok * price.input + b.cache_read_tokens * price.cacheRead + b.cache_write_tokens * price.cacheWrite + outTok * price.output) /
        1_000_000;
      let acc = 0;
      let action = "escalated";
      for (const [name, prob] of Object.entries(p.actions)) {
        acc += prob;
        if (u[18]! < acc) {
          action = name;
          break;
        }
      }
      const judged = u[19]! < sampleRate;
      let score = 0;
      let factual = false;
      if (judged) {
        if (u[20]! < p.judge_pass) score = u[21]! < 0.55 ? 5 : 4;
        else score = u[21]! < 0.7 ? 3 : u[21]! < 0.9 ? 2 : 1;
        factual = u[22]! < p.factual_error;
      }
      reqs.push({ day, hour, status, cause, latencyS: latency, outputTokens: outTok, cost, action, judged, score, factual, v14 });
    }
  }
  return reqs;
}

function stats(reqs: Req[], thresholdS: number, judgeCost: number): Stats {
  const valid = reqs.filter((r) => r.status !== "client_error");
  const good = valid.filter((r) => r.status === "ok");
  const lat = good.map((r) => r.latencyS).sort((a, b) => a - b);
  const judged = good.filter((r) => r.judged);
  const passed = judged.filter((r) => r.score >= 4).length;
  const q = wilson(passed, judged.length);
  let spend = 0;
  for (const r of good) spend += r.cost;
  spend += judgeCost * judged.length;
  const completed = good.filter((r) => r.action === "approved" || r.action === "edited").length;
  const esc = good.filter((r) => r.action === "rejected" || r.action === "escalated").length;
  let outSum = 0;
  for (const r of good) outSum += r.outputTokens;
  return {
    total: reqs.length, valid: valid.length, good: good.length, bad: valid.length - good.length,
    availability: valid.length ? good.length / valid.length : 1, ok: good.length,
    fastShare: lat.length ? lat.filter((v) => v <= thresholdS).length / lat.length : 1,
    p50: nearestRank(lat, 50), p95: nearestRank(lat, 95), p99: nearestRank(lat, 99),
    judged: judged.length, passed, qPoint: q[0], qLow: q[1], qHigh: q[2],
    factual: judged.filter((r) => r.factual).length, spend, completed,
    costPerLetter: completed ? spend / completed : 0,
    escalationRate: good.length ? esc / good.length : 0,
    meanOutTokens: good.length ? outSum / good.length : 0,
  };
}

function qualityStatus(s: Stats, target: number, minN: number): string {
  if (s.judged < minN) return "LOW N";
  if (s.qLow >= target) return "MET";
  if (s.qPoint >= target) return "AT RISK";
  return "MISSED";
}

async function main(): Promise<void> {
  banner("D6.3 - Stakeholder Feedback Loops and SLA Alignment");
  const read = <T>(f: string): T => JSON.parse(fs.readFileSync(path.join(DATA, f), "utf-8")) as T;
  const cat = read<Catalog>("slo_catalog.json");
  const sim = read<Simulation>("simulation.json");
  const feedback = read<{ items: FeedbackItem[] }>("feedback.json").items;
  const rel = read<Releases>("releases.json");
  const lim = read<Limitations>("limitations.json");
  const slo = new Map(cat.slos.map((s) => [s.kind, s]));
  const sAv = slo.get("availability")!;
  const sLat = slo.get("latency")!;
  const sQ = slo.get("quality")!;
  const sCost = slo.get("cost")!;
  const sEsc = slo.get("escalation")!;
  const month = cat.period.month;
  const client = getClient(responder);

  // -- 1. Catalog -----------------------------------------------------------------
  section("1. SLI / SLO / SLA catalog for a Claude-powered service");
  for (const s of cat.slos) {
    const target =
      s.target !== undefined ? pct(s.target) : s.target_max_usd !== undefined ? `<= $${s.target_max_usd.toFixed(2)}` : `<= ${pct(s.target_max!, 0)}`;
    const sla =
      s.sla === null ? "internal SLO only" : s.sla.target !== undefined ? `SLA ${pct(s.sla.target)}` : `SLA ${s.sla.metric} <= ${s.sla.max_s} s`;
    console.log(`  ${s.id} ${s.name.padEnd(30)} target ${target.padEnd(9)} | ${sla}`);
    console.log(`        SLI: ${s.sli}`);
  }
  console.log("  Why quality is different: an LLM has no single 'correct' output, so quality is an eval-based SLI -");
  console.log("  a calibrated judge scores a random sample against a rubric - and is reported as an estimate with");
  console.log("  a confidence interval. Promise 'at least 90% of sampled drafts meet the rubric', never 'always correct'.");

  // -- 2. Simulation --------------------------------------------------------------
  section("2. Simulated month of request logs (seeded)");
  const reqs = simulate(sim, cat);
  const jt = cat.judge.tokens_per_judgement;
  const jp = PRICING[cat.judge.model]!;
  const judgeCost = (jt.input * jp.input + jt.output * jp.output) / 1_000_000;
  const counts = new Map<string, number>();
  for (const r of reqs) counts.set(r.status, (counts.get(r.status) ?? 0) + 1);
  console.log(
    `  ${intc(reqs.length)} drafting requests over ${cat.period.days} days (${cat.period.label}), ` +
      `business hours ${cat.business_hours.open}:00-${cat.business_hours.close}:00 ${cat.business_hours.tz}`,
  );
  console.log("  Status mix: " + [...counts].sort((a, b) => cmp(a[0], b[0])).map(([k, v]) => `${k}=${v}`).join(", "));
  const inc = sim.incident;
  const pc = sim.prompt_change;
  console.log(`  Injected: ${inc.id} on ${month} ${inc.day} ${inc.start_hour}:00 for ${inc.hours} h (${inc.cause})`);
  console.log(
    `  Injected: prompt ${pc.version} live ${month} ${pc.start[0]} ${pc.start[1]}:00 -> ${month} ${pc.end[0]} ` +
      `${pc.end[1]}:00 (longer outputs, slower, lower judge scores)`,
  );

  // -- 3. SLIs and attainment --------------------------------------------------------
  section("3. SLIs and SLO attainment (month and weeks)");
  const thr = sLat.threshold_s!;
  const m = stats(reqs, thr, judgeCost);
  const qTarget = sQ.target!;
  const minN = sQ.min_samples!;
  const qStatus = qualityStatus(m, qTarget, minN);
  const scorecard: ScoreRow[] = [
    ["SLO-1", "Availability", pct(m.availability, 2), pct(sAv.target!), m.availability >= sAv.target! ? "MET" : "MISSED"],
    ["SLO-2", `Drafts <= ${thr} s`, pct(m.fastShare), pct(sLat.target!), m.fastShare >= sLat.target! ? "MET" : "MISSED"],
    ["SLO-3", "Quality (judge >= 4/5)", `${pct(m.qPoint)} [${pct(m.qLow)}-${pct(m.qHigh)}] n=${m.judged}`, pct(qTarget), qStatus],
    ["SLO-4", "Cost per completed letter", `$${m.costPerLetter.toFixed(4)}`, `<= $${sCost.target_max_usd!.toFixed(2)}`,
      m.costPerLetter <= sCost.target_max_usd! ? "MET" : "MISSED"],
    ["SLO-5", "Rejection + escalation", pct(m.escalationRate), `<= ${pct(sEsc.target_max!, 0)}`, m.escalationRate <= sEsc.target_max! ? "MET" : "MISSED"],
  ];
  console.log(`  ${"SLO".padEnd(6)}${"Objective".padEnd(27)}${"Actual".padEnd(31)}${"Target".padEnd(9)}Status`);
  for (const row of scorecard) console.log(`  ${row[0].padEnd(6)}${row[1].padEnd(27)}${row[2].padEnd(31)}${row[3].padEnd(9)}${row[4]}`);
  console.log(
    `  Latency distribution: p50 ${m.p50.toFixed(2)} s, p95 ${m.p95.toFixed(2)} s, p99 ${m.p99.toFixed(2)} s; ` +
      `spend ${money(m.spend)} for ${intc(m.completed)} completed letters`,
  );

  const weeks: [number, number][] = [[1, 7], [8, 14], [15, 21], [22, 28], [29, 30]];
  const weekly: [string, Stats][] = [];
  console.log(
    `\n  ${"Week".padEnd(12)}${"avail".padStart(8)}${"<=8s".padStart(8)}${"p95".padStart(7)}${"quality [95% CI]".padStart(24)}` +
      `${"n".padStart(5)}${"rej+esc".padStart(9)}${"$/letter".padStart(10)}`,
  );
  for (const [a, b] of weeks) {
    const w = stats(reqs.filter((r) => a <= r.day && r.day <= b), thr, judgeCost);
    const label = `${month} ${a}-${b}`;
    weekly.push([label, w]);
    const flags: string[] = [];
    if (w.fastShare < sLat.target!) flags.push("latency");
    if (w.qPoint < qTarget) flags.push("quality");
    if (w.escalationRate > sEsc.target_max!) flags.push("rejections");
    if (w.availability < sAv.target!) flags.push("availability");
    console.log(
      `  ${label.padEnd(12)}${pct(w.availability, 2).padStart(8)}${pct(w.fastShare).padStart(8)}${w.p95.toFixed(2).padStart(6)}s` +
        `${`${pct(w.qPoint)} [${pct(w.qLow)}-${pct(w.qHigh)}]`.padStart(24)}${String(w.judged).padStart(5)}` +
        `${pct(w.escalationRate).padStart(9)}${w.costPerLetter.toFixed(4).padStart(10)}` + (flags.length ? `  <- ${flags.join(", ")}` : ""),
    );
  }
  const daily: Stats[] = [];
  for (let d = 1; d <= cat.period.days; d++) daily.push(stats(reqs.filter((r) => r.day === d), thr, judgeCost));
  console.log(`\n  Daily p95 latency   ${spark(daily.map((s) => s.p95), 5.0, 10.0)}  (scale 5 s ' ' .. 10 s '#')`);
  console.log(`  Daily availability  ${spark(daily.map((s) => s.availability), 0.95, 1.0)}  (scale 95% ' ' .. 100% '#')`);
  console.log(`                      ${Array.from({ length: cat.period.days }, (_, i) => String((i + 1) % 10)).join("")}  (day of month)`);

  // -- 4. Error budget + burn alerts -------------------------------------------------
  section("4. Error budget and multiwindow burn-rate alerts");
  const avTarget = sAv.target!;
  const allowed = (1 - avTarget) * m.valid;
  const consumed = m.bad / allowed;
  const incidentBad = reqs.filter((r) => r.cause === "dependency").length;
  const incidentShare = incidentBad / allowed;
  console.log(`  Budget: (1 - ${avTarget}) x ${intc(m.valid)} valid requests = ${allowed.toFixed(1)} failed requests allowed`);
  console.log(
    `  Spent: ${m.bad} failed -> ${pct(consumed, 0)} of budget (${consumed >= 1 ? "EXHAUSTED" : "remaining " + pct(1 - consumed, 0)}); ` +
      `${inc.id} alone used ${pct(incidentShare, 0)}`,
  );
  const hoursList: [number, number][] = [];
  for (let d = 1; d <= cat.period.days; d++) for (let h = cat.business_hours.open; h < cat.business_hours.close; h++) hoursList.push([d, h]);
  const idxOf = new Map(hoursList.map(([d, h], i) => [d * 100 + h, i]));
  const vSeq = hoursList.map(() => 0);
  const bSeq = hoursList.map(() => 0);
  for (const r of reqs) {
    if (r.status === "client_error") continue;
    const i = idxOf.get(r.day * 100 + r.hour)!;
    vSeq[i]! += 1;
    if (r.status !== "ok") bSeq[i]! += 1;
  }
  const guard = cat.burn_alert_guard;
  /** [valid, failed, burn rate] over the last `length` business hours ending at bucket i. */
  const windowAt = (i: number, length: number): [number, number, number] => {
    const lo = Math.max(0, i - length + 1);
    let v = 0;
    let bd = 0;
    for (let k = lo; k <= i; k++) {
      v += vSeq[k]!;
      bd += bSeq[k]!;
    }
    return [v, bd, v ? bd / v / (1 - avTarget) : 0];
  };
  const burn = (i: number, length: number): number => windowAt(i, length)[2];
  const episodes: Episode[] = [];
  let suppressed = 0;
  for (const rule of cat.burn_alerts) {
    let runStart = -1;
    let peak = 0;
    for (let i = 0; i <= hoursList.length; i++) {
      let firing = false;
      if (i < hoursList.length && i + 1 >= rule.long_h) {
        // warm-up: evaluate only full windows
        const [vLong, fLong, bLong] = windowAt(i, rule.long_h);
        const hot = bLong >= rule.threshold && burn(i, rule.short_h) >= rule.threshold;
        // Low-traffic guard: 1 failure in a 4-request weekend hour is a 50x "burn" but not an incident.
        const enough = vLong >= guard.min_requests && fLong >= guard.min_failures;
        if (hot && !enough) suppressed += 1;
        firing = hot && enough;
      }
      if (firing) {
        if (runStart < 0) {
          runStart = i;
          peak = 0;
        }
        peak = Math.max(peak, burn(i, rule.long_h));
      } else if (runStart >= 0) {
        const [d0, h0] = hoursList[runStart]!;
        const [d1, h1] = hoursList[i - 1]!;
        episodes.push([
          rule.severity, rule.name, `${month} ${d0} ${String(h0).padStart(2, "0")}:00`, `${month} ${d1} ${String(h1 + 1).padStart(2, "0")}:00`,
          i - runStart, peak,
        ]);
        runStart = -1;
      }
    }
  }
  console.log(
    "  Rules (business-hour windows): " +
      cat.burn_alerts.map((r) => `${r.severity} ${r.name}: burn >= ${g(r.threshold)} over ${r.long_h}h and ${r.short_h}h`).join("; "),
  );
  for (const [sev, name, start, end, nH, peak] of episodes) {
    console.log(`  ${sev.padEnd(6)} ${name.padEnd(12)} ${start} -> ${end} (${nH} h firing, peak burn ${peak.toFixed(1)}x)`);
  }
  console.log(`  Suppressed ${suppressed} hot window-hours with < ${guard.min_requests} requests or < ${guard.min_failures} failures (low-traffic noise)`);
  const triggers: Record<string, boolean> = {
    budget_exhausted: consumed >= 1, incident_over_20pct: incidentShare > 0.2, quality_not_met: qStatus !== "MET",
  };
  const policyActions: [string, string][] = cat.error_budget_policy.filter((p) => triggers[p.trigger]).map((p) => [p.id, p.action]);
  console.log("  Error-budget policy triggered:");
  for (const [pid, action] of policyActions) console.log(`    ${pid}: ${action}`);

  // -- 5. SLA ----------------------------------------------------------------------
  section("5. SLA breach detection and service credits");
  const excluded = inc.sla_excluded ? incidentBad : 0;
  const slaValid = m.valid - excluded;
  const slaAvail = m.good / slaValid;
  const slaAv = sAv.sla!;
  const avCredit = Math.max(0, ...slaAv.credits!.filter((c) => slaAvail < c.below).map((c) => c.credit_pct));
  const latSla = sLat.sla!;
  const latCredit = m.p95 > latSla.max_s! ? latSla.credit_pct! : 0;
  const terms = cat.sla_terms;
  const creditPct = Math.min(terms.credit_cap_pct, avCredit + latCredit);
  const creditUsd = (terms.monthly_fee_usd * creditPct) / 100;
  console.log(`  SLA availability excludes ${excluded} requests failed by the customer-owned dependency (${inc.id}):`);
  console.log(
    `    ${intc(m.good)} / ${intc(slaValid)} = ${pct(slaAvail, 2)} vs SLA ${pct(slaAv.target!)} -> ` +
      `${slaAvail < slaAv.target! ? "BREACH" : "met"} (credit ${avCredit}%)`,
  );
  console.log(`  SLA latency: monthly p95 ${m.p95.toFixed(2)} s vs <= ${latSla.max_s} s -> ${latCredit ? "BREACH" : "met"} (credit ${latCredit}%)`);
  console.log(`  Service credit: ${creditPct}% of $${intc(terms.monthly_fee_usd)} = ${money(creditUsd)}`);
  const sloMissed = m.availability < avTarget;
  const slaBreached = slaAvail < slaAv.target!;
  if (sloMissed && !slaBreached) {
    console.log(`  Lesson: the SLO (user experience, ${pct(m.availability, 2)}) was missed while the SLA (contract, ${pct(slaAvail, 2)}) held -`);
    console.log("  the SLO is the internal early-warning line; the SLA sits below it and excludes what the provider can't control.");
  } else {
    console.log(`  SLO ${sloMissed ? "missed" : "met"}, SLA ${slaBreached ? "breached" : "met"}.`);
  }

  // -- 6. Feedback loop ----------------------------------------------------------------
  section("6. Feedback loop: Claude classifies, code corroborates and prioritizes");
  const fbText = feedback.map((f) => `${f.id} | ${month} ${f.day} | ${f.channel} | ${f.role} | ${f.text}`).join("\n");
  const resp = await client.messages.parse({
    model: MODEL_HAIKU, // high-volume classification: the small model is the right tool
    max_tokens: 4000,
    output_config: { effort: "low", format: zodOutputFormat(FeedbackClassification) },
    system: CLASSIFY_SYSTEM,
    messages: [{ role: "user", content: "Classify these feedback items:\n" + fbText }],
  });
  const labels = resp.parsed_output;
  if (!labels) throw new Error("classification did not parse");
  console.log(`  Classified ${labels.items.length} items with ${MODEL_HAIKU} | ${usageLine(resp)}`);
  const ids = labels.items.map((x) => x.id);
  const expected = feedback.map((f) => f.id);
  const dups = [...new Set(ids.filter((i) => ids.filter((j) => j === i).length > 1))].sort(cmp);
  console.log(`  Validation: missing=${listOrNone(expected.filter((i) => !ids.includes(i)))}, duplicates=${listOrNone(dups)}`);
  const byFb = new Map(feedback.map((f) => [f.id, f]));
  const labelOf = new Map(labels.items.map((x) => [x.id, x]));

  const medianTokens = daily.map((s) => s.meanOutTokens).sort((a, b) => a - b)[Math.floor(daily.length / 2)]!;
  const daysWhere = (pred: (s: Stats) => boolean): Set<number> => new Set(daily.flatMap((s, i) => (pred(s) ? [i + 1] : [])));
  const signalDays = new Map<string, Set<number>>([
    ["availability", daysWhere((s) => s.availability < avTarget)],
    ["latency", daysWhere((s) => s.p95 > thr)],
    ["verbosity_tone", daysWhere((s) => s.meanOutTokens > 1.2 * medianTokens)],
    ["factual_error", daysWhere((s) => s.factual > 0)],
  ]);
  console.log(
    "  Telemetry signal days: " +
      [...signalDays].map(([k, v]) => `${k} [${[...v].sort((a, b) => a - b).join(", ")}]`).join("; "),
  );

  const corroborated = (fid: string): boolean => {
    const days = signalDays.get(labelOf.get(fid)!.theme);
    const day = byFb.get(fid)!.day;
    return days !== undefined && (days.has(day) || days.has(day - 1));
  };
  const loopStatus = (fid: string): [string, string] => {
    const lab = labelOf.get(fid)!;
    if (lab.theme === "praise") return ["acknowledged", "-"];
    for (const r of rel.releases) {
      if (r.addresses_themes.includes(lab.theme) && r.day >= byFb.get(fid)!.day) return ["closed", `${r.version} (${month} ${r.day})`];
    }
    return ["open", rel.theme_backlog[lab.theme]!.id];
  };

  const themeRows: ThemeRow[] = [];
  for (const th of THEMES) {
    const items = labels.items.filter((x) => x.theme === th);
    if (!items.length) continue;
    const corr = items.filter((x) => corroborated(x.id)).length;
    const hasSignal = signalDays.has(th);
    themeRows.push({
      theme: th, n: items.length, sev: items.reduce((a, x) => a + x.severity, 0), corr, hasSignal,
      themeCorroborated: hasSignal && corr * 2 >= items.length,
      open: items.filter((x) => loopStatus(x.id)[0] === "open"),
      submitters: new Set(items.map((x) => byFb.get(x.id)!.submitter)).size,
    });
  }
  console.log(`\n  ${"theme".padEnd(16)}${"items".padStart(6)}${"sev".padStart(5)}${"corroborated".padStart(14)}${"open".padStart(6)}`);
  for (const t of themeRows) {
    const corrTxt = t.hasSignal ? `${t.corr}/${t.n}` : "n/a";
    console.log(`  ${t.theme.padEnd(16)}${String(t.n).padStart(6)}${String(t.sev).padStart(5)}${corrTxt.padStart(14)}${String(t.open.length).padStart(6)}`);
  }

  const backlog: BacklogItem[] = [];
  for (const t of themeRows) {
    if (t.theme === "praise" || !t.open.length) continue;
    const bl = rel.theme_backlog[t.theme]!;
    const openSev = t.open.reduce((a, x) => a + x.severity, 0);
    const uniq = new Set(t.open.map((x) => byFb.get(x.id)!.submitter)).size;
    const score = (openSev * (t.themeCorroborated ? 1.5 : 1.0) * (1 + 0.1 * (uniq - 1))) / bl.effort;
    const override = rel.risk_override_themes.includes(t.theme) && t.open.some((x) => x.severity === 4);
    backlog.push({ theme: t.theme, id: bl.id, title: bl.title, effort: bl.effort, score, override, items: t.open.map((x) => x.id), priority: "" });
  }
  const themeIdx = (th: string): number => (THEMES as readonly string[]).indexOf(th);
  backlog.sort((a, b) => (a.override ? 0 : 1) - (b.override ? 0 : 1) || b.score - a.score || themeIdx(a.theme) - themeIdx(b.theme));
  backlog.forEach((b, i) => {
    b.priority = i < 2 ? "P1" : i < 4 ? "P2" : "P3";
  });
  console.log("\n  Backlog (score = open severity x corroboration x reach / effort; compliance risk overrides):");
  for (const b of backlog) {
    console.log(
      `  ${b.priority} ${b.id} score ${b.score.toFixed(2).padStart(5)}${(b.override ? " [risk override]" : "").padEnd(16)} ` +
        `${b.title.slice(0, 70)}  <- ${b.items.join(", ")}`,
    );
  }

  // -- 7. Close the loop ---------------------------------------------------------------
  section("7. Closing the loop: release notes and 'you said, we did'");
  const closed = feedback.filter((f) => loopStatus(f.id)[0] === "closed").map((f) => f.id);
  const actionable = feedback.filter((f) => labelOf.get(f.id)!.theme !== "praise").map((f) => f.id);
  const fixedBy = (r: Release): string[] =>
    feedback.filter((f) => {
      const [st, ref] = loopStatus(f.id);
      return st === "closed" && ref === `${r.version} (${month} ${r.day})`;
    }).map((f) => f.id);
  for (const r of rel.releases) {
    const fixed = fixedBy(r);
    console.log(`  ${r.version} (${month} ${r.day}, ${r.type}): ${r.summary}`);
    if (fixed.length) console.log(`      closes feedback: ${fixed.join(", ")}`);
  }
  console.log(
    `  Loop closure: ${closed.length}/${actionable.length} actionable items closed this month ` +
      `(${pct(closed.length / actionable.length, 0)}); the rest are mapped to backlog items with an owner.`,
  );
  console.log("  'You said, we did' notices go to every submitter whose item closed (and a status note to the rest).");

  // -- 8. Monthly service review -----------------------------------------------------
  section("8. Monthly service review (Claude narrative from computed facts)");
  const w3 = weekly[2]![1];
  const factsLines = [
    `Availability ${pct(m.availability, 2)} vs SLO ${pct(avTarget)} (${sloMissed ? "MISSED" : "met"}: ${m.bad} failed of ${m.valid} valid requests); ` +
      `error budget consumed ${pct(consumed, 0)}; incident ${inc.id} used ${pct(incidentShare, 0)} of the budget`,
    `SLA availability ${pct(slaAvail, 2)} vs ${pct(slaAv.target!)} (${slaBreached ? "BREACHED" : "met"}; dependency outage excluded); service credit ${money(creditUsd)}`,
    `Latency: ${pct(m.fastShare)} of drafts within ${thr} s (SLO ${pct(sLat.target!)}, met); p50 ${m.p50.toFixed(2)} s, p95 ${m.p95.toFixed(2)} s, p99 ${m.p99.toFixed(2)} s`,
    `Week ${weekly[2]![0]}: ${pct(w3.fastShare)} within ${thr} s, p95 ${w3.p95.toFixed(2)} s, quality ${pct(w3.qPoint)}, rejections+escalations ${pct(w3.escalationRate)} (prompt ${pc.version})`,
    `Quality: ${pct(m.qPoint)} of ${m.judged} sampled drafts scored >= 4/5, 95% CI ${pct(m.qLow)}-${pct(m.qHigh)}, target ${pct(qTarget)} -> ${qStatus}`,
    `Cost per completed letter $${m.costPerLetter.toFixed(4)} (target <= $${sCost.target_max_usd!.toFixed(2)}); total spend ${money(m.spend)}`,
    `Rejection + escalation rate ${pct(m.escalationRate)} (target <= ${pct(sEsc.target_max!, 0)})`,
    `Feedback: ${feedback.length} items, ${closed.length} of ${actionable.length} actionable items closed (${pct(closed.length / actionable.length, 0)})`,
    "Top backlog: " + backlog.slice(0, 3).map((b) => `${b.priority} ${b.id} ${b.title}`).join("; "),
    "Releases: " + rel.releases.map((r) => `${r.version} ${month} ${r.day}: ${r.summary}`).join("; "),
  ];
  const fq = wilson(m.factual, m.judged);
  factsLines.push(`Factual-error flags: ${m.factual} of ${m.judged} sampled drafts (${pct(fq[0])}, 95% CI ${pct(fq[1])}-${pct(fq[2])})`);
  const factsText = factsLines.join("\n");
  const factsNum: number[] = [];
  for (const line of factsLines) {
    for (const pat of [...NUMBER_PATTERNS, /(\d[\d,]*(?:\.\d+)?)/gi]) {
      for (const mm of line.matchAll(pat)) factsNum.push(Number(mm[1]!.replace(/,/g, "")));
    }
  }
  const nresp = await client.messages.parse({
    model: MODEL_OPUS,
    max_tokens: 3000,
    thinking: { type: "adaptive" },
    output_config: { effort: "medium", format: zodOutputFormat(ServiceReviewNarrative) },
    system: NARRATIVE_SYSTEM,
    messages: [{ role: "user", content: "FACTS:\n" + factsText + "\n\nWrite the narrative sections." }],
  });
  const narrative = nresp.parsed_output;
  if (!narrative) throw new Error("narrative did not parse");
  const allText = [narrative.executive_summary, ...narrative.highlights, ...narrative.concerns, ...narrative.expectation_notes, ...narrative.next_month_focus].join("\n");
  const badNums = ungrounded(allText, factsNum);
  console.log(`  Narrative drafted | ${usageLine(nresp)}`);
  console.log(`  Grounding check: ${badNums.length ? "UNGROUNDED: " + badNums.join(", ") : "all figures trace to telemetry"}`);

  const report = renderReview({
    cat, sim, rel, lim, feedback, labelOf, m, weekly, scorecard, allowed, consumed, incidentShare, episodes, policyActions,
    slaAvail, excluded, creditPct, creditUsd, themeRows, backlog, loopStatus, narrative, fq, closed, actionable, fixedBy,
  });
  fs.mkdirSync(OUT, { recursive: true });
  const name = `service_review_${cat.period.label.replace(/ /g, "_").toLowerCase()}.md`;
  fs.writeFileSync(path.join(OUT, name), report, "utf-8");
  fs.writeFileSync(path.join(OUT, "service_review_facts.txt"), factsText + "\n", "utf-8");
  console.log(`  wrote labs_output/${name} (${lineCount(report)} lines) and labs_output/service_review_facts.txt; samples in examples/`);
}

interface ReviewInput {
  cat: Catalog; sim: Simulation; rel: Releases; lim: Limitations; feedback: FeedbackItem[]; labelOf: Map<string, FeedbackLabel>;
  m: Stats; weekly: [string, Stats][]; scorecard: ScoreRow[]; allowed: number; consumed: number; incidentShare: number;
  episodes: Episode[]; policyActions: [string, string][]; slaAvail: number; excluded: number; creditPct: number; creditUsd: number;
  themeRows: ThemeRow[]; backlog: BacklogItem[]; loopStatus: (fid: string) => [string, string]; narrative: ServiceReviewNarrative;
  fq: [number, number, number]; closed: string[]; actionable: string[]; fixedBy: (r: Release) => string[];
}

function renderReview(x: ReviewInput): string {
  const { cat, sim, rel, lim, feedback, labelOf, m, narrative, fq } = x;
  const slo = new Map(cat.slos.map((s) => [s.kind, s]));
  const month = cat.period.month;
  const inc = sim.incident;
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  a(`# Monthly Service Review - ${cat.service} - ${cat.period.label}`);
  a("");
  a(`**Provider:** ${cat.provider} | **Customer:** ${cat.customer}`);
  a("");
  a("## 1. Executive summary");
  a("");
  a(narrative.executive_summary);
  a("");
  a("**Highlights**");
  a("");
  for (const h of narrative.highlights) a(`- ${h}`);
  a("");
  a("**Concerns**");
  a("");
  for (const c of narrative.concerns) a(`- ${c}`);
  a("");
  a("## 2. SLO and SLA scorecard");
  a("");
  a("| SLO | Objective | Actual | Target | Status |");
  a("|---|---|---|---|---|");
  for (const row of x.scorecard) a(`| ${row[0]} | ${row[1]} | ${row[2]} | ${row[3]} | **${row[4]}** |`);
  a("");
  const slaAv = slo.get("availability")!.sla!;
  const latSla = slo.get("latency")!.sla!;
  a("| SLA term | Actual | Commitment | Result |");
  a("|---|---|---|---|");
  a(
    `| Availability (excl. ${x.excluded} dependency-caused failures) | ${pct(x.slaAvail, 2)} | >= ${pct(slaAv.target!)} | ` +
      `${x.slaAvail < slaAv.target! ? "BREACH" : "met"} |`,
  );
  a(`| Monthly p95 full-draft latency | ${m.p95.toFixed(2)} s | <= ${latSla.max_s} s | ${m.p95 > latSla.max_s! ? "BREACH" : "met"} |`);
  a(`| **Service credit** | ${x.creditPct}% | fee $${intc(cat.sla_terms.monthly_fee_usd)} | **${money(x.creditUsd)}** |`);
  a("");
  a("SLA exclusions: " + cat.sla_terms.exclusions.join("; ") + ".");
  a("");
  a("## 3. Error budget and incidents");
  a("");
  a(
    `- Budget: ${x.allowed.toFixed(1)} failed requests allowed (${pct(1 - slo.get("availability")!.target!)} of ${intc(m.valid)} valid); ` +
      `${m.bad} failed -> **${pct(x.consumed, 0)} consumed**.`,
  );
  a(`- ${inc.id} (${month} ${inc.day}, ${inc.start_hour}:00 for ${inc.hours} h): ${inc.cause}; used ${pct(x.incidentShare, 0)} of the budget.`);
  a("");
  a("| Alert | Rule | Window | Duration | Peak burn |");
  a("|---|---|---|---|---:|");
  for (const [sev, name, start, end, nH, peak] of x.episodes) a(`| ${sev} | ${name} | ${start} -> ${end} | ${nH} h | ${peak.toFixed(1)}x |`);
  a("");
  a("Error-budget policy actions triggered:");
  a("");
  for (const [pid, action] of x.policyActions) a(`- **${pid}** ${action}`);
  a("");
  a("## 4. Weekly trend");
  a("");
  a("| Week | Availability | Drafts <= 8 s | p95 | Quality [95% CI] (n) | Rejected + escalated | $/letter |");
  a("|---|---:|---:|---:|---|---:|---:|");
  for (const [label, w] of x.weekly) {
    a(
      `| ${label} | ${pct(w.availability, 2)} | ${pct(w.fastShare)} | ${w.p95.toFixed(2)} s | ` +
        `${pct(w.qPoint)} [${pct(w.qLow)}-${pct(w.qHigh)}] (${w.judged}) | ${pct(w.escalationRate)} | $${w.costPerLetter.toFixed(4)} |`,
    );
  }
  a("");
  a(
    `Weekly quality samples are below the ${slo.get("quality")!.min_samples}-sample minimum, so weekly quality is indicative only; ` +
      "the monthly figure is the one we commit to.",
  );
  a("");
  a("## 5. How quality is measured (and how to read it)");
  a("");
  a(`- SLI: ${slo.get("quality")!.sli}.`);
  a(`- Judge: ${cat.judge.model} on a ${pct(cat.judge.sample_rate, 0)} random sample. ${cat.judge.calibration}`);
  a("- Status rule: MET if the lower bound of the 95% Wilson interval is >= target; AT RISK if only the point estimate is; MISSED otherwise.");
  for (const e of narrative.expectation_notes) a(`- ${e}`);
  a("");
  a("## 6. Stakeholder feedback");
  a("");
  a("| Theme | Items | Severity sum | Corroborated by telemetry | Open |");
  a("|---|---:|---:|---|---:|");
  for (const t of x.themeRows) {
    const corrTxt = t.hasSignal ? `${t.corr}/${t.n}` : "n/a (no telemetry signal)";
    a(`| ${t.theme} | ${t.n} | ${t.sev} | ${corrTxt} | ${t.open.length} |`);
  }
  a("");
  a("| Feedback | Day | Theme | Sev | Summary | Loop status |");
  a("|---|---|---|---:|---|---|");
  for (const f of feedback) {
    const lab = labelOf.get(f.id)!;
    const [st, ref] = x.loopStatus(f.id);
    a(`| ${f.id} | ${month} ${f.day} | ${lab.theme} | ${lab.severity} | ${lab.summary} | ${st} ${ref !== "-" ? ref : ""}`.trimEnd() + " |");
  }
  a("");
  a("## 7. What changed this month - you said, we did");
  a("");
  for (const r of rel.releases) {
    const fixed = x.fixedBy(r);
    a(`- **${r.version}** (${month} ${r.day}, ${r.type}): ${r.summary}` + (fixed.length ? ` _You said: ${fixed.join(", ")} - we did: this release._` : ""));
  }
  a("");
  a(`Loop closure: ${x.closed.length}/${x.actionable.length} actionable items closed (${pct(x.closed.length / x.actionable.length, 0)}).`);
  a("");
  a("## 8. What's next");
  a("");
  a("| Priority | Item | Score | Feedback |");
  a("|---|---|---:|---|");
  for (const b of x.backlog) a(`| ${b.priority}${b.override ? " (risk override)" : ""} | ${b.id} ${b.title} | ${b.score.toFixed(2)} | ${b.items.join(", ")} |`);
  a("");
  for (const n of narrative.next_month_focus) a(`- ${n}`);
  a("");
  a("## 9. Known limitations (measured)");
  a("");
  for (const k of lim.known_limitations) {
    let metric = "";
    if (k.metric === "factual_error_rate") metric = ` April: ${m.factual} of ${m.judged} sampled drafts flagged (${pct(fq[0])}, 95% CI ${pct(fq[1])}-${pct(fq[2])}).`;
    else if (k.metric === "quality_ci") metric = ` April: ${pct(m.qPoint)} with a 95% CI of ${pct(m.qLow)}-${pct(m.qHigh)}.`;
    else if (k.metric === "latency_tail") metric = ` April: p50 ${m.p50.toFixed(2)} s, p99 ${m.p99.toFixed(2)} s.`;
    a(`- **${k.id} ${k.title}.** ${k.detail}${metric}`);
  }
  a("");
  a("## 10. What the system will NOT do");
  a("");
  for (const w of lim.will_not) a(`- ${w}`);
  a("");
  a("## Appendix - SLI definitions");
  a("");
  a("| SLO | SLI | Window |");
  a("|---|---|---|");
  for (const s of cat.slos) a(`| ${s.id} ${s.name} | ${s.sli} | ${s.window} |`);
  a("");
  a(
    "_Narrative sections drafted by Claude from the computed facts; every figure was checked against telemetry. " +
      "All tables are computed from the request logs by code._",
  );
  a("");
  return L.join("\n");
}

await main();
