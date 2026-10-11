/**
 * D1.6 - Aligning Solutions to Business Value Pillars
 *
 * A value calculator for three customer-support AI options, scored on five pillars:
 *
 *   efficiency      hours saved, cycle time (first response time)
 *   productivity    tickets resolved per FTE
 *   transformation  new capability / revenue (risk-adjusted)
 *   cost            Claude API cost per task (from token usage via costUsd) + infra + people -> ROI, payback
 *   performance     SLA feasibility per model choice: p95 latency, availability, quality
 *
 * Output: SLA-driven model choice, pillar table, weighted scorecard, KPI tree, and a Claude-drafted
 * executive summary. All arithmetic is plain code - the LLM only writes prose from computed numbers.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  MODEL_OPUS,
  banner,
  costUsd,
  getClient,
  section,
  textBlock,
  textOf,
  usageLine,
  type Json,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");

interface Option {
  id: string;
  name: string;
  candidates: string[];
  calls_per_ticket: number;
  pilot_usage: Json;
  deflection: number;
  aht_reduction: number;
  frt_improvement: number;
  build_usd: number;
  infra_usd_month: number;
  people_usd_month: number;
  new_value_usd_year: number;
  new_capability: string;
  reply_tokens: number;
  tool_ms: number;
  architecture: string[];
  model_fallback: boolean;
  sla: { p95_latency_s: number; availability: number; quality: number };
  pilot_quality: Record<string, number>;
}
interface ValueModel {
  baseline: { tickets_per_month: number; aht_min: number; loaded_cost_per_hour: number; productive_hours_per_fte_month: number; frt_hours: number; csat: number };
  assumptions: { realization_rate: number; transformation_confidence: number; latency_samples: number; latency_seed: number };
  latency_profiles_ms: Record<string, [number, number]>;
  component_availability: Record<string, number>;
  weights: Record<string, number>;
  options: Option[];
}

const VM: ValueModel = JSON.parse(readFileSync(path.join(DATA, "value_model.json"), "utf-8"));
const BASE = VM.baseline;
const ASSUME = VM.assumptions;
const SHORT: Record<string, string> = { "claude-haiku-5-5": "Haiku", "claude-sonnet-5-5": "Sonnet", "claude-opus-5-5": "Opus" };
const OPTIONS = new Map(VM.options.map((o) => [o.id, o]));

const usd = (x: number): string => "$" + Math.round(x).toLocaleString("en-US");
const pct = (x: number): string => `${(x * 100).toFixed(0)}%`;
const n0 = (x: number): string => Math.round(x).toLocaleString("en-US");

// ---------------------------------------------------------------------------
// Performance SLAs: deterministic latency simulation + composite availability
// ---------------------------------------------------------------------------
/** Tiny linear congruential generator: identical sequence in Python and TypeScript. */
class LCG {
  state: number;
  constructor(seed: number) {
    this.state = seed % 4294967296;
  }
  next(): number {
    this.state = (1664525 * this.state + 1013904223) % 4294967296;
    return this.state / 4294967296;
  }
}

/**
 * p50/p95 seconds for one customer-visible reply: TTFT + generation + tool time, with tail noise.
 * Same seed for every candidate (common random numbers) so differences come from the model, not noise.
 * Planning profile only - replace with your own load-test measurements (D3.3, D6.3).
 */
function latencyPercentiles(model: string, replyTokens: number, toolMs: number): [number, number] {
  const [ttft, perTok] = VM.latency_profiles_ms[model]!;
  const base = ttft + perTok * replyTokens + toolMs;
  const rng = new LCG(ASSUME.latency_seed);
  const samples: number[] = [];
  for (let i = 0; i < ASSUME.latency_samples; i++) {
    const u1 = rng.next();
    const u2 = rng.next();
    const u3 = rng.next();
    const factor = 0.85 + 0.3 * u1 + (u2 < 0.1 ? 0.9 * u3 : 0.0); // 10% of calls hit a slow tail
    samples.push(base * factor);
  }
  samples.sort((a, b) => a - b);
  const n = samples.length;
  const p50 = samples[Math.floor((50 * n + 99) / 100) - 1]!;
  const p95 = samples[Math.floor((95 * n + 99) / 100) - 1]!; // nearest-rank percentile
  return [p50 / 1000, p95 / 1000];
}

/** Serial chain = product of availabilities; a redundant model path = 1 - (1 - a)^2. */
function compositeAvailability(option: Option, withFallback: boolean): number {
  let avail = 1.0;
  for (const comp of option.architecture) {
    let a = VM.component_availability[comp]!;
    if (comp === "model_api" && withFallback) a = 1 - (1 - a) ** 2; // e.g. second region / second platform for the same model
    avail *= a;
  }
  return avail;
}

interface SlaResult {
  model: string;
  p50: number;
  p95: number;
  availability: number;
  quality: number;
  passes: Record<"latency" | "availability" | "quality", boolean>;
  allPass: boolean;
}

function checkSla(option: Option, model: string): SlaResult {
  const sla = option.sla;
  const [p50, p95] = latencyPercentiles(model, option.reply_tokens, option.tool_ms);
  const availability = compositeAvailability(option, option.model_fallback);
  const quality = option.pilot_quality[model]!;
  const passes = { latency: p95 <= sla.p95_latency_s, availability: availability >= sla.availability, quality: quality >= sla.quality };
  return { model, p50, p95, availability, quality, passes, allPass: Object.values(passes).every(Boolean) };
}

// ---------------------------------------------------------------------------
// Value pillars
// ---------------------------------------------------------------------------
interface Pillars {
  optionId: string;
  model: string;
  hoursSavedMonth: number;
  frtAfterH: number;
  ahtAfterMin: number;
  ticketsPerFteBefore: number;
  ticketsPerFteAfter: number;
  productivityUplift: number;
  newValueRiskAdjYear: number;
  apiCostPerTicket: number;
  apiMonth: number;
  runMonth: number;
  benefitMonth: number;
  netMonth: number;
  paybackMonths: number;
  roiYear1: number;
  net3yr: number;
  costPerResolvedBefore: number;
  costPerResolvedAfter: number;
}

function computePillars(option: Option, model: string, costPerCall: number): Pillars {
  const t = BASE.tickets_per_month;
  const aht = BASE.aht_min;
  const rate = BASE.loaded_cost_per_hour;
  const hoursFte = BASE.productive_hours_per_fte_month;
  const d = option.deflection;
  const r = option.aht_reduction;
  // Efficiency: automated tickets save the whole AHT; assisted tickets save a share of it.
  const savedMinPerTicket = d * aht + (1 - d) * aht * r;
  const hoursSaved = (t * savedMinPerTicket) / 60;
  const baseHours = (t * aht) / 60;
  const humanHours = baseHours - hoursSaved;
  const ahtAfter = (1 - d) * aht * (1 - r); // blended human minutes per ticket
  const frtAfter = BASE.frt_hours * (1 - option.frt_improvement);
  // Productivity: tickets resolved per FTE actually needed.
  const tpfBefore = t / (baseHours / hoursFte);
  const tpfAfter = t / (humanHours / hoursFte);
  // Transformation: new value, haircut by confidence (it is the least certain pillar).
  const newValue = option.new_value_usd_year * ASSUME.transformation_confidence;
  // Cost: API cost from measured token usage x calls per ticket x volume, plus infra and people.
  const apiPerTicket = costPerCall * option.calls_per_ticket;
  const apiMonth = apiPerTicket * t;
  const runMonth = apiMonth + option.infra_usd_month + option.people_usd_month;
  // Only part of freed capacity becomes money (attrition, avoided hiring) -> realization rate.
  const benefitMonth = hoursSaved * rate * ASSUME.realization_rate + newValue / 12;
  const netMonth = benefitMonth - runMonth;
  const payback = netMonth > 0 ? option.build_usd / netMonth : Infinity;
  const yearCost = 12 * runMonth + option.build_usd;
  const roi = (12 * benefitMonth - yearCost) / yearCost;
  return {
    optionId: option.id,
    model,
    hoursSavedMonth: hoursSaved,
    frtAfterH: frtAfter,
    ahtAfterMin: ahtAfter,
    ticketsPerFteBefore: tpfBefore,
    ticketsPerFteAfter: tpfAfter,
    productivityUplift: tpfAfter / tpfBefore - 1,
    newValueRiskAdjYear: newValue,
    apiCostPerTicket: apiPerTicket,
    apiMonth,
    runMonth,
    benefitMonth,
    netMonth,
    paybackMonths: payback,
    roiYear1: roi,
    net3yr: 36 * netMonth - option.build_usd,
    costPerResolvedBefore: (baseHours * rate) / t,
    costPerResolvedAfter: (humanHours * rate + runMonth) / t,
  };
}

// ---------------------------------------------------------------------------
// Offline responder
// ---------------------------------------------------------------------------
const PILOT_REPLIES: Record<string, string> = {
  A: "Draft reply: Sorry about the double charge - I've refunded the duplicate $49.00; it will show in 5 business days.",
  B: "Resolved: duplicate charge refunded (ref R-2291). Customer notified by email; ticket closed.",
  C: '{"queue": "billing-ops", "priority": "P2", "language": "en"}',
};

function responder(req: MockRequest): ResponderResult {
  const match = /PILOT option=(\w)/.exec(req.lastUserText);
  if (match) {
    // Offline the mock reports the token usage measured in the pilot (production-size prompts with a
    // cached system prompt + knowledge base). Live, usage comes from your real prompt.
    const oid = match[1]!;
    return { content: [textBlock(PILOT_REPLIES[oid]!)], usage: { ...OPTIONS.get(oid)!.pilot_usage } };
  }
  if (req.lastUserText.includes("SCORECARD:")) {
    const p = JSON.parse(req.lastUserText.split("SCORECARD:")[1]!);
    const [top, runner] = [p.ranking[0], p.ranking[1]];
    return (
      `Recommendation: proceed with Option ${top.id} (${top.name}) on ${top.model}, ranked first at ` +
      `${top.score}/5 ahead of Option ${runner.id} (${runner.score}/5).\n` +
      `Value: frees ${top.hours_saved} agent hours a month, lifts tickets per FTE by ` +
      `${top.productivity}, and adds ${top.new_value} of risk-adjusted new value a year.\n` +
      `Economics: year-1 ROI ${top.roi}, payback in ${top.payback} months, 3-year net value ` +
      `${top.net_3yr}; Claude API spend is only ${top.api_month} a month - people and platform ` +
      `dominate run cost.\n` +
      `Performance: ${top.model} is the cheapest model that meets every SLA (${p.sla_note}).\n` +
      `Risks: transformation value is 50% confidence-weighted; the 99.9% availability target needs ` +
      `multi-AZ hosting plus a model fallback path. Next step: 6-week pilot with the KPI tree as the ` +
      `success contract.`
    );
  }
  return "OK";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
banner("D1.6 - Aligning Solutions to Business Value Pillars");
const client = getClient(responder);

section("1. Baseline and solution options");
console.log(
  `  baseline: ${n0(BASE.tickets_per_month)} tickets/month, AHT ${BASE.aht_min} min, ` +
    `$${BASE.loaded_cost_per_hour}/agent-hour, first response ${BASE.frt_hours} h, CSAT ${BASE.csat}`,
);
console.log(
  `  assumptions: realization rate ${pct(ASSUME.realization_rate)} of freed hours, ` +
    `transformation value weighted at ${pct(ASSUME.transformation_confidence)} confidence`,
);
for (const o of VM.options) {
  console.log(
    `  ${o.id}  ${o.name.padEnd(42)} deflect=${pct(o.deflection).padStart(4)} AHT-${pct(o.aht_reduction).padStart(4)} ` +
      `build=${usd(o.build_usd)} candidates=${o.candidates.map((m) => SHORT[m]).join(",")}`,
  );
}

section("2. Pilot token usage -> Claude API cost per task (cost_usd)");
const costPerCall = new Map<string, number>();
const key = (oid: string, model: string) => `${oid}|${model}`;
for (const o of VM.options) {
  for (const model of o.candidates) {
    const msg = await client.messages.create({
      model,
      max_tokens: 500,
      system: "You are the production prompt for this option (system prompt + knowledge base are cached).",
      messages: [{ role: "user", content: `PILOT option=${o.id}: customer says they were charged twice.` }],
    });
    const c = costUsd(msg.model, msg.usage);
    costPerCall.set(key(o.id, model), c);
    const perTicket = c * o.calls_per_ticket;
    console.log(`  ${o.id} ${SHORT[model]!.padEnd(6)} ${usageLine(msg)}`);
    console.log(`           x ${o.calls_per_ticket} calls/ticket = $${perTicket.toFixed(5)}/ticket -> ${usd(perTicket * BASE.tickets_per_month)}/month API`);
  }
}

section("3. Performance SLA feasibility per model choice");
console.log(
  `  ${"opt".padEnd(4)}${"model".padEnd(8)}${"p50 s".padStart(7)}${"p95 s".padStart(7)}${"target".padStart(8)}${"avail".padStart(9)}${"target".padStart(9)}` +
    `${"quality".padStart(8)}${"target".padStart(7)}  verdict`,
);
const chosen = new Map<string, string>();
const slaResults = new Map<string, SlaResult>();
const allSla = new Map<string, SlaResult>();
for (const o of VM.options) {
  const feasible: [number, string, SlaResult][] = [];
  for (const model of o.candidates) {
    const s = checkSla(o, model);
    allSla.set(key(o.id, model), s);
    const fails = Object.entries(s.passes).filter(([, ok]) => !ok).map(([k]) => k);
    const verdict = s.allPass ? "PASS" : "FAIL " + fails.join("+");
    console.log(
      `  ${o.id.padEnd(4)}${SHORT[model]!.padEnd(8)}${s.p50.toFixed(2).padStart(7)}${s.p95.toFixed(2).padStart(7)}${o.sla.p95_latency_s.toFixed(1).padStart(8)}` +
        `${(s.availability * 100).toFixed(2).padStart(8)}%${(o.sla.availability * 100).toFixed(2).padStart(8)}%${s.quality.toFixed(2).padStart(8)}` +
        `${o.sla.quality.toFixed(2).padStart(7)}  ${verdict}`,
    );
    if (s.allPass) feasible.push([costPerCall.get(key(o.id, model))!, model, s]);
  }
  if (!feasible.length) {
    console.log(`  ${o.id}: no candidate meets the SLAs -> redesign (renegotiate SLA, change architecture)`);
    continue;
  }
  feasible.sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : 1)); // cheapest model that meets every SLA
  chosen.set(o.id, feasible[0]![1]);
  slaResults.set(o.id, feasible[0]![2]);
}
const optB = OPTIONS.get("B")!;
const noFb = compositeAvailability(optB, false);
console.log(
  `  availability for B without a model fallback: ${(noFb * 100).toFixed(2)}% (< ${(optB.sla.availability * 100).toFixed(1)}%) ` +
    "-> fallback path is what makes the SLA feasible",
);
console.log(
  "  model choice = cheapest candidate passing ALL SLAs: " +
    [...chosen].map(([k, v]) => `${k}->${SHORT[v]}`).join(", ") +
    "  (model trade-offs in depth: D2.1)",
);

section("4. Value pillars per option");
const pillars = new Map([...chosen].map(([oid, m]) => [oid, computePillars(OPTIONS.get(oid)!, m, costPerCall.get(key(oid, m))!)]));
const rows: [string, (p: Pillars) => string][] = [
  ["Efficiency: agent hours saved / month", (p) => n0(p.hoursSavedMonth)],
  ["Efficiency: first response time (h)", (p) => `${BASE.frt_hours} -> ${p.frtAfterH.toFixed(1)}`],
  ["Productivity: tickets per FTE / month", (p) => `${n0(p.ticketsPerFteBefore)} -> ${n0(p.ticketsPerFteAfter)}`],
  ["Productivity: uplift", (p) => `+${pct(p.productivityUplift)}`],
  ["Transformation: risk-adj. new value / yr", (p) => usd(p.newValueRiskAdjYear)],
  ["Cost: Claude API $ / ticket", (p) => `$${p.apiCostPerTicket.toFixed(5)}`],
  ["Cost: run cost / month (API+infra+people)", (p) => usd(p.runMonth)],
  ["Cost: realised benefit / month", (p) => usd(p.benefitMonth)],
  ["ROI: year-1 ROI", (p) => pct(p.roiYear1)],
  ["ROI: payback (months)", (p) => p.paybackMonths.toFixed(1)],
  ["ROI: 3-year net value", (p) => usd(p.net3yr)],
  ["North star: cost per resolved ticket", (p) => `$${p.costPerResolvedBefore.toFixed(2)} -> $${p.costPerResolvedAfter.toFixed(2)}`],
];
const ids = [...pillars.keys()];
console.log(`  ${"metric".padEnd(44)}` + ids.map((i) => `Option ${i} (${SHORT[chosen.get(i)!]})`.padStart(22)).join(""));
for (const [label, fn] of rows) console.log(`  ${label.padEnd(44)}` + ids.map((i) => fn(pillars.get(i)!).padStart(22)).join(""));

section("5. Weighted scorecard (0-5 per pillar)");
const w = VM.weights;
const rawScores = new Map(
  ids.map((i) => {
    const p = pillars.get(i)!;
    const s = slaResults.get(i)!;
    const passes = Object.values(s.passes);
    return [
      i,
      {
        efficiency: p.hoursSavedMonth,
        productivity: p.productivityUplift,
        transformation: p.newValueRiskAdjYear,
        cost_roi: Math.max(0, p.roiYear1),
        sla: passes.filter(Boolean).length / passes.length,
      } as Record<string, number>,
    ];
  }),
);
const scores = new Map<string, Record<string, number>>();
for (const i of ids) {
  const sc: Record<string, number> = {};
  for (const k of Object.keys(w)) {
    const top = Math.max(...ids.map((j) => rawScores.get(j)![k]!));
    sc[k] = top > 0 ? (5 * rawScores.get(i)![k]!) / top : 0;
  }
  sc.total = Object.keys(w).reduce((s, k) => s + sc[k]! * w[k]!, 0);
  scores.set(i, sc);
}
console.log(`  ${"option".padEnd(8)}` + Object.entries(w).map(([k, v]) => `${k} ${pct(v)}`.padStart(20)).join("") + "TOTAL".padStart(8));
const ranking = [...ids].sort((a, b) => scores.get(b)!.total! - scores.get(a)!.total!);
for (const i of ranking) {
  const sc = scores.get(i)!;
  console.log(`  ${i.padEnd(8)}` + Object.keys(w).map((k) => sc[k]!.toFixed(2).padStart(20)).join("") + sc.total!.toFixed(2).padStart(8));
}
const best = ranking[0]!;
console.log(`  -> recommended: Option ${best} (${OPTIONS.get(best)!.name}) on ${SHORT[chosen.get(best)!]}`);

section("6. KPI tree for the recommended option");
const p = pillars.get(best)!;
const s = slaResults.get(best)!;
const o = OPTIONS.get(best)!;
const tree: [number, string][] = [
  [0, `North star: cost per resolved ticket $${p.costPerResolvedBefore.toFixed(2)} -> $${p.costPerResolvedAfter.toFixed(2)} (guardrail: CSAT >= ${BASE.csat})`],
  [1, "Efficiency"],
  [2, `Blended AHT (min)              ${BASE.aht_min.toFixed(1)} -> ${p.ahtAfterMin.toFixed(1)}`],
  [2, `First response time (h)        ${BASE.frt_hours.toFixed(1)} -> ${p.frtAfterH.toFixed(1)}`],
  [2, `Agent hours saved / month      0 -> ${n0(p.hoursSavedMonth)}`],
  [1, "Productivity"],
  [2, `Tickets per FTE / month        ${n0(p.ticketsPerFteBefore)} -> ${n0(p.ticketsPerFteAfter)}`],
  [1, "Cost"],
  [2, `Claude API $ / ticket          $${p.apiCostPerTicket.toFixed(4)}`],
  [2, `Run cost / month               ${usd(p.runMonth)}`],
  [2, `Payback (months)               ${p.paybackMonths.toFixed(1)}`],
  [1, "Transformation"],
  [2, `Risk-adjusted new value / yr   ${usd(p.newValueRiskAdjYear)} (${o.new_capability})`],
  [1, "Performance SLAs (tracking: D6.3)"],
  [2, `p95 reply latency (s)          ${s.p95.toFixed(2)} vs <= ${o.sla.p95_latency_s.toFixed(1)}`],
  [2, `Availability                   ${(s.availability * 100).toFixed(2)}% vs >= ${(o.sla.availability * 100).toFixed(1)}%`],
  [2, `Resolution quality             ${s.quality.toFixed(2)} vs >= ${o.sla.quality.toFixed(2)}`],
];
for (const [depth, label] of tree) console.log("  " + (depth === 0 ? "" : "   ".repeat(depth - 1) + "+- ") + label);

section("7. Executive summary (Claude Opus drafts prose from computed numbers)");
const payload = {
  ranking: ranking.map((i) => {
    const pi = pillars.get(i)!;
    return {
      id: i,
      name: OPTIONS.get(i)!.name,
      model: SHORT[chosen.get(i)!],
      score: scores.get(i)!.total!.toFixed(1),
      hours_saved: n0(pi.hoursSavedMonth),
      productivity: `+${pct(pi.productivityUplift)}`,
      new_value: usd(pi.newValueRiskAdjYear),
      roi: pct(pi.roiYear1),
      payback: pi.paybackMonths.toFixed(1),
      net_3yr: usd(pi.net3yr),
      api_month: usd(pi.apiMonth),
    };
  }),
  sla_note:
    o.candidates
      .filter((m) => !allSla.get(key(best, m))!.allPass)
      .map((m) => `${SHORT[m]} fails ${Object.entries(allSla.get(key(best, m))!.passes).filter(([, ok]) => !ok).map(([k]) => k).join("+")}`)
      .join("; ") || "every candidate passes",
};
const msg = await client.messages.create({
  model: MODEL_OPUS,
  max_tokens: 1200,
  output_config: { effort: "medium" },
  system:
    "You are a principal architect writing for a CFO and COO. Five short lines: recommendation, value, " +
    "economics, performance, risks. Use ONLY the numbers in the scorecard; do not compute new ones.",
  messages: [{ role: "user", content: "SCORECARD:" + JSON.stringify(payload) }],
});
if (msg.stop_reason === "end_turn") for (const line of textOf(msg).split("\n")) console.log(`  | ${line}`);
console.log(`  [${usageLine(msg)}]`);

section("Summary");
console.log("  Tie every AI option to measurable pillars, compute value in code (not in the prompt), let SLAs pick");
console.log("  the model, risk-adjust the uncertain upside, and hand executives a KPI tree they can hold you to.");
