/**
 * D6.2 - Communicating Architectural Decisions and Trade-offs
 *
 * Decision: how should the Claims Correspondence Copilot ground drafts in policy wording?
 * Options: Hybrid RAG (A), long context + prompt caching (B), agentic search on Sonnet (C)
 * or Opus (D).
 *
 * What this lab builds:
 *   (a) a weighted decision matrix: must-meet screening -> anchored 0-10 scores ->
 *       stakeholder weights -> totals, plus sensitivity analysis (one-at-a-time, break-even
 *       scan, seeded Monte Carlo) to answer "does the winner change if the weights shift?"
 *   (b) an ADR (Michael Nygard format + options/trade-offs) drafted by Claude from the
 *       matrix as a structured output, validated by code (every number must be grounded
 *       in the matrix), repaired once if needed, and rendered deterministically
 *   (c) audience-tailored briefs (executive / engineering lead / security-legal) from ONE
 *       prompt template parameterized by an audience profile, checked by code
 *       (word budget, required topics, jargon, grounded numbers, readability)
 *   (d) a quantified trade-off narrative (cost / latency / quality deltas) built by code
 *
 * Principle: code computes every number; Claude explains them for each audience.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type Anthropic from "@anthropic-ai/sdk";
import { z } from "zod";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { MODEL_OPUS, PRICING, banner, getClient, section, textOf, usageLine } from "../../shared/typescript/claudeClient.ts";
import type { MockRequest } from "../../shared/typescript/claudeClient.ts";

const LAB = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const DATA = path.join(LAB, "data");
const OUT = path.join(LAB, "labs_output");

// ---------------------------------------------------------------------------
// Structured-output contract for the ADR draft. Status is NOT in the schema:
// whether a decision is Proposed/Accepted is a governance fact, not model output.
// ---------------------------------------------------------------------------
const ADRDraft = z.object({
  title: z.string(),
  context: z.string(),
  decision: z.string(),
  decision_drivers: z.array(z.string()),
  considered_options: z.array(
    z.object({
      option_id: z.string(),
      name: z.string(),
      pros: z.array(z.string()),
      cons: z.array(z.string()),
      outcome: z.string(), // "chosen" or "rejected: <reason>"
    }),
  ),
  consequences_positive: z.array(z.string()),
  consequences_negative: z.array(z.string()),
  risks: z.array(z.object({ risk: z.string(), mitigation: z.string() })),
  review_triggers: z.array(z.string()),
});
type ADRDraft = z.infer<typeof ADRDraft>;

const ADR_SYSTEM = `You are a solution architect writing an Architecture Decision Record (ADR, Michael Nygard style).
Write for future maintainers: state the context and forces, the decision in active voice ("We will ..."),
and the consequences - including the negative ones we accept.
Rules: use ONLY the numbers given in the FACTS block (no new or rounded-differently figures);
include every option (also eliminated ones) in considered_options with outcome "chosen" or "rejected: <reason>";
review_triggers are observable conditions that should reopen this decision.`;

const BRIEF_TEMPLATE = `You are a solution architect writing a decision brief.
Audience id: {id}
Audience: {label}
Their goal: {goal}
They care about: {cares}
Avoid: {avoid}
Format: {format}
Hard limit: {max_words} words. Plain-language target: Flesch reading ease >= {min_flesch}.
Use only facts and numbers from the FACTS block. Do not introduce new numbers.`;

// ---------------------------------------------------------------------------
// Offline responder - canned replies shared with the Python version.
// ---------------------------------------------------------------------------
function responder(req: MockRequest): string {
  const mock = path.join(DATA, "mock_responses");
  const revision = req.lastUserText.includes("REVISION REQUEST");
  if (req.systemText.includes("Architecture Decision Record")) {
    return fs.readFileSync(path.join(mock, revision ? "adr_repaired.json" : "adr_draft.json"), "utf-8");
  }
  const m = /Audience id: (\w+)/.exec(req.systemText);
  if (m) return fs.readFileSync(path.join(mock, `brief_${m[1]}${revision ? "_revised" : ""}.md`), "utf-8");
  return "(no canned reply)";
}

// ---------------------------------------------------------------------------
// Deterministic helpers (mirrored 1:1 in Python)
// ---------------------------------------------------------------------------
/** Tiny seeded PRNG with an identical Python implementation -> same Monte Carlo results. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const commas = (s: string): string => s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
function money(x: number, decimals = 0): string {
  const [i, f] = x.toFixed(decimals).split(".");
  return "$" + commas(i!) + (f !== undefined ? "." + f : "");
}
/** Python's "{:g}" for the magnitudes used here (6 significant digits, no trailing zeros). */
const g = (v: number): string => String(Number(v.toPrecision(6)));
const fill = (tpl: string, values: Record<string, string | number>): string =>
  Object.entries(values).reduce((s, [k, v]) => s.split(`{${k}}`).join(String(v)), tpl);

/** Anchored 0-10 scale. Unlike min-max across the options, adding or removing an
 *  option cannot change another option's score (no rank reversal). */
function anchoredScore(value: number, worst: number, best: number): number {
  const s = (10 * (value - worst)) / (best - worst);
  return Math.max(0, Math.min(10, s));
}

const NUMBER_PATTERNS = [
  /\$\s?(\d[\d,]*(?:\.\d+)?)/gi,
  /(\d+(?:\.\d+)?)\s?%/gi,
  /(\d+(?:\.\d+)?)\s?pp\b/gi,
  /(\d+(?:\.\d+)?)\s?(?:s|sec|seconds)\b/gi,
  /(\d+(?:\.\d+)?)\s?weeks?\b/gi,
  /\bx(\d+(?:\.\d+)?)\b/gi,
];

function extractNumbers(text: string): string[] {
  const found: string[] = [];
  for (const pat of NUMBER_PATTERNS) for (const m of text.matchAll(pat)) found.push(m[1]!);
  return found;
}

/** Numbers with units ($, %, pp, s, weeks, xN) that match no computed fact (tolerance 0.5% / 0.05). */
function ungrounded(text: string, facts: number[]): string[] {
  const bad: string[] = [];
  for (const raw of extractNumbers(text)) {
    const v = Number(raw.replace(/,/g, ""));
    if (!facts.some((f) => Math.abs(v - f) <= Math.max(0.05, 0.005 * Math.abs(f)))) bad.push(raw);
  }
  return bad;
}

function syllables(word: string): number {
  const w = word.toLowerCase();
  let n = (w.match(/[aeiouy]+/g) ?? []).length;
  if (w.endsWith("e") && n > 1) n -= 1;
  return Math.max(1, n);
}

function fleschReadingEase(text: string): number {
  const words = text.match(/[A-Za-z]+(?:'[A-Za-z]+)?/g) ?? [];
  const units = text.split(/[.!?]+\s+|\n+/).filter((u) => /[A-Za-z]/.test(u));
  if (!words.length) return 0;
  const syl = words.reduce((a, w) => a + syllables(w), 0);
  return 206.835 - 1.015 * (words.length / Math.max(1, units.length)) - 84.6 * (syl / words.length);
}

const wordCount = (text: string): number => (text.match(/[A-Za-z0-9][A-Za-z0-9'.,%/-]*/g) ?? []).length;
const escapeRe = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const lineCount = (s: string): number => s.split("\n").length - (s.endsWith("\n") ? 1 : 0);

// ---------------------------------------------------------------------------
// Data shapes
// ---------------------------------------------------------------------------
interface Criterion { id: string; label: string; metric: string; unit: string; worst: number; best: number; why: string }
interface ConstraintSpec { id: string; label: string; metric: string; op: string; value: number }
interface OptionSpec {
  id: string; name: string; summary: string; model: string; raw: Record<string, number>;
  tokens_per_request: { input: number; cache_read: number; cache_write: number; output: number };
  fixed_monthly_usd: number;
}
interface StakeholderSpec { id: string; name: string; role: string; influence: number; weights: Record<string, number> }
interface DecisionFile {
  adr_number: number; as_of: string; project: string; question: string; context_facts: string[];
  requests_per_month: number; letters_per_month: number; requirement_links: string[];
  constraints: ConstraintSpec[]; criteria: Criterion[]; options: OptionSpec[]; stakeholders: StakeholderSpec[];
  sensitivity: { perturbation: number; trials: number; seed: number; scan_max_multiplier: number; scan_step: number };
}
interface Audience {
  id: string; label: string; goal: string; cares_about: string[]; avoid: string[]; format: string;
  max_words: number; min_flesch: number; required: string[][]; jargon_blocklist: string[];
}

interface Option {
  id: string; name: string; summary: string; model: string; raw: Record<string, number>;
  costPerRequest: number; monthlyCost: number; costPerLetter: number;
  failures: string[]; scores: Record<string, number>; total: number;
}
type Weights = Record<string, number>;
type PerStakeholder = [string, string, number][]; // [name, winner id, total]
type Breakeven = [string, number, string]; // [criterion id, multiplier, new winner]
interface BriefMetrics { words: number; flesch: number; numbers: number }
type Brief = [Audience, string, string[], BriefMetrics, number];

function weightedTotal(opt: Option, weights: Weights, criteria: Criterion[]): number {
  const wsum = criteria.reduce((a, c) => a + weights[c.id]!, 0);
  return criteria.reduce((a, c) => a + weights[c.id]! * opt.scores[c.id]!, 0) / wsum;
}

function winnerOf(options: Option[], weights: Weights, criteria: Criterion[]): Option {
  let best = options[0]!;
  let bestTotal = -1;
  for (const o of options) {
    const t = weightedTotal(o, weights, criteria);
    if (t > bestTotal) {
      best = o;
      bestTotal = t;
    }
  }
  return best;
}

const checkConstraint = (value: number, op: string, target: number): boolean =>
  op === "<=" ? value <= target : op === ">=" ? value >= target : value === target;

async function main(): Promise<void> {
  banner("D6.2 - Communicating Architectural Decisions and Trade-offs");
  const d: DecisionFile = JSON.parse(fs.readFileSync(path.join(DATA, "decision.json"), "utf-8"));
  const audiences: Audience[] = JSON.parse(fs.readFileSync(path.join(DATA, "audiences.json"), "utf-8")).audiences;
  const criteria = d.criteria;
  const critLabel = new Map(criteria.map((c) => [c.id, c.label]));
  const client = getClient(responder);

  // -- 1. Framing ------------------------------------------------------------
  section("1. Decision framing");
  console.log(`Question: ${d.question}`);
  for (const o of d.options) console.log(`  Option ${o.id}: ${o.name}`);
  console.log("Must-meet constraints (screen first - filters, not weights):");
  for (const c of d.constraints) console.log(`  ${c.id}: ${c.label}`);

  // -- 2. Cost model -----------------------------------------------------------
  section("2. Cost model from token profiles x model pricing");
  const options: Option[] = [];
  for (const o of d.options) {
    const p = PRICING[o.model]!;
    const t = o.tokens_per_request;
    const perReq = (t.input * p.input + t.cache_read * p.cacheRead + t.cache_write * p.cacheWrite + t.output * p.output) / 1_000_000;
    const monthly = perReq * d.requests_per_month + o.fixed_monthly_usd;
    const opt: Option = {
      id: o.id, name: o.name, summary: o.summary, model: o.model, raw: { ...o.raw },
      costPerRequest: perReq, monthlyCost: monthly, costPerLetter: monthly / d.letters_per_month,
      failures: [], scores: {}, total: 0,
    };
    opt.raw.monthly_cost_usd = monthly;
    options.push(opt);
    console.log(
      `  ${o.id} ${o.model.padEnd(18)} in=${String(t.input).padStart(6)} cache_read=${String(t.cache_read).padStart(6)} ` +
        `cache_write=${String(t.cache_write).padStart(5)} out=${String(t.output).padStart(5)} -> $${perReq.toFixed(4)}/request, ` +
        `${money(monthly)}/month (incl. ${money(o.fixed_monthly_usd)} fixed)`,
    );
  }
  console.log(`  Volume: ${commas(String(d.requests_per_month))} requests/month for ${commas(String(d.letters_per_month))} letters`);

  // -- 3. Must-meet screening ----------------------------------------------------
  section("3. Must-meet screening");
  for (const opt of options) {
    for (const c of d.constraints) {
      if (!checkConstraint(opt.raw[c.metric]!, c.op, c.value)) opt.failures.push(`${c.id} (${g(opt.raw[c.metric]!)} vs ${c.op} ${g(c.value)})`);
    }
    console.log(`  ${opt.id}: ${opt.failures.length ? "ELIMINATED - fails " + opt.failures.join(", ") : "PASS"}`);
  }
  const viable = options.filter((o) => !o.failures.length);
  const eliminated = options.filter((o) => o.failures.length);

  // -- 4. Weights -----------------------------------------------------------------
  section("4. Stakeholder weights (points out of 100) and influence");
  console.log(`  ${"stakeholder".padEnd(34)}${"infl".padStart(5)}  ` + criteria.map((c) => c.id.slice(0, 7).padStart(7)).join(" "));
  const combined: Weights = Object.fromEntries(criteria.map((c) => [c.id, 0]));
  for (const s of d.stakeholders) {
    console.log(
      `  ${`${s.name} (${s.id})`.padEnd(34)}${s.influence.toFixed(2).padStart(5)}  ` +
        criteria.map((c) => String(s.weights[c.id]).padStart(7)).join(" "),
    );
    for (const c of criteria) combined[c.id] = combined[c.id]! + s.influence * s.weights[c.id]!;
  }
  console.log(`  ${"COMBINED (influence-weighted)".padEnd(34)}${"".padStart(5)}  ` + criteria.map((c) => combined[c.id]!.toFixed(1).padStart(7)).join(" "));

  // -- 5. Matrix --------------------------------------------------------------------
  section("5. Weighted decision matrix (anchored 0-10 scores)");
  for (const opt of options) {
    for (const c of criteria) opt.scores[c.id] = anchoredScore(opt.raw[c.metric]!, c.worst, c.best);
    opt.total = weightedTotal(opt, combined, criteria);
  }
  console.log(`  ${"criterion".padEnd(24)}${"weight".padStart(7)}` + options.map((o) => o.id.padStart(8)).join(""));
  for (const c of criteria) {
    console.log(`  ${c.label.padEnd(24)}${combined[c.id]!.toFixed(1).padStart(7)}` + options.map((o) => o.scores[c.id]!.toFixed(2).padStart(8)).join(""));
  }
  console.log(`  ${"WEIGHTED TOTAL".padEnd(24)}${"".padStart(7)}` + options.map((o) => o.total.toFixed(2).padStart(8)).join(""));
  const ranked = [...viable].sort((a, b) => b.total - a.total);
  const win = ranked[0]!;
  const runner = ranked[1]!;
  console.log(
    `\n  Recommended: Option ${win.id} (${win.name}) ${win.total.toFixed(2)}; runner-up ${runner.id} ${runner.total.toFixed(2)}; ` +
      `margin ${(win.total - runner.total).toFixed(2)}`,
  );
  if (eliminated.length) console.log(`  Not eligible: ${eliminated.map((o) => o.id).join(", ")} (scored for transparency, excluded by must-meet)`);
  console.log("  Winner through each stakeholder's own weights:");
  const perStakeholder: PerStakeholder = [];
  for (const s of d.stakeholders) {
    const w: Weights = { ...s.weights };
    const sw = winnerOf(viable, w, criteria);
    perStakeholder.push([s.name, sw.id, weightedTotal(sw, w, criteria)]);
    console.log(
      `    ${s.name.padEnd(16)} -> ${sw.id} (${weightedTotal(sw, w, criteria).toFixed(2)})` +
        (sw.id === win.id ? "" : "   <- disagrees: address this concern explicitly"),
    );
  }

  // -- 6. Sensitivity ----------------------------------------------------------------
  section("6. Sensitivity analysis: does the winner change if weights shift?");
  const sens = d.sensitivity;
  const pert = sens.perturbation;
  console.log(`  One-at-a-time: each weight x${(1 - pert).toFixed(1)} and x${(1 + pert).toFixed(1)} (others unchanged, renormalized)`);
  const oatFlips: string[] = [];
  for (const c of criteria) {
    const res: string[] = [];
    for (const mult of [1 - pert, 1 + pert]) {
      const w = { ...combined, [c.id]: combined[c.id]! * mult };
      res.push(winnerOf(viable, w, criteria).id);
    }
    const flag = res[0] === win.id && res[1] === win.id ? "" : "  <- FLIPS";
    if (flag) oatFlips.push(c.id);
    console.log(`    ${c.label.padEnd(24)} low -> ${res[0]}   high -> ${res[1]}${flag}`);
  }

  console.log(
    `  Break-even scan: smallest single-weight multiplier (step ${sens.scan_step}, up to x${g(sens.scan_max_multiplier)}) that changes the winner`,
  );
  const breakevens: Breakeven[] = [];
  const stepsUp = Math.round((sens.scan_max_multiplier - 1) / sens.scan_step);
  const stepsDown = Math.round(1 / sens.scan_step);
  for (const c of criteria) {
    let hit: [number, string] | null = null;
    for (let k = 1; k <= Math.max(stepsUp, stepsDown); k++) {
      for (const mult of [1 + k * sens.scan_step, 1 - k * sens.scan_step]) {
        if (mult < 0 || mult > sens.scan_max_multiplier + 1e-9) continue;
        const w = { ...combined, [c.id]: combined[c.id]! * mult };
        const wn = winnerOf(viable, w, criteria);
        if (wn.id !== win.id && hit === null) hit = [mult, wn.id];
      }
      if (hit) break;
    }
    if (hit) {
      breakevens.push([c.id, hit[0], hit[1]]);
      console.log(
        `    ${c.label.padEnd(24)} x${hit[0].toFixed(2)} (${combined[c.id]!.toFixed(1)} -> ${(combined[c.id]! * hit[0]).toFixed(1)} pts) -> ${hit[1]} wins`,
      );
    } else {
      console.log(`    ${c.label.padEnd(24)} no change within x0.00 - x${sens.scan_max_multiplier.toFixed(2)}`);
    }
  }

  const rng = mulberry32(sens.seed);
  const wins = new Map(viable.map((o) => [o.id, 0]));
  for (let i = 0; i < sens.trials; i++) {
    const w: Weights = {};
    for (const c of criteria) w[c.id] = combined[c.id]! * (1 - pert + 2 * pert * rng());
    const id = winnerOf(viable, w, criteria).id;
    wins.set(id, wins.get(id)! + 1);
  }
  const mc = new Map([...wins].map(([k, v]) => [k, (v / sens.trials) * 100]));
  console.log(
    `  Monte Carlo: ${commas(String(sens.trials))} trials, every weight x U(${(1 - pert).toFixed(1)}, ${(1 + pert).toFixed(1)}), seed ${sens.seed}`,
  );
  for (const [k, v] of mc) console.log(`    Option ${k} wins ${v.toFixed(1).padStart(5)}% ${"#".repeat(Math.floor(v / 2))}`);
  const robust = mc.get(win.id)! >= 80 ? "ROBUST" : "FRAGILE";
  console.log(
    `  Verdict: ${robust} - ${win.id} wins ${mc.get(win.id)!.toFixed(1)}% of perturbations; ` +
      (breakevens.length
        ? `only a large shift toward ${breakevens.map((b) => critLabel.get(b[0])).join(", ")} changes it.`
        : "no single weight shift changes it."),
  );

  // -- 7. Trade-off narrative ----------------------------------------------------------
  section("7. Quantified trade-off narrative (code-generated)");
  const tradeoffs = [...ranked.filter((x) => x.id !== win.id), ...eliminated].map((o) => tradeoffSentence(win, o));
  for (const t of tradeoffs) console.log(`  - ${t}`);
  const headroom = 8000 - win.monthlyCost;
  console.log(
    `  - Budget: ${win.id} uses ${money(win.monthlyCost)} of the $8,000/month cap (${money(headroom)} headroom), ` +
      `about $${win.costPerLetter.toFixed(3)} per letter.`,
  );

  // Facts = the only numbers Claude may use. Collected once; checked everywhere.
  const [factsText, factsNum] = buildFacts(d, options, win, runner, combined, criteria, perStakeholder, mc, breakevens, tradeoffs, headroom);

  // -- 8. ADR ---------------------------------------------------------------------------
  section("8. ADR drafted by Claude (structured output), validated by code");
  const adrNo = String(d.adr_number).padStart(4, "0");
  const adrUser =
    `Write ADR-${adrNo} for this decision.\nQUESTION: ${d.question}\n` +
    `CONTEXT:\n- ` + d.context_facts.join("\n- ") + "\n\nFACTS:\n" + factsText;
  let adr: ADRDraft | null = null;
  let issues: string[] = [];
  let adrAttempts = 0;
  let messages: Anthropic.MessageParam[] = [{ role: "user", content: adrUser }];
  while (adrAttempts < 2) {
    adrAttempts += 1;
    const resp = await client.messages.parse({
      model: MODEL_OPUS,
      max_tokens: 6000,
      thinking: { type: "adaptive" },
      output_config: { effort: "high", format: zodOutputFormat(ADRDraft) },
      system: ADR_SYSTEM,
      messages,
    });
    adr = resp.parsed_output;
    if (!adr) throw new Error("ADR did not parse");
    issues = validateAdr(adr, options, win, factsNum);
    console.log(`  attempt ${adrAttempts}: ${issues.length} issue(s) | ${usageLine(resp)}`);
    for (const i of issues) console.log(`    - ${i}`);
    if (!issues.length) break;
    // Validate -> repair loop: give the model its own draft plus the exact problems.
    messages = [
      {
        role: "user",
        content:
          adrUser + "\n\nREVISION REQUEST - fix these issues and return the full ADR:\n- " + issues.join("\n- ") +
          "\n\nPrevious draft:\n" + JSON.stringify(adr),
      },
    ];
  }
  const adrMd = renderAdr(d, adr!, options, win, combined, tradeoffs, perStakeholder, mc, breakevens, headroom, issues, adrAttempts);
  console.log(`  Rendered ADR-${adrNo} in Nygard format: Status / Context / Decision / Consequences (+ options, trade-offs)`);

  // -- 9. Audience briefs -------------------------------------------------------------
  section("9. Audience-tailored briefs from one parameterized template");
  const briefs: Brief[] = [];
  for (const aud of audiences) {
    const system = fill(BRIEF_TEMPLATE, {
      id: aud.id, label: aud.label, goal: aud.goal, cares: aud.cares_about.join("; "),
      avoid: aud.avoid.join("; "), format: aud.format, max_words: aud.max_words, min_flesch: aud.min_flesch,
    });
    const user = "FACTS:\n" + factsText + "\n\nWrite the brief for this audience.";
    let text = "";
    let checks: string[] = [];
    let metrics: BriefMetrics = { words: 0, flesch: 0, numbers: 0 };
    let attempts = 0;
    let msgs: Anthropic.MessageParam[] = [{ role: "user", content: user }];
    while (attempts < 2) {
      attempts += 1;
      const resp = await client.messages.create({
        model: MODEL_OPUS,
        max_tokens: 1500,
        thinking: { type: "adaptive" },
        output_config: { effort: "medium" },
        system,
        messages: msgs,
      });
      text = textOf(resp).trim();
      [checks, metrics] = checkBrief(text, aud, factsNum);
      console.log(
        `  ${aud.id.padEnd(17)} attempt ${attempts}: ${metrics.words}/${aud.max_words} words, ` +
          `Flesch ${metrics.flesch.toFixed(1)}, issues=${checks.length} | ${usageLine(resp)}`,
      );
      for (const ch of checks) console.log(`      - ${ch}`);
      if (!checks.length) break;
      msgs = [{ role: "user", content: user + "\n\nREVISION REQUEST - fix: " + checks.join("; ") + "\n\nPrevious draft:\n" + text }];
    }
    briefs.push([aud, text, checks, metrics, attempts]);
  }
  console.log("  Same facts, different framing: exec = value/cost/risk/ask; engineering = design/ops; security = data flows/controls.");

  // -- 10. Outputs -------------------------------------------------------------------
  section("10. Outputs");
  fs.mkdirSync(OUT, { recursive: true });
  const files: [string, string][] = [
    [`ADR-${adrNo}-policy-grounding-strategy.md`, adrMd],
    ["decision_matrix.md", renderMatrix(d, options, win, combined, criteria, perStakeholder, oatFlips, breakevens, mc, robust)],
    ["stakeholder_briefs.md", renderBriefs(d, win, briefs)],
    ["decision_facts.txt", factsText + "\n"],
  ];
  for (const [name, content] of files) {
    fs.writeFileSync(path.join(OUT, name), content, "utf-8");
    console.log(`  wrote labs_output/${name} (${lineCount(content)} lines)`);
  }
  console.log("  Samples committed under examples/");
}

// ---------------------------------------------------------------------------
// Narrative, facts, validation and rendering
// ---------------------------------------------------------------------------
function tradeoffSentence(w: Option, o: Option): string {
  const dm = w.monthlyCost - o.monthlyCost;
  const pct = (Math.abs(dm) / o.monthlyCost) * 100;
  const cost = dm < 0 ? `saves ${money(Math.abs(dm))}/month (-${pct.toFixed(1)}%)` : `costs ${money(dm)}/month more (+${pct.toFixed(1)}%)`;
  const dl = w.raw.p95_latency_s! - o.raw.p95_latency_s!;
  const lat = `${Math.abs(dl).toFixed(1)} s ${dl < 0 ? "faster" : "slower"} at p95`;
  const dq = w.raw.eval_pass_rate_pct! - o.raw.eval_pass_rate_pct!;
  const qual = `${Math.abs(dq).toFixed(1)} pp ${dq >= 0 ? "higher" : "lower"} answer quality`;
  const dc = w.raw.citation_accuracy_pct! - o.raw.citation_accuracy_pct!;
  const cit = `${Math.abs(dc).toFixed(1)} pp ${dc >= 0 ? "higher" : "lower"} citation accuracy`;
  const dw = w.raw.build_weeks! - o.raw.build_weeks!;
  const weeks = dw === 0 ? "same build time" : `ships ${g(Math.abs(dw))} weeks ${dw < 0 ? "sooner" : "later"}`;
  const tail = o.failures.length ? ` (${o.id} is not eligible: fails ${o.failures.join(", ")})` : "";
  return `${w.id} over ${o.id}: ${cost}; ${lat}; ${qual}; ${cit}; ${weeks}.${tail}`;
}

function buildFacts(
  d: DecisionFile, options: Option[], win: Option, runner: Option, combined: Weights, criteria: Criterion[],
  perStakeholder: PerStakeholder, mc: Map<string, number>, breakevens: Breakeven[], tradeoffs: string[], headroom: number,
): [string, number[]] {
  const lines: string[] = [];
  const nums: number[] = [...d.constraints.map((c) => c.value), d.requests_per_month, d.letters_per_month];
  for (const o of options) {
    const r = o.raw;
    const status = o.failures.length ? "NOT ELIGIBLE - fails " + o.failures.join(", ") : "eligible";
    lines.push(
      `Option ${o.id} (${o.name}): eval pass rate ${g(r.eval_pass_rate_pct!)}%, p95 ${g(r.p95_latency_s!)} s, ` +
        `monthly cost ${money(o.monthlyCost)} (${money(o.costPerRequest, 4)}/request, ${money(o.costPerLetter, 3)}/letter), ` +
        `build ${g(r.build_weeks!)} weeks, citation accuracy ${g(r.citation_accuracy_pct!)}%, ` +
        `freshness ${g(r.freshness_minutes!)} min, weighted score ${o.total.toFixed(2)}/10 - ${status}`,
    );
    nums.push(...criteria.map((c) => r[c.metric]!), o.costPerRequest, o.monthlyCost, o.costPerLetter, o.total);
  }
  lines.push(`Recommended: Option ${win.id}; runner-up Option ${runner.id} (margin ${(win.total - runner.total).toFixed(2)})`);
  nums.push(win.total - runner.total, headroom);
  lines.push(`Budget headroom with ${win.id}: ${money(headroom)}/month under the $8,000 cap`);
  lines.push("Combined weights: " + criteria.map((c) => `${c.label} ${combined[c.id]!.toFixed(1)}`).join(", "));
  nums.push(...Object.values(combined));
  lines.push("Winner by stakeholder weights: " + perStakeholder.map(([n, wid]) => `${n} -> ${wid}`).join(", "));
  nums.push(...perStakeholder.map(([, , t]) => t));
  lines.push(
    `Monte Carlo (${d.sensitivity.trials} trials, weights +/-${g(d.sensitivity.perturbation * 100)}%): ` +
      [...mc].map(([k, v]) => `${k} wins ${v.toFixed(1)}%`).join(", "),
  );
  nums.push(...mc.values(), d.sensitivity.trials, d.sensitivity.perturbation * 100);
  for (const [cid, mult, wid] of breakevens) {
    lines.push(
      `Break-even: ${cid} weight x${mult.toFixed(2)} (${combined[cid]!.toFixed(1)} -> ${(combined[cid]! * mult).toFixed(1)} points) makes ${wid} the winner`,
    );
    nums.push(mult, combined[cid]! * mult);
  }
  for (const t of tradeoffs) {
    lines.push("Trade-off: " + t);
    nums.push(...extractNumbers(t).map((x) => Number(x.replace(/,/g, ""))));
  }
  return [lines.join("\n"), nums];
}

/** Every free-text field of the draft - all of it is subject to the grounding check. */
function adrText(adr: ADRDraft): string {
  const texts = [
    adr.title, adr.context, adr.decision, ...adr.decision_drivers, ...adr.consequences_positive,
    ...adr.consequences_negative, ...adr.review_triggers,
  ];
  for (const o of adr.considered_options) texts.push(...o.pros, ...o.cons, o.outcome);
  for (const r of adr.risks) texts.push(r.risk, r.mitigation);
  return texts.join("\n");
}

function validateAdr(adr: ADRDraft, options: Option[], win: Option, facts: number[]): string[] {
  const issues: string[] = [];
  const dec = adr.decision.toLowerCase();
  if (!dec.includes(win.name.toLowerCase()) && !dec.includes(`option ${win.id}`.toLowerCase())) {
    issues.push(`decision text does not name the recommended option (${win.id})`);
  }
  const ids = adr.considered_options.map((o) => o.option_id);
  const missing = options.filter((o) => !ids.includes(o.id)).map((o) => o.id);
  if (missing.length) issues.push(`considered_options missing: ${missing.join(", ")}`);
  const chosen = adr.considered_options.filter((o) => o.outcome.toLowerCase().startsWith("chosen")).map((o) => o.option_id);
  if (!(chosen.length === 1 && chosen[0] === win.id)) {
    issues.push(`exactly one option must be 'chosen' and it must be ${win.id} (got [${chosen.map((c) => `'${c}'`).join(", ")}])`);
  }
  if (!adr.consequences_negative.length) issues.push("no negative consequences - every real decision has a cost");
  if (!adr.review_triggers.length) issues.push("no review triggers");
  const bad = ungrounded(adrText(adr), facts);
  if (bad.length) issues.push("ungrounded numbers (not in the decision matrix): " + bad.join(", "));
  return issues;
}

function checkBrief(text: string, aud: Audience, facts: number[]): [string[], BriefMetrics] {
  const issues: string[] = [];
  const words = wordCount(text);
  const low = text.toLowerCase();
  if (words > aud.max_words) issues.push(`over word budget (${words} > ${aud.max_words})`);
  for (const group of aud.required) {
    if (!group.some((term) => low.includes(term.toLowerCase()))) issues.push(`missing required topic: ${group.join(" / ")}`);
  }
  const jargon = aud.jargon_blocklist.filter((j) => new RegExp(`(?<![a-z])${escapeRe(j.toLowerCase())}(?![a-z])`).test(low));
  if (jargon.length) issues.push(`jargon for this audience: ${jargon.join(", ")}`);
  const bad = ungrounded(text, facts);
  if (bad.length) issues.push(`ungrounded numbers: ${bad.join(", ")}`);
  const flesch = fleschReadingEase(text);
  if (flesch < aud.min_flesch) issues.push(`reading ease ${flesch.toFixed(1)} < ${aud.min_flesch} (too dense for this audience)`);
  return [issues, { words, flesch, numbers: extractNumbers(text).length }];
}

function renderAdr(
  d: DecisionFile, adr: ADRDraft, options: Option[], win: Option, combined: Weights, tradeoffs: string[],
  perStakeholder: PerStakeholder, mc: Map<string, number>, breakevens: Breakeven[], headroom: number,
  issues: string[], attempts: number,
): string {
  const deciders = d.stakeholders.map((s) => `${s.name} (${s.role})`).join(", ");
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  a(`# ADR-${String(d.adr_number).padStart(4, "0")}: ${adr.title}`);
  a("");
  a("## Status");
  a("");
  a(`Proposed - ${d.as_of}. Awaiting approval by the deciders below; becomes Accepted when signed off.`);
  a("");
  a(`- **Deciders:** ${deciders}`);
  a(`- **Related requirements:** ${d.requirement_links.join(", ")}`);
  a("");
  a("## Context");
  a("");
  a(adr.context);
  a("");
  a("**Decision drivers**");
  a("");
  for (const x of adr.decision_drivers) a(`- ${x}`);
  a("");
  a("## Decision");
  a("");
  a(adr.decision);
  a("");
  a("## Consequences");
  a("");
  a("**Positive**");
  a("");
  for (const x of adr.consequences_positive) a(`- ${x}`);
  a("");
  a("**Negative (accepted trade-offs)**");
  a("");
  for (const x of adr.consequences_negative) a(`- ${x}`);
  a("");
  a("## Considered options");
  a("");
  a("| Option | Weighted score (0-10) | Must-meet | Outcome |");
  a("|---|---:|---|---|");
  const byId = new Map(options.map((o) => [o.id, o]));
  for (const co of adr.considered_options) {
    const o = byId.get(co.option_id);
    const score = o ? o.total.toFixed(2) : "-";
    const mm = o ? (o.failures.length ? "FAIL: " + o.failures.join(", ") : "PASS") : "-";
    a(`| ${co.option_id} - ${co.name} | ${score} | ${mm} | ${co.outcome} |`);
  }
  a("");
  for (const co of adr.considered_options) {
    a(`**${co.option_id} - ${co.name}**`);
    a("");
    a("- Pros: " + co.pros.join("; "));
    a("- Cons: " + co.cons.join("; "));
    a("");
  }
  a("## Trade-offs (quantified, computed from the decision matrix)");
  a("");
  for (const t of tradeoffs) a(`- ${t}`);
  a(`- Budget: ${money(win.monthlyCost)} of the $8,000/month cap (${money(headroom)} headroom), about $${win.costPerLetter.toFixed(3)} per letter.`);
  a("");
  a("## Sensitivity");
  a("");
  for (const [name, wid, total] of perStakeholder) a(`- ${name}'s weights -> Option ${wid} (${total.toFixed(2)})`);
  a(
    `- Monte Carlo (${commas(String(d.sensitivity.trials))} weight perturbations of +/-${g(d.sensitivity.perturbation * 100)}%): ` +
      [...mc].map(([k, v]) => `${k} wins ${v.toFixed(1)}%`).join(", "),
  );
  for (const [cid, mult, wid] of breakevens) {
    a(
      `- Break-even: the ${cid} weight would need to rise x${mult.toFixed(2)} (${combined[cid]!.toFixed(1)} -> ${(combined[cid]! * mult).toFixed(1)} points) for Option ${wid} to win.`,
    );
  }
  a("");
  a("## Risks and mitigations");
  a("");
  a("| Risk | Mitigation |");
  a("|---|---|");
  for (const r of adr.risks) a(`| ${r.risk} | ${r.mitigation} |`);
  a("");
  a("## Review triggers (reopen this ADR if...)");
  a("");
  for (const x of adr.review_triggers) a(`- ${x}`);
  a("");
  a("---");
  const nChecked = extractNumbers(adrText(adr)).length;
  const rounds = attempts === 1 ? "first draft" : `after ${attempts - 1} repair round(s)`;
  a(
    `_Drafted by Claude from the decision matrix (structured output). Status, scores, trade-offs and sensitivity ` +
      `are computed by code; ${nChecked} figures in Claude's text were checked against the matrix ` +
      `(${rounds}: ${issues.length ? "issues: " + issues.join("; ") : "0 ungrounded"})._`,
  );
  a("");
  return L.join("\n");
}

function renderMatrix(
  d: DecisionFile, options: Option[], win: Option, combined: Weights, criteria: Criterion[], perStakeholder: PerStakeholder,
  oatFlips: string[], breakevens: Breakeven[], mc: Map<string, number>, robust: string,
): string {
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  const s = d.sensitivity;
  a(`# Decision matrix - ${d.question}`);
  a("");
  a(
    `As of ${d.as_of}. Scores use anchored 0-10 scales (worst -> 0, best -> 10), so adding or removing an option ` +
      "cannot reorder the others. Must-meet constraints are applied before scoring.",
  );
  a("");
  a("## Raw measurements");
  a("");
  a("| Criterion | Unit | Worst / best anchor | " + options.map((o) => o.id).join(" | ") + " |");
  a("|---|---|---|" + "---:|".repeat(options.length));
  for (const c of criteria) {
    const vals = options.map((o) => (c.id === "cost" ? money(o.raw[c.metric]!) : g(o.raw[c.metric]!)));
    a(`| ${c.label} | ${c.unit} | ${g(c.worst)} / ${g(c.best)} | ` + vals.join(" | ") + " |");
  }
  a("| Must-meet | - | - | " + options.map((o) => (o.failures.length ? "FAIL" : "PASS")).join(" | ") + " |");
  a("");
  a("## Weighted scores");
  a("");
  a("| Criterion | Weight | " + options.map((o) => o.id).join(" | ") + " |");
  a("|---|---:|" + "---:|".repeat(options.length));
  for (const c of criteria) {
    a(`| ${c.label} | ${combined[c.id]!.toFixed(1)} | ` + options.map((o) => o.scores[c.id]!.toFixed(2)).join(" | ") + " |");
  }
  a("| **Weighted total** | 100 | " + options.map((o) => `**${o.total.toFixed(2)}**`).join(" | ") + " |");
  a("");
  a(`**Recommendation:** Option ${win.id} - ${win.name}.`);
  a("");
  a("## Stakeholder weights");
  a("");
  a("| Stakeholder | Influence | " + criteria.map((c) => c.label).join(" | ") + " | Own winner |");
  a("|---|---:|" + "---:|".repeat(criteria.length) + "---|");
  d.stakeholders.forEach((st, i) => {
    const [, wid, total] = perStakeholder[i]!;
    a(
      `| ${st.name} (${st.role}) | ${st.influence.toFixed(2)} | ` + criteria.map((c) => String(st.weights[c.id])).join(" | ") +
        ` | ${wid} (${total.toFixed(2)}) |`,
    );
  });
  a("");
  a("## Sensitivity");
  a("");
  a(
    `- One-at-a-time (each weight x${(1 - s.perturbation).toFixed(1)} / x${(1 + s.perturbation).toFixed(1)}): ` +
      (oatFlips.length ? `winner flips for ${oatFlips.join(", ")}` : "winner never changes"),
  );
  for (const [cid, mult, wid] of breakevens) {
    a(`- Break-even: ${cid} weight x${mult.toFixed(2)} (${combined[cid]!.toFixed(1)} -> ${(combined[cid]! * mult).toFixed(1)} points) -> Option ${wid}`);
  }
  a(
    `- Monte Carlo (${commas(String(s.trials))} trials, seed ${s.seed}): ` +
      [...mc].map(([k, v]) => `${k} ${v.toFixed(1)}%`).join(", ") + ` -> **${robust}**`,
  );
  a("");
  return L.join("\n");
}

function renderBriefs(d: DecisionFile, win: Option, briefs: Brief[]): string {
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  a(`# Stakeholder briefs - ADR-${String(d.adr_number).padStart(4, "0")} (Option ${win.id}: ${win.name})`);
  a("");
  a(
    "One decision, three framings. Each brief came from the same prompt template parameterized by an audience " +
      "profile (`data/audiences.json`) and was checked by code before release.",
  );
  a("");
  for (const [aud, text, checks, metrics, attempts] of briefs) {
    a(`## For: ${aud.label}`);
    a("");
    a(text);
    a("");
    const revised = attempts === 1 ? "first draft" : `after ${attempts - 1} revision(s)`;
    a(
      `> Checks (${revised}): ${metrics.words}/${aud.max_words} words; Flesch reading ease ${metrics.flesch.toFixed(1)}; ` +
        `${metrics.numbers} figures checked against the matrix; ` +
        (checks.length ? `OPEN ISSUES: ${checks.join("; ")}` : "required topics, jargon and grounding all passed") + ".",
    );
    a("");
  }
  return L.join("\n");
}

await main();
