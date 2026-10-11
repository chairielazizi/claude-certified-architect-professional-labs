/**
 * D1.1 - Translate Business Problems into Claude Solutions
 *
 * Pipeline built here (the way an architect runs a discovery-to-design pass):
 *
 *   intake notes -> structured framing (Claude + structured outputs)
 *                -> AI-suitability rubric (deterministic: complexity, value, viability, cost of error)
 *                -> tier decision (single call / workflow / agent / not-an-LLM-problem)
 *                -> capability mapping (classification, extraction, vision, RAG, tool use, ...)
 *                -> solution blueprint (+ a Claude-drafted one-pager for the top candidate)
 *
 * Design principle: Claude does the *language* work (turning messy interview notes into
 * typed fields). The *decision* is made by transparent code, so it is repeatable,
 * reviewable and defensible in front of stakeholders.
 */
import type Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  MODEL_HAIKU,
  MODEL_OPUS,
  MODEL_SONNET,
  banner,
  costUsd,
  elapsedMs,
  getClient,
  section,
  textOf,
  usageLine,
  type MockRequest,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");

// ---------------------------------------------------------------------------
// 1. The framing contract. Claude fills this; code consumes it.
//    Enums keep the downstream rubric total: every value has a rule.
// ---------------------------------------------------------------------------
const ProblemFraming = z.object({
  pain_point: z.string(),
  primary_users: z.array(z.string()),
  monthly_volume: z.number().int().describe("Work items per month"),
  minutes_per_item_today: z.number().describe("Human handling minutes per item today"),
  loaded_cost_per_hour_usd: z.number(),
  input_modalities: z.array(z.enum(["text", "image", "pdf", "structured", "audio"])),
  data_shape: z.enum(["structured", "semi_structured", "unstructured"]),
  task_nature: z.enum([
    "deterministic_rules",
    "judgment_on_language",
    "open_ended_investigation",
    "generation_or_summarization",
  ]),
  steps_predictable: z.boolean().describe("Can the sequence of steps be written down in advance?"),
  needs_reference_knowledge: z.boolean(),
  needs_external_systems: z.boolean(),
  systems_involved: z.array(z.string()),
  latency_budget_ms: z.number().int(),
  error_severity: z.enum(["low", "medium", "high", "critical"]),
  human_review_feasible: z.boolean(),
  data_available: z.boolean(),
  constraints: z.array(z.string()),
  success_kpis: z.array(z.string()),
});
type ProblemFraming = z.infer<typeof ProblemFraming>;

const FRAMING_SYSTEM =
  "You are a solution architect running AI discovery. Extract a structured problem framing from " +
  "stakeholder interview notes. Use only facts stated or directly implied by the notes; if a number " +
  "is not given, use 0. Classify task_nature by what the work fundamentally is, not by what tool " +
  "might solve it.";

// ---------------------------------------------------------------------------
// 2. Deterministic AI-suitability rubric (the four criteria)
// ---------------------------------------------------------------------------
const NATURE_COMPLEXITY: Record<ProblemFraming["task_nature"], number> = {
  deterministic_rules: 1,
  generation_or_summarization: 2,
  judgment_on_language: 3,
  open_ended_investigation: 5,
};
const SEVERITY: Record<ProblemFraming["error_severity"], number> = { low: 1, medium: 2, high: 3, critical: 4 };
// Annual addressable value bands (USD) -> score. Upper bound exclusive; above the last band = 5.
const VALUE_BANDS: [number, number][] = [
  [25_000, 1],
  [100_000, 2],
  [500_000, 3],
  [2_000_000, 4],
];
// The more autonomous the tier, the more value it must create to justify build + run cost.
const TIER_MIN_VALUE = { single_call: 1, workflow: 2, agent: 3 };

type Tier = "single_call" | "workflow" | "agent" | "not_llm";
type Capability = [name: string, why: string];

interface Assessment {
  complexity: number;
  value: number;
  viability: number;
  cost_of_error: number;
  annual_hours: number;
  annual_value_usd: number;
  notes: string[];
}

function assess(f: ProblemFraming): Assessment {
  const notes: string[] = [];
  // Complexity: what kind of thinking the task needs (rules < summarise < judge < investigate).
  let complexity = NATURE_COMPLEXITY[f.task_nature];
  const rich = f.input_modalities.filter((m) => m === "image" || m === "pdf" || m === "audio");
  if (rich.length && complexity < 5) {
    complexity += 1;
    notes.push(`complexity +1: multimodal inputs (${rich.join(", ")})`);
  }
  // Value: size of the prize = human time currently spent x loaded cost.
  const annualHours = ((f.monthly_volume * f.minutes_per_item_today) / 60) * 12;
  const annualValue = annualHours * f.loaded_cost_per_hour_usd;
  const value = VALUE_BANDS.find(([limit]) => annualValue < limit)?.[1] ?? 5;
  if (annualValue === 0) notes.push("value: no human minutes per item to save at runtime");
  // Viability: can an LLM meet the hard constraints with the data available?
  let viability = 5;
  if (f.latency_budget_ms < 200) {
    viability -= 3;
    notes.push(`viability -3: ${f.latency_budget_ms} ms budget is below LLM inference latency`);
  } else if (f.latency_budget_ms < 2000) {
    viability -= 1;
    notes.push(`viability -1: tight ${f.latency_budget_ms} ms budget (small model + streaming)`);
  }
  if (!f.data_available) {
    viability -= 3;
    notes.push("viability -3: required data/systems not accessible");
  }
  viability = Math.max(1, viability);
  // Cost of error: severity, worsened when no human can catch mistakes in time.
  const costOfError = SEVERITY[f.error_severity] + (f.human_review_feasible ? 0 : 1);
  if (!f.human_review_feasible) notes.push("cost of error +1: no practical human checkpoint at this volume/latency");
  return {
    complexity,
    value,
    viability,
    cost_of_error: costOfError,
    annual_hours: annualHours,
    annual_value_usd: annualValue,
    notes,
  };
}

// ---------------------------------------------------------------------------
// 3. Capability mapping: framing facts -> Claude platform capabilities
// ---------------------------------------------------------------------------
const CAPABILITY_MODEL: Record<string, string> = {
  // starting point only - model selection is D2.1
  classification: MODEL_HAIKU,
  "extraction (structured outputs)": MODEL_SONNET,
  vision: MODEL_SONNET,
  "PDF support": MODEL_SONNET,
  "RAG + citations": MODEL_SONNET,
  "tool use": "-",
  "summarization / generation": MODEL_SONNET,
  "agent loop": MODEL_OPUS,
};

function mapCapabilities(f: ProblemFraming): Capability[] {
  const caps: Capability[] = [];
  if (f.task_nature === "judgment_on_language") {
    caps.push(["classification", "assign severity/route by judgment over language"]);
    if (f.data_shape !== "structured")
      caps.push(["extraction (structured outputs)", "turn free text into typed fields for downstream code"]);
  }
  if (f.input_modalities.includes("image")) caps.push(["vision", "read photos/screenshots directly"]);
  if (f.input_modalities.includes("pdf")) caps.push(["PDF support", "read reports/forms without a separate OCR step"]);
  if (f.needs_reference_knowledge) caps.push(["RAG + citations", "ground outputs in policy/runbook text and cite it"]);
  if (f.needs_external_systems) caps.push(["tool use", `typed tools for: ${f.systems_involved.join(", ")}`]);
  if (f.task_nature === "generation_or_summarization")
    caps.push(["summarization / generation", "condense many items into a narrative"]);
  if (f.task_nature === "open_ended_investigation")
    caps.push(["agent loop", "model chooses the next tool call until end_turn"]);
  return caps;
}

// ---------------------------------------------------------------------------
// 4. Tier decision table (first matching rule wins - order matters)
// ---------------------------------------------------------------------------
function decideTier(f: ProblemFraming, a: Assessment, caps: Capability[]): [Tier, string] {
  if (f.task_nature === "deterministic_rules" && f.data_shape === "structured")
    return ["not_llm", "R1 deterministic rules over structured data -> code / rules engine"];
  if (a.viability <= 2) return ["not_llm", "R2 hard constraints (latency/data) an LLM cannot meet"];
  if (a.cost_of_error >= 5) return ["not_llm", "R3 critical errors and no human checkpoint"];
  if (a.complexity >= 4 && !f.steps_predictable) {
    if (a.value >= TIER_MIN_VALUE.agent)
      return ["agent", "R4 open-ended, unpredictable steps, value justifies agent cost"];
    return ["workflow", "R5 open-ended but value too low for an agent -> constrained workflow"];
  }
  if (f.steps_predictable && (caps.length >= 2 || f.needs_external_systems || a.complexity >= 3)) {
    if (a.value >= TIER_MIN_VALUE.workflow)
      return ["workflow", "R6 predictable multi-step / multi-capability -> workflow in code"];
    return ["single_call", "R6b multi-step but value too low -> start with one call"];
  }
  return ["single_call", "R7 one well-scoped transformation -> single LLM call"];
}

// ---------------------------------------------------------------------------
// 5. Blueprint assembly (deterministic) - the artefact you take into design review
// ---------------------------------------------------------------------------
interface Blueprint {
  problem_id: string;
  title: string;
  tier: Tier;
  components: string[];
  controls: string[];
  models: string[];
  kpis: string[];
}

function buildBlueprint(pid: string, title: string, f: ProblemFraming, a: Assessment, tier: Tier, caps: Capability[]): Blueprint {
  const names = caps.map(([n]) => n);
  let components: string[];
  if (tier === "not_llm") {
    components = [
      "Deterministic service (rules engine / tax API / SQL) at runtime",
      "Claude off the runtime path: Claude Code to maintain rule tables, tests and change summaries",
    ];
  } else if (tier === "single_call") {
    components = ["Batch the inputs (code)", `One Claude call: ${names.join(", ")}`, "Deliver output (code)"];
  } else if (tier === "workflow") {
    // Group capabilities into the fewest sensible steps: one "perception" call reads all
    // modalities at once; retrieval and classification are separate, testable steps.
    components = ["Intake + normalisation + PII redaction (code)"];
    const perceive = names.filter((n) => ["extraction (structured outputs)", "vision", "PDF support"].includes(n));
    if (perceive.length) components.push(`Claude step: ${perceive.join(" + ")} -> typed fields`);
    if (names.includes("RAG + citations")) components.push("Retrieval step: RAG + citations over reference documents");
    if (names.includes("classification")) components.push("Claude step: classification (small, fast model)");
    if (names.includes("summarization / generation")) components.push("Claude step: summarization / generation");
    components.push("Routing rules + write-back via typed tool (code controls the flow)");
  } else {
    components = [
      "Trigger -> context pack (alert, service, recent changes) (code)",
      `Agent loop: Claude + read-only tools for ${f.systems_involved.join(", ")}`,
      "Step limit, token budget and transcript logging (harness)",
      "Hypothesis report -> human decides remediation",
    ];
  }
  const controls: string[] = [];
  const joined = f.constraints.join(" ").toLowerCase();
  if (tier !== "not_llm" && a.cost_of_error >= 3) controls.push("Human-in-the-loop before any consequential action (see D5.3)");
  if (tier === "agent") controls.push("Least-privilege tools: read-only by default; side effects gated by approval");
  if (joined.includes("audit"))
    controls.push(
      tier === "not_llm"
        ? "Audit log: inputs, rule-table version, result"
        : "Audit log: inputs, prompt version, model id, outputs, reviewer",
    );
  if (joined.includes("personal") || joined.includes("pii"))
    controls.push("PII minimisation/redaction and retention policy (see D5.4)");
  if (joined.includes("licensed") || joined.includes("regulator"))
    controls.push("Final decision stays with an accountable licensed human");
  controls.push(
    tier !== "not_llm" ? "Eval set with gold labels + go-live quality gate (see D4.2)" : "Golden test suite for every jurisdiction/rule change",
  );
  const models =
    tier !== "not_llm"
      ? [...new Set(names.map((n) => CAPABILITY_MODEL[n] ?? "-").filter((m) => m !== "-"))].sort()
      : [];
  const kpis = [...f.success_kpis];
  if (a.annual_hours) kpis.push(`baseline: ${Math.round(a.annual_hours).toLocaleString("en-US")} h/yr of human handling`);
  return { problem_id: pid, title, tier, components, controls, models, kpis };
}

// ---------------------------------------------------------------------------
// Offline responder: what Claude would return live
// ---------------------------------------------------------------------------
const MOCK_FRAMINGS: Record<string, unknown> = JSON.parse(readFileSync(path.join(DATA, "mock_framings.json"), "utf-8"));

const BLUEPRINT_ONE_PAGER = `Solution blueprint - Motor claims triage (workflow tier)

Problem: triage adjusters spend ~25 minutes per FNOL claim; misroutes add 3-5 days to cycle time.

Design: a workflow, not an agent. The steps are known in advance, so control flow stays in code and
Claude is used only where judgment over unstructured content is needed.
  1. Intake normalises the claim and redacts personal data not needed for triage.
  2. Claude Sonnet reads the description, photos and police report (vision + PDF) and returns typed
     fields through structured outputs.
  3. Claude Haiku classifies severity and target team; the relevant policy wording is retrieved (RAG)
     so every coverage flag cites its clause.
  4. Code applies routing rules and writes to the claims system through a typed tool.
  5. An adjuster confirms or corrects each triage; corrections feed the eval set.

Value: ~60,000 adjuster hours/year addressable (~$3.3M). Target: under 5 minutes of human time per claim.
Controls: human confirmation, full audit log, PII redaction, licensed adjusters keep settlement authority.
Next step: 4-week pilot on 500 historical claims with gold labels; go/no-go on misrouting rate.`;

function responder(req: MockRequest): string {
  const text = req.lastUserText;
  if (text.includes("BLUEPRINT")) return BLUEPRINT_ONE_PAGER;
  if (text.includes("Compute the sales tax")) {
    // A plausible-looking LLM answer that is off by one cent - exactly the failure mode
    // that makes probabilistic generation wrong for exact, audited arithmetic.
    return "Sales tax: $7.25 (combined rate 7.75% applied to $93.47).";
  }
  const match = /Problem (P\d)/.exec(text);
  if (match && MOCK_FRAMINGS[match[1]!]) return JSON.stringify(MOCK_FRAMINGS[match[1]!]);
  return "{}";
}

// ---------------------------------------------------------------------------
// Tiny deterministic rules engine for the P2 contrast (integer cents, half-up rounding)
// ---------------------------------------------------------------------------
const RATES_BP = { state: 625, county: 100, city: 50 }; // basis points: 6.25% + 1.00% + 0.50%

function rulesEngineTaxCents(amountCents: number): number {
  const rateBp = Object.values(RATES_BP).reduce((s, r) => s + r, 0);
  return Math.floor((amountCents * rateBp + 5_000) / 10_000);
}

const yn = (flag: boolean): string => (flag ? "yes" : "no");
const fmt = (n: number): string => Math.round(n).toLocaleString("en-US");

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
banner("D1.1 - Translate Business Problems into Claude Solutions");
const client = getClient(responder);
const problems: { id: string; title: string; sponsor: string; intake_notes: string }[] = JSON.parse(
  readFileSync(path.join(DATA, "problems.json"), "utf-8"),
);

section("1. Intake: four business problems from discovery");
for (const p of problems) console.log(`  ${p.id}  ${p.title.padEnd(46)} sponsor: ${p.sponsor}`);
console.log("  Raw notes are free text - the first job is to turn them into a comparable framing.");

section("2. Structured problem framing (Claude + structured outputs)");
const framings = new Map<string, ProblemFraming>();
for (const p of problems) {
  const msg = await client.messages.parse({
    model: MODEL_SONNET,
    max_tokens: 2048,
    system: FRAMING_SYSTEM,
    // zodOutputFormat -> output_config.format (JSON schema); effort rides in the same object.
    output_config: { effort: "medium", format: zodOutputFormat(ProblemFraming) },
    messages: [{ role: "user", content: `Problem ${p.id} - ${p.title}\n\nInterview notes:\n${p.intake_notes}` }],
  });
  if (msg.stop_reason !== "end_turn" || !msg.parsed_output) {
    console.log(`  ${p.id}: framing failed (stop_reason=${msg.stop_reason}) - send back to discovery`);
    continue;
  }
  const f = msg.parsed_output;
  framings.set(p.id, f);
  console.log(`  ${p.id} pain: ${f.pain_point}`);
  console.log(
    `      users=${f.primary_users.join(", ")} | volume=${fmt(f.monthly_volume)}/mo | ` +
      `today=${f.minutes_per_item_today} min/item @ $${f.loaded_cost_per_hour_usd}/h`,
  );
  console.log(
    `      nature=${f.task_nature} | data=${f.data_shape} (${f.input_modalities.join("+")}) | ` +
      `predictable=${yn(f.steps_predictable)} | latency budget=${fmt(f.latency_budget_ms)} ms`,
  );
  console.log(
    `      severity=${f.error_severity} | human review feasible=${yn(f.human_review_feasible)} | ` +
      `KPIs: ${f.success_kpis.join("; ")}`,
  );
  console.log(`      [${usageLine(msg)}]`);
}

section("3. AI-suitability rubric (deterministic, 1-5 each)");
console.log("  Criteria: complexity (is an LLM needed?), value (is it worth it?), viability (can it meet");
console.log("  the constraints?), cost of error (what happens when it is wrong? higher = riskier)");
console.log(
  `  ${"id".padEnd(4)}${"complexity".padStart(11)}${"value".padStart(7)}${"$/yr addressable".padStart(18)}` +
    `${"viability".padStart(11)}${"cost-of-error".padStart(15)}`,
);
const assessments = new Map<string, Assessment>();
for (const [pid, f] of framings) {
  const a = assess(f);
  assessments.set(pid, a);
  console.log(
    `  ${pid.padEnd(4)}${String(a.complexity).padStart(11)}${String(a.value).padStart(7)}` +
      `${("$" + fmt(a.annual_value_usd)).padStart(18)}${String(a.viability).padStart(11)}${String(a.cost_of_error).padStart(15)}`,
  );
}
for (const [pid, a] of assessments) for (const note of a.notes) console.log(`    ${pid}: ${note}`);

section("4. Capability mapping + tier decision");
const titles = new Map(problems.map((p) => [p.id, p.title]));
const decisions = new Map<string, { tier: Tier; caps: Capability[] }>();
for (const [pid, f] of framings) {
  const caps = mapCapabilities(f);
  const [tier, rule] = decideTier(f, assessments.get(pid)!, caps);
  decisions.set(pid, { tier, caps });
  console.log(`  ${pid} ${titles.get(pid)}`);
  console.log(`     tier -> ${tier.toUpperCase().padEnd(12)} (${rule})`);
  if (tier === "not_llm") console.log("     capabilities -> none at runtime");
  else for (const [name, why] of caps) console.log(`     - ${name.padEnd(32)} ${why}`);
}

section("5. Solution blueprints");
const blueprints = new Map<string, Blueprint>();
for (const [pid, f] of framings) {
  const { tier, caps } = decisions.get(pid)!;
  const bp = buildBlueprint(pid, titles.get(pid)!, f, assessments.get(pid)!, tier, caps);
  blueprints.set(pid, bp);
  console.log(`  [${pid}] ${bp.title} - ${bp.tier.toUpperCase()}`);
  bp.components.forEach((c, i) => console.log(`     ${i + 1}. ${c}`));
  console.log(`     controls: ${bp.controls.join(" | ")}`);
  if (bp.models.length) console.log(`     starting models (validate in D2.1): ${bp.models.join(", ")}`);
  console.log(`     KPIs: ${bp.kpis.join("; ")}`);
}

section("6. Claude drafts the one-pager for the top LLM candidate");
const candidates = [...framings.keys()].filter((pid) => decisions.get(pid)!.tier !== "not_llm");
const top = candidates.reduce((best, pid) =>
  assessments.get(pid)!.annual_value_usd > assessments.get(best)!.annual_value_usd ? pid : best,
);
const { notes: _notes, ...topAssessment } = assessments.get(top)!;
const payload = { blueprint: blueprints.get(top), framing: framings.get(top), assessment: topAssessment };
let msg: Anthropic.Message = await client.messages.create({
  model: MODEL_OPUS,
  max_tokens: 1500,
  system:
    "You are a principal solution architect. Write crisp, decision-ready one-pagers for executives. " +
    "Use only the facts in the payload; do not invent numbers.",
  messages: [
    { role: "user", content: "Draft a BLUEPRINT one-pager (<= 220 words) from this payload:\n" + JSON.stringify(payload, null, 2) },
  ],
});
if (msg.stop_reason === "end_turn") {
  for (const line of textOf(msg).split("\n")) console.log(`  | ${line}`);
} else {
  console.log(`  draft incomplete (stop_reason=${msg.stop_reason}) - raise max_tokens or shorten the payload`);
}
console.log(`  [${usageLine(msg)}]`);

section("7. Why P2 is NOT an LLM problem - the numbers");
const amountCents = 9_347;
const exact = rulesEngineTaxCents(amountCents);
const t0 = performance.now();
msg = await client.messages.create({
  model: MODEL_HAIKU,
  max_tokens: 200,
  messages: [
    {
      role: "user",
      content:
        `Compute the sales tax for an order of $${(amountCents / 100).toFixed(2)} ` +
        "shipped to a jurisdiction with 6.25% state, 1.00% county and 0.50% city tax.",
    },
  ],
});
const llmMs = elapsedMs(t0);
console.log(`  rules engine : $${(exact / 100).toFixed(2)}  (integer cents, half-up rounding, ~microseconds, exact every time)`);
console.log(`  Claude Haiku : ${textOf(msg)}`);
console.log(`                 latency ~${Math.round(llmMs / 10) * 10} ms vs a 50 ms budget; result not guaranteed to the cent`);
const monthlyCalls = framings.get("P2")?.monthly_volume ?? 60_000_000;
const perCall = { input_tokens: 600, output_tokens: 60 } as Anthropic.Usage;
for (const model of [MODEL_HAIKU, MODEL_SONNET]) {
  console.log(
    `  LLM at runtime on ${model.padEnd(18)}: ~$${fmt(costUsd(model, perCall) * monthlyCalls)}/month ` +
      `for ${fmt(monthlyCalls)} calls (600 in / 60 out tokens each)`,
  );
}
console.log("  Verdict: exact, structured, high-volume, sub-50 ms, unreviewable -> rules engine at runtime.");
console.log("  Claude still adds value OFF the hot path (Claude Code to update rate tables and generate tests).");

section("Summary");
for (const pid of framings.keys()) {
  const { tier } = decisions.get(pid)!;
  const a = assessments.get(pid)!;
  console.log(
    `  ${pid} ${titles.get(pid)!.padEnd(46)} -> ${tier.toUpperCase().padEnd(12)} (C${a.complexity} V${a.value} ` +
      `Vi${a.viability} E${a.cost_of_error})`,
  );
}
console.log("  Lesson: start from the business problem, score it transparently, choose the SIMPLEST tier");
console.log("  that works - and be willing to say 'this is not an LLM problem'.");
