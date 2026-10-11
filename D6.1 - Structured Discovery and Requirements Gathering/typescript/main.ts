/**
 * D6.1 - Structured Discovery and Requirements Gathering
 *
 * What this lab builds (a discovery toolkit an architect can actually run):
 *
 *   interview guide (data/interview_guide.json)  +  3 raw stakeholder transcripts
 *     -> Claude extracts ONE typed record per interview (structured outputs, verbatim evidence)
 *     -> code: provenance check (is every quote really in the transcript?)
 *     -> code: merge + de-duplicate requirements across stakeholders (keeps attribution)
 *     -> code: gap analysis - vague NFRs, missing owners, stakeholder conflicts,
 *              guide coverage, missing baselines
 *     -> code: follow-up questions, routed to the right stakeholder
 *     -> Claude drafts value/effort estimates; code applies MoSCoW + value/effort rules
 *        and the DSDM "Must <= 60% of effort" check
 *     -> code: use-case feasibility / readiness scoring
 *     -> code: renders a requirements document (labs_output/requirements_document.md)
 *
 * Design principle: Claude does the language work (reading messy interviews), code does
 * the judgement that must be repeatable and defensible (rules, scoring, routing).
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { MODEL_OPUS, banner, getClient, section, usageLine } from "../../shared/typescript/claudeClient.ts";
import type { MockRequest } from "../../shared/typescript/claudeClient.ts";

const LAB = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const DATA = path.join(LAB, "data");
const OUT = path.join(LAB, "labs_output");

// ---------------------------------------------------------------------------
// 1. Extraction contract. One record per interview keeps provenance: every
//    requirement can be traced to a stakeholder and a verbatim quote.
//    .nullable() = the model must say null rather than invent a number.
// ---------------------------------------------------------------------------
const Topic = z.enum(["human_review_policy", "phase1_scope", "go_live_date", "ai_budget", "latency_target"]);

const InterviewExtraction = z.object({
  stakeholder: z.object({ name: z.string(), role: z.string(), department: z.string() }),
  business_goals: z.array(z.object({ statement: z.string(), metric: z.string().nullable(), target: z.string().nullable() })),
  personas: z.array(z.object({ name: z.string(), needs: z.string() })),
  pain_points: z.array(z.string()),
  volumes: z.array(z.object({ metric: z.string(), value: z.number(), unit: z.string() })),
  data_sources: z.array(
    z.object({
      name: z.string(),
      sensitivity: z.enum(["public", "internal", "confidential", "restricted"]),
      contains_pii: z.boolean(),
    }),
  ),
  integrations: z.array(
    z.object({ system: z.string(), direction: z.enum(["read", "write", "read_write"]), mechanism: z.string() }),
  ),
  functional: z.array(
    z.object({
      id: z.string(),
      statement: z.string(),
      priority_hint: z.enum(["must", "should", "could", "wont", "unspecified"]),
      owner: z.string().nullable().describe("Team/person explicitly named as accountable, else null"),
      related_topic: Topic.nullable(),
      source_quote: z.string().describe("Verbatim sentence from the transcript"),
    }),
  ),
  non_functional: z.array(
    z.object({
      id: z.string(),
      category: z.enum([
        "latency", "availability", "cost", "quality", "security", "privacy", "compliance", "scalability", "usability",
      ]),
      statement: z.string(),
      metric: z.string().nullable(),
      target_value: z.number().nullable().describe("Numeric target ONLY if stated; never guessed"),
      unit: z.string().nullable(),
      source_quote: z.string(),
    }),
  ),
  constraints: z.array(
    z.object({
      category: z.enum(["regulatory", "budget", "technical", "timeline", "organizational"]),
      statement: z.string(),
      source_quote: z.string(),
    }),
  ),
  positions: z.array(
    z.object({
      topic: Topic,
      stance: z.string(),
      normalized_value: z.string().describe("Short comparable label, e.g. 2027-03-01 or exclude_injury_claims"),
    }),
  ),
  success_metrics: z.array(z.object({ name: z.string(), baseline: z.string().nullable(), target: z.string() })),
  escalation_paths: z.array(z.string()),
  risks: z.array(z.string()),
  assumptions: z.array(z.string()),
  open_questions: z.array(z.string()),
});
type InterviewExtraction = z.infer<typeof InterviewExtraction>;
type NonFunctionalReq = InterviewExtraction["non_functional"][number];

const PrioritizationDraft = z.object({
  estimates: z.array(
    z.object({
      req_id: z.string(),
      value: z.number().int().min(1).max(5).describe("Business value 1-5"),
      effort: z.number().int().min(1).max(5).describe("Relative delivery effort 1-5"),
      moscow: z.enum(["must", "should", "could", "wont"]),
      rationale: z.string(),
    }),
  ),
  notes: z.string(),
});

const EXTRACTION_SYSTEM = `You are a senior requirements analyst on a Claude solution-architecture team.
Turn one raw stakeholder interview transcript into a typed requirements record.
Rules:
- Extract only what this stakeholder actually said. Put the supporting sentence, copied verbatim, in source_quote.
- Never invent numbers. If a non-functional requirement has no measurable target in the transcript,
  set metric, target_value and unit to null. Vague words ("fast", "accurate", "scale") stay vague.
- owner = the team or person explicitly named as accountable; null if unstated or uncertain.
- positions: the stakeholder's stance on these decision topics, only if expressed: human_review_policy,
  phase1_scope, go_live_date, ai_budget, latency_target. normalized_value is a short comparable label.
- escalation_paths: what should happen when the AI is unsure or wrong (empty if not discussed).
- open_questions: anything the stakeholder was unsure about or deferred.`;

const PRIORITIZATION_SYSTEM = `You are the delivery lead on a Claude solution team. For each requirement give a
relative business value (1-5) and delivery effort (1-5), and a MoSCoW suggestion with a one-line rationale.
Respect regulatory constraints: a requirement that contradicts a stated regulatory constraint cannot be a Must.
Flag requirements that depend on open stakeholder conflicts in the rationale. Estimates are a draft for the team.`;

// ---------------------------------------------------------------------------
// 2. Offline responder: canned Claude replies live in data/mock_responses so
//    Python and TypeScript share the exact same fixtures.
// ---------------------------------------------------------------------------
function responder(req: MockRequest): string {
  if (req.systemText.includes("requirements analyst")) {
    const m = /Interview ID: (INT-\d+)/.exec(req.lastUserText);
    const iid = m ? m[1] : "INT-01";
    return fs.readFileSync(path.join(DATA, "mock_responses", `extraction_${iid}.json`), "utf-8");
  }
  if (req.systemText.includes("delivery lead")) {
    return fs.readFileSync(path.join(DATA, "mock_responses", "prioritization.json"), "utf-8");
  }
  return "{}";
}

// ---------------------------------------------------------------------------
// 3. Small deterministic helpers (identical logic in the Python version)
// ---------------------------------------------------------------------------
const STOPWORDS = new Set([
  "the", "and", "for", "from", "with", "into", "that", "this", "our", "any", "all", "each", "every",
  "before", "after", "not", "must", "should", "can", "will", "using", "their", "they", "them", "its",
]);

const normText = (s: string): string => s.replace(/\s+/g, " ").trim().toLowerCase();

function contentTokens(s: string): Set<string> {
  const out = new Set<string>();
  for (let t of s.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (t.length < 3 || STOPWORDS.has(t)) continue;
    if (t.length > 4 && t.endsWith("s")) t = t.slice(0, -1); // naive plural stemming: letters -> letter
    out.add(t);
  }
  return out;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  const union = new Set([...a, ...b]);
  const inter = [...a].filter((t) => b.has(t)).length;
  return union.size ? inter / union.size : 0;
}

function fill(template: string, values: Record<string, string>): string {
  let out = template;
  for (const [k, v] of Object.entries(values)) out = out.split(`{${k}}`).join(v);
  return out;
}

const escapeRe = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function findVagueTerm(text: string, terms: string[]): string | null {
  const low = text.toLowerCase();
  for (const term of terms) {
    if (new RegExp(`(?<![a-z])${escapeRe(term)}(?![a-z])`).test(low)) return term;
  }
  return null;
}

const trunc = (s: string, n: number): string => (s.length <= n ? s : s.slice(0, n - 3) + "...");
const mdCell = (s: string): string => s.replace(/\|/g, "/");
const num = (n: number): string => String(n); // same as Python's :g for the values used here
const cmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

// ---------------------------------------------------------------------------
// 4. Merged (cross-stakeholder) view
// ---------------------------------------------------------------------------
const DEDUPE_THRESHOLD = 0.5;

interface MergedFR {
  id: string;
  statement: string;
  owner: string | null;
  relatedTopic: string | null;
  tokens: Set<string>;
  sources: string[]; // "INT-01/FR-1"
  interviews: string[];
  hints: string[];
  quotes: [string, string][]; // [interview, quote]
}

interface MergedNFR {
  id: string;
  interview: string;
  localId: string;
  item: NonFunctionalReq;
  measurable: boolean;
}

interface Gap {
  kind: string;
  severity: string;
  ref: string;
  detail: string;
  ask: string; // interview id, "joint" or "all"
  question: string;
}

interface Conflict {
  id: string;
  topic: string;
  label: string;
  severity: string;
  owner: string;
  hint: string;
  positions: [string, string, string][]; // [interview, stance, normalized_value]
}

interface Prioritized {
  fr: MergedFR;
  value: number;
  effort: number;
  moscow: string;
  rationale: string;
  quadrant: string;
  score: number;
  blockedBy: string | null;
}

interface UseCaseScore {
  id: string;
  name: string;
  value: number;
  feasibility: number;
  risk: number;
  readiness: number;
  decision: string;
  blockers: string[];
}

interface ConflictTopicMeta {
  label: string;
  severity: string;
  decision_owner: string;
  compromise_hint: string;
}
interface GuideSection {
  id: string;
  title: string;
  why: string;
  evidence_fields: string[];
  questions: string[];
}
interface Guide {
  project: string;
  client: string;
  as_of: string;
  sponsor_interview: string;
  interviews: { id: string; file: string }[];
  sections: GuideSection[];
  conflict_topics: Record<string, ConflictTopicMeta>;
  measurability: { categories_requiring_metric: string[]; vague_terms: string[] };
  followup_templates: Record<string, string>;
  prioritization: { must_effort_share_max: number; high_value_min: number; low_effort_max: number };
}
interface UseCase {
  id: string;
  name: string;
  value: number;
  data_readiness: number;
  technical_fit: number;
  integration_complexity: number;
  data_sensitivity: number;
  error_impact: number;
  regulatory_exposure: number;
  human_in_loop: boolean;
  measurable_success_metric: boolean;
  hard_constraint_conflicts: string[];
}
interface UseCaseFile {
  weights: { value: number; feasibility: number; safety: number };
  thresholds: { go_readiness: number; conditional_readiness: number; min_data_readiness: number; max_risk_for_go: number };
  use_cases: UseCase[];
}

type QuoteCheck = [string, string, string, boolean]; // [interview, ref, quote, verified]
type Dsdm = { mustEffort: number; totalEffort: number; share: number; limit: number; ok: boolean; demotions: string[] };

const MOSCOW_RANK: Record<string, number> = { must: 0, should: 1, could: 2, wont: 3 };
const MOSCOW_LABEL: Record<string, string> = { must: "Must", should: "Should", could: "Could", wont: "Won't" };

function mergeFunctional(extractions: Map<string, InterviewExtraction>): [MergedFR[], string[]] {
  const merged: MergedFR[] = [];
  const log: string[] = [];
  for (const [iid, ex] of extractions) {
    for (const fr of ex.functional) {
      const toks = contentTokens(fr.statement);
      let best: MergedFR | null = null;
      let bestSim = 0;
      for (const m of merged) {
        const sim = jaccard(toks, m.tokens);
        if (sim > bestSim) {
          best = m;
          bestSim = sim;
        }
      }
      if (best !== null && bestSim >= DEDUPE_THRESHOLD) {
        best.sources.push(`${iid}/${fr.id}`);
        if (!best.interviews.includes(iid)) best.interviews.push(iid);
        best.hints.push(fr.priority_hint);
        best.quotes.push([iid, fr.source_quote]);
        best.owner = best.owner ?? fr.owner;
        best.relatedTopic = best.relatedTopic ?? fr.related_topic;
        log.push(`${iid}/${fr.id} merged into ${best.id} (similarity ${bestSim.toFixed(2)}) - same need, 2nd source`);
        continue;
      }
      merged.push({
        id: `FR-${String(merged.length + 1).padStart(2, "0")}`,
        statement: fr.statement,
        owner: fr.owner,
        relatedTopic: fr.related_topic,
        tokens: toks,
        sources: [`${iid}/${fr.id}`],
        interviews: [iid],
        hints: [fr.priority_hint],
        quotes: [[iid, fr.source_quote]],
      });
    }
  }
  return [merged, log];
}

function detectConflicts(extractions: Map<string, InterviewExtraction>, guide: Guide): Conflict[] {
  const conflicts: Conflict[] = [];
  for (const [topic, meta] of Object.entries(guide.conflict_topics)) {
    const positions: [string, string, string][] = [];
    for (const [iid, ex] of extractions) {
      for (const p of ex.positions) if (p.topic === topic) positions.push([iid, p.stance, p.normalized_value]);
    }
    const distinct: string[] = [];
    for (const [, , v] of positions) if (!distinct.includes(v)) distinct.push(v);
    if (distinct.length >= 2) {
      conflicts.push({
        id: `C-${conflicts.length + 1}`,
        topic,
        label: meta.label,
        severity: meta.severity,
        owner: meta.decision_owner,
        hint: meta.compromise_hint,
        positions,
      });
    }
  }
  return conflicts;
}

function coverageMatrix(extractions: Map<string, InterviewExtraction>, guide: Guide): Map<string, Map<string, number>> {
  const matrix = new Map<string, Map<string, number>>();
  for (const s of guide.sections) {
    const row = new Map<string, number>();
    for (const [iid, ex] of extractions) {
      let n = 0;
      for (const f of s.evidence_fields) n += (ex[f as keyof InterviewExtraction] as unknown[]).length;
      row.set(iid, n);
    }
    matrix.set(s.id, row);
  }
  return matrix;
}

const sumRow = (row: Map<string, number>): number => [...row.values()].reduce((a, b) => a + b, 0);
const coverageStatus = (row: Map<string, number>): string =>
  sumRow(row) === 0 ? "GAP" : [...row.values()].filter((v) => v > 0).length === 1 ? "single-source" : "ok";

async function main(): Promise<void> {
  banner("D6.1 - Structured Discovery and Requirements Gathering");
  const guide: Guide = JSON.parse(fs.readFileSync(path.join(DATA, "interview_guide.json"), "utf-8"));
  const useCases: UseCaseFile = JSON.parse(fs.readFileSync(path.join(DATA, "use_cases.json"), "utf-8"));
  const tpl = guide.followup_templates;
  const client = getClient(responder);

  // -- 1. Toolkit -----------------------------------------------------------
  section("1. Discovery toolkit: structured interview guide");
  console.log(`Project: ${guide.project} | Client: ${guide.client} | As of ${guide.as_of}`);
  for (const s of guide.sections) {
    console.log(`  [${s.id.padEnd(17)}] ${s.title.padEnd(52)} ${s.questions.length} q -> ${s.evidence_fields.join(", ")}`);
  }
  console.log("Each guide section maps to typed schema fields, so coverage is measured - not remembered.");

  // -- 2. Extraction (Claude, structured outputs) -----------------------------
  section("2. Claude extracts a typed record per interview");
  const transcripts = new Map<string, string>();
  const extractions = new Map<string, InterviewExtraction>();
  for (const it of guide.interviews) {
    const iid = it.id;
    const text = fs.readFileSync(path.join(DATA, "transcripts", it.file), "utf-8");
    transcripts.set(iid, text);
    // Strongest model: low volume (a handful of interviews), high stakes (it shapes the whole project).
    const resp = await client.messages.parse({
      model: MODEL_OPUS,
      max_tokens: 8000,
      thinking: { type: "adaptive" },
      output_config: { effort: "medium", format: zodOutputFormat(InterviewExtraction) },
      system: EXTRACTION_SYSTEM,
      messages: [
        {
          role: "user",
          content: `Extract the requirements record for this interview.\n\n<transcript>\n${text}\n</transcript>`,
        },
      ],
    });
    const ex = resp.parsed_output;
    if (!ex) throw new Error(`${iid}: structured output did not parse`);
    extractions.set(iid, ex);
    console.log(
      `  ${iid} ${ex.stakeholder.name.padEnd(16)} ${ex.functional.length} FR, ${ex.non_functional.length} NFR, ` +
        `${ex.constraints.length} constraints, ${ex.positions.length} positions | ${usageLine(resp)}`,
    );
  }
  const iids = [...extractions.keys()];
  const names = new Map(iids.map((iid) => [iid, extractions.get(iid)!.stakeholder.name]));
  const nameOf = (iid: string): string => names.get(iid) ?? iid;

  // -- 3. Provenance + merge ---------------------------------------------------
  section("3. Provenance check and cross-stakeholder merge");
  const quoteChecks: QuoteCheck[] = [];
  for (const [iid, ex] of extractions) {
    const body = normText(transcripts.get(iid)!);
    const items: [string, string][] = [
      ...ex.functional.map((f): [string, string] => [f.id, f.source_quote]),
      ...ex.non_functional.map((n): [string, string] => [n.id, n.source_quote]),
      ...ex.constraints.map((c, i): [string, string] => [`CON-${i + 1}`, c.source_quote]),
    ];
    for (const [ref, quote] of items) quoteChecks.push([iid, `${iid}/${ref}`, quote, body.includes(normText(quote))]);
  }
  const verified = quoteChecks.filter((q) => q[3]).length;
  console.log(`Evidence quotes found verbatim in transcripts: ${verified}/${quoteChecks.length}`);
  for (const [, ref, quote, ok] of quoteChecks) {
    if (!ok) console.log(`  ! ${ref}: "${quote}" is a paraphrase - not verbatim (flagged for confirmation)`);
  }

  const [merged, mergeLog] = mergeFunctional(extractions);
  const totalFr = [...extractions.values()].reduce((a, ex) => a + ex.functional.length, 0);
  console.log(
    `Functional requirements: ${totalFr} extracted -> ${merged.length} after de-duplication ` +
      `(Jaccard >= ${DEDUPE_THRESHOLD.toFixed(2)} on content words)`,
  );
  for (const line of mergeLog) console.log(`  ${line}`);
  const nfrs: MergedNFR[] = [];
  for (const [iid, ex] of extractions) {
    for (const n of ex.non_functional) {
      nfrs.push({ id: `NFR-${String(nfrs.length + 1).padStart(2, "0")}`, interview: iid, localId: n.id, item: n, measurable: true });
    }
  }

  // -- 4. Gap analysis -----------------------------------------------------
  section("4. Gap analysis");
  const meas = guide.measurability;
  const gaps: Gap[] = [];
  const conflicts = detectConflicts(extractions, guide);
  for (const c of conflicts) {
    const [a, b] = [c.positions[0]!, c.positions[1]!];
    const q = fill(tpl.conflict!, {
      a_name: nameOf(a[0]), a_stance: a[1], b_name: nameOf(b[0]), b_stance: b[1], owner: c.owner, hint: c.hint,
    });
    gaps.push({
      kind: "conflict", severity: c.severity, ref: c.id,
      detail: `${c.label}: ` + c.positions.map((p) => p[2]).join(" vs "), ask: "joint", question: q,
    });
  }

  const matrix = coverageMatrix(extractions, guide);
  for (const s of guide.sections) {
    if (sumRow(matrix.get(s.id)!) === 0) {
      const q = fill(tpl.coverage!, { title: s.title, question: s.questions[0]! });
      gaps.push({ kind: "coverage", severity: "high", ref: s.id, detail: `No interview covered '${s.title}'`, ask: "all", question: q });
    }
  }

  for (const n of nfrs) {
    const it = n.item;
    if (it.target_value !== null) continue;
    const term = findVagueTerm(`${it.statement} ${it.source_quote}`, meas.vague_terms);
    if (term !== null) {
      n.measurable = false;
      const q = fill(tpl.vague_nfr!, { quote: it.source_quote, category: it.category });
      gaps.push({
        kind: "vague_nfr", severity: "high", ref: n.id,
        detail: `'${term}' with no target (${it.category}): ${it.statement}`, ask: n.interview, question: q,
      });
    } else if (meas.categories_requiring_metric.includes(it.category)) {
      n.measurable = false;
      const q = fill(tpl.unmeasured_nfr!, { statement: it.statement, category: it.category });
      gaps.push({
        kind: "unmeasured_nfr", severity: "medium", ref: n.id,
        detail: `${it.category} requirement without metric: ${it.statement}`, ask: n.interview, question: q,
      });
    }
  }

  for (const [iid, ref, quote, ok] of quoteChecks) {
    if (!ok) {
      const q = fill(tpl.unverified_quote!, { ref, quote });
      gaps.push({ kind: "unverified_quote", severity: "medium", ref, detail: "Evidence quote not verbatim in transcript", ask: iid, question: q });
    }
  }

  for (const m of merged) {
    if (m.owner === null) {
      const q = fill(tpl.missing_owner!, { req_id: m.id, statement: m.statement });
      gaps.push({
        kind: "missing_owner", severity: "medium", ref: m.id, detail: `No accountable owner: ${m.statement}`,
        ask: guide.sponsor_interview, question: q,
      });
    }
  }

  for (const [iid, ex] of extractions) {
    for (const sm of ex.success_metrics) {
      if (sm.baseline === null) {
        const q = fill(tpl.no_baseline!, { metric: sm.name });
        gaps.push({ kind: "no_baseline", severity: "medium", ref: iid, detail: `Success metric without baseline: ${sm.name}`, ask: iid, question: q });
      }
    }
  }

  const kinds = ["conflict", "coverage", "vague_nfr", "unmeasured_nfr", "unverified_quote", "missing_owner", "no_baseline"];
  for (const k of kinds) {
    const items = gaps.filter((g) => g.kind === k);
    if (items.length) console.log(`  ${k.padEnd(17)} x${items.length}: ` + items.map((g) => g.ref).join(", "));
  }
  console.log("\nStakeholder conflicts (same topic, different positions):");
  for (const c of conflicts) {
    console.log(`  ${c.id} [${c.severity.padEnd(6)}] ${c.label}`);
    for (const [iid, , value] of c.positions) console.log(`      ${iid} ${nameOf(iid).padEnd(16)} -> ${value}`);
  }
  console.log("\nGuide coverage (evidence items per interview):");
  console.log(`  ${"section".padEnd(18)}` + iids.map((i) => i.padStart(8)).join("") + "   status");
  for (const s of guide.sections) {
    const row = matrix.get(s.id)!;
    console.log(`  ${s.id.padEnd(18)}` + iids.map((i) => String(row.get(i)).padStart(8)).join("") + `   ${coverageStatus(row)}`);
  }

  // -- 5. Follow-up questions ---------------------------------------------------
  section("5. Follow-up questions, routed to who can answer them");
  const audiences = [...iids, "joint", "all"];
  const audienceLabel = new Map<string, string>(iids.map((iid) => [iid, `${nameOf(iid)} (${extractions.get(iid)!.stakeholder.role})`]));
  audienceLabel.set("joint", "Joint decision session (steering committee)");
  audienceLabel.set("all", "All stakeholders (next workshop)");
  const followups = new Map<string, Gap[]>(audiences.map((a) => [a, gaps.filter((g) => g.ask === a)]));
  for (const a of audiences) {
    const items = followups.get(a)!;
    if (!items.length) continue;
    console.log(`${audienceLabel.get(a)} - ${items.length} question(s)`);
    items.forEach((g, i) => console.log(`  ${i + 1}. [${g.ref}] ${trunc(g.question, 150)}`));
  }

  // -- 6. Prioritization ---------------------------------------------------------
  section("6. Prioritization: Claude drafts estimates, code applies the rules");
  const openTopics = new Map(conflicts.map((c) => [c.topic, c]));
  const payload = {
    requirements: merged.map((m) => ({
      id: m.id,
      statement: m.statement,
      stakeholder_priority_hints: m.hints,
      sources: m.sources,
      depends_on_open_conflict: m.relatedTopic !== null && openTopics.has(m.relatedTopic),
    })),
    business_goals: [...extractions.values()].flatMap((ex) => ex.business_goals.map((g) => g.statement)),
    regulatory_constraints: [...extractions.values()].flatMap((ex) =>
      ex.constraints.filter((c) => c.category === "regulatory").map((c) => c.statement),
    ),
  };
  const presp = await client.messages.parse({
    model: MODEL_OPUS,
    max_tokens: 6000,
    thinking: { type: "adaptive" },
    output_config: { effort: "medium", format: zodOutputFormat(PrioritizationDraft) },
    system: PRIORITIZATION_SYSTEM,
    messages: [{ role: "user", content: "Estimate these requirements:\n" + JSON.stringify(payload, null, 2) }],
  });
  const draft = presp.parsed_output;
  if (!draft) throw new Error("prioritization did not parse");
  console.log(`  Claude estimates received | ${usageLine(presp)}`);
  const byId = new Map(draft.estimates.map((e) => [e.req_id, e]));
  const mergedIds = new Set(merged.map((m) => m.id));
  const missing = merged.filter((m) => !byId.has(m.id)).map((m) => m.id);
  const unknown = draft.estimates.filter((e) => !mergedIds.has(e.req_id)).map((e) => e.req_id);
  const listOrNone = (xs: string[]): string => (xs.length ? `[${xs.map((x) => `'${x}'`).join(", ")}]` : "none");
  console.log(`  Validation: ${byId.size} estimates, missing=${listOrNone(missing)}, unknown ids=${listOrNone(unknown)}`);

  const prCfg = guide.prioritization;
  const prioritized: Prioritized[] = [];
  for (const m of merged) {
    const e = byId.get(m.id);
    if (!e) continue;
    const hv = e.value >= prCfg.high_value_min;
    const le = e.effort <= prCfg.low_effort_max;
    const quadrant = hv && le ? "Quick win" : hv ? "Major project" : le ? "Fill-in" : "Money pit";
    const blocked = m.relatedTopic !== null && openTopics.has(m.relatedTopic) ? openTopics.get(m.relatedTopic)!.label : null;
    prioritized.push({ fr: m, value: e.value, effort: e.effort, moscow: e.moscow, rationale: e.rationale, quadrant, score: e.value / e.effort, blockedBy: blocked });
  }
  prioritized.sort((a, b) => MOSCOW_RANK[a.moscow]! - MOSCOW_RANK[b.moscow]! || b.score - a.score || cmp(a.fr.id, b.fr.id));

  console.log(`\n  ${"ID".padEnd(6)}${"MoSCoW".padEnd(8)}${"V".padStart(2)}${"E".padStart(3)}${"V/E".padStart(6)}  ${"Quadrant".padEnd(14)}${"Owner".padEnd(19)}Requirement`);
  for (const p of prioritized) {
    const flag = p.blockedBy ? "  <- DECISION NEEDED" : "";
    console.log(
      `  ${p.fr.id.padEnd(6)}${p.moscow.toUpperCase().padEnd(8)}${String(p.value).padStart(2)}${String(p.effort).padStart(3)}` +
        `${p.score.toFixed(2).padStart(6)}  ${p.quadrant.padEnd(14)}${trunc(p.fr.owner ?? "(none)", 18).padEnd(19)}${trunc(p.fr.statement, 52)}${flag}`,
    );
  }

  const inScope = prioritized.filter((p) => p.moscow !== "wont");
  const totalEffort = inScope.reduce((a, p) => a + p.effort, 0);
  const mustEffort = inScope.filter((p) => p.moscow === "must").reduce((a, p) => a + p.effort, 0);
  const share = totalEffort ? mustEffort / totalEffort : 0;
  const limit = prCfg.must_effort_share_max;
  const dsdmOk = share <= limit;
  console.log(
    `\n  DSDM check: Must = ${mustEffort}/${totalEffort} effort points = ${(share * 100).toFixed(1)}% ` +
      `(guideline <= ${(limit * 100).toFixed(0)}%) -> ${dsdmOk ? "OK" : "OVER - no contingency left"}`,
  );
  const demotions: string[] = [];
  if (!dsdmOk) {
    // Demote contested Musts first (largest effort first), then the lowest value/effort Musts.
    const musts = inScope.filter((p) => p.moscow === "must");
    const candidates = [
      ...musts.filter((p) => p.blockedBy).sort((a, b) => b.effort - a.effort || cmp(a.fr.id, b.fr.id)),
      ...musts.filter((p) => !p.blockedBy).sort((a, b) => a.score - b.score || cmp(a.fr.id, b.fr.id)),
    ];
    let remaining = mustEffort;
    for (const p of candidates) {
      if (remaining / totalEffort <= limit) break;
      remaining -= p.effort;
      demotions.push(p.fr.id);
    }
    console.log(
      `  Recommendation: move ${demotions.join(", ")} from Must to phase 2 / Should -> Must share ` +
        `${((remaining / totalEffort) * 100).toFixed(1)}%`,
    );
  }

  // -- 7. Use-case feasibility ------------------------------------------------
  section("7. Use-case feasibility and readiness scoring");
  const w = useCases.weights;
  const th = useCases.thresholds;
  const ucScores: UseCaseScore[] = [];
  for (const uc of useCases.use_cases) {
    const feasibility = (uc.data_readiness + uc.technical_fit + (6 - uc.integration_complexity)) / 3;
    // A human in the loop reduces the *effective* impact of an error (but never below 1).
    const err = uc.human_in_loop ? Math.max(1, uc.error_impact - 2) : uc.error_impact;
    const risk = (uc.data_sensitivity + err + uc.regulatory_exposure) / 3;
    const readiness = w.value * uc.value + w.feasibility * feasibility + w.safety * (6 - risk);
    let blockers: string[] = [];
    let decision: string;
    if (uc.hard_constraint_conflicts.length) {
      decision = "NO-GO";
      blockers = uc.hard_constraint_conflicts.map((c) => `violates hard constraint: ${c}`);
    } else {
      if (uc.data_readiness < th.min_data_readiness) blockers.push(`data readiness ${uc.data_readiness} < ${th.min_data_readiness}`);
      if (!uc.measurable_success_metric) blockers.push("no measurable success metric");
      if (risk >= th.max_risk_for_go) blockers.push(`risk ${risk.toFixed(2)} >= ${th.max_risk_for_go.toFixed(1)}: mitigation plan needed`);
      if (readiness >= th.go_readiness && !blockers.length) decision = "GO (pilot)";
      else if (readiness >= th.conditional_readiness) decision = "CONDITIONAL";
      else decision = "NOT NOW";
    }
    ucScores.push({ id: uc.id, name: uc.name, value: uc.value, feasibility, risk, readiness, decision, blockers });
  }
  console.log(`  readiness = ${w.value}*value + ${w.feasibility}*feasibility + ${w.safety}*(6 - risk)   (1-5 scale)`);
  console.log(`  ${"UC".padEnd(6)}${"Use case".padEnd(52)}${"Val".padStart(4)}${"Feas".padStart(6)}${"Risk".padStart(6)}${"Ready".padStart(7)}  Decision`);
  for (const u of ucScores) {
    console.log(
      `  ${u.id.padEnd(6)}${trunc(u.name, 50).padEnd(52)}${String(u.value).padStart(4)}${u.feasibility.toFixed(2).padStart(6)}` +
        `${u.risk.toFixed(2).padStart(6)}${u.readiness.toFixed(2).padStart(7)}  ${u.decision}`,
    );
    for (const b of u.blockers) console.log(`        - ${b}`);
  }

  // -- 8. Discovery exit criteria + document -------------------------------------
  section("8. Discovery exit check and requirements document");
  const mustUnowned = prioritized.filter((p) => p.moscow === "must" && p.fr.owner === null).map((p) => p.fr.id);
  const exitChecks: [string, boolean][] = [
    ["No open high-severity stakeholder conflicts", !conflicts.some((c) => c.severity === "high")],
    ["Every interview-guide section covered", !gaps.some((g) => g.kind === "coverage")],
    ["Every NFR measurable (metric + target)", nfrs.every((n) => n.measurable)],
    ["Every Must requirement has an owner", mustUnowned.length === 0],
    ["MoSCoW within DSDM effort guideline", dsdmOk],
    ["All evidence quotes verified", verified === quoteChecks.length],
  ];
  for (const [label, ok] of exitChecks) console.log(`  [${ok ? "PASS" : "FAIL"}] ${label}`);
  const exitMet = exitChecks.every(([, ok]) => ok);
  console.log(
    `  => Discovery exit ${exitMet ? "MET" : "NOT MET"}: ` +
      `${exitChecks.filter(([, ok]) => !ok).length} criteria open - schedule the follow-ups above.`,
  );

  const doc = renderDocument(
    guide, extractions, merged, nfrs, conflicts, gaps, followups, audienceLabel, prioritized,
    { mustEffort, totalEffort, share, limit, ok: dsdmOk, demotions }, ucScores, useCases, matrix, quoteChecks, exitChecks,
  );
  fs.mkdirSync(OUT, { recursive: true });
  fs.writeFileSync(path.join(OUT, "requirements_document.md"), doc, "utf-8");
  const lineCount = doc.split("\n").length - (doc.endsWith("\n") ? 1 : 0);
  console.log(`\nWrote labs_output/requirements_document.md (${lineCount} lines)`);
  console.log("Sample output committed at examples/requirements_document.md");
}

// ---------------------------------------------------------------------------
// 5. Requirements document (deterministic template rendering)
// ---------------------------------------------------------------------------
function renderDocument(
  guide: Guide,
  extractions: Map<string, InterviewExtraction>,
  merged: MergedFR[],
  nfrs: MergedNFR[],
  conflicts: Conflict[],
  gaps: Gap[],
  followups: Map<string, Gap[]>,
  audienceLabel: Map<string, string>,
  prioritized: Prioritized[],
  dsdm: Dsdm,
  ucScores: UseCaseScore[],
  useCases: UseCaseFile,
  matrix: Map<string, Map<string, number>>,
  quoteChecks: QuoteCheck[],
  exitChecks: [string, boolean][],
): string {
  const iids = [...extractions.keys()];
  const nameOf = (iid: string): string => extractions.get(iid)!.stakeholder.name;
  const counts: Record<string, number> = {};
  for (const k of Object.keys(MOSCOW_RANK)) counts[k] = prioritized.filter((p) => p.moscow === k).length;
  const measurable = nfrs.filter((n) => n.measurable).length;
  const nConstraints = [...extractions.values()].reduce((a, ex) => a + ex.constraints.length, 0);
  const highConflicts = conflicts.filter((c) => c.severity === "high").length;
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  a(`# Requirements Document - ${guide.project}`);
  a("");
  a(`**Client:** ${guide.client}  `);
  a(`**Status:** DRAFT v0.1 (discovery)  `);
  a(`**As of:** ${guide.as_of}  `);
  a(`**Sources:** ${iids.length} stakeholder interviews (${iids.join(", ")})`);
  a("");
  a("> Generated by the D6.1 lab. Claude extracted one typed record per interview (structured outputs);");
  a("> merging, gap analysis, prioritization rules and scoring are deterministic code. Estimates are drafts");
  a("> for the delivery team to confirm.");
  a("");
  a("## 1. Summary");
  a("");
  a(
    `- **${merged.length} functional requirements**: ${counts.must} Must, ${counts.should} Should, ` +
      `${counts.could} Could, ${counts.wont} Won't (this phase)`,
  );
  a(`- **${nfrs.length} non-functional requirements**: ${measurable} measurable, ${nfrs.length - measurable} need a target`);
  a(
    `- **${nConstraints} constraints**, **${conflicts.length} stakeholder conflicts** (${highConflicts} high severity), ` +
      `**${gaps.length} open gaps**`,
  );
  const exitOpen = exitChecks.filter(([, ok]) => !ok).map(([label]) => label);
  a(`- **Discovery exit: ${exitOpen.length ? "NOT MET" : "MET"}** - open: ` + (exitOpen.length ? exitOpen.join("; ") : "none"));
  a("");
  a("## 2. Stakeholders");
  a("");
  a("| Interview | Name | Role | Department |");
  a("|---|---|---|---|");
  for (const [iid, ex] of extractions) {
    const s = ex.stakeholder;
    a(`| ${iid} | ${s.name} | ${s.role} | ${s.department} |`);
  }
  a("");
  a("## 3. Business goals and success metrics");
  a("");
  a("| Goal | Metric | Target | Source |");
  a("|---|---|---|---|");
  for (const [iid, ex] of extractions) {
    for (const g of ex.business_goals) {
      a(`| ${mdCell(g.statement)} | ${g.metric ?? "_not stated_"} | ${g.target ?? "_not stated_"} | ${iid} |`);
    }
  }
  a("");
  a("| Success metric | Baseline | Target | Source |");
  a("|---|---|---|---|");
  for (const [iid, ex] of extractions) {
    for (const sm of ex.success_metrics) a(`| ${sm.name} | ${sm.baseline ?? "**MISSING**"} | ${sm.target} | ${iid} |`);
  }
  a("");
  a("## 4. Functional requirements (prioritized)");
  a("");
  a(
    "MoSCoW from the delivery-lead draft; value/effort on a 1-5 scale; quadrant: Quick win (V>=4, E<=2), " +
      "Major project (V>=4, E>=3), Fill-in (V<4, E<=2), Money pit (V<4, E>=3).",
  );
  a("");
  a("| ID | Requirement | MoSCoW | Value | Effort | V/E | Quadrant | Owner | Sources | Status |");
  a("|---|---|---|---:|---:|---:|---|---|---|---|");
  for (const p of prioritized) {
    const status = p.blockedBy ? `Decision needed: ${p.blockedBy}` : "Ready";
    a(
      `| ${p.fr.id} | ${mdCell(p.fr.statement)} | ${MOSCOW_LABEL[p.moscow]} | ${p.value} | ${p.effort} | ${p.score.toFixed(2)} | ` +
        `${p.quadrant} | ${p.fr.owner ?? "**MISSING**"} | ${p.fr.sources.join(", ")} | ${status} |`,
    );
  }
  a("");
  const verdict = dsdm.ok ? "within guideline" : "OVER guideline - no contingency left";
  a(
    `**DSDM check:** Must = ${dsdm.mustEffort}/${dsdm.totalEffort} effort points = ${(dsdm.share * 100).toFixed(1)}% ` +
      `(guideline <= ${(dsdm.limit * 100).toFixed(0)}%) - ${verdict}.`,
  );
  if (dsdm.demotions.length) a(`Recommendation: move ${dsdm.demotions.join(", ")} out of Must (contested and/or lowest value per effort).`);
  a("");
  a("## 5. Non-functional requirements");
  a("");
  a("| ID | Category | Requirement | Target | Measurable | Source |");
  a("|---|---|---|---|---|---|");
  for (const n of nfrs) {
    const it = n.item;
    const target = it.target_value !== null ? `${num(it.target_value)} ${it.unit ?? ""}`.trim() : "_none_";
    a(`| ${n.id} | ${it.category} | ${mdCell(it.statement)} | ${target} | ${n.measurable ? "yes" : "**NO**"} | ${n.interview}/${n.localId} |`);
  }
  a("");
  a("## 6. Constraints");
  a("");
  a("| Category | Constraint | Source |");
  a("|---|---|---|");
  for (const [iid, ex] of extractions) {
    for (const c of ex.constraints) a(`| ${c.category} | ${mdCell(c.statement)} | ${iid} |`);
  }
  a("");
  a("## 7. Conflicts and decisions needed");
  a("");
  for (const c of conflicts) {
    a(`### ${c.id} - ${c.label} (severity: ${c.severity})`);
    a("");
    a("| Stakeholder | Position | Normalized |");
    a("|---|---|---|");
    for (const [iid, stance, value] of c.positions) a(`| ${nameOf(iid)} (${iid}) | ${mdCell(stance)} | \`${value}\` |`);
    a("");
    a(`- **Decision owner:** ${c.owner}`);
    a(`- **Option to table:** ${c.hint}`);
    a("");
  }
  a("## 8. Gaps and follow-up questions");
  a("");
  a("| # | Type | Severity | Ref | Detail |");
  a("|---|---|---|---|---|");
  gaps.forEach((g, i) => a(`| ${i + 1} | ${g.kind} | ${g.severity} | ${g.ref} | ${mdCell(g.detail)} |`));
  a("");
  for (const [aud, items] of followups) {
    if (!items.length) continue;
    a(`**${audienceLabel.get(aud)}**`);
    a("");
    items.forEach((g, i) => a(`${i + 1}. [${g.ref}] ${g.question}`));
    a("");
  }
  a("## 9. Use-case feasibility and readiness");
  a("");
  const w = useCases.weights;
  a(
    `readiness = ${w.value} x value + ${w.feasibility} x feasibility + ${w.safety} x (6 - risk); ` +
      "feasibility = mean(data readiness, technical fit, 6 - integration complexity); " +
      "risk = mean(data sensitivity, effective error impact, regulatory exposure), where a human in the loop " +
      "lowers effective error impact by 2 (min 1).",
  );
  a("");
  a("| UC | Use case | Value | Feasibility | Risk | Readiness | Decision | Blockers |");
  a("|---|---|---:|---:|---:|---:|---|---|");
  for (const u of ucScores) {
    a(
      `| ${u.id} | ${u.name} | ${u.value} | ${u.feasibility.toFixed(2)} | ${u.risk.toFixed(2)} | ${u.readiness.toFixed(2)} | ` +
        `**${u.decision}** | ${u.blockers.length ? u.blockers.join("; ") : "-"} |`,
    );
  }
  a("");
  a("## 10. Context: users, volumes, data and integrations");
  a("");
  a("| Type | Item | Detail | Source |");
  a("|---|---|---|---|");
  for (const [iid, ex] of extractions) {
    for (const p of ex.personas) a(`| Persona | ${p.name} | ${mdCell(p.needs)} | ${iid} |`);
    for (const pp of ex.pain_points) a(`| Pain point | ${mdCell(pp)} | - | ${iid} |`);
    for (const v of ex.volumes) a(`| Volume | ${v.metric} | ${num(v.value)} ${v.unit} | ${iid} |`);
    for (const d of ex.data_sources) a(`| Data source | ${d.name} | ${d.sensitivity}${d.contains_pii ? ", PII" : ""} | ${iid} |`);
    for (const itg of ex.integrations) a(`| Integration | ${itg.system} | ${itg.direction} via ${itg.mechanism} | ${iid} |`);
  }
  a("");
  a("## 11. Risks, assumptions and open questions");
  a("");
  const lists: [string, "risks" | "assumptions" | "open_questions"][] = [
    ["Risk", "risks"],
    ["Assumption", "assumptions"],
    ["Open question", "open_questions"],
  ];
  for (const [label, attr] of lists) {
    for (const [iid, ex] of extractions) for (const item of ex[attr]) a(`- **${label}** (${iid}): ${item}`);
  }
  a("");
  a("## 12. Traceability (requirement -> evidence)");
  a("");
  a("| Requirement | Source | Evidence (verbatim quote) | Verified |");
  a("|---|---|---|---|");
  const verifiedMap = new Map(quoteChecks.map(([, ref, , ok]) => [ref, ok]));
  const mark = (src: string): string => (verifiedMap.get(src) ? "yes" : "**NO - confirm**");
  for (const m of merged) {
    m.sources.forEach((src, i) => a(`| ${m.id} | ${src} | "${mdCell(m.quotes[i]![1])}" | ${mark(src)} |`));
  }
  for (const n of nfrs) {
    const src = `${n.interview}/${n.localId}`;
    a(`| ${n.id} | ${src} | "${mdCell(n.item.source_quote)}" | ${mark(src)} |`);
  }
  a("");
  a("## Appendix - interview-guide coverage");
  a("");
  a("| Section | " + iids.join(" | ") + " | Status |");
  a("|---|" + "---:|".repeat(iids.length) + "---|");
  for (const s of guide.sections) {
    const row = matrix.get(s.id)!;
    const st = coverageStatus(row);
    const status = st === "GAP" ? "**GAP**" : st;
    a(`| ${s.title} | ` + iids.map((i) => String(row.get(i))).join(" | ") + ` | ${status} |`);
  }
  a("");
  return L.join("\n");
}

await main();
