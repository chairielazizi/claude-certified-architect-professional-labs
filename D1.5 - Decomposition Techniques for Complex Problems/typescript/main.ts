/**
 * D1.5 - Decomposition Techniques for Complex Problems
 *
 * Goal: produce a launch-readiness brief that needs (1) several dependent analyses, (2) obligations
 * from a long contract and (3) a multi-step capacity calculation.
 *
 *   0. Monolithic baseline   one giant prompt -> truncation (max_tokens) + a silently missed requirement
 *   1. Task DAG              planner (structured output) -> validate (unknown deps, cycles) -> topological
 *                            execution, each task sees ONLY its dependencies' outputs
 *   2. Map-reduce            chunk the contract by section -> map: extract obligations -> reduce in code
 *   3. Least-to-most         ordered sub-questions, each answer gated by deterministic checks
 *   4. Comparison            requirement coverage + calls/tokens/cost, monolithic vs decomposed
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
  getClient,
  section,
  textOf,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
const CONTRACT = readFileSync(path.join(DATA, "vendor_contract.md"), "utf-8");
const GOAL = "Prepare the launch-readiness brief for the new 'Export to PDF' dashboard feature.";
const REQUIRED = ["Requirements", "Security", "Pricing", "Docs and support", "Risk register", "Contract obligations", "Black Friday capacity", "Go/no-go"];
const FACTS = { last_year_volume: 18400, growth_pct: 25, deflection_pct: 38, agents: 42, tickets_per_agent_day: 55, days: 7 };
const cj = (obj: unknown): string => JSON.stringify(obj);
const fmt0 = (n: number): string => Math.round(n).toLocaleString("en-US");

// ---------------------------------------------------------------------------
// Ledger
// ---------------------------------------------------------------------------
const LEDGER = new Map<string, { calls: number; tokens: number; cost: number }>();
function record(phase: string, msg: Anthropic.Message): void {
  const r = LEDGER.get(phase) ?? { calls: 0, tokens: 0, cost: 0 };
  r.calls += 1;
  r.tokens += msg.usage.input_tokens + msg.usage.output_tokens;
  r.cost += costUsd(msg.model, msg.usage);
  LEDGER.set(phase, r);
}
async function call(client: Anthropic, phase: string, params: Anthropic.MessageCreateParamsNonStreaming): Promise<Anthropic.Message> {
  const msg = await client.messages.create(params);
  record(phase, msg);
  return msg;
}

/** A required section counts only if its heading exists AND it has a non-empty body. */
function coverage(text: string): Map<string, boolean> {
  const lines = text.split("\n");
  const found = new Map<string, boolean>();
  for (const req of REQUIRED) {
    let ok = false;
    lines.forEach((line, i) => {
      if (line.trim().toLowerCase() === `## ${req}`.toLowerCase()) {
        const body: string[] = [];
        for (const nxt of lines.slice(i + 1)) {
          if (nxt.startsWith("## ")) break;
          body.push(nxt);
        }
        ok = body.join("").trim().length > 0;
      }
    });
    found.set(req, ok);
  }
  return found;
}
const covCount = (cov: Map<string, boolean>): number => [...cov.values()].filter(Boolean).length;

// ===========================================================================
// 1. TASK DAG: plan -> validate -> topological execution
// ===========================================================================
const PlannedTask = z.object({ id: z.string(), title: z.string(), instruction: z.string(), depends_on: z.array(z.string()) });
type PlannedTask = z.infer<typeof PlannedTask>;
const TaskPlan = z.object({ tasks: z.array(PlannedTask) });

/** DFS with colours; returns one cycle as a path (e.g. t2 -> t5 -> t2) or []. */
function findCycle(tasks: PlannedTask[]): string[] {
  const ids = new Set(tasks.map((t) => t.id));
  const graph = new Map(tasks.map((t) => [t.id, t.depends_on.filter((d) => ids.has(d))]));
  const color = new Map([...graph.keys()].map((k) => [k, 0])); // 0 white, 1 grey (on stack), 2 black
  const stack: string[] = [];
  const dfs = (node: string): string[] => {
    color.set(node, 1);
    stack.push(node);
    for (const dep of graph.get(node)!) {
      if (color.get(dep) === 1) return [...stack.slice(stack.indexOf(dep)), dep];
      if (color.get(dep) === 0) {
        const cyc = dfs(dep);
        if (cyc.length) return cyc;
      }
    }
    stack.pop();
    color.set(node, 2);
    return [];
  };
  for (const node of [...graph.keys()].sort()) {
    if (color.get(node) === 0) {
      const cyc = dfs(node);
      if (cyc.length) return cyc;
    }
  }
  return [];
}

function validateDag(tasks: PlannedTask[]): string[] {
  const ids = tasks.map((t) => t.id);
  const errors = [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))].map((id) => `duplicate task id ${id}`);
  for (const t of tasks) errors.push(...t.depends_on.filter((d) => !ids.includes(d)).map((d) => `${t.id} depends on unknown task ${d}`));
  const cycle = findCycle(tasks);
  if (cycle.length) errors.push("cycle: " + cycle.join(" -> "));
  return errors;
}

/** Kahn-style layering: every task in level N depends only on levels < N (so a level can run in parallel). */
function topoLevels(tasks: PlannedTask[]): string[][] {
  const deps = new Map(tasks.map((t) => [t.id, t.depends_on]));
  const done = new Set<string>();
  const levels: string[][] = [];
  while (done.size < deps.size) {
    const ready = [...deps].filter(([id, d]) => !done.has(id) && d.every((x) => done.has(x))).map(([id]) => id).sort();
    if (!ready.length) throw new Error("unresolvable dependencies");
    levels.push(ready);
    ready.forEach((id) => done.add(id));
  }
  return levels;
}

const PLANNER_SYSTEM =
  "You plan work as a dependency graph. Return tasks with ids t1..tn, a one-line instruction each, " +
  "and depends_on listing only the task ids whose OUTPUT the task needs. The final task must " +
  "produce the go/no-go decision.";

async function runDag(client: Anthropic): Promise<Map<string, string>> {
  let user = GOAL;
  let plan: z.infer<typeof TaskPlan> | null = null;
  let valid = false;
  for (const attempt of [1, 2]) {
    const msg = await client.messages.parse({
      model: MODEL_OPUS,
      max_tokens: 2000,
      output_config: { format: zodOutputFormat(TaskPlan) },
      system: PLANNER_SYSTEM,
      messages: [{ role: "user", content: user }],
    });
    record("decomposed", msg);
    plan = msg.parsed_output;
    const errors = plan ? validateDag(plan.tasks) : ["no plan returned"];
    console.log(`  plan attempt ${attempt}: ${plan ? plan.tasks.length : 0} tasks -> ${errors.length ? "INVALID: " + errors.join("; ") : "VALID"}`);
    if (!errors.length) {
      valid = true;
      break;
    }
    user = `${GOAL}\n\nValidator feedback - fix and return the full plan: ${errors.join("; ")}`;
  }
  if (!valid || !plan) throw new Error("planner could not produce a valid DAG");
  const byId = new Map(plan.tasks.map((t) => [t.id, t]));
  const outputs = new Map<string, string>();
  for (const [n, level] of topoLevels(plan.tasks).entries()) {
    const note = level.length > 1 ? `  (${level.length} independent tasks - could run concurrently)` : "";
    console.log(`  level ${n + 1}: ${level.join(", ")}${note}`);
    for (const tid of level) {
      const t = byId.get(tid)!;
      const context = t.depends_on.map((d) => `[${d} output] ${outputs.get(d)}`).join("\n"); // ONLY its dependencies
      const msg = await call(client, "decomposed", {
        model: MODEL_SONNET,
        max_tokens: 800,
        system: "You execute one task of a larger plan. Use only the provided inputs. Be concise.",
        messages: [{ role: "user", content: `Task ${tid}: ${t.instruction}\n\nInputs:\n${context || "(none)"}` }],
      });
      outputs.set(tid, textOf(msg));
      const deps = t.depends_on.join(",") || "-";
      console.log(`    ${tid} ${t.title.padEnd(22)} deps=${deps.padEnd(6)} -> ${outputs.get(tid)!.slice(0, 70)}...`);
    }
  }
  return new Map([...outputs].map(([k, v]) => [byId.get(k)!.title, v]));
}

// ===========================================================================
// 2. MAP-REDUCE over a long document
// ===========================================================================
interface Chunk {
  label: string;
  heading: string;
  text: string;
}
const MAX_CHUNK_CHARS = 900;

/** Split on '## ' headings; split oversized sections on paragraph boundaries (never mid-sentence). */
function chunkBySection(doc: string): [string, Chunk[]] {
  const title = doc.split("\n")[0]!.replace(/^[# ]+/, "").trim();
  const chunks: Chunk[] = [];
  for (const block of doc.split(/\n(?=## )/).slice(1)) {
    const nl = block.indexOf("\n");
    const heading = block.slice(0, nl).replace(/^[# ]+/, "").trim();
    const body = block.slice(nl + 1);
    const number = heading.split(".")[0]!;
    const paras = body.trim().split("\n\n").map((p) => p.trim()).filter(Boolean);
    if (body.length <= MAX_CHUNK_CHARS) {
      chunks.push({ label: number, heading, text: body.trim() });
      continue;
    }
    paras.forEach((para, i) => chunks.push({ label: `${number}.${i + 1}`, heading: `${heading} (part ${i + 1}/${paras.length})`, text: para }));
  }
  return [title, chunks];
}

const Obligation = z.object({ party: z.string(), obligation: z.string(), deadline_hours: z.number().nullable(), penalty: z.string().nullable() });
type Obligation = z.infer<typeof Obligation>;
const ChunkObligations = z.object({ obligations: z.array(Obligation) });

const fmtDeadline = (hours: number | null): string => (hours === null ? "-" : hours < 48 ? `${hours}h` : `${hours / 24}d`);

async function runMapReduce(client: Anthropic): Promise<string> {
  const [title, chunks] = chunkBySection(CONTRACT);
  console.log(
    `  document: ${title} (${CONTRACT.length} chars) -> ${chunks.length} chunks ` +
      `(max ${MAX_CHUNK_CHARS} chars; oversized sections split on paragraphs)`,
  );
  const rows: [string, Obligation][] = [];
  for (const ch of chunks) {
    // MAP (independent -> parallelisable)
    const msg = await client.messages.parse({
      model: MODEL_HAIKU,
      max_tokens: 1000,
      output_config: { format: zodOutputFormat(ChunkObligations) },
      system:
        "Extract every contractual obligation in this section: party, obligation, deadline in hours " +
        "(null if none) and penalty (null if none). Return an empty list if there are none.",
      messages: [{ role: "user", content: `Document: ${title}\nSection ${ch.heading}\n\n${ch.text}` }],
    });
    record("decomposed", msg);
    const found = msg.parsed_output?.obligations ?? [];
    rows.push(...found.map((o): [string, Obligation] => [ch.label, o]));
    console.log(`    map ${ch.label.padEnd(4)} ${ch.heading.slice(0, 44).padEnd(44)} -> ${found.length} obligation(s)`);
  }
  // REDUCE in code: deterministic merge, sort, aggregate - no LLM needed for arithmetic
  rows.sort(([la, a], [lb, b]) => {
    const na = a.deadline_hours === null ? 1 : 0;
    const nb = b.deadline_hours === null ? 1 : 0;
    return na - nb || (a.deadline_hours ?? 0) - (b.deadline_hours ?? 0) || (la < lb ? -1 : la > lb ? 1 : 0);
  });
  const byParty = new Map<string, number>();
  for (const [, o] of rows) byParty.set(o.party, (byParty.get(o.party) ?? 0) + 1);
  const partySorted = [...byParty].sort(([a], [b]) => (a < b ? -1 : 1));
  const penalties = rows.filter(([, o]) => o.penalty).map(([lbl, o]): [string, string] => [lbl, o.penalty!]);
  console.log(`  reduce: ${rows.length} obligations; by party: ` + partySorted.map(([p, n]) => `${p}=${n}`).join(", "));
  console.log("  tightest deadlines:");
  for (const [lbl, o] of rows.filter(([, o]) => o.deadline_hours !== null).slice(0, 4))
    console.log(`    ${fmtDeadline(o.deadline_hours).padStart(4)}  [${lbl}] ${o.party}: ${o.obligation}`);
  for (const [lbl, pen] of penalties) console.log(`    penalty [${lbl}]: ${pen}`);
  const reduced = {
    total: rows.length,
    by_party: Object.fromEntries(partySorted),
    tightest: rows.slice(0, 4).map(([lbl, o]) => ({ section: lbl, party: o.party, obligation: o.obligation, deadline: fmtDeadline(o.deadline_hours) })),
    penalties: penalties.map(([, p]) => p),
  };
  const msg = await call(client, "decomposed", {
    model: MODEL_SONNET,
    max_tokens: 600,
    system: "Summarise contract obligations for a launch brief in 3 sentences. Use only the data given.",
    messages: [{ role: "user", content: cj(reduced) }],
  });
  const summary = textOf(msg);
  console.log(`  summary: ${summary}`);
  return summary;
}

// ===========================================================================
// 3. LEAST-TO-MOST with validation gates between steps
// ===========================================================================
const Decomposition = z.object({ subquestions: z.array(z.object({ question: z.string(), fields: z.array(z.string()) })) });

function stepSchema(fields: string[]): Record<string, unknown> {
  const props: Record<string, unknown> = Object.fromEntries(fields.map((f) => [f, { type: "number" }]));
  props.explanation = { type: "string" };
  return { type: "object", properties: props, required: [...fields, "explanation"], additionalProperties: false };
}

/** Deterministic checks against the input facts and earlier (already-gated) answers. */
function gate(values: Record<string, unknown>, known: Record<string, number>): string[] {
  const problems = Object.entries(values)
    .filter(([, v]) => typeof v !== "number" || v < 0)
    .map(([k]) => `${k} must be a non-negative number`);
  if (problems.length) return problems;
  const v = values as Record<string, number>;
  if ("volume" in v && !(FACTS.last_year_volume <= v.volume! && v.volume! <= 2 * FACTS.last_year_volume))
    problems.push("volume outside plausible range (1x-2x last year)");
  if ("deflected" in v && "remaining" in v) {
    const vol = known.volume!;
    if (Math.abs(v.deflected! + v.remaining! - vol) > 1) problems.push(`deflected + remaining != volume ${fmt0(vol)}`);
    const share = (v.deflected! / vol) * 100;
    if (Math.abs(share - FACTS.deflection_pct) > 2) problems.push(`deflected share ${share.toFixed(0)}% contradicts the ${FACTS.deflection_pct}% pilot rate`);
  }
  if ("capacity" in v) {
    const expected = FACTS.agents * FACTS.tickets_per_agent_day * FACTS.days;
    if (Math.abs(v.capacity! - expected) > expected * 0.01) problems.push(`capacity ${fmt0(v.capacity!)} inconsistent with staffing facts`);
  }
  if ("surplus" in v && Math.abs(v.surplus! - (known.capacity! - known.remaining!)) > 1) problems.push("surplus != capacity - remaining");
  if ("utilization_pct" in v && Math.abs(v.utilization_pct! - (known.remaining! / known.capacity!) * 100) > 0.5)
    problems.push("utilisation inconsistent with remaining / capacity");
  return problems;
}

async function runLeastToMost(client: Anthropic): Promise<string> {
  const question = "Can our support team handle Black Friday week once the AI assistant is live?";
  const factsText = Object.entries(FACTS).map(([k, v]) => `${k}=${v}`).join(", ");
  const msg = await client.messages.parse({
    model: MODEL_SONNET,
    max_tokens: 1000,
    output_config: { format: zodOutputFormat(Decomposition) },
    system:
      "Decompose the question into the simplest ordered sub-questions (least to most complex). " +
      "For each, name the numeric output fields it must produce.",
    messages: [{ role: "user", content: `Question: ${question}\nFacts: ${factsText}` }],
  });
  record("decomposed", msg);
  const decomposition = msg.parsed_output;
  if (!decomposition) return "decomposition failed";
  const known: Record<string, number> = {};
  for (const [idx, sq] of decomposition.subquestions.entries()) {
    const k = idx + 1;
    let feedback = "";
    let passed = false;
    for (const attempt of [1, 2, 3]) {
      const prior = Object.entries(known).map(([f, v]) => `${f}=${v}`).join(", ") || "(none)";
      const user = `Facts: ${factsText}\nEarlier answers: ${prior}\nSubquestion ${k}: ${sq.question}` + (feedback ? `\nGate feedback: ${feedback}` : "");
      const m = await call(client, "decomposed", {
        model: MODEL_SONNET,
        max_tokens: 500,
        system: "Answer one sub-question using the facts and earlier answers. Return JSON only.",
        output_config: { format: { type: "json_schema", schema: stepSchema(sq.fields) } },
        messages: [{ role: "user", content: user }],
      });
      const data: Record<string, unknown> = m.stop_reason === "end_turn" ? JSON.parse(textOf(m)) : {};
      const values = Object.fromEntries(sq.fields.map((f) => [f, data[f]]));
      const problems = Object.keys(data).length ? gate(values, known) : ["no complete answer"];
      const shown = Object.entries(values).map(([f, v]) => `${f}=${v}`).join(", ");
      console.log(
        `  Q${k}.${attempt} ${sq.question.slice(0, 58).padEnd(58)} -> ${shown}  ` +
          `gate: ${problems.length ? "FAIL (" + problems.join("; ") + ")" : "PASS"}`,
      );
      if (!problems.length) {
        for (const [f, val] of Object.entries(values)) known[f] = Number(val); // only gated values flow forward
        passed = true;
        break;
      }
      feedback = problems.join("; ");
    }
    if (!passed) return `stopped: sub-question ${k} failed its gate 3 times -> escalate to an analyst`;
  }
  const m = await call(client, "decomposed", {
    model: MODEL_SONNET,
    max_tokens: 600,
    system: "Answer the original question from the verified sub-answers in 2-3 sentences.",
    messages: [{ role: "user", content: `Question: ${question}\nVerified: ${cj(known)}` }],
  });
  const answer = textOf(m);
  console.log(`  answer: ${answer}`);
  return answer;
}

// ===========================================================================
// Offline responder
// ===========================================================================
const TASKS_OK = [
  { id: "t1", title: "Requirements summary", instruction: "Summarise functional and non-functional requirements.", depends_on: [] as string[] },
  { id: "t2", title: "Security review", instruction: "Review security and privacy of PDF export.", depends_on: ["t1"] },
  { id: "t3", title: "Pricing and packaging", instruction: "Decide plan availability and limits.", depends_on: ["t1"] },
  { id: "t4", title: "Docs and support plan", instruction: "Plan help content and support readiness.", depends_on: ["t1"] },
  { id: "t5", title: "Risk register", instruction: "List top risks with mitigations.", depends_on: ["t2", "t3"] },
  { id: "t6", title: "Go/no-go decision", instruction: "Recommend go/no-go with rollout gates.", depends_on: ["t4", "t5"] },
];
const TASKS_BAD = TASKS_OK.map((t) => ({ ...t }));
TASKS_BAD[1] = { ...TASKS_OK[1]!, depends_on: ["t1", "t5"] }; // t2 <-> t5 cycle
TASKS_BAD[5] = { ...TASKS_OK[5]!, depends_on: ["t4", "t5", "t7"] }; // unknown task
const TASK_OUTPUTS: Record<string, string> = {
  t1: "One-click PDF export of any dashboard (p95 under 10 s for 20 widgets) plus an optional weekly email; Team and Enterprise plans.",
  t2: "Render server-side with the viewer's permissions; signed download URLs expire after 24 h; no PII in file names.",
  t3: "Included in Team and Enterprise; Team limited to 10 scheduled emails; no add-on price.",
  t4: "Help-centre article, in-app tooltip and a support macro for 'export failed' tickets, ready at launch.",
  t5: "Renderer load spikes (queue + 10/min rate limit); shared-link leakage (24 h expiry); schedule abuse on Team (cap of 10).",
  t6: "GO for a staged rollout: 10% of Team accounts for one week, gated on export error rate < 1% and p95 < 10 s.",
};
const ob = (party: string, obligation: string, deadline_hours: number | null, penalty: string | null = null): Obligation => ({ party, obligation, deadline_hours, penalty });
const MAP_OUTPUTS: Record<string, Obligation[]> = {
  "1": [],
  "2": [ob("Vendor", "Provide the Services in the Order Form", null), ob("Customer", "Provide ticketing API access and nominate an administrator", null)],
  "3": [
    ob("Vendor", "Keep 99.9% monthly availability", null, "service credit 10% of Monthly Fee per 0.1% below target (max 50%)"),
    ob("Vendor", "Respond to Priority 1 Incidents", 1),
  ],
  "4.1": [
    ob("Vendor", "Process Personal Data only in EU data centres", null),
    ob("Vendor", "Assist with data subject requests and DPIAs at no charge", null),
    ob("Vendor", "Give notice before engaging new sub-processors", 720),
  ],
  "4.2": [ob("Vendor", "Notify Customer of a Personal Data breach", 72)],
  "5": [ob("Customer", "Pay each invoice", 720, "late interest 1% per month"), ob("Vendor", "Give notice of a fee increase (max 5%/year)", 1440)],
  "6": [
    ob("Vendor", "Share penetration-test summary after completion", 720),
    ob("Vendor", "Remediate critical vulnerabilities", 336),
    ob("Vendor", "Remediate high vulnerabilities", 720),
  ],
  "7": [ob("Either", "Give notice to terminate for convenience", 2160), ob("Vendor", "Return or delete Customer data after termination", 720)],
  "8": [],
};
const OBLIGATION_SUMMARY =
  "Vendor carries 12 of the 15 obligations, with the tightest deadlines being a 1-hour response to " +
  "Priority 1 incidents and 72-hour breach notification. Financial levers are availability service " +
  "credits (10% per 0.1% below 99.9%, max 50%) and 1% monthly late-payment interest on Customer. " +
  "Exit requires 90 days' notice, with data returned or deleted within 30 days.";
const DECOMPOSITION = {
  subquestions: [
    { question: "What ticket volume should we expect in Black Friday week?", fields: ["volume"] },
    { question: "How many tickets will the AI deflect and how many remain for agents?", fields: ["deflected", "remaining"] },
    { question: "How many tickets can the agent team handle that week?", fields: ["capacity"] },
    { question: "What is the surplus or shortfall and resulting utilisation?", fields: ["surplus", "utilization_pct"] },
  ],
};
const STEP_ANSWERS: Record<number, Record<string, unknown>> = {
  1: { volume: 23000, explanation: "18,400 x 1.25" },
  2: { deflected: 8740, remaining: 14260, explanation: "23,000 x 38% deflected; the rest remain" },
  3: { capacity: 16170, explanation: "42 agents x 55 tickets/day x 7 days" },
  4: { surplus: 1910, utilization_pct: 88.2, explanation: "16,170 - 14,260; 14,260 / 16,170" },
};
const CAPACITY_ANSWER =
  "Yes. Expect ~23,000 tickets; the AI assistant deflects ~8,740, leaving 14,260 for agents against a " +
  "capacity of 16,170, a 1,910-ticket buffer at 88.2% utilisation. If deflection drops below ~30% the " +
  "buffer is gone, so keep an on-call overflow rota.";

function monolithicReply(): string {
  return (
    "## Requirements\nThe feature lets any user export a dashboard to PDF with one click and optionally receive a weekly " +
    "email with the PDF attached. Exports must complete quickly for typical dashboards, preserve the on-screen layout " +
    "including charts, tables and filters, and respect the viewer's permissions. Large dashboards with more than twenty " +
    "widgets should still render, potentially with pagination, and the export must be available on desktop browsers. " +
    "Scheduled emails must allow choosing a weekday and time, and recipients must be existing workspace members. " +
    "The export must include a cover page with the dashboard title, the filters applied and a timestamp in the " +
    "viewer's time zone, and charts must be rendered as vectors so that they stay sharp when printed. Exported " +
    "tables need real text rather than images so that screen readers and search work on the PDF.\n" +
    "## Security\nRendering should happen server-side using the requesting viewer's permissions so no hidden widgets " +
    "leak into the document. Download links must be signed and time-limited, and file names must avoid personal data. " +
    "Scheduled exports run as the schedule owner, so revoking a user's access must also disable their schedules. All " +
    "export events should be logged for audit, and generated files should be deleted from storage after a retention " +
    "window. Rate limits protect the renderer from abuse, and the endpoint must reject cross-tenant identifiers. " +
    "Penetration testing of the new endpoint is required before general availability, and the renderer must run " +
    "in an isolated sandbox without access to internal networks.\n" +
    "## Docs and support\nPublish a help-centre article with screenshots, add an in-app tooltip next to the new export " +
    "button, and prepare a support macro for failed exports. Train tier-one agents on the most common failure modes, " +
    "such as timeouts on very large dashboards and permission errors on shared dashboards, and update the status page " +
    "templates so incidents affecting exports can be communicated quickly to affected customers.\n" +
    "## Risk register\nThe main operational risk is load on the headless renderer during peak hours, which could slow " +
    "exports for everyone; mitigation is a queue with per-tenant rate limits and autoscaling of renderer workers. A " +
    "second risk is leakage through forwarded download links; mitigation is short expiry and signed URLs bound to the " +
    "tenant. A third risk is abuse of scheduled emails to send large volumes of mail; mitigation is a cap on schedules " +
    "per workspace and monitoring of bounce rates. A fourth risk is inconsistent rendering of custom chart plugins, " +
    "which we mitigate with a compatibility test suite that runs against the top fifty public dashboards before each " +
    "release, and a feature flag so that problematic plugins can fall back to static images without blocking the export.\n" +
    "## Contract obligations\nThe vendor must notify breaches within 72 hours and respond to P1 incidents in 1 hour...\n" +
    "## Black Friday capacity\nWith 23,000 expected tickets and 38% deflection, 14,260 remain against 16,170 capacity...\n" +
    "## Go/no-go\nGO for a staged rollout gated on error rate and latency.\n"
  );
}

function responder(req: MockRequest): ResponderResult {
  const s = req.systemText;
  const u = req.lastUserText;
  if (s.includes("dependency graph")) return JSON.stringify({ tasks: u.includes("Validator feedback") ? TASKS_OK : TASKS_BAD });
  if (s.includes("You execute one task")) return TASK_OUTPUTS[/Task (t\d+)/.exec(u)![1]!]!;
  if (s.includes("Extract every contractual obligation")) {
    const num = /Section (\d+)\./.exec(u)![1]!;
    const part = /\(part (\d+)\//.exec(u);
    return JSON.stringify({ obligations: MAP_OUTPUTS[part ? `${num}.${part[1]}` : num] });
  }
  if (s.includes("Summarise contract obligations")) return OBLIGATION_SUMMARY;
  if (s.includes("Decompose the question")) return JSON.stringify(DECOMPOSITION);
  if (s.includes("Answer one sub-question")) {
    const k = Number(/Subquestion (\d+):/.exec(u)![1]);
    // Classic slip: deflected and remaining swapped. The gate catches it before it propagates.
    if (k === 2 && !u.includes("Gate feedback")) return JSON.stringify({ deflected: 14260, remaining: 8740, explanation: "38% of 23,000 remain" });
    return JSON.stringify(STEP_ANSWERS[k]);
  }
  if (s.includes("original question from the verified")) return CAPACITY_ANSWER;
  if (s.includes("complete launch-readiness brief")) return monolithicReply();
  return "OK";
}

// ===========================================================================
// Main
// ===========================================================================
banner("D1.5 - Decomposition Techniques for Complex Problems");
const client = getClient(responder);

section("0. Monolithic baseline: one prompt does everything");
const monoPrompt = `${GOAL}\nCover these sections as '## ' headings: ${REQUIRED.join(", ")}.\nCapacity facts: ${cj(FACTS)}\n\nContract:\n${CONTRACT}`;
const mono = await call(client, "monolithic", {
  model: MODEL_SONNET,
  max_tokens: 600,
  system: "You write the complete launch-readiness brief in one pass.",
  messages: [{ role: "user", content: monoPrompt }],
});
const monoText = textOf(mono);
const monoCov = coverage(monoText);
console.log(`  stop_reason=${mono.stop_reason}  output=${monoText.length} chars  (input ${mono.usage.input_tokens} tokens incl. the whole contract)`);
console.log(`  coverage ${covCount(monoCov)}/${REQUIRED.length}: ` + [...monoCov].map(([k, v]) => `${k}=${v ? "ok" : "MISSING"}`).join(", "));
console.log("  -> truncated mid-document AND 'Pricing' was skipped entirely; nothing tells you which is which");

section("1. Task DAG: planner -> validator -> topological execution");
const dagOutputs = await runDag(client);

section("2. Map-reduce over the contract");
const obligations = await runMapReduce(client);

section("3. Least-to-most with validation gates");
const capacity = await runLeastToMost(client);

section("4. Assemble + compare");
const brief = [
  "## Requirements", dagOutputs.get("Requirements summary"),
  "## Security", dagOutputs.get("Security review"),
  "## Pricing", dagOutputs.get("Pricing and packaging"),
  "## Docs and support", dagOutputs.get("Docs and support plan"),
  "## Risk register", dagOutputs.get("Risk register"),
  "## Contract obligations", obligations,
  "## Black Friday capacity", capacity,
  "## Go/no-go", dagOutputs.get("Go/no-go decision"),
].join("\n");
const decCov = coverage(brief);
console.log(`  ${"approach".padEnd(12)}${"calls".padStart(6)}${"tokens".padStart(8)}${"cost $".padStart(10)}${"coverage".padStart(10)}  failure visibility`);
for (const [name, cov, note] of [
  ["monolithic", monoCov, "silent: truncation + omission"],
  ["decomposed", decCov, "per-step: validator, gates, stop_reason"],
] as const) {
  const r = LEDGER.get(name)!;
  console.log(
    `  ${name.padEnd(12)}${String(r.calls).padStart(6)}${String(r.tokens).padStart(8)}${r.cost.toFixed(5).padStart(10)}` +
      `${`${covCount(cov)}/${REQUIRED.length}`.padStart(10)}  ${note}`,
  );
}
console.log("  Decomposition costs more calls, but every step is small, checkable, retryable - and parallelisable.");

section("Summary");
console.log("  DAG for dependent sub-tasks, map-reduce for long inputs, least-to-most for multi-step reasoning;");
console.log("  validate between steps in CODE so errors are caught where they happen, not in the final answer.");
