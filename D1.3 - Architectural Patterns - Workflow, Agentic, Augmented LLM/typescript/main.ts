/**
 * D1.3 - Architectural Patterns - Workflow, Agentic, Augmented LLM
 *
 * Small, runnable versions of the building blocks from Anthropic's "Building effective agents":
 *
 *   1. Augmented LLM        LLM + retrieval + tools + memory (the unit everything else is built from)
 *   2. Prompt chaining      fixed steps with a programmatic gate between them
 *   3. Routing              classify, then hand off to a specialised prompt/model
 *   4. Parallelization      sectioning (independent aspects) and voting (independent judgments)
 *   5. Orchestrator-workers the LLM decides the subtasks at runtime
 *   6. Evaluator-optimizer  generate -> evaluate -> refine until the criteria pass
 *   7. Autonomous agent     tool use until end_turn, bounded by a step limit
 *   8. Pattern selector     deterministic recommendation with trade-offs
 *
 * Workflows = code owns the control flow (predictable). Agents = the model owns it (flexible).
 */
import type Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
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
  textBlock,
  textOf,
  toolUseBlock,
  type Json,
  type MockRequest,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const TODAY = "2026-10-11"; // fixed "today" keeps the demo deterministic
const cj = (obj: unknown): string => JSON.stringify(obj); // compact JSON for printing

// ---------------------------------------------------------------------------
// Ledger: every call is attributed to a pattern so cost can be compared
// ---------------------------------------------------------------------------
const LEDGER = new Map<string, { calls: number; in: number; out: number; cost: number }>();
function record(pattern: string, msg: Anthropic.Message): void {
  const row = LEDGER.get(pattern) ?? { calls: 0, in: 0, out: 0, cost: 0 };
  row.calls += 1;
  row.in += msg.usage.input_tokens;
  row.out += msg.usage.output_tokens;
  row.cost += costUsd(msg.model, msg.usage);
  LEDGER.set(pattern, row);
}
async function call(client: Anthropic, pattern: string, params: Anthropic.MessageCreateParamsNonStreaming): Promise<Anthropic.Message> {
  const msg = await client.messages.create(params);
  record(pattern, msg);
  return msg;
}

// ===========================================================================
// 1. AUGMENTED LLM: retrieval + tools + memory around one model
// ===========================================================================
const KB: [string, string][] = [
  ["returns-policy", "Returns: you can return most items within 30 days of delivery for a full refund. Refunds go to the original payment method within 5 business days after we receive the return."],
  ["warranty", "Warranty: Aurora appliances carry a 2-year limited warranty covering manufacturing defects. Damage in transit must be reported within 7 days of delivery for a free replacement."],
  ["shipping", "Shipping: standard delivery takes 3-5 business days; express takes 1-2. Tracking links are emailed at dispatch."],
  ["loyalty", "Aurora Rewards: Gold members get free return shipping and a dedicated support line. Points are earned on every purchase."],
  ["blender-manual", "Aurora blender quick start: lock the jar onto the base, choose a program and press start. Never run it empty."],
];
const STOP = new Set(["a", "an", "and", "the", "to", "of", "in", "is", "it", "my", "i", "can", "how", "would", "still", "for",
  "on", "be", "are", "with", "most", "all", "if", "we", "you", "your", "do", "from"]);

function tokenize(text: string): string[] {
  const out: string[] = [];
  for (let tok of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (tok.length > 3 && tok.endsWith("s")) tok = tok.slice(0, -1); // naive stemming: returns -> return
    if (!STOP.has(tok)) out.push(tok);
  }
  return out;
}

/** Tiny TF-IDF retriever - the 'retrieval' augmentation (D3.5/D3.6 go deeper). */
function retrieve(query: string, k = 2): [number, string, string][] {
  const df = new Map<string, number>();
  for (const [, text] of KB) for (const t of new Set(tokenize(text))) df.set(t, (df.get(t) ?? 0) + 1);
  const qTerms = new Set(tokenize(query));
  const scored: [number, string, string][] = KB.map(([docId, text]) => {
    const tf = new Map<string, number>();
    for (const t of tokenize(text)) tf.set(t, (tf.get(t) ?? 0) + 1);
    let score = 0;
    for (const t of qTerms) if (tf.has(t)) score += tf.get(t)! * Math.log(1 + KB.length / df.get(t)!);
    return [Math.round(score * 1e4) / 1e4, docId, text];
  });
  scored.sort((a, b) => b[0] - a[0] || a[1].localeCompare(b[1]));
  return scored.slice(0, k).filter((s) => s[0] > 0);
}

const MEMORY: Record<string, string[]> = { "C-1042": ["Gold Aurora Rewards member", "prefers email updates"] };
const ORDERS_DB: Record<string, Json> = {
  "A-7731": { order_id: "A-7731", item: "Aurora blender", status: "delivered", delivered_on: "2026-10-02" },
};

const AUGMENTED_TOOLS: Anthropic.Tool[] = [
  {
    name: "get_order_status",
    description: "Look up an order by id. Call this whenever the customer mentions an order.",
    strict: true,
    input_schema: { type: "object", properties: { order_id: { type: "string" } }, required: ["order_id"], additionalProperties: false },
  },
  {
    name: "save_memory",
    description: "Persist a durable fact about the customer for future conversations.",
    strict: true,
    input_schema: { type: "object", properties: { fact: { type: "string" } }, required: ["fact"], additionalProperties: false },
  },
];

async function runAugmented(client: Anthropic): Promise<void> {
  const customer = "C-1042";
  const question = "Can I still return my blender from order A-7731, and do I pay for return shipping?";
  const docs = retrieve(question);
  console.log(`  question : ${question}`);
  console.log(`  memory   : ${MEMORY[customer]!.join("; ")}`);
  for (const [score, docId] of docs) console.log(`  retrieved: ${docId.padEnd(15)} score=${score.toFixed(2)}`);
  const system =
    `You are the support assistant for Aurora Home. Today is ${TODAY}.\n` +
    `Known facts about this customer: ${MEMORY[customer]!.join("; ")}.\n` +
    "Answer only from the documents and tool results; cite document ids in [brackets].";
  const context = docs.map(([, d, t]) => `<document id="${d}">${t}</document>`).join("\n");
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: `<documents>\n${context}\n</documents>\n\n${question}` }];
  let msg!: Anthropic.Message;
  for (let i = 0; i < 3; i++) {
    // tiny loop: the augmented LLM may call tools before answering
    msg = await call(client, "1 augmented LLM", { model: MODEL_SONNET, max_tokens: 1024, system, tools: AUGMENTED_TOOLS, messages });
    if (msg.stop_reason !== "tool_use") break;
    messages.push({ role: "assistant", content: msg.content });
    const results: Anthropic.ToolResultBlockParam[] = [];
    for (const block of msg.content) {
      if (block.type !== "tool_use") continue;
      const input = block.input as Record<string, string>;
      let out: Json;
      if (block.name === "get_order_status") out = ORDERS_DB[input.order_id!] ?? { error: "not found" };
      else {
        MEMORY[customer]!.push(input.fact!);
        out = { saved: true };
      }
      console.log(`  tool     : ${block.name}(${cj(input)}) -> ${cj(out)}`);
      results.push({ type: "tool_result", tool_use_id: block.id, content: cj(out) });
    }
    messages.push({ role: "user", content: results }); // all parallel results in ONE user turn
  }
  console.log(`  answer   : ${textOf(msg)}`);
  console.log(`  memory now: ${MEMORY[customer]!.join("; ")}`);
}

// ===========================================================================
// 2. PROMPT CHAINING with a programmatic gate
// ===========================================================================
const CHANGELOG = `JIRA-4821 fix NPE in PdfExporter when dashboard has no widgets
JIRA-4822 bump pdfkit to 3.2
JIRA-4830 add 'Export to PDF' button on dashboards
JIRA-4835 cache widget queries: dashboard p95 load 2.1s -> 0.9s
JIRA-4840 refactor auth middleware (no behaviour change)`;
const INTERNAL_TERMS = ["pdfkit", "refactor", "middleware", "npe"];

function gateBullets(text: string): string[] {
  const bullets = text.split("\n").filter((b) => b.trim().startsWith("-"));
  const problems: string[] = [];
  if (bullets.length < 2 || bullets.length > 6) problems.push(`expected 2-6 bullets, got ${bullets.length}`);
  const ids = text.match(/\b[A-Z]{2,}-\d+\b/g) ?? [];
  if (ids.length) problems.push(`internal ticket ids present: ${ids.join(", ")}`);
  const terms = INTERNAL_TERMS.filter((t) => text.toLowerCase().includes(t));
  if (terms.length) problems.push(`internal-only changes present: ${terms.join(", ")}`);
  return problems;
}

async function runChaining(client: Anthropic): Promise<void> {
  const step1System = "Extract the customer-visible changes from an engineering changelog as '-' bullets.";
  let user = CHANGELOG;
  let bullets = "";
  for (const attempt of [1, 2]) {
    const msg = await call(client, "2 prompt chaining", { model: MODEL_HAIKU, max_tokens: 400, system: step1System, messages: [{ role: "user", content: user }] });
    bullets = textOf(msg);
    const problems = gateBullets(bullets);
    console.log(`  step 1 attempt ${attempt} (Haiku) -> gate: ${problems.length ? "FAIL - " + problems.join("; ") : "PASS"}`);
    if (!problems.length) break;
    user = `${CHANGELOG}\n\nGate feedback - fix these problems: ${problems.join("; ")}`;
  }
  for (const line of bullets.split("\n")) console.log(`      ${line}`);
  const msg = await call(client, "2 prompt chaining", {
    model: MODEL_SONNET,
    max_tokens: 600,
    system: "Write a customer-facing release note (<= 60 words, friendly, no jargon) from these bullets.",
    messages: [{ role: "user", content: bullets }],
  });
  console.log(`  step 2 (Sonnet) release note: ${textOf(msg)}`);
}

// ===========================================================================
// 3. ROUTING: cheap classifier -> specialised handler (prompt + model)
// ===========================================================================
const RouteDecision = z.object({ route: z.enum(["refund", "technical", "general"]), reason: z.string() });
const HANDLERS: Record<z.infer<typeof RouteDecision>["route"], [string, string]> = {
  refund: [MODEL_SONNET, "You are a refund specialist. Apply the refund policy precisely and state next steps."],
  technical: [MODEL_OPUS, "You are a senior technical support engineer. Diagnose step by step."],
  general: [MODEL_HAIKU, "You are a friendly general assistant. Answer briefly."],
};

async function runRouting(client: Anthropic): Promise<void> {
  const queries = [
    "I was charged twice for my subscription this month - can I get a refund?",
    "The app crashes with error E1043 whenever I open a shared dashboard.",
    "Do you have a dark mode?",
  ];
  for (const q of queries) {
    const decision = await client.messages.parse({
      model: MODEL_HAIKU,
      max_tokens: 200,
      output_config: { format: zodOutputFormat(RouteDecision) },
      system: "You route customer queries to refund, technical or general handling.",
      messages: [{ role: "user", content: q }],
    });
    record("3 routing", decision);
    const route = decision.parsed_output?.route ?? "general"; // safe default
    const [model, system] = HANDLERS[route];
    const answer = await call(client, "3 routing", { model, max_tokens: 600, system, messages: [{ role: "user", content: q }] });
    console.log(q.length > 55 ? `  "${q.slice(0, 52)}..."` : `  "${q}"`);
    console.log(`      route=${route.padEnd(9)} handler=${model.padEnd(18)} -> ${textOf(answer).slice(0, 88)}`);
  }
}

// ===========================================================================
// 4. PARALLELIZATION: sectioning + voting (concurrent calls)
// ===========================================================================
const SNIPPET = `def get_user(conn, user_id):
    rows = conn.execute(f"SELECT * FROM users WHERE id = {user_id}").fetchall()
    for r in rows:
        time.sleep(0.1)
    return rows[0]`;
const ASPECTS = ["security", "performance", "readability"];
const COMMENT = "You people are idiots and I know where your office is.";
const PERSPECTIVES = ["strict", "lenient", "policy-literal"];

async function runParallel(client: Anthropic): Promise<void> {
  const timed = async (p: Promise<Anthropic.Message>): Promise<[Anthropic.Message, number]> => {
    const t = performance.now();
    const m = await p;
    return [m, elapsedMs(t)];
  };
  const t0 = performance.now();
  const reviews = await Promise.all(
    ASPECTS.map((aspect) =>
      timed(
        call(client, "4 parallelization", {
          model: MODEL_SONNET,
          max_tokens: 300,
          system: `You are a ${aspect} reviewer. Report the single most important ${aspect} issue.`,
          messages: [{ role: "user", content: SNIPPET }],
        }),
      ),
    ),
  );
  const wall = elapsedMs(t0);
  console.log("  sectioning (3 independent aspects, concurrently):");
  ASPECTS.forEach((aspect, i) => console.log(`    ${aspect.padEnd(12)} ${textOf(reviews[i]![0])}`));
  const seq = reviews.reduce((s, [, ms]) => s + ms, 0);
  console.log(`    wall-clock ~${Math.round(wall / 10) * 10} ms vs ~${Math.round(seq / 10) * 10} ms if run sequentially`);

  // Sampling params (temperature) are not available on current models, so diversity comes
  // from framing: each voter gets a different reviewing perspective.
  const votes = await Promise.all(
    PERSPECTIVES.map((p) =>
      call(client, "4 parallelization", {
        model: MODEL_HAIKU,
        max_tokens: 50,
        system: `You are a ${p} content-policy reviewer. Reply VIOLATION or OK, then a 5-word reason.`,
        messages: [{ role: "user", content: COMMENT }],
      }),
    ),
  );
  const labels = votes.map((v) => textOf(v).split(/\s+/)[0]!.replace(/^[.:]+|[.:]+$/g, ""));
  const tally = new Map<string, number>();
  for (const l of labels) tally.set(l, (tally.get(l) ?? 0) + 1);
  let [winner, count] = ["", 0];
  for (const [label, c] of tally) if (c > count) [winner, count] = [label, c];
  console.log("  voting (3 perspectives on one comment):");
  PERSPECTIVES.forEach((p, i) => console.log(`    ${p.padEnd(15)} ${textOf(votes[i]!)}`));
  const agreement = count / labels.length;
  const action = agreement < 1 ? "remove + escalate to human (split vote)" : "auto-remove";
  console.log(`    majority=${winner} (${count}/${labels.length}, agreement ${Math.round(agreement * 100)}%) -> ${action}`);
}

// ===========================================================================
// 5. ORCHESTRATOR-WORKERS: subtasks decided at runtime by the LLM
// ===========================================================================
const Plan = z.object({ subtasks: z.array(z.object({ id: z.string(), area: z.string(), instruction: z.string() })) });

async function runOrchestrator(client: Anthropic): Promise<void> {
  const feature = "Add 'Export to PDF' for dashboards, plus an optional weekly email that attaches the PDF.";
  const planMsg = await client.messages.parse({
    model: MODEL_OPUS,
    max_tokens: 1000,
    output_config: { format: zodOutputFormat(Plan) },
    system: "You are an orchestrator. Break the feature into the minimum set of independent subtasks.",
    messages: [{ role: "user", content: feature }],
  });
  record("5 orchestrator-workers", planMsg);
  const plan = planMsg.parsed_output;
  if (!plan) {
    console.log("  orchestrator returned no plan - stop");
    return;
  }
  console.log(`  orchestrator (Opus) decided ${plan.subtasks.length} subtasks for: ${feature}`);
  const outputs: string[] = [];
  for (const st of plan.subtasks) {
    // workers get ONLY their subtask: isolated, focused context
    const m = await call(client, "5 orchestrator-workers", {
      model: MODEL_SONNET,
      max_tokens: 400,
      system: `You are a worker engineer for the ${st.area} area. Produce a 1-2 line change plan.`,
      messages: [{ role: "user", content: st.instruction }],
    });
    outputs.push(`[${st.id} ${st.area}] ${textOf(m)}`);
    console.log(`    worker ${st.id} (${st.area.padEnd(9)}) -> ${textOf(m)}`);
  }
  const synth = await call(client, "5 orchestrator-workers", {
    model: MODEL_OPUS,
    max_tokens: 600,
    system: "You are the orchestrator. Integrate the worker outputs into one ordered delivery plan.",
    messages: [{ role: "user", content: outputs.join("\n") }],
  });
  console.log(`  synthesis: ${textOf(synth)}`);
}

// ===========================================================================
// 6. EVALUATOR-OPTIMIZER: cheap code checks first, LLM judge for the subjective part
// ===========================================================================
const FACTS = "Aurora blender: 1200 W motor, 1.5 L glass jar, 6 programs, 2-year warranty.";
const BANNED = ["best", "ultimate", "perfect"];
const Verdict = z.object({ passed: z.boolean(), feedback: z.string() });

function codeChecks(text: string): string[] {
  const problems: string[] = [];
  const words = text.split(/\s+/).filter(Boolean).length;
  if (words > 60) problems.push(`${words} words (max 60)`);
  const hits = BANNED.filter((b) => new RegExp(`\\b${b}\\b`).test(text.toLowerCase()));
  if (hits.length) problems.push(`banned superlatives: ${hits.join(", ")}`);
  if (!text.toLowerCase().includes("warranty")) problems.push("must mention the 2-year warranty");
  return problems;
}

async function runEvaluatorOptimizer(client: Anthropic): Promise<void> {
  let feedback = "";
  for (let iteration = 1; iteration <= 3; iteration++) {
    const prompt = `Product facts: ${FACTS}\nWrite the description.` + (feedback ? `\nReviewer feedback: ${feedback}` : "");
    const draft = textOf(
      await call(client, "6 evaluator-optimizer", {
        model: MODEL_SONNET,
        max_tokens: 400,
        system: "You write product descriptions: max 60 words, mention the warranty, no superlatives.",
        messages: [{ role: "user", content: prompt }],
      }),
    );
    const problems = codeChecks(draft);
    if (problems.length) {
      feedback = problems.join("; ");
      console.log(`  iter ${iteration}: code checks FAIL (${feedback})`);
      continue;
    }
    const verdictMsg = await client.messages.parse({
      model: MODEL_SONNET,
      max_tokens: 300,
      output_config: { format: zodOutputFormat(Verdict) },
      system: "You evaluate product descriptions against the product facts. Fail any claim not in the facts.",
      messages: [{ role: "user", content: `Facts: ${FACTS}\nCandidate: ${draft}` }],
    });
    record("6 evaluator-optimizer", verdictMsg);
    const verdict = verdictMsg.parsed_output;
    if (verdict?.passed) {
      console.log(`  iter ${iteration}: code checks PASS, LLM judge PASS`);
      console.log(`  final: ${draft}`);
      return;
    }
    feedback = verdict ? verdict.feedback : "judge unavailable";
    console.log(`  iter ${iteration}: code checks PASS, LLM judge FAIL (${feedback})`);
  }
  console.log("  stopped after 3 iterations -> send best draft to a human editor");
}

// ===========================================================================
// 7. AUTONOMOUS AGENT: tool use until end_turn, bounded by a step limit
// ===========================================================================
const CUSTOMERS: Record<string, Json> = { "jane@example.com": { customer_id: "C-1042", name: "Jane Park", tier: "Gold" } };
const ORDERS: Record<string, { order_id: string; item: string; delivered_on: string }[]> = {
  "C-1042": [
    { order_id: "A-7731", item: "Aurora blender", delivered_on: "2026-10-02" },
    { order_id: "A-7790", item: "Aurora blender", delivered_on: "2026-10-08" },
  ],
};
const REPLACEMENTS: Json[] = [];

function schema(props: Record<string, string>): Anthropic.Tool.InputSchema {
  return {
    type: "object",
    properties: Object.fromEntries(Object.entries(props).map(([k, v]) => [k, { type: v }])),
    required: Object.keys(props),
    additionalProperties: false,
  };
}

const AGENT_TOOLS: Anthropic.Tool[] = [
  { name: "lookup_customer", description: "Find a customer by email. Call first when only an email is known.", strict: true, input_schema: schema({ email: "string" }) },
  { name: "list_orders", description: "List a customer's orders with delivery dates.", strict: true, input_schema: schema({ customer_id: "string" }) },
  { name: "check_warranty", description: "Check transit-damage and warranty eligibility for an order.", strict: true, input_schema: schema({ order_id: "string" }) },
  { name: "create_replacement", description: "Ship a free replacement. Only after check_warranty says eligible.", strict: true, input_schema: schema({ order_id: "string", reason: "string" }) },
];

/** Real tool implementations. Returns [result, isError]. Policy is enforced HERE, not just in the prompt. */
function executeTool(name: string, args: Record<string, string>): [unknown, boolean] {
  const allOrders = new Map(Object.values(ORDERS).flat().map((o) => [o.order_id, o]));
  if (name === "lookup_customer") {
    const c = CUSTOMERS[args.email!];
    return c ? [c, false] : [{ error: "no such customer" }, true];
  }
  if (name === "list_orders") return [ORDERS[args.customer_id!] ?? [], false];
  if (name === "check_warranty" || name === "create_replacement") {
    const order = allOrders.get(args.order_id!);
    if (!order) return [{ error: "unknown order" }, true];
    const days = Math.round((Date.parse(`${TODAY}T00:00:00Z`) - Date.parse(`${order.delivered_on}T00:00:00Z`)) / 86_400_000);
    const eligible = days <= 7;
    if (name === "check_warranty") return [{ order_id: order.order_id, days_since_delivery: days, transit_damage_eligible: eligible }, false];
    if (!eligible) return [{ error: "outside 7-day transit-damage window; escalate to a human" }, true];
    const rep = { replacement_id: `R-${5001 + REPLACEMENTS.length}`, order_id: order.order_id, ship_eta: "2026-10-14" };
    REPLACEMENTS.push(rep);
    return [rep, false];
  }
  return [{ error: `unknown tool ${name}` }, true];
}

const AGENT_SYSTEM =
  "You are an autonomous support agent for Aurora Home. Today is 2026-10-11. Use the tools to " +
  "investigate and resolve the issue end to end, then summarise what you did.";

async function runAgent(client: Anthropic, task: string, maxSteps: number): Promise<void> {
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: task }];
  for (let step = 1; step <= maxSteps; step++) {
    const msg = await call(client, "7 autonomous agent", { model: MODEL_OPUS, max_tokens: 2048, system: AGENT_SYSTEM, tools: AGENT_TOOLS, messages });
    if (msg.stop_reason === "end_turn") {
      console.log(`    step ${step} (in=${msg.usage.input_tokens}): end_turn -> ${textOf(msg)}`);
      return;
    }
    if (msg.stop_reason !== "tool_use") {
      console.log(`    step ${step}: stop_reason=${msg.stop_reason} -> hand off to a human`);
      return;
    }
    messages.push({ role: "assistant", content: msg.content });
    const results: Anthropic.ToolResultBlockParam[] = [];
    for (const block of msg.content) {
      if (block.type !== "tool_use") continue;
      const [out, isError] = executeTool(block.name, block.input as Record<string, string>);
      console.log(`    step ${step} (in=${msg.usage.input_tokens}): ${block.name}(${cj(block.input)}) -> ${cj(out)}${isError ? " (is_error)" : ""}`);
      results.push({ type: "tool_result", tool_use_id: block.id, content: cj(out), is_error: isError });
    }
    messages.push({ role: "user", content: results });
  }
  console.log(`    step limit (${maxSteps}) reached -> stop, escalate to a human with the transcript`);
}

// ===========================================================================
// 8. PATTERN SELECTOR (deterministic)
// ===========================================================================
interface Scenario {
  name: string;
  stepsKnown?: boolean;
  multiStep?: boolean;
  distinctCategories?: boolean;
  independentSubtasks?: boolean;
  subtasksKnownUpfront?: boolean;
  needsConsensus?: boolean;
  clearCriteriaIterative?: boolean;
  openEnded?: boolean;
}

const PROFILE: Record<string, [string, string, string]> = {
  // predictability, relative cost, latency shape
  "augmented LLM": ["high", "1x", "one call (+tool round trips)"],
  "prompt chaining": ["high", "2-4x", "sum of steps"],
  routing: ["high", "~1.1x", "router + one handler"],
  "parallel: sectioning": ["high", "Nx", "slowest branch"],
  "parallel: voting": ["high", "Nx", "slowest voter"],
  "orchestrator-workers": ["medium", "3-10x", "plan + workers + synthesis"],
  "evaluator-optimizer": ["medium", "2-6x", "iterations x (gen + eval)"],
  "autonomous agent": ["low", "5-20x+", "unbounded -> step limit"],
};

function selectPattern(s: Scenario): [string, string] {
  const stepsKnown = s.stepsKnown ?? true;
  const knownUpfront = s.subtasksKnownUpfront ?? true;
  if (s.openEnded && !stepsKnown) return ["autonomous agent", "path depends on what each tool call reveals"];
  if (s.independentSubtasks && !knownUpfront) return ["orchestrator-workers", "subtasks depend on the input; cannot be enumerated in advance"];
  if (s.needsConsensus) return ["parallel: voting", "independent judgments + majority raise recall/confidence"];
  if (s.independentSubtasks) return ["parallel: sectioning", "known independent aspects; latency = slowest branch"];
  if (s.distinctCategories) return ["routing", "distinct input classes served best by specialised prompts/models"];
  if (s.clearCriteriaIterative) return ["evaluator-optimizer", "clear criteria and iteration measurably improves output"];
  if (s.multiStep) return ["prompt chaining", "fixed sequence of simpler steps with gates"];
  return ["augmented LLM", "one call with retrieval/tools is enough"];
}

const SCENARIOS: Scenario[] = [
  { name: "FAQ answers from a product knowledge base" },
  { name: "Inbound email: sales vs support vs billing replies", distinctCategories: true },
  { name: "Contract memo: extract clauses -> compare to playbook -> draft", multiStep: true },
  { name: "Code review for security, performance and style", independentSubtasks: true },
  { name: "Flag fraudulent marketplace listings (misses are costly)", needsConsensus: true },
  { name: "Implement a feature touching an unknown set of files", independentSubtasks: true, subtasksKnownUpfront: false },
  { name: "Localise brand copy until a tone rubric is met", clearCriteriaIterative: true },
  { name: "Resolve open-ended issues across several back-office systems", stepsKnown: false, openEnded: true },
];

// ===========================================================================
// Offline responder - one function serving every pattern (dispatch on the system prompt)
// ===========================================================================
function responder(req: MockRequest): ResponderResult {
  const s = req.systemText;
  const u = req.lastUserText;
  if (s.includes("support assistant for Aurora Home")) {
    if (!req.toolResults.length)
      return [
        textBlock("Let me check that order."),
        toolUseBlock("get_order_status", { order_id: "A-7731" }),
        toolUseBlock("save_memory", { fact: "started a return for order A-7731 (Aurora blender) on 2026-10-11" }),
      ];
    return (
      "Yes. Order A-7731 (Aurora blender) was delivered on 2026-10-02, so you are inside the 30-day window " +
      "and can return it until 2026-11-01 [returns-policy]. As a Gold member your return shipping is free " +
      "[loyalty]. The refund reaches your original payment method within 5 business days of us receiving it " +
      "[returns-policy]. I will email the prepaid label, as you prefer email."
    );
  }
  if (s.includes("Extract the customer-visible changes")) {
    if (u.includes("Gate feedback"))
      return "- New 'Export to PDF' button on dashboards\n- Dashboards load more than twice as fast\n- Fixed an error when exporting an empty dashboard";
    return (
      "- JIRA-4830: new 'Export to PDF' button on dashboards\n- Dashboards load faster (p95 2.1s -> 0.9s)\n" +
      "- Fixed an error when exporting an empty dashboard\n- Upgraded pdfkit to 3.2"
    );
  }
  if (s.includes("customer-facing release note"))
    return "New this week: export any dashboard to PDF with one click, dashboards now load more than twice as fast, and exporting an empty dashboard no longer shows an error.";
  if (s.includes("You route customer queries")) {
    if (/refund|charged/i.test(u)) return JSON.stringify({ route: "refund", reason: "billing dispute with refund request" });
    if (/error|crash/i.test(u)) return JSON.stringify({ route: "technical", reason: "reproducible crash with an error code" });
    return JSON.stringify({ route: "general", reason: "simple product question" });
  }
  if (s.includes("refund specialist")) return "Duplicate charges are refunded in full: I have flagged the second charge; expect it back in 5 business days.";
  if (s.includes("technical support engineer")) return "E1043 means the shared dashboard references a deleted data source: re-link the source, then clear the cache.";
  if (s.includes("general assistant")) return "Yes - turn it on under Settings > Appearance > Dark mode.";
  const findings: [string, string][] = [
    ["security", "SQL injection: user_id is interpolated into the query; use a parameterised query."],
    ["performance", "SELECT * plus a sleep per row; select one row by primary key instead."],
    ["readability", "rows[0] raises IndexError when no user exists; return None and add type hints."],
  ];
  for (const [aspect, finding] of findings) if (s.includes(`a ${aspect} reviewer`)) return finding;
  if (s.includes("content-policy reviewer")) return s.includes("lenient") ? "OK - insulting but no explicit threat" : "VIOLATION - implied threat to staff safety";
  if (s.includes("You are an orchestrator"))
    return JSON.stringify({
      subtasks: [
        { id: "w1", area: "frontend", instruction: "Add an Export to PDF button with progress state to the dashboard toolbar." },
        { id: "w2", area: "backend", instruction: "Create POST /dashboards/{id}/export that renders the dashboard to PDF." },
        { id: "w3", area: "scheduler", instruction: "Add a weekly schedule option and an email job attaching the PDF." },
      ],
    });
  if (s.includes("worker engineer for the frontend")) return "ExportButton component in DashboardToolbar; poll export job status; show toast with download link.";
  if (s.includes("worker engineer for the backend")) return "Async export job: headless renderer -> PDF in object storage; endpoint returns job id; 10/min rate limit.";
  if (s.includes("worker engineer for the scheduler")) return "weekly_export cron per subscription; reuse export job; email via notification service with signed URL.";
  if (s.includes("Integrate the worker outputs"))
    return "Order: (1) backend export job + endpoint, (2) frontend button against it, (3) scheduler reusing the job. Ship 1+2 behind a flag, then 3.";
  if (s.includes("You write product descriptions")) {
    if (u.includes("unsupported"))
      return "Meet the Aurora blender: a 1200 W motor, a 1.5 L glass jar and 6 programs for smoothies, soups and sauces. Backed by a 2-year warranty.";
    if (u.includes("Reviewer feedback"))
      return "Meet the Aurora blender: a 1200 W motor, a 1.5 L glass jar, 6 programs and dishwasher-safe titanium blades. Backed by a 2-year warranty.";
    return (
      "The Aurora is the best blender you will ever own. Its ultimate 1200 W motor crushes ice in seconds, " +
      "the elegant 1.5 L glass jar looks perfect on any counter, and six smart programs handle smoothies, " +
      "soups, nut butters, sauces and frozen desserts so you never have to guess again. It is simply the " +
      "most versatile kitchen companion for busy families who love fresh, healthy food every day."
    );
  }
  if (s.includes("You evaluate product descriptions")) {
    if (u.includes("titanium")) return JSON.stringify({ passed: false, feedback: "unsupported claim: 'dishwasher-safe titanium blades' is not in the facts" });
    return JSON.stringify({ passed: true, feedback: "accurate, within limits, warranty mentioned" });
  }
  if (s.includes("autonomous support agent")) {
    const called = req.messages
      .filter((m) => m.role === "assistant" && Array.isArray(m.content))
      .flatMap((m) => m.content as Json[])
      .filter((b) => b?.type === "tool_use")
      .map((b) => String(b.name));
    if (!called.includes("lookup_customer")) return [textBlock("I'll identify the customer first."), toolUseBlock("lookup_customer", { email: "jane@example.com" })];
    if (!called.includes("list_orders")) return [textBlock("Now her orders."), toolUseBlock("list_orders", { customer_id: "C-1042" })];
    if (!called.includes("check_warranty"))
      return [textBlock("A-7790 is the recent blender; checking eligibility."), toolUseBlock("check_warranty", { order_id: "A-7790" })];
    if (!called.includes("create_replacement"))
      return [
        textBlock("Eligible - creating the replacement."),
        toolUseBlock("create_replacement", { order_id: "A-7790", reason: "glass jar arrived cracked (transit damage)" }),
      ];
    return (
      "Resolved: replacement R-5001 ships for order A-7790 (delivered 3 days ago, inside the 7-day " +
      "transit-damage window), ETA 2026-10-14. No return of the cracked jar needed."
    );
  }
  return "OK";
}

// ===========================================================================
// Main
// ===========================================================================
banner("D1.3 - Architectural Patterns - Workflow, Agentic, Augmented LLM");
const client = getClient(responder);

section("1. Augmented LLM: retrieval + tools + memory");
await runAugmented(client);

section("2. Workflow - prompt chaining with a gate");
await runChaining(client);

section("3. Workflow - routing");
await runRouting(client);

section("4. Workflow - parallelization (sectioning + voting)");
await runParallel(client);

section("5. Workflow - orchestrator-workers");
await runOrchestrator(client);

section("6. Workflow - evaluator-optimizer");
await runEvaluatorOptimizer(client);

section("7. Autonomous agent loop (tool use until end_turn)");
const task = "Customer jane@example.com says the glass jar of her new Aurora blender arrived cracked. Resolve it.";
console.log("  run A: max_steps=6   (note 'in=' grows: the whole transcript is resent every step)");
await runAgent(client, task, 6);
console.log("  run B: same task, max_steps=2 (guardrail demo)");
await runAgent(client, task, 2);

section("Cost per pattern (this run)");
console.log(`  ${"pattern".padEnd(24)}${"calls".padStart(6)}${"in tok".padStart(8)}${"out tok".padStart(9)}${"cost $".padStart(10)}`);
for (const [name, row] of LEDGER)
  console.log(`  ${name.padEnd(24)}${String(row.calls).padStart(6)}${String(row.in).padStart(8)}${String(row.out).padStart(9)}${row.cost.toFixed(5).padStart(10)}`);
console.log("  (agent row includes run B) - autonomy costs tokens: tool schemas + growing history on every call");

section("8. Pattern selector: scenario -> recommended pattern");
for (const s of SCENARIOS) {
  const [pattern, why] = selectPattern(s);
  const [pred, cost, lat] = PROFILE[pattern]!;
  console.log(`  ${s.name}`);
  console.log(`     -> ${pattern.padEnd(21)} | ${why}`);
  console.log(`        predictability=${pred}, relative cost=${cost}, latency=${lat}`);
}

section("Summary");
console.log("  Start with an augmented LLM; add workflow structure when steps are known; reach for an");
console.log("  agent only when the path is genuinely open-ended - and then bound it (step limits, tool policy).");
