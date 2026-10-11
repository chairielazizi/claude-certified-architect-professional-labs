/**
 * D2.3 - Prompting Techniques: Zero-shot, Few-shot, Chain-of-Thought
 * ==================================================================
 *
 * One labelled task set, four techniques, one code grader:
 *
 *   - Task A - classify support emails into a nuanced 6-label taxonomy with
 *     precedence rules (12 labelled emails, several deliberately ambiguous).
 *   - Task B - small multi-step arithmetic problems (5, exact numeric answers).
 *
 * Techniques (same model - Claude Haiku 5.5 - so only the prompt changes):
 *   zero-shot  : instructions + taxonomy only
 *   few-shot   : + 18 balanced, diverse examples in <example> tags (cached prefix)
 *   cot        : + "brief analysis in <reasoning>, label in <answer>" (thinking off)
 *   thinking   : zero-shot prompt + adaptive thinking (display "summarized") -
 *                the model-native alternative to prompted chain-of-thought
 *
 * The grader is real code: strict format check + lenient label/number
 * extraction, accuracy, tokens, cost and latency per technique. Also shown:
 * the `reasoning_extraction` refusal trap on Opus 5.5, and few-shot caching math.
 */

import type Anthropic from "@anthropic-ai/sdk";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  MODEL_HAIKU,
  MODEL_OPUS,
  PRICING,
  type MockRequest,
  type ResponderResult,
  banner,
  costUsd,
  elapsedMs,
  getClient,
  section,
  textOf,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");

interface Example { id: string; label: string; email: string }
interface TestItem extends Example { rationale: string }
interface Problem { id: string; question: string; worked: string; answer: number }
interface Tasks {
  taxonomy: { label: string; definition: string }[];
  rules: string[];
  few_shot_pool: Example[];
  test_set: TestItem[];
  reasoning_examples: { question: string; worked: string; answer: number }[];
  reasoning_test: Problem[];
  offline_simulation: {
    classification_errors: Record<string, Record<string, string>>;
    zero_shot_verbose: Record<string, string>;
    cot_wrong_rationale: Record<string, string>;
    reasoning_errors: Record<string, Record<string, number>>;
  };
}

const TASKS = JSON.parse(readFileSync(path.join(DATA, "tasks.json"), "utf-8")) as Tasks;
const TAXONOMY = TASKS.taxonomy;
const LABELS = TAXONOMY.map((t) => t.label);
const SIM = TASKS.offline_simulation;

type Technique = "zero_shot" | "few_shot" | "cot" | "thinking";
const TECHNIQUES: Technique[] = ["zero_shot", "few_shot", "cot", "thinking"];
const NAMES: Record<Technique, string> = { zero_shot: "zero-shot", few_shot: "few-shot", cot: "chain-of-thought", thinking: "adaptive thinking" };

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------
const CLASSIFY_ROLE =
  "You are a support email classifier for a B2B SaaS company. Assign exactly one label from the " +
  "taxonomy to the email inside <email> tags. Read the definitions and rules carefully: several labels " +
  "overlap and the rules decide between them.";
const MATH_ROLE = "You are a precise analyst for a SaaS operations team. Solve the problem inside <problem> tags.";
const LABEL_ONLY = "Respond with only the label, nothing else.";
const NUMBER_ONLY = "Respond with only the final number: no units, no words.";
const CLASSIFY_COT =
  "Before answering, write a brief analysis inside <reasoning> tags: name the candidate labels and the rule " +
  "that decides between them. Then output only the label inside <answer> tags.";
const MATH_COT = "Work through the problem step by step inside <reasoning> tags, then give only the final number inside <answer> tags.";

function taxonomyBlock(): string {
  const lines = ["<taxonomy>", ...TAXONOMY.map((t) => `- ${t.label}: ${t.definition}`), "</taxonomy>", "<rules>"];
  lines.push(...TASKS.rules.map((r, i) => `${i + 1}. ${r}`), "</rules>");
  return lines.join("\n");
}

/**
 * Balanced (same count per label) and interleaved (no label clustered at the end,
 * which would bias the model toward the most recent label).
 */
function selectExamples(pool: Example[], perLabel: number): Example[] {
  const byLabel = Object.fromEntries(LABELS.map((l) => [l, pool.filter((e) => e.label === l).slice(0, perLabel)]));
  const picked: Example[] = [];
  for (let i = 0; i < perLabel; i++) for (const l of LABELS) if (i < byLabel[l]!.length) picked.push(byLabel[l]![i]!);
  return picked;
}

const words = (text: string): Set<string> => new Set(text.toLowerCase().match(/[a-z0-9']+/g) ?? []);
function jaccard(a: string, b: string): number {
  const wa = words(a);
  const wb = words(b);
  const inter = [...wa].filter((w) => wb.has(w)).length;
  const union = new Set([...wa, ...wb]).size;
  return inter / Math.max(1, union);
}

function examplesBlock(examples: Example[]): string {
  const parts = ["<examples>"];
  for (const e of examples) parts.push(`<example>\n<email>${e.email}</email>\n<label>${e.label}</label>\n</example>`);
  parts.push("</examples>");
  return parts.join("\n");
}

function classifySystem(technique: Technique, examples: Example[]): Anthropic.TextBlockParam[] {
  let text = CLASSIFY_ROLE + "\n\n" + taxonomyBlock();
  if (technique === "few_shot") text += "\n\n" + examplesBlock(examples);
  text += "\n\n" + (technique === "cot" ? CLASSIFY_COT : LABEL_ONLY);
  // Same breakpoint for every technique; only a prefix above the model's minimum
  // cacheable size is actually cached (silently skipped otherwise).
  return [{ type: "text", text, cache_control: { type: "ephemeral" } }];
}

function mathSystem(technique: Technique): Anthropic.TextBlockParam[] {
  let text = MATH_ROLE;
  if (technique === "few_shot") {
    const ex = TASKS.reasoning_examples
      .map((e) => `<example>\n<problem>${e.question}</problem>\n<answer>${e.answer}</answer>\n</example>`)
      .join("\n");
    text += "\n\n<examples>\n" + ex + "\n</examples>";
  }
  text += "\n\n" + (technique === "cot" ? MATH_COT : NUMBER_ONLY);
  return [{ type: "text", text }];
}

type RequestOptions = Pick<Anthropic.MessageCreateParamsNonStreaming, "max_tokens" | "thinking" | "output_config">;

function requestOptions(technique: Technique): RequestOptions {
  if (technique === "thinking") {
    // Model-native reasoning: thinking tokens bill as output; display "summarized"
    // returns a readable summary (the default on current models is omitted text).
    return { max_tokens: 4000, thinking: { type: "adaptive", display: "summarized" }, output_config: { effort: "medium" } };
  }
  // Haiku 5.5 accepts thinking "disabled" at effort <= high: isolates the prompt technique.
  // (Never send "disabled" to Opus 5.5 / Sonnet 5.5 - it is a 400 there.)
  return { max_tokens: 600, thinking: { type: "disabled" }, output_config: { effort: "low" } };
}

// ---------------------------------------------------------------------------
// Graders (pure code)
// ---------------------------------------------------------------------------
const ANSWER_RE = /<answer>\s*([\s\S]*?)\s*<\/answer>/;
const NUMBER_RE = /-?\d[\d,]*\.?\d*/g;

function extractAnswer(text: string): string {
  const m = text.match(ANSWER_RE);
  return m ? m[1]! : text.trim();
}

/** Return [predicted label or null, strict-format-ok]. */
function gradeLabel(text: string, technique: Technique): [string | null, boolean] {
  const strict =
    technique === "cot" ? /<reasoning>[\s\S]+<\/reasoning>/.test(text) && ANSWER_RE.test(text) : LABELS.includes(text.trim());
  const ans = extractAnswer(text).trim().toLowerCase().replace(/\.+$/, "");
  if (LABELS.includes(ans)) return [ans, strict];
  const found = LABELS.filter((l) => new RegExp(`\\b${l}\\b`).test(ans));
  return [found.length === 1 ? found[0]! : null, strict];
}

function gradeNumber(text: string, expected: number): [number | null, boolean] {
  const nums = extractAnswer(text).match(NUMBER_RE);
  if (!nums) return [null, false];
  const value = parseFloat(nums[nums.length - 1]!.replace(/,/g, ""));
  return [value, Math.abs(value - expected) <= 1e-6 * Math.max(1, Math.abs(expected))];
}

// ---------------------------------------------------------------------------
// Offline responder: simulates typical technique-dependent behaviour
// ---------------------------------------------------------------------------
function detectTechnique(req: MockRequest): Technique {
  if (req.body.thinking?.type === "adaptive") return "thinking";
  if (req.systemText.includes("<reasoning>")) return "cot";
  if (req.systemText.includes("<examples>")) return "few_shot";
  return "zero_shot";
}

function responder(req: MockRequest): ResponderResult {
  const user = req.lastUserText;
  const tech = detectTechnique(req);
  if (user.includes("internal thinking verbatim")) {
    return {
      content: [],
      stop_reason: "refusal",
      stop_details: { type: "refusal", category: "reasoning_extraction", explanation: "The request asks the model to reproduce its internal reasoning." },
    };
  }
  if (req.systemText.includes("support email classifier")) {
    const item = TASKS.test_set.find((e) => user.includes(e.email))!;
    const pred = SIM.classification_errors[tech]![item.id] ?? item.label;
    if (tech === "zero_shot") return pred === item.label ? (SIM.zero_shot_verbose[item.id] ?? pred) : pred;
    if (tech === "few_shot") return pred;
    const rationale = pred !== item.label ? (SIM.cot_wrong_rationale[item.id] ?? item.rationale) : item.rationale;
    if (tech === "cot") return `<reasoning>${rationale}</reasoning>\n<answer>${pred}</answer>`;
    return [
      { type: "thinking", thinking: `Checking the taxonomy rules. ${rationale}`, signature: "sig_offline" },
      { type: "text", text: pred },
    ];
  }
  const prob = TASKS.reasoning_test.find((p) => user.includes(p.question))!;
  const value = SIM.reasoning_errors[tech]![prob.id] ?? prob.answer;
  if (user.includes("one-sentence justification")) {
    return [
      { type: "thinking", thinking: `Summary: ${prob.worked}`, signature: "sig_offline" },
      { type: "text", text: `<answer>${value}</answer>\nJustification: 0.5% of the 43,200 minutes in a 30-day month.` },
    ];
  }
  if (tech === "cot") return `<reasoning>${prob.worked}</reasoning>\n<answer>${value}</answer>`;
  if (tech === "thinking") {
    return [
      { type: "thinking", thinking: `Summary: ${prob.worked}`, signature: "sig_offline" },
      { type: "text", text: String(value) },
    ];
  }
  return String(value);
}

// ---------------------------------------------------------------------------
// Runner + metrics
// ---------------------------------------------------------------------------
class Stats {
  n = 0;
  correct = 0;
  strict = 0;
  inputTokens = 0;
  outputTokens = 0;
  cacheRead = 0;
  cost = 0;
  latencyMs = 0;
  misses: Record<string, string> = {};

  add(msg: Anthropic.Message, ms: number): void {
    const u = msg.usage;
    this.n += 1;
    this.inputTokens += u.input_tokens + (u.cache_creation_input_tokens ?? 0) + (u.cache_read_input_tokens ?? 0);
    this.outputTokens += u.output_tokens;
    this.cacheRead += u.cache_read_input_tokens ?? 0;
    this.cost += costUsd(msg.model, u);
    this.latencyMs += ms;
  }
}

async function call(
  client: Anthropic,
  system: Anthropic.TextBlockParam[],
  user: string,
  technique: Technique,
  model: string = MODEL_HAIKU,
): Promise<[Anthropic.Message, number]> {
  const t0 = performance.now();
  const msg = await client.messages.create({ model, system, messages: [{ role: "user", content: user }], ...requestOptions(technique) });
  return [msg, elapsedMs(t0)];
}

function table(headers: string[], rows: string[][]): void {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i]!.length)));
  console.log("  " + headers.map((h, i) => h.padEnd(widths[i]!)).join("  "));
  console.log("  " + widths.map((w) => "-".repeat(w)).join("  "));
  for (const r of rows) console.log("  " + r.map((c, i) => c.padEnd(widths[i]!)).join("  "));
}

const pct = (a: number, b: number): string => `${a}/${b} (${Math.round((100 * a) / b)}%)`;
const num = (x: number, d = 0): string => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const thinkingOf = (msg: Anthropic.Message): string =>
  msg.content.find((b): b is Anthropic.ThinkingBlock => b.type === "thinking")?.thinking ?? "";

banner("D2.3 - Prompting Techniques - Zero-shot, Few-shot, Chain-of-Thought");
const client = getClient(responder);
const tests = TASKS.test_set;
const problems = TASKS.reasoning_test;

// ------------------------------------------------------------------ 1
section("1. Task set and graders");
console.log(
  `  Task A: ${tests.length} labelled emails, ${LABELS.length} labels: ` +
    LABELS.map((l) => `${l}=${tests.filter((e) => e.label === l).length}`).join(", "),
);
console.log(`  Task B: ${problems.length} multi-step arithmetic problems with exact answers`);
console.log("  Grader: strict format check + lenient extraction (<answer> tag, single label / last number).");
console.log("  Model: Claude Haiku 5.5 for every technique, so only the prompt changes.");

// ------------------------------------------------------------------ 2
section("2. Few-shot example selection: balanced, diverse, no leakage");
const examples = selectExamples(TASKS.few_shot_pool, 3);
console.log("  order : " + examples.map((e) => `${e.id}:${e.label.slice(0, 8)}`).join(" "));
console.log("  counts: " + LABELS.map((l) => `${l}=${examples.filter((e) => e.label === l).length}`).join(", "));
let worst: [number, string, string] = [-1, "", ""];
for (const e of examples) {
  for (const t of tests) {
    const s = jaccard(e.email, t.email);
    const cand: [number, string, string] = [s, e.id, t.id];
    // tuple max like Python: compare score, then ids
    if (cand[0] > worst[0] || (cand[0] === worst[0] && (cand[1] > worst[1] || (cand[1] === worst[1] && cand[2] > worst[2])))) worst = cand;
  }
}
console.log(
  `  leakage check: max word-overlap example vs test = ${worst[0].toFixed(2)} (${worst[1]} vs ${worst[2]}) -> ${worst[0] < 0.6 ? "OK" : "LEAK - replace example"}`,
);
const sampleUser = `<email>${tests[0]!.email}</email>`;
const zeroTok = (await client.messages.countTokens({ model: MODEL_HAIKU, system: classifySystem("zero_shot", examples), messages: [{ role: "user", content: sampleUser }] })).input_tokens;
const fewTok = (await client.messages.countTokens({ model: MODEL_HAIKU, system: classifySystem("few_shot", examples), messages: [{ role: "user", content: sampleUser }] })).input_tokens;
const overhead = fewTok - zeroTok;
console.log(
  `  count_tokens: zero-shot request ${zeroTok} tok, few-shot request ${fewTok} tok -> examples add ${overhead} input tokens to EVERY call`,
);

// ------------------------------------------------------------------ 3
section("3. Task A - email classification, 4 techniques");
const clsStats = {} as Record<Technique, Stats>;
let sampleCot = "";
let sampleThinking = "";
for (const tech of TECHNIQUES) {
  const st = new Stats();
  const system = classifySystem(tech, examples);
  for (const item of tests) {
    const [msg, ms] = await call(client, system, `<email>${item.email}</email>`, tech);
    st.add(msg, ms);
    if (msg.stop_reason !== "end_turn") {
      st.misses[item.id] = `stop:${msg.stop_reason}`;
      continue;
    }
    const text = textOf(msg);
    const [pred, strict] = gradeLabel(text, tech);
    if (strict) st.strict += 1;
    if (pred === item.label) st.correct += 1;
    else st.misses[item.id] = pred ?? "unparseable";
    if (item.id === "E07" && tech === "cot") sampleCot = text;
    if (item.id === "E07" && tech === "thinking") sampleThinking = thinkingOf(msg);
  }
  clsStats[tech] = st;
}
table(
  ["technique", "accuracy", "strict format", "avg in", "avg out", "cache rd", "$/1k req", "avg latency"],
  TECHNIQUES.map((tech) => {
    const st = clsStats[tech];
    return [
      NAMES[tech], pct(st.correct, st.n), pct(st.strict, st.n), String(Math.floor(st.inputTokens / st.n)),
      String(Math.floor(st.outputTokens / st.n)), String(st.cacheRead), `$${((st.cost / st.n) * 1000).toFixed(4)}`,
      `${num(st.latencyMs / st.n)} ms`,
    ];
  }),
);
console.log("  Per-item misses (item: predicted):");
for (const tech of TECHNIQUES) {
  const misses = Object.entries(clsStats[tech].misses).map(([k, v]) => `${k}:${v}`);
  console.log(`    ${NAMES[tech].padEnd(18)} ${misses.join(", ") || "none"}`);
}
console.log("  E07 under chain-of-thought (the precedence rule is applied explicitly):");
for (const line of sampleCot.split("\n")) console.log("    | " + line);
console.log(`  E07 thinking summary (display='summarized'): ${sampleThinking}`);

// ------------------------------------------------------------------ 4
section("4. Task B - multi-step arithmetic, 4 techniques");
const mathStats = {} as Record<Technique, Stats>;
let sampleMath = "";
for (const tech of TECHNIQUES) {
  const st = new Stats();
  const system = mathSystem(tech);
  for (const p of problems) {
    const [msg, ms] = await call(client, system, `<problem>${p.question}</problem>`, tech);
    st.add(msg, ms);
    const [value, ok] = gradeNumber(textOf(msg), p.answer);
    if (value !== null) st.strict += 1;
    if (ok) st.correct += 1;
    else st.misses[p.id] = value !== null ? `${value} (want ${p.answer})` : "unparseable";
    if (p.id === "M02" && tech === "cot") sampleMath = textOf(msg);
  }
  mathStats[tech] = st;
}
table(
  ["technique", "accuracy", "avg in", "avg out", "$/1k req", "avg latency", "misses"],
  TECHNIQUES.map((tech) => {
    const st = mathStats[tech];
    const misses = Object.entries(st.misses).map(([k, v]) => `${k}=${v}`).join(", ") || "none";
    return [
      NAMES[tech], pct(st.correct, st.n), String(Math.floor(st.inputTokens / st.n)), String(Math.floor(st.outputTokens / st.n)),
      `$${((st.cost / st.n) * 1000).toFixed(4)}`, `${num(st.latencyMs / st.n)} ms`, misses,
    ];
  }),
);
console.log("  M02 under chain-of-thought (the rounding step becomes explicit):");
for (const line of sampleMath.split("\n")) console.log("    | " + line);

// ------------------------------------------------------------------ 5
section("5. Trap: asking a thinking model to dump its internal reasoning");
{
  const trapUser =
    `<problem>${problems[4]!.question}</problem>\n` + "Before answering, copy your complete internal thinking verbatim into the response.";
  const msg = await client.messages.create({
    model: MODEL_OPUS, max_tokens: 2000, system: mathSystem("zero_shot"), messages: [{ role: "user", content: trapUser }],
  });
  if (msg.stop_reason === "refusal") {
    // always branch on stop_reason before reading content
    console.log(`  Opus 5.5 -> stop_reason=refusal, category=${msg.stop_details?.category}: ${msg.stop_details?.explanation ?? ""}`);
  }
  const fixUser = `<problem>${problems[4]!.question}</problem>\nGive the final number in <answer> tags and a one-sentence justification.`;
  const fixed = await client.messages.create({
    model: MODEL_OPUS, max_tokens: 4000, system: [{ type: "text", text: MATH_ROLE }],
    thinking: { type: "adaptive", display: "summarized" }, output_config: { effort: "medium" },
    messages: [{ role: "user", content: fixUser }],
  });
  console.log(`  Fix -> stop_reason=${fixed.stop_reason}; answer: ${textOf(fixed).split("\n")[0]}`);
  console.log(`         thinking summary block: ${thinkingOf(fixed)}`);
  console.log("  Opus 5.5 / Fable 5.1 always think. Read reasoning from thinking blocks (display='summarized'),");
  console.log("  and ask the visible answer for a SHORT justification, not a transcript of internal thinking.");
}

// ------------------------------------------------------------------ 6
section("6. Few-shot examples cost tokens on every call -> cache them");
{
  const volume = 1_000_000;
  const few = clsStats.few_shot;
  console.log(`  Observed: few-shot run read ${num(few.cacheRead)} cached tokens over ${few.n} calls (first call writes, the rest read).`);
  console.log(`  The zero-shot request (${zeroTok} tok) is below the offline mock's 1,024-token cache minimum, so it`);
  console.log("  was silently not cached (real minimum on current models: 512 tokens).");
  const zs = clsStats.zero_shot;
  if (few.cost / few.n < zs.cost / zs.n) {
    console.log(
      `  With a warm cache, few-shot cost $${((few.cost / few.n) * 1000).toFixed(4)}/1k requests vs $${((zs.cost / zs.n) * 1000).toFixed(4)} ` +
        "for uncached zero-shot: bigger, cached prompts can be CHEAPER.",
    );
  }
  const rows: string[][] = [];
  for (const [model, name] of [[MODEL_HAIKU, "Haiku 5.5"], [MODEL_OPUS, "Opus 5.5"]] as const) {
    const p = PRICING[model]!;
    const uncached = (overhead * volume * p.input) / 1e6;
    const cached = (overhead * volume * p.cacheRead) / 1e6;
    rows.push([name, `${overhead} tok`, `$${num(uncached, 2)}`, `$${num(cached, 2)}`, `${Math.round((1 - cached / uncached) * 100)}%`]);
  }
  table(["model", "example overhead", "uncached / 1M req", "cached / 1M req", "saved"], rows);
  console.log("  Keep examples in the stable prefix (system), never interleaved with per-request content.");
}

// ------------------------------------------------------------------ 7
section("7. When each technique wins");
{
  const zs = clsStats.zero_shot;
  const fs = clsStats.few_shot;
  const th = clsStats.thinking;
  console.log(
    `  zero-shot        : smallest prompt, fine for unambiguous labels; missed the precedence rules (${pct(zs.correct, zs.n)}) and drifted off-format (${pct(zs.strict, zs.n)} strict).`,
  );
  console.log(`  few-shot         : fixes label boundaries AND output format (${pct(fs.strict, fs.n)} strict); costs +${overhead} input tok/call - cache them.`);
  console.log(
    `  chain-of-thought : best when the task needs explicit steps (math ${pct(mathStats.cot.correct, problems.length)}); pays in output tokens and latency.`,
  );
  console.log(`  adaptive thinking: model-native reasoning, answer stays clean (${pct(th.correct, th.n)} on Task A); tune cost with effort, not prompt hacks.`);
  console.log("  Combine them: few-shot examples of the FORMAT + thinking for hard reasoning, all on a cached prefix.");
}
