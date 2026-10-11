/**
 * D7.2 - AI-Assisted Developer Workflows (TypeScript twin of python/main.py)
 *
 * Four workflows a platform team typically automates with Claude, each wrapped in deterministic code:
 *   a) AI PR reviewer: parse diff -> chunk -> structured findings -> ground/filter/dedupe -> PR comments
 *   b) Conventional Commit message + changelog, with a validate-and-repair loop
 *   c) test-case generation for a small function, validated and executed against code AND spec
 *   d) how it runs in CI: GitHub Actions + headless `claude -p --output-format json` (linted, not executed)
 * and measures tokens/cost per PR and findings by severity.
 */

import type Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET, PRICING, banner, costUsd, getClient, section, textOf,
} from "../../shared/typescript/claudeClient.ts";
import type { Json, MockRequest } from "../../shared/typescript/claudeClient.ts";
import {
  CC_TYPES, CHANGELOG_SECTIONS, additions, applyDiscountFromDiff, applyDiscountReference, buildChunks, commitHeader,
  coverage, deletions, gateHeadless, lintWorkflow, parseUnifiedDiff, postprocess, renderChangelog, renderCommitMessage,
  renderPrMarkdown, renderPytest, reviewPayload, runCase, severityCounts, splitLines, triage, validateCases, validateCommit,
} from "./devflow.ts";
import type { Chunk, CommitDraft, FileDiff, GeneratedCase, ReviewComment } from "./devflow.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(HERE, "..", "data");
const OUT = path.join(HERE, "..", "labs_output");
const MAX_CHUNK_TOKENS = 160; // deliberately small so the demo shows a file split across chunks
const MIN_CONFIDENCE = 0.6;
const PRS_PER_MONTH = 400;

const read = (...p: string[]): string => readFileSync(path.join(DATA, ...p), "utf8").replace(/\r\n/g, "\n");

// ---------------------------------------------------------------------------
// Structured-output schemas (zodOutputFormat turns these into output_config.format JSON schemas)
// ---------------------------------------------------------------------------
const ReviewFinding = z.object({
  file: z.string(),
  line: z.number().int(),
  severity: z.enum(["critical", "high", "medium", "low", "nit"]),
  category: z.enum(["security", "correctness", "performance", "maintainability", "tests", "style"]),
  title: z.string(),
  suggestion: z.string(),
  confidence: z.number(),
});
const ReviewResult = z.object({ findings: z.array(ReviewFinding) });
const CommitDraftSchema = z.object({
  type: z.enum(CC_TYPES),
  scope: z.string(),
  subject: z.string(),
  body: z.array(z.string()),
  breaking: z.boolean(),
  changelog: z.array(z.object({ section: z.enum(CHANGELOG_SECTIONS), text: z.string() })),
});
const GeneratedSuite = z.object({
  cases: z.array(
    z.object({
      name: z.string(),
      price_cents: z.number().int(),
      percent: z.number().int(),
      expected: z.number().int().nullable(),
      raises: z.string().nullable(),
      rationale: z.string(),
    }),
  ),
});

// ---------------------------------------------------------------------------
// Offline responder: realistic canned model output (live mode calls the real API)
// ---------------------------------------------------------------------------
const f = (file: string, line: number, severity: string, category: string, title: string, suggestion: string, confidence: number): Json => ({
  file, line, severity, category, title, suggestion, confidence,
});

const CANNED_REVIEW: Record<string, Json[]> = {
  "app/api/orders.py": [
    f("app/api/orders.py", 24, "critical", "security", "User input is interpolated into SQL with an f-string (SQL injection).",
      'Use a bound parameter: conn.execute("SELECT id, total_cents FROM orders WHERE email = :email", {"email": customer_email}).', 0.97),
    f("app/api/orders.py", 21, "high", "security", "New endpoint has no auth dependency, so anyone can read any customer's order history.",
      "Add user=Depends(require_user) and only return orders that belong to user.id.", 0.92),
    f("app/api/orders.py", 20, "low", "security", "Customer email in the URL path leaks PII into access logs.",
      "Identify the customer from the authenticated user instead of an email path parameter.", 0.55),
    f("app/api/orders.py", 24, "high", "security", "Unsanitised customer_email reaches the SQL query.", "Parameterise the query.", 0.81),
    f("app/api/orders.py", 99, "medium", "performance", "Query result is unbounded; add pagination.", "Add limit/offset parameters.", 0.7),
  ],
  "app/services/pricing.py": [
    f("app/services/pricing.py", 10, "high", "correctness", "Money is computed in floating point and truncated by int(): 1005 at 10% gives 904, not 905.",
      "Use integer math with half-up rounding: (price_cents * (100 - percent) + 50) // 100.", 0.9),
    f("app/services/pricing.py", 8, "medium", "correctness", "percent and price_cents are not validated; negative or >50 values yield wrong prices.",
      "Raise ValueError unless 0 <= percent <= MAX_DISCOUNT_PERCENT and price_cents >= 0.", 0.86),
    f("app/services/pricing.py", 14, "low", "maintainability", "MAX_DISCOUNT_PERCENT is defined after the function and never used.",
      "Move the constant above apply_discount and enforce it in the validation.", 0.66),
  ],
  "app/config.py": [
    f("app/config.py", 4, "high", "security", "DEBUG is hard-coded to True, exposing stack traces in production.",
      'Restore the environment-driven default: DEBUG = os.getenv("DEBUG", "false") == "true".', 0.93),
    f("app/config.py", 3, "medium", "performance", "A 300 s request timeout ties up workers during upstream incidents.",
      "Keep a short default (30 s) and move long-running work to a background job.", 0.72),
  ],
  "tests/test_pricing.py": [
    f("tests/test_pricing.py", 5, "medium", "tests", "Only the happy path is tested; rounding, boundaries (0, 50) and invalid inputs are not.",
      "Add cases for 1005 at 10% (905), percent 0 and 50, percent 51 and -5, and a negative price.", 0.84),
    f("tests/conftest.py", 1, "low", "tests", "Shared fixtures are missing.", "Add a conftest.py with a database fixture.", 0.62),
  ],
};

const COMMIT_ATTEMPT_1: CommitDraft = {
  type: "feat", scope: "Orders API",
  subject: "Added a new order history endpoint and discount support to the pricing service.",
  body: ["Adds an endpoint and a helper."], breaking: false,
  changelog: [{ section: "Added", text: "Order history endpoint." }],
};
const COMMIT_ATTEMPT_2: CommitDraft = {
  type: "feat", scope: "orders",
  subject: "add order history endpoint with discount pricing",
  body: [
    "- Add GET /orders/history/{customer_email} returning order totals.",
    "- Add apply_discount() to the pricing service.",
    "- Raise REQUEST_TIMEOUT to 300 s and set DEBUG in app/config.py.",
  ],
  breaking: false,
  changelog: [
    { section: "Added", text: "Order history endpoint `GET /orders/history/{customer_email}`." },
    { section: "Added", text: "`apply_discount()` helper for percentage discounts." },
    { section: "Changed", text: "Default request timeout raised from 30 s to 300 s." },
    { section: "Changed", text: "`DEBUG` is now always enabled." },
  ],
};

const tc = (name: string, price_cents: number, percent: number, expected: number | null, raises: string | null, rationale: string): GeneratedCase => ({
  name, price_cents, percent, expected, raises, rationale,
});

const CANNED_TESTS = {
  cases: [
    tc("test_no_discount_returns_same_price", 1999, 0, 1999, null, "boundary: 0% leaves price unchanged"),
    tc("test_ten_percent_rounds_half_up", 1005, 10, 905, null, "904.5 cents must round half-up to 905"),
    tc("test_fifteen_percent_typical", 1999, 15, 1699, null, "typical discount, 1699.15 rounds to 1699"),
    tc("test_max_discount_boundary", 1000, 50, 500, null, "boundary: maximum allowed discount"),
    tc("test_discount_above_max_raises", 1000, 51, null, "ValueError", "51% exceeds MAX_DISCOUNT_PERCENT"),
    tc("test_negative_percent_raises", 1000, -5, null, "ValueError", "negative discount would raise the price"),
    tc("test_negative_price_raises", -100, 10, null, "ValueError", "prices cannot be negative"),
    tc("test_thirty_three_percent_rounding", 999, 33, 670, null, "669.33 rounds to 670"), // wrong on purpose
    tc("test_zero_price", 0, 25, 0, null, "free items stay free"),
    tc("test_zero_price", 0, 10, 0, null, "duplicate name"),
    tc("test_half_price_or_error", 1000, 50, 500, "ValueError", "ambiguous: both outcomes set"),
  ],
};

function responder(req: MockRequest): string {
  const system = req.systemText;
  const text = req.lastUserText;
  if (system.includes("automated code reviewer")) {
    const m = /^File: (\S+)/m.exec(text);
    const file = m ? m[1]! : "";
    if (file === "app/api/orders.py" && !text.includes("+15,16")) return JSON.stringify({ findings: [] }); // import-only hunk is clean
    return JSON.stringify({ findings: CANNED_REVIEW[file] ?? [] });
  }
  if (system.includes("commit message")) return JSON.stringify(text.includes("Validation errors") ? COMMIT_ATTEMPT_2 : COMMIT_ATTEMPT_1);
  if (system.includes("test cases")) return JSON.stringify(CANNED_TESTS);
  return "{}";
}

// ---------------------------------------------------------------------------
// Cost ledger: every call is attributed to a workflow step
// ---------------------------------------------------------------------------
interface CallRecord {
  step: string;
  model: string;
  usage: Anthropic.Usage;
}
const LEDGER: CallRecord[] = [];
const record = (step: string, msg: Anthropic.Message): void => {
  LEDGER.push({ step, model: msg.model, usage: msg.usage });
};
const cw = (u: Anthropic.Usage): number => u.cache_creation_input_tokens ?? 0;
const cr = (u: Anthropic.Usage): number => u.cache_read_input_tokens ?? 0;
const money = (n: number): string => n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------
function stepParse(): FileDiff[] {
  section("1. Parse the unified diff (data/sample.diff)");
  const [files, warnings] = parseUnifiedDiff(read("sample.diff"));
  for (const fd of files) {
    console.log(`  ${fd.path.padEnd(26)} ${fd.status.padEnd(9)} +${String(additions(fd)).padEnd(3)} -${String(deletions(fd)).padEnd(3)} hunks=${fd.hunks.length}`);
  }
  const hunks = files.reduce((n, fd) => n + fd.hunks.length, 0);
  console.log(`  ${files.length} files, ${hunks} hunks; hunk header counts verified: ${warnings.length ? "NO" : "yes"}`);
  for (const w of warnings) console.log(`  WARNING: ${w}`);
  return files;
}

function stepChunk(files: FileDiff[]): Chunk[] {
  section(`2. Chunk for review (budget ~${MAX_CHUNK_TOKENS} tokens per chunk)`);
  const [chunks, skipped] = buildChunks(files, MAX_CHUNK_TOKENS);
  for (const c of chunks) {
    const lines = [...c.validLines];
    console.log(`  ${c.id}  ${c.file.padEnd(26)} hunks=${c.hunkHeaders.length} ~${String(c.estTokens).padStart(3)} tokens  lines ${Math.min(...lines)}-${Math.max(...lines)}`);
  }
  for (const [p, reason] of skipped) console.log(`  skip ${p.padEnd(26)} ${reason}`);
  console.log("  Chunks carry NEW-file line numbers so the model can cite lines we can verify.");
  return chunks;
}

async function stepReview(client: Anthropic, chunks: Chunk[]): Promise<ReviewComment[]> {
  section("3. AI review per chunk (Sonnet 5.5, structured output, cached rubric)");
  const guidelines = read("review_guidelines.md");
  const raw: ReviewComment[] = [];
  for (const c of chunks) {
    const msg = await client.messages.parse({
      model: MODEL_SONNET,
      max_tokens: 4000,
      thinking: { type: "adaptive" },
      output_config: { effort: "medium", format: zodOutputFormat(ReviewResult) },
      // Stable rubric first + cache_control: chunks 2..N read it from cache at ~10% of the input price.
      system: [{ type: "text", text: guidelines, cache_control: { type: "ephemeral" } }],
      messages: [{ role: "user", content: `Review this chunk of PR #482.\n\n${c.text}` }],
    });
    record("review", msg);
    const found = msg.parsed_output?.findings ?? [];
    raw.push(...found.map((x) => ({ ...x, chunkId: c.id })));
    const us = msg.usage;
    console.log(`  ${c.id} ${c.file.padEnd(26)} raw findings=${found.length}  in=${us.input_tokens} out=${us.output_tokens} cache_write=${cw(us)} cache_read=${cr(us)}`);
  }
  return raw;
}

function stepPostprocess(raw: ReviewComment[], chunks: Chunk[]): ReviewComment[] {
  section(`4. Ground, filter (confidence >= ${MIN_CONFIDENCE}), dedupe -> PR comments`);
  const [kept, dropped] = postprocess(raw, chunks, MIN_CONFIDENCE);
  for (const [x, reason] of dropped) console.log(`  drop ${`${x.file}:${x.line}`.padEnd(28)} ${x.severity.padEnd(8)} ${reason}`);
  console.log();
  for (const x of kept) {
    console.log(`  keep ${`${x.file}:${x.line}`.padEnd(28)} ${x.severity.padEnd(8)} ${x.category.padEnd(15)} conf=${x.confidence.toFixed(2)}  ${x.title.slice(0, 46)}`);
  }
  const counts = severityCounts(kept);
  console.log("\n  findings by severity: " + Object.entries(counts).map(([s, n]) => `${s}=${n}`).join(", "));
  const blocking = kept.filter((x) => (x.severity === "critical" || x.severity === "high") && x.confidence >= 0.8);
  console.log(`  advisory verdict: ${blocking.length ? "changes suggested" : "looks good"} (${blocking.length} critical/high) - ` +
    "posted as COMMENT; branch protection still requires a human approval");
  mkdirSync(OUT, { recursive: true });
  writeFileSync(path.join(OUT, "pr_review.md"), renderPrMarkdown(kept, dropped.length), "utf8");
  writeFileSync(path.join(OUT, "review_payload.json"), JSON.stringify(reviewPayload(kept), null, 2), "utf8");
  console.log("  wrote labs_output/pr_review.md and labs_output/review_payload.json");
  return kept;
}

async function stepCommit(client: Anthropic, files: FileDiff[]): Promise<void> {
  section("5. Commit message + changelog (Haiku 5.5, validate-and-repair loop)");
  const stat = files.map((fd) => `${fd.path} | +${additions(fd)} -${deletions(fd)} (${fd.status})`).join("\n");
  for (const line of stat.split("\n")) console.log(`  ${line}`);
  const diffText = read("sample.diff");
  const system =
    "You write git commit messages in Conventional Commits format and Keep a Changelog entries. " +
    "Subject: imperative mood, lowercase, no trailing period, header <= 72 chars. Body lines <= 72 chars.";
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: `Diffstat:\n${stat}\n\nDiff:\n${diffText}\n\nWrite the commit.` }];
  let draft: CommitDraft | null = null;
  for (const attempt of [1, 2, 3]) {
    const msg = await client.messages.parse({
      model: MODEL_HAIKU,
      max_tokens: 1000,
      thinking: { type: "adaptive" },
      output_config: { effort: "low", format: zodOutputFormat(CommitDraftSchema) },
      system,
      messages,
    });
    record("commit-message", msg);
    draft = msg.parsed_output ?? null;
    const errors = draft ? validateCommit(draft) : ["unparseable output"];
    console.log(`\n  attempt ${attempt}: ${draft ? commitHeader(draft) : "(none)"}`);
    if (!errors.length) {
      console.log("    valid Conventional Commit");
      break;
    }
    for (const e of errors) console.log(`    x ${e}`);
    // Repair turn: show the model its own output plus the validator's errors (no prefill needed).
    messages.push(
      { role: "assistant", content: textOf(msg) },
      { role: "user", content: "Validation errors:\n" + errors.map((e) => `- ${e}`).join("\n") + "\nReturn a corrected JSON object." },
    );
  }
  if (!draft) return;
  const commitMsg = renderCommitMessage(draft);
  const changelog = renderChangelog(draft.changelog);
  console.log("\n  --- commit message ---");
  for (const line of splitLines(commitMsg)) console.log(`  ${line}`);
  console.log("  --- CHANGELOG.md fragment ---");
  for (const line of splitLines(changelog)) console.log(`  ${line}`);
  writeFileSync(path.join(OUT, "COMMIT_MSG.txt"), commitMsg, "utf8");
  writeFileSync(path.join(OUT, "CHANGELOG.fragment.md"), changelog, "utf8");
}

async function stepTests(client: Anthropic): Promise<void> {
  section("6. Test-case generation for apply_discount(), validated by code");
  const spec = read("apply_discount_spec.md");
  const msg = await client.messages.parse({
    model: MODEL_SONNET,
    max_tokens: 4000,
    thinking: { type: "adaptive" },
    output_config: { effort: "medium", format: zodOutputFormat(GeneratedSuite) },
    system: "You design unit test cases from a specification. Return test cases as data, one behaviour per case.",
    messages: [{
      role: "user",
      content: `Specification:\n${spec}\nGenerate 8-12 test cases covering typical values, boundaries, fractional-cent rounding and invalid inputs.`,
    }],
  });
  record("test-generation", msg);
  const cases: GeneratedCase[] = msg.parsed_output?.cases ?? [];
  const [valid, rejected] = validateCases(cases);
  console.log(`  generated ${cases.length} cases; structurally valid ${valid.length}; rejected ${rejected.length}`);
  for (const [c, reason] of rejected) console.log(`    reject ${c.name}: ${reason}`);
  console.log("  coverage: " + Object.entries(coverage(valid)).map(([k, v]) => `${k}=${v ? "yes" : "NO"}`).join(", "));
  console.log("\n  execute each case against the executable spec AND the code in the PR:");
  const suspect = new Set<string>();
  const tallies = new Map<string, number>();
  for (const c of valid) {
    const [specOk] = runCase(applyDiscountReference, c);
    const [codeOk, codeObs] = runCase(applyDiscountFromDiff, c);
    const verdict = triage(specOk, codeOk);
    const key = verdict.split(" - ")[0]!;
    tallies.set(key, (tallies.get(key) ?? 0) + 1);
    if (!specOk) suspect.add(c.name);
    const want = c.raises ? `raises ${c.raises}` : `== ${c.expected}`;
    console.log(`    ${c.name.padEnd(36)} want ${want.padEnd(18)} spec:${specOk ? "ok " : "BAD"} code:${codeOk ? "ok " : "BAD"} (${codeObs}) -> ${verdict}`);
  }
  console.log("  summary: " + [...tallies.entries()].map(([k, n]) => `${k}=${n}`).join(", "));
  writeFileSync(path.join(OUT, "test_pricing_generated.py"), renderPytest(valid, suspect), "utf8");
  console.log("  wrote labs_output/test_pricing_generated.py (suspect cases commented out for human review)");
}

function stepCi(): void {
  section("7. Running it in CI (documented - requires Claude Code / GitHub, not executed)");
  for (const name of ["claude-pr-review.yml", "risky-pr-review.yml"]) {
    const findings = lintWorkflow(read("ci", name));
    console.log(`  lint data/ci/${name}: ${findings.length} findings`);
    for (const x of findings) console.log(`    [${x.severity.padEnd(5)}] ${x.code} ${x.message}`);
  }
  console.log("\n  headless mode as used in the workflow:");
  console.log('    claude -p "Summarise pr.diff for release notes" --output-format json --max-turns 3 --allowedTools "Read"');
  const result = JSON.parse(read("ci", "headless_result.json")) as Json;
  const [status, notes] = gateHeadless(result, 0.5);
  console.log(`  gate on data/ci/headless_result.json -> ${status}`);
  for (const n of notes) console.log(`    ${n}`);
  console.log("\n  guard rails for AI in CI:");
  for (const line of [
    "secrets: ANTHROPIC_API_KEY only via ${{ secrets.* }}, scoped to the step that needs it",
    "least privilege: permissions contents:read + pull-requests:write; --allowedTools read-only",
    "never pull_request_target + PR-head checkout; never --dangerously-skip-permissions with write tokens",
    "cost: paths/draft filters, concurrency cancel, --max-turns, Sonnet/Haiku per step, cached rubric",
    "humans merge: bot posts COMMENT reviews; branch protection requires an approving human review",
  ]) console.log(`    - ${line}`);
}

function stepEconomics(kept: ReviewComment[]): void {
  section("8. PR economics and outcome metrics");
  console.log(`  ${"step".padEnd(16)} ${"model".padEnd(18)} ${"calls".padStart(5)} ${"in".padStart(6)} ${"out".padStart(5)} ${"c_write".padStart(7)} ${"c_read".padStart(6)} ${"cost".padStart(10)}`);
  const steps = [...new Set(LEDGER.map((r) => r.step))];
  let totalCost = 0;
  let totalCalls = 0;
  const sum = (recs: CallRecord[], fn: (u: Anthropic.Usage) => number) => recs.reduce((n, r) => n + fn(r.usage), 0);
  for (const step of steps) {
    const recs = LEDGER.filter((r) => r.step === step);
    const cost = recs.reduce((n, r) => n + costUsd(r.model, r.usage), 0);
    totalCost += cost;
    totalCalls += recs.length;
    console.log(
      `  ${step.padEnd(16)} ${recs[0]!.model.padEnd(18)} ${String(recs.length).padStart(5)} ${String(sum(recs, (u) => u.input_tokens)).padStart(6)} ` +
        `${String(sum(recs, (u) => u.output_tokens)).padStart(5)} ${String(sum(recs, cw)).padStart(7)} ${String(sum(recs, cr)).padStart(6)} $${cost.toFixed(5).padStart(9)}`,
    );
  }
  console.log(`  ${"TOTAL per PR".padEnd(16)} ${"".padEnd(18)} ${String(totalCalls).padStart(5)} ${"".padStart(6)} ${"".padStart(5)} ${"".padStart(7)} ${"".padStart(6)} $${totalCost.toFixed(5).padStart(9)}`);
  console.log(`  projection: ${PRS_PER_MONTH} PRs/month -> $${money(totalCost * PRS_PER_MONTH)}/month (offline estimates are illustrative)`);

  const review = LEDGER.filter((r) => r.step === "review");
  const p = PRICING[MODEL_SONNET]!;
  const noCache = review.reduce((n, r) => n + (r.usage.input_tokens + cw(r.usage) + cr(r.usage)) * p.input + r.usage.output_tokens * p.output, 0) / 1_000_000;
  const withCache = review.reduce((n, r) => n + costUsd(r.model, r.usage), 0);
  console.log(`\n  review step: with cached rubric $${withCache.toFixed(5)} vs no caching $${noCache.toFixed(5)}`);
  for (const model of [MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS]) {
    const alt = review.reduce((n, r) => n + costUsd(model, r.usage), 0);
    console.log(`    same review tokens priced on ${model.padEnd(18)} $${alt.toFixed(5)}`);
  }
  const counts = severityCounts(kept);
  console.log("\n  outcome: " + Object.entries(counts).map(([s, n]) => `${s}=${n}`).join(", ") +
    ` | total=${kept.length} | cost per kept finding $${(totalCost / Math.max(1, kept.length)).toFixed(5)}`);
  console.log("  Track over time: acceptance rate of AI comments, false-positive rate, time-to-first-review.");
}

banner("D7.2 - AI-Assisted Developer Workflows");
const client = getClient(responder);
const files = stepParse();
const chunks = stepChunk(files);
const raw = await stepReview(client, chunks);
const kept = stepPostprocess(raw, chunks);
await stepCommit(client, files);
await stepTests(client);
stepCi();
stepEconomics(kept);
