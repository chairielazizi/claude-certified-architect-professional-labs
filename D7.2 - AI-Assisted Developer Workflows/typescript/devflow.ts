/**
 * Deterministic building blocks for AI-assisted developer workflows (D7.2) - twin of python/devflow.py.
 *
 * Everything here is plain code - no model calls - because this is where reliability comes from:
 *   - unified-diff parser + review chunker (what the model sees)
 *   - review post-processing: validate line numbers, confidence filter, dedupe (what humans see)
 *   - PR comment rendering (markdown + a GitHub review payload)
 *   - Conventional Commits validator + Keep a Changelog renderer
 *   - generated-test validation, coverage check, and execution against code and spec
 *   - GitHub Actions workflow linter + headless `claude -p --output-format json` result gate
 */

export type Json = Record<string, any>;

// ---------------------------------------------------------------------------
// 1. Unified diff parsing
// ---------------------------------------------------------------------------
const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$/;

export interface DiffLine {
  kind: "+" | "-" | " ";
  text: string;
  oldNo: number | null;
  newNo: number | null;
}

export interface Hunk {
  header: string;
  oldStart: number;
  oldCount: number;
  newStart: number;
  newCount: number;
  lines: DiffLine[];
}

export interface FileDiff {
  path: string;
  status: "modified" | "added" | "deleted" | "renamed" | "binary";
  hunks: Hunk[];
}

export const additions = (f: FileDiff): number => f.hunks.reduce((n, h) => n + h.lines.filter((l) => l.kind === "+").length, 0);
export const deletions = (f: FileDiff): number => f.hunks.reduce((n, h) => n + h.lines.filter((l) => l.kind === "-").length, 0);

/** Like Python's str.splitlines(): handles CRLF and drops the phantom line after a final newline. */
export function splitLines(text: string): string[] {
  const lines = text.split(/\r?\n/);
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

/**
 * Parse `git diff` output. Hunk bodies are consumed by the header COUNTS, not by guessing:
 * a removed SQL comment ("--- old") or an added "+++" line would otherwise be misread as a file header.
 */
export function parseUnifiedDiff(text: string): [FileDiff[], string[]] {
  const files: FileDiff[] = [];
  const warnings: string[] = [];
  let cur: FileDiff | null = null;
  let hunk: Hunk | null = null;
  let oldLeft = 0;
  let newLeft = 0;
  let oldNo = 0;
  let newNo = 0;

  for (const raw of splitLines(text)) {
    if (hunk !== null && (oldLeft > 0 || newLeft > 0)) {
      const kind = raw ? raw[0]! : " "; // some tools strip the space of empty context lines
      if (kind === "\\") continue; // "\ No newline at end of file"
      if (kind === "+" || kind === "-" || kind === " ") {
        const body = raw.slice(1);
        if (kind === " ") {
          hunk.lines.push({ kind: " ", text: body, oldNo, newNo });
          oldNo++, newNo++, oldLeft--, newLeft--;
        } else if (kind === "-") {
          hunk.lines.push({ kind: "-", text: body, oldNo, newNo: null });
          oldNo++, oldLeft--;
        } else {
          hunk.lines.push({ kind: "+", text: body, oldNo: null, newNo });
          newNo++, newLeft--;
        }
        continue;
      }
      warnings.push(`${cur ? cur.path : "?"}: hunk ended early (${hunk.header})`);
      hunk = null;
    }

    let m: RegExpExecArray | null;
    if (raw.startsWith("diff --git ")) {
      const g = /^diff --git a\/(.+) b\/(.+)$/.exec(raw);
      cur = { path: g ? g[2]! : raw.slice(11), status: "modified", hunks: [] };
      files.push(cur);
      hunk = null;
    } else if (cur === null) {
      continue;
    } else if (raw.startsWith("new file mode")) {
      cur.status = "added";
    } else if (raw.startsWith("deleted file mode")) {
      cur.status = "deleted";
    } else if (raw.startsWith("rename to ")) {
      cur.status = "renamed";
      cur.path = raw.slice("rename to ".length);
    } else if (raw.startsWith("Binary files ")) {
      cur.status = "binary";
    } else if (raw.startsWith("+++ ") && raw.slice(4) !== "/dev/null") {
      const p = raw.slice(4);
      cur.path = p.startsWith("b/") ? p.slice(2) : p;
    } else if ((m = HUNK_RE.exec(raw))) {
      const oldStart = Number(m[1]);
      const newStart = Number(m[3]);
      const oldCount = m[2] !== undefined ? Number(m[2]) : 1;
      const newCount = m[4] !== undefined ? Number(m[4]) : 1;
      hunk = { header: raw, oldStart, oldCount, newStart, newCount, lines: [] };
      cur.hunks.push(hunk);
      [oldNo, newNo, oldLeft, newLeft] = [oldStart, newStart, oldCount, newCount];
    }
  }
  if (hunk !== null && (oldLeft > 0 || newLeft > 0)) warnings.push(`${cur ? cur.path : "?"}: diff truncated inside ${hunk.header}`);
  return [files, warnings];
}

// ---------------------------------------------------------------------------
// 2. Chunking for review
// ---------------------------------------------------------------------------
const SKIP_PATTERNS = [/(^|\/)package-lock\.json$/, /(^|\/)yarn\.lock$/, /\.lock$/, /(^|\/)dist\//, /\.min\.js$/];

export const estTokens = (text: string): number => (text ? Math.max(1, Math.floor(text.length / 4)) : 0);

/** Prefix every line with its NEW-file line number so the model can cite real lines. */
export function renderHunk(h: Hunk): string {
  const out = [h.header];
  for (const ln of h.lines) {
    const num = ln.newNo !== null ? String(ln.newNo).padStart(4) : "    ";
    out.push(`${num} ${ln.kind}${ln.text}`);
  }
  return out.join("\n");
}

export interface Chunk {
  id: string;
  file: string;
  status: string;
  hunkHeaders: string[];
  text: string;
  estTokens: number;
  validLines: Set<number>;
}

export function buildChunks(files: FileDiff[], maxTokens: number): [Chunk[], [string, string][]] {
  const chunks: Chunk[] = [];
  const skipped: [string, string][] = [];
  for (const f of files) {
    if (f.status === "binary") {
      skipped.push([f.path, "binary file"]);
      continue;
    }
    if (SKIP_PATTERNS.some((p) => p.test(f.path))) {
      skipped.push([f.path, "lockfile / generated - not worth model tokens"]);
      continue;
    }
    if (f.status === "deleted") {
      skipped.push([f.path, "deleted file - nothing to comment on"]);
      continue;
    }
    let group: Hunk[] = [];
    const flush = (): void => {
      if (!group.length) return;
      const body = group.map(renderHunk).join("\n");
      const text = `File: ${f.path} (${f.status})\n${body}`;
      const lines = new Set<number>();
      for (const h of group) for (const ln of h.lines) if (ln.newNo !== null) lines.add(ln.newNo);
      chunks.push({ id: `C${chunks.length + 1}`, file: f.path, status: f.status, hunkHeaders: group.map((h) => h.header), text, estTokens: estTokens(text), validLines: lines });
      group = [];
    };
    for (const h of f.hunks) {
      const candidate = [...group, h].map(renderHunk).join("\n");
      if (group.length && estTokens(candidate) > maxTokens) flush();
      group.push(h);
    }
    flush();
  }
  return [chunks, skipped];
}

// ---------------------------------------------------------------------------
// 3. Review post-processing
// ---------------------------------------------------------------------------
export const SEVERITY_RANK: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3, nit: 4 };

export interface ReviewComment {
  file: string;
  line: number;
  severity: string;
  category: string;
  title: string;
  suggestion: string;
  confidence: number;
  chunkId: string;
}

const cmp = (a: string | number, b: string | number): number => (a < b ? -1 : a > b ? 1 : 0);

/** Never post model output verbatim: ground it in the diff, filter noise, dedupe. */
export function postprocess(raw: ReviewComment[], chunks: Chunk[], minConfidence: number): [ReviewComment[], [ReviewComment, string][]] {
  const valid = new Map<string, Set<number>>();
  for (const c of chunks) {
    const s = valid.get(c.file) ?? new Set<number>();
    for (const n of c.validLines) s.add(n);
    valid.set(c.file, s);
  }
  const kept = new Map<string, ReviewComment>();
  const dropped: [ReviewComment, string][] = [];
  for (const f of raw) {
    if (!valid.has(f.file)) dropped.push([f, "file not in diff (hallucinated)"]);
    else if (!valid.get(f.file)!.has(f.line)) dropped.push([f, `line ${f.line} not in diff (hallucinated)`]);
    else if (f.confidence < minConfidence) dropped.push([f, `confidence ${f.confidence.toFixed(2)} < ${minConfidence.toFixed(2)}`]);
    else {
      const key = `${f.file}|${f.line}|${f.category}`;
      const prev = kept.get(key);
      if (!prev) kept.set(key, f);
      else if (f.confidence > prev.confidence) {
        dropped.push([prev, "duplicate (lower confidence)"]);
        kept.set(key, f);
      } else dropped.push([f, "duplicate (lower confidence)"]);
    }
  }
  const ordered = [...kept.values()].sort(
    (a, b) => SEVERITY_RANK[a.severity]! - SEVERITY_RANK[b.severity]! || cmp(a.file, b.file) || a.line - b.line,
  );
  return [ordered, dropped];
}

export function severityCounts(findings: ReviewComment[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const s of Object.keys(SEVERITY_RANK)) out[s] = findings.filter((f) => f.severity === s).length;
  return out;
}

export function renderPrMarkdown(findings: ReviewComment[], droppedCount: number): string {
  const counts = severityCounts(findings);
  const out = [`## AI review (advisory) - ${findings.length} findings`, "", "| Severity | Count |", "|---|---|"];
  for (const [s, n] of Object.entries(counts)) out.push(`| ${s} | ${n} |`);
  out.push("", `_${droppedCount} raw findings were filtered (ungrounded, low confidence, or duplicates)._`,
    "> These comments are suggestions. A human reviewer approves or rejects this PR.", "");
  let current: string | null = null;
  for (const f of [...findings].sort((a, b) => cmp(a.file, b.file) || a.line - b.line)) {
    if (f.file !== current) {
      out.push(`### \`${f.file}\``, "");
      current = f.file;
    }
    out.push(`- **L${f.line} [${f.severity} | ${f.category}]** ${f.title}`);
    out.push(`  - Suggestion: ${f.suggestion}`);
    out.push(`  - Confidence: ${f.confidence.toFixed(2)}`);
  }
  return out.join("\n") + "\n";
}

/** Shape of GitHub's 'create a review' request. event=COMMENT: the bot never approves or blocks. */
export function reviewPayload(findings: ReviewComment[]): Json {
  const counts = severityCounts(findings);
  const summary = Object.entries(counts).filter(([, n]) => n).map(([s, n]) => `${n} ${s}`).join(", ");
  return {
    event: "COMMENT",
    body: `AI review (advisory): ${summary}. A human makes the merge decision.`,
    comments: findings.map((f) => ({
      path: f.file, line: f.line, side: "RIGHT", body: `**[${f.severity} | ${f.category}]** ${f.title}\n\nSuggestion: ${f.suggestion}`,
    })),
  };
}

// ---------------------------------------------------------------------------
// 4. Conventional Commits + changelog
// ---------------------------------------------------------------------------
export const CC_TYPES = ["feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert"] as const;
export const CHANGELOG_SECTIONS = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"] as const;
const NON_IMPERATIVE = new Set(["adds", "fixes", "updates", "removes", "changes", "implements"]);

export interface CommitDraft {
  type: string;
  scope: string;
  subject: string;
  body: string[];
  breaking: boolean;
  changelog: { section: string; text: string }[];
}

export function commitHeader(d: CommitDraft): string {
  const scope = d.scope ? `(${d.scope})` : "";
  return `${d.type}${scope}${d.breaking ? "!" : ""}: ${d.subject}`;
}

export function validateCommit(d: CommitDraft): string[] {
  const errors: string[] = [];
  if (!(CC_TYPES as readonly string[]).includes(d.type)) errors.push(`type "${d.type}" is not a Conventional Commits type`);
  const scope = d.scope ?? "";
  if (scope && !/^[a-z0-9-]+$/.test(scope)) errors.push(`scope "${scope}" must be lowercase kebab-case`);
  const subject = d.subject ?? "";
  const words = subject.split(/\s+/).filter(Boolean);
  const first = words.length ? words[0]!.toLowerCase() : "";
  if (!subject) errors.push("subject is empty");
  else if (subject[0] !== subject[0]!.toLowerCase()) errors.push("subject must start with a lowercase letter");
  if (first && (first.endsWith("ed") || first.endsWith("ing") || NON_IMPERATIVE.has(first))) {
    errors.push(`subject must use the imperative mood ("${words[0]}" -> e.g. "add")`);
  }
  if (subject.endsWith(".")) errors.push("subject must not end with a period");
  const header = commitHeader(d);
  if (header.length > 72) errors.push(`header is ${header.length} chars (max 72)`);
  (d.body ?? []).forEach((line, i) => {
    if (line.length > 72) errors.push(`body line ${i + 1} is ${line.length} chars (max 72)`);
  });
  if (!d.changelog?.length) errors.push("at least one changelog entry is required");
  return errors;
}

export function renderCommitMessage(d: CommitDraft): string {
  const parts = [commitHeader(d)];
  if (d.body?.length) parts.push("", ...d.body);
  if (d.breaking) parts.push("", "BREAKING CHANGE: see changelog");
  return parts.join("\n") + "\n";
}

export function renderChangelog(entries: { section: string; text: string }[], version = "Unreleased"): string {
  const out = [`## [${version}]`];
  for (const sec of CHANGELOG_SECTIONS) {
    const items = entries.filter((e) => e.section === sec).map((e) => e.text);
    if (items.length) out.push("", `### ${sec}`, ...items.map((t) => `- ${t}`));
  }
  return out.join("\n") + "\n";
}

// ---------------------------------------------------------------------------
// 5. Generated test cases: validate, measure coverage, execute
// ---------------------------------------------------------------------------
export const MAX_DISCOUNT_PERCENT = 50;

export interface GeneratedCase {
  name: string;
  price_cents: number;
  percent: number;
  expected: number | null;
  raises: string | null;
  rationale: string;
}

class ValueError extends Error {}

/** Faithful port of the function in sample.diff (float math, truncation, no validation). */
export function applyDiscountFromDiff(priceCents: number, percent: number): number {
  return Math.trunc(priceCents * (1 - percent / 100));
}

/** Executable spec (data/apply_discount_spec.md): half-up rounding, validated inputs. */
export function applyDiscountReference(priceCents: number, percent: number): number {
  if (!(percent >= 0 && percent <= MAX_DISCOUNT_PERCENT)) throw new ValueError("percent out of range");
  if (priceCents < 0) throw new ValueError("negative price");
  return Math.floor((priceCents * (100 - percent) + 50) / 100);
}

/** Structured outputs guarantee the JSON *shape*; these are the *semantic* rules a schema can't express. */
export function validateCases(cases: GeneratedCase[]): [GeneratedCase[], [GeneratedCase, string][]] {
  const valid: GeneratedCase[] = [];
  const rejected: [GeneratedCase, string][] = [];
  const names = new Set<string>();
  for (const c of cases) {
    const hasExpected = c.expected !== null && c.expected !== undefined;
    const hasRaises = c.raises !== null && c.raises !== undefined;
    if (!/^test_[a-z0-9_]+$/.test(c.name)) rejected.push([c, "name must match test_[a-z0-9_]+"]);
    else if (names.has(c.name)) rejected.push([c, "duplicate test name"]);
    else if (hasExpected === hasRaises) rejected.push([c, "exactly one of expected / raises must be set"]);
    else if (hasRaises && c.raises !== "ValueError") rejected.push([c, `unexpected exception type ${c.raises}`]);
    else {
      names.add(c.name);
      valid.push(c);
    }
  }
  return [valid, rejected];
}

export function coverage(cases: GeneratedCase[]): Record<string, boolean> {
  const rounding = (c: GeneratedCase) => c.expected !== null && (c.price_cents * (100 - c.percent)) % 100 !== 0;
  return {
    "typical value": cases.some((c) => c.expected !== null && c.percent > 0 && c.percent < MAX_DISCOUNT_PERCENT),
    "boundary percent=0": cases.some((c) => c.percent === 0),
    "boundary percent=max": cases.some((c) => c.percent === MAX_DISCOUNT_PERCENT),
    "fractional-cent rounding": cases.some(rounding),
    "invalid input raises": cases.some((c) => Boolean(c.raises)),
  };
}

export function runCase(fn: (p: number, pct: number) => number, c: GeneratedCase): [boolean, string] {
  let got: number;
  try {
    got = fn(c.price_cents, c.percent);
  } catch (err) {
    if (err instanceof ValueError) return [c.raises === "ValueError", "raised ValueError"];
    throw err;
  }
  if (c.raises) return [false, `returned ${got}`];
  return [got === c.expected, `returned ${got}`];
}

export function triage(specOk: boolean, codeOk: boolean): string {
  if (!specOk) return "SUSPECT TEST - contradicts the spec, needs human review";
  if (!codeOk) return "BUG FOUND in PR code";
  return "pass (regression guard)";
}

export function renderPytest(cases: GeneratedCase[], suspect: Set<string>): string {
  const out = ['"""Generated by Claude (D7.2 lab), validated by code, reviewed by a human before commit."""',
    "import pytest", "", "from app.services.pricing import apply_discount", ""];
  for (const c of cases) {
    let prefix = "";
    if (suspect.has(c.name)) {
      out.push("# REVIEW: this expectation contradicts the spec - excluded until a human decides");
      prefix = "# ";
    }
    out.push(`${prefix}def ${c.name}():`);
    if (c.raises) {
      out.push(`${prefix}    with pytest.raises(${c.raises}):`);
      out.push(`${prefix}        apply_discount(${c.price_cents}, ${c.percent})`);
    } else {
      out.push(`${prefix}    assert apply_discount(${c.price_cents}, ${c.percent}) == ${c.expected}`);
    }
    out.push("");
  }
  return out.join("\n");
}

// ---------------------------------------------------------------------------
// 6. CI: workflow linter + headless result gate
// ---------------------------------------------------------------------------
const SECRET_RE = /sk-ant-[A-Za-z0-9_\-]{8,}|gh[pousr]_[A-Za-z0-9_]{10,}/;

export interface WorkflowFinding {
  severity: "ERROR" | "WARN" | "INFO";
  code: string;
  message: string;
}

export function lintWorkflow(rawText: string): WorkflowFinding[] {
  const text = rawText.replace(/\r\n/g, "\n");
  const out: WorkflowFinding[] = [];
  const add = (severity: WorkflowFinding["severity"], code: string, message: string) => out.push({ severity, code, message });
  const has = (rx: RegExp) => rx.test(text);

  if (has(/^\s*pull_request_target\s*:/m) && has(/pull_request\.head\.(sha|ref)/)) {
    add("ERROR", "W001", "pull_request_target + checkout of the PR head runs untrusted code with secrets");
  }
  if (SECRET_RE.test(text)) add("ERROR", "W002", "literal API key/token in the workflow; use ${{ secrets.NAME }}");
  const perms = /^permissions:[ \t]*(\S.*)?$/m.exec(text);
  if (!perms) add("WARN", "W003", "no top-level permissions block; the token gets repository defaults");
  else if (perms[1] && perms[1].includes("write-all")) add("ERROR", "W003", "permissions: write-all - grant only contents: read, pull-requests: write");
  if (!has(/timeout-minutes:/)) add("WARN", "W004", "no timeout-minutes; a stuck agent burns runner time and tokens");
  if (!has(/^concurrency:/m)) add("WARN", "W005", "no concurrency group; superseded pushes keep paying for stale reviews");
  for (const m of text.matchAll(/uses:\s*([\w.-]+\/[\w.-]+)@([\w.-]+)/g)) {
    if (["main", "master", "latest", "HEAD"].includes(m[2]!)) {
      add("WARN", "W006", `${m[1]}@${m[2]} is a moving target; pin a release tag or commit SHA`);
    }
  }
  if (has(/--dangerously-skip-permissions|bypassPermissions/)) add("ERROR", "W007", "permission checks disabled for an agent that can push code");
  if (has(/\bclaude\s+-p\b|claude-code-action/) && !has(/--max-turns/)) add("WARN", "W008", "no --max-turns cap on the agent (runaway cost)");
  const topEnv = /^env:[ \t]*\n((?:[ \t]+.*\n?)+)/m.exec(text);
  if (topEnv && topEnv[1]!.includes("secrets.")) {
    add("WARN", "W009", "secrets in workflow-level env are exposed to every step; scope them to one step");
  }
  if (!has(/draft\s*==\s*false/)) add("INFO", "W010", "no draft filter; draft PRs trigger paid reviews");
  if (!has(/^\s*paths:/m)) add("INFO", "W011", "no paths filter; docs-only changes trigger reviews");
  return out;
}

/** Decide the CI step outcome from `claude -p ... --output-format json` output. */
export function gateHeadless(result: Json, budgetUsd: number): [string, string[]] {
  const cost = Number(result.total_cost_usd ?? 0);
  const notes = [
    `subtype=${result.subtype} is_error=${JSON.stringify(result.is_error)} turns=${result.num_turns} ` +
      `duration=${result.duration_ms}ms cost=$${cost.toFixed(4)}`,
  ];
  if (result.is_error || result.subtype !== "success") return ["FAIL", [...notes, "run did not succeed - fail the job, do not post partial output"]];
  if (cost > budgetUsd) return ["WARN", [...notes, `over the $${budgetUsd.toFixed(2)} per-PR budget - alert the owning team`]];
  return ["PASS", [...notes, "post `result` as a PR comment; keep session_id for audit"]];
}
