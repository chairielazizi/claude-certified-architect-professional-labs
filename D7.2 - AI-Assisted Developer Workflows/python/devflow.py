"""
Deterministic building blocks for AI-assisted developer workflows (D7.2).

Everything here is plain code - no model calls - because this is where reliability comes from:
  * unified-diff parser + review chunker (what the model sees)
  * review post-processing: validate line numbers, confidence filter, dedupe (what humans see)
  * PR comment rendering (markdown + a GitHub review payload)
  * Conventional Commits validator + Keep a Changelog renderer
  * generated-test validation, coverage check, and execution against code and spec
  * GitHub Actions workflow linter + headless `claude -p --output-format json` result gate
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 1. Unified diff parsing
# ---------------------------------------------------------------------------
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


@dataclass
class DiffLine:
    kind: str                 # "+", "-" or " "
    text: str
    old_no: int | None
    new_no: int | None


@dataclass
class Hunk:
    header: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class FileDiff:
    path: str
    status: str = "modified"  # modified | added | deleted | renamed | binary
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def additions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "+")

    @property
    def deletions(self) -> int:
        return sum(1 for h in self.hunks for ln in h.lines if ln.kind == "-")


def parse_unified_diff(text: str) -> tuple[list[FileDiff], list[str]]:
    """Parse `git diff` output. Hunk bodies are consumed by the header COUNTS, not by guessing:
    a removed SQL comment ("--- old") or an added "+++" line would otherwise be misread as a file header."""
    files: list[FileDiff] = []
    warnings: list[str] = []
    cur: FileDiff | None = None
    hunk: Hunk | None = None
    old_left = new_left = old_no = new_no = 0

    for raw in text.splitlines():
        if hunk is not None and (old_left > 0 or new_left > 0):
            kind = raw[:1] if raw else " "      # some tools strip the space of empty context lines
            if kind == "\\":                     # "\ No newline at end of file"
                continue
            if kind in ("+", "-", " "):
                body = raw[1:]
                if kind == " ":
                    hunk.lines.append(DiffLine(" ", body, old_no, new_no))
                    old_no, new_no, old_left, new_left = old_no + 1, new_no + 1, old_left - 1, new_left - 1
                elif kind == "-":
                    hunk.lines.append(DiffLine("-", body, old_no, None))
                    old_no, old_left = old_no + 1, old_left - 1
                else:
                    hunk.lines.append(DiffLine("+", body, None, new_no))
                    new_no, new_left = new_no + 1, new_left - 1
                continue
            warnings.append(f"{cur.path if cur else '?'}: hunk ended early ({hunk.header})")
            hunk = None

        if raw.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+) b/(.+)$", raw)
            cur = FileDiff(path=m.group(2) if m else raw[11:])
            files.append(cur)
            hunk = None
        elif cur is None:
            continue
        elif raw.startswith("new file mode"):
            cur.status = "added"
        elif raw.startswith("deleted file mode"):
            cur.status = "deleted"
        elif raw.startswith("rename to "):
            cur.status, cur.path = "renamed", raw[len("rename to "):]
        elif raw.startswith("Binary files "):
            cur.status = "binary"
        elif raw.startswith("+++ ") and raw[4:] != "/dev/null":
            cur.path = raw[4:][2:] if raw[4:].startswith("b/") else raw[4:]
        elif m := HUNK_RE.match(raw):
            old_start, new_start = int(m.group(1)), int(m.group(3))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            hunk = Hunk(raw, old_start, old_count, new_start, new_count)
            cur.hunks.append(hunk)
            old_no, new_no, old_left, new_left = old_start, new_start, old_count, new_count

    if hunk is not None and (old_left > 0 or new_left > 0):
        warnings.append(f"{cur.path if cur else '?'}: diff truncated inside {hunk.header}")
    return files, warnings


# ---------------------------------------------------------------------------
# 2. Chunking for review
# ---------------------------------------------------------------------------
SKIP_PATTERNS = [r"(^|/)package-lock\.json$", r"(^|/)yarn\.lock$", r"\.lock$", r"(^|/)dist/", r"\.min\.js$"]


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def render_hunk(h: Hunk) -> str:
    """Prefix every line with its NEW-file line number so the model can cite real lines."""
    out = [h.header]
    for ln in h.lines:
        num = f"{ln.new_no:>4}" if ln.new_no is not None else "    "
        out.append(f"{num} {ln.kind}{ln.text}")
    return "\n".join(out)


@dataclass
class Chunk:
    id: str
    file: str
    status: str
    hunk_headers: list[str]
    text: str
    est_tokens: int
    valid_lines: set[int]


def build_chunks(files: list[FileDiff], max_tokens: int) -> tuple[list[Chunk], list[tuple[str, str]]]:
    chunks: list[Chunk] = []
    skipped: list[tuple[str, str]] = []
    for f in files:
        if f.status == "binary":
            skipped.append((f.path, "binary file"))
            continue
        if any(re.search(p, f.path) for p in SKIP_PATTERNS):
            skipped.append((f.path, "lockfile / generated - not worth model tokens"))
            continue
        if f.status == "deleted":
            skipped.append((f.path, "deleted file - nothing to comment on"))
            continue
        group: list[Hunk] = []

        def flush() -> None:
            if not group:
                return
            body = "\n".join(render_hunk(h) for h in group)
            text = f"File: {f.path} ({f.status})\n{body}"
            lines = {ln.new_no for h in group for ln in h.lines if ln.new_no is not None}
            chunks.append(Chunk(f"C{len(chunks) + 1}", f.path, f.status, [h.header for h in group], text, est_tokens(text), lines))
            group.clear()

        for h in f.hunks:
            candidate = "\n".join(render_hunk(x) for x in group + [h])
            if group and est_tokens(candidate) > max_tokens:
                flush()
            group.append(h)
        flush()
    return chunks, skipped


# ---------------------------------------------------------------------------
# 3. Review post-processing
# ---------------------------------------------------------------------------
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "nit": 4}


@dataclass
class ReviewComment:
    file: str
    line: int
    severity: str
    category: str
    title: str
    suggestion: str
    confidence: float
    chunk_id: str


def postprocess(raw: list[ReviewComment], chunks: list[Chunk], min_confidence: float) -> tuple[list[ReviewComment], list[tuple[ReviewComment, str]]]:
    """Never post model output verbatim: ground it in the diff, filter noise, dedupe."""
    valid: dict[str, set[int]] = {}
    for c in chunks:
        valid.setdefault(c.file, set()).update(c.valid_lines)
    kept: dict[tuple[str, int, str], ReviewComment] = {}
    dropped: list[tuple[ReviewComment, str]] = []
    for f in raw:
        if f.file not in valid:
            dropped.append((f, "file not in diff (hallucinated)"))
        elif f.line not in valid[f.file]:
            dropped.append((f, f"line {f.line} not in diff (hallucinated)"))
        elif f.confidence < min_confidence:
            dropped.append((f, f"confidence {f.confidence:.2f} < {min_confidence:.2f}"))
        else:
            key = (f.file, f.line, f.category)
            prev = kept.get(key)
            if prev is None:
                kept[key] = f
            elif f.confidence > prev.confidence:
                dropped.append((prev, "duplicate (lower confidence)"))
                kept[key] = f
            else:
                dropped.append((f, "duplicate (lower confidence)"))
    ordered = sorted(kept.values(), key=lambda x: (SEVERITY_RANK[x.severity], x.file, x.line))
    return ordered, dropped


def severity_counts(findings: list[ReviewComment]) -> dict[str, int]:
    return {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_RANK}


def render_pr_markdown(findings: list[ReviewComment], dropped_count: int) -> str:
    counts = severity_counts(findings)
    out = [f"## AI review (advisory) - {len(findings)} findings", "",
           "| Severity | Count |", "|---|---|"]
    out += [f"| {s} | {n} |" for s, n in counts.items()]
    out += ["", f"_{dropped_count} raw findings were filtered (ungrounded, low confidence, or duplicates)._",
            "> These comments are suggestions. A human reviewer approves or rejects this PR.", ""]
    current = None
    for f in sorted(findings, key=lambda x: (x.file, x.line)):
        if f.file != current:
            out += [f"### `{f.file}`", ""]
            current = f.file
        out.append(f"- **L{f.line} [{f.severity} | {f.category}]** {f.title}")
        out.append(f"  - Suggestion: {f.suggestion}")
        out.append(f"  - Confidence: {f.confidence:.2f}")
    return "\n".join(out) + "\n"


def review_payload(findings: list[ReviewComment]) -> dict[str, Any]:
    """Shape of GitHub's 'create a review' request. event=COMMENT: the bot never approves or blocks."""
    counts = severity_counts(findings)
    summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
    return {
        "event": "COMMENT",
        "body": f"AI review (advisory): {summary}. A human makes the merge decision.",
        "comments": [
            {"path": f.file, "line": f.line, "side": "RIGHT",
             "body": f"**[{f.severity} | {f.category}]** {f.title}\n\nSuggestion: {f.suggestion}"}
            for f in findings
        ],
    }


# ---------------------------------------------------------------------------
# 4. Conventional Commits + changelog
# ---------------------------------------------------------------------------
CC_TYPES = ["feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert"]
CHANGELOG_SECTIONS = ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]
NON_IMPERATIVE = {"adds", "fixes", "updates", "removes", "changes", "implements"}


def commit_header(d: dict[str, Any]) -> str:
    scope = f"({d['scope']})" if d.get("scope") else ""
    bang = "!" if d.get("breaking") else ""
    return f"{d['type']}{scope}{bang}: {d['subject']}"


def validate_commit(d: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if d.get("type") not in CC_TYPES:
        errors.append(f'type "{d.get("type")}" is not a Conventional Commits type')
    scope = d.get("scope") or ""
    if scope and not re.fullmatch(r"[a-z0-9-]+", scope):
        errors.append(f'scope "{scope}" must be lowercase kebab-case')
    subject = d.get("subject") or ""
    words = subject.split()
    first = words[0].lower() if words else ""
    if not subject:
        errors.append("subject is empty")
    elif subject[0] != subject[0].lower():
        errors.append("subject must start with a lowercase letter")
    if first and (first.endswith("ed") or first.endswith("ing") or first in NON_IMPERATIVE):
        errors.append(f'subject must use the imperative mood ("{words[0]}" -> e.g. "add")')
    if subject.endswith("."):
        errors.append("subject must not end with a period")
    header = commit_header(d)
    if len(header) > 72:
        errors.append(f"header is {len(header)} chars (max 72)")
    for i, line in enumerate(d.get("body") or [], 1):
        if len(line) > 72:
            errors.append(f"body line {i} is {len(line)} chars (max 72)")
    if not d.get("changelog"):
        errors.append("at least one changelog entry is required")
    return errors


def render_commit_message(d: dict[str, Any]) -> str:
    parts = [commit_header(d)]
    if d.get("body"):
        parts += ["", *d["body"]]
    if d.get("breaking"):
        parts += ["", "BREAKING CHANGE: see changelog"]
    return "\n".join(parts) + "\n"


def render_changelog(entries: list[dict[str, str]], version: str = "Unreleased") -> str:
    out = [f"## [{version}]"]
    for sec in CHANGELOG_SECTIONS:
        items = [e["text"] for e in entries if e["section"] == sec]
        if items:
            out += ["", f"### {sec}", *[f"- {t}" for t in items]]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 5. Generated test cases: validate, measure coverage, execute
# ---------------------------------------------------------------------------
MAX_DISCOUNT_PERCENT = 50


def apply_discount_from_diff(price_cents: int, percent: int) -> int:
    """Faithful port of the function in sample.diff (float math, truncation, no validation)."""
    return int(price_cents * (1 - percent / 100))


def apply_discount_reference(price_cents: int, percent: int) -> int:
    """Executable spec (data/apply_discount_spec.md): half-up rounding, validated inputs."""
    if not 0 <= percent <= MAX_DISCOUNT_PERCENT:
        raise ValueError("percent out of range")
    if price_cents < 0:
        raise ValueError("negative price")
    return (price_cents * (100 - percent) + 50) // 100


def validate_cases(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    """Structured outputs guarantee the JSON *shape*; these are the *semantic* rules a schema can't express."""
    valid: list[dict[str, Any]] = []
    rejected: list[tuple[dict[str, Any], str]] = []
    names: set[str] = set()
    for c in cases:
        name = c.get("name", "")
        has_expected, has_raises = c.get("expected") is not None, c.get("raises") is not None
        if not re.fullmatch(r"test_[a-z0-9_]+", name):
            rejected.append((c, "name must match test_[a-z0-9_]+"))
        elif name in names:
            rejected.append((c, "duplicate test name"))
        elif has_expected == has_raises:
            rejected.append((c, "exactly one of expected / raises must be set"))
        elif has_raises and c["raises"] != "ValueError":
            rejected.append((c, f"unexpected exception type {c['raises']}"))
        else:
            names.add(name)
            valid.append(c)
    return valid, rejected


def coverage(cases: list[dict[str, Any]]) -> dict[str, bool]:
    def rounding(c: dict[str, Any]) -> bool:
        return c.get("expected") is not None and (c["price_cents"] * (100 - c["percent"])) % 100 != 0

    return {
        "typical value": any(c.get("expected") is not None and 0 < c["percent"] < MAX_DISCOUNT_PERCENT for c in cases),
        "boundary percent=0": any(c["percent"] == 0 for c in cases),
        "boundary percent=max": any(c["percent"] == MAX_DISCOUNT_PERCENT for c in cases),
        "fractional-cent rounding": any(rounding(c) for c in cases),
        "invalid input raises": any(c.get("raises") for c in cases),
    }


def run_case(fn: Callable[[int, int], int], c: dict[str, Any]) -> tuple[bool, str]:
    try:
        got = fn(c["price_cents"], c["percent"])
    except ValueError:
        return c.get("raises") == "ValueError", "raised ValueError"
    if c.get("raises"):
        return False, f"returned {got}"
    return got == c["expected"], f"returned {got}"


def triage(spec_ok: bool, code_ok: bool) -> str:
    if not spec_ok:
        return "SUSPECT TEST - contradicts the spec, needs human review"
    if not code_ok:
        return "BUG FOUND in PR code"
    return "pass (regression guard)"


def render_pytest(cases: list[dict[str, Any]], suspect: set[str]) -> str:
    out = ['"""Generated by Claude (D7.2 lab), validated by code, reviewed by a human before commit."""',
           "import pytest", "", "from app.services.pricing import apply_discount", ""]
    for c in cases:
        prefix = ""
        if c["name"] in suspect:
            out.append("# REVIEW: this expectation contradicts the spec - excluded until a human decides")
            prefix = "# "
        out.append(f"{prefix}def {c['name']}():")
        if c.get("raises"):
            out.append(f"{prefix}    with pytest.raises({c['raises']}):")
            out.append(f"{prefix}        apply_discount({c['price_cents']}, {c['percent']})")
        else:
            out.append(f"{prefix}    assert apply_discount({c['price_cents']}, {c['percent']}) == {c['expected']}")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 6. CI: workflow linter + headless result gate
# ---------------------------------------------------------------------------
SECRET_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}|gh[pousr]_[A-Za-z0-9_]{10,}")


@dataclass
class WorkflowFinding:
    severity: str
    code: str
    message: str


def lint_workflow(text: str) -> list[WorkflowFinding]:
    out: list[WorkflowFinding] = []

    def has(rx: str) -> bool:
        return re.search(rx, text, re.MULTILINE) is not None

    if has(r"^\s*pull_request_target\s*:") and has(r"pull_request\.head\.(sha|ref)"):
        out.append(WorkflowFinding("ERROR", "W001", "pull_request_target + checkout of the PR head runs untrusted code with secrets"))
    if SECRET_RE.search(text):
        out.append(WorkflowFinding("ERROR", "W002", "literal API key/token in the workflow; use ${{ secrets.NAME }}"))
    perms = re.search(r"^permissions:[ \t]*(\S.*)?$", text, re.MULTILINE)
    if not perms:
        out.append(WorkflowFinding("WARN", "W003", "no top-level permissions block; the token gets repository defaults"))
    elif perms.group(1) and "write-all" in perms.group(1):
        out.append(WorkflowFinding("ERROR", "W003", "permissions: write-all - grant only contents: read, pull-requests: write"))
    if not has(r"timeout-minutes:"):
        out.append(WorkflowFinding("WARN", "W004", "no timeout-minutes; a stuck agent burns runner time and tokens"))
    if not has(r"^concurrency:"):
        out.append(WorkflowFinding("WARN", "W005", "no concurrency group; superseded pushes keep paying for stale reviews"))
    for m in re.finditer(r"uses:\s*([\w.-]+/[\w.-]+)@([\w.-]+)", text):
        if m.group(2) in ("main", "master", "latest", "HEAD"):
            out.append(WorkflowFinding("WARN", "W006", f"{m.group(1)}@{m.group(2)} is a moving target; pin a release tag or commit SHA"))
    if has(r"--dangerously-skip-permissions|bypassPermissions"):
        out.append(WorkflowFinding("ERROR", "W007", "permission checks disabled for an agent that can push code"))
    if has(r"\bclaude\s+-p\b|claude-code-action") and not has(r"--max-turns"):
        out.append(WorkflowFinding("WARN", "W008", "no --max-turns cap on the agent (runaway cost)"))
    top_env = re.search(r"^env:[ \t]*\n((?:[ \t]+.*\n?)+)", text, re.MULTILINE)
    if top_env and "secrets." in top_env.group(1):
        out.append(WorkflowFinding("WARN", "W009", "secrets in workflow-level env are exposed to every step; scope them to one step"))
    if not has(r"draft\s*==\s*false"):
        out.append(WorkflowFinding("INFO", "W010", "no draft filter; draft PRs trigger paid reviews"))
    if not has(r"^\s*paths:"):
        out.append(WorkflowFinding("INFO", "W011", "no paths filter; docs-only changes trigger reviews"))
    return out


def gate_headless(result: dict[str, Any], budget_usd: float) -> tuple[str, list[str]]:
    """Decide the CI step outcome from `claude -p ... --output-format json` output."""
    notes = [f"subtype={result.get('subtype')} is_error={compact(result.get('is_error'))} turns={result.get('num_turns')} "
             f"duration={result.get('duration_ms')}ms cost=${result.get('total_cost_usd', 0):.4f}"]
    if result.get("is_error") or result.get("subtype") != "success":
        return "FAIL", notes + ["run did not succeed - fail the job, do not post partial output"]
    if result.get("total_cost_usd", 0) > budget_usd:
        return "WARN", notes + [f"over the ${budget_usd:.2f} per-PR budget - alert the owning team"]
    return "PASS", notes + ["post `result` as a PR comment; keep session_id for audit"]


def compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))
