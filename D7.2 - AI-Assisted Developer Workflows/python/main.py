"""
D7.2 - AI-Assisted Developer Workflows

Four workflows a platform team typically automates with Claude, each wrapped in deterministic code:
  a) AI PR reviewer: parse diff -> chunk -> structured findings -> ground/filter/dedupe -> PR comments
  b) Conventional Commit message + changelog, with a validate-and-repair loop
  c) test-case generation for a small function, validated and executed against code AND spec
  d) how it runs in CI: GitHub Actions + headless `claude -p --output-format json` (linted, not executed)
and measures tokens/cost per PR and findings by severity.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET, PRICING, MockRequest, banner, cost_usd, get_client, section, text_of,
)

from devflow import (  # noqa: E402
    ReviewComment, apply_discount_from_diff, apply_discount_reference, build_chunks, commit_header, compact, coverage,
    gate_headless, lint_workflow, parse_unified_diff, postprocess, render_changelog, render_commit_message,
    render_pr_markdown, render_pytest, review_payload, run_case, severity_counts, triage, validate_cases,
    validate_commit,
)

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
OUT = LAB / "labs_output"
MAX_CHUNK_TOKENS = 160     # deliberately small so the demo shows a file split across chunks
MIN_CONFIDENCE = 0.6
PRS_PER_MONTH = 400


# ---------------------------------------------------------------------------
# Structured-output schemas (the SDK turns these into output_config.format JSON schemas)
# ---------------------------------------------------------------------------
class ReviewFindingModel(BaseModel):
    file: str
    line: int
    severity: Literal["critical", "high", "medium", "low", "nit"]
    category: Literal["security", "correctness", "performance", "maintainability", "tests", "style"]
    title: str
    suggestion: str
    confidence: float


class ReviewResultModel(BaseModel):
    findings: list[ReviewFindingModel]


class ChangelogEntryModel(BaseModel):
    section: Literal["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]
    text: str


class CommitDraftModel(BaseModel):
    type: Literal["feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert"]
    scope: str
    subject: str
    body: list[str]
    breaking: bool
    changelog: list[ChangelogEntryModel]


class GeneratedCaseModel(BaseModel):
    name: str
    price_cents: int
    percent: int
    expected: int | None
    raises: str | None
    rationale: str


class GeneratedSuiteModel(BaseModel):
    cases: list[GeneratedCaseModel]


# ---------------------------------------------------------------------------
# Offline responder: realistic canned model output (live mode calls the real API)
# ---------------------------------------------------------------------------
def _f(file: str, line: int, sev: str, cat: str, title: str, fix: str, conf: float) -> dict[str, Any]:
    return {"file": file, "line": line, "severity": sev, "category": cat, "title": title, "suggestion": fix, "confidence": conf}


CANNED_REVIEW: dict[str, list[dict[str, Any]]] = {
    "app/api/orders.py": [
        _f("app/api/orders.py", 24, "critical", "security", "User input is interpolated into SQL with an f-string (SQL injection).",
           'Use a bound parameter: conn.execute("SELECT id, total_cents FROM orders WHERE email = :email", {"email": customer_email}).', 0.97),
        _f("app/api/orders.py", 21, "high", "security", "New endpoint has no auth dependency, so anyone can read any customer's order history.",
           "Add user=Depends(require_user) and only return orders that belong to user.id.", 0.92),
        _f("app/api/orders.py", 20, "low", "security", "Customer email in the URL path leaks PII into access logs.",
           "Identify the customer from the authenticated user instead of an email path parameter.", 0.55),
        _f("app/api/orders.py", 24, "high", "security", "Unsanitised customer_email reaches the SQL query.",
           "Parameterise the query.", 0.81),
        _f("app/api/orders.py", 99, "medium", "performance", "Query result is unbounded; add pagination.",
           "Add limit/offset parameters.", 0.70),
    ],
    "app/services/pricing.py": [
        _f("app/services/pricing.py", 10, "high", "correctness", "Money is computed in floating point and truncated by int(): 1005 at 10% gives 904, not 905.",
           "Use integer math with half-up rounding: (price_cents * (100 - percent) + 50) // 100.", 0.90),
        _f("app/services/pricing.py", 8, "medium", "correctness", "percent and price_cents are not validated; negative or >50 values yield wrong prices.",
           "Raise ValueError unless 0 <= percent <= MAX_DISCOUNT_PERCENT and price_cents >= 0.", 0.86),
        _f("app/services/pricing.py", 14, "low", "maintainability", "MAX_DISCOUNT_PERCENT is defined after the function and never used.",
           "Move the constant above apply_discount and enforce it in the validation.", 0.66),
    ],
    "app/config.py": [
        _f("app/config.py", 4, "high", "security", "DEBUG is hard-coded to True, exposing stack traces in production.",
           'Restore the environment-driven default: DEBUG = os.getenv("DEBUG", "false") == "true".', 0.93),
        _f("app/config.py", 3, "medium", "performance", "A 300 s request timeout ties up workers during upstream incidents.",
           "Keep a short default (30 s) and move long-running work to a background job.", 0.72),
    ],
    "tests/test_pricing.py": [
        _f("tests/test_pricing.py", 5, "medium", "tests", "Only the happy path is tested; rounding, boundaries (0, 50) and invalid inputs are not.",
           "Add cases for 1005 at 10% (905), percent 0 and 50, percent 51 and -5, and a negative price.", 0.84),
        _f("tests/conftest.py", 1, "low", "tests", "Shared fixtures are missing.", "Add a conftest.py with a database fixture.", 0.62),
    ],
}

COMMIT_ATTEMPT_1 = {
    "type": "feat", "scope": "Orders API",
    "subject": "Added a new order history endpoint and discount support to the pricing service.",
    "body": ["Adds an endpoint and a helper."], "breaking": False,
    "changelog": [{"section": "Added", "text": "Order history endpoint."}],
}
COMMIT_ATTEMPT_2 = {
    "type": "feat", "scope": "orders",
    "subject": "add order history endpoint with discount pricing",
    "body": [
        "- Add GET /orders/history/{customer_email} returning order totals.",
        "- Add apply_discount() to the pricing service.",
        "- Raise REQUEST_TIMEOUT to 300 s and set DEBUG in app/config.py.",
    ],
    "breaking": False,
    "changelog": [
        {"section": "Added", "text": "Order history endpoint `GET /orders/history/{customer_email}`."},
        {"section": "Added", "text": "`apply_discount()` helper for percentage discounts."},
        {"section": "Changed", "text": "Default request timeout raised from 30 s to 300 s."},
        {"section": "Changed", "text": "`DEBUG` is now always enabled."},
    ],
}


def _case(name: str, price: int, pct: int, expected: int | None, raises: str | None, why: str) -> dict[str, Any]:
    return {"name": name, "price_cents": price, "percent": pct, "expected": expected, "raises": raises, "rationale": why}


CANNED_TESTS = {"cases": [
    _case("test_no_discount_returns_same_price", 1999, 0, 1999, None, "boundary: 0% leaves price unchanged"),
    _case("test_ten_percent_rounds_half_up", 1005, 10, 905, None, "904.5 cents must round half-up to 905"),
    _case("test_fifteen_percent_typical", 1999, 15, 1699, None, "typical discount, 1699.15 rounds to 1699"),
    _case("test_max_discount_boundary", 1000, 50, 500, None, "boundary: maximum allowed discount"),
    _case("test_discount_above_max_raises", 1000, 51, None, "ValueError", "51% exceeds MAX_DISCOUNT_PERCENT"),
    _case("test_negative_percent_raises", 1000, -5, None, "ValueError", "negative discount would raise the price"),
    _case("test_negative_price_raises", -100, 10, None, "ValueError", "prices cannot be negative"),
    _case("test_thirty_three_percent_rounding", 999, 33, 670, None, "669.33 rounds to 670"),  # wrong on purpose
    _case("test_zero_price", 0, 25, 0, None, "free items stay free"),
    _case("test_zero_price", 0, 10, 0, None, "duplicate name"),
    _case("test_half_price_or_error", 1000, 50, 500, "ValueError", "ambiguous: both outcomes set"),
]}


def responder(req: MockRequest) -> str:
    system = req.system_text
    text = req.last_user_text
    if "automated code reviewer" in system:
        m = re.search(r"^File: (\S+)", text, re.MULTILINE)
        file = m.group(1) if m else ""
        if file == "app/api/orders.py" and "+15,16" not in text:
            return compact({"findings": []})   # the import-only hunk is clean
        return compact({"findings": CANNED_REVIEW.get(file, [])})
    if "commit message" in system:
        return compact(COMMIT_ATTEMPT_2 if "Validation errors" in text else COMMIT_ATTEMPT_1)
    if "test cases" in system:
        return compact(CANNED_TESTS)
    return "{}"


# ---------------------------------------------------------------------------
# Cost ledger: every call is attributed to a workflow step
# ---------------------------------------------------------------------------
@dataclass
class CallRecord:
    step: str
    model: str
    usage: Any


LEDGER: list[CallRecord] = []


def record(step: str, msg: Any) -> None:
    LEDGER.append(CallRecord(step, msg.model, msg.usage))


def u(usage: Any, key: str) -> int:
    return getattr(usage, key, 0) or 0


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_parse() -> list:
    section("1. Parse the unified diff (data/sample.diff)")
    files, warnings = parse_unified_diff((DATA / "sample.diff").read_text(encoding="utf-8"))
    for f in files:
        print(f"  {f.path:<26} {f.status:<9} +{f.additions:<3} -{f.deletions:<3} hunks={len(f.hunks)}")
    hunks = sum(len(f.hunks) for f in files)
    print(f"  {len(files)} files, {hunks} hunks; hunk header counts verified: {'yes' if not warnings else 'NO'}")
    for w in warnings:
        print(f"  WARNING: {w}")
    return files


def step_chunk(files: list) -> list:
    section(f"2. Chunk for review (budget ~{MAX_CHUNK_TOKENS} tokens per chunk)")
    chunks, skipped = build_chunks(files, MAX_CHUNK_TOKENS)
    for c in chunks:
        print(f"  {c.id}  {c.file:<26} hunks={len(c.hunk_headers)} ~{c.est_tokens:>3} tokens  lines {min(c.valid_lines)}-{max(c.valid_lines)}")
    for path, reason in skipped:
        print(f"  skip {path:<26} {reason}")
    print("  Chunks carry NEW-file line numbers so the model can cite lines we can verify.")
    return chunks


def step_review(client, chunks: list) -> list[ReviewComment]:
    section("3. AI review per chunk (Sonnet 5.5, structured output, cached rubric)")
    guidelines = (DATA / "review_guidelines.md").read_text(encoding="utf-8")
    raw: list[ReviewComment] = []
    for c in chunks:
        msg = client.messages.parse(
            model=MODEL_SONNET,
            max_tokens=4000,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            # Stable rubric first + cache_control: chunks 2..N read it from cache at ~10% of the input price.
            system=[{"type": "text", "text": guidelines, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": f"Review this chunk of PR #482.\n\n{c.text}"}],
            output_format=ReviewResultModel,
        )
        record("review", msg)
        result = msg.parsed_output
        found = result.findings if result else []
        raw += [ReviewComment(f.file, f.line, f.severity, f.category, f.title, f.suggestion, f.confidence, c.id) for f in found]
        us = msg.usage
        print(f"  {c.id} {c.file:<26} raw findings={len(found)}  in={us.input_tokens} out={us.output_tokens} "
              f"cache_write={u(us, 'cache_creation_input_tokens')} cache_read={u(us, 'cache_read_input_tokens')}")
    return raw


def step_postprocess(raw: list[ReviewComment], chunks: list) -> list[ReviewComment]:
    section(f"4. Ground, filter (confidence >= {MIN_CONFIDENCE}), dedupe -> PR comments")
    kept, dropped = postprocess(raw, chunks, MIN_CONFIDENCE)
    for f, reason in dropped:
        print(f"  drop {f.file + ':' + str(f.line):<28} {f.severity:<8} {reason}")
    print()
    for f in kept:
        print(f"  keep {f.file + ':' + str(f.line):<28} {f.severity:<8} {f.category:<15} conf={f.confidence:.2f}  {f.title[:46]}")
    counts = severity_counts(kept)
    print("\n  findings by severity: " + ", ".join(f"{s}={n}" for s, n in counts.items()))
    blocking = [f for f in kept if f.severity in ("critical", "high") and f.confidence >= 0.8]
    print(f"  advisory verdict: {'changes suggested' if blocking else 'looks good'} ({len(blocking)} critical/high) - "
          "posted as COMMENT; branch protection still requires a human approval")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pr_review.md").write_text(render_pr_markdown(kept, len(dropped)), encoding="utf-8")
    (OUT / "review_payload.json").write_text(json.dumps(review_payload(kept), indent=2), encoding="utf-8")
    print("  wrote labs_output/pr_review.md and labs_output/review_payload.json")
    return kept


def step_commit(client, files: list) -> None:
    section("5. Commit message + changelog (Haiku 5.5, validate-and-repair loop)")
    stat = "\n".join(f"{f.path} | +{f.additions} -{f.deletions} ({f.status})" for f in files)
    for line in stat.splitlines():
        print(f"  {line}")
    diff_text = (DATA / "sample.diff").read_text(encoding="utf-8")
    system = ("You write git commit messages in Conventional Commits format and Keep a Changelog entries. "
              "Subject: imperative mood, lowercase, no trailing period, header <= 72 chars. Body lines <= 72 chars.")
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"Diffstat:\n{stat}\n\nDiff:\n{diff_text}\n\nWrite the commit."}]
    draft: dict[str, Any] = {}
    for attempt in (1, 2, 3):
        msg = client.messages.parse(
            model=MODEL_HAIKU, max_tokens=1000, thinking={"type": "adaptive"}, output_config={"effort": "low"},
            system=system, messages=messages, output_format=CommitDraftModel,
        )
        record("commit-message", msg)
        draft = msg.parsed_output.model_dump() if msg.parsed_output else {}
        errors = validate_commit(draft) if draft else ["unparseable output"]
        print(f"\n  attempt {attempt}: {commit_header(draft) if draft else '(none)'}")
        if not errors:
            print("    valid Conventional Commit")
            break
        for e in errors:
            print(f"    x {e}")
        # Repair turn: show the model its own output plus the validator's errors (no prefill needed).
        messages += [{"role": "assistant", "content": text_of(msg)},
                     {"role": "user", "content": "Validation errors:\n" + "\n".join(f"- {e}" for e in errors) + "\nReturn a corrected JSON object."}]
    commit_msg = render_commit_message(draft)
    changelog = render_changelog(draft.get("changelog", []))
    print("\n  --- commit message ---")
    for line in commit_msg.splitlines():
        print(f"  {line}")
    print("  --- CHANGELOG.md fragment ---")
    for line in changelog.splitlines():
        print(f"  {line}")
    (OUT / "COMMIT_MSG.txt").write_text(commit_msg, encoding="utf-8")
    (OUT / "CHANGELOG.fragment.md").write_text(changelog, encoding="utf-8")


def step_tests(client) -> None:
    section("6. Test-case generation for apply_discount(), validated by code")
    spec = (DATA / "apply_discount_spec.md").read_text(encoding="utf-8")
    msg = client.messages.parse(
        model=MODEL_SONNET, max_tokens=4000, thinking={"type": "adaptive"}, output_config={"effort": "medium"},
        system="You design unit test cases from a specification. Return test cases as data, one behaviour per case.",
        messages=[{"role": "user", "content": f"Specification:\n{spec}\nGenerate 8-12 test cases covering typical values, "
                                               "boundaries, fractional-cent rounding and invalid inputs."}],
        output_format=GeneratedSuiteModel,
    )
    record("test-generation", msg)
    cases = [c.model_dump() for c in msg.parsed_output.cases] if msg.parsed_output else []
    valid, rejected = validate_cases(cases)
    print(f"  generated {len(cases)} cases; structurally valid {len(valid)}; rejected {len(rejected)}")
    for c, reason in rejected:
        print(f"    reject {c['name']}: {reason}")
    print("  coverage: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in coverage(valid).items()))
    print("\n  execute each case against the executable spec AND the code in the PR:")
    suspect: set[str] = set()
    tallies: dict[str, int] = {}
    for c in valid:
        spec_ok, spec_obs = run_case(apply_discount_reference, c)
        code_ok, code_obs = run_case(apply_discount_from_diff, c)
        verdict = triage(spec_ok, code_ok)
        tallies[verdict.split(" - ")[0]] = tallies.get(verdict.split(" - ")[0], 0) + 1
        if not spec_ok:
            suspect.add(c["name"])
        want = f"raises {c['raises']}" if c.get("raises") else f"== {c['expected']}"
        print(f"    {c['name']:<36} want {want:<18} spec:{'ok ' if spec_ok else 'BAD'} code:{'ok ' if code_ok else 'BAD'} ({code_obs}) -> {verdict}")
    print("  summary: " + ", ".join(f"{k}={n}" for k, n in tallies.items()))
    (OUT / "test_pricing_generated.py").write_text(render_pytest(valid, suspect), encoding="utf-8")
    print("  wrote labs_output/test_pricing_generated.py (suspect cases commented out for human review)")


def step_ci() -> None:
    section("7. Running it in CI (documented - requires Claude Code / GitHub, not executed)")
    for name in ("claude-pr-review.yml", "risky-pr-review.yml"):
        findings = lint_workflow((DATA / "ci" / name).read_text(encoding="utf-8"))
        print(f"  lint data/ci/{name}: {len(findings)} findings")
        for f in findings:
            print(f"    [{f.severity:<5}] {f.code} {f.message}")
    print("\n  headless mode as used in the workflow:")
    print('    claude -p "Summarise pr.diff for release notes" --output-format json --max-turns 3 --allowedTools "Read"')
    result = json.loads((DATA / "ci" / "headless_result.json").read_text(encoding="utf-8"))
    status, notes = gate_headless(result, budget_usd=0.50)
    print(f"  gate on data/ci/headless_result.json -> {status}")
    for n in notes:
        print(f"    {n}")
    print("\n  guard rails for AI in CI:")
    for line in (
        "secrets: ANTHROPIC_API_KEY only via ${{ secrets.* }}, scoped to the step that needs it",
        "least privilege: permissions contents:read + pull-requests:write; --allowedTools read-only",
        "never pull_request_target + PR-head checkout; never --dangerously-skip-permissions with write tokens",
        "cost: paths/draft filters, concurrency cancel, --max-turns, Sonnet/Haiku per step, cached rubric",
        "humans merge: bot posts COMMENT reviews; branch protection requires an approving human review",
    ):
        print(f"    - {line}")


def step_economics(kept: list[ReviewComment]) -> None:
    section("8. PR economics and outcome metrics")
    print(f"  {'step':<16} {'model':<18} {'calls':>5} {'in':>6} {'out':>5} {'c_write':>7} {'c_read':>6} {'cost':>10}")
    steps = []
    for r in LEDGER:
        if r.step not in steps:
            steps.append(r.step)
    total_cost = 0.0
    total_calls = 0
    for step in steps:
        recs = [r for r in LEDGER if r.step == step]
        cost = sum(cost_usd(r.model, r.usage) for r in recs)
        total_cost += cost
        total_calls += len(recs)
        print(f"  {step:<16} {recs[0].model:<18} {len(recs):>5} {sum(r.usage.input_tokens for r in recs):>6} "
              f"{sum(r.usage.output_tokens for r in recs):>5} {sum(u(r.usage, 'cache_creation_input_tokens') for r in recs):>7} "
              f"{sum(u(r.usage, 'cache_read_input_tokens') for r in recs):>6} ${cost:>9.5f}")
    print(f"  {'TOTAL per PR':<16} {'':<18} {total_calls:>5} {'':>6} {'':>5} {'':>7} {'':>6} ${total_cost:>9.5f}")
    print(f"  projection: {PRS_PER_MONTH} PRs/month -> ${total_cost * PRS_PER_MONTH:,.2f}/month (offline estimates are illustrative)")

    review = [r for r in LEDGER if r.step == "review"]
    p = PRICING[MODEL_SONNET]
    no_cache = sum((r.usage.input_tokens + u(r.usage, "cache_creation_input_tokens") + u(r.usage, "cache_read_input_tokens")) * p["input"]
                   + r.usage.output_tokens * p["output"] for r in review) / 1_000_000
    with_cache = sum(cost_usd(r.model, r.usage) for r in review)
    print(f"\n  review step: with cached rubric ${with_cache:.5f} vs no caching ${no_cache:.5f}")
    for model in (MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS):
        alt = sum(cost_usd(model, r.usage) for r in review)
        print(f"    same review tokens priced on {model:<18} ${alt:.5f}")
    counts = severity_counts(kept)
    print("\n  outcome: " + ", ".join(f"{s}={n}" for s, n in counts.items())
          + f" | total={len(kept)} | cost per kept finding ${total_cost / max(1, len(kept)):.5f}")
    print("  Track over time: acceptance rate of AI comments, false-positive rate, time-to-first-review.")


def main() -> None:
    banner("D7.2 - AI-Assisted Developer Workflows")
    client = get_client(responder)
    files = step_parse()
    chunks = step_chunk(files)
    raw = step_review(client, chunks)
    kept = step_postprocess(raw, chunks)
    step_commit(client, files)
    step_tests(client)
    step_ci()
    step_economics(kept)


if __name__ == "__main__":
    main()
