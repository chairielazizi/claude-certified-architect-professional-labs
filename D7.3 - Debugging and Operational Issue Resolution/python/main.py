"""
D7.3 - Debugging and Operational Issue Resolution

Incident: after deploy v2.14.0 the summarizer service hits Claude API 429s and 529s.
  1. deterministic log analysis: parse, cluster error signatures, timeline, deploy correlation
  2. triage agent: manual tool-use loop (search_logs, get_metrics, get_recent_deploys, get_runbook)
     -> structured incident report, then grounding checks against the deterministic facts
  3. resilient client: typed SDK errors most-specific-first, retry-after, jittered backoff,
     fallback model, circuit breaker, request_id logging (errors simulated - see resilience.py)
  4. operational debugging checklist as code: max_tokens truncation, cache_read=0, request linter
  5. the incident report
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET, MockRequest, banner, cost_usd, get_client, section, text_of,
    tool_use_block,
)

from diagnostics import CHECKLIST, cache_check, diagnose_response, lint_request  # noqa: E402
from loganalysis import bar, cluster_errors, correlate, epoch, mean, parse_log  # noqa: E402
from resilience import AllModelsFailedError, Fault, FaultInjector, ResilientClaude, VirtualClock, mulberry32  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
OUT = LAB / "labs_output"
NOW = "2026-10-08T14:15:00Z"   # frozen "current time" so the demo is deterministic
MAX_TURNS = 8


def compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def load_json(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def wrap(text: str, width: int = 96, indent: str = "    ") -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return [indent + line for line in lines]


# ---------------------------------------------------------------------------
# Incident report schema: JSON schema for the API, Pydantic for validation
# ---------------------------------------------------------------------------
class TimelineEntry(BaseModel):
    time: str
    event: str


class IncidentReport(BaseModel):
    title: str
    severity: Literal["SEV1", "SEV2", "SEV3"]
    impact: str
    trigger: str
    root_cause: str
    contributing_factors: list[str]
    evidence: list[str]
    timeline: list[TimelineEntry]
    immediate_actions: list[str]
    follow_up_actions: list[str]
    runbook_id: str
    runbook_url: str
    confidence: float


# ---------------------------------------------------------------------------
# Triage tools: real implementations over the incident data
# ---------------------------------------------------------------------------
METRIC_NAMES = ["rps", "p95_latency_ms", "error_rate", "claude_429", "claude_529", "cache_hit_ratio", "input_tokens_per_min", "in_flight"]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_logs",
        "description": "Search application log lines (case-insensitive substring). Returns the total number of matches and "
                       "the first `limit` lines. since/until are HH:MM in UTC, or empty strings for no bound.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "level": {"type": "string", "enum": ["ANY", "INFO", "WARN", "ERROR"]},
                "since": {"type": "string"},
                "until": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query", "level", "since", "until", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_metrics",
        "description": "Per-minute metrics for the summarizer service between start and end (HH:MM, UTC). "
                       "Returns the points plus max and mean before/after the latest deploy.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"metric": {"type": "string", "enum": METRIC_NAMES}, "start": {"type": "string"}, "end": {"type": "string"}},
            "required": ["metric", "start", "end"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_recent_deploys",
        "description": "Deploys that finished within the last `hours` hours. Empty service string means all services.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"service": {"type": "string"}, "hours": {"type": "integer"}},
            "required": ["service", "hours"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_runbook",
        "description": "Find operational runbooks relevant to a topic (keywords such as error codes or symptoms).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
]


class TriageTools:
    def __init__(self, records: list, metrics: dict[str, Any], deploys: list[dict[str, Any]], runbooks: list[dict[str, Any]]):
        self.records, self.metrics, self.deploys, self.runbooks = records, metrics, deploys, runbooks

    def search_logs(self, query: str, level: str, since: str, until: str, limit: int) -> dict[str, Any]:
        q = query.lower()
        hits = [r for r in self.records
                if q in r.raw.lower() and (level == "ANY" or r.level == level)
                and (not since or r.time[:5] >= since) and (not until or r.time[:5] <= until)]
        limit = max(1, min(limit, 20))   # cap output: tool results cost context tokens
        return {"matches": len(hits), "returned": min(limit, len(hits)), "lines": [r.raw for r in hits[:limit]]}

    def get_metrics(self, metric: str, start: str, end: str) -> dict[str, Any]:
        pts = [p for p in self.metrics["points"] if start <= p["minute"] <= end]
        before = [p[metric] for p in pts if p["minute"] < "13:59"]
        after = [p[metric] for p in pts if p["minute"] >= "14:00"]
        return {"metric": metric, "points": [[p["minute"], p[metric]] for p in pts],
                "max": max((p[metric] for p in pts), default=0),
                "mean_before_deploy": f"{mean(before):.3f}", "mean_after_deploy": f"{mean(after):.3f}"}

    def get_recent_deploys(self, service: str, hours: int) -> dict[str, Any]:
        now = epoch(NOW)
        found = [d for d in self.deploys if now - epoch(d["finished"]) <= hours * 3600 and (not service or d["service"] == service)]
        return {"now": NOW, "deploys": found}

    def get_runbook(self, topic: str) -> dict[str, Any]:
        t = topic.lower()
        scored = [(sum(1 for k in rb["keywords"] if k.lower() in t), rb) for rb in self.runbooks]
        ranked = sorted([s for s in scored if s[0] > 0], key=lambda s: -s[0])[:2]
        return {"results": [{"id": rb["id"], "title": rb["title"], "url": rb["url"], "steps": rb["steps"], "score": score}
                            for score, rb in ranked]}

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handlers = {"search_logs": self.search_logs, "get_metrics": self.get_metrics,
                    "get_recent_deploys": self.get_recent_deploys, "get_runbook": self.get_runbook}
        if name not in handlers:
            raise ValueError(f"unknown tool {name}")
        return handlers[name](**args)


def summarize_tool_result(name: str, result: dict[str, Any]) -> str:
    if name == "search_logs":
        return f"matches={result['matches']} returned={result['returned']}"
    if name == "get_metrics":
        return f"{result['metric']}: max={result['max']} mean_before={result['mean_before_deploy']} mean_after={result['mean_after_deploy']}"
    if name == "get_recent_deploys":
        return "deploys=" + ", ".join(f"{d['service']}:{d['version']}@{d['finished'][11:16]}" for d in result["deploys"])
    if name == "get_runbook":
        return "results=" + ", ".join(f"{r['id']}(score {r['score']})" for r in result["results"])
    return "ok"


# ---------------------------------------------------------------------------
# Offline responder
# ---------------------------------------------------------------------------
CANNED_REPORT = load_json("offline/incident_report.json")
LONG_POSTMORTEM = (
    "INC-2291 postmortem summary. At 13:59:40 UTC deploy v2.14.0 of the summarizer service completed. The release "
    "changed three things at once: the prompt template, the worker concurrency and the retry policy. Within two minutes "
    "the Claude API began returning 429 rate_limit_error responses because uncached input tokens per minute exceeded "
    "the organisation limit. The new retry policy retried every 500 ms regardless of retry-after, which multiplied "
    "traffic and later amplified 529 overloaded_error responses. Customer impact lasted until the rollback. "
) * 2


def triage_responder(req: MockRequest) -> Any:
    turn = sum(1 for m in req.messages if m.get("role") == "assistant")
    if turn == 0:   # broad first: what changed, and what do the golden signals say?
        return [
            tool_use_block("get_recent_deploys", {"service": "summarizer", "hours": 24}),
            tool_use_block("get_metrics", {"metric": "claude_429", "start": "13:50", "end": "14:14"}),
            tool_use_block("get_metrics", {"metric": "cache_hit_ratio", "start": "13:50", "end": "14:14"}),
        ]
    if turn == 1:   # then narrow: confirm the hypothesis in the logs
        return [
            tool_use_block("search_logs", {"query": "status=429", "level": "ERROR", "since": "13:55", "until": "14:15", "limit": 3}),
            tool_use_block("search_logs", {"query": "retrying", "level": "WARN", "since": "", "until": "", "limit": 3}),
            tool_use_block("get_metrics", {"metric": "input_tokens_per_min", "start": "13:50", "end": "14:14"}),
        ]
    if turn == 2:
        return [
            tool_use_block("search_logs", {"query": "status=529", "level": "ERROR", "since": "", "until": "", "limit": 3}),
            tool_use_block("get_runbook", {"topic": "429 rate limit retry-after concurrency"}),
        ]
    return compact(CANNED_REPORT)


def responder(req: MockRequest) -> Any:
    if req.tools:
        return triage_responder(req)
    if "detailed postmortem" in req.last_user_text:
        return LONG_POSTMORTEM
    return "Q3 saw two incidents, both rate-limit related; caching and throttling fixes cut error rates by 90%."


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_log_analysis(records, unparsed, metrics, deploys):
    section("1. Deterministic log analysis (ground truth before any model call)")
    levels = {lvl: sum(1 for r in records if r.level == lvl) for lvl in ("INFO", "WARN", "ERROR")}
    print(f"  parsed {len(records)} lines ({unparsed} unparsed): " + ", ".join(f"{k}={v}" for k, v in levels.items()))
    print("\n  error/warning clusters (volatile fields like request_id, latency, attempt removed):")
    for c in cluster_errors(records):
        print(f"    {c.count:>3}x {c.level:<5} {c.first}-{c.last}  {c.signature[:92]}")

    facts = correlate(records, metrics, deploys)
    print("\n  per-minute timeline from metrics.json  (429 bar | 529 bar | error rate | cache hit | in-flight)")
    pts = metrics["points"]
    max429 = max(p["claude_429"] for p in pts)
    max529 = max(p["claude_529"] for p in pts)
    deploy_minute = facts.deploy_finished[:5]
    for p in pts:
        marker = f"  <- deploy {facts.deploy_version} finished {facts.deploy_finished}" if p["minute"] == deploy_minute else ""
        print(f"    {p['minute']}  {bar(p['claude_429'], max429, 20)} {p['claude_429']:>3} | {bar(p['claude_529'], max529, 10)} {p['claude_529']:>2} | "
              f"err {p['error_rate'] * 100:5.1f}% | cache {p['cache_hit_ratio'] * 100:3.0f}% | {p['in_flight']:>2}{marker}")

    print("\n  correlation with the deploy:")
    rows = [
        ("suspect deploy", f"{facts.deploy_version} (finished {facts.deploy_finished}); web-v5.2.0 ruled out (other service, 3h earlier)"),
        ("first 429 / first 529", f"{facts.first_429} ({facts.seconds_deploy_to_first_429} s after deploy) / {facts.first_529}"),
        ("cache hit ratio (logs)", f"{facts.cache_ratio_before:.2f} -> {facts.cache_ratio_after:.2f}"),
        ("uncached input tokens per call", f"{facts.uncached_input_per_call_before:.0f} -> {facts.uncached_input_per_call_after:.0f}"),
        ("input tokens/min vs limit", f"peak {facts.itpm_peak} vs {facts.itpm_limit} ({facts.itpm_peak / facts.itpm_limit * 100:.0f}% of limit)"),
        ("in-flight requests", f"{facts.in_flight_before:.0f} -> {facts.in_flight_after:.0f}"),
        ("error rate (metrics)", f"{facts.error_rate_before * 100:.2f}% -> {facts.error_rate_after * 100:.1f}%"),
        ("p95 latency (metrics)", f"{facts.p95_before_ms:.0f} ms -> {facts.p95_after_ms:.0f} ms"),
        ("retries ignoring retry-after", f"{facts.retries_ignoring_retry_after} (retry-after up to {facts.max_retry_after_s} s, retried after 500 ms)"),
    ]
    for k, v in rows:
        print(f"    {k:<32} {v}")
    return facts


def step_triage(client, tools: TriageTools, schema: dict[str, Any]) -> tuple[IncidentReport | None, dict[str, Any]]:
    section("2. Triage agent: manual tool-use loop (Opus 5.5, structured final answer)")
    system = (
        "You are an SRE incident triage agent for the summarizer service. Investigate with the tools before "
        "concluding: check recent deploys, metrics and logs, then find the matching runbook. Every claim must be "
        "backed by tool output with concrete times and numbers. When you are done, answer with the incident report."
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": (
        "PagerDuty: SummarizerErrorRateHigh fired at 14:08 UTC (error_rate 0.31). Find the root cause, the "
        f"remediation and the runbook. Current time {NOW}.")}]
    stats = {"turns": 0, "tool_calls": 0, "input": 0, "output": 0, "cost": 0.0}
    for turn in range(1, MAX_TURNS + 1):
        resp = client.messages.create(
            model=MODEL_OPUS,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        stats["turns"] = turn
        stats["input"] += resp.usage.input_tokens
        stats["output"] += resp.usage.output_tokens
        stats["cost"] += cost_usd(resp.model, resp.usage)
        if resp.stop_reason == "tool_use":
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": resp.content})
            results: list[dict[str, Any]] = []
            for tu in tool_uses:
                stats["tool_calls"] += 1
                try:
                    out = tools.dispatch(tu.name, dict(tu.input))
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": compact(out)})
                    print(f"  turn {turn}  {tu.name:<18} {compact(tu.input)[:62]:<62} => {summarize_tool_result(tu.name, out)}")
                except Exception as e:  # a failing tool becomes an is_error result the model can react to
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": f"tool error: {e}", "is_error": True})
                    print(f"  turn {turn}  {tu.name:<18} ERROR {e}")
            messages.append({"role": "user", "content": results})   # all results for this turn in ONE user message
            continue
        print(f"  turn {turn}  final answer: stop_reason={resp.stop_reason} -> {diagnose_response(resp)}")
        if resp.stop_reason != "end_turn":
            return None, stats
        report = IncidentReport.model_validate_json(text_of(resp))
        print(f"  agent: {stats['turns']} turns, {stats['tool_calls']} tool calls, in={stats['input']} out={stats['output']} cost=${stats['cost']:.5f}")
        return report, stats
    print(f"  stopped after MAX_TURNS={MAX_TURNS} without an answer")
    return None, stats


def grounding_checks(report: IncidentReport, facts, records, deploys, runbooks) -> list[tuple[str, bool, str]]:
    """Trust, but verify: every checkable claim is compared with the deterministic facts."""
    versions = {d["version"] for d in deploys}
    m = re.search(r"v\d+\.\d+\.\d+", report.trigger)
    log_times = {r.time for r in records}
    times = [t.time for t in report.timeline]
    rb = next((r for r in runbooks if r["id"] == report.runbook_id), None)
    evidence = " ".join(report.evidence)
    delay = f"{facts.seconds_deploy_to_first_429} s"
    return [
        ("trigger names a real deploy", bool(m and m.group(0) in versions), m.group(0) if m else "none"),
        ("runbook id exists and URL matches", bool(rb and rb["url"] == report.runbook_url), report.runbook_id),
        ("timeline times exist in the log", all(t in log_times for t in times), f"{sum(1 for t in times if t in log_times)}/{len(times)}"),
        ("timeline is chronological", times == sorted(times), f"{len(times)} entries"),
        ("evidence cites the first 429 time", facts.first_429 in evidence, facts.first_429),
        ("evidence cites deploy -> 429 delay", delay in evidence, delay),
        ("confidence within [0, 1]", 0 <= report.confidence <= 1, f"{report.confidence:.2f}"),
    ]


def step_resilience(client) -> None:
    section("3. Resilient client: typed errors, retry-after, jitter, fallback, breaker")
    print("  Errors are SIMULATED by FaultInjector using the SDK's own exception classes (the offline mock only")
    print("  returns 200/404). Sleeps use a virtual clock. Primary model: Opus 5.5, fallback: Sonnet 5.5.")
    injector, clock = FaultInjector(), VirtualClock()
    rc = ResilientClaude(client, [MODEL_OPUS, MODEL_SONNET], injector, clock, mulberry32(7))
    params = {"max_tokens": 600, "system": "You are the summarization service. Summarise the document in one sentence.",
              "messages": [{"role": "user", "content": "Document: Q3 reliability review (two incidents, both rate-limit related)."}]}
    f429 = lambda ra: Fault("status", 429, "rate_limit_error", "input tokens per minute rate limit exceeded", ra)  # noqa: E731
    f529 = Fault("status", 529, "overloaded_error", "Overloaded")
    scenarios: list[tuple[str, dict[str, list[Fault | None]], float]] = [
        ("S1 429 + retry-after 3s: wait exactly what the API asks", {MODEL_OPUS: [f429(3), None]}, 0),
        ("S2 529 twice: exponential backoff with full jitter", {MODEL_OPUS: [f529, f529, None]}, 0),
        ("S3 sustained 529: breaker opens after 3 failures -> fallback", {MODEL_OPUS: [f529, f529, f529]}, 0),
        ("S4 breaker OPEN: fail fast, no request sent to Opus", {}, 5),
        ("S5 cooldown elapsed: half-open probe closes the breaker", {}, 30),
        ("S6 400 invalid_request_error: never retry", {MODEL_OPUS: [Fault("status", 400, "invalid_request_error", "temperature: not supported for this model")]}, 0),
        ("S7 429 + retry-after 120s exceeds the 30s budget -> fallback", {MODEL_OPUS: [f429(120)]}, 0),
        ("S8 network timeout: retry with backoff", {MODEL_OPUS: [Fault("timeout"), None]}, 0),
    ]
    outcomes: list[str] = []
    for name, plan, advance in scenarios:
        print(f"\n  {name}")
        if advance:
            clock.sleep(advance)
            print(f"    (virtual clock advanced {advance:.0f}s)")
        for model, faults in plan.items():
            injector.script(model, faults)
        try:
            msg = rc.create(**params)
            outcome = f"OK, served by {msg.model}"
        except anthropic.BadRequestError as e:
            outcome = f"raised BadRequestError to the caller (request_id={e.request_id})"
        except AllModelsFailedError:
            outcome = "raised AllModelsFailedError"
        outcomes.append(outcome)
        states = ", ".join(f"{m.split('-')[1]}={b.state}" for m, b in rc.breakers.items())
        print(f"    => {outcome} | breakers: {states}")
    sdk_tries, caller_tries = 1 + 2, 3
    print(f"\n  wrapper attempts: {rc.sent} ({injector.injected} simulated failures) across {len(scenarios)} scenarios; "
          f"virtual time elapsed {clock.now:.2f}s")
    print(f"  retry-storm math: SDK default ({sdk_tries} tries) x wrapper ({rc.max_attempts}) x caller ({caller_tries}) = "
          f"{sdk_tries * rc.max_attempts * caller_tries} requests per user action -> keep ONE retry layer")


def static_prompt() -> str:
    rules = [f"Rule {i:02d}: keep summaries factual, cite the source section, avoid speculation, and use plain "
             f"language suitable for an operations audience (style guide item {i})." for i in range(1, 41)]
    return "You are the summarization service for operations reports.\n" + "\n".join(rules)


def step_checklist(client) -> None:
    section("4. Operational debugging checklist (as code)")
    print("  4a) max_tokens truncation")
    for max_tokens in (60, 1024):
        msg = client.messages.create(model=MODEL_HAIKU, max_tokens=max_tokens,
                                     system="You are the summarization service.",
                                     messages=[{"role": "user", "content": "Write a detailed postmortem summary of INC-2291."}])
        print(f"    max_tokens={max_tokens:<5} stop_reason={msg.stop_reason:<10} out={msg.usage.output_tokens:<4} -> {diagnose_response(msg)}")

    print("\n  4b) cache_read_input_tokens == 0 investigation (the v2.14.0 prompt template bug)")
    static = static_prompt()
    doc = "Document: weekly operations report for the summarizer service."
    stamps = ["2026-10-08T14:03:11Z", "2026-10-08T14:03:12Z"]
    groups = {
        "v7 timestamp first in system": [([{"type": "text", "text": f"Request time: {ts}\n{static}", "cache_control": {"type": "ephemeral"}}], doc) for ts in stamps],
        "v6 static system, time in user": [([{"type": "text", "text": static, "cache_control": {"type": "ephemeral"}}], f"Request time: {ts}\n{doc}") for ts in stamps],
    }
    for label, calls in groups.items():
        usages, cost = [], 0.0
        for system, user in calls:
            msg = client.messages.create(model=MODEL_SONNET, max_tokens=300, system=system, messages=[{"role": "user", "content": user}])
            usages.append(msg.usage)
            cost += cost_usd(msg.model, msg.usage)
            u = msg.usage
            print(f"    {label:<31} in={u.input_tokens:<4} cache_write={u.cache_creation_input_tokens or 0:<5} cache_read={u.cache_read_input_tokens or 0}")
        print(f"    -> {cache_check(usages)}  (2 calls: ${cost:.5f})")

    print("\n  4c) request linter: catch the 400s before sending")
    user = {"role": "user", "content": "Summarise the incident."}
    tool = {"name": "get_runbook", "description": "Find runbooks", "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}
    bad_requests: list[tuple[str, dict[str, Any]]] = [
        ("temperature on Opus 5.5", {"model": MODEL_OPUS, "max_tokens": 1024, "temperature": 0.2, "messages": [user]}),
        ("budget_tokens thinking", {"model": MODEL_OPUS, "max_tokens": 16000, "thinking": {"type": "enabled", "budget_tokens": 8000}, "messages": [user]}),
        ("thinking disabled on Sonnet 5.5", {"model": MODEL_SONNET, "max_tokens": 1024, "thinking": {"type": "disabled"}, "messages": [user]}),
        ("assistant prefill", {"model": MODEL_SONNET, "max_tokens": 512, "messages": [user, {"role": "assistant", "content": "{"}]}),
        ("forced tool_choice on Sonnet 5.5", {"model": MODEL_SONNET, "max_tokens": 512, "tools": [tool], "tool_choice": {"type": "any"}, "messages": [user]}),
        ("broken tool_use/tool_result pairing", {"model": MODEL_OPUS, "max_tokens": 1024, "tools": [tool], "messages": [
            user,
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "get_runbook", "input": {"topic": "429"}}]},
            {"role": "user", "content": [{"type": "text", "text": "here you go"}, {"type": "tool_result", "tool_use_id": "toolu_02", "content": "[]"}]},
        ]}),
        ("dotted model id", {"model": "claude-opus-5.5", "max_tokens": 1024, "messages": [user]}),
        ("correct request", {"model": MODEL_OPUS, "max_tokens": 4096, "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}, "messages": [user]}),
    ]
    for label, params in bad_requests:
        issues = lint_request(params)
        if not issues:
            print(f"    {label:<36} OK - no issues")
        for code, problem, fix in issues:
            print(f"    {label:<36} {code} {problem}")
            print(f"    {'':<36}      fix: {fix}")

    print("\n  4d) checklist: symptom -> check -> fix")
    for symptom, check, fix in CHECKLIST:
        print(f"    {symptom:<30} | {check:<62} | {fix}")


def render_report(report: IncidentReport, checks: list[tuple[str, bool, str]], stats: dict[str, Any]) -> list[str]:
    out = [f"INCIDENT REPORT - {report.title}",
           f"Severity: {report.severity}   Confidence: {report.confidence:.2f}   Runbook: {report.runbook_id} {report.runbook_url}"]
    for label, text in (("Impact", report.impact), ("Trigger", report.trigger), ("Root cause", report.root_cause)):
        out.append(f"{label}:")
        out += wrap(text)
    for label, items in (("Contributing factors", report.contributing_factors), ("Evidence", report.evidence)):
        out.append(f"{label}:")
        for item in items:
            lines = wrap(item, indent="      ")
            out += ["    - " + lines[0].strip()] + lines[1:]
    out.append("Timeline (UTC):")
    out += [f"    {t.time}  {t.event}" for t in report.timeline]
    for label, items in (("Immediate actions", report.immediate_actions), ("Follow-up actions", report.follow_up_actions)):
        out.append(f"{label}:")
        for i, item in enumerate(items, 1):
            lines = wrap(item, indent="       ")
            out += [f"    {i}. " + lines[0].strip()] + lines[1:]
    passed = sum(1 for _, ok, _ in checks if ok)
    out.append(f"Verification: {passed}/{len(checks)} grounding checks passed; agent used {stats['turns']} turns, "
               f"{stats['tool_calls']} tool calls, ${stats['cost']:.5f}")
    return out


def main() -> None:
    banner("D7.3 - Debugging and Operational Issue Resolution")
    records, unparsed = parse_log((DATA / "app.log.txt").read_text(encoding="utf-8"))
    metrics, deploys, runbooks = load_json("metrics.json"), load_json("deploys.json"), load_json("runbooks.json")
    schema = load_json("incident_report.schema.json")
    client = get_client(responder)

    facts = step_log_analysis(records, unparsed, metrics, deploys)
    report, stats = step_triage(client, TriageTools(records, metrics, deploys, runbooks), schema)
    checks: list[tuple[str, bool, str]] = []
    if report:
        checks = grounding_checks(report, facts, records, deploys, runbooks)
        print("\n  grounding checks (agent claims vs deterministic facts):")
        for name, ok, detail in checks:
            print(f"    [{'PASS' if ok else 'FAIL'}] {name:<36} {detail}")
    step_resilience(client)
    step_checklist(client)

    section("5. Incident report")
    if not report:
        print("  no report produced")
        return
    lines = render_report(report, checks, stats)
    for line in lines:
        print(f"  {line}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "incident_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (OUT / "incident_report.md").write_text("```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
    print("\n  wrote labs_output/incident_report.json and labs_output/incident_report.md")


if __name__ == "__main__":
    main()
