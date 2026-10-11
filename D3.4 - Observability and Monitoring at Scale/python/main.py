"""
D3.4 - Observability and Monitoring at Scale
Exam objective: Analyze observability challenges and select monitoring strategies at scale.

What this lab builds
--------------------
A seeded simulation of one day of LLM traffic (10,000 requests across 8 tenants,
4 features, 3 models, 2,000 users) with four injected incidents:
    INC1 latency spike on Sonnet (10:00-11:00)
    INC2 cost runaway from an agent loop (tenant initech, from 15:00)
    INC3 rising refusal rate (tenant umbrella, feature extract, from 16:00)
    INC4 silent quality drop after a prompt deploy (feature summarize, from 20:00)
Then it applies - and compares - the monitoring strategies an architect must choose between:
    histogram aggregation (p50/p95/p99) vs averages
    bounded metric cardinality + exemplars (user_id is NOT a metric label)
    head-based vs tail-based trace sampling
    W3C trace context across agent steps, tools and subagents (+ loop detection)
    per-tenant cost attribution + anomaly alerts
    SLO burn-rate alerts vs static thresholds (alert fatigue)
    sampled LLM-as-judge quality monitoring (real calls through the shared SDK client)
This lab is about SCALE and STRATEGY SELECTION; D4.6 covers instrumenting a single app.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    PRICING,
    MockRequest,
    banner,
    cost_usd,
    estimate_tokens,
    get_async_client,
    is_live,
    section,
    text_block,
    text_of,
)

SEED = 3404
N_REQUESTS = 10_000
MINUTES = 1440
N_USERS = 2000
TENANTS = [("acme", 0.24), ("globex", 0.18), ("initech", 0.14), ("umbrella", 0.12),
           ("hooli", 0.10), ("stark", 0.09), ("wayne", 0.07), ("wonka", 0.06)]
FEATURES = [("chat", 0.45), ("summarize", 0.25), ("extract", 0.18), ("agent", 0.12)]
FEATURE_MODEL = {"chat": MODEL_SONNET, "summarize": MODEL_SONNET, "extract": MODEL_HAIKU, "agent": MODEL_OPUS}
TOKENS = {"chat": (1200, 300), "summarize": (4000, 250), "extract": (2500, 150), "agent": (6000, 600)}
LAT = {MODEL_HAIKU: (250.0, 3.0), MODEL_SONNET: (500.0, 7.0), MODEL_OPUS: (900.0, 12.0)}  # ttft ms, ms/token
SLO_LATENCY_MS = {"chat": 8000, "summarize": 15000, "extract": 5000, "agent": 600000}  # agents run async
TOOL_MS = 200.0
CONTEXT_GROWTH = 800  # tokens appended to the agent context per looping step
ERRORS = {"overloaded_529", "api_error_500", "rate_limit_429"}
SHORT = {MODEL_HAIKU: "haiku", MODEL_SONNET: "sonnet", MODEL_OPUS: "opus"}


# ---------------------------------------------------------------------------
# Deterministic helpers (identical in the TypeScript version)
# ---------------------------------------------------------------------------
class Mulberry32:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def hash01(key: str, salt: str) -> float:
    """Deterministic [0,1) from an id: the basis of consistent (trace-id) sampling."""
    return int(sha(salt + key)[:8], 16) / 4294967296.0


def weighted(u: float, items: list[tuple[str, float]]) -> str:
    cum = 0.0
    for name, w in items:
        cum += w
        if u < cum:
            return name
    return items[-1][0]


def hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def pct_exact(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------------------
# 1. Traffic simulation with injected incidents
# ---------------------------------------------------------------------------
@dataclass
class Req:
    idx: int
    minute: int
    trace_id: str
    tenant: str
    feature: str
    model: str
    user: str
    steps: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str
    cost: float
    good: bool   # hidden ground truth for answer quality
    loop: bool   # ground truth: INC2 agent loop

    @property
    def slow(self) -> bool:
        return self.latency_ms > SLO_LATENCY_MS[self.feature]

    @property
    def bad(self) -> bool:  # SLO "bad event": server error or too slow (refusals are valid responses)
        return self.status in ERRORS or self.slow


def diurnal(minute: int) -> float:
    return 0.3 + 0.7 * max(0.0, 1 - abs(minute - 840) / 600)


def simulate(rng: Mulberry32) -> list[Req]:
    reqs: list[Req] = []
    for idx in range(N_REQUESTS):
        while True:  # rejection sampling against a diurnal curve (peak 14:00)
            m = int(rng.next() * MINUTES)
            if rng.next() < diurnal(m):
                break
        tenant = weighted(rng.next(), TENANTS)
        feature = weighted(rng.next(), FEATURES)
        model = FEATURE_MODEL[feature]
        user = f"u{int(rng.next() * N_USERS):04d}"
        base_in, base_out = TOKENS[feature]
        in_tok = int(base_in * (0.6 + 0.8 * rng.next()))
        out_tok = int(base_out * (0.5 + rng.next()))
        u_steps = rng.next()
        loop = feature == "agent" and tenant == "initech" and m >= 900          # INC2
        steps = 1 if feature != "agent" else (25 + int(u_steps * 20) if loop else 3 + int(u_steps * 5))
        ttft, decode = LAT[model]
        per_call = (ttft + decode * out_tok) * (0.85 + 0.3 * rng.next())
        u_tail, u_tail2, u_spike = rng.next(), rng.next(), rng.next()
        if u_tail < 0.01:
            per_call *= 2 + 2 * u_tail2                                          # heavy tail
        latency = per_call * steps + TOOL_MS * (steps - 1)
        if model == MODEL_SONNET and 600 <= m < 660:
            latency *= 3 + 2 * u_spike                                           # INC1
        total_in = steps * in_tok + (CONTEXT_GROWTH * steps * (steps + 1) // 2 if loop else 0)
        total_out = steps * out_tok
        cost = (total_in * PRICING[model]["input"] + total_out * PRICING[model]["output"]) / 1e6
        p_refusal = 0.005
        if tenant == "umbrella" and feature == "extract" and m >= 960:
            p_refusal = 0.005 + 0.495 * min(1.0, (m - 960) / 60)                 # INC3 (bad deploy)
        u = rng.next()
        if u < 0.001:
            status = "overloaded_529"
        elif u < 0.0015:
            status = "api_error_500"
        elif u < 0.002:
            status = "rate_limit_429"
        elif u < 0.002 + p_refusal:
            status = "refusal"
        elif u < 0.005 + p_refusal:
            status = "max_tokens"
        else:
            status = "ok"
        p_good = 0.45 if (feature == "summarize" and m >= 1200) else 0.93        # INC4
        good = rng.next() < p_good
        reqs.append(Req(idx, m, sha(f"trace-{idx}")[:32], tenant, feature, model, user, steps,
                        total_in, total_out, latency, status, cost, good, loop))
    return reqs


# ---------------------------------------------------------------------------
# 2. Histogram aggregation
# ---------------------------------------------------------------------------
BUCKETS = [100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]


def histogram(values: list[float]) -> list[int]:
    counts = [0] * (len(BUCKETS) + 1)
    for v in values:
        i = 0
        while i < len(BUCKETS) and v > BUCKETS[i]:
            i += 1
        counts[i] += 1
    return counts


def pct_hist(counts: list[int], p: float) -> float:
    """Percentile from bucket counts with linear interpolation (how metric backends do it)."""
    rank = p / 100 * sum(counts)
    cum = 0
    for i, c in enumerate(counts):
        if c and cum + c >= rank:
            lower = BUCKETS[i - 1] if i > 0 else 0
            if i == len(BUCKETS):
                return float(lower)
            return lower + (BUCKETS[i] - lower) * (rank - cum) / c
        cum += c
    return float(BUCKETS[-1])


# ---------------------------------------------------------------------------
# 5. Trace model (W3C trace context)
# ---------------------------------------------------------------------------
@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    start_ms: float
    dur_ms: float
    attrs: dict[str, Any] = field(default_factory=dict)


def build_agent_trace(r: Req) -> list[Span]:
    spans: list[Span] = []

    def add(name: str, parent: str | None, start: float, dur: float, attrs: dict[str, Any]) -> str:
        sid = sha(f"span-{r.idx}-{len(spans)}")[:16]
        spans.append(Span(sid, parent, name, start, dur, attrs))
        return sid

    root = add("POST /v1/agent", None, 0.0, r.latency_ms, {"tenant": r.tenant, "feature": r.feature})
    orch = add("agent.orchestrator", root, 2.0, r.latency_ms - 4, {"gen_ai.request.model": r.model})
    per_call = (r.latency_ms - TOOL_MS * (r.steps - 1)) / r.steps
    tools = ["search_orders{order_id=O-77}", "get_shipment{order_id=O-77}", "draft_reply{ticket=T-9}"]
    t = 4.0
    for i in range(r.steps):
        add(f"llm.call {SHORT[r.model]}", orch, t, per_call,
            {"step": i + 1, "anthropic.request_id": "req_" + sha(f"req-{r.idx}-{i}")[:12]})
        t += per_call
        if i < r.steps - 1:
            tool = tools[0] if r.loop else tools[i % len(tools)]
            add(f"tool.{tool}", orch, t, TOOL_MS, {})
            t += TOOL_MS
    return spans


# ---------------------------------------------------------------------------
# 7. Alerting: static threshold vs multi-window burn rate
# ---------------------------------------------------------------------------
SLO_TARGET = 0.995
BUDGET = 1 - SLO_TARGET


def episodes(firing: list[tuple[int, bool]]) -> list[tuple[int, int]]:
    """Collapse per-evaluation booleans into (start_minute, end_minute) alert episodes."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    last = 0
    for minute, on in firing:
        if on and start is None:
            start = minute
        if not on and start is not None:
            out.append((start, last))
            start = None
        last = minute
    if start is not None:
        out.append((start, last))
    return out


# ---------------------------------------------------------------------------
# 8. LLM-as-judge (real calls through the shared client)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are a strict faithfulness judge. Compare the SUMMARY with the SOURCE. "
    "A summary is faithful only if every number and claim in it is supported by the source."
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithful": {"type": "boolean"},
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "issue": {"type": "string"},
    },
    "required": ["faithful", "score", "issue"],
    "additionalProperties": False,
}


def make_doc(r: Req) -> tuple[str, str]:
    h = bytes.fromhex(sha(f"doc-{r.idx}"))
    q, growth, revenue, churn = 1 + h[0] % 4, 5 + h[1] % 20, 10 + h[2] % 90, 1 + h[3] % 9
    source = f"In Q{q}, revenue grew {growth}% to ${revenue}M while churn fell to {churn}%. The board approved a buyback."
    shown_growth = growth if r.good else growth + 7  # an unfaithful summary hallucinates a number
    summary = f"Q{q}: revenue up {shown_growth}% to ${revenue}M; churn down to {churn}%."
    return source, summary


def judge_responder(req: MockRequest) -> Any:
    text = req.last_user_text
    src = re.search(r"<source>(.*?)</source>", text, re.S)
    summ = re.search(r"<summary>(.*?)</summary>", text, re.S)
    nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s))  # noqa: E731
    unsupported = sorted(nums(summ.group(1) if summ else "") - nums(src.group(1) if src else ""), key=float)
    if unsupported:
        verdict = {"faithful": False, "score": 2, "issue": f"summary states {', '.join(unsupported)} which the source does not support"}
    else:
        verdict = {"faithful": True, "score": 5, "issue": "none"}
    body = json.dumps(verdict)
    return {"content": [text_block(body)], "usage": {"output_tokens": estimate_tokens(body)}}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
async def main() -> None:
    banner("D3.4 - Observability and Monitoring at Scale")

    section("1. Simulated day of traffic (seeded) with 4 injected incidents")
    reqs = simulate(Mulberry32(SEED))
    total_cost = sum(r.cost for r in reqs)
    n_err = sum(r.status in ERRORS for r in reqs)
    print(f"{len(reqs):,} requests | {len({r.tenant for r in reqs})} tenants | {len({r.feature for r in reqs})} features | "
          f"{len({r.model for r in reqs})} models | {len({r.user for r in reqs}):,} distinct users")
    print(f"LLM calls (agent steps included): {sum(r.steps for r in reqs):,} | tokens in/out: "
          f"{sum(r.input_tokens for r in reqs) / 1e6:.1f}M / {sum(r.output_tokens for r in reqs) / 1e6:.2f}M | "
          f"cost ${total_cost:,.2f} | server errors {n_err} | refusals {sum(r.status == 'refusal' for r in reqs)}")
    print("Incidents: INC1 Sonnet latency 10:00-11:00 | INC2 initech agent loop from 15:00 |")
    print("           INC3 umbrella/extract refusals from 16:00 | INC4 summarize quality drop from 20:00")

    section("2. Aggregation: averages hide tails; histograms give mergeable percentiles")
    print(f"{'MODEL':<7} {'N':>5} {'MEAN':>7} | {'P50 exact/hist':>15} | {'P95 exact/hist':>15} | {'P99 exact/hist':>17}")
    for model in (MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS):
        lat = [r.latency_ms for r in reqs if r.model == model]
        h = histogram(lat)
        cells = [f"{pct_exact(lat, p):>7.0f}/{pct_hist(h, p):<7.0f}" for p in (50, 95, 99)]
        print(f"{SHORT[model]:<7} {len(lat):>5} {sum(lat) / len(lat):>7.0f} | {cells[0]:>15} | {cells[1]:>15} | {cells[2]:>17}")
    print(f"A histogram is {len(BUCKETS) + 1} counters per series regardless of volume, and histograms from many")
    print("hosts/minutes can be summed before computing percentiles - averages and per-host p95s cannot.")

    section("3. Cardinality: what may be a metric label")
    base = {(r.tenant, r.feature, r.model, r.status) for r in reqs}
    with_user = {(r.tenant, r.feature, r.model, r.status, r.user) for r in reqs}
    statuses = {r.status for r in reqs}
    print(f"labels {{tenant, feature, model, status}}        -> {len(base):>6,} series observed (max {8 * 4 * 3 * len(statuses):,})")
    print(f"... + user_id                                -> {len(with_user):>6,} series observed today")
    print(f"... + user_id at 200k monthly users          -> up to {8 * 4 * 3 * len(statuses) * 200_000:,} series")
    print("Rule: metric labels must be bounded (tenant tier, feature, model, status). Put user_id, trace_id and")
    print("request_id in logs/traces, and link metrics to traces with EXEMPLARS:")
    spike = [r for r in reqs if r.model == MODEL_SONNET and 600 <= r.minute < 660]
    worst: dict[int, Req] = {}
    for r in spike:
        i = histogram([r.latency_ms]).index(1)
        if i not in worst or r.latency_ms > worst[i].latency_ms:
            worst[i] = r
    top = max(worst)
    ex = worst[top]
    print(f"  sonnet 10:00-11:00 slowest bucket ({BUCKETS[top - 1]:,}-{BUCKETS[top]:,} ms] exemplar: trace_id={ex.trace_id}"
          f" ({ex.latency_ms:,.0f} ms, tenant={ex.tenant}, user={ex.user})")

    section("4. Trace sampling: head-based vs tail-based")
    cats = {
        "server errors": lambda r: r.status in ERRORS,
        "refusals": lambda r: r.status == "refusal",
        "SLO-slow": lambda r: r.slow,
        "agent-loop (INC2)": lambda r: r.loop,
        "cost > $1": lambda r: r.cost > 1.0,
    }
    head = {r.idx for r in reqs if hash01(r.trace_id, "head") < 0.05}

    def tail_keep(r: Req) -> bool:
        # Decided AFTER the trace completes (needs a buffering collector): keep every
        # error/refusal/slow/expensive/long trace + a 2% baseline of normal traffic.
        return (r.status in ERRORS or r.status == "refusal" or r.slow or r.cost > 1.0 or r.steps > 15
                or hash01(r.trace_id, "tail") < 0.02)

    tail = {r.idx for r in reqs if tail_keep(r)}
    print(f"{'POLICY':<24} {'KEPT':>6} {'VOLUME':>7} | " + " | ".join(f"{c:>17}" for c in cats))
    for name, kept in (("head 5% (at trace start)", head), ("tail (after completion)", tail)):
        cov = []
        for fn in cats.values():
            matching = [r for r in reqs if fn(r)]
            got = sum(r.idx in kept for r in matching)
            cov.append(f"{got:>4}/{len(matching):<4} {got / len(matching) * 100:>5.0f}%" if matching else "n/a")
        print(f"{name:<24} {len(kept):>6} {len(kept) / len(reqs) * 100:>6.1f}% | " + " | ".join(f"{c:>17}" for c in cov))
    print("Head sampling is cheap and stateless but keeps ~5% of the traces you need most. Tail sampling keeps")
    print("all of them at similar volume, at the price of buffering whole traces before deciding.")

    section("5. Distributed trace correlation across agent steps")
    loop_req = next(r for r in reqs if r.loop)
    spans = build_agent_trace(loop_req)
    by_id = {s.span_id: s for s in spans}

    def depth(s: Span) -> int:
        return 0 if s.parent_id is None else 1 + depth(by_id[s.parent_id])

    print(f"trace_id={loop_req.trace_id} tenant={loop_req.tenant} started {hhmm(loop_req.minute)}, {len(spans)} spans:")
    for s in spans[:7]:
        extra = f" request_id={s.attrs['anthropic.request_id']}" if "anthropic.request_id" in s.attrs else ""
        print(f"  {'  ' * depth(s)}- {s.name:<34} span={s.span_id} +{s.start_ms:>9,.0f}ms {s.dur_ms:>9,.0f}ms{extra}")
    print(f"  {'  ' * 2}... {len(spans) - 7} more spans")
    llm = spans[2]
    print(f"Propagate to tools/subagents as a W3C header: traceparent: 00-{loop_req.trace_id}-{llm.span_id}-01")
    counts: dict[str, int] = {}
    for s in spans:
        if s.name.startswith("tool."):
            counts[s.name] = counts.get(s.name, 0) + 1
    tool, n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    print(f"Loop detector: {tool} called {n}x with identical args in one trace (steps={loop_req.steps}) -> LOOP SUSPECTED")

    section("6. Cost attribution per tenant (from usage tokens)")
    hourly: dict[str, list[float]] = {t: [0.0] * 24 for t, _ in TENANTS}
    for r in reqs:
        hourly[r.tenant][r.minute // 60] += r.cost
    print(f"{'TENANT':<9} {'REQS':>5} {'COST':>9} {'SHARE':>6} {'$/HR 00-15':>10} {'$/HR 15-24':>10} {'AGENT P99 STEPS':>16}  ANOMALY")
    anomaly_first: dict[str, int] = {}
    for t, _ in TENANTS:
        rs = [r for r in reqs if r.tenant == t]
        cost = sum(r.cost for r in rs)
        before, after = sum(hourly[t][:15]) / 15, sum(hourly[t][15:]) / 9
        steps = [float(r.steps) for r in rs if r.feature == "agent"]
        flag = ""
        for h in range(6, 24):
            window = sorted(hourly[t][h - 6:h])
            baseline = (window[2] + window[3]) / 2  # trailing 6h median
            if hourly[t][h] > 3 * baseline and hourly[t][h] > 2.0:
                anomaly_first[t] = h
                flag = f"cost {hourly[t][h] / baseline:.1f}x 6h median at {h:02d}:00"
                break
        print(f"{t:<9} {len(rs):>5} {cost:>9.2f} {cost / total_cost * 100:>5.1f}% {before:>10.2f} {after:>10.2f} "
              f"{pct_exact(steps, 99) if steps else 0:>16.0f}  {flag}")
    print("Guardrails: per-tenant budgets + anomaly alerts, a max_steps cap per agent run, per-tenant rate limits,")
    print("and (for hard isolation) separate workspaces/keys whose spend limits and usage reports attribute cost natively.")

    section("7. Alerting: static threshold vs SLO burn rate (SLO 99.5% good events)")
    tot = [0] * MINUTES
    bad = [0] * MINUTES
    for r in reqs:
        tot[r.minute] += 1
        bad[r.minute] += r.bad
    ptot = [0] * (MINUTES + 1)
    pbad = [0] * (MINUTES + 1)
    for m in range(MINUTES):
        ptot[m + 1] = ptot[m] + tot[m]
        pbad[m + 1] = pbad[m] + bad[m]

    def rate(end: int, window: int) -> float:
        lo = max(0, end + 1 - window)
        n = ptot[end + 1] - ptot[lo]
        return (pbad[end + 1] - pbad[lo]) / n if n else 0.0

    evals = list(range(60, MINUTES, 5))
    static = [(m, rate(m, 5) > 0.02) for m in evals]
    page = [(m, rate(m, 60) / BUDGET > 14.4 and rate(m, 5) / BUDGET > 14.4) for m in evals]
    ticket = [(m, rate(m, 360) / BUDGET > 6 and rate(m, 30) / BUDGET > 6) for m in evals if m >= 360]  # full 6h window
    day_bad = sum(bad) / sum(tot)
    print(f"Day: {sum(bad)} bad events / {sum(tot):,} = {day_bad * 100:.2f}% (budget 0.50%, burn {day_bad / BUDGET:.1f}x)")
    for label, rule in (("static: bad rate > 2% over 5m", static),
                        ("PAGE: burn > 14.4x over 1h AND 5m", page),
                        ("TICKET: burn > 6x over 6h AND 30m", ticket)):
        eps = episodes(rule)
        shown = ", ".join(f"{hhmm(a)}-{hhmm(b)}" for a, b in eps[:4]) + (" ..." if len(eps) > 4 else "")
        print(f"  {label:<36} {sum(on for _, on in rule):>3} firing evals, {len(eps):>3} episodes: {shown or '-'}")
    print("The static rule fires all day on random blips (alert fatigue) - the real incident is buried.")
    print("Burn-rate alerts fire on budget consumption speed: one page for INC1, nothing for noise.")
    print("Note: INC2-INC4 never breach the availability SLO - cost, refusal and quality need their own monitors.")

    # Refusal-rate monitor: per-segment ratio alert with a minimum volume and a confidence bound.
    seg_n: dict[tuple[str, str], list[int]] = {(t, f): [0] * 24 for t, _ in TENANTS for f, _ in FEATURES}
    seg_k: dict[tuple[str, str], list[int]] = {(t, f): [0] * 24 for t, _ in TENANTS for f, _ in FEATURES}
    for r in reqs:
        seg_n[(r.tenant, r.feature)][r.minute // 60] += 1
        seg_k[(r.tenant, r.feature)][r.minute // 60] += r.status == "refusal"
    refusal_alert: tuple[str, str, int, int, int] | None = None
    for end_h in range(3, 25):  # evaluate hourly over a trailing 3h window
        for seg in seg_n:
            n, k = sum(seg_n[seg][end_h - 3:end_h]), sum(seg_k[seg][end_h - 3:end_h])
            if refusal_alert is None and n >= 20 and wilson(k, n)[0] > 0.05:
                refusal_alert = (seg[0], seg[1], end_h * 60, k, n)
    before = [r for r in reqs if r.minute < 960]
    after = [r for r in reqs if r.minute >= 960]
    rate_before = sum(r.status == "refusal" for r in before) / len(before)
    rate_after = sum(r.status == "refusal" for r in after) / len(after)
    print(f"Refusal rate across ALL traffic: {rate_before * 100:.1f}% before 16:00 vs {rate_after * 100:.1f}% after"
          f" - easy to miss on a global dashboard.")
    if refusal_alert:
        t, f, end, k, n = refusal_alert
        print(f"Segment monitor (tenant x feature, 3h window, n>=20, Wilson lower bound > 5%): {t}/{f} at {hhmm(end)}"
              f" - {k}/{n} refused ({k / n * 100:.0f}%)")

    section("8. Quality: sampled LLM-as-judge (real calls via the shared client)")
    sample: list[Req] = []
    for hour in range(24):
        in_hour = [r for r in reqs if r.feature == "summarize" and r.minute // 60 == hour]
        in_hour.sort(key=lambda r: hash01(r.trace_id, "judge"))
        sample.extend(sorted(in_hour[:6], key=lambda r: r.idx))  # stratified: 6 per hour
    client = get_async_client(judge_responder)
    sem = asyncio.Semaphore(8)

    async def judge(r: Req) -> tuple[Req, bool, str, float]:
        source, summary = make_doc(r)
        async with sem:
            resp = await client.messages.create(
                model=MODEL_SONNET,
                max_tokens=300,
                system=JUDGE_SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                messages=[{"role": "user", "content": f"<source>{source}</source>\n<summary>{summary}</summary>"}],
            )
        if resp.stop_reason != "end_turn":
            return r, False, f"judge stopped: {resp.stop_reason}", 0.0
        verdict = json.loads(text_of(resp))
        return r, bool(verdict["faithful"]), verdict["issue"], cost_usd(MODEL_SONNET, resp.usage)

    verdicts = await asyncio.gather(*[judge(r) for r in sample])
    judge_cost = sum(v[3] for v in verdicts)
    print(f"Judged {len(verdicts)} of {sum(r.feature == 'summarize' for r in reqs):,} summarize responses "
          f"(6/hour, stratified) for ${judge_cost:.4f}. Example verdicts:")
    for r, ok, issue, _ in [v for v in verdicts if v[0].minute >= 1200][:2] + [verdicts[0]]:
        print(f"  {hhmm(r.minute)} trace={r.trace_id[:12]}.. faithful={str(ok).lower():<5} issue={issue}")
    print(f"{'WINDOW':<12} {'N':>3} {'PASS':>5} {'RATE':>6}  {'95% WILSON CI':<15} STATUS")
    judge_alert = None
    for w in range(0, 24, 2):
        vs = [v for v in verdicts if w <= v[0].minute // 60 < w + 2]
        k = sum(v[1] for v in vs)
        lo, hi = wilson(k, len(vs))
        alert = hi < 0.85  # even the optimistic bound is below the 85% quality target
        if alert and judge_alert is None:
            judge_alert = w
        print(f"{w:02d}:00-{w + 2:02d}:00  {len(vs):>3} {k:>5} {k / len(vs) * 100 if vs else 0:>5.0f}%  "
              f"[{lo * 100:>3.0f}%, {hi * 100:>3.0f}%]      {'ALERT' if alert else 'ok'}")
    batch = [{"custom_id": f"judge-{r.idx}", "params": {
        "model": MODEL_SONNET, "max_tokens": 300, "system": JUDGE_SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        "messages": [{"role": "user", "content": "<source>{0}</source>\n<summary>{1}</summary>".format(*make_doc(r))}]}}
        for r in sample]
    if is_live():
        mb = await client.messages.batches.create(requests=batch)  # type: ignore[arg-type]
        print(f"Submitted nightly judge batch {mb.id} ({len(batch)} requests, 50% price).")
    else:
        print(f"[live-only] client.messages.batches.create(requests=<{len(batch)} judge requests>) - the Batches API")
        print("            halves judge cost; results arrive asynchronously (within 24h), fine for trend monitoring.")

    section("Incident detection matrix (time to detect)")
    static_eps = episodes(static)
    burn_eps = sorted(episodes(page) + episodes(ticket))

    def static_cell(start: int, end: int) -> str:
        k = sum(1 for a, b in static_eps if a <= end and b >= start)
        return f"{k} of {len(static_eps)} episodes overlap" if k else "silent"

    def burn_cell(start: int, end: int) -> str:
        first = next((a for a, _ in burn_eps if start <= a <= end), None)
        return f"{first - start} min" if first is not None else "missed"

    rows = [
        ("INC1 Sonnet latency spike", 600, 660, "-", "-"),
        ("INC2 initech agent loop", 900, MINUTES,
         f"{anomaly_first['initech'] * 60 + 60 - 900} min (cost anomaly)" if "initech" in anomaly_first else "missed", "-"),
        ("INC3 umbrella refusals", 960, MINUTES,
         f"{refusal_alert[2] - 960} min (refusal monitor)" if refusal_alert else "missed", "-"),
        ("INC4 summarize quality", 1200, MINUTES, "missed",
         f"<= {(judge_alert + 2) * 60 - 1200} min (judge window)" if judge_alert is not None else "missed"),
    ]
    print(f"{'INCIDENT':<27} {'STATIC THRESHOLD':<27} {'BURN-RATE':<10} {'COST/REFUSAL MONITOR':<30} {'LLM JUDGE'}")
    for name, start, end, c, j in rows:
        print(f"{name:<27} {static_cell(start, end):<27} {burn_cell(start, end):<10} {c:<30} {j}")
    print("A static threshold that fires on noise all day is not a detector even when it 'fires' during an incident.")

    section("Monitoring strategy recommendations")
    table = [
        ("Tail latency at volume", "Histograms per model/feature; p95/p99 SLIs", "usage + latency per call; TTFT when streaming"),
        ("Alert fatigue", "Multi-window SLO burn-rate alerts", "page on fast burn, ticket on slow burn"),
        ("High-cardinality dims", "Bounded labels; user/trace ids in logs + exemplars", "never user_id as a metric label"),
        ("Trace storage cost", "Tail-based sampling at the collector", "keep 100% errors/slow/costly/refusals + 2%"),
        ("Multi-step agents", "W3C traceparent through tools/subagents", "span per LLM call/tool; log request_id"),
        ("Runaway cost", "Per-tenant attribution, budgets, anomalies", "usage tokens x price; max_steps; workspace limits"),
        ("Refusals / truncation", "stop_reason ratio monitors (min volume)", "refusal, max_tokens, pause_turn per tenant"),
        ("Silent quality drift", "Stratified LLM-as-judge, calibrated", "Batches API for cost; compare pre/post deploy"),
        ("Rate limits / overload", "429/529 + retry rates per workspace", "anthropic-ratelimit-* headers, retry-after"),
    ]
    print(f"{'CHALLENGE':<24} {'STRATEGY':<51} SIGNAL / HOOK")
    for c, s, h in table:
        print(f"{c:<24} {s:<51} {h}")


if __name__ == "__main__":
    asyncio.run(main())
