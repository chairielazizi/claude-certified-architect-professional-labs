"""
D6.3 - Stakeholder Feedback Loops and SLA Alignment

Service: the Claims Correspondence Copilot (Harbor Mutual, fictional), one month after go-live.

What this lab builds:
  (a) an SLI / SLO / SLA catalog for a Claude-powered service: availability, latency,
      eval-based quality (LLM judge on a sample, reported with a confidence interval),
      cost per completed task and rejection/escalation rate
  (b) a seeded simulation of a month of request logs -> SLIs, error budget, multiwindow
      burn-rate alerts, SLA breach detection and service credits (with SLA exclusions)
  (c) a stakeholder feedback loop: Claude (Haiku, structured output) classifies feedback,
      code corroborates themes against telemetry, maps them to backlog priorities and
      closes the loop with "you said, we did" release notes
  (d) expectation alignment: a monthly service review with SLO attainment, top issues,
      what changed, what's next, known limitations (measured) and what the system will NOT do

Principle: numbers come from telemetry and code; Claude classifies and writes the narrative.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import MODEL_HAIKU, MODEL_OPUS, PRICING, MockRequest, banner, get_client, section, usage_line  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
OUT = LAB / "labs_output"

THEMES = ["availability", "latency", "verbosity_tone", "factual_error", "missing_context", "policy_risk",
          "formatting", "feature_request", "praise"]
Theme = Literal["availability", "latency", "verbosity_tone", "factual_error", "missing_context", "policy_risk",
                "formatting", "feature_request", "praise"]


class FeedbackLabel(BaseModel):
    id: str
    theme: Theme
    severity: int = Field(ge=1, le=4, description="1 = cosmetic ... 4 = compliance risk or customer harm")
    sentiment: Literal["positive", "neutral", "negative"]
    component: Literal["prompt", "retrieval", "integration", "ui", "model", "process"]
    summary: str


class FeedbackClassification(BaseModel):
    items: list[FeedbackLabel]


class ServiceReviewNarrative(BaseModel):
    executive_summary: str
    highlights: list[str]
    concerns: list[str]
    expectation_notes: list[str]
    next_month_focus: list[str]


CLASSIFY_SYSTEM = """You triage user feedback for a Claude-powered letter-drafting service.
For each item return: one theme (availability, latency, verbosity_tone, factual_error, missing_context,
policy_risk, formatting, feature_request, praise), severity 1-4 (1 cosmetic, 4 compliance risk or customer harm),
sentiment, the component most likely responsible, and a one-line neutral summary. Return every id exactly once."""

NARRATIVE_SYSTEM = """You write the narrative sections of a monthly service review for business stakeholders.
Be candid: lead with what missed its target and why, then what was done about it.
Use ONLY numbers that appear in the FACTS block. Describe LLM quality probabilistically
(sample-based estimate with a confidence interval), never as a guarantee."""


def responder(req: MockRequest) -> str:
    mock = DATA / "mock_responses"
    if "triage user feedback" in req.system_text:
        return (mock / "feedback_classification.json").read_text(encoding="utf-8")
    if "monthly service review" in req.system_text:
        return (mock / "service_review_narrative.json").read_text(encoding="utf-8")
    return "{}"


# ---------------------------------------------------------------------------
# Deterministic helpers (identical in TypeScript)
# ---------------------------------------------------------------------------
def mulberry32(seed: int) -> Callable[[], float]:
    state = seed & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((state ^ (state >> 15)) * (state | 1)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return nxt


def nearest_rank(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = math.ceil(p / 100 * len(sorted_vals))
    return sorted_vals[max(1, k) - 1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval: honest uncertainty for a sampled pass rate (works for small n and p near 1)."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def spark(values: list[float], lo: float, hi: float) -> str:
    levels = " .:-=+*#"
    out = ""
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(levels) - 1)) if hi > lo else 0
        out += levels[max(0, min(len(levels) - 1, idx))]
    return out


NUMBER_PATTERNS = [r"\$\s?(\d[\d,]*(?:\.\d+)?)", r"(\d+(?:\.\d+)?)\s?%", r"(\d+(?:\.\d+)?)\s?pp\b",
                   r"(\d+(?:\.\d+)?)\s?(?:s|sec|seconds)\b"]


def ungrounded(text: str, facts: list[float]) -> list[str]:
    bad: list[str] = []
    for pat in NUMBER_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            v = float(m.group(1).replace(",", ""))
            if not any(abs(v - f) <= max(0.05, 0.005 * abs(f)) for f in facts):
                bad.append(m.group(1))
    return bad


# ---------------------------------------------------------------------------
# Simulation: one record per drafting request
# ---------------------------------------------------------------------------
@dataclass
class Req:
    day: int
    hour: int
    status: str  # ok | client_error | server_error | timeout | throttled
    cause: str  # service | dependency
    latency_s: float
    output_tokens: int
    cost: float
    action: str  # approved | edited | rejected | escalated | ""
    judged: bool
    score: int
    factual: bool
    v14: bool


def simulate(sim: dict[str, Any], cat: dict[str, Any]) -> list[Req]:
    rng = mulberry32(sim["seed"])
    b, inc, pc = sim["baseline"], sim["incident"], sim["prompt_change"]
    open_h, close_h = cat["business_hours"]["open"], cat["business_hours"]["close"]
    price = PRICING[sim["model"]]
    start_key, end_key = pc["start"][0] * 100 + pc["start"][1], pc["end"][0] * 100 + pc["end"][1]
    sample_rate = cat["judge"]["sample_rate"]
    reqs: list[Req] = []
    for day in range(1, cat["period"]["days"] + 1):
        weekday = (cat["period"]["start_weekday"] + day - 1) % 7  # 0 = Monday
        base = sim["weekend_requests"] if weekday >= 5 else sim["weekday_requests"]
        n = base + int(rng() * (2 * sim["jitter"] + 1)) - sim["jitter"]
        for _ in range(n):
            u = [rng() for _ in range(23)]  # fixed draw count per request keeps both languages in lockstep
            hour = open_h + int(u[0] * (close_h - open_h))
            v14 = start_key <= day * 100 + hour < end_key
            in_incident = day == inc["day"] and inc["start_hour"] <= hour < inc["start_hour"] + inc["hours"]
            p = pc if v14 else b
            status, cause = "ok", "service"
            if u[1] < b["p_client_error"]:
                status = "client_error"
            elif in_incident and u[2] < inc["error_rate"]:
                status, cause = "timeout", "dependency"
            elif u[3] < b["p_server_error"]:
                status = "server_error"
            elif u[3] < b["p_server_error"] + b["p_timeout"]:
                status = "timeout"
            elif u[3] < b["p_server_error"] + b["p_timeout"] + b["p_throttled"]:
                status = "throttled"
            if status != "ok":
                reqs.append(Req(day, hour, status, cause, 0.0, 0, 0.0, "", False, 0, False, v14))
                continue
            # Approx. log-normal latency using only + and * (Irwin-Hall normal, 2nd-order exp),
            # so Python and TypeScript produce bit-identical floats.
            z = sum(u[4:16]) - 6.0
            x = b["latency_sigma"] * z
            latency = b["latency_median_s"] * (1 + x + x * x / 2) * (pc["latency_mult"] if v14 else 1.0)
            in_tok = math.floor(b["input_tokens"] * (0.8 + 0.4 * u[16]) + 0.5)
            out_tok = math.floor(b["output_tokens"] * (0.8 + 0.4 * u[17]) * (pc["output_mult"] if v14 else 1.0) + 0.5)
            cost = (in_tok * price["input"] + b["cache_read_tokens"] * price["cache_read"]
                    + b["cache_write_tokens"] * price["cache_write"] + out_tok * price["output"]) / 1_000_000
            acc, action = 0.0, "escalated"
            for name, prob in p["actions"].items():
                acc += prob
                if u[18] < acc:
                    action = name
                    break
            judged = u[19] < sample_rate
            score, factual = 0, False
            if judged:
                if u[20] < p["judge_pass"]:
                    score = 5 if u[21] < 0.55 else 4
                else:
                    score = 3 if u[21] < 0.7 else 2 if u[21] < 0.9 else 1
                factual = u[22] < p["factual_error"]
            reqs.append(Req(day, hour, status, cause, latency, out_tok, cost, action, judged, score, factual, v14))
    return reqs


@dataclass
class Stats:
    total: int
    valid: int
    good: int
    bad: int
    availability: float
    ok: int
    fast_share: float
    p50: float
    p95: float
    p99: float
    judged: int
    passed: int
    q_point: float
    q_low: float
    q_high: float
    factual: int
    spend: float
    completed: int
    cost_per_letter: float
    escalation_rate: float
    mean_out_tokens: float


def stats(reqs: list[Req], threshold_s: float, judge_cost: float) -> Stats:
    valid = [r for r in reqs if r.status != "client_error"]
    good = [r for r in valid if r.status == "ok"]
    lat = sorted(r.latency_s for r in good)
    judged = [r for r in good if r.judged]
    passed = sum(1 for r in judged if r.score >= 4)
    q = wilson(passed, len(judged))
    spend = sum(r.cost for r in good) + judge_cost * len(judged)
    completed = sum(1 for r in good if r.action in ("approved", "edited"))
    esc = sum(1 for r in good if r.action in ("rejected", "escalated"))
    return Stats(
        total=len(reqs), valid=len(valid), good=len(good), bad=len(valid) - len(good),
        availability=len(good) / len(valid) if valid else 1.0, ok=len(good),
        fast_share=sum(1 for v in lat if v <= threshold_s) / len(lat) if lat else 1.0,
        p50=nearest_rank(lat, 50), p95=nearest_rank(lat, 95), p99=nearest_rank(lat, 99),
        judged=len(judged), passed=passed, q_point=q[0], q_low=q[1], q_high=q[2],
        factual=sum(1 for r in judged if r.factual), spend=spend, completed=completed,
        cost_per_letter=spend / completed if completed else 0.0,
        escalation_rate=esc / len(good) if good else 0.0,
        mean_out_tokens=sum(r.output_tokens for r in good) / len(good) if good else 0.0,
    )


def quality_status(s: Stats, target: float, min_n: int) -> str:
    if s.judged < min_n:
        return "LOW N"
    if s.q_low >= target:
        return "MET"
    if s.q_point >= target:
        return "AT RISK"
    return "MISSED"


def main() -> None:
    banner("D6.3 - Stakeholder Feedback Loops and SLA Alignment")
    cat = json.loads((DATA / "slo_catalog.json").read_text(encoding="utf-8"))
    sim = json.loads((DATA / "simulation.json").read_text(encoding="utf-8"))
    feedback = json.loads((DATA / "feedback.json").read_text(encoding="utf-8"))["items"]
    rel = json.loads((DATA / "releases.json").read_text(encoding="utf-8"))
    lim = json.loads((DATA / "limitations.json").read_text(encoding="utf-8"))
    slo = {s["kind"]: s for s in cat["slos"]}
    month = cat["period"]["month"]
    client = get_client(responder)

    # -- 1. Catalog -----------------------------------------------------------------
    section("1. SLI / SLO / SLA catalog for a Claude-powered service")
    for s in cat["slos"]:
        target = (pct(s["target"]) if "target" in s else f"<= ${s['target_max_usd']:.2f}" if "target_max_usd" in s
                  else f"<= {pct(s['target_max'], 0)}")
        sla = "internal SLO only" if s["sla"] is None else (
            f"SLA {pct(s['sla']['target'])}" if "target" in s["sla"] else f"SLA {s['sla']['metric']} <= {s['sla']['max_s']} s")
        print(f"  {s['id']} {s['name']:<30} target {target:<9} | {sla}")
        print(f"        SLI: {s['sli']}")
    print("  Why quality is different: an LLM has no single 'correct' output, so quality is an eval-based SLI -")
    print("  a calibrated judge scores a random sample against a rubric - and is reported as an estimate with")
    print("  a confidence interval. Promise 'at least 90% of sampled drafts meet the rubric', never 'always correct'.")

    # -- 2. Simulation --------------------------------------------------------------
    section("2. Simulated month of request logs (seeded)")
    reqs = simulate(sim, cat)
    jt = cat["judge"]["tokens_per_judgement"]
    jp = PRICING[cat["judge"]["model"]]
    judge_cost = (jt["input"] * jp["input"] + jt["output"] * jp["output"]) / 1_000_000
    counts: dict[str, int] = {}
    for r in reqs:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"  {len(reqs):,} drafting requests over {cat['period']['days']} days ({cat['period']['label']}), "
          f"business hours {cat['business_hours']['open']}:00-{cat['business_hours']['close']}:00 {cat['business_hours']['tz']}")
    print("  Status mix: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    inc, pc = sim["incident"], sim["prompt_change"]
    print(f"  Injected: {inc['id']} on {month} {inc['day']} {inc['start_hour']}:00 for {inc['hours']} h ({inc['cause']})")
    print(f"  Injected: prompt {pc['version']} live {month} {pc['start'][0]} {pc['start'][1]}:00 -> {month} {pc['end'][0]} "
          f"{pc['end'][1]}:00 (longer outputs, slower, lower judge scores)")

    # -- 3. SLIs and attainment --------------------------------------------------------
    section("3. SLIs and SLO attainment (month and weeks)")
    thr = slo["latency"]["threshold_s"]
    m = stats(reqs, thr, judge_cost)
    q_target, min_n = slo["quality"]["target"], slo["quality"]["min_samples"]
    q_status = quality_status(m, q_target, min_n)
    scorecard = [
        ("SLO-1", "Availability", pct(m.availability, 2), pct(slo["availability"]["target"]),
         "MET" if m.availability >= slo["availability"]["target"] else "MISSED"),
        ("SLO-2", f"Drafts <= {thr} s", pct(m.fast_share), pct(slo["latency"]["target"]),
         "MET" if m.fast_share >= slo["latency"]["target"] else "MISSED"),
        ("SLO-3", "Quality (judge >= 4/5)", f"{pct(m.q_point)} [{pct(m.q_low)}-{pct(m.q_high)}] n={m.judged}",
         pct(q_target), q_status),
        ("SLO-4", "Cost per completed letter", f"${m.cost_per_letter:.4f}", f"<= ${slo['cost']['target_max_usd']:.2f}",
         "MET" if m.cost_per_letter <= slo["cost"]["target_max_usd"] else "MISSED"),
        ("SLO-5", "Rejection + escalation", pct(m.escalation_rate), f"<= {pct(slo['escalation']['target_max'], 0)}",
         "MET" if m.escalation_rate <= slo["escalation"]["target_max"] else "MISSED"),
    ]
    print(f"  {'SLO':<6}{'Objective':<27}{'Actual':<31}{'Target':<9}Status")
    for row in scorecard:
        print(f"  {row[0]:<6}{row[1]:<27}{row[2]:<31}{row[3]:<9}{row[4]}")
    print(f"  Latency distribution: p50 {m.p50:.2f} s, p95 {m.p95:.2f} s, p99 {m.p99:.2f} s; "
          f"spend ${m.spend:,.2f} for {m.completed:,} completed letters")

    weeks = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 30)]
    weekly: list[tuple[str, Stats]] = []
    print(f"\n  {'Week':<12}{'avail':>8}{'<=8s':>8}{'p95':>7}{'quality [95% CI]':>24}{'n':>5}{'rej+esc':>9}{'$/letter':>10}")
    for a, b in weeks:
        w = stats([r for r in reqs if a <= r.day <= b], thr, judge_cost)
        label = f"{month} {a}-{b}"
        weekly.append((label, w))
        flags = []
        if w.fast_share < slo["latency"]["target"]:
            flags.append("latency")
        if w.q_point < q_target:
            flags.append("quality")
        if w.escalation_rate > slo["escalation"]["target_max"]:
            flags.append("rejections")
        if w.availability < slo["availability"]["target"]:
            flags.append("availability")
        print(f"  {label:<12}{pct(w.availability, 2):>8}{pct(w.fast_share):>8}{w.p95:>6.2f}s"
              f"{pct(w.q_point) + ' [' + pct(w.q_low) + '-' + pct(w.q_high) + ']':>24}{w.judged:>5}"
              f"{pct(w.escalation_rate):>9}{w.cost_per_letter:>10.4f}" + (f"  <- {', '.join(flags)}" if flags else ""))
    daily = [stats([r for r in reqs if r.day == d], thr, judge_cost) for d in range(1, cat["period"]["days"] + 1)]
    print(f"\n  Daily p95 latency   {spark([s.p95 for s in daily], 5.0, 10.0)}  (scale 5 s ' ' .. 10 s '#')")
    print(f"  Daily availability  {spark([s.availability for s in daily], 0.95, 1.0)}  (scale 95% ' ' .. 100% '#')")
    print(f"                      {''.join(str(d % 10) for d in range(1, cat['period']['days'] + 1))}  (day of month)")

    # -- 4. Error budget + burn alerts -------------------------------------------------
    section("4. Error budget and multiwindow burn-rate alerts")
    av_target = slo["availability"]["target"]
    allowed = (1 - av_target) * m.valid
    consumed = m.bad / allowed
    incident_bad = sum(1 for r in reqs if r.cause == "dependency")
    incident_share = incident_bad / allowed
    print(f"  Budget: (1 - {av_target}) x {m.valid:,} valid requests = {allowed:.1f} failed requests allowed")
    print(f"  Spent: {m.bad} failed -> {pct(consumed, 0)} of budget ({'EXHAUSTED' if consumed >= 1 else 'remaining ' + pct(1 - consumed, 0)}); "
          f"{inc['id']} alone used {pct(incident_share, 0)}")
    hours_list = [(d, h) for d in range(1, cat["period"]["days"] + 1)
                  for h in range(cat["business_hours"]["open"], cat["business_hours"]["close"])]
    bucket_valid = {k: 0 for k in hours_list}
    bucket_bad = {k: 0 for k in hours_list}
    for r in reqs:
        if r.status == "client_error":
            continue
        bucket_valid[(r.day, r.hour)] += 1
        if r.status != "ok":
            bucket_bad[(r.day, r.hour)] += 1
    v_seq = [bucket_valid[k] for k in hours_list]
    b_seq = [bucket_bad[k] for k in hours_list]

    guard = cat["burn_alert_guard"]

    def window(i: int, length: int) -> tuple[int, int, float]:
        """(valid, failed, burn rate) over the last `length` business hours ending at bucket i."""
        lo = max(0, i - length + 1)
        v, bd = sum(v_seq[lo:i + 1]), sum(b_seq[lo:i + 1])
        return v, bd, ((bd / v) / (1 - av_target) if v else 0.0)

    def burn(i: int, length: int) -> float:
        return window(i, length)[2]

    episodes: list[tuple[str, str, str, str, int, float]] = []
    suppressed = 0
    for rule in cat["burn_alerts"]:
        run_start, peak = -1, 0.0
        for i in range(len(hours_list) + 1):
            firing = False
            if i < len(hours_list) and i + 1 >= rule["long_h"]:  # warm-up: evaluate only full windows
                v_long, f_long, b_long = window(i, rule["long_h"])
                hot = b_long >= rule["threshold"] and burn(i, rule["short_h"]) >= rule["threshold"]
                # Low-traffic guard: 1 failure in a 4-request weekend hour is a 50x "burn" but not an incident.
                enough = v_long >= guard["min_requests"] and f_long >= guard["min_failures"]
                if hot and not enough:
                    suppressed += 1
                firing = hot and enough
            if firing:
                if run_start < 0:
                    run_start, peak = i, 0.0
                peak = max(peak, burn(i, rule["long_h"]))
            elif run_start >= 0:
                d0, h0 = hours_list[run_start]
                d1, h1 = hours_list[i - 1]
                episodes.append((rule["severity"], rule["name"], f"{month} {d0} {h0:02d}:00", f"{month} {d1} {h1 + 1:02d}:00",
                                 i - run_start, peak))
                run_start = -1
    print(f"  Rules (business-hour windows): " + "; ".join(
        f"{r['severity']} {r['name']}: burn >= {r['threshold']:g} over {r['long_h']}h and {r['short_h']}h" for r in cat["burn_alerts"]))
    for sev, name, start, end, n_h, peak in episodes:
        print(f"  {sev:<6} {name:<12} {start} -> {end} ({n_h} h firing, peak burn {peak:.1f}x)")
    print(f"  Suppressed {suppressed} hot window-hours with < {guard['min_requests']} requests or < {guard['min_failures']} failures (low-traffic noise)")
    triggers = {"budget_exhausted": consumed >= 1, "incident_over_20pct": incident_share > 0.2, "quality_not_met": q_status != "MET"}
    policy_actions = [(p["id"], p["action"]) for p in cat["error_budget_policy"] if triggers[p["trigger"]]]
    print("  Error-budget policy triggered:")
    for pid, action in policy_actions:
        print(f"    {pid}: {action}")

    # -- 5. SLA ----------------------------------------------------------------------
    section("5. SLA breach detection and service credits")
    excluded = incident_bad if inc["sla_excluded"] else 0
    sla_valid = m.valid - excluded
    sla_avail = (m.good) / sla_valid
    sla_av = slo["availability"]["sla"]
    av_credit = max([c["credit_pct"] for c in sla_av["credits"] if sla_avail < c["below"]], default=0)
    lat_sla = slo["latency"]["sla"]
    lat_credit = lat_sla["credit_pct"] if m.p95 > lat_sla["max_s"] else 0
    terms = cat["sla_terms"]
    credit_pct = min(terms["credit_cap_pct"], av_credit + lat_credit)
    credit_usd = terms["monthly_fee_usd"] * credit_pct / 100
    print(f"  SLA availability excludes {excluded} requests failed by the customer-owned dependency ({inc['id']}):")
    print(f"    {m.good:,} / {sla_valid:,} = {pct(sla_avail, 2)} vs SLA {pct(sla_av['target'])} -> "
          f"{'BREACH' if sla_avail < sla_av['target'] else 'met'} (credit {av_credit}%)")
    print(f"  SLA latency: monthly p95 {m.p95:.2f} s vs <= {lat_sla['max_s']} s -> "
          f"{'BREACH' if lat_credit else 'met'} (credit {lat_credit}%)")
    print(f"  Service credit: {credit_pct}% of ${terms['monthly_fee_usd']:,} = ${credit_usd:,.2f}")
    slo_missed = m.availability < av_target
    sla_breached = sla_avail < sla_av["target"]
    if slo_missed and not sla_breached:
        print(f"  Lesson: the SLO (user experience, {pct(m.availability, 2)}) was missed while the SLA (contract, {pct(sla_avail, 2)}) held -")
        print("  the SLO is the internal early-warning line; the SLA sits below it and excludes what the provider can't control.")
    else:
        print(f"  SLO {'missed' if slo_missed else 'met'}, SLA {'breached' if sla_breached else 'met'}.")

    # -- 6. Feedback loop ----------------------------------------------------------------
    section("6. Feedback loop: Claude classifies, code corroborates and prioritizes")
    fb_text = "\n".join(f"{f['id']} | {month} {f['day']} | {f['channel']} | {f['role']} | {f['text']}" for f in feedback)
    resp = client.messages.parse(
        model=MODEL_HAIKU,  # high-volume classification: the small model is the right tool
        max_tokens=4000,
        output_config={"effort": "low"},
        system=CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": "Classify these feedback items:\n" + fb_text}],
        output_format=FeedbackClassification,
    )
    labels = resp.parsed_output
    if labels is None:
        raise RuntimeError("classification did not parse")
    print(f"  Classified {len(labels.items)} items with {MODEL_HAIKU} | {usage_line(resp)}")
    ids = [x.id for x in labels.items]
    expected = [f["id"] for f in feedback]
    print(f"  Validation: missing={[i for i in expected if i not in ids] or 'none'}, "
          f"duplicates={sorted({i for i in ids if ids.count(i) > 1}) or 'none'}")
    by_fb = {f["id"]: f for f in feedback}
    label_of = {x.id: x for x in labels.items}

    median_tokens = sorted(s.mean_out_tokens for s in daily)[len(daily) // 2]
    signal_days = {
        "availability": {d + 1 for d, s in enumerate(daily) if s.availability < av_target},
        "latency": {d + 1 for d, s in enumerate(daily) if s.p95 > thr},
        "verbosity_tone": {d + 1 for d, s in enumerate(daily) if s.mean_out_tokens > 1.2 * median_tokens},
        "factual_error": {d + 1 for d, s in enumerate(daily) if s.factual > 0},
    }
    print("  Telemetry signal days: " + "; ".join(f"{k} {sorted(v)}" for k, v in signal_days.items()))

    def corroborated(fid: str) -> bool:
        lab = label_of[fid]
        days = signal_days.get(lab.theme)
        return days is not None and (by_fb[fid]["day"] in days or by_fb[fid]["day"] - 1 in days)

    def loop_status(fid: str) -> tuple[str, str]:
        lab = label_of[fid]
        if lab.theme == "praise":
            return "acknowledged", "-"
        for r in rel["releases"]:
            if lab.theme in r["addresses_themes"] and r["day"] >= by_fb[fid]["day"]:
                return "closed", f"{r['version']} ({month} {r['day']})"
        return "open", rel["theme_backlog"][lab.theme]["id"]

    theme_rows: list[dict[str, Any]] = []
    for th in THEMES:
        items = [x for x in labels.items if x.theme == th]
        if not items:
            continue
        corr = sum(1 for x in items if corroborated(x.id))
        has_signal = th in signal_days
        open_items = [x for x in items if loop_status(x.id)[0] == "open"]
        theme_rows.append({
            "theme": th, "n": len(items), "sev": sum(x.severity for x in items), "corr": corr, "has_signal": has_signal,
            "theme_corroborated": has_signal and corr * 2 >= len(items),
            "open": open_items, "submitters": len({by_fb[x.id]["submitter"] for x in items}),
        })
    print(f"\n  {'theme':<16}{'items':>6}{'sev':>5}{'corroborated':>14}{'open':>6}")
    for t in theme_rows:
        corr_txt = f"{t['corr']}/{t['n']}" if t["has_signal"] else "n/a"
        print(f"  {t['theme']:<16}{t['n']:>6}{t['sev']:>5}{corr_txt:>14}{len(t['open']):>6}")

    backlog: list[dict[str, Any]] = []
    for t in theme_rows:
        if t["theme"] == "praise" or not t["open"]:
            continue
        bl = rel["theme_backlog"][t["theme"]]
        open_sev = sum(x.severity for x in t["open"])
        uniq = len({by_fb[x.id]["submitter"] for x in t["open"]})
        score = open_sev * (1.5 if t["theme_corroborated"] else 1.0) * (1 + 0.1 * (uniq - 1)) / bl["effort"]
        override = t["theme"] in rel["risk_override_themes"] and any(x.severity == 4 for x in t["open"])
        backlog.append({"theme": t["theme"], "id": bl["id"], "title": bl["title"], "effort": bl["effort"],
                        "score": score, "override": override, "items": [x.id for x in t["open"]]})
    backlog.sort(key=lambda x: (0 if x["override"] else 1, -x["score"], THEMES.index(x["theme"])))
    for i, b in enumerate(backlog):
        b["priority"] = "P1" if i < 2 else "P2" if i < 4 else "P3"
    print("\n  Backlog (score = open severity x corroboration x reach / effort; compliance risk overrides):")
    for b in backlog:
        print(f"  {b['priority']} {b['id']} score {b['score']:5.2f}{' [risk override]' if b['override'] else '':<16} "
              f"{b['title'][:70]}  <- {', '.join(b['items'])}")

    # -- 7. Close the loop ---------------------------------------------------------------
    section("7. Closing the loop: release notes and 'you said, we did'")
    closed = [f["id"] for f in feedback if loop_status(f["id"])[0] == "closed"]
    actionable = [f["id"] for f in feedback if label_of[f["id"]].theme != "praise"]
    for r in rel["releases"]:
        fixed = [f["id"] for f in feedback if loop_status(f["id"]) == ("closed", f"{r['version']} ({month} {r['day']})")]
        print(f"  {r['version']} ({month} {r['day']}, {r['type']}): {r['summary']}")
        if fixed:
            print(f"      closes feedback: {', '.join(fixed)}")
    print(f"  Loop closure: {len(closed)}/{len(actionable)} actionable items closed this month "
          f"({pct(len(closed) / len(actionable), 0)}); the rest are mapped to backlog items with an owner.")
    print("  'You said, we did' notices go to every submitter whose item closed (and a status note to the rest).")

    # -- 8. Monthly service review -----------------------------------------------------
    section("8. Monthly service review (Claude narrative from computed facts)")
    w3 = weekly[2][1]
    facts_lines = [
        f"Availability {pct(m.availability, 2)} vs SLO {pct(av_target)} ({'MISSED' if slo_missed else 'met'}: {m.bad} failed of {m.valid} valid requests); "
        f"error budget consumed {pct(consumed, 0)}; incident {inc['id']} used {pct(incident_share, 0)} of the budget",
        f"SLA availability {pct(sla_avail, 2)} vs {pct(sla_av['target'])} ({'BREACHED' if sla_breached else 'met'}; dependency outage excluded); service credit ${credit_usd:,.2f}",
        f"Latency: {pct(m.fast_share)} of drafts within {thr} s (SLO {pct(slo['latency']['target'])}, met); p50 {m.p50:.2f} s, p95 {m.p95:.2f} s, p99 {m.p99:.2f} s",
        f"Week {weekly[2][0]}: {pct(w3.fast_share)} within {thr} s, p95 {w3.p95:.2f} s, quality {pct(w3.q_point)}, rejections+escalations {pct(w3.escalation_rate)} (prompt {pc['version']})",
        f"Quality: {pct(m.q_point)} of {m.judged} sampled drafts scored >= 4/5, 95% CI {pct(m.q_low)}-{pct(m.q_high)}, target {pct(q_target)} -> {q_status}",
        f"Cost per completed letter ${m.cost_per_letter:.4f} (target <= ${slo['cost']['target_max_usd']:.2f}); total spend ${m.spend:,.2f}",
        f"Rejection + escalation rate {pct(m.escalation_rate)} (target <= {pct(slo['escalation']['target_max'], 0)})",
        f"Feedback: {len(feedback)} items, {len(closed)} of {len(actionable)} actionable items closed ({pct(len(closed) / len(actionable), 0)})",
        "Top backlog: " + "; ".join(f"{b['priority']} {b['id']} {b['title']}" for b in backlog[:3]),
        "Releases: " + "; ".join(f"{r['version']} {month} {r['day']}: {r['summary']}" for r in rel["releases"]),
    ]
    fq = wilson(m.factual, m.judged)
    facts_lines.append(f"Factual-error flags: {m.factual} of {m.judged} sampled drafts ({pct(fq[0])}, 95% CI {pct(fq[1])}-{pct(fq[2])})")
    facts_text = "\n".join(facts_lines)
    facts_num: list[float] = []
    for line in facts_lines:
        for pat in NUMBER_PATTERNS + [r"(\d[\d,]*(?:\.\d+)?)"]:
            for mm in re.finditer(pat, line, flags=re.IGNORECASE):
                facts_num.append(float(mm.group(1).replace(",", "")))
    resp = client.messages.parse(
        model=MODEL_OPUS, max_tokens=3000, thinking={"type": "adaptive"}, output_config={"effort": "medium"},
        system=NARRATIVE_SYSTEM,
        messages=[{"role": "user", "content": "FACTS:\n" + facts_text + "\n\nWrite the narrative sections."}],
        output_format=ServiceReviewNarrative,
    )
    narrative = resp.parsed_output
    if narrative is None:
        raise RuntimeError("narrative did not parse")
    all_text = "\n".join([narrative.executive_summary, *narrative.highlights, *narrative.concerns,
                          *narrative.expectation_notes, *narrative.next_month_focus])
    bad_nums = ungrounded(all_text, facts_num)
    print(f"  Narrative drafted | {usage_line(resp)}")
    print(f"  Grounding check: {'all figures trace to telemetry' if not bad_nums else 'UNGROUNDED: ' + ', '.join(bad_nums)}")

    report = render_review(cat, sim, rel, lim, feedback, label_of, m, weekly, scorecard, allowed, consumed,
                           incident_share, episodes, policy_actions, sla_avail, excluded, credit_pct, credit_usd,
                           theme_rows, backlog, loop_status, narrative, fq, closed, actionable)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"service_review_{cat['period']['label'].replace(' ', '_').lower()}.md"
    (OUT / name).write_text(report, encoding="utf-8")
    (OUT / "service_review_facts.txt").write_text(facts_text + "\n", encoding="utf-8")
    print(f"  wrote labs_output/{name} ({len(report.splitlines())} lines) and labs_output/service_review_facts.txt; samples in examples/")


def render_review(cat, sim, rel, lim, feedback, label_of, m: Stats, weekly, scorecard, allowed, consumed, incident_share,
                  episodes, policy_actions, sla_avail, excluded, credit_pct, credit_usd, theme_rows, backlog, loop_status,
                  narrative: ServiceReviewNarrative, fq, closed, actionable) -> str:
    slo = {s["kind"]: s for s in cat["slos"]}
    month = cat["period"]["month"]
    inc, pc = sim["incident"], sim["prompt_change"]
    L: list[str] = []
    a = L.append
    a(f"# Monthly Service Review - {cat['service']} - {cat['period']['label']}")
    a("")
    a(f"**Provider:** {cat['provider']} | **Customer:** {cat['customer']}")
    a("")
    a("## 1. Executive summary")
    a("")
    a(narrative.executive_summary)
    a("")
    a("**Highlights**")
    a("")
    for x in narrative.highlights:
        a(f"- {x}")
    a("")
    a("**Concerns**")
    a("")
    for x in narrative.concerns:
        a(f"- {x}")
    a("")
    a("## 2. SLO and SLA scorecard")
    a("")
    a("| SLO | Objective | Actual | Target | Status |")
    a("|---|---|---|---|---|")
    for row in scorecard:
        a(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | **{row[4]}** |")
    a("")
    sla_av = slo["availability"]["sla"]
    lat_sla = slo["latency"]["sla"]
    a("| SLA term | Actual | Commitment | Result |")
    a("|---|---|---|---|")
    a(f"| Availability (excl. {excluded} dependency-caused failures) | {pct(sla_avail, 2)} | >= {pct(sla_av['target'])} | "
      f"{'BREACH' if sla_avail < sla_av['target'] else 'met'} |")
    a(f"| Monthly p95 full-draft latency | {m.p95:.2f} s | <= {lat_sla['max_s']} s | {'BREACH' if m.p95 > lat_sla['max_s'] else 'met'} |")
    a(f"| **Service credit** | {credit_pct}% | fee ${cat['sla_terms']['monthly_fee_usd']:,} | **${credit_usd:,.2f}** |")
    a("")
    a("SLA exclusions: " + "; ".join(cat["sla_terms"]["exclusions"]) + ".")
    a("")
    a("## 3. Error budget and incidents")
    a("")
    a(f"- Budget: {allowed:.1f} failed requests allowed ({pct(1 - slo['availability']['target'])} of {m.valid:,} valid); "
      f"{m.bad} failed -> **{pct(consumed, 0)} consumed**.")
    a(f"- {inc['id']} ({month} {inc['day']}, {inc['start_hour']}:00 for {inc['hours']} h): {inc['cause']}; used {pct(incident_share, 0)} of the budget.")
    a("")
    a("| Alert | Rule | Window | Duration | Peak burn |")
    a("|---|---|---|---|---:|")
    for sev, name, start, end, n_h, peak in episodes:
        a(f"| {sev} | {name} | {start} -> {end} | {n_h} h | {peak:.1f}x |")
    a("")
    a("Error-budget policy actions triggered:")
    a("")
    for pid, action in policy_actions:
        a(f"- **{pid}** {action}")
    a("")
    a("## 4. Weekly trend")
    a("")
    a("| Week | Availability | Drafts <= 8 s | p95 | Quality [95% CI] (n) | Rejected + escalated | $/letter |")
    a("|---|---:|---:|---:|---|---:|---:|")
    for label, w in weekly:
        a(f"| {label} | {pct(w.availability, 2)} | {pct(w.fast_share)} | {w.p95:.2f} s | "
          f"{pct(w.q_point)} [{pct(w.q_low)}-{pct(w.q_high)}] ({w.judged}) | {pct(w.escalation_rate)} | ${w.cost_per_letter:.4f} |")
    a("")
    a(f"Weekly quality samples are below the {slo['quality']['min_samples']}-sample minimum, so weekly quality is indicative only; "
      "the monthly figure is the one we commit to.")
    a("")
    a("## 5. How quality is measured (and how to read it)")
    a("")
    a(f"- SLI: {slo['quality']['sli']}.")
    a(f"- Judge: {cat['judge']['model']} on a {pct(cat['judge']['sample_rate'], 0)} random sample. {cat['judge']['calibration']}")
    a(f"- Status rule: MET if the lower bound of the 95% Wilson interval is >= target; AT RISK if only the point estimate is; MISSED otherwise.")
    for x in narrative.expectation_notes:
        a(f"- {x}")
    a("")
    a("## 6. Stakeholder feedback")
    a("")
    a("| Theme | Items | Severity sum | Corroborated by telemetry | Open |")
    a("|---|---:|---:|---|---:|")
    for t in theme_rows:
        corr_txt = f"{t['corr']}/{t['n']}" if t["has_signal"] else "n/a (no telemetry signal)"
        a(f"| {t['theme']} | {t['n']} | {t['sev']} | {corr_txt} | {len(t['open'])} |")
    a("")
    a("| Feedback | Day | Theme | Sev | Summary | Loop status |")
    a("|---|---|---|---:|---|---|")
    for f in feedback:
        lab = label_of[f["id"]]
        st, ref = loop_status(f["id"])
        a(f"| {f['id']} | {month} {f['day']} | {lab.theme} | {lab.severity} | {lab.summary} | {st} {ref if ref != '-' else ''}".rstrip() + " |")
    a("")
    a("## 7. What changed this month - you said, we did")
    a("")
    for r in rel["releases"]:
        fixed = [f["id"] for f in feedback if loop_status(f["id"]) == ("closed", f"{r['version']} ({month} {r['day']})")]
        a(f"- **{r['version']}** ({month} {r['day']}, {r['type']}): {r['summary']}"
          + (f" _You said: {', '.join(fixed)} - we did: this release._" if fixed else ""))
    a("")
    a(f"Loop closure: {len(closed)}/{len(actionable)} actionable items closed ({pct(len(closed) / len(actionable), 0)}).")
    a("")
    a("## 8. What's next")
    a("")
    a("| Priority | Item | Score | Feedback |")
    a("|---|---|---:|---|")
    for b in backlog:
        a(f"| {b['priority']}{' (risk override)' if b['override'] else ''} | {b['id']} {b['title']} | {b['score']:.2f} | {', '.join(b['items'])} |")
    a("")
    for x in narrative.next_month_focus:
        a(f"- {x}")
    a("")
    a("## 9. Known limitations (measured)")
    a("")
    for k in lim["known_limitations"]:
        metric = ""
        if k["metric"] == "factual_error_rate":
            metric = f" April: {m.factual} of {m.judged} sampled drafts flagged ({pct(fq[0])}, 95% CI {pct(fq[1])}-{pct(fq[2])})."
        elif k["metric"] == "quality_ci":
            metric = f" April: {pct(m.q_point)} with a 95% CI of {pct(m.q_low)}-{pct(m.q_high)}."
        elif k["metric"] == "latency_tail":
            metric = f" April: p50 {m.p50:.2f} s, p99 {m.p99:.2f} s."
        a(f"- **{k['id']} {k['title']}.** {k['detail']}{metric}")
    a("")
    a("## 10. What the system will NOT do")
    a("")
    for x in lim["will_not"]:
        a(f"- {x}")
    a("")
    a("## Appendix - SLI definitions")
    a("")
    a("| SLO | SLI | Window |")
    a("|---|---|---|")
    for s in cat["slos"]:
        a(f"| {s['id']} {s['name']} | {s['sli']} | {s['window']} |")
    a("")
    a("_Narrative sections drafted by Claude from the computed facts; every figure was checked against telemetry. "
      "All tables are computed from the request logs by code._")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
