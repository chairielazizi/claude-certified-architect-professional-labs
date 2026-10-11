"""
D1.6 - Aligning Solutions to Business Value Pillars

A value calculator for three customer-support AI options, scored on five pillars:

  efficiency      hours saved, cycle time (first response time)
  productivity    tickets resolved per FTE
  transformation  new capability / revenue (risk-adjusted)
  cost            Claude API cost per task (from token usage via cost_usd) + infra + people -> ROI, payback
  performance     SLA feasibility per model choice: p95 latency, availability, quality

Output: SLA-driven model choice, pillar table, weighted scorecard, KPI tree, and a Claude-drafted
executive summary. All arithmetic is plain code - the LLM only writes prose from computed numbers.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_OPUS,
    MockRequest,
    banner,
    cost_usd,
    get_client,
    section,
    text_block,
    text_of,
    usage_line,
)

DATA = Path(__file__).resolve().parents[1] / "data"
VM: dict[str, Any] = json.loads((DATA / "value_model.json").read_text(encoding="utf-8"))
BASE = VM["baseline"]
ASSUME = VM["assumptions"]
SHORT = {"claude-haiku-5-5": "Haiku", "claude-sonnet-5-5": "Sonnet", "claude-opus-5-5": "Opus"}


def usd(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


# ---------------------------------------------------------------------------
# Performance SLAs: deterministic latency simulation + composite availability
# ---------------------------------------------------------------------------
class LCG:
    """Tiny linear congruential generator: identical sequence in Python and TypeScript."""

    def __init__(self, seed: int) -> None:
        self.state = seed % 4294967296

    def next(self) -> float:
        self.state = (1664525 * self.state + 1013904223) % 4294967296
        return self.state / 4294967296


def latency_percentiles(model: str, reply_tokens: int, tool_ms: float) -> tuple[float, float]:
    """p50/p95 seconds for one customer-visible reply: TTFT + generation + tool time, with tail noise.

    Same seed for every candidate (common random numbers) so differences come from the model, not noise.
    Planning profile only - replace with your own load-test measurements (D3.3, D6.3).
    """
    ttft, per_tok = VM["latency_profiles_ms"][model]
    base = ttft + per_tok * reply_tokens + tool_ms
    rng = LCG(ASSUME["latency_seed"])
    samples = []
    for _ in range(ASSUME["latency_samples"]):
        u1, u2, u3 = rng.next(), rng.next(), rng.next()
        factor = 0.85 + 0.30 * u1 + (0.9 * u3 if u2 < 0.10 else 0.0)   # 10% of calls hit a slow tail
        samples.append(base * factor)
    samples.sort()
    n = len(samples)
    p50 = samples[(50 * n + 99) // 100 - 1]
    p95 = samples[(95 * n + 99) // 100 - 1]                            # nearest-rank percentile
    return p50 / 1000, p95 / 1000


def composite_availability(option: dict[str, Any], with_fallback: bool) -> float:
    """Serial chain = product of availabilities; a redundant model path = 1 - (1 - a)^2."""
    avail = 1.0
    for comp in option["architecture"]:
        a = VM["component_availability"][comp]
        if comp == "model_api" and with_fallback:
            a = 1 - (1 - a) ** 2          # e.g. second region / second platform for the same model
        avail *= a
    return avail


@dataclass
class SlaResult:
    model: str
    p50: float
    p95: float
    availability: float
    quality: float
    passes: dict[str, bool]

    @property
    def all_pass(self) -> bool:
        return all(self.passes.values())


def check_sla(option: dict[str, Any], model: str) -> SlaResult:
    sla = option["sla"]
    p50, p95 = latency_percentiles(model, option["reply_tokens"], option["tool_ms"])
    avail = composite_availability(option, option["model_fallback"])
    quality = option["pilot_quality"][model]
    passes = {"latency": p95 <= sla["p95_latency_s"], "availability": avail >= sla["availability"],
              "quality": quality >= sla["quality"]}
    return SlaResult(model, p50, p95, avail, quality, passes)


# ---------------------------------------------------------------------------
# Value pillars
# ---------------------------------------------------------------------------
@dataclass
class Pillars:
    option_id: str
    model: str
    hours_saved_month: float
    frt_after_h: float
    aht_after_min: float
    tickets_per_fte_before: float
    tickets_per_fte_after: float
    productivity_uplift: float
    new_value_risk_adj_year: float
    api_cost_per_ticket: float
    api_month: float
    run_month: float
    benefit_month: float
    net_month: float
    payback_months: float
    roi_year1: float
    net_3yr: float
    cost_per_resolved_before: float
    cost_per_resolved_after: float


def compute_pillars(option: dict[str, Any], model: str, cost_per_call: float) -> Pillars:
    t, aht, rate = BASE["tickets_per_month"], BASE["aht_min"], BASE["loaded_cost_per_hour"]
    hours_fte = BASE["productive_hours_per_fte_month"]
    d, r = option["deflection"], option["aht_reduction"]
    # Efficiency: automated tickets save the whole AHT; assisted tickets save a share of it.
    saved_min_per_ticket = d * aht + (1 - d) * aht * r
    hours_saved = t * saved_min_per_ticket / 60
    base_hours = t * aht / 60
    human_hours = base_hours - hours_saved
    aht_after = (1 - d) * aht * (1 - r)                       # blended human minutes per ticket
    frt_after = BASE["frt_hours"] * (1 - option["frt_improvement"])
    # Productivity: tickets resolved per FTE actually needed.
    tpf_before = t / (base_hours / hours_fte)
    tpf_after = t / (human_hours / hours_fte)
    # Transformation: new value, haircut by confidence (it is the least certain pillar).
    new_value = option["new_value_usd_year"] * ASSUME["transformation_confidence"]
    # Cost: API cost from measured token usage x calls per ticket x volume, plus infra and people.
    api_per_ticket = cost_per_call * option["calls_per_ticket"]
    api_month = api_per_ticket * t
    run_month = api_month + option["infra_usd_month"] + option["people_usd_month"]
    # Only part of freed capacity becomes money (attrition, avoided hiring) -> realization rate.
    benefit_month = hours_saved * rate * ASSUME["realization_rate"] + new_value / 12
    net_month = benefit_month - run_month
    payback = option["build_usd"] / net_month if net_month > 0 else float("inf")
    year_cost = 12 * run_month + option["build_usd"]
    roi = (12 * benefit_month - year_cost) / year_cost
    net_3yr = 36 * net_month - option["build_usd"]
    cpr_before = base_hours * rate / t
    cpr_after = (human_hours * rate + run_month) / t
    return Pillars(option["id"], model, hours_saved, frt_after, aht_after, tpf_before, tpf_after,
                   tpf_after / tpf_before - 1, new_value, api_per_ticket, api_month, run_month, benefit_month,
                   net_month, payback, roi, net_3yr, cpr_before, cpr_after)


# ---------------------------------------------------------------------------
# Offline responder
# ---------------------------------------------------------------------------
OPTIONS = {o["id"]: o for o in VM["options"]}
PILOT_REPLIES = {
    "A": "Draft reply: Sorry about the double charge - I've refunded the duplicate $49.00; it will show in 5 business days.",
    "B": "Resolved: duplicate charge refunded (ref R-2291). Customer notified by email; ticket closed.",
    "C": '{"queue": "billing-ops", "priority": "P2", "language": "en"}',
}


def responder(req: MockRequest) -> Any:
    match = re.search(r"PILOT option=(\w)", req.last_user_text)
    if match:
        # Offline the mock reports the token usage measured in the pilot (production-size prompts with a
        # cached system prompt + knowledge base). Live, usage comes from your real prompt.
        oid = match.group(1)
        return {"content": [text_block(PILOT_REPLIES[oid])], "usage": dict(OPTIONS[oid]["pilot_usage"])}
    if "SCORECARD:" in req.last_user_text:
        p = json.loads(req.last_user_text.split("SCORECARD:", 1)[1])
        top, runner = p["ranking"][0], p["ranking"][1]
        return (f"Recommendation: proceed with Option {top['id']} ({top['name']}) on {top['model']}, ranked first at "
                f"{top['score']}/5 ahead of Option {runner['id']} ({runner['score']}/5).\n"
                f"Value: frees {top['hours_saved']} agent hours a month, lifts tickets per FTE by "
                f"{top['productivity']}, and adds {top['new_value']} of risk-adjusted new value a year.\n"
                f"Economics: year-1 ROI {top['roi']}, payback in {top['payback']} months, 3-year net value "
                f"{top['net_3yr']}; Claude API spend is only {top['api_month']} a month - people and platform "
                f"dominate run cost.\n"
                f"Performance: {top['model']} is the cheapest model that meets every SLA ({p['sla_note']}).\n"
                f"Risks: transformation value is 50% confidence-weighted; the 99.9% availability target needs "
                f"multi-AZ hosting plus a model fallback path. Next step: 6-week pilot with the KPI tree as the "
                f"success contract.")
    return "OK"


def main() -> None:
    banner("D1.6 - Aligning Solutions to Business Value Pillars")
    client = get_client(responder)

    section("1. Baseline and solution options")
    print(f"  baseline: {BASE['tickets_per_month']:,} tickets/month, AHT {BASE['aht_min']} min, "
          f"${BASE['loaded_cost_per_hour']}/agent-hour, first response {BASE['frt_hours']:g} h, CSAT {BASE['csat']}")
    print(f"  assumptions: realization rate {pct(ASSUME['realization_rate'])} of freed hours, "
          f"transformation value weighted at {pct(ASSUME['transformation_confidence'])} confidence")
    for o in VM["options"]:
        print(f"  {o['id']}  {o['name']:<42} deflect={pct(o['deflection']):>4} AHT-{pct(o['aht_reduction']):>4} "
              f"build={usd(o['build_usd'])} candidates={','.join(SHORT[m] for m in o['candidates'])}")

    section("2. Pilot token usage -> Claude API cost per task (cost_usd)")
    cost_per_call: dict[tuple[str, str], float] = {}
    for o in VM["options"]:
        for model in o["candidates"]:
            msg = client.messages.create(
                model=model, max_tokens=500,
                system="You are the production prompt for this option (system prompt + knowledge base are cached).",
                messages=[{"role": "user", "content": f"PILOT option={o['id']}: customer says they were charged twice."}],
            )
            c = cost_usd(msg.model, msg.usage)
            cost_per_call[(o["id"], model)] = c
            per_ticket = c * o["calls_per_ticket"]
            print(f"  {o['id']} {SHORT[model]:<6} {usage_line(msg)}")
            print(f"           x {o['calls_per_ticket']:g} calls/ticket = ${per_ticket:.5f}/ticket -> "
                  f"{usd(per_ticket * BASE['tickets_per_month'])}/month API")

    section("3. Performance SLA feasibility per model choice")
    print(f"  {'opt':<4}{'model':<8}{'p50 s':>7}{'p95 s':>7}{'target':>8}{'avail':>9}{'target':>9}"
          f"{'quality':>8}{'target':>7}  verdict")
    chosen: dict[str, str] = {}
    sla_results: dict[str, SlaResult] = {}
    all_sla: dict[tuple[str, str], SlaResult] = {}
    for o in VM["options"]:
        feasible = []
        for model in o["candidates"]:
            s = check_sla(o, model)
            all_sla[(o["id"], model)] = s
            fails = [k for k, ok in s.passes.items() if not ok]
            verdict = "PASS" if s.all_pass else "FAIL " + "+".join(fails)
            print(f"  {o['id']:<4}{SHORT[model]:<8}{s.p50:>7.2f}{s.p95:>7.2f}{o['sla']['p95_latency_s']:>8.1f}"
                  f"{s.availability * 100:>8.2f}%{o['sla']['availability'] * 100:>8.2f}%{s.quality:>8.2f}"
                  f"{o['sla']['quality']:>7.2f}  {verdict}")
            if s.all_pass:
                feasible.append((cost_per_call[(o["id"], model)], model, s))
        if not feasible:
            print(f"  {o['id']}: no candidate meets the SLAs -> redesign (renegotiate SLA, change architecture)")
            continue
        _, model, s = min(feasible)                     # cheapest model that meets every SLA
        chosen[o["id"]] = model
        sla_results[o["id"]] = s
    b = OPTIONS["B"]
    no_fb = composite_availability(b, with_fallback=False)
    print(f"  availability for B without a model fallback: {no_fb * 100:.2f}% (< {b['sla']['availability'] * 100:.1f}%) "
          f"-> fallback path is what makes the SLA feasible")
    print("  model choice = cheapest candidate passing ALL SLAs: "
          + ", ".join(f"{k}->{SHORT[v]}" for k, v in chosen.items()) + "  (model trade-offs in depth: D2.1)")

    section("4. Value pillars per option")
    pillars = {oid: compute_pillars(OPTIONS[oid], m, cost_per_call[(oid, m)]) for oid, m in chosen.items()}
    rows = [
        ("Efficiency: agent hours saved / month", lambda p: f"{p.hours_saved_month:,.0f}"),
        ("Efficiency: first response time (h)", lambda p: f"{BASE['frt_hours']:g} -> {p.frt_after_h:.1f}"),
        ("Productivity: tickets per FTE / month", lambda p: f"{p.tickets_per_fte_before:,.0f} -> {p.tickets_per_fte_after:,.0f}"),
        ("Productivity: uplift", lambda p: f"+{pct(p.productivity_uplift)}"),
        ("Transformation: risk-adj. new value / yr", lambda p: usd(p.new_value_risk_adj_year)),
        ("Cost: Claude API $ / ticket", lambda p: f"${p.api_cost_per_ticket:.5f}"),
        ("Cost: run cost / month (API+infra+people)", lambda p: usd(p.run_month)),
        ("Cost: realised benefit / month", lambda p: usd(p.benefit_month)),
        ("ROI: year-1 ROI", lambda p: pct(p.roi_year1)),
        ("ROI: payback (months)", lambda p: f"{p.payback_months:.1f}"),
        ("ROI: 3-year net value", lambda p: usd(p.net_3yr)),
        ("North star: cost per resolved ticket", lambda p: f"${p.cost_per_resolved_before:.2f} -> ${p.cost_per_resolved_after:.2f}"),
    ]
    ids = list(pillars)
    print(f"  {'metric':<44}" + "".join(f"{('Option ' + i + ' (' + SHORT[chosen[i]] + ')'):>22}" for i in ids))
    for label, fn in rows:
        print(f"  {label:<44}" + "".join(f"{fn(pillars[i]):>22}" for i in ids))

    section("5. Weighted scorecard (0-5 per pillar)")
    w = VM["weights"]
    raw = {i: {"efficiency": pillars[i].hours_saved_month, "productivity": pillars[i].productivity_uplift,
               "transformation": pillars[i].new_value_risk_adj_year, "cost_roi": max(0.0, pillars[i].roi_year1),
               "sla": sum(sla_results[i].passes.values()) / len(sla_results[i].passes)} for i in ids}
    scores: dict[str, dict[str, float]] = {}
    for i in ids:
        scores[i] = {}
        for k in w:
            top = max(raw[j][k] for j in ids)
            scores[i][k] = 5 * raw[i][k] / top if top > 0 else 0.0
        scores[i]["total"] = sum(scores[i][k] * w[k] for k in w)
    print(f"  {'option':<8}" + "".join(f"{k + ' ' + pct(v):>20}" for k, v in w.items()) + f"{'TOTAL':>8}")
    ranking = sorted(ids, key=lambda i: -scores[i]["total"])
    for i in ranking:
        print(f"  {i:<8}" + "".join(f"{scores[i][k]:>20.2f}" for k in w) + f"{scores[i]['total']:>8.2f}")
    best = ranking[0]
    print(f"  -> recommended: Option {best} ({OPTIONS[best]['name']}) on {SHORT[chosen[best]]}")

    section("6. KPI tree for the recommended option")
    p, s, o = pillars[best], sla_results[best], OPTIONS[best]
    tree = [
        (0, f"North star: cost per resolved ticket ${p.cost_per_resolved_before:.2f} -> ${p.cost_per_resolved_after:.2f} "
            f"(guardrail: CSAT >= {BASE['csat']})"),
        (1, "Efficiency"),
        (2, f"Blended AHT (min)              {BASE['aht_min']:.1f} -> {p.aht_after_min:.1f}"),
        (2, f"First response time (h)        {BASE['frt_hours']:.1f} -> {p.frt_after_h:.1f}"),
        (2, f"Agent hours saved / month      0 -> {p.hours_saved_month:,.0f}"),
        (1, "Productivity"),
        (2, f"Tickets per FTE / month        {p.tickets_per_fte_before:,.0f} -> {p.tickets_per_fte_after:,.0f}"),
        (1, "Cost"),
        (2, f"Claude API $ / ticket          ${p.api_cost_per_ticket:.4f}"),
        (2, f"Run cost / month               {usd(p.run_month)}"),
        (2, f"Payback (months)               {p.payback_months:.1f}"),
        (1, "Transformation"),
        (2, f"Risk-adjusted new value / yr   {usd(p.new_value_risk_adj_year)} ({o['new_capability']})"),
        (1, "Performance SLAs (tracking: D6.3)"),
        (2, f"p95 reply latency (s)          {s.p95:.2f} vs <= {o['sla']['p95_latency_s']:.1f}"),
        (2, f"Availability                   {s.availability * 100:.2f}% vs >= {o['sla']['availability'] * 100:.1f}%"),
        (2, f"Resolution quality             {s.quality:.2f} vs >= {o['sla']['quality']:.2f}"),
    ]
    for depth, label in tree:
        print("  " + ("" if depth == 0 else "   " * (depth - 1) + "+- ") + label)

    section("7. Executive summary (Claude Opus drafts prose from computed numbers)")
    payload = {
        "ranking": [{"id": i, "name": OPTIONS[i]["name"], "model": SHORT[chosen[i]], "score": f"{scores[i]['total']:.1f}",
                     "hours_saved": f"{pillars[i].hours_saved_month:,.0f}", "productivity": f"+{pct(pillars[i].productivity_uplift)}",
                     "new_value": usd(pillars[i].new_value_risk_adj_year), "roi": pct(pillars[i].roi_year1),
                     "payback": f"{pillars[i].payback_months:.1f}", "net_3yr": usd(pillars[i].net_3yr),
                     "api_month": usd(pillars[i].api_month)} for i in ranking],
        "sla_note": "; ".join(f"{SHORT[m]} fails {'+'.join(k for k, ok in all_sla[(best, m)].passes.items() if not ok)}"
                              for m in OPTIONS[best]["candidates"] if not all_sla[(best, m)].all_pass)
                    or "every candidate passes",
    }
    msg = client.messages.create(
        model=MODEL_OPUS, max_tokens=1200,
        output_config={"effort": "medium"},
        system="You are a principal architect writing for a CFO and COO. Five short lines: recommendation, value, "
               "economics, performance, risks. Use ONLY the numbers in the scorecard; do not compute new ones.",
        messages=[{"role": "user", "content": "SCORECARD:" + json.dumps(payload)}],
    )
    if msg.stop_reason == "end_turn":
        for line in text_of(msg).splitlines():
            print(f"  | {line}")
    print(f"  [{usage_line(msg)}]")

    section("Summary")
    print("  Tie every AI option to measurable pillars, compute value in code (not in the prompt), let SLAs pick")
    print("  the model, risk-adjust the uncertain upside, and hand executives a KPI tree they can hold you to.")


if __name__ == "__main__":
    main()
