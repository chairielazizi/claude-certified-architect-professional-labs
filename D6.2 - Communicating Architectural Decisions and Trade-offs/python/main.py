"""
D6.2 - Communicating Architectural Decisions and Trade-offs

Decision: how should the Claims Correspondence Copilot ground drafts in policy wording?
Options: Hybrid RAG (A), long context + prompt caching (B), agentic search on Sonnet (C)
or Opus (D).

What this lab builds:
  (a) a weighted decision matrix: must-meet screening -> anchored 0-10 scores ->
      stakeholder weights -> totals, plus sensitivity analysis (one-at-a-time, break-even
      scan, seeded Monte Carlo) to answer "does the winner change if the weights shift?"
  (b) an ADR (Michael Nygard format + options/trade-offs) drafted by Claude from the
      matrix as a structured output, validated by code (every number must be grounded
      in the matrix), repaired once if needed, and rendered deterministically
  (c) audience-tailored briefs (executive / engineering lead / security-legal) from ONE
      prompt template parameterized by an audience profile, checked by code
      (word budget, required topics, jargon, grounded numbers, readability)
  (d) a quantified trade-off narrative (cost / latency / quality deltas) built by code

Principle: code computes every number; Claude explains them for each audience.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_OPUS,
    PRICING,
    MockRequest,
    banner,
    get_client,
    section,
    text_of,
    usage_line,
)

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
OUT = LAB / "labs_output"


# ---------------------------------------------------------------------------
# Structured-output contract for the ADR draft. Status is NOT in the schema:
# whether a decision is Proposed/Accepted is a governance fact, not model output.
# ---------------------------------------------------------------------------
class ConsideredOption(BaseModel):
    option_id: str
    name: str
    pros: list[str]
    cons: list[str]
    outcome: str  # "chosen" or "rejected: <reason>"


class RiskItem(BaseModel):
    risk: str
    mitigation: str


class ADRDraft(BaseModel):
    title: str
    context: str
    decision: str
    decision_drivers: list[str]
    considered_options: list[ConsideredOption]
    consequences_positive: list[str]
    consequences_negative: list[str]
    risks: list[RiskItem]
    review_triggers: list[str]


ADR_SYSTEM = """You are a solution architect writing an Architecture Decision Record (ADR, Michael Nygard style).
Write for future maintainers: state the context and forces, the decision in active voice ("We will ..."),
and the consequences - including the negative ones we accept.
Rules: use ONLY the numbers given in the FACTS block (no new or rounded-differently figures);
include every option (also eliminated ones) in considered_options with outcome "chosen" or "rejected: <reason>";
review_triggers are observable conditions that should reopen this decision."""

BRIEF_TEMPLATE = """You are a solution architect writing a decision brief.
Audience id: {id}
Audience: {label}
Their goal: {goal}
They care about: {cares}
Avoid: {avoid}
Format: {format}
Hard limit: {max_words} words. Plain-language target: Flesch reading ease >= {min_flesch}.
Use only facts and numbers from the FACTS block. Do not introduce new numbers."""


# ---------------------------------------------------------------------------
# Offline responder - canned replies shared with the TypeScript version.
# ---------------------------------------------------------------------------
def responder(req: MockRequest) -> str:
    mock = DATA / "mock_responses"
    revision = "REVISION REQUEST" in req.last_user_text
    if "Architecture Decision Record" in req.system_text:
        return (mock / ("adr_repaired.json" if revision else "adr_draft.json")).read_text(encoding="utf-8")
    m = re.search(r"Audience id: (\w+)", req.system_text)
    if m:
        name = f"brief_{m.group(1)}{'_revised' if revision else ''}.md"
        return (mock / name).read_text(encoding="utf-8")
    return "(no canned reply)"


# ---------------------------------------------------------------------------
# Deterministic helpers (mirrored 1:1 in TypeScript)
# ---------------------------------------------------------------------------
def mulberry32(seed: int) -> Callable[[], float]:
    """Tiny seeded PRNG with an identical TypeScript implementation -> same Monte Carlo results."""
    state = seed & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((state ^ (state >> 15)) * (state | 1)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return nxt


def money(x: float, decimals: int = 0) -> str:
    return "$" + f"{x:,.{decimals}f}"


def anchored_score(value: float, worst: float, best: float) -> float:
    """Anchored 0-10 scale. Unlike min-max across the options, adding or removing an
    option cannot change another option's score (no rank reversal)."""
    s = 10 * (value - worst) / (best - worst)
    return max(0.0, min(10.0, s))


NUMBER_PATTERNS = [
    r"\$\s?(\d[\d,]*(?:\.\d+)?)",
    r"(\d+(?:\.\d+)?)\s?%",
    r"(\d+(?:\.\d+)?)\s?pp\b",
    r"(\d+(?:\.\d+)?)\s?(?:s|sec|seconds)\b",
    r"(\d+(?:\.\d+)?)\s?weeks?\b",
    r"\bx(\d+(?:\.\d+)?)\b",
]


def extract_numbers(text: str) -> list[str]:
    found: list[str] = []
    for pat in NUMBER_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            found.append(m.group(1))
    return found


def ungrounded(text: str, facts: list[float]) -> list[str]:
    """Numbers with units ($, %, pp, s, weeks, xN) that match no computed fact (tolerance 0.5% / 0.05)."""
    bad: list[str] = []
    for raw in extract_numbers(text):
        v = float(raw.replace(",", ""))
        if not any(abs(v - f) <= max(0.05, 0.005 * abs(f)) for f in facts):
            bad.append(raw)
    return bad


def syllables(word: str) -> int:
    w = word.lower()
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    units = [u for u in re.split(r"[.!?]+\s+|\n+", text) if re.search(r"[A-Za-z]", u)]
    if not words:
        return 0.0
    syl = sum(syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / max(1, len(units))) - 84.6 * (syl / len(words))


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.,%/-]*", text))


# ---------------------------------------------------------------------------
# Matrix model
# ---------------------------------------------------------------------------
@dataclass
class Option:
    id: str
    name: str
    summary: str
    model: str
    raw: dict[str, float]
    cost_per_request: float = 0.0
    monthly_cost: float = 0.0
    cost_per_letter: float = 0.0
    failures: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    total: float = 0.0


def weighted_total(opt: Option, weights: dict[str, float], criteria: list[dict[str, Any]]) -> float:
    wsum = sum(weights[c["id"]] for c in criteria)
    return sum(weights[c["id"]] * opt.scores[c["id"]] for c in criteria) / wsum


def winner_of(options: list[Option], weights: dict[str, float], criteria: list[dict[str, Any]]) -> Option:
    best, best_total = options[0], -1.0
    for o in options:
        t = weighted_total(o, weights, criteria)
        if t > best_total:
            best, best_total = o, t
    return best


def check_constraint(value: float, op: str, target: float) -> bool:
    return value <= target if op == "<=" else value >= target if op == ">=" else value == target


def main() -> None:
    banner("D6.2 - Communicating Architectural Decisions and Trade-offs")
    d = json.loads((DATA / "decision.json").read_text(encoding="utf-8"))
    audiences = json.loads((DATA / "audiences.json").read_text(encoding="utf-8"))["audiences"]
    criteria: list[dict[str, Any]] = d["criteria"]
    crit_label = {c["id"]: c["label"] for c in criteria}
    client = get_client(responder)

    # -- 1. Framing ------------------------------------------------------------
    section("1. Decision framing")
    print(f"Question: {d['question']}")
    for o in d["options"]:
        print(f"  Option {o['id']}: {o['name']}")
    print("Must-meet constraints (screen first - filters, not weights):")
    for c in d["constraints"]:
        print(f"  {c['id']}: {c['label']}")

    # -- 2. Cost model -----------------------------------------------------------
    section("2. Cost model from token profiles x model pricing")
    options: list[Option] = []
    for o in d["options"]:
        p, t = PRICING[o["model"]], o["tokens_per_request"]
        per_req = (t["input"] * p["input"] + t["cache_read"] * p["cache_read"]
                   + t["cache_write"] * p["cache_write"] + t["output"] * p["output"]) / 1_000_000
        opt = Option(o["id"], o["name"], o["summary"], o["model"], dict(o["raw"]))
        opt.cost_per_request = per_req
        opt.monthly_cost = per_req * d["requests_per_month"] + o["fixed_monthly_usd"]
        opt.cost_per_letter = opt.monthly_cost / d["letters_per_month"]
        opt.raw["monthly_cost_usd"] = opt.monthly_cost
        options.append(opt)
        print(f"  {o['id']} {o['model']:<18} in={t['input']:>6} cache_read={t['cache_read']:>6} "
              f"cache_write={t['cache_write']:>5} out={t['output']:>5} -> ${per_req:.4f}/request, "
              f"{money(opt.monthly_cost)}/month (incl. {money(o['fixed_monthly_usd'])} fixed)")
    print(f"  Volume: {d['requests_per_month']:,} requests/month for {d['letters_per_month']:,} letters")

    # -- 3. Must-meet screening ----------------------------------------------------
    section("3. Must-meet screening")
    for opt in options:
        for c in d["constraints"]:
            if not check_constraint(opt.raw[c["metric"]], c["op"], c["value"]):
                opt.failures.append(f"{c['id']} ({opt.raw[c['metric']]:g} vs {c['op']} {c['value']:g})")
        print(f"  {opt.id}: {'PASS' if not opt.failures else 'ELIMINATED - fails ' + ', '.join(opt.failures)}")
    viable = [o for o in options if not o.failures]
    eliminated = [o for o in options if o.failures]

    # -- 4. Weights -----------------------------------------------------------------
    section("4. Stakeholder weights (points out of 100) and influence")
    print(f"  {'stakeholder':<34}{'infl':>5}  " + " ".join(f"{c['id'][:7]:>7}" for c in criteria))
    combined = {c["id"]: 0.0 for c in criteria}
    for s in d["stakeholders"]:
        print(f"  {s['name'] + ' (' + s['id'] + ')':<34}{s['influence']:>5.2f}  "
              + " ".join(f"{s['weights'][c['id']]:>7}" for c in criteria))
        for c in criteria:
            combined[c["id"]] += s["influence"] * s["weights"][c["id"]]
    print(f"  {'COMBINED (influence-weighted)':<34}{'':>5}  " + " ".join(f"{combined[c['id']]:>7.1f}" for c in criteria))

    # -- 5. Matrix --------------------------------------------------------------------
    section("5. Weighted decision matrix (anchored 0-10 scores)")
    for opt in options:
        for c in criteria:
            opt.scores[c["id"]] = anchored_score(opt.raw[c["metric"]], c["worst"], c["best"])
        opt.total = weighted_total(opt, combined, criteria)
    print(f"  {'criterion':<24}{'weight':>7}" + "".join(f"{o.id:>8}" for o in options))
    for c in criteria:
        print(f"  {c['label']:<24}{combined[c['id']]:>7.1f}" + "".join(f"{o.scores[c['id']]:>8.2f}" for o in options))
    print(f"  {'WEIGHTED TOTAL':<24}{'':>7}" + "".join(f"{o.total:>8.2f}" for o in options))
    ranked = sorted(viable, key=lambda o: -o.total)
    win, runner = ranked[0], ranked[1]
    print(f"\n  Recommended: Option {win.id} ({win.name}) {win.total:.2f}; runner-up {runner.id} {runner.total:.2f}; "
          f"margin {win.total - runner.total:.2f}")
    if eliminated:
        print(f"  Not eligible: {', '.join(o.id for o in eliminated)} (scored for transparency, excluded by must-meet)")
    print("  Winner through each stakeholder's own weights:")
    per_stakeholder: list[tuple[str, str, float]] = []
    for s in d["stakeholders"]:
        w = {k: float(v) for k, v in s["weights"].items()}
        sw = winner_of(viable, w, criteria)
        per_stakeholder.append((s["name"], sw.id, weighted_total(sw, w, criteria)))
        print(f"    {s['name']:<16} -> {sw.id} ({weighted_total(sw, w, criteria):.2f})"
              + ("" if sw.id == win.id else "   <- disagrees: address this concern explicitly"))

    # -- 6. Sensitivity ----------------------------------------------------------------
    section("6. Sensitivity analysis: does the winner change if weights shift?")
    sens = d["sensitivity"]
    pert = sens["perturbation"]
    print(f"  One-at-a-time: each weight x{1 - pert:.1f} and x{1 + pert:.1f} (others unchanged, renormalized)")
    oat_flips: list[str] = []
    for c in criteria:
        res = []
        for mult in (1 - pert, 1 + pert):
            w = dict(combined)
            w[c["id"]] = combined[c["id"]] * mult
            res.append(winner_of(viable, w, criteria).id)
        flag = "" if res == [win.id, win.id] else "  <- FLIPS"
        if flag:
            oat_flips.append(c["id"])
        print(f"    {c['label']:<24} low -> {res[0]}   high -> {res[1]}{flag}")

    print(f"  Break-even scan: smallest single-weight multiplier (step {sens['scan_step']}, up to x{sens['scan_max_multiplier']:g}) that changes the winner")
    breakevens: list[tuple[str, float, str]] = []
    steps_up = int(round((sens["scan_max_multiplier"] - 1) / sens["scan_step"]))
    steps_down = int(round(1 / sens["scan_step"]))
    for c in criteria:
        hit: Optional[tuple[float, str]] = None
        for k in range(1, max(steps_up, steps_down) + 1):
            for mult in (1 + k * sens["scan_step"], 1 - k * sens["scan_step"]):
                if mult < 0 or mult > sens["scan_max_multiplier"] + 1e-9:
                    continue
                w = dict(combined)
                w[c["id"]] = combined[c["id"]] * mult
                wn = winner_of(viable, w, criteria)
                if wn.id != win.id and hit is None:
                    hit = (mult, wn.id)
            if hit:
                break
        if hit:
            breakevens.append((c["id"], hit[0], hit[1]))
            print(f"    {c['label']:<24} x{hit[0]:.2f} ({combined[c['id']]:.1f} -> {combined[c['id']] * hit[0]:.1f} pts) -> {hit[1]} wins")
        else:
            print(f"    {c['label']:<24} no change within x0.00 - x{sens['scan_max_multiplier']:.2f}")

    rng = mulberry32(sens["seed"])
    wins = {o.id: 0 for o in viable}
    for _ in range(sens["trials"]):
        w = {c["id"]: combined[c["id"]] * (1 - pert + 2 * pert * rng()) for c in criteria}
        wins[winner_of(viable, w, criteria).id] += 1
    mc = {k: v / sens["trials"] * 100 for k, v in wins.items()}
    print(f"  Monte Carlo: {sens['trials']:,} trials, every weight x U({1 - pert:.1f}, {1 + pert:.1f}), seed {sens['seed']}")
    for k, v in mc.items():
        print(f"    Option {k} wins {v:5.1f}% {'#' * int(v // 2)}")
    robust = "ROBUST" if mc[win.id] >= 80 else "FRAGILE"
    print(f"  Verdict: {robust} - {win.id} wins {mc[win.id]:.1f}% of perturbations; "
          + (f"only a large shift toward {', '.join(crit_label[b[0]] for b in breakevens)} changes it."
             if breakevens else "no single weight shift changes it."))

    # -- 7. Trade-off narrative ----------------------------------------------------------
    section("7. Quantified trade-off narrative (code-generated)")
    tradeoffs: list[str] = []
    for o in [x for x in ranked if x.id != win.id] + eliminated:
        tradeoffs.append(tradeoff_sentence(win, o))
    for t in tradeoffs:
        print(f"  - {t}")
    headroom = 8000 - win.monthly_cost
    print(f"  - Budget: {win.id} uses {money(win.monthly_cost)} of the $8,000/month cap ({money(headroom)} headroom), "
          f"about ${win.cost_per_letter:.3f} per letter.")

    # Facts = the only numbers Claude may use. Collected once; checked everywhere.
    facts_text, facts_num = build_facts(d, options, win, runner, combined, criteria, per_stakeholder,
                                        mc, breakevens, tradeoffs, headroom)

    # -- 8. ADR ---------------------------------------------------------------------------
    section("8. ADR drafted by Claude (structured output), validated by code")
    adr_user = (f"Write ADR-{d['adr_number']:04d} for this decision.\nQUESTION: {d['question']}\n"
                f"CONTEXT:\n- " + "\n- ".join(d["context_facts"]) + "\n\nFACTS:\n" + facts_text)
    adr, issues, attempts = None, [], 0
    messages: list[dict[str, Any]] = [{"role": "user", "content": adr_user}]
    while attempts < 2:
        attempts += 1
        resp = client.messages.parse(
            model=MODEL_OPUS, max_tokens=6000, thinking={"type": "adaptive"}, output_config={"effort": "high"},
            system=ADR_SYSTEM, messages=messages, output_format=ADRDraft,
        )
        adr = resp.parsed_output
        if adr is None:
            raise RuntimeError("ADR did not parse")
        issues = validate_adr(adr, options, win, facts_num)
        print(f"  attempt {attempts}: {len(issues)} issue(s) | {usage_line(resp)}")
        for i in issues:
            print(f"    - {i}")
        if not issues:
            break
        # Validate -> repair loop: give the model its own draft plus the exact problems.
        messages = [{"role": "user", "content": adr_user + "\n\nREVISION REQUEST - fix these issues and return the full ADR:\n- "
                     + "\n- ".join(issues) + "\n\nPrevious draft:\n" + adr.model_dump_json()}]
    assert adr is not None
    adr_md = render_adr(d, adr, options, win, combined, criteria, tradeoffs, per_stakeholder, mc, breakevens,
                        headroom, issues, attempts)
    print(f"  Rendered ADR-{d['adr_number']:04d} in Nygard format: Status / Context / Decision / Consequences (+ options, trade-offs)")

    # -- 9. Audience briefs -------------------------------------------------------------
    section("9. Audience-tailored briefs from one parameterized template")
    briefs: list[tuple[dict[str, Any], str, list[str], dict[str, Any], int]] = []
    for aud in audiences:
        system = BRIEF_TEMPLATE.format(
            id=aud["id"], label=aud["label"], goal=aud["goal"], cares="; ".join(aud["cares_about"]),
            avoid="; ".join(aud["avoid"]), format=aud["format"], max_words=aud["max_words"],
            min_flesch=aud["min_flesch"],
        )
        user = "FACTS:\n" + facts_text + "\n\nWrite the brief for this audience."
        text, checks, attempts = "", [], 0
        msgs: list[dict[str, Any]] = [{"role": "user", "content": user}]
        while attempts < 2:
            attempts += 1
            resp = client.messages.create(
                model=MODEL_OPUS, max_tokens=1500, thinking={"type": "adaptive"}, output_config={"effort": "medium"},
                system=system, messages=msgs,
            )
            text = text_of(resp).strip()
            checks, metrics = check_brief(text, aud, facts_num)
            print(f"  {aud['id']:<17} attempt {attempts}: {metrics['words']}/{aud['max_words']} words, "
                  f"Flesch {metrics['flesch']:.1f}, issues={len(checks)} | {usage_line(resp)}")
            for ch in checks:
                print(f"      - {ch}")
            if not checks:
                break
            msgs = [{"role": "user", "content": user + "\n\nREVISION REQUEST - fix: " + "; ".join(checks)
                     + "\n\nPrevious draft:\n" + text}]
        briefs.append((aud, text, checks, metrics, attempts))
    print("  Same facts, different framing: exec = value/cost/risk/ask; engineering = design/ops; security = data flows/controls.")

    # -- 10. Outputs -------------------------------------------------------------------
    section("10. Outputs")
    OUT.mkdir(parents=True, exist_ok=True)
    adr_name = f"ADR-{d['adr_number']:04d}-policy-grounding-strategy.md"
    files = {
        adr_name: adr_md,
        "decision_matrix.md": render_matrix(d, options, win, combined, criteria, per_stakeholder, oat_flips, breakevens, mc, robust),
        "stakeholder_briefs.md": render_briefs(d, win, briefs),
        "decision_facts.txt": facts_text + "\n",
    }
    for name, content in files.items():
        (OUT / name).write_text(content, encoding="utf-8")
        print(f"  wrote labs_output/{name} ({len(content.splitlines())} lines)")
    print("  Samples committed under examples/")


# ---------------------------------------------------------------------------
# Narrative, facts, validation and rendering
# ---------------------------------------------------------------------------
def tradeoff_sentence(w: Option, o: Option) -> str:
    dm = w.monthly_cost - o.monthly_cost
    pct = abs(dm) / o.monthly_cost * 100
    cost = (f"saves {money(abs(dm))}/month (-{pct:.1f}%)" if dm < 0 else f"costs {money(dm)}/month more (+{pct:.1f}%)")
    dl = w.raw["p95_latency_s"] - o.raw["p95_latency_s"]
    lat = f"{abs(dl):.1f} s {'faster' if dl < 0 else 'slower'} at p95"
    dq = w.raw["eval_pass_rate_pct"] - o.raw["eval_pass_rate_pct"]
    qual = f"{abs(dq):.1f} pp {'higher' if dq >= 0 else 'lower'} answer quality"
    dc = w.raw["citation_accuracy_pct"] - o.raw["citation_accuracy_pct"]
    cit = f"{abs(dc):.1f} pp {'higher' if dc >= 0 else 'lower'} citation accuracy"
    dw = w.raw["build_weeks"] - o.raw["build_weeks"]
    weeks = "same build time" if dw == 0 else f"ships {abs(dw):g} weeks {'sooner' if dw < 0 else 'later'}"
    tail = f" ({o.id} is not eligible: fails {', '.join(o.failures)})" if o.failures else ""
    return f"{w.id} over {o.id}: {cost}; {lat}; {qual}; {cit}; {weeks}.{tail}"


def build_facts(d, options, win, runner, combined, criteria, per_stakeholder, mc, breakevens, tradeoffs, headroom):
    lines: list[str] = []
    nums: list[float] = [c["value"] for c in d["constraints"]] + [d["requests_per_month"], d["letters_per_month"]]
    for o in options:
        r = o.raw
        status = "eligible" if not o.failures else "NOT ELIGIBLE - fails " + ", ".join(o.failures)
        lines.append(
            f"Option {o.id} ({o.name}): eval pass rate {r['eval_pass_rate_pct']:g}%, p95 {r['p95_latency_s']:g} s, "
            f"monthly cost {money(o.monthly_cost)} ({money(o.cost_per_request, 4)}/request, {money(o.cost_per_letter, 3)}/letter), "
            f"build {r['build_weeks']:g} weeks, citation accuracy {r['citation_accuracy_pct']:g}%, "
            f"freshness {r['freshness_minutes']:g} min, weighted score {o.total:.2f}/10 - {status}"
        )
        nums += [r[c["metric"]] for c in criteria] + [o.cost_per_request, o.monthly_cost, o.cost_per_letter, o.total]
    lines.append(f"Recommended: Option {win.id}; runner-up Option {runner.id} (margin {win.total - runner.total:.2f})")
    nums += [win.total - runner.total, headroom]
    lines.append(f"Budget headroom with {win.id}: {money(headroom)}/month under the $8,000 cap")
    lines.append("Combined weights: " + ", ".join(f"{c['label']} {combined[c['id']]:.1f}" for c in criteria))
    nums += list(combined.values())
    lines.append("Winner by stakeholder weights: " + ", ".join(f"{n} -> {wid}" for n, wid, _ in per_stakeholder))
    nums += [t for _, _, t in per_stakeholder]
    lines.append(f"Monte Carlo ({d['sensitivity']['trials']} trials, weights +/-{d['sensitivity']['perturbation'] * 100:g}%): "
                 + ", ".join(f"{k} wins {v:.1f}%" for k, v in mc.items()))
    nums += list(mc.values()) + [d["sensitivity"]["trials"], d["sensitivity"]["perturbation"] * 100]
    for cid, mult, wid in breakevens:
        lines.append(f"Break-even: {cid} weight x{mult:.2f} ({combined[cid]:.1f} -> {combined[cid] * mult:.1f} points) makes {wid} the winner")
        nums += [mult, combined[cid] * mult]
    for t in tradeoffs:
        lines.append("Trade-off: " + t)
        nums += [float(x.replace(",", "")) for x in extract_numbers(t)]
    return "\n".join(lines), nums


def adr_text(adr: ADRDraft) -> str:
    """Every free-text field of the draft - all of it is subject to the grounding check."""
    texts = [adr.title, adr.context, adr.decision, *adr.decision_drivers, *adr.consequences_positive,
             *adr.consequences_negative, *adr.review_triggers]
    for o in adr.considered_options:
        texts += [*o.pros, *o.cons, o.outcome]
    for r in adr.risks:
        texts += [r.risk, r.mitigation]
    return "\n".join(texts)


def validate_adr(adr: ADRDraft, options: list[Option], win: Option, facts: list[float]) -> list[str]:
    issues: list[str] = []
    if win.name.lower() not in adr.decision.lower() and f"option {win.id}".lower() not in adr.decision.lower():
        issues.append(f"decision text does not name the recommended option ({win.id})")
    ids = [o.option_id for o in adr.considered_options]
    missing = [o.id for o in options if o.id not in ids]
    if missing:
        issues.append(f"considered_options missing: {', '.join(missing)}")
    chosen = [o.option_id for o in adr.considered_options if o.outcome.lower().startswith("chosen")]
    if chosen != [win.id]:
        issues.append(f"exactly one option must be 'chosen' and it must be {win.id} (got {chosen})")
    if not adr.consequences_negative:
        issues.append("no negative consequences - every real decision has a cost")
    if not adr.review_triggers:
        issues.append("no review triggers")
    bad = ungrounded(adr_text(adr), facts)
    if bad:
        issues.append("ungrounded numbers (not in the decision matrix): " + ", ".join(bad))
    return issues


def check_brief(text: str, aud: dict[str, Any], facts: list[float]) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    words = word_count(text)
    low = text.lower()
    if words > aud["max_words"]:
        issues.append(f"over word budget ({words} > {aud['max_words']})")
    for group in aud["required"]:
        if not any(term.lower() in low for term in group):
            issues.append(f"missing required topic: {' / '.join(group)}")
    jargon = [j for j in aud["jargon_blocklist"] if re.search(r"(?<![a-z])" + re.escape(j.lower()) + r"(?![a-z])", low)]
    if jargon:
        issues.append(f"jargon for this audience: {', '.join(jargon)}")
    bad = ungrounded(text, facts)
    if bad:
        issues.append(f"ungrounded numbers: {', '.join(bad)}")
    flesch = flesch_reading_ease(text)
    if flesch < aud["min_flesch"]:
        issues.append(f"reading ease {flesch:.1f} < {aud['min_flesch']} (too dense for this audience)")
    return issues, {"words": words, "flesch": flesch, "numbers": len(extract_numbers(text))}


def render_adr(d, adr: ADRDraft, options, win, combined, criteria, tradeoffs, per_stakeholder, mc, breakevens,
               headroom, issues, attempts) -> str:
    deciders = ", ".join(f"{s['name']} ({s['role']})" for s in d["stakeholders"])
    L: list[str] = []
    a = L.append
    a(f"# ADR-{d['adr_number']:04d}: {adr.title}")
    a("")
    a("## Status")
    a("")
    a(f"Proposed - {d['as_of']}. Awaiting approval by the deciders below; becomes Accepted when signed off.")
    a("")
    a(f"- **Deciders:** {deciders}")
    a(f"- **Related requirements:** {', '.join(d['requirement_links'])}")
    a("")
    a("## Context")
    a("")
    a(adr.context)
    a("")
    a("**Decision drivers**")
    a("")
    for x in adr.decision_drivers:
        a(f"- {x}")
    a("")
    a("## Decision")
    a("")
    a(adr.decision)
    a("")
    a("## Consequences")
    a("")
    a("**Positive**")
    a("")
    for x in adr.consequences_positive:
        a(f"- {x}")
    a("")
    a("**Negative (accepted trade-offs)**")
    a("")
    for x in adr.consequences_negative:
        a(f"- {x}")
    a("")
    a("## Considered options")
    a("")
    a("| Option | Weighted score (0-10) | Must-meet | Outcome |")
    a("|---|---:|---|---|")
    by_id = {o.id: o for o in options}
    for co in adr.considered_options:
        o = by_id.get(co.option_id)
        score = f"{o.total:.2f}" if o else "-"
        mm = ("PASS" if not o.failures else "FAIL: " + ", ".join(o.failures)) if o else "-"
        a(f"| {co.option_id} - {co.name} | {score} | {mm} | {co.outcome} |")
    a("")
    for co in adr.considered_options:
        a(f"**{co.option_id} - {co.name}**")
        a("")
        a("- Pros: " + "; ".join(co.pros))
        a("- Cons: " + "; ".join(co.cons))
        a("")
    a("## Trade-offs (quantified, computed from the decision matrix)")
    a("")
    for t in tradeoffs:
        a(f"- {t}")
    a(f"- Budget: {money(win.monthly_cost)} of the $8,000/month cap ({money(headroom)} headroom), about ${win.cost_per_letter:.3f} per letter.")
    a("")
    a("## Sensitivity")
    a("")
    for name, wid, total in per_stakeholder:
        a(f"- {name}'s weights -> Option {wid} ({total:.2f})")
    a(f"- Monte Carlo ({d['sensitivity']['trials']:,} weight perturbations of +/-{d['sensitivity']['perturbation'] * 100:g}%): "
      + ", ".join(f"{k} wins {v:.1f}%" for k, v in mc.items()))
    for cid, mult, wid in breakevens:
        a(f"- Break-even: the {cid} weight would need to rise x{mult:.2f} ({combined[cid]:.1f} -> {combined[cid] * mult:.1f} points) for Option {wid} to win.")
    a("")
    a("## Risks and mitigations")
    a("")
    a("| Risk | Mitigation |")
    a("|---|---|")
    for r in adr.risks:
        a(f"| {r.risk} | {r.mitigation} |")
    a("")
    a("## Review triggers (reopen this ADR if...)")
    a("")
    for x in adr.review_triggers:
        a(f"- {x}")
    a("")
    a("---")
    n_checked = len(extract_numbers(adr_text(adr)))
    rounds = "first draft" if attempts == 1 else f"after {attempts - 1} repair round(s)"
    a(f"_Drafted by Claude from the decision matrix (structured output). Status, scores, trade-offs and sensitivity "
      f"are computed by code; {n_checked} figures in Claude's text were checked against the matrix "
      f"({rounds}: {'0 ungrounded' if not issues else 'issues: ' + '; '.join(issues)})._")
    a("")
    return "\n".join(L)


def render_matrix(d, options, win, combined, criteria, per_stakeholder, oat_flips, breakevens, mc, robust) -> str:
    L: list[str] = []
    a = L.append
    a(f"# Decision matrix - {d['question']}")
    a("")
    a(f"As of {d['as_of']}. Scores use anchored 0-10 scales (worst -> 0, best -> 10), so adding or removing an option "
      "cannot reorder the others. Must-meet constraints are applied before scoring.")
    a("")
    a("## Raw measurements")
    a("")
    a("| Criterion | Unit | Worst / best anchor | " + " | ".join(f"{o.id}" for o in options) + " |")
    a("|---|---|---|" + "---:|" * len(options))
    for c in criteria:
        vals = []
        for o in options:
            v = o.raw[c["metric"]]
            vals.append(money(v) if c["id"] == "cost" else f"{v:g}")
        a(f"| {c['label']} | {c['unit']} | {c['worst']:g} / {c['best']:g} | " + " | ".join(vals) + " |")
    a("| Must-meet | - | - | " + " | ".join("PASS" if not o.failures else "FAIL" for o in options) + " |")
    a("")
    a("## Weighted scores")
    a("")
    a("| Criterion | Weight | " + " | ".join(o.id for o in options) + " |")
    a("|---|---:|" + "---:|" * len(options))
    for c in criteria:
        a(f"| {c['label']} | {combined[c['id']]:.1f} | " + " | ".join(f"{o.scores[c['id']]:.2f}" for o in options) + " |")
    a("| **Weighted total** | 100 | " + " | ".join(f"**{o.total:.2f}**" for o in options) + " |")
    a("")
    a(f"**Recommendation:** Option {win.id} - {win.name}.")
    a("")
    a("## Stakeholder weights")
    a("")
    a("| Stakeholder | Influence | " + " | ".join(c["label"] for c in criteria) + " | Own winner |")
    a("|---|---:|" + "---:|" * len(criteria) + "---|")
    for s, (_, wid, total) in zip(d["stakeholders"], per_stakeholder):
        a(f"| {s['name']} ({s['role']}) | {s['influence']:.2f} | " + " | ".join(str(s["weights"][c["id"]]) for c in criteria)
          + f" | {wid} ({total:.2f}) |")
    a("")
    a("## Sensitivity")
    a("")
    a(f"- One-at-a-time (each weight x{1 - d['sensitivity']['perturbation']:.1f} / x{1 + d['sensitivity']['perturbation']:.1f}): "
      + (f"winner flips for {', '.join(oat_flips)}" if oat_flips else "winner never changes"))
    for cid, mult, wid in breakevens:
        a(f"- Break-even: {cid} weight x{mult:.2f} ({combined[cid]:.1f} -> {combined[cid] * mult:.1f} points) -> Option {wid}")
    a(f"- Monte Carlo ({d['sensitivity']['trials']:,} trials, seed {d['sensitivity']['seed']}): "
      + ", ".join(f"{k} {v:.1f}%" for k, v in mc.items()) + f" -> **{robust}**")
    a("")
    return "\n".join(L)


def render_briefs(d, win: Option, briefs) -> str:
    L: list[str] = []
    a = L.append
    a(f"# Stakeholder briefs - ADR-{d['adr_number']:04d} (Option {win.id}: {win.name})")
    a("")
    a("One decision, three framings. Each brief came from the same prompt template parameterized by an audience "
      "profile (`data/audiences.json`) and was checked by code before release.")
    a("")
    for aud, text, checks, metrics, attempts in briefs:
        a(f"## For: {aud['label']}")
        a("")
        a(text)
        a("")
        revised = "first draft" if attempts == 1 else f"after {attempts - 1} revision(s)"
        a(f"> Checks ({revised}): {metrics['words']}/{aud['max_words']} words; Flesch reading ease {metrics['flesch']:.1f}; "
          f"{metrics['numbers']} figures checked against the matrix; "
          + ("required topics, jargon and grounding all passed" if not checks else f"OPEN ISSUES: {'; '.join(checks)}") + ".")
        a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
