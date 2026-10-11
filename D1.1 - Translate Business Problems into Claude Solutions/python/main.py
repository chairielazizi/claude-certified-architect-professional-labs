"""
D1.1 - Translate Business Problems into Claude Solutions

Pipeline built here (the way an architect runs a discovery-to-design pass):

  intake notes -> structured framing (Claude + structured outputs)
               -> AI-suitability rubric (deterministic: complexity, value, viability, cost of error)
               -> tier decision (single call / workflow / agent / not-an-LLM-problem)
               -> capability mapping (classification, extraction, vision, RAG, tool use, ...)
               -> solution blueprint (+ a Claude-drafted one-pager for the top candidate)

Design principle: Claude does the *language* work (turning messy interview notes into
typed fields). The *decision* is made by transparent code, so it is repeatable,
reviewable and defensible in front of stakeholders.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    MockRequest,
    banner,
    cost_usd,
    elapsed_ms,
    get_client,
    section,
    text_of,
    usage_line,
)

DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# 1. The framing contract. Claude fills this; code consumes it.
#    Enums (Literal) keep the downstream rubric total: every value has a rule.
# ---------------------------------------------------------------------------
class ProblemFraming(BaseModel):
    pain_point: str
    primary_users: list[str]
    monthly_volume: int = Field(description="Work items per month")
    minutes_per_item_today: float = Field(description="Human handling minutes per item today")
    loaded_cost_per_hour_usd: float
    input_modalities: list[Literal["text", "image", "pdf", "structured", "audio"]]
    data_shape: Literal["structured", "semi_structured", "unstructured"]
    task_nature: Literal[
        "deterministic_rules", "judgment_on_language", "open_ended_investigation", "generation_or_summarization"
    ]
    steps_predictable: bool = Field(description="Can the sequence of steps be written down in advance?")
    needs_reference_knowledge: bool
    needs_external_systems: bool
    systems_involved: list[str]
    latency_budget_ms: int
    error_severity: Literal["low", "medium", "high", "critical"]
    human_review_feasible: bool
    data_available: bool
    constraints: list[str]
    success_kpis: list[str]


FRAMING_SYSTEM = (
    "You are a solution architect running AI discovery. Extract a structured problem framing from "
    "stakeholder interview notes. Use only facts stated or directly implied by the notes; if a number "
    "is not given, use 0. Classify task_nature by what the work fundamentally is, not by what tool "
    "might solve it."
)


# ---------------------------------------------------------------------------
# 2. Deterministic AI-suitability rubric (the four criteria)
# ---------------------------------------------------------------------------
NATURE_COMPLEXITY = {
    "deterministic_rules": 1,
    "generation_or_summarization": 2,
    "judgment_on_language": 3,
    "open_ended_investigation": 5,
}
SEVERITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}
# Annual addressable value bands (USD) -> score. Upper bound exclusive; above the last band = 5.
VALUE_BANDS = [(25_000, 1), (100_000, 2), (500_000, 3), (2_000_000, 4)]
# The more autonomous the tier, the more value it must create to justify build + run cost.
TIER_MIN_VALUE = {"single_call": 1, "workflow": 2, "agent": 3}


@dataclass
class Assessment:
    complexity: int
    value: int
    viability: int
    cost_of_error: int
    annual_hours: float
    annual_value_usd: float
    notes: list[str] = field(default_factory=list)


def assess(f: ProblemFraming) -> Assessment:
    notes: list[str] = []
    # Complexity: what kind of thinking the task needs (rules < summarise < judge < investigate).
    complexity = NATURE_COMPLEXITY[f.task_nature]
    rich = [m for m in f.input_modalities if m in ("image", "pdf", "audio")]
    if rich and complexity < 5:
        complexity += 1
        notes.append(f"complexity +1: multimodal inputs ({', '.join(rich)})")
    # Value: size of the prize = human time currently spent x loaded cost.
    annual_hours = f.monthly_volume * f.minutes_per_item_today / 60 * 12
    annual_value = annual_hours * f.loaded_cost_per_hour_usd
    value = next((score for limit, score in VALUE_BANDS if annual_value < limit), 5)
    if annual_value == 0:
        notes.append("value: no human minutes per item to save at runtime")
    # Viability: can an LLM meet the hard constraints with the data available?
    viability = 5
    if f.latency_budget_ms < 200:
        viability -= 3
        notes.append(f"viability -3: {f.latency_budget_ms} ms budget is below LLM inference latency")
    elif f.latency_budget_ms < 2000:
        viability -= 1
        notes.append(f"viability -1: tight {f.latency_budget_ms} ms budget (small model + streaming)")
    if not f.data_available:
        viability -= 3
        notes.append("viability -3: required data/systems not accessible")
    viability = max(1, viability)
    # Cost of error: severity, worsened when no human can catch mistakes in time.
    cost_of_error = SEVERITY[f.error_severity] + (0 if f.human_review_feasible else 1)
    if not f.human_review_feasible:
        notes.append("cost of error +1: no practical human checkpoint at this volume/latency")
    return Assessment(complexity, value, viability, cost_of_error, annual_hours, annual_value, notes)


# ---------------------------------------------------------------------------
# 3. Capability mapping: framing facts -> Claude platform capabilities
# ---------------------------------------------------------------------------
CAPABILITY_MODEL = {  # starting point only - model selection is D2.1
    "classification": MODEL_HAIKU,
    "extraction (structured outputs)": MODEL_SONNET,
    "vision": MODEL_SONNET,
    "PDF support": MODEL_SONNET,
    "RAG + citations": MODEL_SONNET,
    "tool use": "-",
    "summarization / generation": MODEL_SONNET,
    "agent loop": MODEL_OPUS,
}


def map_capabilities(f: ProblemFraming) -> list[tuple[str, str]]:
    caps: list[tuple[str, str]] = []
    if f.task_nature == "judgment_on_language":
        caps.append(("classification", "assign severity/route by judgment over language"))
        if f.data_shape != "structured":
            caps.append(("extraction (structured outputs)", "turn free text into typed fields for downstream code"))
    if "image" in f.input_modalities:
        caps.append(("vision", "read photos/screenshots directly"))
    if "pdf" in f.input_modalities:
        caps.append(("PDF support", "read reports/forms without a separate OCR step"))
    if f.needs_reference_knowledge:
        caps.append(("RAG + citations", "ground outputs in policy/runbook text and cite it"))
    if f.needs_external_systems:
        caps.append(("tool use", f"typed tools for: {', '.join(f.systems_involved)}"))
    if f.task_nature == "generation_or_summarization":
        caps.append(("summarization / generation", "condense many items into a narrative"))
    if f.task_nature == "open_ended_investigation":
        caps.append(("agent loop", "model chooses the next tool call until end_turn"))
    return caps


# ---------------------------------------------------------------------------
# 4. Tier decision table (first matching rule wins - order matters)
# ---------------------------------------------------------------------------
def decide_tier(f: ProblemFraming, a: Assessment, caps: list[tuple[str, str]]) -> tuple[str, str]:
    if f.task_nature == "deterministic_rules" and f.data_shape == "structured":
        return "not_llm", "R1 deterministic rules over structured data -> code / rules engine"
    if a.viability <= 2:
        return "not_llm", "R2 hard constraints (latency/data) an LLM cannot meet"
    if a.cost_of_error >= 5:
        return "not_llm", "R3 critical errors and no human checkpoint"
    if a.complexity >= 4 and not f.steps_predictable:
        if a.value >= TIER_MIN_VALUE["agent"]:
            return "agent", "R4 open-ended, unpredictable steps, value justifies agent cost"
        return "workflow", "R5 open-ended but value too low for an agent -> constrained workflow"
    if f.steps_predictable and (len(caps) >= 2 or f.needs_external_systems or a.complexity >= 3):
        if a.value >= TIER_MIN_VALUE["workflow"]:
            return "workflow", "R6 predictable multi-step / multi-capability -> workflow in code"
        return "single_call", "R6b multi-step but value too low -> start with one call"
    return "single_call", "R7 one well-scoped transformation -> single LLM call"


# ---------------------------------------------------------------------------
# 5. Blueprint assembly (deterministic) - the artefact you take into design review
# ---------------------------------------------------------------------------
@dataclass
class Blueprint:
    problem_id: str
    title: str
    tier: str
    components: list[str]
    controls: list[str]
    models: list[str]
    kpis: list[str]


def build_blueprint(pid: str, title: str, f: ProblemFraming, a: Assessment, tier: str,
                    caps: list[tuple[str, str]]) -> Blueprint:
    names = [c for c, _ in caps]
    if tier == "not_llm":
        components = [
            "Deterministic service (rules engine / tax API / SQL) at runtime",
            "Claude off the runtime path: Claude Code to maintain rule tables, tests and change summaries",
        ]
    elif tier == "single_call":
        components = ["Batch the inputs (code)", f"One Claude call: {', '.join(names)}", "Deliver output (code)"]
    elif tier == "workflow":
        # Group capabilities into the fewest sensible steps: one "perception" call reads all
        # modalities at once; retrieval and classification are separate, testable steps.
        components = ["Intake + normalisation + PII redaction (code)"]
        perceive = [n for n in names if n in ("extraction (structured outputs)", "vision", "PDF support")]
        if perceive:
            components.append(f"Claude step: {' + '.join(perceive)} -> typed fields")
        if "RAG + citations" in names:
            components.append("Retrieval step: RAG + citations over reference documents")
        if "classification" in names:
            components.append("Claude step: classification (small, fast model)")
        if "summarization / generation" in names:
            components.append("Claude step: summarization / generation")
        components.append("Routing rules + write-back via typed tool (code controls the flow)")
    else:  # agent
        components = [
            "Trigger -> context pack (alert, service, recent changes) (code)",
            f"Agent loop: Claude + read-only tools for {', '.join(f.systems_involved)}",
            "Step limit, token budget and transcript logging (harness)",
            "Hypothesis report -> human decides remediation",
        ]
    controls: list[str] = []
    joined = " ".join(f.constraints).lower()
    if tier != "not_llm" and a.cost_of_error >= 3:
        controls.append("Human-in-the-loop before any consequential action (see D5.3)")
    if tier == "agent":
        controls.append("Least-privilege tools: read-only by default; side effects gated by approval")
    if "audit" in joined:
        controls.append("Audit log: inputs, rule-table version, result" if tier == "not_llm"
                        else "Audit log: inputs, prompt version, model id, outputs, reviewer")
    if "personal" in joined or "pii" in joined:
        controls.append("PII minimisation/redaction and retention policy (see D5.4)")
    if "licensed" in joined or "regulator" in joined:
        controls.append("Final decision stays with an accountable licensed human")
    if tier != "not_llm":
        controls.append("Eval set with gold labels + go-live quality gate (see D4.2)")
    else:
        controls.append("Golden test suite for every jurisdiction/rule change")
    models = sorted({CAPABILITY_MODEL[n] for n in names if CAPABILITY_MODEL.get(n, "-") != "-"}) if tier != "not_llm" else []
    kpis = list(f.success_kpis)
    if a.annual_hours:
        kpis.append(f"baseline: {a.annual_hours:,.0f} h/yr of human handling")
    return Blueprint(pid, title, tier, components, controls, models, kpis)


# ---------------------------------------------------------------------------
# Offline responder: what Claude would return live
# ---------------------------------------------------------------------------
MOCK_FRAMINGS: dict[str, dict] = json.loads((DATA / "mock_framings.json").read_text(encoding="utf-8"))

BLUEPRINT_ONE_PAGER = """Solution blueprint - Motor claims triage (workflow tier)

Problem: triage adjusters spend ~25 minutes per FNOL claim; misroutes add 3-5 days to cycle time.

Design: a workflow, not an agent. The steps are known in advance, so control flow stays in code and
Claude is used only where judgment over unstructured content is needed.
  1. Intake normalises the claim and redacts personal data not needed for triage.
  2. Claude Sonnet reads the description, photos and police report (vision + PDF) and returns typed
     fields through structured outputs.
  3. Claude Haiku classifies severity and target team; the relevant policy wording is retrieved (RAG)
     so every coverage flag cites its clause.
  4. Code applies routing rules and writes to the claims system through a typed tool.
  5. An adjuster confirms or corrects each triage; corrections feed the eval set.

Value: ~60,000 adjuster hours/year addressable (~$3.3M). Target: under 5 minutes of human time per claim.
Controls: human confirmation, full audit log, PII redaction, licensed adjusters keep settlement authority.
Next step: 4-week pilot on 500 historical claims with gold labels; go/no-go on misrouting rate."""


def responder(req: MockRequest) -> str:
    text = req.last_user_text
    if "BLUEPRINT" in text:
        return BLUEPRINT_ONE_PAGER
    if "Compute the sales tax" in text:
        # A plausible-looking LLM answer that is off by one cent - exactly the failure mode
        # that makes probabilistic generation wrong for exact, audited arithmetic.
        return "Sales tax: $7.25 (combined rate 7.75% applied to $93.47)."
    match = re.search(r"Problem (P\d)", text)
    if match and match.group(1) in MOCK_FRAMINGS:
        return json.dumps(MOCK_FRAMINGS[match.group(1)])
    return "{}"


# ---------------------------------------------------------------------------
# Tiny deterministic rules engine for the P2 contrast (integer cents, half-up rounding)
# ---------------------------------------------------------------------------
RATES_BP = {"state": 625, "county": 100, "city": 50}  # basis points: 6.25% + 1.00% + 0.50%


def rules_engine_tax_cents(amount_cents: int) -> int:
    rate_bp = sum(RATES_BP.values())
    return (amount_cents * rate_bp + 5_000) // 10_000


def yn(flag: bool) -> str:
    return "yes" if flag else "no"


def main() -> None:
    banner("D1.1 - Translate Business Problems into Claude Solutions")
    client = get_client(responder)
    problems = json.loads((DATA / "problems.json").read_text(encoding="utf-8"))

    section("1. Intake: four business problems from discovery")
    for p in problems:
        print(f"  {p['id']}  {p['title']:<46} sponsor: {p['sponsor']}")
    print("  Raw notes are free text - the first job is to turn them into a comparable framing.")

    section("2. Structured problem framing (Claude + structured outputs)")
    framings: dict[str, ProblemFraming] = {}
    for p in problems:
        msg = client.messages.parse(
            model=MODEL_SONNET,
            max_tokens=2048,
            system=FRAMING_SYSTEM,
            output_format=ProblemFraming,           # SDK -> output_config.format (JSON schema)
            output_config={"effort": "medium"},     # merged with the format by the SDK
            messages=[{"role": "user", "content": f"Problem {p['id']} - {p['title']}\n\nInterview notes:\n{p['intake_notes']}"}],
        )
        if msg.stop_reason != "end_turn" or msg.parsed_output is None:
            print(f"  {p['id']}: framing failed (stop_reason={msg.stop_reason}) - send back to discovery")
            continue
        f = msg.parsed_output
        framings[p["id"]] = f
        print(f"  {p['id']} pain: {f.pain_point}")
        print(f"      users={', '.join(f.primary_users)} | volume={f.monthly_volume:,}/mo | "
              f"today={f.minutes_per_item_today:g} min/item @ ${f.loaded_cost_per_hour_usd:g}/h")
        print(f"      nature={f.task_nature} | data={f.data_shape} ({'+'.join(f.input_modalities)}) | "
              f"predictable={yn(f.steps_predictable)} | latency budget={f.latency_budget_ms:,} ms")
        print(f"      severity={f.error_severity} | human review feasible={yn(f.human_review_feasible)} | "
              f"KPIs: {'; '.join(f.success_kpis)}")
        print(f"      [{usage_line(msg)}]")

    section("3. AI-suitability rubric (deterministic, 1-5 each)")
    print("  Criteria: complexity (is an LLM needed?), value (is it worth it?), viability (can it meet")
    print("  the constraints?), cost of error (what happens when it is wrong? higher = riskier)")
    print(f"  {'id':<4}{'complexity':>11}{'value':>7}{'$/yr addressable':>18}{'viability':>11}{'cost-of-error':>15}")
    assessments: dict[str, Assessment] = {}
    for pid, f in framings.items():
        a = assess(f)
        assessments[pid] = a
        print(f"  {pid:<4}{a.complexity:>11}{a.value:>7}{'$' + format(round(a.annual_value_usd), ','):>18}"
              f"{a.viability:>11}{a.cost_of_error:>15}")
    for pid, a in assessments.items():
        for note in a.notes:
            print(f"    {pid}: {note}")

    section("4. Capability mapping + tier decision")
    titles = {p["id"]: p["title"] for p in problems}
    decisions: dict[str, tuple[str, list[tuple[str, str]]]] = {}
    for pid, f in framings.items():
        caps = map_capabilities(f)
        tier, rule = decide_tier(f, assessments[pid], caps)
        decisions[pid] = (tier, caps)
        print(f"  {pid} {titles[pid]}")
        print(f"     tier -> {tier.upper():<12} ({rule})")
        if tier == "not_llm":
            print("     capabilities -> none at runtime")
        else:
            for name, why in caps:
                print(f"     - {name:<32} {why}")

    section("5. Solution blueprints")
    blueprints: dict[str, Blueprint] = {}
    for pid, f in framings.items():
        tier, caps = decisions[pid]
        bp = build_blueprint(pid, titles[pid], f, assessments[pid], tier, caps)
        blueprints[pid] = bp
        print(f"  [{pid}] {bp.title} - {bp.tier.upper()}")
        for i, c in enumerate(bp.components, 1):
            print(f"     {i}. {c}")
        print(f"     controls: {' | '.join(bp.controls)}")
        if bp.models:
            print(f"     starting models (validate in D2.1): {', '.join(bp.models)}")
        print(f"     KPIs: {'; '.join(bp.kpis)}")

    section("6. Claude drafts the one-pager for the top LLM candidate")
    candidates = [pid for pid in framings if decisions[pid][0] != "not_llm"]
    top = max(candidates, key=lambda pid: assessments[pid].annual_value_usd)
    payload = {
        "blueprint": blueprints[top].__dict__,
        "framing": framings[top].model_dump(),
        "assessment": {k: v for k, v in assessments[top].__dict__.items() if k != "notes"},
    }
    msg = client.messages.create(
        model=MODEL_OPUS,
        max_tokens=1500,
        system="You are a principal solution architect. Write crisp, decision-ready one-pagers for executives. "
               "Use only the facts in the payload; do not invent numbers.",
        messages=[{"role": "user", "content": "Draft a BLUEPRINT one-pager (<= 220 words) from this payload:\n"
                   + json.dumps(payload, indent=2)}],
    )
    if msg.stop_reason == "end_turn":
        for line in text_of(msg).splitlines():
            print(f"  | {line}")
    else:
        print(f"  draft incomplete (stop_reason={msg.stop_reason}) - raise max_tokens or shorten the payload")
    print(f"  [{usage_line(msg)}]")

    section("7. Why P2 is NOT an LLM problem - the numbers")
    amount_cents = 9_347
    exact = rules_engine_tax_cents(amount_cents)
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=MODEL_HAIKU,
        max_tokens=200,
        messages=[{"role": "user", "content": f"Compute the sales tax for an order of ${amount_cents / 100:.2f} "
                   "shipped to a jurisdiction with 6.25% state, 1.00% county and 0.50% city tax."}],
    )
    llm_ms = elapsed_ms(t0)
    print(f"  rules engine : ${exact / 100:.2f}  (integer cents, half-up rounding, ~microseconds, exact every time)")
    print(f"  Claude Haiku : {text_of(msg)}")
    print(f"                 latency ~{round(llm_ms, -1):.0f} ms vs a 50 ms budget; result not guaranteed to the cent")
    monthly_calls = framings["P2"].monthly_volume if "P2" in framings else 60_000_000
    per_call = {"input_tokens": 600, "output_tokens": 60}
    for model in (MODEL_HAIKU, MODEL_SONNET):
        print(f"  LLM at runtime on {model:<18}: ~${cost_usd(model, per_call) * monthly_calls:,.0f}/month "
              f"for {monthly_calls:,} calls (600 in / 60 out tokens each)")
    print("  Verdict: exact, structured, high-volume, sub-50 ms, unreviewable -> rules engine at runtime.")
    print("  Claude still adds value OFF the hot path (Claude Code to update rate tables and generate tests).")

    section("Summary")
    for pid in framings:
        tier = decisions[pid][0]
        a = assessments[pid]
        print(f"  {pid} {titles[pid]:<46} -> {tier.upper():<12} (C{a.complexity} V{a.value} "
              f"Vi{a.viability} E{a.cost_of_error})")
    print("  Lesson: start from the business problem, score it transparently, choose the SIMPLEST tier")
    print("  that works - and be willing to say 'this is not an LLM problem'.")


if __name__ == "__main__":
    main()
