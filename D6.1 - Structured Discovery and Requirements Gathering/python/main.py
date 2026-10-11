"""
D6.1 - Structured Discovery and Requirements Gathering

What this lab builds (a discovery toolkit an architect can actually run):

  interview guide (data/interview_guide.json)  +  3 raw stakeholder transcripts
     -> Claude extracts ONE typed record per interview (structured outputs, verbatim evidence)
     -> code: provenance check (is every quote really in the transcript?)
     -> code: merge + de-duplicate requirements across stakeholders (keeps attribution)
     -> code: gap analysis - vague NFRs, missing owners, stakeholder conflicts,
              guide coverage, missing baselines
     -> code: follow-up questions, routed to the right stakeholder
     -> Claude drafts value/effort estimates; code applies MoSCoW + value/effort rules
        and the DSDM "Must <= 60% of effort" check
     -> code: use-case feasibility / readiness scoring
     -> code: renders a requirements document (labs_output/requirements_document.md)

Design principle: Claude does the language work (reading messy interviews), code does
the judgement that must be repeatable and defensible (rules, scoring, routing).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import MODEL_OPUS, MockRequest, banner, get_client, section, usage_line  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
OUT = LAB / "labs_output"

# ---------------------------------------------------------------------------
# 1. Extraction contract. One record per interview keeps provenance: every
#    requirement can be traced to a stakeholder and a verbatim quote.
#    Optional[...] = "nullable": the model must say null rather than invent a number.
# ---------------------------------------------------------------------------
Topic = Literal["human_review_policy", "phase1_scope", "go_live_date", "ai_budget", "latency_target"]


class Stakeholder(BaseModel):
    name: str
    role: str
    department: str


class BusinessGoal(BaseModel):
    statement: str
    metric: Optional[str]
    target: Optional[str]


class Persona(BaseModel):
    name: str
    needs: str


class Volume(BaseModel):
    metric: str
    value: float
    unit: str


class DataSource(BaseModel):
    name: str
    sensitivity: Literal["public", "internal", "confidential", "restricted"]
    contains_pii: bool


class Integration(BaseModel):
    system: str
    direction: Literal["read", "write", "read_write"]
    mechanism: str


class FunctionalReq(BaseModel):
    id: str
    statement: str
    priority_hint: Literal["must", "should", "could", "wont", "unspecified"]
    owner: Optional[str] = Field(description="Team/person explicitly named as accountable, else null")
    related_topic: Optional[Topic]
    source_quote: str = Field(description="Verbatim sentence from the transcript")


class NonFunctionalReq(BaseModel):
    id: str
    category: Literal[
        "latency", "availability", "cost", "quality", "security", "privacy", "compliance", "scalability", "usability"
    ]
    statement: str
    metric: Optional[str]
    target_value: Optional[float] = Field(description="Numeric target ONLY if stated; never guessed")
    unit: Optional[str]
    source_quote: str


class Constraint(BaseModel):
    category: Literal["regulatory", "budget", "technical", "timeline", "organizational"]
    statement: str
    source_quote: str


class Position(BaseModel):
    topic: Topic
    stance: str
    normalized_value: str = Field(description="Short comparable label, e.g. 2027-03-01 or exclude_injury_claims")


class SuccessMetric(BaseModel):
    name: str
    baseline: Optional[str]
    target: str


class InterviewExtraction(BaseModel):
    stakeholder: Stakeholder
    business_goals: list[BusinessGoal]
    personas: list[Persona]
    pain_points: list[str]
    volumes: list[Volume]
    data_sources: list[DataSource]
    integrations: list[Integration]
    functional: list[FunctionalReq]
    non_functional: list[NonFunctionalReq]
    constraints: list[Constraint]
    positions: list[Position]
    success_metrics: list[SuccessMetric]
    escalation_paths: list[str]
    risks: list[str]
    assumptions: list[str]
    open_questions: list[str]


class Estimate(BaseModel):
    req_id: str
    value: int = Field(ge=1, le=5, description="Business value 1-5")
    effort: int = Field(ge=1, le=5, description="Relative delivery effort 1-5")
    moscow: Literal["must", "should", "could", "wont"]
    rationale: str


class PrioritizationDraft(BaseModel):
    estimates: list[Estimate]
    notes: str


EXTRACTION_SYSTEM = """You are a senior requirements analyst on a Claude solution-architecture team.
Turn one raw stakeholder interview transcript into a typed requirements record.
Rules:
- Extract only what this stakeholder actually said. Put the supporting sentence, copied verbatim, in source_quote.
- Never invent numbers. If a non-functional requirement has no measurable target in the transcript,
  set metric, target_value and unit to null. Vague words ("fast", "accurate", "scale") stay vague.
- owner = the team or person explicitly named as accountable; null if unstated or uncertain.
- positions: the stakeholder's stance on these decision topics, only if expressed: human_review_policy,
  phase1_scope, go_live_date, ai_budget, latency_target. normalized_value is a short comparable label.
- escalation_paths: what should happen when the AI is unsure or wrong (empty if not discussed).
- open_questions: anything the stakeholder was unsure about or deferred."""

PRIORITIZATION_SYSTEM = """You are the delivery lead on a Claude solution team. For each requirement give a
relative business value (1-5) and delivery effort (1-5), and a MoSCoW suggestion with a one-line rationale.
Respect regulatory constraints: a requirement that contradicts a stated regulatory constraint cannot be a Must.
Flag requirements that depend on open stakeholder conflicts in the rationale. Estimates are a draft for the team."""


# ---------------------------------------------------------------------------
# 2. Offline responder: canned Claude replies live in data/mock_responses so
#    Python and TypeScript share the exact same fixtures.
# ---------------------------------------------------------------------------
def responder(req: MockRequest) -> str:
    if "requirements analyst" in req.system_text:
        m = re.search(r"Interview ID: (INT-\d+)", req.last_user_text)
        iid = m.group(1) if m else "INT-01"
        return (DATA / "mock_responses" / f"extraction_{iid}.json").read_text(encoding="utf-8")
    if "delivery lead" in req.system_text:
        return (DATA / "mock_responses" / "prioritization.json").read_text(encoding="utf-8")
    return "{}"


# ---------------------------------------------------------------------------
# 3. Small deterministic helpers (identical logic in the TypeScript version)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "the", "and", "for", "from", "with", "into", "that", "this", "our", "any", "all", "each", "every",
    "before", "after", "not", "must", "should", "can", "will", "using", "their", "they", "them", "its",
}


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def content_tokens(s: str) -> set[str]:
    out: set[str] = set()
    for t in re.findall(r"[a-z0-9]+", s.lower()):
        if len(t) < 3 or t in STOPWORDS:
            continue
        if len(t) > 4 and t.endswith("s"):
            t = t[:-1]  # naive plural stemming: letters -> letter
        out.add(t)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def fill(template: str, values: dict[str, str]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return out


def find_vague_term(text: str, terms: list[str]) -> Optional[str]:
    low = text.lower()
    for term in terms:
        if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", low):
            return term
    return None


def trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def md_cell(s: str) -> str:
    return s.replace("|", "/")


# ---------------------------------------------------------------------------
# 4. Merged (cross-stakeholder) view
# ---------------------------------------------------------------------------
DEDUPE_THRESHOLD = 0.5


@dataclass
class MergedFR:
    id: str
    statement: str
    owner: Optional[str]
    related_topic: Optional[str]
    tokens: set[str]
    sources: list[str] = field(default_factory=list)  # "INT-01/FR-1"
    interviews: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    quotes: list[tuple[str, str]] = field(default_factory=list)  # (interview, quote)


@dataclass
class MergedNFR:
    id: str
    interview: str
    local_id: str
    item: NonFunctionalReq
    measurable: bool = True


@dataclass
class Gap:
    kind: str
    severity: str
    ref: str
    detail: str
    ask: str  # interview id, "joint" or "all"
    question: str


@dataclass
class Conflict:
    id: str
    topic: str
    label: str
    severity: str
    owner: str
    hint: str
    positions: list[tuple[str, str, str]]  # (interview, stance, normalized_value)


@dataclass
class Prioritized:
    fr: MergedFR
    value: int
    effort: int
    moscow: str
    rationale: str
    quadrant: str
    score: float
    blocked_by: Optional[str]


@dataclass
class UseCaseScore:
    id: str
    name: str
    value: int
    feasibility: float
    risk: float
    readiness: float
    decision: str
    blockers: list[str]


MOSCOW_RANK = {"must": 0, "should": 1, "could": 2, "wont": 3}
MOSCOW_LABEL = {"must": "Must", "should": "Should", "could": "Could", "wont": "Won't"}


def merge_functional(extractions: dict[str, InterviewExtraction]) -> tuple[list[MergedFR], list[str]]:
    merged: list[MergedFR] = []
    log: list[str] = []
    for iid, ex in extractions.items():
        for fr in ex.functional:
            toks = content_tokens(fr.statement)
            best, best_sim = None, 0.0
            for m in merged:
                sim = jaccard(toks, m.tokens)
                if sim > best_sim:
                    best, best_sim = m, sim
            if best is not None and best_sim >= DEDUPE_THRESHOLD:
                best.sources.append(f"{iid}/{fr.id}")
                if iid not in best.interviews:
                    best.interviews.append(iid)
                best.hints.append(fr.priority_hint)
                best.quotes.append((iid, fr.source_quote))
                best.owner = best.owner or fr.owner
                best.related_topic = best.related_topic or fr.related_topic
                log.append(f"{iid}/{fr.id} merged into {best.id} (similarity {best_sim:.2f}) - same need, 2nd source")
                continue
            m = MergedFR(
                id=f"FR-{len(merged) + 1:02d}", statement=fr.statement, owner=fr.owner,
                related_topic=fr.related_topic, tokens=toks,
            )
            m.sources.append(f"{iid}/{fr.id}")
            m.interviews.append(iid)
            m.hints.append(fr.priority_hint)
            m.quotes.append((iid, fr.source_quote))
            merged.append(m)
    return merged, log


def detect_conflicts(extractions: dict[str, InterviewExtraction], guide: dict[str, Any]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    for topic, meta in guide["conflict_topics"].items():
        positions = [
            (iid, p.stance, p.normalized_value) for iid, ex in extractions.items() for p in ex.positions if p.topic == topic
        ]
        distinct: list[str] = []
        for _, _, v in positions:
            if v not in distinct:
                distinct.append(v)
        if len(distinct) >= 2:
            conflicts.append(Conflict(
                id=f"C-{len(conflicts) + 1}", topic=topic, label=meta["label"], severity=meta["severity"],
                owner=meta["decision_owner"], hint=meta["compromise_hint"], positions=positions,
            ))
    return conflicts


def coverage_matrix(extractions: dict[str, InterviewExtraction], guide: dict[str, Any]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for s in guide["sections"]:
        row: dict[str, int] = {}
        for iid, ex in extractions.items():
            row[iid] = sum(len(getattr(ex, f)) for f in s["evidence_fields"])
        matrix[s["id"]] = row
    return matrix


def main() -> None:
    banner("D6.1 - Structured Discovery and Requirements Gathering")
    guide = json.loads((DATA / "interview_guide.json").read_text(encoding="utf-8"))
    use_cases = json.loads((DATA / "use_cases.json").read_text(encoding="utf-8"))
    tpl = guide["followup_templates"]
    client = get_client(responder)

    # -- 1. Toolkit -----------------------------------------------------------
    section("1. Discovery toolkit: structured interview guide")
    print(f"Project: {guide['project']} | Client: {guide['client']} | As of {guide['as_of']}")
    for s in guide["sections"]:
        print(f"  [{s['id']:<17}] {s['title']:<52} {len(s['questions'])} q -> {', '.join(s['evidence_fields'])}")
    print("Each guide section maps to typed schema fields, so coverage is measured - not remembered.")

    # -- 2. Extraction (Claude, structured outputs) -----------------------------
    section("2. Claude extracts a typed record per interview")
    transcripts: dict[str, str] = {}
    extractions: dict[str, InterviewExtraction] = {}
    for it in guide["interviews"]:
        iid = it["id"]
        text = (DATA / "transcripts" / it["file"]).read_text(encoding="utf-8")
        transcripts[iid] = text
        # Strongest model: low volume (a handful of interviews), high stakes (it shapes the whole project).
        resp = client.messages.parse(
            model=MODEL_OPUS,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=EXTRACTION_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Extract the requirements record for this interview.\n\n<transcript>\n{text}\n</transcript>",
            }],
            output_format=InterviewExtraction,
        )
        ex = resp.parsed_output
        if ex is None:
            raise RuntimeError(f"{iid}: structured output did not parse")
        extractions[iid] = ex
        print(
            f"  {iid} {ex.stakeholder.name:<16} {len(ex.functional)} FR, {len(ex.non_functional)} NFR, "
            f"{len(ex.constraints)} constraints, {len(ex.positions)} positions | {usage_line(resp)}"
        )
    names = {iid: ex.stakeholder.name for iid, ex in extractions.items()}

    # -- 3. Provenance + merge ---------------------------------------------------
    section("3. Provenance check and cross-stakeholder merge")
    quote_checks: list[tuple[str, str, str, bool]] = []  # (interview, ref, quote, verified)
    for iid, ex in extractions.items():
        body = norm_text(transcripts[iid])
        items: list[tuple[str, str]] = [(f.id, f.source_quote) for f in ex.functional]
        items += [(n.id, n.source_quote) for n in ex.non_functional]
        items += [(f"CON-{i + 1}", c.source_quote) for i, c in enumerate(ex.constraints)]
        for ref, quote in items:
            quote_checks.append((iid, f"{iid}/{ref}", quote, norm_text(quote) in body))
    verified = sum(1 for q in quote_checks if q[3])
    print(f"Evidence quotes found verbatim in transcripts: {verified}/{len(quote_checks)}")
    for iid, ref, quote, ok in quote_checks:
        if not ok:
            print(f"  ! {ref}: \"{quote}\" is a paraphrase - not verbatim (flagged for confirmation)")

    merged, merge_log = merge_functional(extractions)
    total_fr = sum(len(ex.functional) for ex in extractions.values())
    print(f"Functional requirements: {total_fr} extracted -> {len(merged)} after de-duplication "
          f"(Jaccard >= {DEDUPE_THRESHOLD:.2f} on content words)")
    for line in merge_log:
        print(f"  {line}")
    nfrs: list[MergedNFR] = []
    for iid, ex in extractions.items():
        for n in ex.non_functional:
            nfrs.append(MergedNFR(id=f"NFR-{len(nfrs) + 1:02d}", interview=iid, local_id=n.id, item=n))

    # -- 4. Gap analysis -----------------------------------------------------
    section("4. Gap analysis")
    meas = guide["measurability"]
    gaps: list[Gap] = []
    conflicts = detect_conflicts(extractions, guide)
    for c in conflicts:
        a, b = c.positions[0], c.positions[1]
        q = fill(tpl["conflict"], {
            "a_name": names[a[0]], "a_stance": a[1], "b_name": names[b[0]], "b_stance": b[1],
            "owner": c.owner, "hint": c.hint,
        })
        gaps.append(Gap("conflict", c.severity, c.id, f"{c.label}: " + " vs ".join(p[2] for p in c.positions), "joint", q))

    matrix = coverage_matrix(extractions, guide)
    for s in guide["sections"]:
        if sum(matrix[s["id"]].values()) == 0:
            q = fill(tpl["coverage"], {"title": s["title"], "question": s["questions"][0]})
            gaps.append(Gap("coverage", "high", s["id"], f"No interview covered '{s['title']}'", "all", q))

    for n in nfrs:
        it = n.item
        if it.target_value is not None:
            continue
        term = find_vague_term(it.statement + " " + it.source_quote, meas["vague_terms"])
        if term is not None:
            n.measurable = False
            q = fill(tpl["vague_nfr"], {"quote": it.source_quote, "category": it.category})
            gaps.append(Gap("vague_nfr", "high", n.id, f"'{term}' with no target ({it.category}): {it.statement}", n.interview, q))
        elif it.category in meas["categories_requiring_metric"]:
            n.measurable = False
            q = fill(tpl["unmeasured_nfr"], {"statement": it.statement, "category": it.category})
            gaps.append(Gap("unmeasured_nfr", "medium", n.id, f"{it.category} requirement without metric: {it.statement}", n.interview, q))

    for iid, ref, quote, ok in quote_checks:
        if not ok:
            q = fill(tpl["unverified_quote"], {"ref": ref, "quote": quote})
            gaps.append(Gap("unverified_quote", "medium", ref, "Evidence quote not verbatim in transcript", iid, q))

    for m in merged:
        if m.owner is None:
            q = fill(tpl["missing_owner"], {"req_id": m.id, "statement": m.statement})
            gaps.append(Gap("missing_owner", "medium", m.id, f"No accountable owner: {m.statement}", guide["sponsor_interview"], q))

    for iid, ex in extractions.items():
        for sm in ex.success_metrics:
            if sm.baseline is None:
                q = fill(tpl["no_baseline"], {"metric": sm.name})
                gaps.append(Gap("no_baseline", "medium", iid, f"Success metric without baseline: {sm.name}", iid, q))

    kinds = ["conflict", "coverage", "vague_nfr", "unmeasured_nfr", "unverified_quote", "missing_owner", "no_baseline"]
    for k in kinds:
        items = [g for g in gaps if g.kind == k]
        if items:
            print(f"  {k:<17} x{len(items)}: " + ", ".join(g.ref for g in items))
    print("\nStakeholder conflicts (same topic, different positions):")
    for c in conflicts:
        print(f"  {c.id} [{c.severity:<6}] {c.label}")
        for iid, stance, value in c.positions:
            print(f"      {iid} {names[iid]:<16} -> {value}")
    print("\nGuide coverage (evidence items per interview):")
    iids = list(extractions.keys())
    print(f"  {'section':<18}" + "".join(f"{i:>8}" for i in iids) + "   status")
    for s in guide["sections"]:
        row = matrix[s["id"]]
        status = "GAP" if sum(row.values()) == 0 else ("single-source" if sum(1 for v in row.values() if v) == 1 else "ok")
        print(f"  {s['id']:<18}" + "".join(f"{row[i]:>8}" for i in iids) + f"   {status}")

    # -- 5. Follow-up questions ---------------------------------------------------
    section("5. Follow-up questions, routed to who can answer them")
    audiences = iids + ["joint", "all"]
    audience_label = {iid: f"{names[iid]} ({extractions[iid].stakeholder.role})" for iid in iids}
    audience_label["joint"] = "Joint decision session (steering committee)"
    audience_label["all"] = "All stakeholders (next workshop)"
    followups: dict[str, list[Gap]] = {a: [g for g in gaps if g.ask == a] for a in audiences}
    for a in audiences:
        if not followups[a]:
            continue
        print(f"{audience_label[a]} - {len(followups[a])} question(s)")
        for i, g in enumerate(followups[a], 1):
            print(f"  {i}. [{g.ref}] {trunc(g.question, 150)}")

    # -- 6. Prioritization ---------------------------------------------------------
    section("6. Prioritization: Claude drafts estimates, code applies the rules")
    open_topics = {c.topic: c for c in conflicts}
    payload = {
        "requirements": [
            {"id": m.id, "statement": m.statement, "stakeholder_priority_hints": m.hints,
             "sources": m.sources, "depends_on_open_conflict": m.related_topic in open_topics}
            for m in merged
        ],
        "business_goals": [g.statement for ex in extractions.values() for g in ex.business_goals],
        "regulatory_constraints": [c.statement for ex in extractions.values() for c in ex.constraints if c.category == "regulatory"],
    }
    resp = client.messages.parse(
        model=MODEL_OPUS,
        max_tokens=6000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=PRIORITIZATION_SYSTEM,
        messages=[{"role": "user", "content": "Estimate these requirements:\n" + json.dumps(payload, indent=2)}],
        output_format=PrioritizationDraft,
    )
    draft = resp.parsed_output
    if draft is None:
        raise RuntimeError("prioritization did not parse")
    print(f"  Claude estimates received | {usage_line(resp)}")
    by_id = {e.req_id: e for e in draft.estimates}
    missing = [m.id for m in merged if m.id not in by_id]
    unknown = [e.req_id for e in draft.estimates if e.req_id not in {m.id for m in merged}]
    print(f"  Validation: {len(by_id)} estimates, missing={missing or 'none'}, unknown ids={unknown or 'none'}")

    pr_cfg = guide["prioritization"]
    prioritized: list[Prioritized] = []
    for m in merged:
        e = by_id.get(m.id)
        if e is None:
            continue
        hv, le = e.value >= pr_cfg["high_value_min"], e.effort <= pr_cfg["low_effort_max"]
        quadrant = "Quick win" if hv and le else "Major project" if hv else "Fill-in" if le else "Money pit"
        blocked = open_topics[m.related_topic].label if m.related_topic in open_topics else None
        prioritized.append(Prioritized(m, e.value, e.effort, e.moscow, e.rationale, quadrant, e.value / e.effort, blocked))
    prioritized.sort(key=lambda p: (MOSCOW_RANK[p.moscow], -p.score, p.fr.id))

    print(f"\n  {'ID':<6}{'MoSCoW':<8}{'V':>2}{'E':>3}{'V/E':>6}  {'Quadrant':<14}{'Owner':<19}Requirement")
    for p in prioritized:
        flag = "  <- DECISION NEEDED" if p.blocked_by else ""
        print(f"  {p.fr.id:<6}{p.moscow.upper():<8}{p.value:>2}{p.effort:>3}{p.score:>6.2f}  {p.quadrant:<14}"
              f"{trunc(p.fr.owner or '(none)', 18):<19}{trunc(p.fr.statement, 52)}{flag}")

    in_scope = [p for p in prioritized if p.moscow != "wont"]
    total_effort = sum(p.effort for p in in_scope)
    must_effort = sum(p.effort for p in in_scope if p.moscow == "must")
    share = must_effort / total_effort if total_effort else 0.0
    limit = pr_cfg["must_effort_share_max"]
    dsdm_ok = share <= limit
    print(f"\n  DSDM check: Must = {must_effort}/{total_effort} effort points = {share * 100:.1f}% "
          f"(guideline <= {limit * 100:.0f}%) -> {'OK' if dsdm_ok else 'OVER - no contingency left'}")
    demotions: list[str] = []
    if not dsdm_ok:
        # Demote contested Musts first (largest effort first), then the lowest value/effort Musts.
        musts = [p for p in in_scope if p.moscow == "must"]
        candidates = sorted([p for p in musts if p.blocked_by], key=lambda p: (-p.effort, p.fr.id))
        candidates += sorted([p for p in musts if not p.blocked_by], key=lambda p: (p.score, p.fr.id))
        remaining = must_effort
        for p in candidates:
            if remaining / total_effort <= limit:
                break
            remaining -= p.effort
            demotions.append(p.fr.id)
        print(f"  Recommendation: move {', '.join(demotions)} from Must to phase 2 / Should -> Must share "
              f"{remaining / total_effort * 100:.1f}%")

    # -- 7. Use-case feasibility ------------------------------------------------
    section("7. Use-case feasibility and readiness scoring")
    w, th = use_cases["weights"], use_cases["thresholds"]
    uc_scores: list[UseCaseScore] = []
    for uc in use_cases["use_cases"]:
        feasibility = (uc["data_readiness"] + uc["technical_fit"] + (6 - uc["integration_complexity"])) / 3
        # A human in the loop reduces the *effective* impact of an error (but never below 1).
        err = max(1, uc["error_impact"] - 2) if uc["human_in_loop"] else uc["error_impact"]
        risk = (uc["data_sensitivity"] + err + uc["regulatory_exposure"]) / 3
        readiness = w["value"] * uc["value"] + w["feasibility"] * feasibility + w["safety"] * (6 - risk)
        blockers: list[str] = []
        if uc["hard_constraint_conflicts"]:
            decision = "NO-GO"
            blockers = [f"violates hard constraint: {c}" for c in uc["hard_constraint_conflicts"]]
        else:
            if uc["data_readiness"] < th["min_data_readiness"]:
                blockers.append(f"data readiness {uc['data_readiness']} < {th['min_data_readiness']}")
            if not uc["measurable_success_metric"]:
                blockers.append("no measurable success metric")
            if risk >= th["max_risk_for_go"]:
                blockers.append(f"risk {risk:.2f} >= {th['max_risk_for_go']:.1f}: mitigation plan needed")
            if readiness >= th["go_readiness"] and not blockers:
                decision = "GO (pilot)"
            elif readiness >= th["conditional_readiness"]:
                decision = "CONDITIONAL"
            else:
                decision = "NOT NOW"
        uc_scores.append(UseCaseScore(uc["id"], uc["name"], uc["value"], feasibility, risk, readiness, decision, blockers))
    print(f"  readiness = {w['value']}*value + {w['feasibility']}*feasibility + {w['safety']}*(6 - risk)   (1-5 scale)")
    print(f"  {'UC':<6}{'Use case':<52}{'Val':>4}{'Feas':>6}{'Risk':>6}{'Ready':>7}  Decision")
    for u in uc_scores:
        print(f"  {u.id:<6}{trunc(u.name, 50):<52}{u.value:>4}{u.feasibility:>6.2f}{u.risk:>6.2f}{u.readiness:>7.2f}  {u.decision}")
        for b in u.blockers:
            print(f"        - {b}")

    # -- 8. Discovery exit criteria + document -------------------------------------
    section("8. Discovery exit check and requirements document")
    must_unowned = [p.fr.id for p in prioritized if p.moscow == "must" and p.fr.owner is None]
    exit_checks = [
        ("No open high-severity stakeholder conflicts", not any(c.severity == "high" for c in conflicts)),
        ("Every interview-guide section covered", not any(g.kind == "coverage" for g in gaps)),
        ("Every NFR measurable (metric + target)", all(n.measurable for n in nfrs)),
        ("Every Must requirement has an owner", not must_unowned),
        ("MoSCoW within DSDM effort guideline", dsdm_ok),
        ("All evidence quotes verified", verified == len(quote_checks)),
    ]
    for label, ok in exit_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    exit_met = all(ok for _, ok in exit_checks)
    print(f"  => Discovery exit {'MET' if exit_met else 'NOT MET'}: "
          f"{sum(1 for _, ok in exit_checks if not ok)} criteria open - schedule the follow-ups above.")

    doc = render_document(
        guide, extractions, merged, nfrs, conflicts, gaps, followups, audience_label, prioritized,
        (must_effort, total_effort, share, limit, dsdm_ok, demotions), uc_scores, use_cases, matrix,
        quote_checks, exit_checks,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "requirements_document.md"
    out_path.write_text(doc, encoding="utf-8")
    print(f"\nWrote labs_output/{out_path.name} ({len(doc.splitlines())} lines)")
    print("Sample output committed at examples/requirements_document.md")


# ---------------------------------------------------------------------------
# 5. Requirements document (deterministic template rendering)
# ---------------------------------------------------------------------------
def render_document(
    guide: dict[str, Any],
    extractions: dict[str, InterviewExtraction],
    merged: list[MergedFR],
    nfrs: list[MergedNFR],
    conflicts: list[Conflict],
    gaps: list[Gap],
    followups: dict[str, list[Gap]],
    audience_label: dict[str, str],
    prioritized: list[Prioritized],
    dsdm: tuple[int, int, float, float, bool, list[str]],
    uc_scores: list[UseCaseScore],
    use_cases: dict[str, Any],
    matrix: dict[str, dict[str, int]],
    quote_checks: list[tuple[str, str, str, bool]],
    exit_checks: list[tuple[str, bool]],
) -> str:
    must_effort, total_effort, share, limit, dsdm_ok, demotions = dsdm
    names = {iid: ex.stakeholder.name for iid, ex in extractions.items()}
    iids = list(extractions.keys())
    counts = {k: sum(1 for p in prioritized if p.moscow == k) for k in MOSCOW_RANK}
    measurable = sum(1 for n in nfrs if n.measurable)
    n_constraints = sum(len(ex.constraints) for ex in extractions.values())
    high_conflicts = sum(1 for c in conflicts if c.severity == "high")
    L: list[str] = []
    a = L.append
    a(f"# Requirements Document - {guide['project']}")
    a("")
    a(f"**Client:** {guide['client']}  ")
    a(f"**Status:** DRAFT v0.1 (discovery)  ")
    a(f"**As of:** {guide['as_of']}  ")
    a(f"**Sources:** {len(iids)} stakeholder interviews ({', '.join(iids)})")
    a("")
    a("> Generated by the D6.1 lab. Claude extracted one typed record per interview (structured outputs);")
    a("> merging, gap analysis, prioritization rules and scoring are deterministic code. Estimates are drafts")
    a("> for the delivery team to confirm.")
    a("")
    a("## 1. Summary")
    a("")
    a(f"- **{len(merged)} functional requirements**: {counts['must']} Must, {counts['should']} Should, "
      f"{counts['could']} Could, {counts['wont']} Won't (this phase)")
    a(f"- **{len(nfrs)} non-functional requirements**: {measurable} measurable, {len(nfrs) - measurable} need a target")
    a(f"- **{n_constraints} constraints**, **{len(conflicts)} stakeholder conflicts** ({high_conflicts} high severity), "
      f"**{len(gaps)} open gaps**")
    exit_open = [label for label, ok in exit_checks if not ok]
    a(f"- **Discovery exit: {'MET' if not exit_open else 'NOT MET'}** - open: " + ("; ".join(exit_open) if exit_open else "none"))
    a("")
    a("## 2. Stakeholders")
    a("")
    a("| Interview | Name | Role | Department |")
    a("|---|---|---|---|")
    for iid, ex in extractions.items():
        s = ex.stakeholder
        a(f"| {iid} | {s.name} | {s.role} | {s.department} |")
    a("")
    a("## 3. Business goals and success metrics")
    a("")
    a("| Goal | Metric | Target | Source |")
    a("|---|---|---|---|")
    for iid, ex in extractions.items():
        for g in ex.business_goals:
            a(f"| {md_cell(g.statement)} | {g.metric or '_not stated_'} | {g.target or '_not stated_'} | {iid} |")
    a("")
    a("| Success metric | Baseline | Target | Source |")
    a("|---|---|---|---|")
    for iid, ex in extractions.items():
        for sm in ex.success_metrics:
            a(f"| {sm.name} | {sm.baseline or '**MISSING**'} | {sm.target} | {iid} |")
    a("")
    a("## 4. Functional requirements (prioritized)")
    a("")
    a("MoSCoW from the delivery-lead draft; value/effort on a 1-5 scale; quadrant: Quick win (V>=4, E<=2), "
      "Major project (V>=4, E>=3), Fill-in (V<4, E<=2), Money pit (V<4, E>=3).")
    a("")
    a("| ID | Requirement | MoSCoW | Value | Effort | V/E | Quadrant | Owner | Sources | Status |")
    a("|---|---|---|---:|---:|---:|---|---|---|---|")
    for p in prioritized:
        status = f"Decision needed: {p.blocked_by}" if p.blocked_by else "Ready"
        a(f"| {p.fr.id} | {md_cell(p.fr.statement)} | {MOSCOW_LABEL[p.moscow]} | {p.value} | {p.effort} | {p.score:.2f} | "
          f"{p.quadrant} | {p.fr.owner or '**MISSING**'} | {', '.join(p.fr.sources)} | {status} |")
    a("")
    verdict = "within guideline" if dsdm_ok else "OVER guideline - no contingency left"
    a(f"**DSDM check:** Must = {must_effort}/{total_effort} effort points = {share * 100:.1f}% "
      f"(guideline <= {limit * 100:.0f}%) - {verdict}.")
    if demotions:
        a(f"Recommendation: move {', '.join(demotions)} out of Must (contested and/or lowest value per effort).")
    a("")
    a("## 5. Non-functional requirements")
    a("")
    a("| ID | Category | Requirement | Target | Measurable | Source |")
    a("|---|---|---|---|---|---|")
    for n in nfrs:
        it = n.item
        target = f"{it.target_value:g} {it.unit or ''}".strip() if it.target_value is not None else "_none_"
        a(f"| {n.id} | {it.category} | {md_cell(it.statement)} | {target} | {'yes' if n.measurable else '**NO**'} | "
          f"{n.interview}/{n.local_id} |")
    a("")
    a("## 6. Constraints")
    a("")
    a("| Category | Constraint | Source |")
    a("|---|---|---|")
    for iid, ex in extractions.items():
        for c in ex.constraints:
            a(f"| {c.category} | {md_cell(c.statement)} | {iid} |")
    a("")
    a("## 7. Conflicts and decisions needed")
    a("")
    for c in conflicts:
        a(f"### {c.id} - {c.label} (severity: {c.severity})")
        a("")
        a("| Stakeholder | Position | Normalized |")
        a("|---|---|---|")
        for iid, stance, value in c.positions:
            a(f"| {names[iid]} ({iid}) | {md_cell(stance)} | `{value}` |")
        a("")
        a(f"- **Decision owner:** {c.owner}")
        a(f"- **Option to table:** {c.hint}")
        a("")
    a("## 8. Gaps and follow-up questions")
    a("")
    a("| # | Type | Severity | Ref | Detail |")
    a("|---|---|---|---|---|")
    for i, g in enumerate(gaps, 1):
        a(f"| {i} | {g.kind} | {g.severity} | {g.ref} | {md_cell(g.detail)} |")
    a("")
    for aud, items in followups.items():
        if not items:
            continue
        a(f"**{audience_label[aud]}**")
        a("")
        for i, g in enumerate(items, 1):
            a(f"{i}. [{g.ref}] {g.question}")
        a("")
    a("## 9. Use-case feasibility and readiness")
    a("")
    w = use_cases["weights"]
    a(f"readiness = {w['value']} x value + {w['feasibility']} x feasibility + {w['safety']} x (6 - risk); "
      "feasibility = mean(data readiness, technical fit, 6 - integration complexity); "
      "risk = mean(data sensitivity, effective error impact, regulatory exposure), where a human in the loop "
      "lowers effective error impact by 2 (min 1).")
    a("")
    a("| UC | Use case | Value | Feasibility | Risk | Readiness | Decision | Blockers |")
    a("|---|---|---:|---:|---:|---:|---|---|")
    for u in uc_scores:
        a(f"| {u.id} | {u.name} | {u.value} | {u.feasibility:.2f} | {u.risk:.2f} | {u.readiness:.2f} | **{u.decision}** | "
          f"{'; '.join(u.blockers) if u.blockers else '-'} |")
    a("")
    a("## 10. Context: users, volumes, data and integrations")
    a("")
    a("| Type | Item | Detail | Source |")
    a("|---|---|---|---|")
    for iid, ex in extractions.items():
        for p in ex.personas:
            a(f"| Persona | {p.name} | {md_cell(p.needs)} | {iid} |")
        for pp in ex.pain_points:
            a(f"| Pain point | {md_cell(pp)} | - | {iid} |")
        for v in ex.volumes:
            a(f"| Volume | {v.metric} | {v.value:g} {v.unit} | {iid} |")
        for d in ex.data_sources:
            a(f"| Data source | {d.name} | {d.sensitivity}{', PII' if d.contains_pii else ''} | {iid} |")
        for itg in ex.integrations:
            a(f"| Integration | {itg.system} | {itg.direction} via {itg.mechanism} | {iid} |")
    a("")
    a("## 11. Risks, assumptions and open questions")
    a("")
    for label, attr in (("Risk", "risks"), ("Assumption", "assumptions"), ("Open question", "open_questions")):
        for iid, ex in extractions.items():
            for item in getattr(ex, attr):
                a(f"- **{label}** ({iid}): {item}")
    a("")
    a("## 12. Traceability (requirement -> evidence)")
    a("")
    a("| Requirement | Source | Evidence (verbatim quote) | Verified |")
    a("|---|---|---|---|")
    verified = {ref: ok for _, ref, _, ok in quote_checks}
    for m in merged:
        for src, (iid, quote) in zip(m.sources, m.quotes):
            a(f"| {m.id} | {src} | \"{md_cell(quote)}\" | {'yes' if verified.get(src, False) else '**NO - confirm**'} |")
    for n in nfrs:
        src = f"{n.interview}/{n.local_id}"
        a(f"| {n.id} | {src} | \"{md_cell(n.item.source_quote)}\" | {'yes' if verified.get(src, False) else '**NO - confirm**'} |")
    a("")
    a("## Appendix - interview-guide coverage")
    a("")
    a("| Section | " + " | ".join(iids) + " | Status |")
    a("|---|" + "---:|" * len(iids) + "---|")
    for s in guide["sections"]:
        row = matrix[s["id"]]
        status = "**GAP**" if sum(row.values()) == 0 else ("single-source" if sum(1 for v in row.values() if v) == 1 else "ok")
        a(f"| {s['title']} | " + " | ".join(str(row[i]) for i in iids) + f" | {status} |")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
