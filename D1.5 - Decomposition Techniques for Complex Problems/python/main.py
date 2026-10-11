"""
D1.5 - Decomposition Techniques for Complex Problems

Goal: produce a launch-readiness brief that needs (1) several dependent analyses, (2) obligations
from a long contract and (3) a multi-step capacity calculation.

  0. Monolithic baseline   one giant prompt -> truncation (max_tokens) + a silently missed requirement
  1. Task DAG              planner (structured output) -> validate (unknown deps, cycles) -> topological
                           execution, each task sees ONLY its dependencies' outputs
  2. Map-reduce            chunk the contract by section -> map: extract obligations -> reduce in code
  3. Least-to-most         ordered sub-questions, each answer gated by deterministic checks
  4. Comparison            requirement coverage + calls/tokens/cost, monolithic vs decomposed
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    MockRequest,
    banner,
    cost_usd,
    get_client,
    section,
    text_of,
)

DATA = Path(__file__).resolve().parents[1] / "data"
CONTRACT = (DATA / "vendor_contract.md").read_text(encoding="utf-8")
GOAL = "Prepare the launch-readiness brief for the new 'Export to PDF' dashboard feature."
REQUIRED = ["Requirements", "Security", "Pricing", "Docs and support", "Risk register",
            "Contract obligations", "Black Friday capacity", "Go/no-go"]
FACTS = {"last_year_volume": 18400, "growth_pct": 25, "deflection_pct": 38, "agents": 42,
         "tickets_per_agent_day": 55, "days": 7}


def cj(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
LEDGER: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})


def call(client: anthropic.Anthropic, phase: str, **kwargs: Any) -> anthropic.types.Message:
    msg = client.messages.create(**kwargs)
    LEDGER[phase]["calls"] += 1
    LEDGER[phase]["tokens"] += msg.usage.input_tokens + msg.usage.output_tokens
    LEDGER[phase]["cost"] += cost_usd(msg.model, msg.usage)
    return msg


def coverage(text: str) -> dict[str, bool]:
    """A required section counts only if its heading exists AND it has a non-empty body."""
    found: dict[str, bool] = {}
    lines = text.splitlines()
    for req in REQUIRED:
        ok = False
        for i, line in enumerate(lines):
            if line.strip().lower() == f"## {req}".lower():
                body = []
                for nxt in lines[i + 1:]:
                    if nxt.startswith("## "):
                        break
                    body.append(nxt)
                ok = bool("".join(body).strip())
        found[req] = ok
    return found


# ===========================================================================
# 1. TASK DAG: plan -> validate -> topological execution
# ===========================================================================
class PlannedTask(BaseModel):
    id: str
    title: str
    instruction: str
    depends_on: list[str]


class TaskPlan(BaseModel):
    tasks: list[PlannedTask]


def find_cycle(tasks: list[PlannedTask]) -> list[str]:
    """DFS with colours; returns one cycle as a path (e.g. t2 -> t5 -> t2) or []."""
    graph = {t.id: [d for d in t.depends_on if d in {x.id for x in tasks}] for t in tasks}
    color: dict[str, int] = {t: 0 for t in graph}       # 0 white, 1 grey (on stack), 2 black
    stack: list[str] = []

    def dfs(node: str) -> list[str]:
        color[node] = 1
        stack.append(node)
        for dep in graph[node]:
            if color[dep] == 1:
                return stack[stack.index(dep):] + [dep]
            if color[dep] == 0:
                cyc = dfs(dep)
                if cyc:
                    return cyc
        stack.pop()
        color[node] = 2
        return []

    for node in sorted(graph):
        if color[node] == 0:
            cyc = dfs(node)
            if cyc:
                return cyc
    return []


def validate_dag(tasks: list[PlannedTask]) -> list[str]:
    ids = [t.id for t in tasks]
    errors = [f"duplicate task id {i}" for i, n in Counter(ids).items() if n > 1]
    for t in tasks:
        errors += [f"{t.id} depends on unknown task {d}" for d in t.depends_on if d not in ids]
    cycle = find_cycle(tasks)
    if cycle:
        errors.append("cycle: " + " -> ".join(cycle))
    return errors


def topo_levels(tasks: list[PlannedTask]) -> list[list[str]]:
    """Kahn-style layering: every task in level N depends only on levels < N (so a level can run in parallel)."""
    deps = {t.id: set(t.depends_on) for t in tasks}
    done: set[str] = set()
    levels: list[list[str]] = []
    while len(done) < len(deps):
        ready = sorted(i for i, d in deps.items() if i not in done and d <= done)
        if not ready:
            raise ValueError("unresolvable dependencies")
        levels.append(ready)
        done |= set(ready)
    return levels


PLANNER_SYSTEM = ("You plan work as a dependency graph. Return tasks with ids t1..tn, a one-line instruction each, "
                  "and depends_on listing only the task ids whose OUTPUT the task needs. The final task must "
                  "produce the go/no-go decision.")


def run_dag(client: anthropic.Anthropic) -> dict[str, str]:
    user = GOAL
    plan: TaskPlan | None = None
    for attempt in (1, 2):
        msg = client.messages.parse(model=MODEL_OPUS, max_tokens=2000, output_format=TaskPlan,
                                    system=PLANNER_SYSTEM, messages=[{"role": "user", "content": user}])
        LEDGER["decomposed"]["calls"] += 1
        LEDGER["decomposed"]["tokens"] += msg.usage.input_tokens + msg.usage.output_tokens
        LEDGER["decomposed"]["cost"] += cost_usd(msg.model, msg.usage)
        plan = msg.parsed_output
        errors = validate_dag(plan.tasks) if plan else ["no plan returned"]
        print(f"  plan attempt {attempt}: {len(plan.tasks) if plan else 0} tasks -> "
              f"{'VALID' if not errors else 'INVALID: ' + '; '.join(errors)}")
        if not errors:
            break
        user = f"{GOAL}\n\nValidator feedback - fix and return the full plan: {'; '.join(errors)}"
    else:
        raise SystemExit("planner could not produce a valid DAG")
    assert plan is not None
    by_id = {t.id: t for t in plan.tasks}
    outputs: dict[str, str] = {}
    for n, level in enumerate(topo_levels(plan.tasks), 1):
        note = f"  ({len(level)} independent tasks - could run concurrently)" if len(level) > 1 else ""
        print(f"  level {n}: {', '.join(level)}{note}")
        for tid in level:
            t = by_id[tid]
            context = "\n".join(f"[{d} output] {outputs[d]}" for d in t.depends_on)   # ONLY its dependencies
            msg = call(client, "decomposed", model=MODEL_SONNET, max_tokens=800,
                       system="You execute one task of a larger plan. Use only the provided inputs. Be concise.",
                       messages=[{"role": "user", "content": f"Task {tid}: {t.instruction}\n\nInputs:\n{context or '(none)'}"}])
            outputs[tid] = text_of(msg)
            deps = ",".join(t.depends_on) or "-"
            print(f"    {tid} {t.title:<22} deps={deps:<6} -> {outputs[tid][:70]}...")
    return {by_id[k].title: v for k, v in outputs.items()}


# ===========================================================================
# 2. MAP-REDUCE over a long document
# ===========================================================================
@dataclass
class Chunk:
    label: str
    heading: str
    text: str


MAX_CHUNK_CHARS = 900


def chunk_by_section(doc: str) -> tuple[str, list[Chunk]]:
    """Split on '## ' headings; split oversized sections on paragraph boundaries (never mid-sentence)."""
    title = doc.splitlines()[0].lstrip("# ").strip()
    chunks: list[Chunk] = []
    for block in re.split(r"\n(?=## )", doc)[1:]:
        heading, _, body = block.partition("\n")
        heading = heading.lstrip("# ").strip()
        number = heading.split(".")[0]
        paras = [p.strip() for p in body.strip().split("\n\n") if p.strip()]
        if len(body) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(number, heading, body.strip()))
            continue
        for i, para in enumerate(paras, 1):
            chunks.append(Chunk(f"{number}.{i}", f"{heading} (part {i}/{len(paras)})", para))
    return title, chunks


class Obligation(BaseModel):
    party: str
    obligation: str
    deadline_hours: float | None
    penalty: str | None


class ChunkObligations(BaseModel):
    obligations: list[Obligation]


def fmt_deadline(hours: float | None) -> str:
    if hours is None:
        return "-"
    return f"{hours:g}h" if hours < 48 else f"{hours / 24:g}d"


def run_map_reduce(client: anthropic.Anthropic) -> str:
    title, chunks = chunk_by_section(CONTRACT)
    print(f"  document: {title} ({len(CONTRACT)} chars) -> {len(chunks)} chunks "
          f"(max {MAX_CHUNK_CHARS} chars; oversized sections split on paragraphs)")
    rows: list[tuple[str, Obligation]] = []
    for ch in chunks:                                              # MAP (independent -> parallelisable)
        msg = client.messages.parse(
            model=MODEL_HAIKU, max_tokens=1000, output_format=ChunkObligations,
            system="Extract every contractual obligation in this section: party, obligation, deadline in hours "
                   "(null if none) and penalty (null if none). Return an empty list if there are none.",
            messages=[{"role": "user", "content": f"Document: {title}\nSection {ch.heading}\n\n{ch.text}"}],
        )
        LEDGER["decomposed"]["calls"] += 1
        LEDGER["decomposed"]["tokens"] += msg.usage.input_tokens + msg.usage.output_tokens
        LEDGER["decomposed"]["cost"] += cost_usd(msg.model, msg.usage)
        found = msg.parsed_output.obligations if msg.parsed_output else []
        rows += [(ch.label, o) for o in found]
        print(f"    map {ch.label:<4} {ch.heading[:44]:<44} -> {len(found)} obligation(s)")
    # REDUCE in code: deterministic merge, sort, aggregate - no LLM needed for arithmetic
    rows.sort(key=lambda r: (r[1].deadline_hours is None, r[1].deadline_hours or 0, r[0]))
    by_party = Counter(o.party for _, o in rows)
    penalties = [(lbl, o.penalty) for lbl, o in rows if o.penalty]
    print(f"  reduce: {len(rows)} obligations; by party: "
          + ", ".join(f"{p}={n}" for p, n in sorted(by_party.items())))
    print("  tightest deadlines:")
    for lbl, o in [r for r in rows if r[1].deadline_hours is not None][:4]:
        print(f"    {fmt_deadline(o.deadline_hours):>4}  [{lbl}] {o.party}: {o.obligation}")
    for lbl, pen in penalties:
        print(f"    penalty [{lbl}]: {pen}")
    reduced = {"total": len(rows), "by_party": dict(sorted(by_party.items())),
               "tightest": [{"section": lbl, "party": o.party, "obligation": o.obligation,
                             "deadline": fmt_deadline(o.deadline_hours)} for lbl, o in rows[:4]],
               "penalties": [p for _, p in penalties]}
    msg = call(client, "decomposed", model=MODEL_SONNET, max_tokens=600,
               system="Summarise contract obligations for a launch brief in 3 sentences. Use only the data given.",
               messages=[{"role": "user", "content": cj(reduced)}])
    summary = text_of(msg)
    print(f"  summary: {summary}")
    return summary


# ===========================================================================
# 3. LEAST-TO-MOST with validation gates between steps
# ===========================================================================
class SubQuestion(BaseModel):
    question: str
    fields: list[str]


class Decomposition(BaseModel):
    subquestions: list[SubQuestion]


def step_schema(fields: list[str]) -> dict[str, Any]:
    props: dict[str, Any] = {f: {"type": "number"} for f in fields}
    props["explanation"] = {"type": "string"}
    return {"type": "object", "properties": props, "required": [*fields, "explanation"], "additionalProperties": False}


def gate(values: dict[str, float], known: dict[str, float]) -> list[str]:
    """Deterministic checks against the input facts and earlier (already-gated) answers."""
    problems = [f"{k} must be a non-negative number" for k, v in values.items()
                if not isinstance(v, (int, float)) or v < 0]
    if problems:
        return problems
    if "volume" in values and not FACTS["last_year_volume"] <= values["volume"] <= 2 * FACTS["last_year_volume"]:
        problems.append("volume outside plausible range (1x-2x last year)")
    if "deflected" in values and "remaining" in values:
        vol = known["volume"]
        if abs(values["deflected"] + values["remaining"] - vol) > 1:
            problems.append(f"deflected + remaining != volume {vol:,.0f}")
        share = values["deflected"] / vol * 100
        if abs(share - FACTS["deflection_pct"]) > 2:
            problems.append(f"deflected share {share:.0f}% contradicts the {FACTS['deflection_pct']}% pilot rate")
    if "capacity" in values:
        expected = FACTS["agents"] * FACTS["tickets_per_agent_day"] * FACTS["days"]
        if abs(values["capacity"] - expected) > expected * 0.01:
            problems.append(f"capacity {values['capacity']:,.0f} inconsistent with staffing facts")
    if "surplus" in values and abs(values["surplus"] - (known["capacity"] - known["remaining"])) > 1:
        problems.append("surplus != capacity - remaining")
    if "utilization_pct" in values and abs(values["utilization_pct"] - known["remaining"] / known["capacity"] * 100) > 0.5:
        problems.append("utilisation inconsistent with remaining / capacity")
    return problems


def run_least_to_most(client: anthropic.Anthropic) -> str:
    question = "Can our support team handle Black Friday week once the AI assistant is live?"
    facts_text = ", ".join(f"{k}={v}" for k, v in FACTS.items())
    msg = client.messages.parse(
        model=MODEL_SONNET, max_tokens=1000, output_format=Decomposition,
        system="Decompose the question into the simplest ordered sub-questions (least to most complex). "
               "For each, name the numeric output fields it must produce.",
        messages=[{"role": "user", "content": f"Question: {question}\nFacts: {facts_text}"}],
    )
    LEDGER["decomposed"]["calls"] += 1
    LEDGER["decomposed"]["tokens"] += msg.usage.input_tokens + msg.usage.output_tokens
    LEDGER["decomposed"]["cost"] += cost_usd(msg.model, msg.usage)
    decomposition = msg.parsed_output
    if decomposition is None:
        return "decomposition failed"
    known: dict[str, float] = {}
    for k, sq in enumerate(decomposition.subquestions, 1):
        feedback = ""
        for attempt in (1, 2, 3):
            prior = ", ".join(f"{f}={v:g}" for f, v in known.items()) or "(none)"
            user = (f"Facts: {facts_text}\nEarlier answers: {prior}\nSubquestion {k}: {sq.question}"
                    + (f"\nGate feedback: {feedback}" if feedback else ""))
            m = call(client, "decomposed", model=MODEL_SONNET, max_tokens=500,
                     system="Answer one sub-question using the facts and earlier answers. Return JSON only.",
                     output_config={"format": {"type": "json_schema", "schema": step_schema(sq.fields)}},
                     messages=[{"role": "user", "content": user}])
            data = json.loads(text_of(m)) if m.stop_reason == "end_turn" else {}
            values = {f: data.get(f) for f in sq.fields}
            problems = gate(values, known) if data else ["no complete answer"]
            shown = ", ".join(f"{f}={v:g}" if isinstance(v, (int, float)) else f"{f}={v}" for f, v in values.items())
            print(f"  Q{k}.{attempt} {sq.question[:58]:<58} -> {shown}  "
                  f"gate: {'PASS' if not problems else 'FAIL (' + '; '.join(problems) + ')'}")
            if not problems:
                known.update({f: float(v) for f, v in values.items()})   # only gated values flow forward
                break
            feedback = "; ".join(problems)
        else:
            return f"stopped: sub-question {k} failed its gate 3 times -> escalate to an analyst"
    m = call(client, "decomposed", model=MODEL_SONNET, max_tokens=600,
             system="Answer the original question from the verified sub-answers in 2-3 sentences.",
             messages=[{"role": "user", "content": f"Question: {question}\nVerified: {cj(known)}"}])
    answer = text_of(m)
    print(f"  answer: {answer}")
    return answer


# ===========================================================================
# Offline responder
# ===========================================================================
TASKS_OK = [
    {"id": "t1", "title": "Requirements summary", "instruction": "Summarise functional and non-functional requirements.", "depends_on": []},
    {"id": "t2", "title": "Security review", "instruction": "Review security and privacy of PDF export.", "depends_on": ["t1"]},
    {"id": "t3", "title": "Pricing and packaging", "instruction": "Decide plan availability and limits.", "depends_on": ["t1"]},
    {"id": "t4", "title": "Docs and support plan", "instruction": "Plan help content and support readiness.", "depends_on": ["t1"]},
    {"id": "t5", "title": "Risk register", "instruction": "List top risks with mitigations.", "depends_on": ["t2", "t3"]},
    {"id": "t6", "title": "Go/no-go decision", "instruction": "Recommend go/no-go with rollout gates.", "depends_on": ["t4", "t5"]},
]
TASKS_BAD = [dict(t) for t in TASKS_OK]
TASKS_BAD[1] = {**TASKS_OK[1], "depends_on": ["t1", "t5"]}          # t2 <-> t5 cycle
TASKS_BAD[5] = {**TASKS_OK[5], "depends_on": ["t4", "t5", "t7"]}    # unknown task
TASK_OUTPUTS = {
    "t1": "One-click PDF export of any dashboard (p95 under 10 s for 20 widgets) plus an optional weekly email; Team and Enterprise plans.",
    "t2": "Render server-side with the viewer's permissions; signed download URLs expire after 24 h; no PII in file names.",
    "t3": "Included in Team and Enterprise; Team limited to 10 scheduled emails; no add-on price.",
    "t4": "Help-centre article, in-app tooltip and a support macro for 'export failed' tickets, ready at launch.",
    "t5": "Renderer load spikes (queue + 10/min rate limit); shared-link leakage (24 h expiry); schedule abuse on Team (cap of 10).",
    "t6": "GO for a staged rollout: 10% of Team accounts for one week, gated on export error rate < 1% and p95 < 10 s.",
}
MAP_OUTPUTS: dict[str, list[dict[str, Any]]] = {
    "1": [],
    "2": [{"party": "Vendor", "obligation": "Provide the Services in the Order Form", "deadline_hours": None, "penalty": None},
          {"party": "Customer", "obligation": "Provide ticketing API access and nominate an administrator", "deadline_hours": None, "penalty": None}],
    "3": [{"party": "Vendor", "obligation": "Keep 99.9% monthly availability", "deadline_hours": None,
           "penalty": "service credit 10% of Monthly Fee per 0.1% below target (max 50%)"},
          {"party": "Vendor", "obligation": "Respond to Priority 1 Incidents", "deadline_hours": 1, "penalty": None}],
    "4.1": [{"party": "Vendor", "obligation": "Process Personal Data only in EU data centres", "deadline_hours": None, "penalty": None},
            {"party": "Vendor", "obligation": "Assist with data subject requests and DPIAs at no charge", "deadline_hours": None, "penalty": None},
            {"party": "Vendor", "obligation": "Give notice before engaging new sub-processors", "deadline_hours": 720, "penalty": None}],
    "4.2": [{"party": "Vendor", "obligation": "Notify Customer of a Personal Data breach", "deadline_hours": 72, "penalty": None}],
    "5": [{"party": "Customer", "obligation": "Pay each invoice", "deadline_hours": 720, "penalty": "late interest 1% per month"},
          {"party": "Vendor", "obligation": "Give notice of a fee increase (max 5%/year)", "deadline_hours": 1440, "penalty": None}],
    "6": [{"party": "Vendor", "obligation": "Share penetration-test summary after completion", "deadline_hours": 720, "penalty": None},
          {"party": "Vendor", "obligation": "Remediate critical vulnerabilities", "deadline_hours": 336, "penalty": None},
          {"party": "Vendor", "obligation": "Remediate high vulnerabilities", "deadline_hours": 720, "penalty": None}],
    "7": [{"party": "Either", "obligation": "Give notice to terminate for convenience", "deadline_hours": 2160, "penalty": None},
          {"party": "Vendor", "obligation": "Return or delete Customer data after termination", "deadline_hours": 720, "penalty": None}],
    "8": [],
}
OBLIGATION_SUMMARY = ("Vendor carries 12 of the 15 obligations, with the tightest deadlines being a 1-hour response to "
                      "Priority 1 incidents and 72-hour breach notification. Financial levers are availability service "
                      "credits (10% per 0.1% below 99.9%, max 50%) and 1% monthly late-payment interest on Customer. "
                      "Exit requires 90 days' notice, with data returned or deleted within 30 days.")
DECOMPOSITION = {"subquestions": [
    {"question": "What ticket volume should we expect in Black Friday week?", "fields": ["volume"]},
    {"question": "How many tickets will the AI deflect and how many remain for agents?", "fields": ["deflected", "remaining"]},
    {"question": "How many tickets can the agent team handle that week?", "fields": ["capacity"]},
    {"question": "What is the surplus or shortfall and resulting utilisation?", "fields": ["surplus", "utilization_pct"]},
]}
STEP_ANSWERS = {
    1: {"volume": 23000, "explanation": "18,400 x 1.25"},
    2: {"deflected": 8740, "remaining": 14260, "explanation": "23,000 x 38% deflected; the rest remain"},
    3: {"capacity": 16170, "explanation": "42 agents x 55 tickets/day x 7 days"},
    4: {"surplus": 1910, "utilization_pct": 88.2, "explanation": "16,170 - 14,260; 14,260 / 16,170"},
}
CAPACITY_ANSWER = ("Yes. Expect ~23,000 tickets; the AI assistant deflects ~8,740, leaving 14,260 for agents against a "
                   "capacity of 16,170, a 1,910-ticket buffer at 88.2% utilisation. If deflection drops below ~30% the "
                   "buffer is gone, so keep an on-call overflow rota.")


def _monolithic_reply() -> str:
    return (
        "## Requirements\nThe feature lets any user export a dashboard to PDF with one click and optionally receive a weekly "
        "email with the PDF attached. Exports must complete quickly for typical dashboards, preserve the on-screen layout "
        "including charts, tables and filters, and respect the viewer's permissions. Large dashboards with more than twenty "
        "widgets should still render, potentially with pagination, and the export must be available on desktop browsers. "
        "Scheduled emails must allow choosing a weekday and time, and recipients must be existing workspace members. "
        "The export must include a cover page with the dashboard title, the filters applied and a timestamp in the "
        "viewer's time zone, and charts must be rendered as vectors so that they stay sharp when printed. Exported "
        "tables need real text rather than images so that screen readers and search work on the PDF.\n"
        "## Security\nRendering should happen server-side using the requesting viewer's permissions so no hidden widgets "
        "leak into the document. Download links must be signed and time-limited, and file names must avoid personal data. "
        "Scheduled exports run as the schedule owner, so revoking a user's access must also disable their schedules. All "
        "export events should be logged for audit, and generated files should be deleted from storage after a retention "
        "window. Rate limits protect the renderer from abuse, and the endpoint must reject cross-tenant identifiers. "
        "Penetration testing of the new endpoint is required before general availability, and the renderer must run "
        "in an isolated sandbox without access to internal networks.\n"
        "## Docs and support\nPublish a help-centre article with screenshots, add an in-app tooltip next to the new export "
        "button, and prepare a support macro for failed exports. Train tier-one agents on the most common failure modes, "
        "such as timeouts on very large dashboards and permission errors on shared dashboards, and update the status page "
        "templates so incidents affecting exports can be communicated quickly to affected customers.\n"
        "## Risk register\nThe main operational risk is load on the headless renderer during peak hours, which could slow "
        "exports for everyone; mitigation is a queue with per-tenant rate limits and autoscaling of renderer workers. A "
        "second risk is leakage through forwarded download links; mitigation is short expiry and signed URLs bound to the "
        "tenant. A third risk is abuse of scheduled emails to send large volumes of mail; mitigation is a cap on schedules "
        "per workspace and monitoring of bounce rates. A fourth risk is inconsistent rendering of custom chart plugins, "
        "which we mitigate with a compatibility test suite that runs against the top fifty public dashboards before each "
        "release, and a feature flag so that problematic plugins can fall back to static images without blocking the export.\n"
        "## Contract obligations\nThe vendor must notify breaches within 72 hours and respond to P1 incidents in 1 hour...\n"
        "## Black Friday capacity\nWith 23,000 expected tickets and 38% deflection, 14,260 remain against 16,170 capacity...\n"
        "## Go/no-go\nGO for a staged rollout gated on error rate and latency.\n"
    )


def responder(req: MockRequest) -> Any:
    s, u = req.system_text, req.last_user_text
    if "dependency graph" in s:
        return json.dumps({"tasks": TASKS_OK if "Validator feedback" in u else TASKS_BAD})
    if "You execute one task" in s:
        return TASK_OUTPUTS[re.search(r"Task (t\d+)", u).group(1)]  # type: ignore[union-attr]
    if "Extract every contractual obligation" in s:
        num = re.search(r"Section (\d+)\.", u).group(1)  # type: ignore[union-attr]
        part = re.search(r"\(part (\d+)/", u)
        return json.dumps({"obligations": MAP_OUTPUTS[f"{num}.{part.group(1)}" if part else num]})
    if "Summarise contract obligations" in s:
        return OBLIGATION_SUMMARY
    if "Decompose the question" in s:
        return json.dumps(DECOMPOSITION)
    if "Answer one sub-question" in s:
        k = int(re.search(r"Subquestion (\d+):", u).group(1))  # type: ignore[union-attr]
        if k == 2 and "Gate feedback" not in u:
            # Classic slip: deflected and remaining swapped. The gate catches it before it propagates.
            return json.dumps({"deflected": 14260, "remaining": 8740, "explanation": "38% of 23,000 remain"})
        return json.dumps(STEP_ANSWERS[k])
    if "original question from the verified" in s:
        return CAPACITY_ANSWER
    if "complete launch-readiness brief" in s:
        return _monolithic_reply()
    return "OK"


def main() -> None:
    banner("D1.5 - Decomposition Techniques for Complex Problems")
    client = get_client(responder)

    section("0. Monolithic baseline: one prompt does everything")
    mono_prompt = (f"{GOAL}\nCover these sections as '## ' headings: {', '.join(REQUIRED)}.\n"
                   f"Capacity facts: {cj(FACTS)}\n\nContract:\n{CONTRACT}")
    mono = call(client, "monolithic", model=MODEL_SONNET, max_tokens=600,
                system="You write the complete launch-readiness brief in one pass.",
                messages=[{"role": "user", "content": mono_prompt}])
    mono_text = text_of(mono)
    mono_cov = coverage(mono_text)
    print(f"  stop_reason={mono.stop_reason}  output={len(mono_text)} chars  "
          f"(input {mono.usage.input_tokens} tokens incl. the whole contract)")
    print(f"  coverage {sum(mono_cov.values())}/{len(REQUIRED)}: "
          + ", ".join(f"{k}={'ok' if v else 'MISSING'}" for k, v in mono_cov.items()))
    print("  -> truncated mid-document AND 'Pricing' was skipped entirely; nothing tells you which is which")

    section("1. Task DAG: planner -> validator -> topological execution")
    dag_outputs = run_dag(client)

    section("2. Map-reduce over the contract")
    obligations = run_map_reduce(client)

    section("3. Least-to-most with validation gates")
    capacity = run_least_to_most(client)

    section("4. Assemble + compare")
    brief = "\n".join([
        "## Requirements", dag_outputs["Requirements summary"],
        "## Security", dag_outputs["Security review"],
        "## Pricing", dag_outputs["Pricing and packaging"],
        "## Docs and support", dag_outputs["Docs and support plan"],
        "## Risk register", dag_outputs["Risk register"],
        "## Contract obligations", obligations,
        "## Black Friday capacity", capacity,
        "## Go/no-go", dag_outputs["Go/no-go decision"],
    ])
    dec_cov = coverage(brief)
    print(f"  {'approach':<12}{'calls':>6}{'tokens':>8}{'cost $':>10}{'coverage':>10}  failure visibility")
    for name, cov, note in (("monolithic", mono_cov, "silent: truncation + omission"),
                            ("decomposed", dec_cov, "per-step: validator, gates, stop_reason")):
        r = LEDGER[name]
        print(f"  {name:<12}{int(r['calls']):>6}{int(r['tokens']):>8}{r['cost']:>10.5f}"
              f"{str(sum(cov.values())) + '/' + str(len(REQUIRED)):>10}  {note}")
    print("  Decomposition costs more calls, but every step is small, checkable, retryable - and parallelisable.")

    section("Summary")
    print("  DAG for dependent sub-tasks, map-reduce for long inputs, least-to-most for multi-step reasoning;")
    print("  validate between steps in CODE so errors are caught where they happen, not in the final answer.")


if __name__ == "__main__":
    main()
