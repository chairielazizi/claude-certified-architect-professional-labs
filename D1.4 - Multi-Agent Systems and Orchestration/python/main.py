"""
D1.4 - Multi-Agent Systems and Orchestration

Hierarchical research system (orchestrator + parallel subagents), plus the other topologies:

  orchestrator (Opus) --plan (structured output)--> subagents (Sonnet/Haiku), each with an
  ISOLATED context and its own tool loop --condensed JSON--> blackboard + orchestrator synthesis

Also shown: sequential vs parallel wall-clock, blackboard conflicts, token/cost multiplier vs a
single agent, failure isolation (chaos test), a sequential pipeline, a peer handoff, and a
deterministic "should this be multi-agent at all?" check.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import anthropic
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    LATENCY_SCALE,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    MockRequest,
    banner,
    cost_usd,
    elapsed_ms,
    estimate_tokens,
    get_async_client,
    is_live,
    section,
    text_block,
    text_of,
    tool_use_block,
)

DATA = Path(__file__).resolve().parents[1] / "data"
CORPUS: list[dict[str, str]] = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))["snippets"]
TASK = ("Should we launch an AI-first customer-support offering for EU mid-market retailers next year? "
        "Cover market, competitors, unit economics and regulation, and give a recommendation.")
MODEL_MAP = {"sonnet": MODEL_SONNET, "haiku": MODEL_HAIKU}
SUBAGENT_DEADLINE_S = 120.0 if is_live() else 120.0 * LATENCY_SCALE   # wall-clock budget per subagent


def cj(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Ledger: tokens and dollars per phase, so multipliers are computed, not asserted
# ---------------------------------------------------------------------------
LEDGER: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})


def record(phase: str, msg: anthropic.types.Message) -> None:
    row = LEDGER[phase]
    row["calls"] += 1
    row["tokens"] += msg.usage.input_tokens + msg.usage.output_tokens
    row["cost"] += cost_usd(msg.model, msg.usage)


# ---------------------------------------------------------------------------
# Contracts between orchestrator and subagents
# ---------------------------------------------------------------------------
class Subtask(BaseModel):
    id: str
    topic: Literal["market", "competitors", "cost", "regulatory"]
    objective: str
    model: Literal["sonnet", "haiku"]
    boundaries: str
    critical: bool


class Plan(BaseModel):
    subtasks: list[Subtask]


class Fact(BaseModel):
    key: str
    value: str
    confidence: float


class SubagentResult(BaseModel):
    topic: str
    findings: list[str]
    facts: list[Fact]
    confidence: float
    open_questions: list[str]


RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "array", "items": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["key", "value", "confidence"], "additionalProperties": False}},
        "confidence": {"type": "number"},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topic", "findings", "facts", "confidence", "open_questions"],
    "additionalProperties": False,
}

SEARCH_TOOL = {
    "name": "search_corpus",
    "description": "Search the research corpus. Call this before answering; returns the top 3 snippets.",
    "strict": True,
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                     "required": ["query"], "additionalProperties": False},
}


def search_corpus(query: str, k: int = 3) -> list[dict[str, str]]:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = [(len(q & set(re.findall(r"[a-z0-9]+", s["text"].lower()))), s["id"], s) for s in CORPUS]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"id": s["id"], "text": s["text"]} for score, _, s in scored[:k] if score > 0]


@dataclass
class Outcome:
    subtask: Subtask
    result: SubagentResult
    calls: int
    consumed_tokens: int        # everything the subagent read and wrote in ITS context
    returned_tokens: int        # what the orchestrator has to read
    sources: list[str]


class SubagentError(Exception):
    pass


async def run_subagent(client: anthropic.AsyncAnthropic, st: Subtask, phase: str) -> Outcome:
    """One subagent = its own system prompt, its own message history, its own tool loop."""
    system = (f"You are a research subagent. Your topic: {st.topic}.\nObjective: {st.objective}\n"
              f"Boundaries: {st.boundaries}\nUse search_corpus, then return ONLY the JSON result "
              "(<= 3 findings, facts with confidence, open questions). Be concise: the orchestrator's "
              "context is the scarce resource.")
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"Brief {st.id}: {st.objective}"}]
    calls, consumed, sources = 0, 0, []
    for _ in range(4):                                          # subagent step limit
        msg = await client.messages.create(
            model=MODEL_MAP[st.model], max_tokens=1024, system=system, tools=[SEARCH_TOOL],
            output_config={"format": {"type": "json_schema", "schema": RESULT_SCHEMA}},
            messages=messages,
        )
        record(phase, msg)
        calls += 1
        consumed += msg.usage.input_tokens + msg.usage.output_tokens
        if msg.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": msg.content})
            results = []
            for block in msg.content:
                if block.type == "tool_use":
                    hits = search_corpus(block.input["query"])
                    sources += [h["id"] for h in hits]
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": cj(hits)})
            messages.append({"role": "user", "content": results})
            continue
        if msg.stop_reason != "end_turn":
            raise SubagentError(f"{st.id}: stop_reason={msg.stop_reason}")
        result = SubagentResult.model_validate_json(text_of(msg))
        return Outcome(st, result, calls, consumed, estimate_tokens(cj(result.model_dump())), sources)
    raise SubagentError(f"{st.id}: step limit reached")


async def run_all_parallel(client: anthropic.AsyncAnthropic, plan: Plan, phase: str) -> list[Outcome | BaseException]:
    # return_exceptions=True: one failing subagent must not cancel its siblings (failure isolation)
    return await asyncio.gather(
        *[asyncio.wait_for(run_subagent(client, st, phase), timeout=SUBAGENT_DEADLINE_S) for st in plan.subtasks],
        return_exceptions=True,
    )


# ---------------------------------------------------------------------------
# Blackboard: shared state the harness maintains; subagents never see each other's context
# ---------------------------------------------------------------------------
@dataclass
class Entry:
    key: str
    value: str
    author: str
    confidence: float


@dataclass
class Blackboard:
    entries: list[Entry] = field(default_factory=list)

    def post(self, author: str, facts: list[Fact]) -> None:
        self.entries += [Entry(f.key, f.value, author, f.confidence) for f in facts]

    def conflicts(self) -> dict[str, list[Entry]]:
        by_key: dict[str, list[Entry]] = defaultdict(list)
        for e in self.entries:
            by_key[e.key].append(e)
        return {k: v for k, v in by_key.items() if len({e.value for e in v}) > 1}

    def resolved(self) -> dict[str, Entry]:
        best: dict[str, Entry] = {}
        for e in self.entries:                       # policy: highest confidence wins, first poster on ties
            if e.key not in best or e.confidence > best[e.key].confidence:
                best[e.key] = e
        return best


async def synthesize(client: anthropic.AsyncAnthropic, outcomes: list[Outcome], board: Blackboard,
                     missing: list[str], phase: str) -> str:
    payload = {
        "results": [o.result.model_dump() for o in outcomes],
        "blackboard": {k: {"value": e.value, "by": e.author, "confidence": e.confidence} for k, e in board.resolved().items()},
        "conflicts": {k: [f"{e.value} ({e.author}, {e.confidence})" for e in v] for k, v in board.conflicts().items()},
    }
    note = f"\nMISSING: {', '.join(missing)}" if missing else ""
    msg = await client.messages.create(
        model=MODEL_OPUS, max_tokens=1500,
        system="You are the lead research orchestrator. Synthesise subagent results into a recommendation. "
               "Surface conflicts and missing inputs explicitly; never fill gaps with guesses.",
        messages=[{"role": "user", "content": f"Task: {TASK}\n\nSubagent results:\n{cj(payload)}{note}"}],
    )
    record(phase, msg)
    return text_of(msg)


# ---------------------------------------------------------------------------
# Topology chooser: when NOT to use multi-agent
# ---------------------------------------------------------------------------
@dataclass
class TaskProfile:
    name: str
    breadth: int = 1                      # independent threads of work
    tightly_coupled: bool = False         # do workers need each other's full context?
    exceeds_one_context: bool = False
    high_value: bool = False              # can it pay for many times the tokens?
    latency_sensitive: bool = False
    fixed_stages: bool = False
    specialist_handoff: bool = False


def recommend_topology(t: TaskProfile) -> tuple[str, str]:
    if t.specialist_handoff:
        return "peer handoff", "conversation ownership moves to a specialist agent"
    if t.latency_sensitive and not t.exceeds_one_context:
        return "single agent", "planning + synthesis round trips would blow the latency budget"
    if t.tightly_coupled:
        return "single agent", "workers would need shared context; coordination costs and conflicting edits"
    if t.fixed_stages:
        return "sequential pipeline", "fixed stage order; each stage's output is the next stage's input"
    if t.breadth >= 3 and t.high_value:
        return "hierarchical multi-agent", "broad, parallelisable, valuable enough for the token multiplier"
    if t.exceeds_one_context:
        return "hierarchical multi-agent", "context isolation: subagents compress what the lead must read"
    return "single agent", "not enough independent breadth to pay for extra tokens"


# ---------------------------------------------------------------------------
# Offline responder
# ---------------------------------------------------------------------------
PLAN_JSON = {"subtasks": [
    {"id": "s1", "topic": "market", "objective": "Size the EU mid-market retail segment and current AI-support adoption",
     "model": "sonnet", "boundaries": "No competitor analysis, no pricing", "critical": False},
    {"id": "s2", "topic": "competitors", "objective": "List direct competitors and their pricing model",
     "model": "haiku", "boundaries": "No market sizing beyond what sources state", "critical": False},
    {"id": "s3", "topic": "cost", "objective": "Estimate unit economics per ticket: AI vs human, incl. overheads",
     "model": "sonnet", "boundaries": "Use pilot data; no revenue forecast", "critical": True},
    {"id": "s4", "topic": "regulatory", "objective": "Identify EU AI Act and GDPR obligations for an AI support assistant",
     "model": "sonnet", "boundaries": "Obligations only; flag items for legal review", "critical": True},
]}
QUERIES = {
    "market": "EU mid-market retailers AI customer support adoption",
    "competitors": "competitors vendors pricing addressable retailers",
    "cost": "ticket cost human ai model usage integration",
    "regulatory": "eu ai act gdpr dpia obligations chatbots data residency",
}
RESULTS = {
    "market": {"topic": "market", "findings": ["~120k EU mid-market retailers (survey)", "AI-support adoption 18% in 2026, doubled since 2024", "24/7 and multilingual coverage drive adoption"],
               "facts": [{"key": "eu_midmarket_retailers", "value": "120000", "confidence": 0.7}, {"key": "ai_support_adoption_pct", "value": "18", "confidence": 0.65}],
               "confidence": 0.7, "open_questions": ["Survey method for the 120k figure"]},
    "competitors": {"topic": "competitors", "findings": ["4 AI-first vendors target this segment", "Pricing per resolved conversation: $0.60-$1.20", "Analyst puts addressable base at ~95k"],
                    "facts": [{"key": "eu_midmarket_retailers", "value": "95000", "confidence": 0.5}, {"key": "direct_competitors", "value": "4", "confidence": 0.8}],
                    "confidence": 0.65, "open_questions": ["Competitor resolution rates"]},
    "cost": {"topic": "cost", "findings": ["AI ticket ~$0.04 model cost with caching", "Human ticket ~$4.10 fully loaded", "Add 20-30% integration/QA in year one"],
             "facts": [{"key": "ai_cost_per_ticket_usd", "value": "0.04", "confidence": 0.8}, {"key": "human_cost_per_ticket_usd", "value": "4.10", "confidence": 0.75}],
             "confidence": 0.75, "open_questions": ["Share of tickets AI can fully resolve"]},
    "regulatory": {"topic": "regulatory", "findings": ["AI Act transparency: disclose AI interaction", "GDPR DPIA likely required", "EU data processing often required"],
                   "facts": [{"key": "ai_act_obligation", "value": "transparency (disclose AI)", "confidence": 0.7}, {"key": "dpia_required", "value": "likely", "confidence": 0.8}],
                   "confidence": 0.7, "open_questions": ["Legal review of sector-specific rules"]},
}
FULL_SYNTHESIS = ("GO for a 2027 pilot, conditional on a DPIA. Market: 95k-120k EU mid-market retailers (sources "
                  "conflict; 120k used at 0.7 confidence - validate before the business case); adoption 18% and "
                  "doubling. Economics: ~$0.04 AI vs ~$4.10 human per ticket, plus 20-30% integration/QA in year one. "
                  "Competition: 4 AI-first vendors at $0.60-$1.20 per resolved conversation. Regulation: AI Act "
                  "transparency duties, DPIA likely, plan EU data processing.")
PARTIAL_SYNTHESIS = ("PARTIAL - NOT DECISION-READY. Regulatory analysis is missing (subagent failed), and it is a "
                     "critical input. Market (95k-120k retailers, 18% adoption) and economics (~$0.04 vs ~$4.10 per "
                     "ticket) look attractive. Re-run the regulatory subagent before any go/no-go.")
BASELINE_ANSWER = ("Recommendation: GO. About 120,000 EU mid-market retailers, 18% already use AI support, AI tickets "
                   "cost ~$0.04 vs $4.10 for humans, four competitors exist, and chatbots must disclose AI use.")


def responder(req: MockRequest) -> Any:
    s, u = req.system_text, req.last_user_text
    if "You plan research" in s:
        return json.dumps(PLAN_JSON)
    if "research subagent" in s:
        topic = re.search(r"Your topic: (\w+)", s).group(1)  # type: ignore[union-attr]
        if not req.tool_results:
            return [text_block("Searching the corpus."), tool_use_block("search_corpus", {"query": QUERIES[topic]})]
        return json.dumps(RESULTS[topic])
    if "lead research orchestrator" in s:
        return PARTIAL_SYNTHESIS if "MISSING:" in u else FULL_SYNTHESIS
    if "single research analyst" in s:
        return BASELINE_ANSWER
    if "extract commercial terms" in s:
        return ("- 3-year term, $180k/year\n- 99.5% uptime SLA\n- Data stored in US-East\n"
                "- Price uplift capped at 7%/year\n- 90-day termination for convenience")
    if "risk analyst" in s:
        return ("1) US-East storage conflicts with EU data-residency needs (high). 2) 99.5% SLA allows ~44 h "
                "downtime/year (medium). 3) 7%/yr uplift = +14.5% by year 3 (medium).")
    if "memo writer" in s:
        return ("Decision: negotiate before signing. Require EU data residency and a 99.9% SLA; cap uplift at 3%. "
                "Termination terms are acceptable.")
    if "triage agent" in s:
        return [text_block("This is a billing issue; handing off."),
                tool_use_block("transfer_to_billing_agent",
                               {"summary": "Upgraded to Team plan yesterday; charged twice; wants the duplicate refunded."})]
    if "billing agent" in s:
        return ("Thanks for your patience - I can see the duplicate Team-plan charge from yesterday. I have refunded "
                "it; it will appear on your statement within 5 business days.")
    return "OK"


def with_fault(inner: Callable[[MockRequest], Any], predicate: Callable[[MockRequest], bool]) -> Callable[[MockRequest], Any]:
    """Chaos testing: inject an upstream failure into matching requests (SDK raises APIConnectionError)."""
    def wrapped(req: MockRequest) -> Any:
        if predicate(req):
            raise ConnectionError("injected fault: upstream unavailable")
        return inner(req)
    return wrapped


async def main_async() -> None:
    banner("D1.4 - Multi-Agent Systems and Orchestration")
    client = get_async_client(responder)

    section("1. Orchestrator plans the work (Opus, structured output)")
    print(f"  task: {TASK}")
    plan_msg = await client.messages.parse(
        model=MODEL_OPUS, max_tokens=2000, output_format=Plan,
        system="You plan research for a team of subagents. For each subtask give an objective, the cheapest "
               "adequate model, explicit boundaries (to avoid duplicated work) and whether it is critical.",
        messages=[{"role": "user", "content": TASK}],
    )
    record("orchestrator: plan", plan_msg)
    plan = plan_msg.parsed_output
    if plan is None:
        print("  no valid plan - abort")
        return
    for st in plan.subtasks:
        print(f"  {st.id} {st.topic:<12} model={st.model:<6} critical={'yes' if st.critical else 'no ':<3} | {st.objective}")
        print(f"     boundaries: {st.boundaries}")

    section("2. Subagents: isolated context, own tool loop, condensed result")
    t0 = time.perf_counter()
    for st in plan.subtasks:                                   # baseline: one after another
        await run_subagent(client, st, "timing baseline (sequential)")
    seq_ms = elapsed_ms(t0)
    t0 = time.perf_counter()
    raw = await run_all_parallel(client, plan, "subagents (parallel)")
    par_ms = elapsed_ms(t0)
    outcomes = [o for o in raw if isinstance(o, Outcome)]
    for o in outcomes:
        ratio = o.consumed_tokens / max(1, o.returned_tokens)
        print(f"  {o.subtask.id} {o.subtask.topic:<12} {MODEL_MAP[o.subtask.model]:<18} calls={o.calls} "
              f"sources={','.join(o.sources)}  consumed={o.consumed_tokens} tok -> returned={o.returned_tokens} tok "
              f"({ratio:.1f}x compression)")
    print(f"  wall-clock: sequential ~{round(seq_ms, -1):.0f} ms vs parallel ~{round(par_ms, -1):.0f} ms "
          f"-> {seq_ms / par_ms:.1f}x faster (latency = slowest subagent, not the sum)")

    section("3. Shared state: blackboard (harness-mediated)")
    board = Blackboard()
    for o in outcomes:
        board.post(o.subtask.topic, o.result.facts)
    for e in board.entries:
        print(f"  {e.author:<12} posted {e.key:<26} = {e.value:<28} conf={e.confidence:.2f}")
    for key, entries in board.conflicts().items():
        winner = board.resolved()[key]
        print(f"  CONFLICT on {key}: " + " vs ".join(f"{e.value} ({e.author})" for e in entries)
              + f" -> keep {winner.value} (higher confidence), surface the range in the synthesis")

    section("4. Orchestrator synthesis (Opus reads condensed results only)")
    print(f"  {await synthesize(client, outcomes, board, [], 'orchestrator: synthesis')}")

    section("5. Token / cost multiplier vs a single agent")
    corpus_text = "\n".join(f"[{s['id']}] {s['text']}" for s in CORPUS)
    single = await client.messages.create(
        model=MODEL_OPUS, max_tokens=1500, system="You are a single research analyst. Answer from the sources.",
        messages=[{"role": "user", "content": f"Task: {TASK}\n\nSources:\n{corpus_text}"}],
    )
    record("single agent baseline", single)
    multi_phases = ["orchestrator: plan", "subagents (parallel)", "orchestrator: synthesis"]
    multi_tokens = sum(LEDGER[p]["tokens"] for p in multi_phases)
    multi_cost = sum(LEDGER[p]["cost"] for p in multi_phases)
    single_row = LEDGER["single agent baseline"]
    print(f"  {'phase':<32}{'calls':>6}{'tokens':>8}{'cost $':>11}")
    for p in multi_phases + ["single agent baseline"]:
        r = LEDGER[p]
        print(f"  {p:<32}{int(r['calls']):>6}{int(r['tokens']):>8}{r['cost']:>11.5f}")
    print(f"  multi-agent total: {int(multi_tokens)} tokens, ${multi_cost:.5f}  ->  "
          f"{multi_tokens / single_row['tokens']:.1f}x tokens, {multi_cost / single_row['cost']:.1f}x cost of the single agent")
    print("  (cheaper subagent models soften the cost multiplier; the single agent also missed the 95k/120k conflict)")

    section("6. Failure isolation (chaos test: regulatory subagent outage)")
    chaos = get_async_client(with_fault(responder, lambda r: "Your topic: regulatory" in r.system_text))
    raw = await run_all_parallel(chaos, plan, "chaos run")
    ok, failed = [], []
    for st, o in zip(plan.subtasks, raw):
        if isinstance(o, Outcome):
            ok.append(o)
            print(f"  {st.id} {st.topic:<12} ok")
        else:
            failed.append(st)
            print(f"  {st.id} {st.topic:<12} FAILED: {type(o).__name__} (siblings unaffected)")
    critical_missing = [st.topic for st in failed if st.critical]
    status = "PARTIAL (critical input missing)" if critical_missing else ("DEGRADED" if failed else "COMPLETE")
    print(f"  orchestrator policy -> status {status}; options: retry subagent, re-plan, or report with gaps")
    board2 = Blackboard()
    for o in ok:
        board2.post(o.subtask.topic, o.result.facts)
    print(f"  {await synthesize(chaos, ok, board2, [st.topic for st in failed], 'chaos run')}")

    section("7. Other topologies: sequential pipeline and peer handoff")
    proposal = ("Vendor X proposes a 3-year contract at $180k/year for an AI support platform, 99.5% uptime SLA, "
                "data stored in US-East, price uplift capped at 7%/year, 90-day termination for convenience.")
    print("  pipeline: extractor (Haiku) -> risk analyst (Sonnet) -> memo writer (Sonnet)")
    stage_input, total_ms = proposal, 0.0
    for name, model, system in (("extractor", MODEL_HAIKU, "You extract commercial terms as bullets."),
                                ("risk analyst", MODEL_SONNET, "You are a risk analyst. Rank the top 3 risks."),
                                ("memo writer", MODEL_SONNET, "You are a memo writer. Write a 2-line decision memo.")):
        t0 = time.perf_counter()
        m = await client.messages.create(model=model, max_tokens=600, system=system,
                                         messages=[{"role": "user", "content": stage_input}])
        total_ms += elapsed_ms(t0)
        record("pipeline", m)
        stage_input = text_of(m)                    # each stage sees ONLY the previous stage's output
        print(f"    [{name}] {stage_input.splitlines()[0]}{' ...' if len(stage_input.splitlines()) > 1 else ''}")
    print(f"    latency is the SUM of stages: ~{round(total_ms, -1):.0f} ms; a bad early stage poisons later ones")

    user_msg = "I upgraded to the Team plan yesterday and was charged twice."
    transfer_tools = [
        {"name": "transfer_to_billing_agent", "description": "Hand the conversation to the billing specialist.",
         "strict": True, "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                                          "required": ["summary"], "additionalProperties": False}},
        {"name": "transfer_to_technical_agent", "description": "Hand the conversation to the technical specialist.",
         "strict": True, "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                                          "required": ["summary"], "additionalProperties": False}},
    ]
    triage = await client.messages.create(
        model=MODEL_HAIKU, max_tokens=400, tools=transfer_tools,
        system="You are the triage agent. Transfer to the right specialist with a one-line summary.",
        messages=[{"role": "user", "content": user_msg}],
    )
    record("handoff", triage)
    handoff = next((b for b in triage.content if b.type == "tool_use"), None)
    if triage.stop_reason == "tool_use" and handoff is not None:
        target = handoff.name.removeprefix("transfer_to_").removesuffix("_agent")
        note = handoff.input["summary"]
        print(f"  peer handoff: triage (Haiku) -> {target} (Sonnet); context passed = handoff note "
              f"(~{estimate_tokens(note)} tok), not the whole transcript")
        reply = await client.messages.create(
            model=MODEL_SONNET, max_tokens=600,
            system=f"You are the {target} agent. You now own this conversation end to end.",
            messages=[{"role": "user", "content": f"{user_msg}\n\n<handoff from=\"triage\">{note}</handoff>"}],
        )
        record("handoff", reply)
        print(f"    {target} agent: {text_of(reply)}")

    section("8. When NOT to use multi-agent")
    profiles = [
        TaskProfile("Due diligence across market, competitors, cost, regulation", breadth=4, high_value=True),
        TaskProfile("Refactor a tightly coupled billing module", breadth=3, tightly_coupled=True, high_value=True),
        TaskProfile("Live chat answer within 2 seconds", latency_sensitive=True),
        TaskProfile("Invoices: extract -> validate -> post to ERP", fixed_stages=True),
        TaskProfile("Support chat that may need billing or tech specialists", specialist_handoff=True),
        TaskProfile("Summarise one 20-page report"),
    ]
    for t in profiles:
        topo, why = recommend_topology(t)
        print(f"  {t.name:<58} -> {topo:<24} ({why})")

    section("Summary")
    print("  Multi-agent buys breadth, parallel speed and context isolation at a token multiplier.")
    print("  Use it for broad, parallelisable, high-value work; give every subagent an objective, boundaries")
    print("  and an output format; isolate failures; and keep tightly coupled or latency-critical work single-agent.")


if __name__ == "__main__":
    asyncio.run(main_async())
