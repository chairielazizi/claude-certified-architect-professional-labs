"""
D1.3 - Architectural Patterns - Workflow, Agentic, Augmented LLM

Small, runnable versions of the building blocks from Anthropic's "Building effective agents":

  1. Augmented LLM        LLM + retrieval + tools + memory (the unit everything else is built from)
  2. Prompt chaining      fixed steps with a programmatic gate between them
  3. Routing              classify, then hand off to a specialised prompt/model
  4. Parallelization      sectioning (independent aspects) and voting (independent judgments)
  5. Orchestrator-workers the LLM decides the subtasks at runtime
  6. Evaluator-optimizer  generate -> evaluate -> refine until the criteria pass
  7. Autonomous agent     tool use until end_turn, bounded by a step limit
  8. Pattern selector     deterministic recommendation with trade-offs

Workflows = code owns the control flow (predictable). Agents = the model owns it (flexible).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

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
    elapsed_ms,
    get_async_client,
    get_client,
    section,
    text_block,
    text_of,
    tool_use_block,
)

TODAY = date(2026, 10, 11)   # fixed "today" keeps the demo deterministic


def cj(obj: Any) -> str:
    """Compact JSON (identical to JSON.stringify) for printing tool inputs/results."""
    return json.dumps(obj, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Ledger: every call is attributed to a pattern so cost/latency can be compared
# ---------------------------------------------------------------------------
class Ledger:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "cost": 0.0})

    def record(self, pattern: str, msg: anthropic.types.Message) -> None:
        row = self.rows[pattern]
        row["calls"] += 1
        row["in"] += msg.usage.input_tokens
        row["out"] += msg.usage.output_tokens
        row["cost"] += cost_usd(msg.model, msg.usage)


LEDGER = Ledger()


def call(client: anthropic.Anthropic, pattern: str, **kwargs: Any) -> anthropic.types.Message:
    msg = client.messages.create(**kwargs)
    LEDGER.record(pattern, msg)
    return msg


async def acall(client: anthropic.AsyncAnthropic, pattern: str, **kwargs: Any) -> anthropic.types.Message:
    msg = await client.messages.create(**kwargs)
    LEDGER.record(pattern, msg)
    return msg


# ===========================================================================
# 1. AUGMENTED LLM: retrieval + tools + memory around one model
# ===========================================================================
KB = [
    ("returns-policy", "Returns: you can return most items within 30 days of delivery for a full refund. Refunds go to "
                       "the original payment method within 5 business days after we receive the return."),
    ("warranty", "Warranty: Aurora appliances carry a 2-year limited warranty covering manufacturing defects. Damage "
                 "in transit must be reported within 7 days of delivery for a free replacement."),
    ("shipping", "Shipping: standard delivery takes 3-5 business days; express takes 1-2. Tracking links are emailed "
                 "at dispatch."),
    ("loyalty", "Aurora Rewards: Gold members get free return shipping and a dedicated support line. Points are "
                "earned on every purchase."),
    ("blender-manual", "Aurora blender quick start: lock the jar onto the base, choose a program and press start. "
                       "Never run it empty."),
]
STOP = {"a", "an", "and", "the", "to", "of", "in", "is", "it", "my", "i", "can", "how", "would", "still", "for",
        "on", "be", "are", "with", "most", "all", "if", "we", "you", "your", "do", "from"}


def tokenize(text: str) -> list[str]:
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if len(tok) > 3 and tok.endswith("s"):     # naive stemming: returns -> return
            tok = tok[:-1]
        if tok not in STOP:
            out.append(tok)
    return out


def retrieve(query: str, k: int = 2) -> list[tuple[float, str, str]]:
    """Tiny TF-IDF retriever - the 'retrieval' augmentation (D3.5/D3.6 go deeper)."""
    n_docs = len(KB)
    df = Counter(t for _, text in KB for t in set(tokenize(text)))
    q_terms = set(tokenize(query))
    scored = []
    for doc_id, text in KB:
        tf = Counter(tokenize(text))
        score = sum(tf[t] * math.log(1 + n_docs / df[t]) for t in q_terms if t in tf)
        scored.append((round(score, 4), doc_id, text))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for s in scored[:k] if s[0] > 0]


MEMORY: dict[str, list[str]] = {"C-1042": ["Gold Aurora Rewards member", "prefers email updates"]}
ORDERS_DB = {"A-7731": {"order_id": "A-7731", "item": "Aurora blender", "status": "delivered", "delivered_on": "2026-10-02"}}

AUGMENTED_TOOLS: list[dict[str, Any]] = [
    {"name": "get_order_status", "description": "Look up an order by id. Call this whenever the customer mentions an order.",
     "strict": True, "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                                      "required": ["order_id"], "additionalProperties": False}},
    {"name": "save_memory", "description": "Persist a durable fact about the customer for future conversations.",
     "strict": True, "input_schema": {"type": "object", "properties": {"fact": {"type": "string"}},
                                      "required": ["fact"], "additionalProperties": False}},
]


def run_augmented(client: anthropic.Anthropic) -> None:
    customer = "C-1042"
    question = "Can I still return my blender from order A-7731, and do I pay for return shipping?"
    docs = retrieve(question)
    print(f"  question : {question}")
    print(f"  memory   : {'; '.join(MEMORY[customer])}")
    for score, doc_id, _ in docs:
        print(f"  retrieved: {doc_id:<15} score={score:.2f}")
    system = (f"You are the support assistant for Aurora Home. Today is {TODAY.isoformat()}.\n"
              f"Known facts about this customer: {'; '.join(MEMORY[customer])}.\n"
              "Answer only from the documents and tool results; cite document ids in [brackets].")
    context = "\n".join(f'<document id="{d}">{t}</document>' for _, d, t in docs)
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"<documents>\n{context}\n</documents>\n\n{question}"}]
    for _ in range(3):  # tiny loop: the augmented LLM may call tools before answering
        msg = call(client, "1 augmented LLM", model=MODEL_SONNET, max_tokens=1024, system=system,
                   tools=AUGMENTED_TOOLS, messages=messages)
        if msg.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for block in msg.content:
            if block.type != "tool_use":
                continue
            if block.name == "get_order_status":
                out: Any = ORDERS_DB.get(block.input["order_id"], {"error": "not found"})
            else:
                MEMORY[customer].append(block.input["fact"])
                out = {"saved": True}
            print(f"  tool     : {block.name}({cj(block.input)}) -> {cj(out)}")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": cj(out)})
        messages.append({"role": "user", "content": results})   # all parallel results in ONE user turn
    print(f"  answer   : {text_of(msg)}")
    print(f"  memory now: {'; '.join(MEMORY[customer])}")


# ===========================================================================
# 2. PROMPT CHAINING with a programmatic gate
# ===========================================================================
CHANGELOG = """JIRA-4821 fix NPE in PdfExporter when dashboard has no widgets
JIRA-4822 bump pdfkit to 3.2
JIRA-4830 add 'Export to PDF' button on dashboards
JIRA-4835 cache widget queries: dashboard p95 load 2.1s -> 0.9s
JIRA-4840 refactor auth middleware (no behaviour change)"""
INTERNAL_TERMS = ("pdfkit", "refactor", "middleware", "npe")


def gate_bullets(text: str) -> list[str]:
    bullets = [b for b in text.splitlines() if b.strip().startswith("-")]
    problems = []
    if not 2 <= len(bullets) <= 6:
        problems.append(f"expected 2-6 bullets, got {len(bullets)}")
    ids = re.findall(r"\b[A-Z]{2,}-\d+\b", text)
    if ids:
        problems.append(f"internal ticket ids present: {', '.join(ids)}")
    terms = [t for t in INTERNAL_TERMS if t in text.lower()]
    if terms:
        problems.append(f"internal-only changes present: {', '.join(terms)}")
    return problems


def run_chaining(client: anthropic.Anthropic) -> None:
    step1_system = "Extract the customer-visible changes from an engineering changelog as '-' bullets."
    user = CHANGELOG
    bullets = ""
    for attempt in (1, 2):
        msg = call(client, "2 prompt chaining", model=MODEL_HAIKU, max_tokens=400, system=step1_system,
                   messages=[{"role": "user", "content": user}])
        bullets = text_of(msg)
        problems = gate_bullets(bullets)
        print(f"  step 1 attempt {attempt} (Haiku) -> gate: {'PASS' if not problems else 'FAIL - ' + '; '.join(problems)}")
        if not problems:
            break
        user = f"{CHANGELOG}\n\nGate feedback - fix these problems: {'; '.join(problems)}"
    for line in bullets.splitlines():
        print(f"      {line}")
    msg = call(client, "2 prompt chaining", model=MODEL_SONNET, max_tokens=600,
               system="Write a customer-facing release note (<= 60 words, friendly, no jargon) from these bullets.",
               messages=[{"role": "user", "content": bullets}])
    print(f"  step 2 (Sonnet) release note: {text_of(msg)}")


# ===========================================================================
# 3. ROUTING: cheap classifier -> specialised handler (prompt + model)
# ===========================================================================
class RouteDecision(BaseModel):
    route: Literal["refund", "technical", "general"]
    reason: str


HANDLERS = {
    "refund": (MODEL_SONNET, "You are a refund specialist. Apply the refund policy precisely and state next steps."),
    "technical": (MODEL_OPUS, "You are a senior technical support engineer. Diagnose step by step."),
    "general": (MODEL_HAIKU, "You are a friendly general assistant. Answer briefly."),
}


def run_routing(client: anthropic.Anthropic) -> None:
    queries = [
        "I was charged twice for my subscription this month - can I get a refund?",
        "The app crashes with error E1043 whenever I open a shared dashboard.",
        "Do you have a dark mode?",
    ]
    for q in queries:
        decision = client.messages.parse(
            model=MODEL_HAIKU, max_tokens=200, output_format=RouteDecision,
            system="You route customer queries to refund, technical or general handling.",
            messages=[{"role": "user", "content": q}],
        )
        LEDGER.record("3 routing", decision)
        route = decision.parsed_output.route if decision.parsed_output else "general"   # safe default
        model, system = HANDLERS[route]
        answer = call(client, "3 routing", model=model, max_tokens=600, system=system,
                      messages=[{"role": "user", "content": q}])
        print(f"  \"{q[:52]}...\"" if len(q) > 55 else f"  \"{q}\"")
        print(f"      route={route:<9} handler={model:<18} -> {text_of(answer)[:88]}")


# ===========================================================================
# 4. PARALLELIZATION: sectioning + voting (async client, concurrent calls)
# ===========================================================================
SNIPPET = '''def get_user(conn, user_id):
    rows = conn.execute(f"SELECT * FROM users WHERE id = {user_id}").fetchall()
    for r in rows:
        time.sleep(0.1)
    return rows[0]'''
ASPECTS = ["security", "performance", "readability"]
COMMENT = "You people are idiots and I know where your office is."
PERSPECTIVES = ["strict", "lenient", "policy-literal"]


async def run_parallel() -> None:
    aclient = get_async_client(responder)

    async def timed(coro: Any) -> tuple[anthropic.types.Message, float]:
        t = time.perf_counter()
        m = await coro
        return m, elapsed_ms(t)

    t0 = time.perf_counter()
    reviews = await asyncio.gather(*[
        timed(acall(aclient, "4 parallelization", model=MODEL_SONNET, max_tokens=300,
                    system=f"You are a {aspect} reviewer. Report the single most important {aspect} issue.",
                    messages=[{"role": "user", "content": SNIPPET}]))
        for aspect in ASPECTS])
    wall = elapsed_ms(t0)
    print("  sectioning (3 independent aspects, concurrently):")
    for aspect, (m, _) in zip(ASPECTS, reviews):
        print(f"    {aspect:<12} {text_of(m)}")
    print(f"    wall-clock ~{round(wall, -1):.0f} ms vs ~{round(sum(ms for _, ms in reviews), -1):.0f} ms if run sequentially")

    # Sampling params (temperature) are not available on current models, so diversity comes
    # from framing: each voter gets a different reviewing perspective.
    votes = await asyncio.gather(*[
        acall(aclient, "4 parallelization", model=MODEL_HAIKU, max_tokens=50,
              system=f"You are a {p} content-policy reviewer. Reply VIOLATION or OK, then a 5-word reason.",
              messages=[{"role": "user", "content": COMMENT}])
        for p in PERSPECTIVES])
    labels = [text_of(v).split()[0].strip(".:") for v in votes]
    tally = Counter(labels)
    winner, count = tally.most_common(1)[0]
    print("  voting (3 perspectives on one comment):")
    for p, v in zip(PERSPECTIVES, votes):
        print(f"    {p:<15} {text_of(v)}")
    agreement = count / len(labels)
    action = "remove + escalate to human (split vote)" if agreement < 1 else "auto-remove"
    print(f"    majority={winner} ({count}/{len(labels)}, agreement {agreement:.0%}) -> {action}")


# ===========================================================================
# 5. ORCHESTRATOR-WORKERS: subtasks decided at runtime by the LLM
# ===========================================================================
class Subtask(BaseModel):
    id: str
    area: str
    instruction: str


class Plan(BaseModel):
    subtasks: list[Subtask]


def run_orchestrator(client: anthropic.Anthropic) -> None:
    feature = "Add 'Export to PDF' for dashboards, plus an optional weekly email that attaches the PDF."
    plan_msg = client.messages.parse(
        model=MODEL_OPUS, max_tokens=1000, output_format=Plan,
        system="You are an orchestrator. Break the feature into the minimum set of independent subtasks.",
        messages=[{"role": "user", "content": feature}],
    )
    LEDGER.record("5 orchestrator-workers", plan_msg)
    plan = plan_msg.parsed_output
    if plan is None:
        print("  orchestrator returned no plan - stop")
        return
    print(f"  orchestrator (Opus) decided {len(plan.subtasks)} subtasks for: {feature}")
    outputs = []
    for st in plan.subtasks:   # workers get ONLY their subtask: isolated, focused context
        m = call(client, "5 orchestrator-workers", model=MODEL_SONNET, max_tokens=400,
                 system=f"You are a worker engineer for the {st.area} area. Produce a 1-2 line change plan.",
                 messages=[{"role": "user", "content": st.instruction}])
        outputs.append(f"[{st.id} {st.area}] {text_of(m)}")
        print(f"    worker {st.id} ({st.area:<9}) -> {text_of(m)}")
    synth = call(client, "5 orchestrator-workers", model=MODEL_OPUS, max_tokens=600,
                 system="You are the orchestrator. Integrate the worker outputs into one ordered delivery plan.",
                 messages=[{"role": "user", "content": "\n".join(outputs)}])
    print(f"  synthesis: {text_of(synth)}")


# ===========================================================================
# 6. EVALUATOR-OPTIMIZER: cheap code checks first, LLM judge for the subjective part
# ===========================================================================
FACTS = "Aurora blender: 1200 W motor, 1.5 L glass jar, 6 programs, 2-year warranty."
BANNED = ("best", "ultimate", "perfect")


class Verdict(BaseModel):
    passed: bool
    feedback: str


def code_checks(text: str) -> list[str]:
    problems = []
    words = len(text.split())
    if words > 60:
        problems.append(f"{words} words (max 60)")
    hits = [b for b in BANNED if re.search(rf"\b{b}\b", text.lower())]
    if hits:
        problems.append(f"banned superlatives: {', '.join(hits)}")
    if "warranty" not in text.lower():
        problems.append("must mention the 2-year warranty")
    return problems


def run_evaluator_optimizer(client: anthropic.Anthropic) -> None:
    feedback = ""
    for iteration in range(1, 4):
        prompt = f"Product facts: {FACTS}\nWrite the description." + (f"\nReviewer feedback: {feedback}" if feedback else "")
        draft = text_of(call(client, "6 evaluator-optimizer", model=MODEL_SONNET, max_tokens=400,
                             system="You write product descriptions: max 60 words, mention the warranty, no superlatives.",
                             messages=[{"role": "user", "content": prompt}]))
        problems = code_checks(draft)
        if problems:
            feedback = "; ".join(problems)
            print(f"  iter {iteration}: code checks FAIL ({feedback})")
            continue
        verdict_msg = client.messages.parse(
            model=MODEL_SONNET, max_tokens=300, output_format=Verdict,
            system="You evaluate product descriptions against the product facts. Fail any claim not in the facts.",
            messages=[{"role": "user", "content": f"Facts: {FACTS}\nCandidate: {draft}"}],
        )
        LEDGER.record("6 evaluator-optimizer", verdict_msg)
        verdict = verdict_msg.parsed_output
        if verdict and verdict.passed:
            print(f"  iter {iteration}: code checks PASS, LLM judge PASS")
            print(f"  final: {draft}")
            return
        feedback = verdict.feedback if verdict else "judge unavailable"
        print(f"  iter {iteration}: code checks PASS, LLM judge FAIL ({feedback})")
    print("  stopped after 3 iterations -> send best draft to a human editor")


# ===========================================================================
# 7. AUTONOMOUS AGENT: tool use until end_turn, bounded by a step limit
# ===========================================================================
CUSTOMERS = {"jane@example.com": {"customer_id": "C-1042", "name": "Jane Park", "tier": "Gold"}}
ORDERS = {"C-1042": [
    {"order_id": "A-7731", "item": "Aurora blender", "delivered_on": "2026-10-02"},
    {"order_id": "A-7790", "item": "Aurora blender", "delivered_on": "2026-10-08"},
]}
REPLACEMENTS: list[dict[str, str]] = []


def _schema(**props: str) -> dict[str, Any]:
    return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()},
            "required": list(props), "additionalProperties": False}


AGENT_TOOLS: list[dict[str, Any]] = [
    {"name": "lookup_customer", "description": "Find a customer by email. Call first when only an email is known.",
     "strict": True, "input_schema": _schema(email="string")},
    {"name": "list_orders", "description": "List a customer's orders with delivery dates.",
     "strict": True, "input_schema": _schema(customer_id="string")},
    {"name": "check_warranty", "description": "Check transit-damage and warranty eligibility for an order.",
     "strict": True, "input_schema": _schema(order_id="string")},
    {"name": "create_replacement", "description": "Ship a free replacement. Only after check_warranty says eligible.",
     "strict": True, "input_schema": _schema(order_id="string", reason="string")},
]


def execute_tool(name: str, args: dict[str, Any]) -> tuple[Any, bool]:
    """Real tool implementations. Returns (result, is_error). Policy is enforced HERE, not just in the prompt."""
    all_orders = {o["order_id"]: o for orders in ORDERS.values() for o in orders}
    if name == "lookup_customer":
        c = CUSTOMERS.get(args["email"])
        return (c, False) if c else ({"error": "no such customer"}, True)
    if name == "list_orders":
        return ORDERS.get(args["customer_id"], []), False
    if name in ("check_warranty", "create_replacement"):
        order = all_orders.get(args["order_id"])
        if not order:
            return {"error": "unknown order"}, True
        days = (TODAY - date.fromisoformat(order["delivered_on"])).days
        eligible = days <= 7
        if name == "check_warranty":
            return {"order_id": order["order_id"], "days_since_delivery": days, "transit_damage_eligible": eligible}, False
        if not eligible:
            return {"error": "outside 7-day transit-damage window; escalate to a human"}, True
        rep = {"replacement_id": f"R-{5001 + len(REPLACEMENTS)}", "order_id": order["order_id"], "ship_eta": "2026-10-14"}
        REPLACEMENTS.append(rep)
        return rep, False
    return {"error": f"unknown tool {name}"}, True


AGENT_SYSTEM = ("You are an autonomous support agent for Aurora Home. Today is 2026-10-11. Use the tools to "
                "investigate and resolve the issue end to end, then summarise what you did.")


def run_agent(client: anthropic.Anthropic, task: str, max_steps: int) -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        msg = call(client, "7 autonomous agent", model=MODEL_OPUS, max_tokens=2048, system=AGENT_SYSTEM,
                   tools=AGENT_TOOLS, messages=messages)
        if msg.stop_reason == "end_turn":
            print(f"    step {step} (in={msg.usage.input_tokens}): end_turn -> {text_of(msg)}")
            return
        if msg.stop_reason != "tool_use":
            print(f"    step {step}: stop_reason={msg.stop_reason} -> hand off to a human")
            return
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for block in msg.content:
            if block.type == "tool_use":
                out, is_error = execute_tool(block.name, block.input)
                flag = " (is_error)" if is_error else ""
                print(f"    step {step} (in={msg.usage.input_tokens}): {block.name}({cj(block.input)}) -> {cj(out)}{flag}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": cj(out), "is_error": is_error})
        messages.append({"role": "user", "content": results})
    print(f"    step limit ({max_steps}) reached -> stop, escalate to a human with the transcript")


# ===========================================================================
# 8. PATTERN SELECTOR (deterministic)
# ===========================================================================
@dataclass
class Scenario:
    name: str
    steps_known: bool = True
    multi_step: bool = False
    distinct_categories: bool = False
    independent_subtasks: bool = False
    subtasks_known_upfront: bool = True
    needs_consensus: bool = False
    clear_criteria_iterative: bool = False
    open_ended: bool = False


PROFILE = {  # predictability, relative cost, latency shape
    "augmented LLM": ("high", "1x", "one call (+tool round trips)"),
    "prompt chaining": ("high", "2-4x", "sum of steps"),
    "routing": ("high", "~1.1x", "router + one handler"),
    "parallel: sectioning": ("high", "Nx", "slowest branch"),
    "parallel: voting": ("high", "Nx", "slowest voter"),
    "orchestrator-workers": ("medium", "3-10x", "plan + workers + synthesis"),
    "evaluator-optimizer": ("medium", "2-6x", "iterations x (gen + eval)"),
    "autonomous agent": ("low", "5-20x+", "unbounded -> step limit"),
}


def select_pattern(s: Scenario) -> tuple[str, str]:
    if s.open_ended and not s.steps_known:
        return "autonomous agent", "path depends on what each tool call reveals"
    if s.independent_subtasks and not s.subtasks_known_upfront:
        return "orchestrator-workers", "subtasks depend on the input; cannot be enumerated in advance"
    if s.needs_consensus:
        return "parallel: voting", "independent judgments + majority raise recall/confidence"
    if s.independent_subtasks:
        return "parallel: sectioning", "known independent aspects; latency = slowest branch"
    if s.distinct_categories:
        return "routing", "distinct input classes served best by specialised prompts/models"
    if s.clear_criteria_iterative:
        return "evaluator-optimizer", "clear criteria and iteration measurably improves output"
    if s.multi_step:
        return "prompt chaining", "fixed sequence of simpler steps with gates"
    return "augmented LLM", "one call with retrieval/tools is enough"


SCENARIOS = [
    Scenario("FAQ answers from a product knowledge base"),
    Scenario("Inbound email: sales vs support vs billing replies", distinct_categories=True),
    Scenario("Contract memo: extract clauses -> compare to playbook -> draft", multi_step=True),
    Scenario("Code review for security, performance and style", independent_subtasks=True),
    Scenario("Flag fraudulent marketplace listings (misses are costly)", needs_consensus=True),
    Scenario("Implement a feature touching an unknown set of files", independent_subtasks=True,
             subtasks_known_upfront=False),
    Scenario("Localise brand copy until a tone rubric is met", clear_criteria_iterative=True),
    Scenario("Resolve open-ended issues across several back-office systems", steps_known=False, open_ended=True),
]


# ===========================================================================
# Offline responder - one function serving every pattern (dispatch on the system prompt)
# ===========================================================================
def responder(req: MockRequest) -> Any:
    s, u = req.system_text, req.last_user_text
    if "support assistant for Aurora Home" in s:
        if not req.tool_results:
            return [text_block("Let me check that order."),
                    tool_use_block("get_order_status", {"order_id": "A-7731"}),
                    tool_use_block("save_memory", {"fact": "started a return for order A-7731 (Aurora blender) on 2026-10-11"})]
        return ("Yes. Order A-7731 (Aurora blender) was delivered on 2026-10-02, so you are inside the 30-day window "
                "and can return it until 2026-11-01 [returns-policy]. As a Gold member your return shipping is free "
                "[loyalty]. The refund reaches your original payment method within 5 business days of us receiving it "
                "[returns-policy]. I will email the prepaid label, as you prefer email.")
    if "Extract the customer-visible changes" in s:
        if "Gate feedback" in u:
            return ("- New 'Export to PDF' button on dashboards\n- Dashboards load more than twice as fast\n"
                    "- Fixed an error when exporting an empty dashboard")
        return ("- JIRA-4830: new 'Export to PDF' button on dashboards\n- Dashboards load faster (p95 2.1s -> 0.9s)\n"
                "- Fixed an error when exporting an empty dashboard\n- Upgraded pdfkit to 3.2")
    if "customer-facing release note" in s:
        return ("New this week: export any dashboard to PDF with one click, dashboards now load more than twice as "
                "fast, and exporting an empty dashboard no longer shows an error.")
    if "You route customer queries" in s:
        if re.search(r"refund|charged", u, re.I):
            return json.dumps({"route": "refund", "reason": "billing dispute with refund request"})
        if re.search(r"error|crash", u, re.I):
            return json.dumps({"route": "technical", "reason": "reproducible crash with an error code"})
        return json.dumps({"route": "general", "reason": "simple product question"})
    if "refund specialist" in s:
        return "Duplicate charges are refunded in full: I have flagged the second charge; expect it back in 5 business days."
    if "technical support engineer" in s:
        return "E1043 means the shared dashboard references a deleted data source: re-link the source, then clear the cache."
    if "general assistant" in s:
        return "Yes - turn it on under Settings > Appearance > Dark mode."
    for aspect, finding in (("security", "SQL injection: user_id is interpolated into the query; use a parameterised query."),
                            ("performance", "SELECT * plus a sleep per row; select one row by primary key instead."),
                            ("readability", "rows[0] raises IndexError when no user exists; return None and add type hints.")):
        if f"a {aspect} reviewer" in s:
            return finding
    if "content-policy reviewer" in s:
        if "lenient" in s:
            return "OK - insulting but no explicit threat"
        return "VIOLATION - implied threat to staff safety"
    if "You are an orchestrator" in s:
        return json.dumps({"subtasks": [
            {"id": "w1", "area": "frontend", "instruction": "Add an Export to PDF button with progress state to the dashboard toolbar."},
            {"id": "w2", "area": "backend", "instruction": "Create POST /dashboards/{id}/export that renders the dashboard to PDF."},
            {"id": "w3", "area": "scheduler", "instruction": "Add a weekly schedule option and an email job attaching the PDF."},
        ]})
    if "worker engineer for the frontend" in s:
        return "ExportButton component in DashboardToolbar; poll export job status; show toast with download link."
    if "worker engineer for the backend" in s:
        return "Async export job: headless renderer -> PDF in object storage; endpoint returns job id; 10/min rate limit."
    if "worker engineer for the scheduler" in s:
        return "weekly_export cron per subscription; reuse export job; email via notification service with signed URL."
    if "Integrate the worker outputs" in s:
        return "Order: (1) backend export job + endpoint, (2) frontend button against it, (3) scheduler reusing the job. Ship 1+2 behind a flag, then 3."
    if "You write product descriptions" in s:
        if "unsupported" in u:
            return ("Meet the Aurora blender: a 1200 W motor, a 1.5 L glass jar and 6 programs for smoothies, soups "
                    "and sauces. Backed by a 2-year warranty.")
        if "Reviewer feedback" in u:
            return ("Meet the Aurora blender: a 1200 W motor, a 1.5 L glass jar, 6 programs and dishwasher-safe "
                    "titanium blades. Backed by a 2-year warranty.")
        return ("The Aurora is the best blender you will ever own. Its ultimate 1200 W motor crushes ice in seconds, "
                "the elegant 1.5 L glass jar looks perfect on any counter, and six smart programs handle smoothies, "
                "soups, nut butters, sauces and frozen desserts so you never have to guess again. It is simply the "
                "most versatile kitchen companion for busy families who love fresh, healthy food every day.")
    if "You evaluate product descriptions" in s:
        if "titanium" in u:
            return json.dumps({"passed": False, "feedback": "unsupported claim: 'dishwasher-safe titanium blades' is not in the facts"})
        return json.dumps({"passed": True, "feedback": "accurate, within limits, warranty mentioned"})
    if "autonomous support agent" in s:
        called = [b.get("name") for m in req.messages if m.get("role") == "assistant"
                  for b in (m.get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_use"]
        if "lookup_customer" not in called:
            return [text_block("I'll identify the customer first."), tool_use_block("lookup_customer", {"email": "jane@example.com"})]
        if "list_orders" not in called:
            return [text_block("Now her orders."), tool_use_block("list_orders", {"customer_id": "C-1042"})]
        if "check_warranty" not in called:
            return [text_block("A-7790 is the recent blender; checking eligibility."), tool_use_block("check_warranty", {"order_id": "A-7790"})]
        if "create_replacement" not in called:
            return [text_block("Eligible - creating the replacement."),
                    tool_use_block("create_replacement", {"order_id": "A-7790", "reason": "glass jar arrived cracked (transit damage)"})]
        return ("Resolved: replacement R-5001 ships for order A-7790 (delivered 3 days ago, inside the 7-day "
                "transit-damage window), ETA 2026-10-14. No return of the cracked jar needed.")
    return "OK"


def main() -> None:
    banner("D1.3 - Architectural Patterns - Workflow, Agentic, Augmented LLM")
    client = get_client(responder)

    section("1. Augmented LLM: retrieval + tools + memory")
    run_augmented(client)

    section("2. Workflow - prompt chaining with a gate")
    run_chaining(client)

    section("3. Workflow - routing")
    run_routing(client)

    section("4. Workflow - parallelization (sectioning + voting)")
    asyncio.run(run_parallel())

    section("5. Workflow - orchestrator-workers")
    run_orchestrator(client)

    section("6. Workflow - evaluator-optimizer")
    run_evaluator_optimizer(client)

    section("7. Autonomous agent loop (tool use until end_turn)")
    task = "Customer jane@example.com says the glass jar of her new Aurora blender arrived cracked. Resolve it."
    print("  run A: max_steps=6   (note 'in=' grows: the whole transcript is resent every step)")
    run_agent(client, task, max_steps=6)
    print("  run B: same task, max_steps=2 (guardrail demo)")
    run_agent(client, task, max_steps=2)

    section("Cost per pattern (this run)")
    print(f"  {'pattern':<24}{'calls':>6}{'in tok':>8}{'out tok':>9}{'cost $':>10}")
    for name, row in LEDGER.rows.items():
        print(f"  {name:<24}{int(row['calls']):>6}{int(row['in']):>8}{int(row['out']):>9}{row['cost']:>10.5f}")
    print("  (agent row includes run B) - autonomy costs tokens: tool schemas + growing history on every call")

    section("8. Pattern selector: scenario -> recommended pattern")
    for s in SCENARIOS:
        pattern, why = select_pattern(s)
        pred, cost, lat = PROFILE[pattern]
        print(f"  {s.name}")
        print(f"     -> {pattern:<21} | {why}")
        print(f"        predictability={pred}, relative cost={cost}, latency={lat}")

    section("Summary")
    print("  Start with an augmented LLM; add workflow structure when steps are known; reach for an")
    print("  agent only when the path is genuinely open-ended - and then bound it (step limits, tool policy).")


if __name__ == "__main__":
    main()
