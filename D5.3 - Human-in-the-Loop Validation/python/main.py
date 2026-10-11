"""
D5.3 - Human-in-the-Loop Validation (Python)

Five HITL strategies wired together around a refund-decisioning assistant:
  (a) confidence/risk-based ROUTING: Claude returns a structured decision + a
      self-reported confidence; deterministic risk rules (amount, regulated category,
      retrieval support, fraud) route to auto-approve / human review / reject.
  (b) APPROVAL GATES in an agent tool loop: high-risk tool calls (issue_refund over
      limit, send_external_email) pause for a scripted human approve/edit/deny, fed
      back as a tool_result.
  (c) QA SAMPLING of auto-approved items (stratified) + reviewer-agreement metric.
  (d) FEEDBACK capture -> a corrections dataset.
  (e) METRICS: automation rate, review load, SLA attainment, escalation latency.

Only the LLM decisions go through the shared client; all routing/QA/metrics are real
deterministic code.

Run (from repo root):
    .venv/Scripts/python "D5.3 - Human-in-the-Loop Validation/python/main.py"
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_OPUS,
    MockRequest,
    banner,
    get_client,
    section,
)

DATA = Path(__file__).resolve().parents[1] / "data"
SAMPLE_RNG = random.Random(4)   # QA sampling (separate stream -> stable sample)
# Deterministic simulated time-in-queue per review item (hours), cycled by position.
REVIEW_LATENCIES = [1.5, 2.0, 3.0, 3.5]


# ===========================================================================
# (a) Confidence/risk-based routing
# ===========================================================================
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "deny"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["decision", "confidence", "rationale"],
    "additionalProperties": False,
}


@dataclass
class Routed:
    item_id: str
    route: str          # AUTO_APPROVE | HUMAN_REVIEW | REJECT
    model_decision: str
    confidence: float
    triggers: list[str]
    sla_hours: float = 0.0


def model_decide(client: Any, item: dict[str, Any]) -> dict[str, Any]:
    """Claude returns a structured decision + a self-reported confidence. Self-reported
    confidence is a USEFUL SIGNAL, not ground truth - it is one input to the router."""
    payload = json.dumps({k: item[k] for k in ("id", "amount", "category", "support", "text")})
    msg = client.messages.create(
        model=MODEL_OPUS,
        max_tokens=256,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
        system="You are a refund adjudicator. Return a decision, a calibrated confidence in [0,1], "
        "and a one-line rationale. Base it only on the provided fields.",
        messages=[{"role": "user", "content": f"Adjudicate this refund:\n{payload}"}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return json.loads(text)


def route(item: dict[str, Any], decision: dict[str, Any], cfg: dict[str, Any], regulated: list[str]) -> Routed:
    triggers: list[str] = []
    # Hard reject: an upstream fraud signal overrides the model entirely.
    if item["fraud_signal"]:
        return Routed(item["id"], "REJECT", decision["decision"], decision["confidence"], ["fraud_signal"])
    if item["amount"] > cfg["auto_amount_limit"]:
        triggers.append(f"amount>{cfg['auto_amount_limit']}")
    if item["category"] in regulated:
        triggers.append(f"regulated:{item['category']}")
    if item["support"] < cfg["support_min"]:
        triggers.append("low_retrieval_support")
    if decision["confidence"] < cfg["confidence_min"]:
        triggers.append("low_confidence")
    r = "HUMAN_REVIEW" if triggers else "AUTO_APPROVE"
    return Routed(item["id"], r, decision["decision"], decision["confidence"], triggers)


# ===========================================================================
# (b) Approval gates in an agent tool loop
# ===========================================================================
AGENT_TOOLS = [
    {"name": "lookup_order", "description": "look up an order", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
    {"name": "issue_refund", "description": "issue a refund", "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}}}},
    {"name": "send_external_email", "description": "email a customer", "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}}},
]
REFUND_APPROVAL_LIMIT = 200


def is_high_risk(name: str, args: dict[str, Any]) -> bool:
    if name == "send_external_email":
        return True
    if name == "issue_refund" and float(args.get("amount", 0)) > REFUND_APPROVAL_LIMIT:
        return True
    return False


@dataclass
class HumanDecision:
    action: str                      # approve | edit | deny
    edited_args: dict[str, Any] | None = None
    note: str = ""


def run_agent_with_gates(client: Any, scripted: list[HumanDecision], corrections: list[dict[str, Any]]) -> list[str]:
    """Manual tool loop that GATES high-risk tool calls behind a (scripted) human
    approve/edit/deny before executing, feeding the outcome back as a tool_result."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": "AGENT: refund order ORD-5 for $250 and email the customer a confirmation."}]
    transcript: list[str] = []
    script = iter(scripted)
    for _ in range(6):  # step budget
        resp = client.messages.create(model=MODEL_OPUS, max_tokens=512, tools=AGENT_TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            transcript.append("agent: " + "".join(b.text for b in resp.content if getattr(b, "type", None) == "text"))
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in [b for b in resp.content if getattr(b, "type", None) == "tool_use"]:
            args = dict(tu.input)
            if is_high_risk(tu.name, args):
                hd = next(script)
                if hd.action == "deny":
                    transcript.append(f"GATE  {tu.name}({args}) -> human DENY ({hd.note})")
                    corrections.append({"source": "approval_gate", "tool": tu.name, "from": args, "action": "deny", "note": hd.note})
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "Denied by human reviewer.", "is_error": True})
                    continue
                if hd.action == "edit" and hd.edited_args:
                    corrections.append({"source": "approval_gate", "tool": tu.name, "from": args, "to": hd.edited_args, "action": "edit", "note": hd.note})
                    args = {**args, **hd.edited_args}
                    transcript.append(f"GATE  {tu.name} -> human EDIT to {args} ({hd.note})")
                else:
                    transcript.append(f"GATE  {tu.name}({args}) -> human APPROVE")
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": f"executed {tu.name} with {args}"})
            else:
                transcript.append(f"auto  {tu.name}({args}) -> executed (low risk)")
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": f"executed {tu.name} with {args}"})
        messages.append({"role": "user", "content": results})
    return transcript


# ===========================================================================
# (c) QA sampling of auto-approved items + reviewer agreement
# ===========================================================================
def stratified_sample(auto: list[dict[str, Any]], frac: float) -> list[dict[str, Any]]:
    """Stratify by category, then random-sample within each stratum (fixed seed)."""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for it in auto:
        by_cat.setdefault(it["category"], []).append(it)
    sample: list[dict[str, Any]] = []
    for cat, items in by_cat.items():
        k = max(1, round(len(items) * frac))
        sample.extend(SAMPLE_RNG.sample(items, min(k, len(items))))
    return sorted(sample, key=lambda it: it["id"])


def qa_reviewer(item: dict[str, Any]) -> str:
    """Simulated independent QA reviewer re-deciding an auto-approved item. Deterministic:
    the reviewer flags one specific item (duplicate-charge edge case) as a disagreement."""
    if item["id"] == "C08":
        return "deny"  # reviewer judges the duplicate-charge evidence insufficient
    return "approve"


# ===========================================================================
# Offline responder
# ===========================================================================
def build_responder() -> Callable[[MockRequest], Any]:
    def responder(req: MockRequest) -> Any:
        q = req.last_user_text
        names = req.tool_names

        # ---- Agent tool loop ----
        if "AGENT:" in q or req.tool_results:
            # Drive: refund first, then email, then finish.
            did_refund = "issue_refund" in req.all_text
            did_email = "send_external_email" in req.all_text
            results_text = "\n".join(
                b.get("content", "") if isinstance(b, dict) else "" for b in req.tool_results
            )
            if not did_refund and names:
                return [{"type": "tool_use", "id": "t_refund", "name": "issue_refund", "input": {"order_id": "ORD-5", "amount": 250}}]
            if did_refund and not did_email and names:
                return [{"type": "tool_use", "id": "t_email", "name": "send_external_email", "input": {"to": "jane.doe@example.com", "body": "Your refund is processed."}}]
            return "Done: refund issued (as approved/edited) and confirmation email sent."

        # ---- Structured refund adjudication ----
        try:
            item = json.loads(q.split("Adjudicate this refund:\n", 1)[1])
        except (IndexError, json.JSONDecodeError):
            return json.dumps({"decision": "deny", "confidence": 0.5, "rationale": "unparseable"})
        if item["support"] < 0.5:
            return json.dumps({"decision": "deny", "confidence": 0.55, "rationale": "weak evidence / no order record"})
        if item["amount"] > 300:
            return json.dumps({"decision": "approve", "confidence": 0.62, "rationale": "large amount, approve with lower confidence"})
        return json.dumps({"decision": "approve", "confidence": 0.90, "rationale": "clear, well-supported claim"})

    return responder


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    banner("D5.3 - Human-in-the-Loop Validation")
    data = json.loads((DATA / "decisions.json").read_text(encoding="utf-8"))
    cfg = data["thresholds"]
    regulated = data["regulated_categories"]
    items = data["items"]
    client = get_client(build_responder())
    corrections: list[dict[str, Any]] = []

    # --- (a) routing ---
    section("(a) Confidence/risk-based routing")
    print(f"  {'id':<5}{'route':<14}{'model':<9}{'conf':<7}triggers")
    print("  " + "-" * 70)
    routed: list[Routed] = []
    review_seen = 0
    for item in items:
        dec = model_decide(client, item)
        r = route(item, dec, cfg, regulated)
        if r.route == "HUMAN_REVIEW":
            r.sla_hours = REVIEW_LATENCIES[review_seen % len(REVIEW_LATENCIES)]
            review_seen += 1
        routed.append(r)
        print(f"  {r.item_id:<5}{r.route:<14}{r.model_decision:<9}{r.confidence:<7}{', '.join(r.triggers) or '-'}")

    auto = [items[i] for i, r in enumerate(routed) if r.route == "AUTO_APPROVE"]
    review = [r for r in routed if r.route == "HUMAN_REVIEW"]
    rejected = [r for r in routed if r.route == "REJECT"]

    # --- (b) approval gates ---
    section("(b) Approval gates in an agent tool loop (scripted human)")
    scripted = [
        HumanDecision("edit", {"amount": 200}, note="capped at policy limit $200"),
        HumanDecision("approve", note="customer confirmation email ok"),
    ]
    for line in run_agent_with_gates(client, scripted, corrections):
        print("  " + line)

    # --- (c) QA sampling + reviewer agreement ---
    section("(c) QA sampling of auto-approved items (stratified) + agreement")
    sample = stratified_sample(auto, frac=0.6)
    agree = 0
    for it in sample:
        reviewer = qa_reviewer(it)
        auto_decision = "approve"  # auto-approved items were effectively approved
        match = reviewer == auto_decision
        agree += int(match)
        flag = "AGREE" if match else "DISAGREE -> correction"
        print(f"  {it['id']:<5} auto=approve  reviewer={reviewer:<8} {flag}")
        if not match:
            corrections.append({"source": "qa_sample", "item": it["id"], "auto": auto_decision, "reviewer": reviewer})
    agreement = agree / len(sample) if sample else 1.0
    print(f"  sampled {len(sample)} of {len(auto)} auto-approved; reviewer agreement = {agreement:.0%}")

    # --- (d) corrections dataset ---
    section("(d) Feedback capture -> corrections dataset")
    if not corrections:
        print("  (none)")
    for c in corrections:
        print("  " + json.dumps(c))
    print(f"  {len(corrections)} correction(s) captured for fine-tuning / prompt-improvement / policy review.")

    # --- (e) metrics ---
    section("(e) Operational metrics")
    total = len(routed)
    review_latencies = [r.sla_hours for r in review]
    within_sla = sum(1 for h in review_latencies if h <= cfg["review_sla_hours"])
    print(f"  total decisions .......... {total}")
    print(f"  automation rate .......... {len(auto) / total:.0%}  ({len(auto)} auto-approved)")
    print(f"  human review load ........ {len(review) / total:.0%}  ({len(review)} queued)")
    print(f"  reject rate .............. {len(rejected) / total:.0%}  ({len(rejected)} rejected)")
    if review_latencies:
        print(f"  avg escalation latency ... {sum(review_latencies) / len(review_latencies):.1f}h  (SLA {cfg['review_sla_hours']}h)")
        print(f"  SLA attainment ........... {within_sla / len(review_latencies):.0%}  ({within_sla}/{len(review_latencies)} within SLA)")

    section("Summary")
    print("  Routing automates the easy, grounded, low-value cases and escalates the rest.")
    print("  High-risk tool actions are gated; every human override becomes training signal.")
    print("  QA sampling keeps auto-approvals honest and measures reviewer agreement.")
    print("  See README for automation-bias and reviewer-fatigue failure modes.")


if __name__ == "__main__":
    main()
