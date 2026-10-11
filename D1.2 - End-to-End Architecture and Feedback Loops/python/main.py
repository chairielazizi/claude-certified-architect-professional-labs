"""
D1.2 - End-to-End Architecture and Feedback Loops

A support-ticket pipeline with explicit, typed stage contracts:

  INPUT       validate -> normalise -> PII scrub -> dedupe            (deterministic code)
  PROCESSING  Claude classify + extract (structured outputs)          (LLM, with retry policy)
  OUTPUT      schema + business validation -> route -> deliver        (deterministic code)
  FEEDBACK    human corrections -> metrics -> few-shot + eval set -> prompt v2 -> gated re-run

Every stage can fail independently; failures are retried when transient and parked in a
dead-letter queue (DLQ) when not, so one bad ticket never blocks the batch.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    LATENCY_SCALE,
    MODEL_HAIKU,
    MockRequest,
    banner,
    get_client,
    is_live,
    section,
    text_of,
)

DATA = Path(__file__).resolve().parents[1] / "data"

# ---------------------------------------------------------------------------
# Stage contracts. Each stage accepts exactly one type and emits exactly one type,
# so stages can be tested, scaled and replaced independently.
# ---------------------------------------------------------------------------
Category = Literal["billing", "access", "bug", "how_to", "privacy", "account", "sales"]
Priority = Literal["P1", "P2", "P3", "P4"]


@dataclass
class RawTicket:                      # INPUT contract (as received from email/chat/web)
    id: str
    channel: str
    customer_email: str
    subject: str
    body: str


@dataclass
class NormalizedTicket:               # INPUT -> PROCESSING contract
    id: str
    channel: str
    text: str                         # scrubbed: safe to send to the model
    pii_found: list[str]
    fingerprint: str


class Classification(BaseModel):      # PROCESSING -> OUTPUT contract (validated in code)
    category: Category
    priority: Priority
    product_area: str
    summary: str
    key_facts: list[str]
    # Numeric ranges are NOT enforceable by structured outputs, so the range check lives here.
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class RoutedTicket:                   # OUTPUT contract (what downstream systems receive)
    ticket_id: str
    classification: Classification
    queue: str
    sla_hours: int
    status: str


@dataclass
class DeadLetter:
    ticket_id: str
    stage: str
    reason: str
    attempts: int
    payload: object = None


# The JSON schema sent as output_config.format. Hand-written on purpose: it IS the contract.
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["billing", "access", "bug", "how_to", "privacy", "account", "sales"]},
        "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
        "product_area": {"type": "string"},
        "summary": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["category", "priority", "product_area", "summary", "key_facts", "confidence"],
    "additionalProperties": False,
}

ALLOWED_CHANNELS = {"email", "chat", "web"}
QUEUES = {"billing": "billing-ops", "access": "identity-team", "bug": "engineering-triage",
          "how_to": "tier1-support", "privacy": "privacy-office", "account": "tier1-support", "sales": "sales-desk"}
SLA_HOURS = {"P1": 1, "P2": 4, "P3": 24, "P4": 72}
CONFIDENCE_FLOOR = 0.70
MAX_ATTEMPTS = 3
INITIAL_MAX_TOKENS = 200   # sized from typical output length; outliers trigger an adaptive retry


# ---------------------------------------------------------------------------
# INPUT stage (deterministic)
# ---------------------------------------------------------------------------
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\+?\d{1,3}[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
QUOTED_REPLY_RE = re.compile(r"\n\s*On .{0,120}wrote:.*", re.DOTALL)


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def validate_raw(t: RawTicket) -> list[str]:
    errors = []
    if not t.body.strip():
        errors.append("empty body")
    if len(t.body) > 20_000:
        errors.append("body exceeds 20k chars")
    if not t.customer_email.strip():
        errors.append("missing customer_email (cannot reply)")
    if t.channel.strip().lower() not in ALLOWED_CHANNELS:
        errors.append(f"unknown channel {t.channel!r}")
    return errors


def scrub_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):   # Luhn avoids redacting order numbers
            found.append("CARD")
            return "[CARD]"
        return m.group(0)

    text = CARD_RE.sub(card, text)
    if EMAIL_RE.search(text):
        found.append("EMAIL")
        text = EMAIL_RE.sub("[EMAIL]", text)
    if PHONE_RE.search(text):
        found.append("PHONE")
        text = PHONE_RE.sub("[PHONE]", text)
    return text, found


def normalize(t: RawTicket) -> NormalizedTicket:
    body = QUOTED_REPLY_RE.sub("", t.body)                       # drop quoted email history
    body = "\n".join(" ".join(line.split()) for line in body.strip().splitlines())
    text, pii = scrub_pii(f"Subject: {t.subject.strip()}\n\n{body}")
    # Fingerprint on whitespace/case-insensitive text so re-sent tickets collapse to one.
    fingerprint = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:12]
    return NormalizedTicket(t.id, t.channel.strip().lower(), text, pii, fingerprint)


# ---------------------------------------------------------------------------
# Prompt registry - prompts are versioned artefacts, not string literals in code paths
# ---------------------------------------------------------------------------
BASE_SYSTEM = """You classify customer-support tickets for a B2B SaaS company.
Categories: billing (charges, invoices, refunds, payments), access (login, SSO, permissions),
bug (product defects, errors), how_to (usage questions), privacy (data-protection requests),
account (profile, users, plan settings without a charge problem), sales (pricing, discounts, quotes).
Priority: P1 = many users blocked or security incident; P2 = single customer blocked, money at risk or a
legal deadline; P3 = degraded but workable; P4 = question or request.
Return the JSON object only. confidence is your probability (0-1) that category AND priority are right."""


@dataclass
class PromptVersion:
    version: str
    system: str
    examples: list[dict] = field(default_factory=list)


def build_prompt(version: str, examples: list[dict]) -> PromptVersion:
    if not examples:
        return PromptVersion(version, BASE_SYSTEM)
    blocks = []
    for ex in examples:
        blocks.append(
            "<example>\n"
            f"<ticket>{ex['text']}</ticket>\n"
            f"<labels>{json.dumps(ex['labels'])}</labels>\n"
            f"<why>{ex['why']}</why>\n"
            "</example>"
        )
    system = BASE_SYSTEM + "\n\nLearn from these reviewed tickets:\n<examples>\n" + "\n".join(blocks) + "\n</examples>"
    return PromptVersion(version, system, examples)


# ---------------------------------------------------------------------------
# PROCESSING stage (Claude) with a retry policy and DLQ hand-off
# ---------------------------------------------------------------------------
def is_transient(err: Exception) -> bool:
    if isinstance(err, anthropic.APIConnectionError):             # includes APITimeoutError
        return True
    if isinstance(err, anthropic.APIStatusError):
        return err.status_code in (408, 409, 429) or err.status_code >= 500
    return False


@dataclass
class ProcessStats:
    retries: int = 0
    log: list[str] = field(default_factory=list)


def process(client: anthropic.Anthropic, t: NormalizedTicket, prompt: PromptVersion,
            stats: ProcessStats, verbose: bool = True) -> Classification | DeadLetter:
    max_tokens = INITIAL_MAX_TOKENS
    last_error = "unknown"

    def note(msg: str) -> None:
        if verbose:
            print(f"    {t.id} {msg}")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            stats.retries += 1
        try:
            msg = client.messages.create(
                model=MODEL_HAIKU,
                max_tokens=max_tokens,
                system=prompt.system,
                output_config={"effort": "low",
                               "format": {"type": "json_schema", "schema": CLASSIFICATION_SCHEMA}},
                messages=[{"role": "user", "content": f"Ticket {t.id} (channel: {t.channel})\n\n{t.text}"}],
            )
        except anthropic.APIError as err:
            if not is_transient(err):
                return DeadLetter(t.id, "processing", f"non-retryable {type(err).__name__}", attempt, t)
            last_error = type(err).__name__
            backoff_ms = 500 * 2 ** (attempt - 1)          # add jitter in production
            note(f"attempt {attempt}: {last_error} (transient) -> backoff {backoff_ms} ms")
            time.sleep(backoff_ms / 1000 * (1.0 if is_live() else LATENCY_SCALE * 0.1))
            continue
        # Always inspect stop_reason before trusting content.
        if msg.stop_reason == "max_tokens":
            last_error = f"truncated at max_tokens={max_tokens}"
            note(f"attempt {attempt}: stop_reason=max_tokens ({max_tokens}) -> retry with {max_tokens * 2}")
            max_tokens *= 2
            continue
        if msg.stop_reason == "refusal":
            return DeadLetter(t.id, "processing", "model refusal - needs human handling", attempt, t)
        try:
            return Classification.model_validate_json(text_of(msg))
        except ValidationError as err:
            last_error = f"contract violation: {err.errors()[0]['msg']}"
            note(f"attempt {attempt}: {last_error} -> retry")
    return DeadLetter(t.id, "processing", f"retries exhausted ({last_error})", MAX_ATTEMPTS, t)


# ---------------------------------------------------------------------------
# OUTPUT stage (deterministic): business validation -> routing -> idempotent delivery
# ---------------------------------------------------------------------------
class Outbox:
    def __init__(self) -> None:
        self.delivered: dict[str, RoutedTicket] = {}

    def deliver(self, r: RoutedTicket) -> str:
        if r.ticket_id in self.delivered:            # idempotency key = ticket id
            return "duplicate-suppressed"
        self.delivered[r.ticket_id] = r
        return "delivered"


def route(t: NormalizedTicket, c: Classification, outbox: Outbox) -> RoutedTicket | DeadLetter:
    leaks = [name for name, rx in (("EMAIL", EMAIL_RE), ("PHONE", PHONE_RE)) if rx.search(c.summary)]
    if leaks:
        return DeadLetter(t.id, "output", f"PII in model summary: {leaks}", 1, c)
    queue = "human-review" if c.confidence < CONFIDENCE_FLOOR else QUEUES[c.category]
    routed = RoutedTicket(t.id, c, queue, SLA_HOURS[c.priority], "pending")
    routed.status = outbox.deliver(routed)
    return routed


# ---------------------------------------------------------------------------
# FEEDBACK: corrections, metrics, promotion
# ---------------------------------------------------------------------------
@dataclass
class Correction:
    ticket_id: str
    field: str
    predicted: str
    corrected: str
    note: str


@dataclass
class Metrics:
    n: int
    category_acc: float
    priority_acc: float
    exact_match: float
    low_conf_rate: float
    conf_when_right: float
    conf_when_wrong: float


def compute_metrics(preds: dict[str, Classification], gold: dict[str, dict]) -> Metrics:
    ids = [i for i in preds if i in gold]
    cat = [preds[i].category == gold[i]["category"] for i in ids]
    pri = [preds[i].priority == gold[i]["priority"] for i in ids]
    exact = [a and b for a, b in zip(cat, pri)]
    right = [preds[i].confidence for i, ok in zip(ids, exact) if ok]
    wrong = [preds[i].confidence for i, ok in zip(ids, exact) if not ok]
    n = len(ids)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return Metrics(n, sum(cat) / n, sum(pri) / n, sum(exact) / n,
                   sum(preds[i].confidence < CONFIDENCE_FLOOR for i in ids) / n, avg(right), avg(wrong))


def print_metrics(label: str, m: Metrics) -> None:
    wrong = f"{m.conf_when_wrong:.2f}" if m.exact_match < 1 else "n/a"
    print(f"  {label:<4} n={m.n}  category={m.category_acc:6.1%}  priority={m.priority_acc:6.1%}  "
          f"exact={m.exact_match:6.1%}  low-conf={m.low_conf_rate:5.1%}  "
          f"conf(right)={m.conf_when_right:.2f}  conf(wrong)={wrong}")


# ---------------------------------------------------------------------------
# Offline responder - canned classifications, plus simulated failures
# ---------------------------------------------------------------------------
MOCK = json.loads((DATA / "mock_classifications.json").read_text(encoding="utf-8"))


class TicketResponder:
    """Stateful so failures are deterministic: T-1009 blips once; T-1010 hits a 3-attempt outage."""

    def __init__(self) -> None:
        self.attempts: dict[str, int] = defaultdict(int)

    def __call__(self, req: MockRequest) -> str:
        match = re.search(r"Ticket (T-\d+)", req.last_user_text)
        tid = match.group(1) if match else ""
        self.attempts[tid] += 1
        n = self.attempts[tid]
        if tid == "T-1009" and n == 1:
            raise ConnectionError("simulated network blip")      # SDK surfaces APIConnectionError
        if tid == "T-1010" and n <= 3:
            raise ConnectionError("simulated upstream outage")
        out = dict(MOCK["outputs"][tid])
        if "<examples>" not in req.system_text and tid in MOCK["v1_mistakes"]:
            out.update(MOCK["v1_mistakes"][tid])                 # zero-shot prompt's systematic errors
        return json.dumps(out)


def main() -> None:
    banner("D1.2 - End-to-End Architecture and Feedback Loops")
    client = get_client(TicketResponder())
    raw_rows = json.loads((DATA / "tickets.json").read_text(encoding="utf-8"))
    raw = [RawTicket(r["id"], r["channel"], r["customer_email"], r["subject"], r["body"]) for r in raw_rows]
    gold = {r["id"]: r["gold"] for r in raw_rows if r["gold"]}
    dlq: list[DeadLetter] = []

    section("0. Stage contracts")
    print("  RawTicket -> [INPUT] -> NormalizedTicket -> [PROCESSING] -> Classification -> [OUTPUT] -> RoutedTicket")
    print("  Any stage may emit DeadLetter(ticket_id, stage, reason, attempts) instead.")
    print(f"  Classification schema fields: {', '.join(CLASSIFICATION_SCHEMA['properties'])}")

    section("1. INPUT: validate -> normalise -> PII scrub -> dedupe")
    seen: dict[str, str] = {}
    normalized: list[NormalizedTicket] = []
    duplicates = 0
    for t in raw:
        errors = validate_raw(t)
        if errors:
            dlq.append(DeadLetter(t.id, "input", "; ".join(errors), 1, t))
            print(f"  {t.id} REJECTED -> DLQ ({'; '.join(errors)})")
            continue
        n = normalize(t)
        if n.fingerprint in seen:
            print(f"  {t.id} DUPLICATE of {seen[n.fingerprint]} (fingerprint {n.fingerprint}) -> dropped")
            duplicates += 1
            continue
        seen[n.fingerprint] = t.id
        normalized.append(n)
        pii = ",".join(n.pii_found) or "none"
        print(f"  {t.id} ok  channel={n.channel:<5} pii={pii:<6} fp={n.fingerprint}  \"{n.text.splitlines()[0][:44]}\"")
    print(f"  -> {len(normalized)} tickets to process; customer_email never leaves the input stage (data minimisation)")
    sample = next(line for n in normalized for line in n.text.splitlines() if "[CARD]" in line)
    print(f"  scrubbed line sent to Claude: \"{sample[:100]}\"")

    section("2. PROCESSING: Claude classify + extract (prompt v1, Haiku, structured outputs)")
    prompt_v1 = build_prompt("v1", [])
    stats = ProcessStats()
    preds: dict[str, Classification] = {}
    by_id = {n.id: n for n in normalized}
    for n in normalized:
        result = process(client, n, prompt_v1, stats)
        if isinstance(result, DeadLetter):
            dlq.append(result)
            print(f"  {n.id} -> DLQ ({result.reason})")
        else:
            preds[n.id] = result
            print(f"  {n.id} -> {result.category:<8} {result.priority}  conf={result.confidence:.2f}  {result.product_area}")
    print(f"  retries used: {stats.retries}  (SDK retries are disabled offline so the pipeline policy is visible)")

    section("3. OUTPUT: validate -> route -> deliver")
    outbox = Outbox()
    for tid, c in preds.items():
        r = route(by_id[tid], c, outbox)
        if isinstance(r, DeadLetter):
            dlq.append(r)
            print(f"  {tid} -> DLQ ({r.reason})")
            continue
        flag = "  <- below confidence floor" if r.queue == "human-review" else ""
        print(f"  {tid} {c.category:<8} {c.priority} -> queue={r.queue:<19} SLA={r.sla_hours:>2}h  {r.status}{flag}")
    again = route(by_id["T-1001"], preds["T-1001"], outbox)
    print(f"  re-delivery of T-1001 (e.g. after a worker crash) -> {getattr(again, 'status', '?')}")

    section("4. Dead-letter queue: park, inspect, replay")
    for d in dlq:
        print(f"  [{d.stage:<10}] {d.ticket_id}: {d.reason} (attempts={d.attempts})")
    print("  ...upstream dependency recovered; replaying processing-stage letters:")
    for d in [d for d in dlq if d.stage == "processing"]:
        result = process(client, by_id[d.ticket_id], prompt_v1, stats)
        if isinstance(result, Classification):
            dlq.remove(d)
            preds[d.ticket_id] = result
            r = route(by_id[d.ticket_id], result, outbox)
            print(f"  {d.ticket_id} replayed -> {result.category} {result.priority} -> "
                  f"queue={getattr(r, 'queue', '?')} {getattr(r, 'status', '')}")
    print(f"  DLQ now holds {len(dlq)} item(s): {', '.join(d.ticket_id for d in dlq)} "
          "(input errors need a human, not a retry)")

    section("5. FEEDBACK: human review captures corrections")
    corrections: list[Correction] = []
    for tid, c in preds.items():
        g = gold[tid]
        for fld in ("category", "priority"):
            if getattr(c, fld) != g[fld]:
                corrections.append(Correction(tid, fld, getattr(c, fld), g[fld], g.get("review_note", "")))
    for corr in corrections:
        print(f"  {corr.ticket_id} {corr.field}: {corr.predicted} -> {corr.corrected}   note: {corr.note}")
    print(f"  {len(corrections)} correction(s) on {len(preds)} reviewed tickets")

    section("6. Metrics for prompt v1 (production batch)")
    m_prod = compute_metrics(preds, gold)
    print_metrics("v1", m_prod)
    print(f"  human override rate = {1 - m_prod.exact_match:.1%}; DLQ rate = {len(dlq)}/{len(raw) - duplicates} "
          f"unique tickets; pipeline retries = {stats.retries}")
    print("  note: wrong answers still carried high self-reported confidence -> never gate on confidence alone")

    section("7. Promote corrections -> few-shot examples + eval set; bump prompt version")
    corrected_ids = sorted({c.ticket_id for c in corrections})
    examples = [{"text": by_id[i].text, "labels": {"category": gold[i]["category"], "priority": gold[i]["priority"]},
                 "why": gold[i].get("review_note", "")} for i in corrected_ids]
    eval_set = {tid: gold[tid] for tid in preds}          # every human-verified ticket becomes a regression case
    prompt_v2 = build_prompt("v2", examples)
    print(f"  few-shot examples added: {', '.join(corrected_ids)}")
    print(f"  eval set size: {len(eval_set)} human-labelled tickets")
    print(f"  prompt {prompt_v1.version} -> {prompt_v2.version}: system prompt {len(prompt_v1.system)} -> "
          f"{len(prompt_v2.system)} chars (examples block appended; base instructions unchanged)")

    section("8. Offline eval on the eval set: v1 vs v2 + promotion gate")
    results: dict[str, dict[str, Classification]] = {}
    for prompt in (prompt_v1, prompt_v2):
        quiet = ProcessStats()
        out: dict[str, Classification] = {}
        for tid in eval_set:
            r = process(client, by_id[tid], prompt, quiet, verbose=False)   # eval runs never deliver
            if isinstance(r, Classification):
                out[tid] = r
        results[prompt.version] = out
    m1 = compute_metrics(results["v1"], eval_set)
    m2 = compute_metrics(results["v2"], eval_set)
    print_metrics("v1", m1)
    print_metrics("v2", m2)
    regressions = [tid for tid in eval_set
                   if tid in results["v1"] and tid in results["v2"]
                   and (results["v1"][tid].category, results["v1"][tid].priority) == (eval_set[tid]["category"], eval_set[tid]["priority"])
                   and (results["v2"][tid].category, results["v2"][tid].priority) != (eval_set[tid]["category"], eval_set[tid]["priority"])]
    promote = m2.exact_match >= m1.exact_match and not regressions
    print(f"  gate: exact-match {m1.exact_match:.1%} -> {m2.exact_match:.1%}, regressions={', '.join(regressions) or 'none'}")
    print(f"  decision: {'PROMOTE v2 to production' if promote else 'KEEP v1 (gate failed)'}")

    section("Summary")
    print("  input -> processing -> output -> feedback is a LOOP: every human correction becomes")
    print("  a few-shot example and a regression test, and a new prompt version ships only through the eval gate.")


if __name__ == "__main__":
    main()
