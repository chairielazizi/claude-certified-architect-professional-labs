"""
D4.6 - Logging and Observability Tooling

Instruments a small tool-using Claude agent with:
  * structured JSON logs per LLM call and tool call (request_id, trace/span ids, model,
    prompt version, tokens incl. cache, cost, latency, stop_reason, tool names, error type)
    with PII redaction BEFORE anything is written
  * OpenTelemetry-style spans: agent.session -> agent.step -> llm.call / tool.call (JSON export)
  * metrics: counters, histograms (p50/p95 from buckets vs exact), cost/hour gauge, text dashboard
  * alert rules: error rate, p95 latency, cost/hour, refusal spike, truncation rate, cache hit ratio
  * typed SDK error handling and logging (429 + retry-after, 529 overloaded, connection errors)
  * the SDK's own debug logging switch (ANTHROPIC_LOG=debug)

Outputs go to <lab>/labs_output/ (gitignored).

Run from the repo root:
    .venv/Scripts/python "D4.6 - Logging and Observability Tooling/python/main.py"
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic
import httpx2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_SONNET,
    MockRequest,
    banner,
    cost_usd,
    elapsed_ms,
    estimate_tokens,
    get_client,
    is_live,
    section,
    text_block,
    tool_use_block,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "labs_output"
MODEL = MODEL_SONNET
MAX_ATTEMPTS = 3
MAX_STEPS = 4

# ---------------------------------------------------------------------------
# Deterministic helpers (same algorithms as the TypeScript version)
# ---------------------------------------------------------------------------
def fnv1a32(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def mulberry32(seed: int) -> Callable[[], float]:
    state = seed & 0xFFFFFFFF

    def rand() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rand


def fmt(x: float, d: int = 2) -> str:
    scaled = math.floor(x * 10**d + 0.5) / 10**d
    return f"{scaled:.{d}f}"


def pct(x: float) -> str:
    return fmt(100.0 * x, 1) + "%"


def total(xs: Any) -> float:
    s = 0.0
    for x in xs:
        s += x
    return s


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    rank = p / 100.0 * (len(s) - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 1. PII redaction - applied to every string BEFORE it reaches a log sink
# ---------------------------------------------------------------------------
REDACTIONS: list[tuple[str, re.Pattern[str]]] = [
    ("[EMAIL]", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[CARD]", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("[PHONE]", re.compile(r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b")),
]


def redact(text: str) -> str:
    for token, pattern in REDACTIONS:
        text = pattern.sub(token, text)
    return text


def preview(text: str, n: int = 80) -> str:
    t = redact(text.replace("\n", " "))
    return t if len(t) <= n else t[: n - 3] + "..."


# ---------------------------------------------------------------------------
# 2. Clock + ids (simulated offline so logs are reproducible; real time live)
# ---------------------------------------------------------------------------
class Clock:
    def __init__(self, start_iso: str) -> None:
        self.ms = int(dt.datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp() * 1000)

    def now(self) -> int:
        return int(time.time() * 1000) if is_live() else self.ms

    def advance(self, ms: float) -> None:
        if not is_live():
            self.ms += math.floor(ms + 0.5)

    def set(self, ms: int) -> None:
        if not is_live():
            self.ms = ms

    def sleep(self, seconds: float) -> None:
        if is_live():
            time.sleep(min(seconds, 2.0))  # keep the live demo short
        else:
            self.advance(seconds * 1000)


def iso(ms: int) -> str:
    d = dt.datetime.fromtimestamp(ms // 1000, tz=dt.timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms % 1000:03d}Z"


START_ISO = "2026-10-11T09:00:00Z"
CLOCK = Clock(START_ISO)
START_MS = CLOCK.ms
ID_RNG = mulberry32(20261011)


def hex_id(chunks: int) -> str:
    return "".join(f"{math.floor(ID_RNG() * 4294967296):08x}" for _ in range(chunks))


# ---------------------------------------------------------------------------
# 3. Minimal OpenTelemetry-style tracer
# ---------------------------------------------------------------------------
@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_ms: int
    end_ms: int = 0
    status: str = "UNSET"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def start(self, name: str, parent: Span | None, **attrs: Any) -> Span:
        span = Span(parent.trace_id if parent else hex_id(4), hex_id(2), parent.span_id if parent else None, name, CLOCK.now(), attributes=dict(attrs))
        self.spans.append(span)
        return span

    def end(self, span: Span, status: str = "OK", **attrs: Any) -> None:
        span.attributes.update(attrs)
        span.status = status
        span.end_ms = CLOCK.now()

    def export(self) -> list[dict[str, Any]]:
        return [{"trace_id": s.trace_id, "span_id": s.span_id, "parent_span_id": s.parent_span_id, "name": s.name,
                 "start_time": iso(s.start_ms), "end_time": iso(s.end_ms), "duration_ms": s.duration_ms,
                 "status": s.status, "attributes": s.attributes} for s in self.spans]


# ---------------------------------------------------------------------------
# 4. Structured logger + metrics registry
# ---------------------------------------------------------------------------
class JsonLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log(self, level: str, event: str, span: Span | None, **fields: Any) -> dict[str, Any]:
        rec: dict[str, Any] = {"ts": iso(CLOCK.now()), "level": level, "event": event,
                               "trace_id": span.trace_id if span else None, "span_id": span.span_id if span else None}
        for k, v in fields.items():
            rec[k] = redact(v) if isinstance(v, str) else v  # defence in depth: redact every string field
        self.records.append(rec)
        return rec


BUCKETS = [300.0, 600.0, 900.0, 1200.0, 2400.0, 4800.0, math.inf]


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.hist: dict[str, list[float]] = {}
        self.sums: dict[str, float] = {}

    def inc(self, name: str, n: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + n

    def observe(self, name: str, value: float) -> None:
        self.hist.setdefault(name, []).append(value)

    def add(self, name: str, value: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + value

    def c(self, name: str) -> int:
        return self.counters.get(name, 0)

    def bucket_counts(self, name: str) -> list[int]:
        vals = self.hist.get(name, [])
        return [sum(1 for v in vals if v <= le) for le in BUCKETS]  # cumulative, Prometheus-style

    def histogram_quantile(self, name: str, q: float) -> float:
        """Estimate a quantile from cumulative buckets with linear interpolation (like PromQL histogram_quantile)."""
        cum = self.bucket_counts(name)
        n = cum[-1]
        if n == 0:
            return 0.0
        rank = q * n
        prev_le, prev_c = 0.0, 0
        for le, cnt in zip(BUCKETS, cum):
            if cnt >= rank:
                if math.isinf(le):
                    return prev_le
                return prev_le + (le - prev_le) * ((rank - prev_c) / (cnt - prev_c) if cnt > prev_c else 0.0)
            prev_le, prev_c = le, cnt
        return prev_le


LOG = JsonLogger()
TRACER = Tracer()
METRICS = Metrics()

# ---------------------------------------------------------------------------
# 5. The agent under observation: tools, sessions, offline responder
# ---------------------------------------------------------------------------
ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {"status": "in_transit", "eta": "2026-10-14"},
    "ORD-1002": {"status": "refunded", "refund_amount": 49.9, "refund_date": "2026-10-09"},
    "ORD-1003": {"status": "in_transit", "eta": "2026-10-16"},
    "ORD-1004": {"status": "delivered", "delivered": "2026-10-01", "return_window_until": "2026-10-31"},
    "ORD-1005": {"status": "processing"},
    "ORD-1008": {"status": "in_transit", "eta": "2026-10-15"},
    "ORD-1009": {"status": "in_transit", "eta": "2026-10-13"},
}
CUSTOMERS = {"jane.doe@example.com": {"name": "Jane Doe", "email": "jane.doe@example.com", "phone": "555-123-4567",
                                      "address": "12 Elm Street", "tier": "gold"}}
TOOLS = [
    {"name": "get_order", "description": "Look up an order's status by order id.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "get_customer", "description": "Look up a customer profile by email address.",
     "input_schema": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}},
]
TOOL_LATENCY = {"get_order": 120.0, "get_customer": 80.0}
PROMPT_VERSION = "support-agent@v7"
SYSTEM_TEXT = ("You are Acme's order-support agent. Use get_order for order questions and get_customer only when the user "
               "gives an email address. Answer in one or two sentences. Never repeat personal data back to the user. ") + " ".join(
    f"Guideline {i + 1}: follow the escalation matrix for case type {i + 1}, confirm identity before account changes, "
    f"and log the outcome in the case system." for i in range(40))
SYSTEM = [{"type": "text", "text": SYSTEM_TEXT, "cache_control": {"type": "ephemeral"}}]
PROMPT_HASH = hashlib.sha256(SYSTEM_TEXT.encode()).hexdigest()[:8]


@dataclass
class Session:
    sid: str
    user: str
    text: str
    max_tokens: int = 1024


SESSIONS = [
    Session("S1", "u1@example.com", "Where is my order ORD-1001?"),
    Session("S2", "jane.doe@example.com", "Hi, I'm Jane Doe (jane.doe@example.com, 555-123-4567). Has my refund for ORD-1002 been processed?"),
    Session("S3", "u3@example.com", "Can you check the delivery date for ORD-1003?"),
    Session("S4", "u4@example.com", "Is ORD-1004 still eligible for a return?"),
    Session("S5", "u5@example.com", "Track ORD-1005 please."),
    Session("S6", "u6@example.com", "[RED-TEAM PROMPT agent-01 | policy area: credential theft | full text held in the red-team store]"),
    Session("S7", "u7@example.com", "Summarize every interaction on my account for ORD-1007 in full detail.", max_tokens=60),  # misconfigured route
    Session("S8", "u8@example.com", "Which arrives first, ORD-1008 or ORD-1009?"),
]


def run_tool(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    if name == "get_order":
        rec = ORDERS.get(str(args.get("order_id")))
        return (compact({"order_id": args.get("order_id"), **rec}), True) if rec else (compact({"error": "order not found"}), False)
    if name == "get_customer":
        cust = CUSTOMERS.get(str(args.get("email")))
        return (compact(cust), True) if cust else (compact({"error": "customer not found"}), False)
    return compact({"error": f"unknown tool {name}"}), False


def final_answer(results: list[dict[str, Any]]) -> str:
    data = [json.loads(r["content"]) for r in results]
    orders = [d for d in data if "order_id" in d]
    if len(orders) == 2 and all("eta" in o for o in orders):
        first = min(orders, key=lambda o: o["eta"])
        return f"{first['order_id']} arrives first, on {first['eta']}."
    o = orders[0]
    if o.get("status") == "refunded":
        return f"Yes - your refund of {fmt(o['refund_amount'], 2)} dollars for {o['order_id']} was processed on {o['refund_date']}."
    if o.get("status") == "delivered":
        return f"Yes - {o['order_id']} was delivered on {o['delivered']} and can be returned until {o['return_window_until']}."
    return f"{o['order_id']} is in transit and should arrive on {o['eta']}."


def responder(req: MockRequest) -> Any:
    if req.tool_results:
        text = final_answer(req.tool_results)
        return {"content": [text_block(text)], "usage": {"output_tokens": estimate_tokens(text)}}
    user = req.last_user_text
    if "RED-TEAM" in user:
        return {"content": [], "stop_reason": "refusal", "usage": {"output_tokens": 0},
                "stop_details": {"type": "refusal", "category": "cyber", "explanation": "The request could enable cyber harm."}}
    if "Summarize every interaction" in user:
        return "Here is a complete history of your account. " + " ".join(
            f"On day {d} you contacted support about delivery, billing and settings, and the agent resolved the case." for d in range(1, 30))
    if "Track ORD-1005" in user:
        raise RuntimeError("connection reset by peer")  # surfaces in the SDK as APIConnectionError
    blocks: list[dict[str, Any]] = []
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user)
    if email:
        blocks.append(tool_use_block("get_customer", {"email": email.group(0)}, id="toolu_customer"))
    for oid in re.findall(r"ORD-\d{4}", user):
        blocks.append(tool_use_block("get_order", {"order_id": oid}, id=f"toolu_{oid.replace('-', '')}"))
    out_tokens = sum(estimate_tokens(b["name"] + compact(b["input"])) + 10 for b in blocks)
    return {"content": blocks, "usage": {"output_tokens": out_tokens}}


# ---------------------------------------------------------------------------
# 6. Fault injection with REAL SDK exception types (offline demo of error paths)
# ---------------------------------------------------------------------------
def status_error(cls: type[anthropic.APIStatusError], status: int, err_type: str, retry_after: str | None, req_id: str) -> anthropic.APIStatusError:
    headers = {"request-id": req_id}
    if retry_after:
        headers["retry-after"] = retry_after
    body = {"type": "error", "error": {"type": err_type, "message": f"injected {err_type}"}}
    response = httpx2.Response(status, headers=headers, json=body, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    return cls(f"injected {err_type}", response=response, body=body)


FAULTS: dict[tuple[str, int, int], Callable[[], Exception]] = {
    ("S3", 1, 0): lambda: status_error(anthropic.RateLimitError, 429, "rate_limit_error", "2", "req_fault_s3_0"),
    ("S4", 1, 0): lambda: status_error(anthropic.OverloadedError, 529, "overloaded_error", None, "req_fault_s4_0"),
    ("S4", 1, 1): lambda: status_error(anthropic.OverloadedError, 529, "overloaded_error", None, "req_fault_s4_1"),
}
FAILED_ATTEMPT_MS = {"rate_limit_error": 40.0, "overloaded_error": 300.0, "connection_error": 1500.0, "api_error": 200.0}
LAT_PROFILE = (500.0, 7.0)  # Sonnet: TTFT ms, ms per output token


def llm_call(client: Any, sess: Session, step: int, messages: list[dict[str, Any]], parent: Span, session_id: str) -> Any:
    """One logical LLM call: attempts with typed error handling, backoff, one log record + span per attempt."""
    for attempt in range(MAX_ATTEMPTS):
        span = TRACER.start("llm.call", parent, **{"gen_ai.system": "anthropic", "gen_ai.request.model": MODEL, "attempt": attempt})
        base = {"session_id": session_id, "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH, "model": MODEL,
                "step": step, "attempt": attempt}
        try:
            fault = FAULTS.get((sess.sid, step, attempt))
            if fault is not None:
                raise fault()
            t0 = time.perf_counter()
            msg = client.messages.create(model=MODEL, max_tokens=sess.max_tokens, system=SYSTEM, tools=TOOLS, messages=messages)
            wall = elapsed_ms(t0)
            rng = mulberry32(fnv1a32(f"{sess.sid}:{step}:{attempt}"))
            latency = wall if is_live() else (LAT_PROFILE[0] * (0.85 + 0.3 * rng()) + LAT_PROFILE[1] * msg.usage.output_tokens) * (2.5 if rng() < 0.15 else 1.0)  # seeded slow tail
            CLOCK.advance(latency)
            u = msg.usage
            cost = cost_usd(msg.model, u)
            tool_names = [b.name for b in msg.content if b.type == "tool_use"]
            LOG.log("WARN" if msg.stop_reason in ("max_tokens", "refusal") else "INFO", "llm.call", span, **base,
                    request_id=msg._request_id, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                    cache_read_tokens=u.cache_read_input_tokens or 0, cache_write_tokens=u.cache_creation_input_tokens or 0,
                    cost_usd=math.floor(cost * 1e8 + 0.5) / 1e8, latency_ms=math.floor(latency + 0.5), stop_reason=msg.stop_reason,
                    tool_names=tool_names, error_type=None, user_message_preview=preview(sess.text))
            METRICS.inc("llm_requests_total{outcome=ok}")
            METRICS.inc(f"llm_stop_reason_total{{reason={msg.stop_reason}}}")
            METRICS.observe("llm_latency_ms", latency)
            for key, val in (("tokens_input", u.input_tokens), ("tokens_output", u.output_tokens), ("cost_usd", cost),
                             ("tokens_cache_read", u.cache_read_input_tokens or 0), ("tokens_cache_write", u.cache_creation_input_tokens or 0)):
                METRICS.add(key, val)
            TRACER.end(span, "OK", **{"gen_ai.response.id": msg.id, "gen_ai.usage.input_tokens": u.input_tokens,
                                      "gen_ai.usage.output_tokens": u.output_tokens, "gen_ai.response.finish_reasons": [msg.stop_reason]})
            return msg
        except anthropic.RateLimitError as e:          # 429: honour retry-after
            err, status, retry_after, req_id, retryable = "rate_limit_error", 429, float(e.response.headers.get("retry-after", "1")), e.request_id, True
        except anthropic.APIStatusError as e:          # other HTTP errors; 5xx and 529 overloaded are retryable
            err, status, retry_after, req_id, retryable = (e.type or "api_error"), e.status_code, None, e.request_id, e.status_code >= 500
        except anthropic.APIConnectionError:           # network failure before any response
            err, status, retry_after, req_id, retryable = "connection_error", None, None, None, True
        CLOCK.advance(FAILED_ATTEMPT_MS.get(err, 200.0))
        last = attempt == MAX_ATTEMPTS - 1 or not retryable
        rng = mulberry32(fnv1a32(f"backoff:{sess.sid}:{step}:{attempt}"))
        delay = retry_after if retry_after is not None else min(8.0, 1.0 * 2**attempt) * (1 + 0.25 * rng())
        LOG.log("ERROR", "llm.call", span, **base, request_id=req_id, error_type=err, status_code=status,
                retry_after_s=retry_after, retryable=retryable, next_retry_in_s=None if last else math.floor(delay * 1000 + 0.5) / 1000,
                user_message_preview=preview(sess.text))
        METRICS.inc(f"llm_requests_total{{outcome=error,error_type={err}}}")
        TRACER.end(span, "ERROR", **{"error.type": err, "http.status_code": status})
        if last:
            return None
        CLOCK.sleep(delay)
    return None


def run_session(client: Any, sess: Session) -> dict[str, Any]:
    session_id = hashlib.sha256(f"acme-salt:{sess.user}".encode()).hexdigest()[:12]  # pseudonymous, never the raw email
    root = TRACER.start("agent.session", None, **{"session.id": session_id, "prompt.version": PROMPT_VERSION})
    messages: list[dict[str, Any]] = [{"role": "user", "content": sess.text}]
    status, stop = "OK", ""
    for step in range(1, MAX_STEPS + 1):
        step_span = TRACER.start("agent.step", root, step=step)
        msg = llm_call(client, sess, step, messages, step_span, session_id)
        if msg is None:
            TRACER.end(step_span, "ERROR")
            status, stop = "ERROR", "error"
            break
        stop = msg.stop_reason or ""
        if msg.stop_reason != "tool_use":
            TRACER.end(step_span, "OK")
            if stop in ("max_tokens", "refusal"):
                status = "DEGRADED"
            break
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for block in (b for b in msg.content if b.type == "tool_use"):
            tspan = TRACER.start("tool.call", step_span, **{"tool.name": block.name, "tool.args": redact(compact(block.input))})
            result, ok = run_tool(block.name, block.input)
            rng = mulberry32(fnv1a32(f"tool:{sess.sid}:{block.id}"))
            tlat = TOOL_LATENCY[block.name] * (0.8 + 0.4 * rng())
            CLOCK.advance(tlat)
            LOG.log("INFO" if ok else "WARN", "tool.call", tspan, session_id=session_id, tool_name=block.name,
                    tool_args=compact(block.input), result_preview=preview(result, 60), latency_ms=math.floor(tlat + 0.5), ok=ok)
            METRICS.inc(f"tool_calls_total{{tool={block.name},ok={str(ok).lower()}}}")
            METRICS.observe(f"tool_latency_ms{{tool={block.name}}}", tlat)
            TRACER.end(tspan, "OK" if ok else "ERROR")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": results})
        TRACER.end(step_span, "OK")
    TRACER.end(root, "ERROR" if status == "ERROR" else "OK", **{"session.status": status})
    METRICS.inc(f"sessions_total{{status={status}}}")
    return {"sid": sess.sid, "status": status, "stop": stop, "root": root}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def render(rec: dict[str, Any]) -> str:
    """Stable, language-neutral rendering of a log record for the console."""
    parts = []
    for k, v in rec.items():
        if isinstance(v, bool) or v is None:
            s = "null" if v is None else str(v).lower()
        elif isinstance(v, float):
            s = str(int(v)) if v.is_integer() else fmt(v, 6)  # same rendering as JS numbers
        elif isinstance(v, int):
            s = str(v)
        elif isinstance(v, list):
            s = "[" + ",".join(f'"{x}"' for x in v) + "]"
        else:
            s = f'"{v}"'
        parts.append(f'"{k}":{s}')
    return "{" + ", ".join(parts) + "}"


def main() -> None:
    banner("D4.6 - Logging and Observability Tooling")
    print(f"Agent: order support on {MODEL}, prompt {PROMPT_VERSION} (hash {PROMPT_HASH}), {len(SESSIONS)} sessions, tools: get_order, get_customer")

    section("1. PII redaction before logging")
    sample = "I'm jane.doe@example.com, call 555-123-4567, card 4111 1111 1111 1111, SSN 123-45-6789"
    print(f"  raw     : {sample}")
    print(f"  redacted: {redact(sample)}")
    print("  Redact at the logger (every string field), not at each call site - one forgotten call site leaks PII.")

    section("2. Run the instrumented agent")
    # SDK auto-retries are disabled so every attempt is visible to OUR handler and logged with its error type.
    client = get_client(responder).with_options(max_retries=0)
    outcomes = []
    for i, sess in enumerate(SESSIONS):
        CLOCK.set(START_MS + i * 300_000)  # simulated traffic: a session every 5 minutes (no-op live)
        outcomes.append(run_session(client, sess))
    print(f"  {'session':<9}{'status':<10}{'final stop':<12}{'llm attempts':>13}{'tool calls':>11}{'duration ms':>13}")
    for o in outcomes:
        tid = o["root"].trace_id
        llm = sum(1 for s in TRACER.spans if s.trace_id == tid and s.name == "llm.call")
        tools = sum(1 for s in TRACER.spans if s.trace_id == tid and s.name == "tool.call")
        print(f"  {o['sid']:<9}{o['status']:<10}{o['stop']:<12}{llm:>13}{tools:>11}{o['root'].duration_ms:>13}")

    section("3. Structured JSON log records (one per LLM / tool call)")
    ok_rec = next(r for r in LOG.records if r["event"] == "llm.call" and r.get("tool_names") and r["level"] == "INFO" and r["session_id"] and "[EMAIL]" in r["user_message_preview"])
    shown = dict(ok_rec, request_id=str(ok_rec["request_id"])[:12] + "...")
    print("  " + render(shown))
    err_rec = next(r for r in LOG.records if r.get("error_type") == "rate_limit_error")
    print("  " + render(dict(err_rec, request_id=str(err_rec["request_id"]))))
    print("  Fields: who (pseudonymous session_id), what (model, prompt_version/hash), cost (tokens incl. cache, $),")
    print("  speed (latency_ms), outcome (stop_reason, error_type), correlation (trace_id/span_id, request_id for support).")

    section("4. Trace for session S2 (agent.session -> agent.step -> llm.call / tool.call)")
    s2 = next(o for o in outcomes if o["sid"] == "S2")["root"]
    depth = {s2.span_id: 0}
    for s in TRACER.spans:
        if s.trace_id != s2.trace_id:
            continue
        d = depth[s.parent_span_id] + 1 if s.parent_span_id else 0
        depth[s.span_id] = d
        attrs = ""
        if s.name == "llm.call":
            attrs = f" in={s.attributes.get('gen_ai.usage.input_tokens')} out={s.attributes.get('gen_ai.usage.output_tokens')} finish={s.attributes.get('gen_ai.response.finish_reasons', ['-'])[0]}"
        elif s.name == "tool.call":
            attrs = f" {s.attributes['tool.name']} args={s.attributes['tool.args']}"
        elif s.name == "agent.step":
            attrs = f" #{s.attributes['step']}"
        print(f"  {'  ' * d}{s.name}{attrs}  [{s.duration_ms} ms, {s.status}]  span={s.span_id}")
    print(f"  trace_id={s2.trace_id} - the same id is on every log record of the session, so logs and traces join.")

    section("5. Metrics dashboard")
    ok_calls = METRICS.c("llm_requests_total{outcome=ok}")
    err_counts = {k.split("error_type=")[1].rstrip("}"): v for k, v in METRICS.counters.items() if "outcome=error" in k}
    attempts = ok_calls + sum(err_counts.values())
    sess_ok = METRICS.c("sessions_total{status=OK}")
    sess_deg = METRICS.c("sessions_total{status=DEGRADED}")
    sess_err = METRICS.c("sessions_total{status=ERROR}")
    lat = METRICS.hist["llm_latency_ms"]
    stops = {k.split("reason=")[1].rstrip("}"): v for k, v in METRICS.counters.items() if k.startswith("llm_stop_reason_total")}
    window_h = (CLOCK.now() - START_MS) / 3_600_000 if not is_live() else max(1e-6, (TRACER.spans[-1].end_ms - TRACER.spans[0].start_ms) / 3_600_000)
    cost = METRICS.sums["cost_usd"]
    cache_ratio = METRICS.sums["tokens_cache_read"] / (METRICS.sums["tokens_cache_read"] + METRICS.sums["tokens_input"])
    print(f"  LLM attempts   : {attempts} total, {ok_calls} ok, {sum(err_counts.values())} errors ({pct(sum(err_counts.values()) / attempts)}) "
          + " ".join(f"{k}={v}" for k, v in sorted(err_counts.items())))
    print(f"  sessions       : OK={sess_ok} DEGRADED={sess_deg} ERROR={sess_err}")
    print(f"  stop reasons   : " + " ".join(f"{k}={v}" for k, v in sorted(stops.items())))
    print(f"  LLM latency ms : exact p50={fmt(percentile(lat, 50), 0)} p95={fmt(percentile(lat, 95), 0)} | from buckets p50~{fmt(METRICS.histogram_quantile('llm_latency_ms', 0.5), 0)} "
          f"p95~{fmt(METRICS.histogram_quantile('llm_latency_ms', 0.95), 0)} (bucket estimates are coarse)")
    cum = METRICS.bucket_counts("llm_latency_ms")
    prev = 0
    for le, cnt in zip(BUCKETS, cum):
        label = "+Inf" if math.isinf(le) else fmt(le, 0)
        print(f"    le {label:>5} | {'#' * (cnt - prev):<12} {cnt - prev}")
        prev = cnt
    print(f"  tokens         : input={int(METRICS.sums['tokens_input'])} output={int(METRICS.sums['tokens_output'])} "
          f"cache_read={int(METRICS.sums['tokens_cache_read'])} cache_write={int(METRICS.sums['tokens_cache_write'])} (cache hit ratio {pct(cache_ratio)})")
    print(f"  cost           : ${fmt(cost, 6)} over {fmt(window_h * 60, 1)} min -> ${fmt(cost / window_h, 4)}/hour")
    tool_names = sorted({k.split("tool=")[1].split(",")[0] for k in METRICS.counters if k.startswith("tool_calls_total")})
    print("  tools          : " + "  ".join(
        f"{t} calls={sum(v for k, v in METRICS.counters.items() if k.startswith(f'tool_calls_total{{tool={t},'))} p95={fmt(percentile(METRICS.hist[f'tool_latency_ms{{tool={t}}}'], 95), 0)}ms"
        for t in tool_names))

    section("6. Alert rules")
    refusals, truncations = stops.get("refusal", 0), stops.get("max_tokens", 0)
    rules = [
        ("LLM error rate (attempts)", sum(err_counts.values()) / attempts, 0.05, ">", "page", pct),
        ("session failure rate", sess_err / len(SESSIONS), 0.05, ">", "page", pct),
        ("LLM p95 latency (ms)", percentile(lat, 95), 2000.0, ">", "ticket", lambda v: fmt(v, 0)),
        ("cost per hour (USD)", cost / window_h, 5.0, ">", "ticket", lambda v: "$" + fmt(v, 4)),
        ("refusal rate", refusals / ok_calls, 0.05, ">", "ticket", pct),
        ("max_tokens truncation rate", truncations / ok_calls, 0.02, ">", "ticket", pct),
        ("cache hit ratio", cache_ratio, 0.50, "<", "ticket", pct),
    ]
    print(f"  {'rule':<30}{'value':>10}  {'threshold':<12}{'severity':<9}state")
    for name, value, thr, op, sev, f in rules:
        firing = value > thr if op == ">" else value < thr
        print(f"  {name:<30}{f(value):>10}  {op + ' ' + f(thr):<12}{sev:<9}{'FIRING' if firing else 'ok'}")
    print("  Alert on symptoms users feel (errors, latency, truncation) and on silent cost regressions (cache hit ratio).")

    section("7. Typed error handling (what each error means operationally)")
    for rec in (r for r in LOG.records if r.get("error_type")):
        retry = f"retry in {fmt(rec['next_retry_in_s'], 2)} s" if rec.get("next_retry_in_s") is not None else "give up"
        print(f"  {rec['ts']} session={rec['session_id']} step={rec['step']} attempt={rec['attempt']} "
              f"{rec['error_type']:<17} status={rec['status_code'] if rec['status_code'] is not None else '-':<4} "
              f"request_id={rec['request_id'] or '-':<15} -> {retry}")
    print("  429 rate_limit_error: honour retry-after | 529 overloaded / 5xx: exponential backoff + jitter |")
    print("  connection errors: retry, then fail the session cleanly | 4xx (400/401/403/404): never retry - fix the request.")

    section("8. SDK debug logging (ANTHROPIC_LOG=debug)")
    captured: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    sdk_logger = logging.getLogger("anthropic")
    handler, old_level = Capture(), sdk_logger.level
    sdk_logger.addHandler(handler)
    sdk_logger.setLevel(logging.DEBUG)  # same effect as starting the process with ANTHROPIC_LOG=debug
    debug_client = get_client(responder)
    debug_client.messages.create(model=MODEL, max_tokens=64, messages=[{"role": "user", "content": "Where is my order ORD-1001?"}])
    sdk_logger.removeHandler(handler)
    sdk_logger.setLevel(old_level)
    print(f"  SDK debug lines captured for one request: {'yes' if captured else 'no'}")
    print("  Enable with ANTHROPIC_LOG=debug (or info) for local debugging; debug output includes request details and can")
    print("  contain prompt text - keep it off in production or route it through the same redaction pipeline.")

    section("9. Export")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "llm_calls.jsonl").open("w", encoding="utf-8") as fh:
        for rec in LOG.records:
            fh.write(json.dumps(rec) + "\n")
    (OUT_DIR / "spans.json").write_text(json.dumps(TRACER.export(), indent=2), encoding="utf-8")
    (OUT_DIR / "metrics.json").write_text(json.dumps({"counters": METRICS.counters, "sums": METRICS.sums,
                                                      "histograms": {k: {"buckets_le": ["+Inf" if math.isinf(b) else fmt(b, 0) for b in BUCKETS], "cumulative": METRICS.bucket_counts(k)} for k in METRICS.hist}},
                                                     indent=2), encoding="utf-8")
    leaked = [r for r in LOG.records if any(isinstance(v, str) and ("jane.doe@example.com" in v or "555-123-4567" in v) for v in r.values())]
    print(f"  labs_output/llm_calls.jsonl : {len(LOG.records)} records "
          f"({sum(1 for r in LOG.records if r['event'] == 'llm.call')} llm.call, {sum(1 for r in LOG.records if r['event'] == 'tool.call')} tool.call)")
    print(f"  labs_output/spans.json      : {len(TRACER.spans)} spans in {len({s.trace_id for s in TRACER.spans})} traces")
    print(f"  labs_output/metrics.json    : {len(METRICS.counters)} counters, {len(METRICS.hist)} histograms")
    print(f"  PII check on written logs   : {len(leaked)} records contain raw email/phone")
    print("  In production, ship these to your log store / OTel collector; this lab is the instrumentation, D3.4 covers")
    print("  the at-scale strategy (sampling, cardinality, retention, centralised dashboards).")


if __name__ == "__main__":
    main()
