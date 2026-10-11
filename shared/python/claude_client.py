"""
Shared Claude client used by every lab in this repository.

Why this module exists
----------------------
Every lab talks to Claude through the *real* Anthropic SDK (`anthropic.Anthropic`)
with real Messages API request shapes. When no credentials are configured, the
SDK is wired to an in-process mock HTTP transport that answers in the Messages
API wire format. That means:

  * every demo runs offline, deterministically, and for free;
  * the code you read is exactly the code that runs against the live API;
  * set ANTHROPIC_API_KEY (or LABS_MODE=live) and the same lab goes live.

Mode selection (env var LABS_MODE):
    live     -> always call the real API
    offline  -> always use the mock transport
    (unset)  -> live if ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is set, else offline

Offline simulation features (so architecture trade-offs are observable):
    * token usage estimated at ~4 characters per token
    * prompt-caching simulation (cache writes, then cache reads on identical prefixes)
    * per-model simulated latency (Haiku < Sonnet < Opus < Fable), scaled down
    * `max_tokens` truncation -> stop_reason "max_tokens"
    * streaming (SSE) for text and tool_use blocks
    * /v1/messages/count_tokens

Usage from a lab (folder layout: <root>/<lab>/python/main.py):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
    from claude_client import get_client, MODEL_OPUS, text_of

    def responder(req):            # only used offline
        return "canned answer for: " + req.last_user_text

    client = get_client(responder)
    msg = client.messages.create(model=MODEL_OPUS, max_tokens=1024,
                                 messages=[{"role": "user", "content": "Hi"}])
    print(text_of(msg))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import anthropic
import httpx2

# ---------------------------------------------------------------------------
# Models & pricing (cached 2026-10; always confirm on the live pricing page)
# ---------------------------------------------------------------------------
MODEL_FABLE = "claude-fable-5-1"    # most capable widely released model
MODEL_OPUS = "claude-opus-5-5"      # default: frontier intelligence, agents, coding
MODEL_SONNET = "claude-sonnet-5-5"  # balanced speed / capability, high volume
MODEL_HAIKU = "claude-haiku-5-5"    # fastest / cheapest: routing, classification
DEFAULT_MODEL = MODEL_OPUS

# USD per 1M tokens. cache_write = 5-minute TTL write (~1.25x input).
PRICING: dict[str, dict[str, float]] = {
    MODEL_FABLE: {"input": 10.00, "output": 50.00, "cache_write": 12.50, "cache_read": 0.25},
    MODEL_OPUS: {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    MODEL_SONNET: {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    # Haiku 5.5 cache pricing assumed at the usual ratios (1.25x write, 0.1x read).
    MODEL_HAIKU: {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}

# Simulated latency profile for offline mode: time-to-first-token + per output token.
_LATENCY_PROFILE_MS: dict[str, tuple[float, float]] = {
    MODEL_HAIKU: (250.0, 3.0),
    MODEL_SONNET: (500.0, 7.0),
    MODEL_OPUS: (900.0, 12.0),
    MODEL_FABLE: (1500.0, 18.0),
}
# Offline sleeps are scaled down so demos stay fast; elapsed_ms() scales them back up.
LATENCY_SCALE = float(os.environ.get("LABS_LATENCY_SCALE", "0.1"))
# Simplified minimum cacheable prefix for the offline cache simulation.
MIN_CACHEABLE_TOKENS = 1024


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------
def is_live() -> bool:
    mode = os.environ.get("LABS_MODE", "").strip().lower()
    if mode == "live":
        return True
    if mode == "offline":
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def mode_label() -> str:
    return "LIVE (real Claude API)" if is_live() else "OFFLINE (mock transport - set ANTHROPIC_API_KEY to go live)"


def banner(title: str) -> None:
    line = "=" * max(60, len(title) + 4)
    print(line)
    print(f"  {title}")
    print(f"  mode: {mode_label()}")
    print(line)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(3, 56 - len(title)))


def elapsed_ms(t0: float) -> float:
    """Milliseconds since t0 (from time.perf_counter()).

    Offline, the mock sleeps a scaled-down simulated latency; this scales it back
    up so printed latencies resemble real-world magnitudes.
    """
    wall = (time.perf_counter() - t0) * 1000.0
    return wall if is_live() else wall / LATENCY_SCALE


# ---------------------------------------------------------------------------
# Token / cost helpers
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Rough heuristic (~4 chars/token). Use client.messages.count_tokens for real counts."""
    return max(1, len(text) // 4) if text else 0


def cost_usd(model: str, usage: Any) -> float:
    """Dollar cost of one response from its `usage` object (SDK Usage or dict)."""
    p = PRICING.get(model, PRICING[DEFAULT_MODEL])
    get = (lambda k: usage.get(k) or 0) if isinstance(usage, dict) else (lambda k: getattr(usage, k, 0) or 0)
    return (
        get("input_tokens") * p["input"]
        + get("output_tokens") * p["output"]
        + get("cache_creation_input_tokens") * p["cache_write"]
        + get("cache_read_input_tokens") * p["cache_read"]
    ) / 1_000_000


def text_of(message: Any) -> str:
    """Concatenate all text blocks of a Message."""
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


def usage_line(message: Any) -> str:
    u = message.usage
    return (
        f"in={u.input_tokens} out={u.output_tokens} "
        f"cache_write={u.cache_creation_input_tokens or 0} cache_read={u.cache_read_input_tokens or 0} "
        f"cost=${cost_usd(message.model, u):.6f}"
    )


# ---------------------------------------------------------------------------
# Offline mock: request view + responder contract
# ---------------------------------------------------------------------------
def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "tool_result":
            parts.append(_block_text(b.get("content", "")))
    return "\n".join(parts)


@dataclass
class MockRequest:
    """Convenience view over a Messages API request body, handed to responders."""

    body: dict[str, Any]
    call_index: int = 0  # how many requests this client has served before this one

    @property
    def model(self) -> str:
        return self.body.get("model", "")

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.body.get("messages", [])

    @property
    def system_text(self) -> str:
        return _block_text(self.body.get("system", ""))

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self.body.get("tools", []) or []

    @property
    def tool_names(self) -> list[str]:
        return [t.get("name", "") for t in self.tools]

    @property
    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return _block_text(m.get("content", ""))
        return ""

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        """tool_result blocks in the most recent user turn (empty if none)."""
        for m in reversed(self.messages):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    return [b for b in c if isinstance(b, dict) and b.get("type") == "tool_result"]
                return []
        return []

    @property
    def all_text(self) -> str:
        """System + every message's text - handy for keyword-based canned replies."""
        return "\n".join([self.system_text] + [_block_text(m.get("content", "")) for m in self.messages])


# A responder returns one of:
#   str                         -> a single text block, stop_reason "end_turn"
#   list[dict]                  -> content blocks (use text_block / tool_use_block helpers);
#                                  stop_reason is "tool_use" if any tool_use block is present
#   dict with "content" key     -> full control: {"content": [...], "stop_reason": "...", ...}
ResponderResult = Union[str, list, dict]
Responder = Callable[[MockRequest], ResponderResult]


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_use_block(name: str, input: dict[str, Any], id: str | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": id or f"toolu_{uuid.uuid4().hex[:20]}", "name": name, "input": input}


def default_responder(req: MockRequest) -> ResponderResult:
    snippet = req.last_user_text.strip().replace("\n", " ")
    if len(snippet) > 140:
        snippet = snippet[:137] + "..."
    return f"[offline mock reply from {req.model}] You asked: \"{snippet}\""


# ---------------------------------------------------------------------------
# Offline mock: transport implementation
# ---------------------------------------------------------------------------
@dataclass
class _MockState:
    responder: Responder
    calls: int = 0
    cache: dict[str, float] = field(default_factory=dict)  # prefix hash -> expiry (epoch s)


def _segments(body: dict[str, Any]) -> list[Any]:
    """Prompt render order is tools -> system -> messages; flatten into cacheable segments."""
    segs: list[Any] = list(body.get("tools", []) or [])
    system = body.get("system")
    if isinstance(system, str) and system:
        segs.append({"type": "text", "text": system})
    elif isinstance(system, list):
        segs.extend(system)
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            segs.append({"role": m.get("role"), "type": "text", "text": c})
        else:
            segs.extend({"role": m.get("role"), **b} if isinstance(b, dict) else b for b in c or [])
    return segs


def _strip_cache_control(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(v) for v in obj]
    return obj


def _usage_for(body: dict[str, Any], state: _MockState) -> dict[str, int]:
    segs = _segments(body)
    total = estimate_tokens(json.dumps(_strip_cache_control(segs), sort_keys=True))
    breakpoint_idx = -1
    for i, s in enumerate(segs):
        if isinstance(s, dict) and s.get("cache_control"):
            breakpoint_idx = i
    if body.get("cache_control") and segs:  # top-level automatic caching
        breakpoint_idx = len(segs) - 1
    usage = {"input_tokens": total, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    if breakpoint_idx < 0:
        return usage
    prefix = json.dumps(_strip_cache_control(segs[: breakpoint_idx + 1]), sort_keys=True)
    prefix_tokens = estimate_tokens(prefix)
    if prefix_tokens < MIN_CACHEABLE_TOKENS:
        return usage  # silently not cached - just like the real API
    key = hashlib.sha256((body.get("model", "") + prefix).encode()).hexdigest()
    now = time.time()
    if state.cache.get(key, 0) > now:
        usage["cache_read_input_tokens"] = prefix_tokens
    else:
        usage["cache_creation_input_tokens"] = prefix_tokens
    ttl = 3600 if '"ttl": "1h"' in prefix else 300
    state.cache[key] = now + ttl
    usage["input_tokens"] = max(0, total - prefix_tokens)
    return usage


def _normalize(result: ResponderResult) -> dict[str, Any]:
    if isinstance(result, str):
        return {"content": [text_block(result)]}
    if isinstance(result, list):
        return {"content": result}
    if isinstance(result, dict) and "content" in result:
        return dict(result)
    raise TypeError(f"Responder must return str, list of blocks, or dict with 'content'; got {type(result)!r}")


def _build_message(body: dict[str, Any], state: _MockState) -> tuple[dict[str, Any], float]:
    req = MockRequest(body=body, call_index=state.calls)
    state.calls += 1
    out = _normalize(state.responder(req))
    content: list[dict[str, Any]] = out["content"]
    stop_reason = out.get("stop_reason") or (
        "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn"
    )
    out_tokens = estimate_tokens(json.dumps(content))
    max_tokens = int(body.get("max_tokens", 4096))
    if out_tokens > max_tokens and stop_reason == "end_turn":
        # Simulate truncation: keep roughly max_tokens worth of text.
        for b in content:
            if b.get("type") == "text":
                b["text"] = b["text"][: max_tokens * 4]
        out_tokens, stop_reason = max_tokens, "max_tokens"
    usage = _usage_for(body, state)
    usage["output_tokens"] = out_tokens
    usage.update(out.get("usage", {}))
    message = {
        "id": f"msg_offline_{uuid.uuid4().hex[:16]}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", DEFAULT_MODEL),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }
    if "stop_details" in out:
        message["stop_details"] = out["stop_details"]
    ttft, per_tok = _LATENCY_PROFILE_MS.get(message["model"], _LATENCY_PROFILE_MS[DEFAULT_MODEL])
    latency_s = (ttft + per_tok * out_tokens) / 1000.0 * LATENCY_SCALE
    return message, latency_s


def _sse(message: dict[str, Any]) -> bytes:
    """Render a complete message as a Messages API SSE stream."""
    events: list[tuple[str, dict[str, Any]]] = []
    start = dict(message, content=[], stop_reason=None)
    start["usage"] = dict(message["usage"], output_tokens=1)
    events.append(("message_start", {"type": "message_start", "message": start}))
    for i, block in enumerate(message["content"]):
        btype = block.get("type")
        if btype == "text":
            events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}))
            text = block["text"]
            for j in range(0, len(text), 24):
                events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": text[j : j + 24]}}))
        elif btype == "tool_use":
            events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}))
            raw = json.dumps(block["input"])
            for j in range(0, len(raw), 32):
                events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": raw[j : j + 32]}}))
        else:
            events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": block}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": message["usage"]["output_tokens"]}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def _route(request: httpx2.Request, state: _MockState) -> tuple[httpx2.Response, float]:
    path = request.url.path
    body = json.loads(request.content or b"{}")
    if path.endswith("/messages/count_tokens"):
        tokens = estimate_tokens(json.dumps(_strip_cache_control(_segments(body)), sort_keys=True))
        return httpx2.Response(200, json={"input_tokens": tokens}), 0.0
    if path.endswith("/messages"):
        message, latency = _build_message(body, state)
        headers = {"request-id": f"req_offline_{uuid.uuid4().hex[:12]}"}
        if body.get("stream"):
            headers["content-type"] = "text/event-stream"
            return httpx2.Response(200, content=_sse(message), headers=headers), latency
        return httpx2.Response(200, json=message, headers=headers), latency
    return httpx2.Response(
        404,
        json={"type": "error", "error": {"type": "not_found_error", "message": f"Offline mock does not implement {path}. Run with LABS_MODE=live."}},
    ), 0.0


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------
def get_client(responder: Responder | None = None, **kwargs: Any) -> anthropic.Anthropic:
    """Return a sync Anthropic client: real when live, mock-transport backed when offline.

    `responder` is only consulted offline. Extra kwargs go to anthropic.Anthropic().
    """
    if is_live():
        return anthropic.Anthropic(**kwargs)
    state = _MockState(responder=responder or default_responder)

    def handler(request: httpx2.Request) -> httpx2.Response:
        response, latency = _route(request, state)
        time.sleep(latency)
        return response

    return anthropic.Anthropic(
        api_key="offline-mock-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        **kwargs,
    )


class _AsyncMockTransport(httpx2.AsyncBaseTransport):
    """Async transport whose simulated latency uses asyncio.sleep, so concurrent calls overlap."""

    def __init__(self, state: _MockState) -> None:
        self._state = state

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        response, latency = _route(request, self._state)
        await asyncio.sleep(latency)
        return response


def get_async_client(responder: Responder | None = None, **kwargs: Any) -> anthropic.AsyncAnthropic:
    """Async variant - offline latency uses asyncio.sleep so concurrency is real."""
    if is_live():
        return anthropic.AsyncAnthropic(**kwargs)
    state = _MockState(responder=responder or default_responder)
    return anthropic.AsyncAnthropic(
        api_key="offline-mock-key",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=_AsyncMockTransport(state)),
        **kwargs,
    )
