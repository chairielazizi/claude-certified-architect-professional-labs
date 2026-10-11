"""
Resilient Claude client wrapper (D7.3): typed SDK errors handled most-specific-first, retry-after honored,
exponential backoff with full jitter, fallback model, per-model circuit breaker, request_id logging.

Two pieces here are SIMULATION SCAFFOLDING, clearly separated from the production pattern:
  * FaultInjector - the shared offline mock only returns 200/404, so this raises the SDK's OWN typed
    exceptions, built exactly as the SDK builds them from an HTTP error (status, retry-after and
    request-id headers, JSON error body). In live mode you would not use it.
  * VirtualClock - sleeps advance a virtual clock so the demo runs in milliseconds. Production code
    would call time.sleep() (or asyncio.sleep()).

Production note: the SDK already retries 408/409/429/5xx (max_retries=2 by default, honoring retry-after).
The shared lab client sets max_retries=0 so this policy is visible. In production use ONE retry layer
(or keep SDK retries low): SDK x wrapper x caller retries multiply into a retry storm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
import httpx2

SIM_URL = "https://api.anthropic.com/v1/messages"


# ---------------------------------------------------------------------------
# Determinism helpers (identical algorithms in the TypeScript twin)
# ---------------------------------------------------------------------------
def mulberry32(seed: int) -> Callable[[], float]:
    """Tiny seeded PRNG so jittered delays are reproducible (and identical in Python and TypeScript)."""
    state = seed & 0xFFFFFFFF

    def rand() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((state ^ (state >> 15)) * (state | 1)) & 0xFFFFFFFF
        t = (((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rand


@dataclass
class VirtualClock:
    now: float = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# SIMULATION ONLY: raise the SDK's typed errors as if the API had returned them
# ---------------------------------------------------------------------------
@dataclass
class Fault:
    kind: str                      # "status" | "timeout" | "connection"
    status: int = 0
    error_type: str = ""
    message: str = ""
    retry_after: float | None = None


def make_status_error(status: int, error_type: str, message: str, request_id: str,
                      retry_after: float | None = None) -> anthropic.APIStatusError:
    """Mirror of the SDK's own status -> exception mapping (anthropic._client._make_status_error)."""
    headers = {"request-id": request_id}
    if retry_after is not None:
        headers["retry-after"] = str(int(retry_after)) if float(retry_after).is_integer() else str(retry_after)
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    response = httpx2.Response(status, headers=headers, json=body, request=httpx2.Request("POST", SIM_URL))
    classes: dict[int, type[anthropic.APIStatusError]] = {
        400: anthropic.BadRequestError, 401: anthropic.AuthenticationError, 403: anthropic.PermissionDeniedError,
        404: anthropic.NotFoundError, 429: anthropic.RateLimitError, 529: anthropic.OverloadedError,
    }
    cls = classes.get(status) or (anthropic.InternalServerError if status >= 500 else anthropic.APIStatusError)
    return cls(f"Error code: {status} - {json.dumps(body)}", response=response, body=body)


class FaultInjector:
    def __init__(self) -> None:
        self.plan: dict[str, list[Fault | None]] = {}
        self.injected = 0

    def script(self, model: str, faults: list[Fault | None]) -> None:
        self.plan[model] = list(faults)

    def before_request(self, model: str) -> None:
        queue = self.plan.get(model)
        if not queue:
            return
        fault = queue.pop(0)
        if fault is None:
            return
        self.injected += 1
        request = httpx2.Request("POST", SIM_URL)
        if fault.kind == "timeout":
            raise anthropic.APITimeoutError(request=request)
        if fault.kind == "connection":
            raise anthropic.APIConnectionError(request=request)
        raise make_status_error(fault.status, fault.error_type, fault.message, f"req_sim_{self.injected:04d}", fault.retry_after)


# ---------------------------------------------------------------------------
# Production pattern: circuit breaker + resilient wrapper
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """CLOSED -> (N consecutive failures) -> OPEN -> (cooldown) -> HALF_OPEN -> probe -> CLOSED / OPEN."""

    def __init__(self, clock: VirtualClock, failure_threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self.clock = clock
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.state = "CLOSED"
        self.failures = 0
        self.opened_at = 0.0

    def allow(self) -> bool:
        if self.state == "OPEN":
            if self.clock.now - self.opened_at >= self.cooldown_s:
                self.state = "HALF_OPEN"   # let exactly one probe through
                return True
            return False                   # fail fast: no request leaves the process
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == "HALF_OPEN" or self.failures >= self.failure_threshold:
            self.state = "OPEN"
            self.opened_at = self.clock.now


class AllModelsFailedError(Exception):
    pass


@dataclass
class ResilientClaude:
    client: anthropic.Anthropic
    models: list[str]                          # primary first, then fallbacks
    injector: FaultInjector
    clock: VirtualClock
    rng: Callable[[], float]
    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    max_retry_after_s: float = 30.0            # longer than this -> fail over instead of waiting
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    sent: int = 0                              # requests that actually left the process (or would have)

    def __post_init__(self) -> None:
        for m in self.models:
            self.breakers.setdefault(m, CircuitBreaker(self.clock))

    def log(self, event: str, pairs: list[tuple[str, str]]) -> None:
        print(f"    t=+{self.clock.now:6.2f}s {event:<16} " + " ".join(f"{k}={v}" for k, v in pairs))

    def backoff(self, attempt: int) -> float:
        """Exponential backoff with FULL jitter: uniform(0, min(cap, base * 2^(attempt-1)))."""
        return self.rng() * min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))

    def _server_error(self, e: anthropic.APIStatusError, model: str, attempt: int, label: str) -> bool:
        """Shared handling for 529 and other 5xx. Returns True to retry the same model."""
        breaker = self.breakers[model]
        breaker.record_failure()
        base = [("model", model), ("attempt", str(attempt)), ("status", str(e.status_code)), ("request_id", str(e.request_id))]
        if breaker.state == "OPEN":
            self.log(label, base + [("action", f"circuit OPEN for {breaker.cooldown_s:.0f}s -> next model")])
            return False
        if attempt == self.max_attempts:
            self.log(label, base + [("action", "attempts exhausted -> next model")])
            return False
        wait = self.backoff(attempt)
        self.log(label, base + [("wait", f"{wait:.2f}s"), ("breaker", f"{breaker.failures}/{breaker.failure_threshold}")])
        self.clock.sleep(wait)
        return True

    def create(self, **params: Any) -> Any:
        for model in self.models:
            breaker = self.breakers[model]
            if not breaker.allow():
                self.log("circuit_open", [("model", model), ("action", "skip without sending a request")])
                continue
            if breaker.state == "HALF_OPEN":
                self.log("half_open_probe", [("model", model)])
            for attempt in range(1, self.max_attempts + 1):
                self.sent += 1
                try:
                    self.injector.before_request(model)    # SIMULATION ONLY (see FaultInjector)
                    msg = self.client.messages.create(model=model, **params)
                # --- most specific first: each class needs a different decision ---------------
                except anthropic.BadRequestError as e:          # 400: the request itself is wrong
                    self.log("bad_request", [("model", model), ("status", "400"), ("request_id", str(e.request_id)),
                                             ("action", "do not retry - fix the request")])
                    raise
                except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:  # 401/403
                    self.log("auth_error", [("model", model), ("status", str(e.status_code)), ("action", "page on-call; do not retry")])
                    raise
                except anthropic.NotFoundError as e:            # 404: model id unknown / not enabled
                    self.log("not_found", [("model", model), ("request_id", str(e.request_id)), ("action", "next model")])
                    break
                except anthropic.RateLimitError as e:           # 429: OUR quota - honor retry-after
                    raw = e.response.headers.get("retry-after")
                    retry_after = float(raw) if raw else None
                    base = [("model", model), ("attempt", str(attempt)), ("status", "429"), ("request_id", str(e.request_id))]
                    if retry_after is not None and retry_after > self.max_retry_after_s:
                        self.log("rate_limited", base + [("retry_after", f"{retry_after:.0f}s"), ("action", "exceeds latency budget -> next model")])
                        break
                    if attempt == self.max_attempts:
                        self.log("rate_limited", base + [("action", "attempts exhausted -> next model")])
                        break
                    wait = retry_after if retry_after is not None else self.backoff(attempt)
                    self.log("rate_limited", base + [("retry_after", f"{retry_after:.0f}s" if retry_after is not None else "none"),
                                                     ("wait", f"{wait:.2f}s")])
                    self.clock.sleep(wait)
                    continue
                except anthropic.OverloadedError as e:          # 529: provider capacity (sibling of InternalServerError)
                    if self._server_error(e, model, attempt, "overloaded"):
                        continue
                    break
                except anthropic.InternalServerError as e:      # other 5xx
                    if self._server_error(e, model, attempt, "server_error"):
                        continue
                    break
                except anthropic.APIStatusError as e:           # any other non-2xx (409, 413, 422, ...)
                    self.log("api_error", [("model", model), ("status", str(e.status_code)), ("action", "raise")])
                    raise
                except anthropic.APITimeoutError:               # subclass of APIConnectionError -> catch first
                    wait = self.backoff(attempt)
                    self.log("timeout", [("model", model), ("attempt", str(attempt)), ("wait", f"{wait:.2f}s")])
                    self.clock.sleep(wait)
                    continue
                except anthropic.APIConnectionError:
                    wait = self.backoff(attempt)
                    self.log("connection_error", [("model", model), ("attempt", str(attempt)), ("wait", f"{wait:.2f}s")])
                    self.clock.sleep(wait)
                    continue
                breaker.record_success()
                self.log("success", [("model", model), ("attempt", str(attempt)), ("stop_reason", str(msg.stop_reason)),
                                     ("request_id", str(msg._request_id))])
                return msg
        raise AllModelsFailedError("primary and fallback models all failed")
