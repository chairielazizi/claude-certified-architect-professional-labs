/**
 * Resilient Claude client wrapper (D7.3) - twin of python/resilience.py.
 *
 * Typed SDK errors handled most-specific-first, retry-after honored, exponential backoff with full
 * jitter, fallback model, per-model circuit breaker, request_id logging.
 *
 * SIMULATION SCAFFOLDING (clearly separated from the production pattern):
 *   - FaultInjector: the shared offline mock only returns 200/404, so this throws the SDK's OWN typed
 *     errors via Anthropic.APIError.generate() - the same factory the SDK uses for HTTP error responses
 *     (status, retry-after / request-id headers, JSON error body). Not used in live mode.
 *   - VirtualClock: sleeps advance a virtual clock so the demo runs instantly.
 *
 * TypeScript SDK note: there is no OverloadedError class - every status >= 500 (including 529) is an
 * InternalServerError, so branch on err.status. APIConnectionTimeoutError extends APIConnectionError,
 * which extends APIError - check them in that order, before the APIError catch-all.
 *
 * Production note: the SDK already retries 408/409/429/5xx (maxRetries=2 by default, honoring
 * retry-after). The shared lab client sets maxRetries=0 so this policy is visible. Use ONE retry layer.
 */

import Anthropic, { APIError } from "@anthropic-ai/sdk";

// ---------------------------------------------------------------------------
// Determinism helpers (identical algorithms in the Python twin)
// ---------------------------------------------------------------------------
/** Tiny seeded PRNG so jittered delays are reproducible (and identical in Python and TypeScript). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class VirtualClock {
  now = 0;
  sleep(seconds: number): void {
    this.now += seconds;
  }
}

// ---------------------------------------------------------------------------
// SIMULATION ONLY: throw the SDK's typed errors as if the API had returned them
// ---------------------------------------------------------------------------
export interface Fault {
  kind: "status" | "timeout" | "connection";
  status?: number;
  errorType?: string;
  message?: string;
  retryAfter?: number | null;
}

export function makeStatusError(status: number, errorType: string, message: string, requestId: string, retryAfter: number | null = null): APIError {
  const headers = new Headers({ "request-id": requestId });
  if (retryAfter !== null) headers.set("retry-after", String(retryAfter));
  const body = { type: "error", error: { type: errorType, message } };
  return APIError.generate(status, body, undefined, headers);
}

export class FaultInjector {
  plan = new Map<string, (Fault | null)[]>();
  injected = 0;

  script(model: string, faults: (Fault | null)[]): void {
    this.plan.set(model, [...faults]);
  }

  beforeRequest(model: string): void {
    const queue = this.plan.get(model);
    if (!queue || !queue.length) return;
    const fault = queue.shift();
    if (!fault) return;
    this.injected++;
    if (fault.kind === "timeout") throw new Anthropic.APIConnectionTimeoutError();
    if (fault.kind === "connection") throw new Anthropic.APIConnectionError({ message: "Connection error." });
    throw makeStatusError(fault.status!, fault.errorType!, fault.message!, `req_sim_${String(this.injected).padStart(4, "0")}`, fault.retryAfter ?? null);
  }
}

// ---------------------------------------------------------------------------
// Production pattern: circuit breaker + resilient wrapper
// ---------------------------------------------------------------------------
/** CLOSED -> (N consecutive failures) -> OPEN -> (cooldown) -> HALF_OPEN -> probe -> CLOSED / OPEN. */
export class CircuitBreaker {
  clock: VirtualClock;
  failureThreshold: number;
  cooldownS: number;
  state: "CLOSED" | "OPEN" | "HALF_OPEN" = "CLOSED";
  failures = 0;
  openedAt = 0;

  constructor(clock: VirtualClock, failureThreshold = 3, cooldownS = 30) {
    this.clock = clock;
    this.failureThreshold = failureThreshold;
    this.cooldownS = cooldownS;
  }

  allow(): boolean {
    if (this.state === "OPEN") {
      if (this.clock.now - this.openedAt >= this.cooldownS) {
        this.state = "HALF_OPEN"; // let exactly one probe through
        return true;
      }
      return false; // fail fast: no request leaves the process
    }
    return true;
  }

  recordSuccess(): void {
    this.failures = 0;
    this.state = "CLOSED";
  }

  recordFailure(): void {
    this.failures++;
    if (this.state === "HALF_OPEN" || this.failures >= this.failureThreshold) {
      this.state = "OPEN";
      this.openedAt = this.clock.now;
    }
  }
}

export class AllModelsFailedError extends Error {}

export type CreateParams = Omit<Anthropic.MessageCreateParamsNonStreaming, "model">;

export class ResilientClaude {
  client: Anthropic;
  models: string[]; // primary first, then fallbacks
  injector: FaultInjector;
  clock: VirtualClock;
  rng: () => number;
  maxAttempts = 4;
  baseDelayS = 0.5;
  maxDelayS = 8;
  maxRetryAfterS = 30; // longer than this -> fail over instead of waiting
  breakers = new Map<string, CircuitBreaker>();
  sent = 0; // requests that actually left the process (or would have)

  constructor(client: Anthropic, models: string[], injector: FaultInjector, clock: VirtualClock, rng: () => number) {
    this.client = client;
    this.models = models;
    this.injector = injector;
    this.clock = clock;
    this.rng = rng;
    for (const m of models) this.breakers.set(m, new CircuitBreaker(clock));
  }

  log(event: string, pairs: [string, string][]): void {
    console.log(`    t=+${this.clock.now.toFixed(2).padStart(6)}s ${event.padEnd(16)} ` + pairs.map(([k, v]) => `${k}=${v}`).join(" "));
  }

  /** Exponential backoff with FULL jitter: uniform(0, min(cap, base * 2^(attempt-1))). */
  backoff(attempt: number): number {
    return this.rng() * Math.min(this.maxDelayS, this.baseDelayS * 2 ** (attempt - 1));
  }

  /** Shared handling for 529 and other 5xx. Returns true to retry the same model. */
  private serverError(e: APIError, model: string, attempt: number, label: string): boolean {
    const breaker = this.breakers.get(model)!;
    breaker.recordFailure();
    const base: [string, string][] = [["model", model], ["attempt", String(attempt)], ["status", String(e.status)], ["request_id", String(e.requestID)]];
    if (breaker.state === "OPEN") {
      this.log(label, [...base, ["action", `circuit OPEN for ${breaker.cooldownS.toFixed(0)}s -> next model`]]);
      return false;
    }
    if (attempt === this.maxAttempts) {
      this.log(label, [...base, ["action", "attempts exhausted -> next model"]]);
      return false;
    }
    const wait = this.backoff(attempt);
    this.log(label, [...base, ["wait", `${wait.toFixed(2)}s`], ["breaker", `${breaker.failures}/${breaker.failureThreshold}`]]);
    this.clock.sleep(wait);
    return true;
  }

  async create(params: CreateParams): Promise<Anthropic.Message> {
    for (const model of this.models) {
      const breaker = this.breakers.get(model)!;
      if (!breaker.allow()) {
        this.log("circuit_open", [["model", model], ["action", "skip without sending a request"]]);
        continue;
      }
      if (breaker.state === "HALF_OPEN") this.log("half_open_probe", [["model", model]]);
      for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
        this.sent++;
        let msg: Anthropic.Message & { _request_id?: string | null };
        try {
          this.injector.beforeRequest(model); // SIMULATION ONLY (see FaultInjector)
          msg = await this.client.messages.create({ model, ...params });
        } catch (err) {
          // --- most specific first: each class needs a different decision ---------------
          if (err instanceof Anthropic.BadRequestError) { // 400: the request itself is wrong
            this.log("bad_request", [["model", model], ["status", "400"], ["request_id", String(err.requestID)], ["action", "do not retry - fix the request"]]);
            throw err;
          }
          if (err instanceof Anthropic.AuthenticationError || err instanceof Anthropic.PermissionDeniedError) { // 401/403
            this.log("auth_error", [["model", model], ["status", String(err.status)], ["action", "page on-call; do not retry"]]);
            throw err;
          }
          if (err instanceof Anthropic.NotFoundError) { // 404: model id unknown / not enabled
            this.log("not_found", [["model", model], ["request_id", String(err.requestID)], ["action", "next model"]]);
            break;
          }
          if (err instanceof Anthropic.RateLimitError) { // 429: OUR quota - honor retry-after
            const raw = err.headers?.get("retry-after");
            const retryAfter = raw ? Number(raw) : null;
            const base: [string, string][] = [["model", model], ["attempt", String(attempt)], ["status", "429"], ["request_id", String(err.requestID)]];
            if (retryAfter !== null && retryAfter > this.maxRetryAfterS) {
              this.log("rate_limited", [...base, ["retry_after", `${retryAfter.toFixed(0)}s`], ["action", "exceeds latency budget -> next model"]]);
              break;
            }
            if (attempt === this.maxAttempts) {
              this.log("rate_limited", [...base, ["action", "attempts exhausted -> next model"]]);
              break;
            }
            const wait = retryAfter ?? this.backoff(attempt);
            this.log("rate_limited", [...base, ["retry_after", retryAfter !== null ? `${retryAfter.toFixed(0)}s` : "none"], ["wait", `${wait.toFixed(2)}s`]]);
            this.clock.sleep(wait);
            continue;
          }
          if (err instanceof Anthropic.InternalServerError) { // >= 500, including 529 overloaded
            if (this.serverError(err, model, attempt, err.status === 529 ? "overloaded" : "server_error")) continue;
            break;
          }
          if (err instanceof Anthropic.APIConnectionTimeoutError) { // subclass of APIConnectionError -> check first
            const wait = this.backoff(attempt);
            this.log("timeout", [["model", model], ["attempt", String(attempt)], ["wait", `${wait.toFixed(2)}s`]]);
            this.clock.sleep(wait);
            continue;
          }
          if (err instanceof Anthropic.APIConnectionError) {
            const wait = this.backoff(attempt);
            this.log("connection_error", [["model", model], ["attempt", String(attempt)], ["wait", `${wait.toFixed(2)}s`]]);
            this.clock.sleep(wait);
            continue;
          }
          if (err instanceof Anthropic.APIError) { // any other non-2xx (409, 413, 422, ...)
            this.log("api_error", [["model", model], ["status", String(err.status)], ["action", "raise"]]);
          }
          throw err;
        }
        breaker.recordSuccess();
        this.log("success", [["model", model], ["attempt", String(attempt)], ["stop_reason", String(msg.stop_reason)], ["request_id", String(msg._request_id)]]);
        return msg;
      }
    }
    throw new AllModelsFailedError("primary and fallback models all failed");
  }
}
