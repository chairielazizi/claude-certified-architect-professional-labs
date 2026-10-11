"""
D4.5 - Token, Latency and Cost-Performance Optimization

Start from a measured baseline workload (12 refund questions answered over a large
static policy manual plus a verbose tool result) and apply levers IN ORDER, measuring
each one: cost per COMPLETED task (retries and escalations included), p95 latency and
quality.

  Free wins first:  1 prompt caching  2 trim input tokens  3 right-size output  4 Batches API
  Trade-offs last:  5 effort tuning   6 model routing cascade   7 parallelism (latency)

Run from the repo root:
    .venv/Scripts/python "D4.5 - Token, Latency and Cost-Performance Optimization/python/main.py"
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    PRICING,
    MockRequest,
    banner,
    cost_usd,
    elapsed_ms,
    estimate_tokens,
    get_async_client,
    get_client,
    is_live,
    section,
    text_block,
)

P95_SLO_MS = 6000.0
MAX_ATTEMPTS = 2

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


def signed_pct(x: float) -> str:
    return ("+" if x >= 0 else "") + fmt(100.0 * x, 1) + "%"


def total(xs: Any) -> float:
    s = 0.0
    for x in xs:
        s += x
    return s


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    rank = p / 100.0 * (len(s) - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


# ---------------------------------------------------------------------------
# Workload: a big static manual (cacheable) + per-task verbose tool results
# ---------------------------------------------------------------------------
RULES = [
    ("R-1", "Unopened items are refunded at the full item price times quantity."),
    ("R-2", "Opened items carry a 10 percent restocking fee: refund 90 percent of price times quantity."),
    ("R-3", "If an order-level promotion applied, first reduce the item price by the promotion percentage, then apply R-1 or R-2."),
]
FILLER_TOPICS = ["warranty claims", "gift cards", "loyalty points", "price matching", "pre-orders", "backorders", "bulk orders",
                 "international duties", "carrier claims", "lost parcels", "address changes", "split shipments", "subscriptions",
                 "trade-ins", "repairs", "extended warranty", "student discounts", "tax exemption", "invoices", "chargebacks",
                 "fraud review", "account merges", "data requests", "accessibility", "corporate accounts", "reseller orders",
                 "events", "rentals", "refurbished items", "bundles", "custom engraving", "personalization", "digital goods",
                 "software licenses", "installation", "recycling", "hazardous goods", "oversized items", "white glove", "pickup points"]
MANUAL = "ACME RETURNS AND REFUNDS MANUAL (v12)\n\n" + "\n".join(f"Section {sid}: {txt}" for sid, txt in RULES) + "\n\n" + "\n".join(
    f"Section G-{i + 1:02d} ({t}): Requests about {t} are handled by the customer care team. Agents verify the order, record the "
    f"request in the case system, and follow the {t} checklist. Escalate unusual {t} requests to a team lead within one business day."
    for i, t in enumerate(FILLER_TOPICS))
ITEM_NAMES = ["desk lamp", "office chair", "monitor arm", "keyboard", "webcam", "headset", "dock", "mouse"]
LEVEL = {"easy": 1, "medium": 2, "hard": 3}


def make_task(i: int) -> dict[str, Any]:
    difficulty = "hard" if i % 6 == 5 else ("medium" if i % 6 == 4 else "easy")
    items = [{"sku": f"SKU-{100 + i * 3 + k}", "name": ITEM_NAMES[(i + k) % len(ITEM_NAMES)],
              "unit_price": 15 + ((i * 37 + k * 11) % 80) + 0.5, "qty": 1 + (i + k) % 3,
              "opened": difficulty != "easy" and k == i % 3} for k in range(3)]
    promo = 20 if difficulty == "hard" else 0
    history = [{"ts": f"2026-09-{1 + e:02d}T{8 + e % 9:02d}:15:00Z", "event": ["created", "paid", "picked", "packed", "shipped", "in transit",
                "out for delivery", "delivered", "survey sent", "return requested", "label issued", "carrier scan", "received", "inspected"][e],
                "actor": "system" if e % 2 else "warehouse-bot", "note": f"Automated status update number {e + 1} for order ORD-5{i:04d}."}
               for e in range(14)]
    record = {"order_id": f"ORD-5{i:04d}", "customer": {"name": f"Customer {i}", "tier": "standard", "since": "2021-03-04"},
              "items": items, "order_promo_pct": promo, "shipping": {"method": "ground", "cost": 7.5, "address": "redacted"},
              "payment": {"method": "card", "last4": "4242", "captured": True}, "history": history}
    ret = items[i % 3]
    price = ret["unit_price"] * (1 - promo / 100)
    refund = price * ret["qty"] * (0.9 if ret["opened"] else 1.0)
    rule = "R-3" if promo else ("R-2" if ret["opened"] else "R-1")
    return {"id": f"T{i + 1:02d}", "order_id": record["order_id"], "difficulty": difficulty, "record": record,
            "return_sku": ret["sku"], "expected_amount": float(fmt(refund, 2)), "expected_rule": rule}


TASKS = [make_task(i) for i in range(12)]
GET_ORDER_TOOL = {"name": "get_order", "description": "Fetch an order record by id.",
                  "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}


def trimmed_record(task: dict[str, Any]) -> dict[str, Any]:
    """Project the tool result to the fields the task needs (lever 2)."""
    item = next(it for it in task["record"]["items"] if it["sku"] == task["return_sku"])
    return {"order_id": task["order_id"], "return_item": {"sku": item["sku"], "unit_price": item["unit_price"], "qty": item["qty"],
            "opened": item["opened"]}, "order_promo_pct": task["record"]["order_promo_pct"]}


# ---------------------------------------------------------------------------
# Offline responder: quality depends on model x effort vs task difficulty
# ---------------------------------------------------------------------------
CAPABILITY = {(MODEL_HAIKU, "low"): 1, (MODEL_HAIKU, "medium"): 1, (MODEL_SONNET, "low"): 1, (MODEL_SONNET, "medium"): 2,
              (MODEL_SONNET, "high"): 2, (MODEL_OPUS, "low"): 2, (MODEL_OPUS, "medium"): 3, (MODEL_OPUS, "high"): 3}
THINKING = {"low": 40, "medium": 250, "high": 700}  # adaptive-thinking tokens - billed as output


def responder(req: MockRequest) -> Any:
    m = re.search(r"ORD-5\d{4}", req.all_text)
    task = next(t for t in TASKS if m and t["order_id"] == m.group(0))
    effort = (req.body.get("output_config") or {}).get("effort", "high")
    ok = CAPABILITY.get((req.model, effort), 1) >= LEVEL[task["difficulty"]]
    if ok:
        amount, rule = task["expected_amount"], task["expected_rule"]
    else:  # typical slip: skip the last rule in the chain
        item = next(it for it in task["record"]["items"] if it["sku"] == task["return_sku"])
        amount = float(fmt(item["unit_price"] * item["qty"] * (0.9 if task["difficulty"] == "hard" else 1.0), 2))
        rule = "R-2" if task["difficulty"] == "hard" else "R-1"
    text = f"Refund: {fmt(amount, 2)} dollars (policy {rule})."
    if "at most two sentences" not in req.all_text:
        text += (" To explain in detail: the manual describes how refunds are computed from the item price, the quantity returned, "
                 "whether the packaging was opened and whether an order-level promotion applied. I checked the order history, the "
                 "payment record and the shipping record, confirmed the item was delivered and inspected, and then applied the "
                 "relevant sections in order. Shipping costs are not refunded for this case, and the refund goes back to the "
                 "original card ending in 4242 within five to seven business days after inspection.")
    think = THINKING[effort]
    return {"content": [text_block(text)],
            "usage": {"output_tokens": estimate_tokens(text) + think, "output_tokens_details": {"thinking_tokens": think}}}


# ---------------------------------------------------------------------------
# Configurations and measurement
# ---------------------------------------------------------------------------
@dataclass
class Config:
    name: str
    lever: str
    kind: str                 # baseline | free | trade-off
    model: str = MODEL_OPUS
    effort: str = "high"
    cache: bool = False
    trim: bool = False
    concise: bool = False
    max_tokens: int = 4096
    cascade: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Attempt:
    model: str
    cost: float
    latency_ms: float
    ok: bool
    input_tokens: int
    cache_read: int
    cache_write: int
    output_tokens: int
    thinking: int


@dataclass
class TaskRun:
    task_id: str
    attempts: list[Attempt]

    @property
    def completed(self) -> bool:
        return self.attempts[-1].ok

    @property
    def latency(self) -> float:
        return total(a.latency_ms for a in self.attempts)  # retries/escalations add user-visible latency


@dataclass
class Result:
    cfg: Config
    runs: list[TaskRun]

    @property
    def attempts(self) -> list[Attempt]:
        return [a for r in self.runs for a in r.attempts]

    @property
    def completed(self) -> int:
        return sum(1 for r in self.runs if r.completed)

    @property
    def cost_per_completed(self) -> float:
        return total(a.cost for a in self.attempts) / max(1, self.completed)

    @property
    def p95(self) -> float:
        return percentile([r.latency for r in self.runs], 95)


LAT = {MODEL_HAIKU: (250.0, 3.0, 0.01), MODEL_SONNET: (500.0, 7.0, 0.02), MODEL_OPUS: (900.0, 12.0, 0.04)}  # ttft, ms/out tok, ms/in tok


def modeled_latency(key: str, model: str, usage: Any) -> float:
    """Illustrative latency model: TTFT + prefill (cache reads ~10% of the cost of fresh input) + generation."""
    ttft, per_out, per_in = LAT[model]
    rng = mulberry32(fnv1a32(key))
    prefill = per_in * (usage.input_tokens + (usage.cache_creation_input_tokens or 0) + 0.1 * (usage.cache_read_input_tokens or 0))
    return ttft * (0.9 + 0.2 * rng()) + prefill + per_out * usage.output_tokens


def build_request(cfg: Config, task: dict[str, Any], model: str, effort: str) -> dict[str, Any]:
    style = ("Answer in at most two sentences: the refund amount and the policy section id." if cfg.concise
             else "Explain your reasoning in detail, including relevant background from the manual.")
    record = json.dumps(trimmed_record(task), separators=(",", ":")) if cfg.trim else json.dumps(task["record"], indent=2)
    system: Any = [{"type": "text", "text": MANUAL, "cache_control": {"type": "ephemeral"}}] if cfg.cache else MANUAL
    return {
        "model": model, "max_tokens": cfg.max_tokens, "system": system, "tools": [GET_ORDER_TOOL],
        "output_config": {"effort": effort},
        "messages": [
            {"role": "user", "content": f"Order {task['order_id']}: the customer returns {task['return_sku']}. How much is refunded? {style}"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"toolu_{task['id']}", "name": "get_order", "input": {"order_id": task["order_id"]}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"toolu_{task['id']}", "content": record}]},
        ],
    }


def validate(task: dict[str, Any], text: str) -> bool:
    m = re.search(r"Refund: (-?\d+(?:\.\d+)?) dollars \(policy (R-\d)\)", text)
    return bool(m) and abs(float(m.group(1)) - task["expected_amount"]) < 0.01 and m.group(2) == task["expected_rule"]


async def call(client: Any, cfg: Config, task: dict[str, Any], model: str, effort: str, key: str) -> Attempt:
    t0 = time.perf_counter()
    msg = await client.messages.create(**build_request(cfg, task, model, effort))
    wall = elapsed_ms(t0)
    text = "".join(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    thinking = u.output_tokens_details.thinking_tokens if u.output_tokens_details else 0
    latency = wall if is_live() else modeled_latency(key, msg.model, u)  # offline: deterministic model, live: wall-clock
    return Attempt(model, cost_usd(msg.model, u), latency, validate(task, text), u.input_tokens,
                   u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0, u.output_tokens, thinking)


async def run_task(client: Any, cfg: Config, task: dict[str, Any]) -> TaskRun:
    attempts: list[Attempt] = []
    if cfg.cascade:  # escalate to the next tier only when the cheap answer fails validation
        for model, effort in cfg.cascade:
            attempts.append(await call(client, cfg, task, model, effort, f"{cfg.name}:{task['id']}:{model}"))
            if attempts[-1].ok:
                break
    else:
        for n in range(MAX_ATTEMPTS):  # a failed validation is retried - and billed
            attempts.append(await call(client, cfg, task, cfg.model, cfg.effort, f"{cfg.name}:{task['id']}:{n}"))
            if attempts[-1].ok:
                break
    return TaskRun(task["id"], attempts)


async def run_config(cfg: Config) -> Result:
    client = get_async_client(responder)  # fresh client = cold cache, so every config pays its own cache writes
    runs = [await run_task(client, cfg, t) for t in TASKS]  # sequential, so later requests can read the cache
    return Result(cfg, runs)


def schedule(latencies: list[float], workers: int) -> float:
    """Makespan of running tasks in order on N workers (greedy: next task -> first free worker)."""
    free = [0.0] * workers
    for lat in latencies:
        i = free.index(min(free))
        free[i] += lat
    return max(free)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main_async() -> None:
    banner("D4.5 - Token, Latency and Cost-Performance Optimization")
    diffs = {d: sum(1 for t in TASKS if t["difficulty"] == d) for d in ("easy", "medium", "hard")}
    print(f"Workload: {len(TASKS)} refund tasks ({', '.join(f'{k}={v}' for k, v in diffs.items())}); static manual ~{estimate_tokens(MANUAL)} tokens;")
    print(f"verbose tool result ~{estimate_tokens(json.dumps(TASKS[0]['record'], indent=2))} tokens vs trimmed ~"
          f"{estimate_tokens(json.dumps(trimmed_record(TASKS[0]), separators=(',', ':')))} tokens. p95 SLO {fmt(P95_SLO_MS, 0)} ms.")
    print("Offline latency is modeled (TTFT + prefill + generation); costs come from real usage objects x PRICING.")

    base = Config("baseline", "0. baseline: Opus/high, raw JSON, verbose", "baseline")
    c1 = Config("cache", "1. prompt caching on the static prefix", "free", cache=True)
    c2 = Config("trim", "2. trim the tool result to needed fields", "free", cache=True, trim=True)
    c3 = Config("output", "3. concise output + right-sized max_tokens", "free", cache=True, trim=True, concise=True, max_tokens=1024)
    effort_med = Config("opus-medium", "5a. effort high -> medium", "trade-off", cache=True, trim=True, concise=True, max_tokens=1024, effort="medium")
    effort_low = Config("opus-low", "5b. effort -> low", "trade-off", cache=True, trim=True, concise=True, max_tokens=1024, effort="low")
    sonnet = Config("sonnet-medium", "6a. all traffic on Sonnet 5.5 (medium)", "trade-off", model=MODEL_SONNET, cache=True, trim=True, concise=True,
                    max_tokens=1024, effort="medium")
    cascade = Config("cascade", "6b. cascade Haiku -> Sonnet -> Opus", "trade-off", cache=True, trim=True, concise=True, max_tokens=1024,
                     cascade=[(MODEL_HAIKU, "low"), (MODEL_SONNET, "medium"), (MODEL_OPUS, "medium")])
    configs = [base, c1, c2, c3, effort_med, effort_low, sonnet, cascade]
    results = {r.cfg.name: r for r in await asyncio.gather(*(run_config(c) for c in configs))}

    # -- profile ------------------------------------------------------------
    section("0. Baseline token profile (where the money goes)")
    b = results["baseline"]
    a0 = b.attempts[0]
    price = PRICING[MODEL_OPUS]
    in_cost = a0.input_tokens * price["input"] / 1e6
    think_cost = a0.thinking * price["output"] / 1e6
    vis_cost = (a0.output_tokens - a0.thinking) * price["output"] / 1e6
    print(f"  per request: input {a0.input_tokens} tok (${fmt(in_cost, 5)}), visible output {a0.output_tokens - a0.thinking} tok (${fmt(vis_cost, 5)}), "
          f"thinking {a0.thinking} tok (${fmt(think_cost, 5)})")
    print("  -> the static manual is re-billed on every call (caching), the tool JSON is mostly irrelevant (trim),")
    print("     and verbose answers + high-effort thinking dominate output spend (concise output, then effort).")

    # -- free wins ----------------------------------------------------------
    section("1-4. Free wins, applied in order (quality must hold)")

    def row(label: str, kind: str, r: Result | None, prev: float, base_cost: float, cost: float | None = None, p95: str | None = None) -> float:
        c = r.cost_per_completed if r else (cost or 0.0)
        lat = p95 if p95 is not None else (fmt(r.p95, 0) if r else "-")
        q = f"{r.completed}/{len(TASKS)}" if r else f"{results['output'].completed}/{len(TASKS)}"
        held = "yes" if (r.completed if r else results["output"].completed) >= b.completed else "NO"
        print(f"  {label:<46}{kind:<10}{'$' + fmt(c, 5):>10}{signed_pct(c / prev - 1):>9}{signed_pct(c / base_cost - 1):>9}{lat:>9}{q:>7}{held:>6}")
        return c

    print(f"  {'step':<46}{'type':<10}{'$/task':>10}{'d_step':>9}{'d_base':>9}{'p95 ms':>9}{'qual':>7}{'held':>6}")
    base_cost = b.cost_per_completed
    prev = row(base.lever, "baseline", b, base_cost, base_cost)
    for cfg in (c1, c2, c3):
        prev = row(cfg.lever, "free", results[cfg.name], prev, base_cost)
    after_free = prev
    batch_cost = after_free * 0.5
    row("4. Batches API for non-urgent work (50% off)", "free", None, after_free, base_cost, cost=batch_cost, p95="async")
    r1 = results["cache"].attempts
    print(f"\n  caching detail: request 1 wrote {r1[0].cache_write} tokens (1.25x), requests 2-12 read {r1[1].cache_read} tokens each "
          f"(Opus 5.5 cache read ${fmt(price['cache_read'], 2)}/MTok vs ${fmt(price['input'], 2)} input)")
    print("  The first cached request costs MORE than uncached; measure from the second request on (warm cache).")

    section("4. Batches API request (live-only in this lab)")
    print("  client.messages.batches.create(requests=[")
    print("      {\"custom_id\": task_id, \"params\": {model, max_tokens, system (with cache_control), tools, messages}}, ...])")
    print("  -> poll client.messages.batches.retrieve(id) until processing_status == 'ended', then iterate")
    print("     client.messages.batches.results(id); results arrive in ANY order - join on custom_id.")
    if is_live():
        sync_client = get_client(responder)
        batch = sync_client.messages.batches.create(requests=[
            {"custom_id": t["id"], "params": build_request(c3, t, MODEL_OPUS, "high")} for t in TASKS[:2]])
        print(f"  LIVE: submitted batch {batch.id} status={batch.processing_status} (not polling in this demo)")
    else:
        print("  OFFLINE: the mock has no /v1/messages/batches endpoint - cost shown above is projected (50% of step 3).")
    print("  Use for evals, backfills and nightly jobs - never for a user waiting on a response (up to 24 h).")

    # -- trade-offs ---------------------------------------------------------
    section("5-6. Trade-offs: effort and model routing (measure, don't assume)")
    print(f"  {'configuration':<40}{'$/task':>10}{'p95 ms':>9}{'qual':>7}{'calls':>7}{'cache-write $':>15}  verdict")
    candidates = [("opus-high (step 3)", results["output"]), ("opus-medium", results["opus-medium"]), ("opus-low", results["opus-low"]),
                  ("sonnet-medium", results["sonnet-medium"]), ("cascade H->S->O", results["cascade"])]
    eligible: list[tuple[float, str]] = []
    for label, r in candidates:
        cw = total(a.cache_write * PRICING[a.model]["cache_write"] / 1e6 for a in r.attempts)
        quality_ok = r.completed == len(TASKS)
        latency_ok = r.p95 <= P95_SLO_MS
        verdict = "OK" if quality_ok and latency_ok else ("quality NOT held" if not quality_ok else "p95 over SLO")
        if quality_ok and latency_ok:
            eligible.append((r.cost_per_completed, label))
        print(f"  {label:<40}{'$' + fmt(r.cost_per_completed, 5):>10}{fmt(r.p95, 0):>9}{f'{r.completed}/{len(TASKS)}':>7}{len(r.attempts):>7}"
              f"{'$' + fmt(cw, 5):>15}  {verdict}")
    chosen_cost, chosen = min(eligible)
    casc = results["cascade"]
    tiers = {m: sum(1 for a in casc.attempts if a.model == m) for m in (MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS)}
    print(f"\n  cascade calls by tier: " + ", ".join(f"{m}={n}" for m, n in tiers.items()) +
          " - each tier wrote its OWN cache entry (caches are model-scoped)")
    print("  Opus at medium effort holds quality; low effort and Sonnet fail the hard tasks (and their retries are billed).")
    print("  The cascade is cheaper per task but stacks latency on escalated tasks - measure the most capable model at")
    print("  lower effort BEFORE building a cascade; build one only if the traffic mix and latency budget justify it.")
    print(f"  CHOSEN: {chosen} at ${fmt(chosen_cost, 5)}/task ({signed_pct(chosen_cost / base_cost - 1)} vs baseline), quality held, p95 within SLO")

    # -- parallelism --------------------------------------------------------
    section("7. Parallelism for wall-clock latency")
    chosen_cfg = effort_med if chosen == "opus-medium" else (c3 if chosen.startswith("opus-high") else cascade)
    client = get_async_client(responder)
    warm = await run_task(client, chosen_cfg, TASKS[0])     # warm the cache with ONE request first...
    sem = asyncio.Semaphore(4)

    async def bounded(t: dict[str, Any]) -> TaskRun:
        async with sem:
            return await run_task(client, chosen_cfg, t)

    rest = await asyncio.gather(*(bounded(t) for t in TASKS[1:]))  # ...then fan out the rest
    lats = [warm.latency] + [r.latency for r in rest]
    sequential = total(lats)
    parallel = warm.latency + schedule(lats[1:], 4)
    print(f"  12 independent tasks on {chosen}: sequential {fmt(sequential / 1000, 1)} s vs warm-up + 4-way parallel {fmt(parallel / 1000, 1)} s "
          f"({fmt(sequential / parallel, 1)}x faster wall-clock)")
    print(f"  per-request p95 is unchanged ({fmt(percentile(lats, 95), 0)} ms) and so is cost - parallelism buys throughput, not cheaper calls.")
    print(f"  fan-out cache reads after warm-up: {sum(1 for r in rest if r.attempts[0].cache_read > 0)}/11 (a cache entry is readable only once the")
    print("  first response has started, so N simultaneous cold requests would all pay full price).")

    # -- summary ------------------------------------------------------------
    section("Waterfall summary: cost per completed task")
    steps = [("baseline", base_cost), ("+ caching", results["cache"].cost_per_completed), ("+ trim input", results["trim"].cost_per_completed),
             ("+ concise output", results["output"].cost_per_completed), (f"+ {chosen}", chosen_cost)]
    width = 40
    for label, c in steps:
        bar = "#" * max(1, math.floor(width * c / base_cost + 0.5))
        print(f"  {label:<20}{'$' + fmt(c, 5):>10}  {bar}")
    print(f"  (batch re-runs of the final config would cost ~${fmt(chosen_cost * 0.5, 5)}/task)")
    print("  Order matters: free wins first (they never cost quality), then trade-offs validated against the eval.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
