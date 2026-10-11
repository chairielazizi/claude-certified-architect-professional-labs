"""
D2.1 - Model Selection and Trade-offs
=====================================

A model-selection engine an architect could actually run against a portfolio:

  1. Model catalog: tier, latency profile, context, pricing, default effort.
  2. Models API lookup (live-only) - discover limits/capabilities at runtime.
  3. Workload profiles -> every (model x effort) configuration is checked against
     hard constraints (capability, latency SLA, context) and priced at volume
     (tokens x PRICING, cache reads, Batch discount, Haiku long-context tier).
  4. Portfolio decision table vs a "one model for everything" baseline.
  5. "Most capable model at lower effort" vs "cheaper model at max effort",
     judged on cost per SOLVED task, not cost per token.
  6. Routing cascade: Haiku triages difficulty -> route to Haiku/Sonnet/Opus.
  7. Same prompt through each model with the shared client -> measured latency + cost.

Everything non-LLM here (constraint filter, cost model, router) is real code.
The heuristic numbers (tiers, effort multipliers) are ILLUSTRATIVE - the method
is the lesson; your own eval sweep supplies the real numbers.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_FABLE,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    PRICING,
    MockRequest,
    banner,
    cost_usd,
    elapsed_ms,
    get_client,
    is_live,
    section,
    text_of,
)

DATA = Path(__file__).resolve().parents[1] / "data"

# ---------------------------------------------------------------------------
# 1. Catalog. Context/max-output/default-effort are from the Oct 2026 model
#    table; tier + latency are ILLUSTRATIVE (latency mirrors the offline mock).
# ---------------------------------------------------------------------------
EFFORTS = ["low", "medium", "high", "xhigh", "max"]

# Thinking tokens are billed as output tokens, so effort mostly moves OUTPUT cost
# and latency. Multipliers and capability bonuses are illustrative assumptions.
EFFORT_OUTPUT_MULT = {"low": 0.5, "medium": 1.0, "high": 1.6, "xhigh": 2.4, "max": 3.5}
EFFORT_CAPABILITY_BONUS = {"low": -0.5, "medium": 0.0, "high": 0.4, "xhigh": 0.7, "max": 0.9}

BATCH_DISCOUNT = 0.50  # Message Batches API: 50% off input + output, async (<= 24h)
HAIKU_LONG_CONTEXT_THRESHOLD = 100_000  # Haiku 5.5 prompt > 100K tokens -> higher price tier
HAIKU_LONG_CONTEXT_PRICE = {"input": 0.50, "output": 2.50, "cache_write": 0.625, "cache_read": 0.05}


@dataclass(frozen=True)
class ModelProfile:
    id: str
    name: str
    tier: float  # 1-5 capability scale used by workload.capability_need
    ttft_ms: float
    ms_per_output_token: float
    context_window: int
    max_output: int
    default_effort: str
    best_for: str


CATALOG: list[ModelProfile] = [
    ModelProfile(MODEL_HAIKU, "Haiku 5.5", 2.0, 250, 3, 1_000_000, 128_000, "medium",
                 "routing, classification, extraction, sub-agents"),
    ModelProfile(MODEL_SONNET, "Sonnet 5.5", 3.0, 500, 7, 1_000_000, 128_000, "high",
                 "high-volume production, near-frontier coding"),
    ModelProfile(MODEL_OPUS, "Opus 5.5", 4.0, 900, 12, 1_000_000, 128_000, "medium",
                 "default: complex reasoning, agents, coding"),
    ModelProfile(MODEL_FABLE, "Fable 5.1", 5.0, 1500, 18, 1_000_000, 128_000, "high",
                 "hardest reasoning, long-horizon agents"),
]
BY_ID = {m.id: m for m in CATALOG}


@dataclass(frozen=True)
class Workload:
    id: str
    description: str
    requests_per_month: int
    avg_input_tokens: int
    avg_output_tokens: int
    cacheable_input_fraction: float
    context_need_tokens: int
    capability_need: float
    latency_sla_ms: Optional[float]
    batchable: bool
    agent_turns: int


@dataclass
class Evaluation:
    model: ModelProfile
    effort: str
    effective_capability: float
    latency_ms: float
    cost_per_request: float
    monthly_usd: float
    long_context_tier: bool
    rejected: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.rejected

    def headroom(self, wl: Workload) -> float:
        return self.effective_capability - wl.capability_need

    @property
    def label(self) -> str:
        return f"{self.model.name} @ {self.effort}"


# ---------------------------------------------------------------------------
# Formatting helpers (kept identical in the TypeScript twin)
# ---------------------------------------------------------------------------
def usd(x: float) -> str:
    if x >= 100:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    if x >= 0.01:
        return f"${x:.4f}"
    return f"${x:.6f}"


def table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))


# ---------------------------------------------------------------------------
# 3. The selection engine: constraints first, then price at volume
# ---------------------------------------------------------------------------
def unit_prices(model_id: str, prompt_tokens: int) -> dict[str, float]:
    """Per-1M-token prices for one call. Haiku 5.5 has a long-context price tier."""
    if model_id == MODEL_HAIKU and prompt_tokens > HAIKU_LONG_CONTEXT_THRESHOLD:
        return dict(HAIKU_LONG_CONTEXT_PRICE)
    return dict(PRICING[model_id])


def cost_per_request(model: ModelProfile, wl: Workload, effort: str) -> float:
    p = unit_prices(model.id, wl.context_need_tokens)
    # Steady state: the cacheable prefix is read from cache (write cost amortized away
    # at volume); the rest is billed at the base input price.
    cached = wl.avg_input_tokens * wl.cacheable_input_fraction
    uncached = wl.avg_input_tokens - cached
    out = wl.avg_output_tokens * EFFORT_OUTPUT_MULT[effort]
    usd_cost = (uncached * p["input"] + cached * p["cache_read"] + out * p["output"]) / 1_000_000
    if wl.batchable:
        usd_cost *= 1 - BATCH_DISCOUNT  # nobody is waiting -> Batches API
    return usd_cost


def latency_ms(model: ModelProfile, wl: Workload, effort: str) -> float:
    # Agent tasks pay time-to-first-token once per turn plus generation time.
    out = wl.avg_output_tokens * EFFORT_OUTPUT_MULT[effort]
    return wl.agent_turns * model.ttft_ms + out * model.ms_per_output_token


def evaluate(model: ModelProfile, wl: Workload, effort: str) -> Evaluation:
    eff_cap = model.tier + EFFORT_CAPABILITY_BONUS[effort]
    lat = latency_ms(model, wl, effort)
    cpr = cost_per_request(model, wl, effort)
    ev = Evaluation(
        model=model,
        effort=effort,
        effective_capability=eff_cap,
        latency_ms=lat,
        cost_per_request=cpr,
        monthly_usd=cpr * wl.requests_per_month,
        long_context_tier=model.id == MODEL_HAIKU and wl.context_need_tokens > HAIKU_LONG_CONTEXT_THRESHOLD,
    )
    # Hard constraints are filters, not weights: a cheap model that misses the
    # quality bar or the SLA is not "a bit worse", it is unusable.
    if eff_cap + 1e-9 < wl.capability_need:
        ev.rejected.append("capability")
    if wl.latency_sla_ms is not None and lat > wl.latency_sla_ms:
        ev.rejected.append("latency")
    if wl.context_need_tokens > model.context_window:
        ev.rejected.append("context")
    if wl.avg_output_tokens * EFFORT_OUTPUT_MULT[effort] / max(1, wl.agent_turns) > model.max_output:
        ev.rejected.append("max_output")
    return ev


def select(wl: Workload) -> tuple[list[Evaluation], list[Evaluation]]:
    """Return (feasible configs sorted cheapest-first, all configs)."""
    all_evals = [evaluate(m, wl, e) for m in CATALOG for e in EFFORTS]
    feasible = [ev for ev in all_evals if ev.feasible]
    # Cheapest first; on a tie prefer more capability headroom.
    feasible.sort(key=lambda ev: (round(ev.monthly_usd, 6), -ev.headroom(wl)))
    return feasible, all_evals


def load_workloads() -> tuple[list[Workload], dict[str, Any]]:
    raw = json.loads((DATA / "workloads.json").read_text(encoding="utf-8"))
    wls = [Workload(**{k: w[k] for k in Workload.__dataclass_fields__}) for w in raw["workloads"]]
    return wls, raw


# ---------------------------------------------------------------------------
# 6. Routing cascade
# ---------------------------------------------------------------------------
DIFFICULTY_ORDER = ["simple", "moderate", "complex"]
ROUTES = {
    "simple": (MODEL_HAIKU, "low"),
    "moderate": (MODEL_SONNET, "medium"),
    "complex": (MODEL_OPUS, "high"),
}
CONFIDENCE_FLOOR = 0.75  # below this, escalate one tier: misrouting UP costs cents, DOWN costs quality

TRIAGE_SYSTEM = (
    "You are a difficulty triage classifier for a customer-operations assistant.\n"
    "Classify the request inside <request> tags:\n"
    "- simple: one lookup or one well-specified action, no judgment\n"
    "- moderate: summarize/draft/compare with a few constraints\n"
    "- complex: multi-step diagnosis, planning, or reasoning across many facts\n"
    "Return JSON only. confidence is your probability (0-1) that the label is right."
)
TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "difficulty": {"type": "string", "enum": DIFFICULTY_ORDER},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["difficulty", "confidence", "reason"],
    "additionalProperties": False,
}
ANSWER_SYSTEM = "You are a customer operations assistant for an online retailer. Answer concisely and accurately."
PROBE_SYSTEM = "Latency probe. Answer in exactly three short bullet points."
PROBE_PROMPT = "Why do payment APIs need idempotency keys?"
PROBE_ANSWER = (
    "- Retries after timeouts are safe: the same key never charges twice.\n"
    "- The server can return the original result instead of re-executing.\n"
    "- Clients can retry aggressively without coordinating state."
)


def make_responder(cascade: list[dict[str, Any]]):
    """Offline stand-in for Claude: canned but realistic replies keyed on the request."""

    def find(text: str) -> dict[str, Any] | None:
        return next((r for r in cascade if r["text"] in text), None)

    def responder(req: MockRequest) -> Any:
        if "difficulty triage" in req.system_text:
            r = find(req.last_user_text)
            mock = r["offline_mock"] if r else {"difficulty": "complex", "confidence": 0.5, "reason": "unknown"}
            return json.dumps({k: mock[k] for k in ("difficulty", "confidence", "reason")})
        if "Latency probe" in req.system_text:
            return PROBE_ANSWER
        r = find(req.last_user_text)
        return r["offline_mock"]["answer"] if r else "I can help with that."

    return responder


def triage(client: Any, text: str) -> dict[str, Any]:
    msg = client.messages.create(
        model=MODEL_HAIKU,
        max_tokens=300,
        system=TRIAGE_SYSTEM,
        # Structured outputs guarantee parseable JSON; low effort keeps the router fast.
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": TRIAGE_SCHEMA}},
        messages=[{"role": "user", "content": f"<request>{text}</request>"}],
    )
    if msg.stop_reason != "end_turn":  # refusal / max_tokens -> fail safe to the strongest route
        return {"difficulty": "complex", "confidence": 0.0, "reason": f"triage stop_reason={msg.stop_reason}", "_msg": msg}
    data = json.loads(text_of(msg))
    data["_msg"] = msg
    return data


def route(t: dict[str, Any]) -> tuple[str, str, str, bool]:
    level = t["difficulty"]
    escalated = False
    if t["confidence"] < CONFIDENCE_FLOOR and level != "complex":
        level = DIFFICULTY_ORDER[DIFFICULTY_ORDER.index(level) + 1]
        escalated = True
    model, effort = ROUTES[level]
    return level, model, effort, escalated


def answer(client: Any, model: str, effort: str, text: str) -> tuple[Any, float]:
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=ANSWER_SYSTEM,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": text}],
    )
    return msg, elapsed_ms(t0)


# ---------------------------------------------------------------------------
# 2. Models API (live-only)
# ---------------------------------------------------------------------------
MODELS_API_SNIPPET = '''\
for m in client.models.list():                      # auto-paginates
    caps = m.capabilities                           # typed in anthropic>=1.x
    print(m.id, m.max_input_tokens, m.max_tokens,
          caps.effort.max.supported,
          caps.thinking.types.adaptive.supported,
          caps.batch.supported,
          caps.context_management.compact_20260112.supported if caps.context_management.compact_20260112 else None)
info = client.models.retrieve("claude-opus-5-5")
'''


def models_api_lookup(client: Any) -> None:
    for m in client.models.list():
        if not m.id.startswith(("claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable-5")):
            continue
        caps = m.capabilities
        if caps is None:
            print(f"  {m.id:<20} ctx={m.max_input_tokens} max_out={m.max_tokens} (no capability data)")
            continue
        print(
            f"  {m.id:<20} ctx={m.max_input_tokens} max_out={m.max_tokens} "
            f"effort.max={caps.effort.max.supported} adaptive={caps.thinking.types.adaptive.supported} "
            f"batch={caps.batch.supported} structured={caps.structured_outputs.supported}"
        )
    info = client.models.retrieve(MODEL_OPUS)
    print(f"  retrieve({MODEL_OPUS}) -> display_name={info.display_name!r} max_input_tokens={info.max_input_tokens}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    banner("D2.1 - Model Selection and Trade-offs")
    workloads, raw = load_workloads()
    cascade: list[dict[str, Any]] = raw["cascade_requests"]
    client = get_client(make_responder(cascade))

    # ---------------------------------------------------------------- 1
    section("1. Model catalog (Oct 2026 prices; tier/latency illustrative)")
    rows = []
    for m in CATALOG:
        p = PRICING[m.id]
        rows.append([
            m.name, m.id, f"{m.tier:.0f}", f"${p['input']:.2f}/${p['output']:.2f}", f"${p['cache_read']:.2f}",
            f"{m.ttft_ms:.0f} ms", f"{m.context_window // 1000}K", m.default_effort, m.best_for,
        ])
    table(["model", "id", "tier", "in/out per 1M", "cache rd", "TTFT", "ctx", "def.effort", "best for"], rows)
    print("  Batch API: -50% on input+output. Haiku 5.5 prompts >100K tokens bill at $0.50/$2.50.")
    print("  Effort is a cost/depth dial on ONE model: thinking tokens bill as output tokens.")

    # ---------------------------------------------------------------- 2
    section("2. Models API lookup - capabilities at runtime")
    print("  Request code (anthropic Python SDK):")
    for line in MODELS_API_SNIPPET.rstrip().splitlines():
        print("    " + line)
    if is_live():
        models_api_lookup(client)
    else:
        print("  [live-only] The offline mock does not serve /v1/models; the catalog above is the")
        print("  cached table. In production, gate features on capabilities (e.g. effort.max,")
        print("  context_management.compact_20260112) instead of hard-coding model names.")

    # ---------------------------------------------------------------- 3
    section("3. Workload profiles -> constraint filter -> priced at volume")
    decisions: list[tuple[Workload, Evaluation]] = []
    for wl in workloads:
        feasible, all_evals = select(wl)
        reasons: dict[str, int] = {}
        for ev in all_evals:
            for r in ev.rejected:
                reasons[r] = reasons.get(r, 0) + 1
        sla = f"{wl.latency_sla_ms:,.0f} ms" if wl.latency_sla_ms is not None else "none"
        print(f"\n  [{wl.id}] {wl.description}")
        print(
            f"    volume={wl.requests_per_month:,}/mo  in={wl.avg_input_tokens:,} out={wl.avg_output_tokens:,} tok  "
            f"cacheable={wl.cacheable_input_fraction:.0%}  need={wl.capability_need:.1f}  SLA={sla}  "
            f"batch={'yes' if wl.batchable else 'no'}  turns={wl.agent_turns}"
        )
        rej = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
        print(f"    feasible configs: {len(feasible)} of {len(all_evals)}  (rejected: {rej or 'none'})")
        if not feasible:
            print("    NO FEASIBLE CONFIG - relax the SLA, decompose the task, or raise the budget.")
            continue
        best = feasible[0]
        decisions.append((wl, best))
        print(
            f"    RECOMMEND  {best.label:<20} {usd(best.monthly_usd)}/mo  ~{best.latency_ms:,.0f} ms  "
            f"headroom {best.headroom(wl):+.1f}"
        )
        alt = next((ev for ev in feasible[1:] if ev.model.id != best.model.id), None)
        if alt:
            print(
                f"    runner-up  {alt.label:<20} {usd(alt.monthly_usd)}/mo  ~{alt.latency_ms:,.0f} ms  "
                f"(+{(alt.monthly_usd / best.monthly_usd - 1):.0%} cost)"
            )
        if best.long_context_tier:
            print("    note: Haiku long-context price tier applied (prompt >100K tokens) - still cheapest here.")
        if wl.batchable:
            print("    note: no interactive SLA -> Message Batches API halves the bill.")
        # The interesting case: a MORE capable model at LOWER effort beats a cheaper
        # model that needs max effort to reach the bar.
        lower_tier = [ev for ev in feasible if ev.model.tier < best.model.tier]
        if lower_tier:
            cheapest_lower = min(lower_tier, key=lambda ev: ev.monthly_usd)
            print(
                f"    note: {best.label} beats {cheapest_lower.label} by "
                f"{usd(cheapest_lower.monthly_usd - best.monthly_usd)}/mo - the stronger model needs fewer thinking tokens."
            )

    # ---------------------------------------------------------------- 4
    section("4. Portfolio decision table vs 'Opus 5.5 @ medium for everything'")
    rows = []
    total_rec = total_base = 0.0
    base_failures = 0
    for wl, best in decisions:
        base = evaluate(BY_ID[MODEL_OPUS], wl, "medium")
        total_rec += best.monthly_usd
        total_base += base.monthly_usd
        base_failures += 0 if base.feasible else 1
        base_note = "ok" if base.feasible else "FAILS " + "+".join(base.rejected)
        rows.append([
            wl.id, best.model.name, best.effort, "yes" if wl.batchable else "no",
            usd(best.monthly_usd), usd(base.monthly_usd), base_note,
        ])
    table(["workload", "model", "effort", "batch", "$/mo (rec)", "$/mo (Opus med)", "Opus med vs constraints"], rows)
    print(f"  Portfolio: {usd(total_rec)}/mo right-sized vs {usd(total_base)}/mo single-model "
          f"({1 - total_rec / total_base:.0%} lower).")
    print(f"  The single-model baseline also violates {base_failures} of {len(decisions)} workloads' constraints:")
    print("  too slow for the real-time SLA, not capable enough for the hardest agents.")

    # ---------------------------------------------------------------- 5
    section("5. Capable model @ low effort vs cheaper model @ max effort")
    ev_cfg = raw["research_agent_eval"]
    wl = next(w for w in workloads if w.id == ev_cfg["workload_id"])
    fallback = float(ev_cfg["human_fallback_cost_usd"])
    print(f"  Workload: {wl.id}. {ev_cfg['_note']}")
    rows = []
    scored: list[tuple[str, float, float, float, float]] = []
    for c in ev_cfg["candidates"]:
        m = BY_ID[c["model"]]
        attempt = cost_per_request(m, wl, c["effort"])
        per_solved = attempt / c["pass_rate"]
        expected = attempt + (1 - c["pass_rate"]) * fallback
        label = f"{m.name} @ {c['effort']}"
        scored.append((label, attempt, c["pass_rate"], per_solved, expected))
        rows.append([label, usd(attempt), f"{c['pass_rate']:.0%}", usd(per_solved), usd(expected)])
    table(["config", "$/attempt", "pass rate", "$/solved task", f"$/task incl ${fallback:.0f} human fallback"], rows)
    cheapest_attempt = min(scored, key=lambda s: s[1])
    best_solved = min(scored, key=lambda s: s[3])
    best_expected = min(scored, key=lambda s: s[4])
    print(f"  Cheapest per attempt : {cheapest_attempt[0]}")
    print(f"  Cheapest per SOLVED  : {best_solved[0]}")
    print(f"  Cheapest incl. errors: {best_expected[0]}")
    print("  Lesson: per-token price lists do not rank models. Price the solved task (and the cost")
    print("  of a failure) on your own eval; sweep effort before switching models.")

    # ---------------------------------------------------------------- 6
    section("6. Routing cascade: Haiku triages difficulty -> route")
    print(f"  Routes: simple->Haiku 5.5@low, moderate->Sonnet 5.5@medium, complex->Opus 5.5@high;"
          f" confidence < {CONFIDENCE_FLOOR} escalates one tier.")
    rows = []
    routed_cost = base_cost = 0.0
    routed_lat = base_lat = 0.0
    correct = 0
    for r in cascade:
        t = triage(client, r["text"])
        level, model, effort, escalated = route(t)
        tri_cost = cost_usd(MODEL_HAIKU, t["_msg"].usage)
        msg, lat = answer(client, model, effort, r["text"])
        if msg.stop_reason != "end_turn":
            print(f"  {r['id']}: stop_reason={msg.stop_reason} - would retry on a stronger route")
        c = tri_cost + cost_usd(model, msg.usage)
        bmsg, blat = answer(client, MODEL_OPUS, "medium", r["text"])
        b = cost_usd(MODEL_OPUS, bmsg.usage)
        routed_cost += c
        base_cost += b
        routed_lat += lat
        base_lat += blat
        correct += t["difficulty"] == r["expected"]
        rows.append([
            r["id"], r["expected"], f"{t['difficulty']} ({t['confidence']:.2f})",
            BY_ID[model].name + "@" + effort + (" ^" if escalated else ""), usd(c), f"{lat:,.0f} ms", usd(b),
        ])
    table(["req", "expected", "triage (conf)", "routed to", "$ routed", "latency", "$ all-Opus"], rows)
    n = len(cascade)
    print(f"  Triage accuracy {correct}/{n}  ('^' = escalated for low confidence - safe direction)")
    print(f"  Routed total {usd(routed_cost)} (incl. triage) vs all-Opus {usd(base_cost)} -> "
          f"{1 - routed_cost / base_cost:.0%} cheaper; mean latency {routed_lat / n:,.0f} ms vs {base_lat / n:,.0f} ms")
    print("  Caveat: route BETWEEN conversations. Switching model inside one conversation")
    print("  cold-starts the prompt cache (caches are model-scoped) - spawn a sub-agent instead.")

    # ---------------------------------------------------------------- 7
    section("7. Same prompt, every model - measured latency and cost")
    # Warm-up call: the first request pays client/connection setup; exclude it from timings.
    client.messages.create(model=MODEL_HAIKU, max_tokens=64, messages=[{"role": "user", "content": "ping"}])
    rows = []
    for m in CATALOG:
        t0 = time.perf_counter()
        msg = client.messages.create(
            model=m.id,
            max_tokens=1024,
            system=PROBE_SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": PROBE_PROMPT}],
        )
        ms = elapsed_ms(t0)
        c = cost_usd(m.id, msg.usage)
        rows.append([
            m.name, f"{ms:,.0f} ms", str(msg.usage.input_tokens), str(msg.usage.output_tokens),
            usd(c), usd(c * 1_000_000),
        ])
    table(["model", "latency", "in tok", "out tok", "$/request", "$ per 1M requests"], rows)
    note = "real measurements" if is_live() else "offline latency is the shared mock's simulated profile"
    print(f"  ({note}; output text is identical, so the spread is pure model price/speed.)")

    section("Takeaways")
    print("  * Filter on hard constraints (quality bar, SLA, context) BEFORE comparing price.")
    print("  * Price at volume: tokens x price x requests, plus cache reads and Batch discounts.")
    print("  * Effort is the first dial; a stronger model at lower effort can be the cheapest config.")
    print("  * Cascades (cheap triage -> right-sized model) cut cost when most traffic is simple.")


if __name__ == "__main__":
    main()
