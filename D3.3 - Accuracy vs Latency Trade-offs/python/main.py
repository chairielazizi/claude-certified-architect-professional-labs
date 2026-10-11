"""
D3.3 - Accuracy vs Latency Trade-offs
Exam objective: Evaluate accuracy-latency trade-offs and justify configuration decisions.

What this lab builds
--------------------
1. A configuration sweep over a labeled eval set (travel-policy Q&A with a BM25 retriever):
       model (Haiku / Sonnet / Opus) x effort (low / medium / high) x retrieval top-k (3 / 8)
   Every cell streams its answers through the real SDK, so each request yields
   accuracy (code grader), total latency, time-to-first-token (TTFT) and cost.
   Streaming on/off is the fourth dimension: it changes PERCEIVED latency (TTFT vs total),
   never accuracy or cost.
2. SLA-driven selection: the cheapest configuration that meets each scenario's gates,
   a Pareto frontier, and a written decision record.
3. Latency levers that do not cost accuracy: streaming, parallel tool calls, prompt
   caching, prompt trimming.

Latency source: LIVE -> measured wall clock (elapsed_ms, first text delta).
OFFLINE -> a transparent latency model fed by each response's real `usage`
(TTFT + prefill x uncached input + decode x thinking/visible tokens). The model
mirrors the shared mock's per-model speeds and keeps decisions deterministic even
when the machine is busy.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    LATENCY_SCALE,
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
    is_live,
    section,
    text_block,
    text_of,
    tool_use_block,
)

DATA = Path(__file__).resolve().parents[1] / "data"
CORPUS: dict[str, str] = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
EVAL: list[dict[str, Any]] = json.loads((DATA / "eval_set.json").read_text(encoding="utf-8"))

MODELS = [MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS]
EFFORTS = ["low", "medium", "high"]
TOP_KS = [3, 8]
SHORT = {MODEL_HAIKU: "haiku", MODEL_SONNET: "sonnet", MODEL_OPUS: "opus"}
SYSTEM = (
    "You answer employee travel-policy questions using ONLY the policy excerpts provided. "
    "Reply in exactly two lines: 'ANSWER: <short answer>' then 'REASON: <one sentence citing excerpt ids>'."
)

# (base TTFT ms, prefill ms per uncached input token, decode ms per output token)
LATENCY_MODEL = {
    MODEL_HAIKU: (250.0, 0.04, 3.0),
    MODEL_SONNET: (500.0, 0.08, 7.0),
    MODEL_OPUS: (900.0, 0.12, 12.0),
}
CACHED_PREFILL_FACTOR = 0.1  # cache reads skip most prefill work


# ---------------------------------------------------------------------------
# Retrieval: a small BM25 index (top-k is an accuracy AND a cost/latency knob)
# ---------------------------------------------------------------------------
STOP = set(
    "a an the and or of to for in on at is are be by with my i do does what which who how much many must "
    "can under per each all any from than after was had have has it this that as".split()
)


def tokenize(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
    return out


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = {d: tokenize(t) for d, t in docs.items()}
        self.k1, self.b = k1, b
        self.avgdl = sum(len(t) for t in self.docs.values()) / len(self.docs)
        self.df: dict[str, int] = {}
        for toks in self.docs.values():
            for w in set(toks):
                self.df[w] = self.df.get(w, 0) + 1

    def search(self, query: str, k: int) -> list[str]:
        n = len(self.docs)
        q = sorted(set(tokenize(query)))
        scored = []
        for doc_id, toks in self.docs.items():
            s = 0.0
            for w in q:
                f = toks.count(w)
                if f:
                    idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl))
            scored.append((-s, doc_id))
        scored.sort()
        return [d for _, d in scored[:k]]


def build_prompt(context_ids: list[str], question: str) -> str:
    excerpts = "\n".join(f"[{d}] {CORPUS[d]}" for d in context_ids)
    return f"Policy excerpts:\n{excerpts}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Latency model, grader, percentiles
# ---------------------------------------------------------------------------
def modelled_latency(model: str, usage: Any) -> tuple[float, float]:
    base, prefill, decode = LATENCY_MODEL[model]
    details = getattr(usage, "output_tokens_details", None)
    thinking = details.thinking_tokens if details else 0
    uncached = usage.input_tokens + (usage.cache_creation_input_tokens or 0)
    cached = usage.cache_read_input_tokens or 0
    ttft = base + prefill * uncached + prefill * CACHED_PREFILL_FACTOR * cached + decode * thinking
    return ttft, ttft + decode * (usage.output_tokens - thinking)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace(",", "")).strip(" .")


def grade(answer_text: str, gold: str) -> bool:
    m = re.search(r"ANSWER:\s*(.+)", answer_text)
    return bool(m) and normalize(m.group(1)) == normalize(gold)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (with n=10, p95 is the slowest request)."""
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


# ---------------------------------------------------------------------------
# Offline responder: capability grows with model tier and effort
# ---------------------------------------------------------------------------
CAPABILITY = {MODEL_HAIKU: 1.0, MODEL_SONNET: 2.0, MODEL_OPUS: 3.0}
EFFORT_BONUS = {"low": 0.0, "medium": 0.5, "high": 1.0}
THINK_PER_SKILL = {"low": 10, "medium": 40, "high": 90}  # adaptive thinking scales with difficulty


def simulate_answer(model: str, effort: str, item: dict[str, Any], context: list[str]) -> tuple[str, str, int]:
    skill = CAPABILITY.get(model, 3.0) + EFFORT_BONUS.get(effort, 0.5)
    missing = [d for d in item["needs"] if d not in context]
    if model == MODEL_HAIKU and "D04" in context and "D03" in item["needs"]:
        # Smaller models are likelier to be misled by a superseded document in a big context.
        answer, reason = item["distractor_wrong"], "D04 lists the applicable rate."
    elif missing:
        answer, reason = item["wrong"], "The excerpts do not fully settle this; best reading of the text."
    elif skill < item["required_skill"]:
        answer, reason = item["wrong"], "Based on a direct reading of the excerpts."
    else:
        answer, reason = item["gold"], item["rationale"]
    thinking = 0 if (effort == "low" and item["required_skill"] <= 1.0) else int(THINK_PER_SKILL.get(effort, 40) * item["required_skill"])
    return answer, reason, thinking


def sweep_responder(req: MockRequest) -> Any:
    user = req.last_user_text
    question = user.split("Question: ", 1)[-1].strip()
    item = next((it for it in EVAL if it["question"] == question), None)
    if item is None:
        return "ANSWER: unknown\nREASON: no matching policy."
    context = re.findall(r"^\[(D\d\d)\]", user, re.M)
    effort = (req.body.get("output_config") or {}).get("effort", "medium")
    answer, reason, thinking = simulate_answer(req.model, effort, item, context)
    text = f"ANSWER: {answer}\nREASON: {reason}"
    visible = estimate_tokens(text)
    return {"content": [text_block(text)],
            "usage": {"output_tokens": visible + thinking, "output_tokens_details": {"thinking_tokens": thinking}}}


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    model: str
    effort: str
    k: int

    def label(self) -> str:
        return f"{SHORT[self.model]}/{self.effort}/k{self.k}"


@dataclass
class CallResult:
    item_id: str
    correct: bool
    ttft_ms: float
    total_ms: float
    cost: float
    stop_reason: str


@dataclass
class Cell:
    cfg: Config
    acc: float
    p50: float
    p95: float
    p95_ttft: float
    cost_per_1k: float
    failures: list[str]


async def run_one(client: Any, sem: asyncio.Semaphore, bm25: BM25, cfg: Config, item: dict[str, Any]) -> CallResult:
    context = bm25.search(item["question"], cfg.k)
    async with sem:
        t0 = time.perf_counter()
        first: float | None = None
        async with client.messages.stream(
            model=cfg.model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            output_config={"effort": cfg.effort},
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(context, item["question"])}],
        ) as stream:
            async for _ in stream.text_stream:
                if first is None:
                    first = elapsed_ms(t0)  # time to first visible token
            msg = await stream.get_final_message()
        wall = elapsed_ms(t0)
    ttft, total = (first or wall, wall) if is_live() else modelled_latency(cfg.model, msg.usage)
    ok = msg.stop_reason == "end_turn" and grade(text_of(msg), item["gold"])
    return CallResult(item["id"], ok, ttft, total, cost_usd(cfg.model, msg.usage), msg.stop_reason or "")


async def run_sweep(client: Any, bm25: BM25) -> list[Cell]:
    sem = asyncio.Semaphore(4 if is_live() else 16)  # bounded concurrency respects rate limits
    configs = [Config(m, e, k) for m in MODELS for e in EFFORTS for k in TOP_KS]
    results = await asyncio.gather(*[run_one(client, sem, bm25, c, it) for c in configs for it in EVAL])
    cells = []
    n = len(EVAL)
    for i, cfg in enumerate(configs):
        rs = results[i * n:(i + 1) * n]
        cells.append(Cell(
            cfg,
            acc=sum(r.correct for r in rs) / n,
            p50=percentile([r.total_ms for r in rs], 50),
            p95=percentile([r.total_ms for r in rs], 95),
            p95_ttft=percentile([r.ttft_ms for r in rs], 95),
            cost_per_1k=sum(r.cost for r in rs) / n * 1000,
            failures=[r.item_id for r in rs if not r.correct],
        ))
    return cells


# ---------------------------------------------------------------------------
# SLA selection + Pareto frontier
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    gates: str
    min_acc: float
    max_ms: float | None
    metric: str  # "total" (stream off / machine consumer) or "ttft" (stream on / human reader)


SCENARIOS = [
    Scenario("S1 Internal API (needs full JSON)", "p95 total <= 2000 ms, acc >= 90%", 0.90, 2000, "total"),
    Scenario("S2 Chat UI, streaming on", "p95 TTFT <= 2800 ms, acc >= 95%", 0.95, 2800, "ttft"),
    Scenario("S2b Same chat UI, streaming off", "p95 total <= 2800 ms, acc >= 95%", 0.95, 2800, "total"),
    Scenario("S3 Nightly bulk re-check", "no latency gate, acc >= 70%", 0.70, None, "total"),
    Scenario("S4 High-volume triage", "p95 total <= 1000 ms, acc >= 50%", 0.50, 1000, "total"),
]


def pick(cells: list[Cell], sc: Scenario) -> Cell | None:
    def latency(c: Cell) -> float:
        return c.p95_ttft if sc.metric == "ttft" else c.p95

    ok = [c for c in cells if c.acc >= sc.min_acc - 1e-9 and (sc.max_ms is None or latency(c) <= sc.max_ms)]
    # Cheapest that meets every gate; ties -> more accurate, then faster.
    return min(ok, key=lambda c: (c.cost_per_1k, -c.acc, latency(c))) if ok else None


def pareto(cells: list[Cell]) -> list[Cell]:
    def dominates(a: Cell, b: Cell) -> bool:
        no_worse = a.acc >= b.acc and a.p95 <= b.p95 and a.cost_per_1k <= b.cost_per_1k
        better = a.acc > b.acc or a.p95 < b.p95 or a.cost_per_1k < b.cost_per_1k
        return no_worse and better

    front = [c for c in cells if not any(dominates(o, c) for o in cells if o is not c)]
    return sorted(front, key=lambda c: (c.cost_per_1k, -c.acc))


# ---------------------------------------------------------------------------
# Latency levers that do not cost accuracy
# ---------------------------------------------------------------------------
CITIES = ["London", "Tokyo", "Chicago"]
CITY_CAPS = {"London": "GBP 260", "Tokyo": "JPY 32000", "Chicago": "USD 280"}
TOOL_MS = 300.0  # simulated downstream API latency per tool call
CAP_TOOL = {
    "name": "get_city_cap",
    "description": "Return the current hotel nightly cap for one city. Call once per city needed.",
    "strict": True,
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"], "additionalProperties": False},
}


def tools_responder(req: MockRequest) -> Any:
    requested = [b["input"]["city"] for m in req.messages if m.get("role") == "assistant"
                 for b in (m["content"] if isinstance(m["content"], list) else []) if b.get("type") == "tool_use"]
    remaining = [c for c in CITIES if c not in requested]
    if not remaining:
        return "London GBP 260, Tokyo JPY 32000, Chicago USD 280 per night (2026 policy)."
    one_at_a_time = bool((req.body.get("tool_choice") or {}).get("disable_parallel_tool_use"))
    batch = remaining[:1] if one_at_a_time else remaining
    return {"content": [tool_use_block("get_city_cap", {"city": c}) for c in batch],
            "usage": {"output_tokens": 25 * len(batch)}}  # ~25 output tokens per tool call


async def get_city_cap(city: str) -> str:
    await asyncio.sleep(TOOL_MS / 1000 * (1.0 if is_live() else LATENCY_SCALE))
    return CITY_CAPS.get(city, "unknown")


async def tool_agent(client: Any, parallel: bool) -> tuple[int, float, float, float]:
    """Returns (model round trips, model ms, tool ms, total ms)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Compare the hotel caps for London, Tokyo and Chicago."}]
    choice: dict[str, Any] = {"type": "auto"} if parallel else {"type": "auto", "disable_parallel_tool_use": True}
    trips, model_ms, tool_ms = 0, 0.0, 0.0
    t0 = time.perf_counter()
    while True:
        t_call = time.perf_counter()
        resp = await client.messages.create(model=MODEL_SONNET, max_tokens=1024, tools=[CAP_TOOL],
                                            tool_choice=choice, messages=messages)
        trips += 1
        model_ms += elapsed_ms(t_call) if is_live() else modelled_latency(MODEL_SONNET, resp.usage)[1]
        if resp.stop_reason != "tool_use":
            break
        calls = [b for b in resp.content if b.type == "tool_use"]
        # Independent tool calls run concurrently; results return in ONE user message.
        outputs = await asyncio.gather(*[get_city_cap(b.input["city"]) for b in calls])
        tool_ms += TOOL_MS  # concurrent batch costs one tool latency, not len(calls)
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": b.id, "content": out} for b, out in zip(calls, outputs)]})
    total = elapsed_ms(t0) if is_live() else model_ms + tool_ms
    return trips, model_ms, tool_ms, total


def cache_responder(req: MockRequest) -> Any:
    return "ANSWER: GBP 260\nREASON: D03 sets the current London cap."


def handbook() -> str:
    examples = []
    for i in range(1, 31):
        cents = i * 17 * 67
        examples.append(f"Example {i}: {i * 17} miles of personal car use is reimbursed at USD {cents // 100}.{cents % 100:02d}.")
    policy = "\n".join(f"[{d}] {t}" for d, t in CORPUS.items())
    return f"Travel policy handbook (static, versioned).\n{policy}\nWorked mileage examples:\n" + "\n".join(examples)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def fmt_cell(c: Cell) -> str:
    return (f"{SHORT[c.cfg.model]:<7}{c.cfg.effort:<8}{c.cfg.k:>2}  {c.acc * 100:>4.0f}%  {c.p50:>6.0f}  {c.p95:>6.0f}"
            f"  {c.p95_ttft:>8.0f}  {c.cost_per_1k:>7.3f}  {','.join(c.failures) or '-'}")


async def main() -> None:
    banner("D3.3 - Accuracy vs Latency Trade-offs")
    bm25 = BM25(CORPUS)
    src = "measured wall clock" if is_live() else "latency model fed by real usage (deterministic)"

    section("1. Eval set and retrieval (BM25 top-k)")
    print(f"{len(EVAL)} labeled questions, {len(CORPUS)} policy excerpts. Needed docs vs retrieved:")
    for it in EVAL:
        k3, k8 = bm25.search(it["question"], 3), bm25.search(it["question"], 8)
        miss3 = [d for d in it["needs"] if d not in k3]
        flag = "k=3 MISSES " + ",".join(miss3) if miss3 else "k=3 ok"
        distractor = "  (superseded D04 in k=8)" if "D04" in k8 and "D04" not in k3 else ""
        print(f"  {it['id']} {it['category']:<12} needs {','.join(it['needs']):<12} {flag}{distractor}")

    section("2. Sweep: model x effort x top-k (180 streamed calls)")
    print(f"Latency source: {src}")
    client = get_async_client(sweep_responder)
    cells = await run_sweep(client, bm25)  # 180 requests, bounded concurrency
    print(f"{'MODEL':<7}{'EFFORT':<8}{'K':>2}  {'ACC':>5}  {'P50ms':>6}  {'P95ms':>6}  {'P95-TTFT':>8}  {'$/1k':>7}  WRONG")
    for c in cells:
        print(fmt_cell(c))
    print("Streaming on/off: same accuracy and cost per row; perceived latency = P95-TTFT (on) or P95ms (off).")

    section("3. SLA-driven selection (cheapest config that passes every gate)")
    winners: dict[str, Cell | None] = {}
    for sc in SCENARIOS:
        w = pick(cells, sc)
        winners[sc.name] = w
        if w:
            lat = w.p95_ttft if sc.metric == "ttft" else w.p95
            print(f"{sc.name:<36} [{sc.gates}]")
            print(f"{'':<4}-> {w.cfg.label():<18} acc {w.acc * 100:.0f}%  p95 {lat:.0f} ms  ${w.cost_per_1k:.3f}/1k")
        else:
            print(f"{sc.name:<36} [{sc.gates}]")
            print(f"{'':<4}-> NO configuration qualifies")

    section("4. Pareto frontier (accuracy up, p95 down, cost down)")
    for c in pareto(cells):
        print(f"  {c.cfg.label():<18} acc {c.acc * 100:>3.0f}%  p95 {c.p95:>5.0f} ms  ${c.cost_per_1k:.3f}/1k")
    print("Everything off the frontier is dominated: another config is at least as good on all three axes.")

    section("5. Latency levers that do not cost accuracy")
    # (a) streaming
    s1 = winners[SCENARIOS[0].name]
    assert s1 is not None
    hard = EVAL[-1]
    ctx = bm25.search(hard["question"], s1.cfg.k)
    t0 = time.perf_counter()
    first_chunk, first_ms = "", None
    async with client.messages.stream(model=s1.cfg.model, max_tokens=4096, thinking={"type": "adaptive"},
                                      output_config={"effort": s1.cfg.effort}, system=SYSTEM,
                                      messages=[{"role": "user", "content": build_prompt(ctx, hard["question"])}]) as stream:
        async for chunk in stream.text_stream:
            if first_ms is None:
                first_ms, first_chunk = elapsed_ms(t0), chunk
        final = await stream.get_final_message()
    ttft, total = (first_ms or 0.0, elapsed_ms(t0)) if is_live() else modelled_latency(s1.cfg.model, final.usage)
    print(f"(a) Streaming [{s1.cfg.label()}, {hard['id']}]: first text '{first_chunk.replace(chr(10), ' ').strip()}' at {ttft:.0f} ms;"
          f" complete at {total:.0f} ms -> user sees progress {total - ttft:.0f} ms sooner (more on long answers).")

    # (b) parallel tool calls
    tclient = get_async_client(tools_responder)
    seq = await tool_agent(tclient, parallel=False)
    par = await tool_agent(tclient, parallel=True)
    print("(b) Parallel tool calls (3 independent city lookups, 300 ms each):")
    print(f"    disable_parallel_tool_use=True : {seq[0]} model round trips, model {seq[1]:.0f} ms + tools {seq[2]:.0f} ms = {seq[3]:.0f} ms")
    print(f"    parallel (default)             : {par[0]} model round trips, model {par[1]:.0f} ms + tools {par[2]:.0f} ms = {par[3]:.0f} ms")

    # (c) prompt caching
    cclient = get_async_client(cache_responder)
    system_blocks = [{"type": "text", "text": handbook(), "cache_control": {"type": "ephemeral"}}]
    print("(c) Prompt caching of a static handbook prefix (same model, 2 sequential requests):")
    for i, q in enumerate([EVAL[2]["question"], EVAL[3]["question"]]):
        r = await cclient.messages.create(model=MODEL_SONNET, max_tokens=512, system=system_blocks,
                                          messages=[{"role": "user", "content": q}])
        t, tot = modelled_latency(MODEL_SONNET, r.usage)
        print(f"    request {i + 1}: cache_write={r.usage.cache_creation_input_tokens} cache_read={r.usage.cache_read_input_tokens}"
              f" uncached={r.usage.input_tokens}  TTFT ~{t:.0f} ms  cost ${cost_usd(MODEL_SONNET, r.usage):.5f}")

    # (d) prompt trimming
    verbose_system = (SYSTEM + " You are a friendly, professional, helpful assistant. Always be polite. "
                      "For completeness, here is the entire travel policy:\n" + "\n".join(CORPUS.values()))
    lean_prompt = build_prompt(bm25.search(hard["question"], 8), hard["question"])
    verbose_n = (await client.messages.count_tokens(model=MODEL_OPUS, system=verbose_system,
                                                    messages=[{"role": "user", "content": lean_prompt}])).input_tokens
    lean_n = (await client.messages.count_tokens(model=MODEL_OPUS, system=SYSTEM,
                                                 messages=[{"role": "user", "content": lean_prompt}])).input_tokens
    saved = verbose_n - lean_n
    print(f"(d) Prompt trimming: whole-policy system prompt {verbose_n} tokens vs retrieved-only {lean_n} tokens"
          f" -> {saved} fewer tokens, ~{saved * LATENCY_MODEL[MODEL_OPUS][1]:.0f} ms less prefill on Opus,"
          f" ${saved * PRICING[MODEL_OPUS]['input']:,.0f} less per 1M requests.")

    section("6. Decision record")
    s2 = winners[SCENARIOS[1].name]
    assert s2 is not None
    runner_up = sorted((c for c in cells if c.acc >= 0.90 and c is not s1), key=lambda c: c.p95)[0]
    print("ADR-D33-01  Travel-policy assistant: production configuration")
    print("Context   : two consumers - an internal API (needs the full answer) and an employee chat UI.")
    print(f"Decision  : API -> {s1.cfg.label()} (acc {s1.acc * 100:.0f}%, p95 {s1.p95:.0f} ms, ${s1.cost_per_1k:.3f}/1k).")
    print(f"            Chat UI -> {s2.cfg.label()} + streaming (acc {s2.acc * 100:.0f}%, p95 TTFT {s2.p95_ttft:.0f} ms).")
    print(f"Rationale : thinking tokens dominate latency; a larger model at LOW effort beat {runner_up.cfg.label()}")
    print(f"            (acc {runner_up.acc * 100:.0f}%, p95 {runner_up.p95:.0f} ms) on latency at equal accuracy. k=8 is required for")
    print("            multi-hop questions; streaming lets the UI afford medium effort for the remaining exception case.")
    print(f"Accepted  : API config misses {','.join(s1.failures)} (exception + date math). Mitigation: validator flags")
    print("            low-confidence answers and re-runs only those at higher effort, asynchronously.")
    print("Revisit if: weekly eval accuracy < 90%, p95 drifts > 10%, traffic mix shifts to multi-hop, or a new model ships.")

    section("Takeaways")
    print("* Decide against explicit SLAs: accuracy floor + latency metric (total vs TTFT) + cost - then pick the cheapest pass.")
    print("* Sweep effort before switching models; for small models, more context (top-k) can hurt via distractors.")
    print("* Streaming fixes PERCEIVED latency for humans only; machines waiting on full JSON gain nothing.")
    print("* Parallel tools, caching and trimming cut latency without touching accuracy - apply them first.")


if __name__ == "__main__":
    asyncio.run(main())
