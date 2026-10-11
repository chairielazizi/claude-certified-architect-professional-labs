"""
D4.3 - A/B Testing and Iterative Improvement

1. Offline A/B: two prompt variants on the SAME eval set -> paired analysis
   (McNemar exact test + paired bootstrap CI).
2. Online A/B simulation: deterministic hash bucketing, sample-ratio-mismatch check,
   power / minimum sample size, two-proportion z-test with 95% CI, guardrail metrics.
3. A/A test: why peeking at p-values inflates false positives.
4. Iteration loop (hill-climb): Claude proposes prompt edits from TRAIN failures ->
   evaluate on VALIDATION -> keep or revert -> final report on held-out TEST.

Run from the repo root:
    .venv/Scripts/python "D4.3 - A-B Testing and Iterative Improvement/python/main.py"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU,
    MODEL_OPUS,
    MockRequest,
    banner,
    cost_usd,
    estimate_tokens,
    get_async_client,
    section,
    text_block,
)

DATA = Path(__file__).resolve().parents[1] / "data"
APP_MODEL = MODEL_HAIKU       # ticket router: classification at volume
ANALYZER_MODEL = MODEL_OPUS   # reads failures and proposes edits (a stronger model than the one under test)
CONCURRENCY = 16

# ---------------------------------------------------------------------------
# Deterministic helpers (identical algorithms in the TypeScript version)
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


def hash01(s: str) -> float:
    return mulberry32(fnv1a32(s))()


def fmt(x: float, d: int = 2) -> str:
    scaled = math.floor(x * 10**d + 0.5) / 10**d
    return f"{scaled:.{d}f}"


def pct(x: float) -> str:
    return fmt(100.0 * x, 1) + "%"


def signed_pct(x: float) -> str:
    return ("+" if x >= 0 else "") + fmt(100.0 * x, 1) + "%"


def fmt_p(p: float) -> str:
    return "<0.0001" if p < 0.0001 else fmt(p, 4)


def total(xs: Any) -> float:
    s = 0.0
    for x in xs:
        s += x
    return s


def mean(xs: list[float]) -> float:
    return total(xs) / len(xs) if xs else 0.0


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    rank = p / 100.0 * (len(s) - 1)
    lo = math.floor(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


# ---------------------------------------------------------------------------
# Statistics (implemented from scratch - no scipy)
# ---------------------------------------------------------------------------
def erf(x: float) -> float:
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)."""
    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return sign * y


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation of the inverse normal CDF."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """b = A right & B wrong, c = A wrong & B right. Returns (exact two-sided p, chi2-with-continuity p)."""
    n = b + c
    if n == 0:
        return 1.0, 1.0
    term = 0.5**n          # P(X = 0) for X ~ Binomial(n, 0.5), then iterate
    tail = 0.0
    for i in range(min(b, c) + 1):
        tail += term
        term = term * (n - i) / (i + 1)
    exact = min(1.0, 2.0 * tail)
    chi2 = (abs(b - c) - 1) ** 2 / n
    return exact, 1.0 - erf(math.sqrt(chi2 / 2.0))


def paired_bootstrap(a: list[int], b: list[int], iters: int, seed: int) -> tuple[float, float]:
    """95% percentile CI for mean(b) - mean(a), resampling CASES (keeps the pairing)."""
    rng = mulberry32(seed)
    n = len(a)
    diffs: list[float] = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            i = math.floor(rng() * n)
            s += b[i] - a[i]
        diffs.append(s / n)
    return percentile(diffs, 2.5), percentile(diffs, 97.5)


def two_prop_ztest(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float, float, float]:
    """Returns (diff p2-p1, z, two-sided p, 95% CI half-width). Pooled SE for the test, unpooled for the CI."""
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se_pool = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool if se_pool > 0 else 0.0
    p = 2 * (1 - norm_cdf(abs(z)))
    half = 1.959964 * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p2 - p1, z, p, half


def sample_size_per_arm(p1: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    p2 = p1 + mde
    pbar = (p1 + p2) / 2
    za, zb = norm_ppf(1 - alpha / 2), norm_ppf(power)
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / mde**2
    return math.ceil(n)


# ---------------------------------------------------------------------------
# Dataset, prompts, offline responder
# ---------------------------------------------------------------------------
SPEC = json.loads((DATA / "ticket_templates.json").read_text(encoding="utf-8"))
LABELS: list[str] = SPEC["labels"]
TICKETS: list[dict[str, Any]] = []
for ti, tpl in enumerate(SPEC["templates"]):
    for fi, filler in enumerate(tpl["fillers"]):
        TICKETS.append({"id": f"{tpl['label'][:4]}-{ti:02d}-{fi}", "label": tpl["label"], "difficulty": tpl["difficulty"],
                        "text": tpl["text"].replace("{x}", filler)})
BY_TEXT = {t["text"]: t for t in TICKETS}

# Stable stratified split: within each label, order by hash and deal round-robin.
SPLIT: dict[str, str] = {}
for lab in LABELS:
    group = sorted((t for t in TICKETS if t["label"] == lab), key=lambda t: fnv1a32("split:" + t["id"]))
    for i, t in enumerate(group):
        SPLIT[t["id"]] = ("train", "validation", "test")[i % 3]

ROLE = "You route customer support tickets to exactly one team: billing, shipping, technical, account, sales."
FORMAT = "Reply in the format:\nLabel: <team>\nReason: <one sentence>"
DEFS = (
    "Label definitions:\n- billing: charges, invoices, refunds, payment methods, fees (including delivery surcharges).\n"
    "- shipping: physical delivery status, addresses, couriers.\n- technical: errors, crashes, outages, broken pages (even the billing page).\n"
    "- account: sign-in identity, security, members, ownership, profile data.\n- sales: pricing questions, discounts, demos, contracts, renewals.\n"
    "Tie-break: route by the action the customer needs, not by the words they use."
)
PROMPT_A = ROLE + "\n" + FORMAT
PROMPT_B = PROMPT_A + "\n" + DEFS

# Offline "model": P(correct) depends on difficulty and on which prompt blocks are present.
BASE_P = {"easy": 0.75, "hard": 0.38}
EFFECTS = {"defs": (0.045, 0.20), "fewshot": (0.03, 0.10)}
MARKERS = {"defs": "Label definitions:", "fewshot": "Examples:", "label_only": "Do not add a reason",
           "overfit": "Known tickets:", "fluff": "Think carefully"}


def features_of(system: str) -> list[str]:
    return sorted(f for f, m in MARKERS.items() if m in system)


def mock_label(ticket: dict[str, Any], system: str) -> str:
    feats = features_of(system)
    if "overfit" in feats and ticket["text"] in system:
        return ticket["label"]  # memorised from the prompt - right on train, no help anywhere else
    hard = ticket["difficulty"] == "hard"
    quality = [f for f in feats if f in EFFECTS]  # only blocks that change the reasoning perturb outcomes
    p = BASE_P[ticket["difficulty"]] + total(EFFECTS[f][1 if hard else 0] for f in quality)
    # 70% shared per-case difficulty (paired designs exploit this) + 30% variant-specific noise
    u = 0.7 * hash01("case:" + ticket["id"]) + 0.3 * hash01("var:" + ticket["id"] + ":" + ",".join(quality))
    if u < p:
        return ticket["label"]
    gold = LABELS.index(ticket["label"])
    return LABELS[(gold + 1 + fnv1a32("wrong:" + ticket["id"]) % 4) % 5]


def analyzer_reply(prompt: str) -> str:
    failing = re.findall(r"^- \[[^\]]+\] (.+?) \| predicted=\w+ \| gold=(\w+)$", prompt, re.M)
    known = "; ".join(f"'{t}' -> {g}" for t, g in failing[:4])
    return json.dumps({"candidates": [
        {"name": "patch the failing tickets", "rationale": "Hard-code the exact train tickets that failed.",
         "block": "Known tickets: " + known},
        {"name": "few-shot examples", "rationale": "Hard tickets use words from another team; show three boundary examples.",
         "block": "Examples:\nTicket: I see a handling fee on my statement. -> billing\nTicket: The invoices page shows error 504. -> technical\nTicket: We want to renegotiate our renewal. -> sales"},
        {"name": "label-only output", "rationale": "Reasons are never read downstream; dropping them cuts output tokens and latency.",
         "block": "Output override: respond with 'Label: <team>' only. Do not add a reason."},
        {"name": "accuracy reminder", "rationale": "Remind the model to be careful.",
         "block": "Think carefully and be accurate."},
    ]}, separators=(",", ":"))


def responder(req: MockRequest) -> Any:
    if req.system_text.startswith("You improve prompts"):
        text = analyzer_reply(req.last_user_text)
        return {"content": [text_block(text)], "usage": {"output_tokens": estimate_tokens(text)}}
    m = re.search(r"Ticket: (.+)$", req.last_user_text, re.M)
    ticket = BY_TEXT.get(m.group(1).strip()) if m else None
    if ticket is None:
        return "Label: technical\nReason: unclear ticket."
    label = mock_label(ticket, req.system_text)
    feats = features_of(req.system_text)
    if "label_only" in feats:
        text = f"Label: {label}"
    elif "defs" in feats:
        text = (f"Label: {label}\nReason: Per the label definitions, the customer needs an action that the {label} team owns; "
                "the ticket wording may mention other areas, such as fees, apps or delivery, but the requested action decides the route.")
    else:
        text = f"Label: {label}\nReason: The customer describes an issue that matches the {label} team based on the main request."
    return {"content": [text_block(text)], "usage": {"output_tokens": estimate_tokens(text)}}


class Candidate(BaseModel):
    name: str
    rationale: str
    block: str


class CandidateList(BaseModel):
    candidates: list[Candidate]


# ---------------------------------------------------------------------------
# Running a prompt variant over a set of tickets
# ---------------------------------------------------------------------------
@dataclass
class Outcome:
    ticket_id: str
    correct: int
    output_tokens: int
    cost: float


async def evaluate(client: Any, sem: asyncio.Semaphore, system: str, tickets: list[dict[str, Any]]) -> list[Outcome]:
    async def one(t: dict[str, Any]) -> Outcome:
        async with sem:
            msg = await client.messages.create(model=APP_MODEL, max_tokens=256, system=system,
                                               output_config={"effort": "low"},
                                               messages=[{"role": "user", "content": "Ticket: " + t["text"]}])
        text = "".join(b.text for b in msg.content if b.type == "text").lower()
        m = re.search(r"label:\s*([a-z]+)", text)
        return Outcome(t["id"], 1 if m and m.group(1) == t["label"] else 0, msg.usage.output_tokens, cost_usd(msg.model, msg.usage))

    return list(await asyncio.gather(*(one(t) for t in tickets)))


def acc(outs: list[Outcome]) -> float:
    return sum(o.correct for o in outs) / len(outs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main_async() -> None:
    banner("D4.3 - A/B Testing and Iterative Improvement")
    client = get_async_client(responder)
    sem = asyncio.Semaphore(CONCURRENCY)
    train = [t for t in TICKETS if SPLIT[t["id"]] == "train"]
    val = [t for t in TICKETS if SPLIT[t["id"]] == "validation"]
    test = [t for t in TICKETS if SPLIT[t["id"]] == "test"]

    # -- 1 ------------------------------------------------------------------
    section("1. Setup: dataset, splits, variants")
    print(f"  {len(TICKETS)} routing tickets ({sum(1 for t in TICKETS if t['difficulty'] == 'hard')} hard), "
          f"split stratified by label: train={len(train)} validation={len(val)} test={len(test)}")
    print(f"  App: {APP_MODEL} (effort=low). Variant A = role + format. Variant B = A + label definitions & tie-break rule.")
    print(f"  Prompt size: A ~{estimate_tokens(PROMPT_A)} tokens, B ~{estimate_tokens(PROMPT_B)} tokens")

    # -- 2 ------------------------------------------------------------------
    section("2. Offline A/B on the same 120 cases (paired)")
    out_a = await evaluate(client, sem, PROMPT_A, TICKETS)
    out_b = await evaluate(client, sem, PROMPT_B, TICKETS)
    for diff in ("easy", "hard"):
        idx = [i for i, t in enumerate(TICKETS) if t["difficulty"] == diff]
        print(f"  {diff:<5} n={len(idx):<3} A={pct(mean([out_a[i].correct for i in idx]))}  B={pct(mean([out_b[i].correct for i in idx]))}")
    print(f"  all   n={len(TICKETS):<3} A={pct(acc(out_a))}  B={pct(acc(out_b))}  diff={signed_pct(acc(out_b) - acc(out_a))}")
    b_ = sum(1 for x, y in zip(out_a, out_b) if x.correct and not y.correct)
    c_ = sum(1 for x, y in zip(out_a, out_b) if not x.correct and y.correct)
    both = sum(1 for x, y in zip(out_a, out_b) if x.correct and y.correct)
    print(f"  paired table: both right={both}, A only={b_}, B only={c_}, both wrong={len(TICKETS) - both - b_ - c_}")
    p_exact, p_chi = mcnemar(b_, c_)
    lo, hi = paired_bootstrap([o.correct for o in out_a], [o.correct for o in out_b], 2000, 7)
    print(f"  McNemar (only discordant pairs matter): exact p={fmt_p(p_exact)}, chi2-corrected p={fmt_p(p_chi)}")
    print(f"  paired bootstrap 95% CI for B-A: [{signed_pct(lo)}, {signed_pct(hi)}] (2000 resamples of cases)")
    tok_a, tok_b = mean([o.output_tokens for o in out_a]), mean([o.output_tokens for o in out_b])
    cost_a, cost_b = mean([o.cost for o in out_a]), mean([o.cost for o in out_b])
    print(f"  output tokens/ticket A={fmt(tok_a, 1)} B={fmt(tok_b, 1)}; cost/ticket A=${fmt(cost_a, 7)} B=${fmt(cost_b, 7)}")
    print("  Decision: B is better offline" + (" (significant at 5%)." if p_exact < 0.05 else " but not significant - collect more cases.")
          + " Next: does it hold for real users?")

    # -- 3 ------------------------------------------------------------------
    section("3. Online A/B design: bucketing, sample-ratio check, power")
    experiment = "ticket-router-defs-v1"
    n_users = 8000

    def bucket(user: str) -> str:
        # Deterministic: same user -> same arm on every request and every server; salt by experiment
        # so different experiments get independent assignments.
        h = int(hashlib.sha256(f"{experiment}:{user}".encode()).hexdigest()[:8], 16)
        return "A" if h % 100 < 50 else "B"

    users = [f"u{i:05d}" for i in range(1, n_users + 1)]
    arm_of = {u: bucket(u) for u in users}
    n_a = sum(1 for u in users if arm_of[u] == "A")
    n_b = n_users - n_a
    chi2 = ((n_a - n_users / 2) ** 2 + (n_b - n_users / 2) ** 2) / (n_users / 2)
    srm_p = 1.0 - erf(math.sqrt(chi2 / 2.0))
    print(f"  sha256('{experiment}:<user>') % 100 < 50 -> A.  A={n_a} users, B={n_b} users")
    print(f"  sample-ratio-mismatch check: chi2={fmt(chi2, 3)} p={fmt_p(srm_p)} -> {'SRM! stop and debug assignment' if srm_p < 0.001 else 'OK (no SRM)'}")
    base_rate, mde = 0.70, 0.04
    need = sample_size_per_arm(base_rate, mde)
    print(f"  power analysis: baseline success {pct(base_rate)}, minimum detectable effect {signed_pct(mde)} abs, alpha=0.05, power=0.80")
    print(f"  -> need {need} users per arm; we have ~{min(n_a, n_b)} -> {'adequately powered' if min(n_a, n_b) >= need else 'UNDERPOWERED'}")

    # Measure each arm's real token/cost profile with a few calls through the shared client.
    prof_a = await evaluate(client, sem, PROMPT_A, train[:10])
    prof_b = await evaluate(client, sem, PROMPT_B, train[:10])
    profile = {
        "A": (mean([o.output_tokens for o in prof_a]), mean([o.cost for o in prof_a])),
        "B": (mean([o.output_tokens for o in prof_b]), mean([o.cost for o in prof_b])),
    }
    print(f"  measured per-call profile (10 calls/arm): A {fmt(profile['A'][0], 1)} out tok ${fmt(profile['A'][1], 7)}, "
          f"B {fmt(profile['B'][0], 1)} out tok ${fmt(profile['B'][1], 7)}")

    # -- 4 ------------------------------------------------------------------
    section("4. Online results (simulated users) and guardrails")
    true_success = {"A": 0.70, "B": 0.75}  # the online lift is smaller than offline - typical
    stats: dict[str, dict[str, list[float]]] = {"A": {"success": [], "thumbs": [], "safety": [], "latency": [], "cost": []},
                                                 "B": {"success": [], "thumbs": [], "safety": [], "latency": [], "cost": []}}
    for u in users:
        arm = arm_of[u]
        rng = mulberry32(fnv1a32("user:" + u))
        success = 1.0 if rng() < true_success[arm] else 0.0
        thumbs = 1.0 if rng() < (0.55 if success else 0.10) else 0.0
        safety = 1.0 if rng() < 0.003 else 0.0
        out_tok, call_cost = profile[arm]
        # latency: Haiku profile (250 ms TTFT + 3 ms/token), +-30% jitter, 5% slow tail
        lat = (250.0 + 3.0 * out_tok) * (0.7 + 0.6 * rng()) * (2.0 if rng() < 0.05 else 1.0)
        cost = call_cost * (0.8 + 0.4 * rng())
        for k, v in (("success", success), ("thumbs", thumbs), ("safety", safety), ("latency", lat), ("cost", cost)):
            stats[arm][k].append(v)

    def rate(arm: str, k: str) -> float:
        return mean(stats[arm][k])

    print(f"  {'metric':<26}{'A':>12}{'B':>12}{'B vs A':>12}")
    rows = [("task success rate", "success", pct), ("thumbs-up rate", "thumbs", pct), ("safety incident rate", "safety", lambda v: fmt(100 * v, 2) + "%")]
    for name, k, f in rows:
        print(f"  {name:<26}{f(rate('A', k)):>12}{f(rate('B', k)):>12}{signed_pct(rate('B', k) - rate('A', k)):>12}")
    p95 = {arm: percentile(stats[arm]["latency"], 95) for arm in ("A", "B")}
    cps = {arm: total(stats[arm]["cost"]) / total(stats[arm]["success"]) for arm in ("A", "B")}
    print(f"  {'p95 latency (ms)':<26}{fmt(p95['A'], 0):>12}{fmt(p95['B'], 0):>12}{signed_pct(p95['B'] / p95['A'] - 1):>12}")
    print(f"  {'cost per successful task':<26}{'$' + fmt(cps['A'], 7):>12}{'$' + fmt(cps['B'], 7):>12}{signed_pct(cps['B'] / cps['A'] - 1):>12}")
    xa, xb = int(total(stats["A"]["success"])), int(total(stats["B"]["success"]))
    diff, z, p, half = two_prop_ztest(xa, n_a, xb, n_b)
    print(f"\n  primary metric (task success): diff={signed_pct(diff)}  z={fmt(z, 2)}  p={fmt_p(p)}  95% CI [{signed_pct(diff - half)}, {signed_pct(diff + half)}]")
    ta, tb = int(total(stats["A"]["thumbs"])), int(total(stats["B"]["thumbs"]))
    d2, z2, p2, h2 = two_prop_ztest(ta, n_a, tb, n_b)
    print(f"  secondary (thumbs-up):         diff={signed_pct(d2)}  z={fmt(z2, 2)}  p={fmt_p(p2)}  95% CI [{signed_pct(d2 - h2)}, {signed_pct(d2 + h2)}]")
    sa, sb = int(total(stats["A"]["safety"])), int(total(stats["B"]["safety"]))
    _, zs, _, _ = two_prop_ztest(sa, n_a, sb, n_b)
    safety_p_one_sided = 1 - norm_cdf(zs)
    guardrails = [
        ("p95 latency regression <= +10%", p95["B"] / p95["A"] - 1 <= 0.10, signed_pct(p95["B"] / p95["A"] - 1)),
        ("cost per success regression <= +15%", cps["B"] / cps["A"] - 1 <= 0.15, signed_pct(cps["B"] / cps["A"] - 1)),
        ("safety incidents not significantly higher", safety_p_one_sided >= 0.05, f"one-sided p={fmt_p(safety_p_one_sided)}"),
    ]
    print("  guardrails (must not regress, even if the primary metric wins):")
    for name, ok, observed in guardrails:
        print(f"    [{'OK  ' if ok else 'FAIL'}] {name:<42} observed {observed}")
    primary_win = p < 0.05 and diff > 0
    all_ok = all(ok for _, ok, _ in guardrails)
    decision = "SHIP B" if primary_win and all_ok else ("DO NOT SHIP - primary metric wins but guardrails fail; iterate" if primary_win else "DO NOT SHIP - no significant win")
    print(f"  DECISION: {decision}")

    # -- 5 ------------------------------------------------------------------
    section("5. A/A tests: fixed horizon vs peeking")
    experiments, per_arm, looks = 200, 1000, 10
    fp_fixed = fp_peek = 0
    for e in range(experiments):
        rng = mulberry32(1000 + e)
        xa_ = xb_ = 0
        peeked = False
        for i in range(1, per_arm + 1):
            xa_ += 1 if rng() < 0.70 else 0
            xb_ += 1 if rng() < 0.70 else 0
            if i % (per_arm // looks) == 0:
                _, _, pv, _ = two_prop_ztest(xa_, i, xb_, i)
                if pv < 0.05:
                    peeked = True
                if i == per_arm and pv < 0.05:
                    fp_fixed += 1
        fp_peek += 1 if peeked else 0
    print(f"  {experiments} simulated A/A experiments (no real difference), {per_arm} users per arm")
    print(f"  false-positive rate, test once at the planned sample size: {pct(fp_fixed / experiments)} (expected ~5%, +/-3% at 200 sims)")
    print(f"  false-positive rate, stop at the first of {looks} peeks with p<0.05: {pct(fp_peek / experiments)}")
    print("  Fix: pre-register the sample size and test once, or use a sequential method (alpha spending / SPRT).")

    # -- 6 ------------------------------------------------------------------
    section("6. Iteration loop: propose -> evaluate on validation -> keep or revert")
    current = PROMPT_B
    cur_train = await evaluate(client, sem, current, train)
    cur_val = await evaluate(client, sem, current, val)
    failing = [(t, o) for t, o in zip(train, cur_train) if not o.correct]
    analysis_prompt = "Current prompt:\n" + current + "\n\nFailing TRAIN tickets:\n" + "\n".join(
        f"- [{t['id']}] {t['text']} | predicted=wrong | gold={t['label']}" for t, _ in failing
    ) + "\n\nPropose independent candidate edits as prompt blocks."

    proposal = await client.messages.parse(model=ANALYZER_MODEL, max_tokens=2048,
                                           system="You improve prompts for a ticket router. Only TRAIN failures are shown to you.",
                                           messages=[{"role": "user", "content": analysis_prompt}], output_format=CandidateList)
    cands = proposal.parsed_output.candidates if proposal.parsed_output else []
    print(f"  start: variant B  train={pct(acc(cur_train))} validation={pct(acc(cur_val))} out_tok={fmt(mean([o.output_tokens for o in cur_val]), 1)}")
    print(f"  analyzer ({ANALYZER_MODEL}) saw {len(failing)} train failures and proposed {len(cands)} edits")
    print(f"  keep rule: validation +2 cases or more with output tokens <= 1.2x, OR validation not worse with output tokens <= 0.7x")
    print(f"  {'#':<3}{'candidate':<27}{'train':>8}{'valid':>8}{'d_val':>7}{'out_tok':>9}  decision")
    for i, cand in enumerate(cands, start=1):
        trial = current + "\n" + cand.block
        tr = await evaluate(client, sem, trial, train)
        va = await evaluate(client, sem, trial, val)
        d_val = sum(o.correct for o in va) - sum(o.correct for o in cur_val)
        tok_ratio = mean([o.output_tokens for o in va]) / mean([o.output_tokens for o in cur_val])
        keep = (d_val >= 2 and tok_ratio <= 1.2) or (d_val >= 0 and tok_ratio <= 0.7)
        if keep:
            why = "KEEP - quality gain" if d_val >= 2 else "KEEP - same quality, far fewer tokens"
        elif acc(tr) - acc(cur_train) >= 0.05 and d_val < 2:
            why = "REVERT - train up, validation flat: overfit"
        else:
            why = "REVERT - no measurable gain"
        print(f"  {i:<3}{cand.name:<27}{pct(acc(tr)):>8}{pct(acc(va)):>8}{d_val:>+7d}{fmt(mean([o.output_tokens for o in va]), 1):>9}  {why}")
        if keep:
            current, cur_train, cur_val = trial, tr, va
    print("  (n=40 per split: a 2-case threshold is illustrative; real loops need more cases or reps to clear noise.)")

    # -- 7 ------------------------------------------------------------------
    section("7. Final report on the held-out TEST split (never used for decisions)")
    final_variants = [("A (baseline)", PROMPT_A), ("B (defs)", PROMPT_B), ("C (hill-climbed)", current)]
    test_out = {name: await evaluate(client, sem, prompt, test) for name, prompt in final_variants}
    print(f"  {'variant':<18}{'test acc':>10}{'out_tok':>9}{'cost/ticket':>13}{'latency*':>10}")
    for name, _ in final_variants:
        o = test_out[name]
        lat = 250.0 + 3.0 * mean([x.output_tokens for x in o])
        print(f"  {name:<18}{pct(acc(o)):>10}{fmt(mean([x.output_tokens for x in o]), 1):>9}{'$' + fmt(mean([x.cost for x in o]), 7):>13}{fmt(lat, 0):>10}")
    a_c = [x.correct for x in test_out["A (baseline)"]]
    c_c = [x.correct for x in test_out["C (hill-climbed)"]]
    lo, hi = paired_bootstrap(a_c, c_c, 2000, 11)
    bb = sum(1 for x, y in zip(a_c, c_c) if x and not y)
    cc = sum(1 for x, y in zip(a_c, c_c) if not x and y)
    pe, _ = mcnemar(bb, cc)
    print("  * modeled latency = 250 ms + 3 ms per output token (Haiku profile); use measured wall-clock in live runs")
    print(f"  C vs A on test: {signed_pct(mean(c_c) - mean(a_c))}, paired 95% CI [{signed_pct(lo)}, {signed_pct(hi)}], McNemar exact p={fmt_p(pe)}")
    print("  Recommendation: run an online A/B of C vs A - C keeps B's quality gain without B's latency and cost regressions.")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
