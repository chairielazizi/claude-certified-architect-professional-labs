"""
D2.3 - Prompting Techniques: Zero-shot, Few-shot, Chain-of-Thought
==================================================================

One labelled task set, four techniques, one code grader:

  * Task A - classify support emails into a nuanced 6-label taxonomy with
    precedence rules (12 labelled emails, several deliberately ambiguous).
  * Task B - small multi-step arithmetic problems (5, exact numeric answers).

Techniques (same model - Claude Haiku 5.5 - so only the prompt changes):
  zero-shot  : instructions + taxonomy only
  few-shot   : + 18 balanced, diverse examples in <example> tags (cached prefix)
  cot        : + "brief analysis in <reasoning>, label in <answer>" (thinking off)
  thinking   : zero-shot prompt + adaptive thinking (display "summarized") -
               the model-native alternative to prompted chain-of-thought

The grader is real code: strict format check + lenient label/number
extraction, accuracy, tokens, cost and latency per technique. Also shown:
the `reasoning_extraction` refusal trap on Opus 5.5, and few-shot caching math.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU,
    MODEL_OPUS,
    PRICING,
    MockRequest,
    banner,
    cost_usd,
    elapsed_ms,
    get_client,
    section,
    text_of,
)

DATA = Path(__file__).resolve().parents[1] / "data"
TASKS: dict[str, Any] = json.loads((DATA / "tasks.json").read_text(encoding="utf-8"))
TAXONOMY: list[dict[str, str]] = TASKS["taxonomy"]
LABELS = [t["label"] for t in TAXONOMY]
SIM = TASKS["offline_simulation"]

TECHNIQUES = ["zero_shot", "few_shot", "cot", "thinking"]
NAMES = {"zero_shot": "zero-shot", "few_shot": "few-shot", "cot": "chain-of-thought", "thinking": "adaptive thinking"}

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
CLASSIFY_ROLE = (
    "You are a support email classifier for a B2B SaaS company. Assign exactly one label from the "
    "taxonomy to the email inside <email> tags. Read the definitions and rules carefully: several labels "
    "overlap and the rules decide between them."
)
MATH_ROLE = "You are a precise analyst for a SaaS operations team. Solve the problem inside <problem> tags."
LABEL_ONLY = "Respond with only the label, nothing else."
NUMBER_ONLY = "Respond with only the final number: no units, no words."
CLASSIFY_COT = (
    "Before answering, write a brief analysis inside <reasoning> tags: name the candidate labels and the rule "
    "that decides between them. Then output only the label inside <answer> tags."
)
MATH_COT = "Work through the problem step by step inside <reasoning> tags, then give only the final number inside <answer> tags."


def taxonomy_block() -> str:
    lines = ["<taxonomy>"] + [f"- {t['label']}: {t['definition']}" for t in TAXONOMY] + ["</taxonomy>", "<rules>"]
    lines += [f"{i}. {r}" for i, r in enumerate(TASKS["rules"], start=1)] + ["</rules>"]
    return "\n".join(lines)


def select_examples(pool: list[dict[str, str]], per_label: int) -> list[dict[str, str]]:
    """Balanced (same count per label) and interleaved (no label clustered at the end,
    which would bias the model toward the most recent label)."""
    by_label = {lbl: [e for e in pool if e["label"] == lbl][:per_label] for lbl in LABELS}
    picked: list[dict[str, str]] = []
    for i in range(per_label):
        for lbl in LABELS:
            if i < len(by_label[lbl]):
                picked.append(by_label[lbl][i])
    return picked


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / max(1, len(wa | wb))


def examples_block(examples: list[dict[str, str]]) -> str:
    parts = ["<examples>"]
    for e in examples:
        parts.append(f"<example>\n<email>{e['email']}</email>\n<label>{e['label']}</label>\n</example>")
    parts.append("</examples>")
    return "\n".join(parts)


def classify_system(technique: str, examples: list[dict[str, str]]) -> list[dict[str, Any]]:
    text = CLASSIFY_ROLE + "\n\n" + taxonomy_block()
    if technique == "few_shot":
        text += "\n\n" + examples_block(examples)
    text += "\n\n" + (CLASSIFY_COT if technique == "cot" else LABEL_ONLY)
    # Same breakpoint for every technique; only a prefix above the model's minimum
    # cacheable size is actually cached (silently skipped otherwise).
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def math_system(technique: str) -> list[dict[str, Any]]:
    text = MATH_ROLE
    if technique == "few_shot":
        ex = "\n".join(
            f"<example>\n<problem>{e['question']}</problem>\n<answer>{e['answer']}</answer>\n</example>"
            for e in TASKS["reasoning_examples"]
        )
        text += "\n\n<examples>\n" + ex + "\n</examples>"
    text += "\n\n" + (MATH_COT if technique == "cot" else NUMBER_ONLY)
    return [{"type": "text", "text": text}]


def request_options(technique: str) -> dict[str, Any]:
    if technique == "thinking":
        # Model-native reasoning: thinking tokens bill as output; display "summarized"
        # returns a readable summary (the default on current models is omitted text).
        return {"max_tokens": 4000, "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "medium"}}
    # Haiku 5.5 accepts thinking "disabled" at effort <= high: isolates the prompt technique.
    # (Never send "disabled" to Opus 5.5 / Sonnet 5.5 - it is a 400 there.)
    return {"max_tokens": 600, "thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}


# ---------------------------------------------------------------------------
# Graders (pure code)
# ---------------------------------------------------------------------------
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text)
    return m.group(1) if m else text.strip()


def grade_label(text: str, technique: str) -> tuple[str | None, bool]:
    """Return (predicted label or None, strict-format-ok)."""
    if technique == "cot":
        strict = bool(re.search(r"<reasoning>.+</reasoning>", text, re.DOTALL)) and ANSWER_RE.search(text) is not None
    else:
        strict = text.strip() in LABELS
    ans = extract_answer(text).strip().lower().rstrip(".")
    if ans in LABELS:
        return ans, strict
    found = [lbl for lbl in LABELS if re.search(rf"\b{lbl}\b", ans)]
    return (found[0] if len(found) == 1 else None), strict


def grade_number(text: str, expected: float) -> tuple[float | None, bool]:
    nums = NUMBER_RE.findall(extract_answer(text))
    if not nums:
        return None, False
    value = float(nums[-1].replace(",", ""))
    return value, abs(value - expected) <= 1e-6 * max(1.0, abs(expected))


# ---------------------------------------------------------------------------
# Offline responder: simulates typical technique-dependent behaviour
# ---------------------------------------------------------------------------
def detect_technique(req: MockRequest) -> str:
    if (req.body.get("thinking") or {}).get("type") == "adaptive":
        return "thinking"
    if "<reasoning>" in req.system_text:
        return "cot"
    if "<examples>" in req.system_text:
        return "few_shot"
    return "zero_shot"


def responder(req: MockRequest) -> Any:
    user = req.last_user_text
    tech = detect_technique(req)
    if "internal thinking verbatim" in user:
        return {"content": [], "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "reasoning_extraction",
                                 "explanation": "The request asks the model to reproduce its internal reasoning."}}
    if "support email classifier" in req.system_text:
        item = next(e for e in TASKS["test_set"] if e["email"] in user)
        pred = SIM["classification_errors"][tech].get(item["id"], item["label"])
        if tech == "zero_shot":
            return SIM["zero_shot_verbose"].get(item["id"], pred) if pred == item["label"] else pred
        if tech == "few_shot":
            return pred
        rationale = SIM["cot_wrong_rationale"].get(item["id"], item["rationale"]) if pred != item["label"] else item["rationale"]
        if tech == "cot":
            return f"<reasoning>{rationale}</reasoning>\n<answer>{pred}</answer>"
        return [{"type": "thinking", "thinking": f"Checking the taxonomy rules. {rationale}", "signature": "sig_offline"},
                {"type": "text", "text": pred}]
    prob = next(p for p in TASKS["reasoning_test"] if p["question"] in user)
    value = SIM["reasoning_errors"][tech].get(prob["id"], prob["answer"])
    if "one-sentence justification" in user:
        return [{"type": "thinking", "thinking": f"Summary: {prob['worked']}", "signature": "sig_offline"},
                {"type": "text", "text": f"<answer>{value}</answer>\nJustification: 0.5% of the 43,200 minutes in a 30-day month."}]
    if tech == "cot":
        return f"<reasoning>{prob['worked']}</reasoning>\n<answer>{value}</answer>"
    if tech == "thinking":
        return [{"type": "thinking", "thinking": f"Summary: {prob['worked']}", "signature": "sig_offline"},
                {"type": "text", "text": str(value)}]
    return str(value)


# ---------------------------------------------------------------------------
# Runner + metrics
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    n: int = 0
    correct: int = 0
    strict: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    misses: dict[str, str] = field(default_factory=dict)

    def add(self, msg: Any, ms: float) -> None:
        u = msg.usage
        self.n += 1
        self.input_tokens += u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0)
        self.output_tokens += u.output_tokens
        self.cache_read += u.cache_read_input_tokens or 0
        self.cost += cost_usd(msg.model, u)
        self.latency_ms += ms


def call(client: Any, system: list[dict[str, Any]], user: str, technique: str, model: str = MODEL_HAIKU) -> tuple[Any, float]:
    t0 = time.perf_counter()
    msg = client.messages.create(model=model, system=system, messages=[{"role": "user", "content": user}],
                                 **request_options(technique))
    return msg, elapsed_ms(t0)


def table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({round(100 * a / b)}%)"


def main() -> None:
    banner("D2.3 - Prompting Techniques - Zero-shot, Few-shot, Chain-of-Thought")
    client = get_client(responder)
    tests = TASKS["test_set"]
    problems = TASKS["reasoning_test"]

    # ---------------------------------------------------------------- 1
    section("1. Task set and graders")
    dist = {lbl: sum(1 for e in tests if e["label"] == lbl) for lbl in LABELS}
    print(f"  Task A: {len(tests)} labelled emails, {len(LABELS)} labels: " + ", ".join(f"{k}={v}" for k, v in dist.items()))
    print(f"  Task B: {len(problems)} multi-step arithmetic problems with exact answers")
    print("  Grader: strict format check + lenient extraction (<answer> tag, single label / last number).")
    print("  Model: Claude Haiku 5.5 for every technique, so only the prompt changes.")

    # ---------------------------------------------------------------- 2
    section("2. Few-shot example selection: balanced, diverse, no leakage")
    examples = select_examples(TASKS["few_shot_pool"], per_label=3)
    ex_dist = {lbl: sum(1 for e in examples if e["label"] == lbl) for lbl in LABELS}
    print("  order : " + " ".join(f"{e['id']}:{e['label'][:8]}" for e in examples))
    print("  counts: " + ", ".join(f"{k}={v}" for k, v in ex_dist.items()))
    worst = max(((jaccard(e["email"], t["email"]), e["id"], t["id"]) for e in examples for t in tests))
    print(f"  leakage check: max word-overlap example vs test = {worst[0]:.2f} ({worst[1]} vs {worst[2]})"
          f" -> {'OK' if worst[0] < 0.6 else 'LEAK - replace example'}")
    sample_user = f"<email>{tests[0]['email']}</email>"
    zero_tok = client.messages.count_tokens(model=MODEL_HAIKU, system=classify_system("zero_shot", examples),
                                            messages=[{"role": "user", "content": sample_user}]).input_tokens
    few_tok = client.messages.count_tokens(model=MODEL_HAIKU, system=classify_system("few_shot", examples),
                                           messages=[{"role": "user", "content": sample_user}]).input_tokens
    overhead = few_tok - zero_tok
    print(f"  count_tokens: zero-shot request {zero_tok} tok, few-shot request {few_tok} tok "
          f"-> examples add {overhead} input tokens to EVERY call")

    # ---------------------------------------------------------------- 3
    section("3. Task A - email classification, 4 techniques")
    cls_stats: dict[str, Stats] = {}
    sample_cot = sample_thinking = ""
    for tech in TECHNIQUES:
        st = Stats()
        system = classify_system(tech, examples)
        for item in tests:
            msg, ms = call(client, system, f"<email>{item['email']}</email>", tech)
            st.add(msg, ms)
            if msg.stop_reason != "end_turn":
                st.misses[item["id"]] = f"stop:{msg.stop_reason}"
                continue
            text = text_of(msg)
            pred, strict = grade_label(text, tech)
            st.strict += strict
            if pred == item["label"]:
                st.correct += 1
            else:
                st.misses[item["id"]] = pred or "unparseable"
            if item["id"] == "E07" and tech == "cot":
                sample_cot = text
            if item["id"] == "E07" and tech == "thinking":
                sample_thinking = next((b.thinking for b in msg.content if b.type == "thinking"), "")
        cls_stats[tech] = st
    rows = []
    for tech in TECHNIQUES:
        st = cls_stats[tech]
        rows.append([NAMES[tech], pct(st.correct, st.n), pct(st.strict, st.n), str(st.input_tokens // st.n),
                     str(st.output_tokens // st.n), str(st.cache_read), f"${st.cost / st.n * 1000:.4f}",
                     f"{st.latency_ms / st.n:,.0f} ms"])
    table(["technique", "accuracy", "strict format", "avg in", "avg out", "cache rd", "$/1k req", "avg latency"], rows)
    print("  Per-item misses (item: predicted):")
    for tech in TECHNIQUES:
        misses = cls_stats[tech].misses
        print(f"    {NAMES[tech]:<18} " + (", ".join(f"{k}:{v}" for k, v in misses.items()) or "none"))
    print("  E07 under chain-of-thought (the precedence rule is applied explicitly):")
    for line in sample_cot.splitlines():
        print("    | " + line)
    print(f"  E07 thinking summary (display='summarized'): {sample_thinking}")

    # ---------------------------------------------------------------- 4
    section("4. Task B - multi-step arithmetic, 4 techniques")
    math_stats: dict[str, Stats] = {}
    sample_math = ""
    for tech in TECHNIQUES:
        st = Stats()
        system = math_system(tech)
        for p in problems:
            msg, ms = call(client, system, f"<problem>{p['question']}</problem>", tech)
            st.add(msg, ms)
            value, ok = grade_number(text_of(msg), p["answer"])
            st.strict += 1 if value is not None else 0
            if ok:
                st.correct += 1
            else:
                st.misses[p["id"]] = f"{value:g} (want {p['answer']})" if value is not None else "unparseable"
            if p["id"] == "M02" and tech == "cot":
                sample_math = text_of(msg)
        math_stats[tech] = st
    rows = []
    for tech in TECHNIQUES:
        st = math_stats[tech]
        rows.append([NAMES[tech], pct(st.correct, st.n), str(st.input_tokens // st.n), str(st.output_tokens // st.n),
                     f"${st.cost / st.n * 1000:.4f}", f"{st.latency_ms / st.n:,.0f} ms",
                     ", ".join(f"{k}={v}" for k, v in st.misses.items()) or "none"])
    table(["technique", "accuracy", "avg in", "avg out", "$/1k req", "avg latency", "misses"], rows)
    print("  M02 under chain-of-thought (the rounding step becomes explicit):")
    for line in sample_math.splitlines():
        print("    | " + line)

    # ---------------------------------------------------------------- 5
    section("5. Trap: asking a thinking model to dump its internal reasoning")
    trap_user = (f"<problem>{problems[4]['question']}</problem>\n"
                 "Before answering, copy your complete internal thinking verbatim into the response.")
    msg = client.messages.create(model=MODEL_OPUS, max_tokens=2000, system=math_system("zero_shot"),
                                 messages=[{"role": "user", "content": trap_user}])
    if msg.stop_reason == "refusal":  # always branch on stop_reason before reading content
        cat = msg.stop_details.category if msg.stop_details else None
        print(f"  Opus 5.5 -> stop_reason=refusal, category={cat}: {msg.stop_details.explanation if msg.stop_details else ''}")
    fix_user = f"<problem>{problems[4]['question']}</problem>\nGive the final number in <answer> tags and a one-sentence justification."
    fixed = client.messages.create(
        model=MODEL_OPUS, max_tokens=4000, system=[{"type": "text", "text": MATH_ROLE}],
        thinking={"type": "adaptive", "display": "summarized"}, output_config={"effort": "medium"},
        messages=[{"role": "user", "content": fix_user}],
    )
    summary = next((b.thinking for b in fixed.content if b.type == "thinking"), "")
    print(f"  Fix -> stop_reason={fixed.stop_reason}; answer: {text_of(fixed).splitlines()[0]}")
    print(f"         thinking summary block: {summary}")
    print("  Opus 5.5 / Fable 5.1 always think. Read reasoning from thinking blocks (display='summarized'),")
    print("  and ask the visible answer for a SHORT justification, not a transcript of internal thinking.")

    # ---------------------------------------------------------------- 6
    section("6. Few-shot examples cost tokens on every call -> cache them")
    volume = 1_000_000
    few = cls_stats["few_shot"]
    print(f"  Observed: few-shot run read {few.cache_read:,} cached tokens over {few.n} calls "
          f"(first call writes, the rest read).")
    print(f"  The zero-shot request ({zero_tok} tok) is below the offline mock's 1,024-token cache minimum, so it")
    print("  was silently not cached (real minimum on current models: 512 tokens).")
    zs = cls_stats["zero_shot"]
    if few.cost / few.n < zs.cost / zs.n:
        print(f"  With a warm cache, few-shot cost ${few.cost / few.n * 1000:.4f}/1k requests vs ${zs.cost / zs.n * 1000:.4f} "
              "for uncached zero-shot: bigger, cached prompts can be CHEAPER.")
    rows = []
    for model, name in ((MODEL_HAIKU, "Haiku 5.5"), (MODEL_OPUS, "Opus 5.5")):
        p = PRICING[model]
        uncached = overhead * volume * p["input"] / 1e6
        cached = overhead * volume * p["cache_read"] / 1e6
        rows.append([name, f"{overhead} tok", f"${uncached:,.2f}", f"${cached:,.2f}", f"{1 - cached / uncached:.0%}"])
    table(["model", "example overhead", "uncached / 1M req", "cached / 1M req", "saved"], rows)
    print("  Keep examples in the stable prefix (system), never interleaved with per-request content.")

    # ---------------------------------------------------------------- 7
    section("7. When each technique wins")
    zs, fs, ct, th = (cls_stats[t] for t in TECHNIQUES)
    print(f"  zero-shot        : smallest prompt, fine for unambiguous labels; missed the precedence rules "
          f"({pct(zs.correct, zs.n)}) and drifted off-format ({pct(zs.strict, zs.n)} strict).")
    print(f"  few-shot         : fixes label boundaries AND output format ({pct(fs.strict, fs.n)} strict);"
          f" costs +{overhead} input tok/call - cache them.")
    print(f"  chain-of-thought : best when the task needs explicit steps (math {pct(math_stats['cot'].correct, len(problems))});"
          f" pays in output tokens and latency.")
    print(f"  adaptive thinking: model-native reasoning, answer stays clean ({pct(th.correct, th.n)} on Task A);"
          f" tune cost with effort, not prompt hacks.")
    print("  Combine them: few-shot examples of the FORMAT + thinking for hard reasoning, all on a cached prefix.")


if __name__ == "__main__":
    main()
