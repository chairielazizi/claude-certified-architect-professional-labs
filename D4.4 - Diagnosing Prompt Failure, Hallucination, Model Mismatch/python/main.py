"""
D4.4 - Diagnosing Prompt Failure, Hallucination, Model Mismatch

A diagnostic harness that walks SYMPTOM -> HYPOTHESES -> PROBES -> EVIDENCE -> ROOT CAUSE
-> FIX -> VERIFY for six broken Claude applications:

  1. ambiguous prompt -> inconsistent output format
  2. hallucination from missing grounding
  3. stale RAG index after a document refresh   (exam sample question 3)
  4. model mismatch on multi-step reasoning
  5. truncation (stop_reason = max_tokens)
  6. context overload / lost-in-the-middle

Run from the repo root:
    .venv/Scripts/python "D4.4 - Diagnosing Prompt Failure, Hallucination, Model Mismatch/python/main.py"
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    MockRequest,
    banner,
    cost_usd,
    estimate_tokens,
    get_client,
    section,
    text_block,
)

# ---------------------------------------------------------------------------
# Small deterministic helpers (same algorithms as the TypeScript version)
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


def compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def reply(text: str) -> dict[str, Any]:
    return {"content": [text_block(text)], "usage": {"output_tokens": estimate_tokens(text)}}


STOP = {"a", "an", "the", "is", "are", "of", "to", "and", "or", "in", "on", "for", "it", "its", "does", "do", "how",
        "what", "can", "be", "by", "with", "into", "many", "much", "long", "which", "this", "that", "next"}


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and (len(w) > 2 or w.isdigit())}


def text_of(msg: Any) -> str:
    return "".join(b.text for b in msg.content if b.type == "text")


# ---------------------------------------------------------------------------
# Diagnostic report helper
# ---------------------------------------------------------------------------
SUMMARY: list[tuple[str, str, str]] = []


class Report:
    def __init__(self, n: int, title: str) -> None:
        self.title = title
        section(f"Scenario {n}: {title}")

    def line(self, label: str, text: str) -> None:
        print(f"  {label:<11}: {text}")

    def hypotheses(self, items: list[str]) -> None:
        self.line("HYPOTHESES", "; ".join(f"{i + 1}) {h}" for i, h in enumerate(items)))

    def probe(self, name: str, evidence: str) -> None:
        self.line("PROBE", name)
        self.line("  evidence", evidence)

    def conclude(self, category: str, short: str, root: str, fix: str, verify: str) -> None:
        self.line("ROOT CAUSE", root)
        self.line("FIX", fix)
        self.line("VERIFY", verify)
        SUMMARY.append((self.title, category, short))


# ---------------------------------------------------------------------------
# Scenario data
# ---------------------------------------------------------------------------
TICKET = "The mobile app crashes every time I upload a photo larger than 10 MB on Android 15. Please fix this soon."
FORMAT_VARIANTS = [
    "The customer reports that the mobile app crashes when uploading photos larger than 10 MB on Android; they want a fix soon.",
    "- Issue: app crashes on large photo upload\n- Product: mobile app (Android)\n- Urgency: medium",
    '{"issue": "App crashes when uploading photos over 10 MB", "product": "mobile app", "urgency": "medium"}',
    "Customer says the Android app crashes on photo uploads over 10 MB and asks for a quick fix.",
    "1. Problem: crash on photo upload\n2. Platform: Android 15\n3. Priority: medium",
    "- Crash when uploading large photos\n- Android 15, mobile app\n- Needs fix soon",
    '```json\n{"issue": "Crash on large photo upload", "product": "mobile app", "urgency": "medium"}\n```',
    '{"issue": "Photo upload crash above 10 MB", "product": "mobile app", "urgency": "medium"}',
]
STRUCTURED_SUMMARY = {"issue": "App crashes when uploading photos larger than 10 MB on Android 15", "product": "mobile app", "urgency": "medium"}

PRODUCT_DOCS = {
    "x200-warranty": "The X200 router has a 2-year limited warranty period that covers hardware defects.",
    "x200-security": "The X200 supports WPA3 and WPA2 encryption on its 2.4 GHz and 5 GHz bands.",
    "x200-setup": "Set up the X200 with the mobile app by scanning the QR code on the base of the router.",
}
FABRICATED_WARRANTY = [
    "The X200 comes with a 3-year warranty.",
    "The X200 includes a 2-year warranty.",
    "The X200 is covered by a 5-year warranty.",
    "The X200 has a 3-year warranty.",
    "The X200 has a 4-year warranty on all parts.",
]

POLICY_V1 = {
    "refunds": "Refunds are accepted within 30 days of delivery for unused items.",
    "shipping": "Standard shipping costs 5 dollars and takes 3 to 5 business days.",
    "returns-label": "Return labels are emailed within one business day of approval.",
}
POLICY_V2 = {
    "refunds": "Refunds are accepted within 14 days of delivery for unused items.",
    "shipping": "Standard shipping costs 7 dollars and takes 3 to 5 business days.",
    "returns-label": "Return labels are emailed within one business day of approval.",
}

# Multi-step proration tasks: list of (days, monthly price) segments in a 30-day cycle, minus a credit.
BILLING_TASKS: list[dict[str, Any]] = [
    {"id": "S1", "kind": "simple", "text": "The Pro plan costs 30 dollars per month. What is the monthly price of the Pro plan?", "answer": 30.0},
    {"id": "S2", "kind": "simple", "text": "A plan costs 50 dollars for a 30-day cycle. What does one full cycle cost?", "answer": 50.0},
    {"id": "S3", "kind": "simple", "text": "A customer has a 5 dollar credit and a 20 dollar invoice. How much do they owe?", "answer": 15.0},
    {"id": "C1", "kind": "complex", "segments": [(10, 30), (10, 50), (10, 30)], "credit": 0.0},
    {"id": "C2", "kind": "complex", "segments": [(15, 20), (15, 40)], "credit": 5.0},
    {"id": "C3", "kind": "complex", "segments": [(5, 50), (25, 30)], "credit": 0.0},
    {"id": "C4", "kind": "complex", "segments": [(12, 45), (18, 15)], "credit": 2.5},
    {"id": "C5", "kind": "complex", "segments": [(7, 60), (8, 30), (15, 90)], "credit": 0.0},
]
for task in BILLING_TASKS:
    if task["kind"] == "complex":
        parts = ", then ".join(f"the {p} dollar/month plan for {d} days" for d, p in task["segments"])
        credit = f" A {fmt(task['credit'], 2)} dollar goodwill credit applies." if task["credit"] else ""
        task["text"] = f"In a 30-day billing cycle the customer was on {parts}. Prorate by day.{credit} What is the net charge for the cycle?"
        # The grader computes the truth with real code - never trust a model to grade arithmetic.
        task["answer"] = total(d * p / 30.0 for d, p in task["segments"]) - task["credit"]
# Which tasks each tier gets right in this simulation (complex multi-step is where small models slip).
CAPABILITY = {MODEL_HAIKU: {"S1", "S2", "S3", "C2"}, MODEL_SONNET: {"S1", "S2", "S3", "C1", "C2", "C3", "C4"},
              MODEL_OPUS: {"S1", "S2", "S3", "C1", "C2", "C3", "C4", "C5"}}

INCIDENTS = [
    {"id": "INC-1", "title": "Login latency spike", "events": 2},
    {"id": "INC-2", "title": "EU region outage", "events": 9},
    {"id": "INC-3", "title": "Webhook retries", "events": 3},
    {"id": "INC-4", "title": "Billing double charge", "events": 8},
    {"id": "INC-5", "title": "Search index lag", "events": 2},
    {"id": "INC-6", "title": "Certificate expiry", "events": 7},
]

TOPICS = [
    "travel booking", "expense receipts", "laptop refresh", "badge access", "parking", "meeting rooms", "printing",
    "visitor registration", "software requests", "password rotation", "remote work equipment", "mobile phones",
    "conference attendance", "training budget", "mentoring", "performance reviews", "office moves", "desk booking",
    "shipping samples", "vendor onboarding", "purchase orders", "contract signatures", "data retention", "email signatures",
    "social media", "press inquiries", "gifts and hospitality", "conflicts of interest", "open source use", "code reviews",
    "incident reporting", "on-call rotation", "change management", "backup restore", "cloud accounts", "encryption keys",
    "customer data access", "call recording", "accessibility", "first aid", "fire drills", "pets in the office",
    "bicycle storage", "kitchen use", "lost and found", "recycling", "lockers", "after-hours access",
    "translation services", "brand assets", "slide templates", "intranet posts", "team events", "charity matching",
    "referral bonuses", "relocation", "name badges", "open-plan noise", "plant care",
]
NEEDLE = ("Policy P-NEEDLE (vacation carry-over): Employees may carry over up to 7 unused vacation days into the next year. "
          "Carrying over more than 7 days requires director approval.")
VACATION_Q = "How many unused vacation days can an employee carry over into the next year?"


def filler_doc(i: int, topic: str) -> str:
    return (f"Policy P-{i:03d} ({topic}): Requests about {topic} go through the internal service portal. "
            f"The owning team reviews {topic} requests within five business days. "
            f"Exceptions to the {topic} rules require written approval from a department head. "
            f"Questions about {topic} go to the operations mailbox and records are kept for audit purposes.")


# ---------------------------------------------------------------------------
# Offline responder: one "model" behaving like each broken app would
# ---------------------------------------------------------------------------
COUNTERS: dict[str, int] = {}


def bump(key: str) -> int:
    n = COUNTERS.get(key, 0)
    COUNTERS[key] = n + 1
    return n


def best_sentence(question: str, context: str) -> tuple[str, str, int]:
    """Faithful extraction: the context sentence (and its doc id) that overlaps the question most."""
    q = content_words(question)
    best = ("", "", 0)
    for m in re.finditer(r'<doc id="([^"]+)"[^>]*>\n(.*?)\n</doc>', context, re.S):
        for sent in re.split(r"(?<=\.)\s+", m.group(2)):
            score = len(q & content_words(sent))
            if score > best[2]:
                best = (sent.strip(), m.group(1), score)
    return best


def responder(req: MockRequest) -> Any:
    system, user = req.system_text, req.last_user_text
    if system.startswith("Summarize"):
        if (req.body.get("output_config") or {}).get("format"):
            return reply(compact(STRUCTURED_SUMMARY))
        return reply(FORMAT_VARIANTS[bump("fmt") % len(FORMAT_VARIANTS)])
    if system.startswith("You are a helpful product expert"):
        if "<doc" not in user:  # no grounding: confident, unstable inventions
            if "6 GHz" in user:
                return reply("Yes, the X200 supports 6 GHz Wi-Fi 6E for the fastest speeds.")
            return reply(FABRICATED_WARRANTY[bump("warranty") % len(FABRICATED_WARRANTY)])
        sent, doc, score = best_sentence(user.split("Question:")[-1], user)
        if score < 3:  # weak overlap = the documents do not answer it
            return reply("I don't know based on the provided documents.")
        return reply(f"{sent} [{doc}]")
    if system.startswith("You answer questions about Acme policies"):
        sent, doc, _ = best_sentence(user.split("Question:")[-1], user)
        return reply(f"{sent} [{doc}]")
    if system.startswith("You are a billing calculator"):
        task = next(t for t in BILLING_TASKS if t["text"] in user)
        if task["id"] in CAPABILITY.get(req.model, set()):
            value = task["answer"]
        else:  # classic slip: charge the last plan for the full month and forget the proration
            value = task["segments"][-1][1] - task["credit"]
        return reply(f"Net charge: {fmt(value, 2)} dollars")
    if system.startswith("Write an incident report"):
        inc = next(i for i in INCIDENTS if i["id"] in user)
        report = {"title": inc["title"], "impact": f"Customers affected by {inc['title'].lower()}",
                  "timeline": [f"T+{k * 5}m: responders took step {k + 1} to mitigate {inc['title'].lower()} and confirmed the result" for k in range(inc["events"])],
                  "root_cause": "Under investigation", "actions": ["Add alerting", "Write a runbook"]}
        text = compact(report)
        if estimate_tokens(text) > int(req.body.get("max_tokens", 4096)):
            return text  # let the API truncate it: stop_reason becomes "max_tokens"
        return reply(text)
    if system.startswith("You are the HR policy assistant"):
        docs = re.findall(r'<doc id="([^"]+)">', user)
        if "P-NEEDLE" not in docs:
            return reply("I could not find a carry-over limit in the documents.")
        pos, n = docs.index("P-NEEDLE"), len(docs)
        rel = pos / (n - 1) if n > 1 else 0.0
        query_first = user.find("Question:") < user.find("<doc")
        missed = n > 20 and ((query_first and 0.2 < rel < 0.8) or (not query_first and 0.4 < rel < 0.6))
        return reply("I could not find a carry-over limit in the documents." if missed
                     else "Employees may carry over up to 7 unused vacation days [P-NEEDLE].")
    return reply("OK")


# ---------------------------------------------------------------------------
# Probes (the reusable diagnostic toolkit)
# ---------------------------------------------------------------------------
def classify_format(text: str) -> str:
    t = text.strip()
    try:
        obj = json.loads(t)
        return "json" if isinstance(obj, dict) else "other-json"
    except ValueError:
        pass
    if t.startswith("```"):
        return "fenced-json"
    lines = t.splitlines()
    if lines and all(line.startswith("- ") for line in lines):
        return "bullets"
    if lines and all(re.match(r"\d+\. ", line) for line in lines):
        return "numbered"
    return "prose"


def coefficient_of_variation(xs: list[float]) -> float:
    m = total(xs) / len(xs)
    var = total((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var) / m if m else 0.0


def grounded_fraction(answer: str, context: str) -> float:
    """Share of answer content words that appear in the supplied context (0 if no context)."""
    a = content_words(re.sub(r"\[[^\]]+\]", "", answer))
    if not a:
        return 1.0
    return len(a & content_words(context)) / len(a)


def retrieve(question: str, docs: dict[str, str], k: int) -> list[tuple[str, str]]:
    q = content_words(question)
    scored = sorted(docs.items(), key=lambda kv: (-len(q & content_words(kv[1])), kv[0]))
    return scored[:k]


def doc_block(items: list[tuple[str, str]], version: dict[str, int] | None = None) -> str:
    out = []
    for doc_id, text in items:
        ver = f' version="{version[doc_id]}"' if version else ""
        out.append(f'<doc id="{doc_id}"{ver}>\n{text}\n</doc>')
    return "\n".join(out)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


LATENCY_PROFILE = {MODEL_HAIKU: (250.0, 3.0), MODEL_SONNET: (500.0, 7.0), MODEL_OPUS: (900.0, 12.0)}


def modeled_latency(key: str, model: str, out_tokens: int) -> float:
    rng = mulberry32(fnv1a32(key))
    ttft, per = LATENCY_PROFILE[model]
    return ttft * (0.85 + 0.3 * rng()) + per * out_tokens


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def scenario_format(client: Any) -> None:
    r = Report(1, "Ambiguous prompt -> inconsistent output format")
    r.line("SYMPTOM", "the downstream parser rejects ~half of the ticket summaries; nothing changed in the code")
    r.hypotheses(["prompt does not specify an output contract", "model regression", "parser bug"])
    outs = [text_of(client.messages.create(model=MODEL_SONNET, max_tokens=512, system="Summarize the support ticket.",
                                           messages=[{"role": "user", "content": TICKET}])) for _ in range(8)]
    formats = [classify_format(o) for o in outs]
    counts = {f: formats.count(f) for f in dict.fromkeys(formats)}
    parse_ok = sum(1 for f in formats if f == "json")
    r.probe("same input sampled 8x (no temperature knob - variance is inherent, so sample it)",
            f"{len(counts)} distinct formats (" + ", ".join(f"{k}={v}" for k, v in counts.items()) + f"); strict-JSON parse success {parse_ok}/8; "
            f"length CV={fmt(coefficient_of_variation([float(len(o)) for o in outs]), 2)}")
    r.probe("parser on a known-good JSON sample", "parses fine -> parser is not the problem")

    class TicketSummary(BaseModel):
        issue: str
        product: str
        urgency: Literal["low", "medium", "high"]

    fixed = []
    for _ in range(8):
        msg = client.messages.parse(model=MODEL_SONNET, max_tokens=512,
                                    system="Summarize the support ticket as JSON with keys issue, product, urgency.",
                                    messages=[{"role": "user", "content": TICKET}], output_format=TicketSummary)
        fixed.append(msg.parsed_output is not None)
    r.conclude("prompt", "no output contract", "underspecified prompt: no output contract, so each sample picks its own format",
               "state the format explicitly AND enforce it with structured outputs (output_config.format / messages.parse)",
               f"8 samples after fix: {sum(fixed)}/8 schema-valid, 1 format")


def scenario_grounding(client: Any) -> None:
    r = Report(2, "Hallucination from missing grounding")
    r.line("SYMPTOM", "the product bot states warranty terms and features with full confidence; some are wrong")
    r.hypotheses(["no source documents in the prompt", "outdated documents", "model ignores provided context"])
    sys_prompt = "You are a helpful product expert for the X200 router."
    q1 = "How long is the warranty period of the X200?"
    q2 = "Does the X200 support 6 GHz Wi-Fi?"
    samples = [text_of(client.messages.create(model=MODEL_SONNET, max_tokens=256, system=sys_prompt,
                                              messages=[{"role": "user", "content": q1}])) for _ in range(5)]
    years = [m.group(1) for s in samples if (m := re.search(r"(\d+)-year", s))]
    r.probe("self-consistency: ask the same factual question 5x",
            f"warranty answers {'/'.join(years)} years -> {len(set(years))} different 'facts' = invention, not knowledge")
    unans = text_of(client.messages.create(model=MODEL_SONNET, max_tokens=256, system=sys_prompt,
                                           messages=[{"role": "user", "content": q2}]))
    r.probe("groundedness of answers vs supplied context", f"context supplied: none -> groundedness {pct(grounded_fraction(samples[0], ''))}; "
            f"unanswerable question answered confidently: '{unans}'")
    grounded_sys = (sys_prompt + " Answer ONLY from the documents. Cite the doc id in brackets. "
                    "If the answer is not in the documents, say you don't know.")
    fixed: list[str] = []
    for q in (q1, q1, q1, q2):
        ctx = doc_block(retrieve(q, PRODUCT_DOCS, 2))
        fixed.append(text_of(client.messages.create(model=MODEL_SONNET, max_tokens=256, system=grounded_sys,
                                                    messages=[{"role": "user", "content": f"{ctx}\n\nQuestion: {q}"}])))
    ctx_all = " ".join(PRODUCT_DOCS.values())
    r.conclude("grounding", "no source documents", "the model was asked for facts it was never given; it filled the gap fluently",
               "retrieve source docs into the prompt, require citations, and give explicit permission to say 'I don't know'",
               f"warranty answer x3: '{fixed[0]}' (consistent: {'yes' if len(set(fixed[:3])) == 1 else 'no'}, "
               f"groundedness {pct(grounded_fraction(fixed[0], ctx_all))}); unanswerable -> '{fixed[3]}'")


def scenario_stale_index(client: Any) -> None:
    r = Report(3, "Stale RAG index after a document refresh")
    r.line("SYMPTOM", "after Monday's policy refresh the assistant gives confident WRONG refund answers;")
    r.line("", "latency, error rate and model are unchanged")
    r.hypotheses(["retrieval serves stale chunks (index not rebuilt)", "model or prompt changed", "infra degradation"])
    sys_prompt = "You answer questions about Acme policies using ONLY the provided context. Cite the doc id."
    index = {doc_id: {"text": text, "version": 1, "hash": sha(text)} for doc_id, text in POLICY_V1.items()}  # built last week
    source = {doc_id: {"text": text, "version": 2 if POLICY_V1[doc_id] != text else 1, "hash": sha(text)} for doc_id, text in POLICY_V2.items()}
    question = "Within how many days of delivery are refunds accepted?"

    def ask(store: dict[str, dict[str, Any]], key: str) -> tuple[str, float]:
        hits = retrieve(question, {k: v["text"] for k, v in store.items()}, 2)
        ctx = doc_block(hits, {k: store[k]["version"] for k, _ in hits})
        msg = client.messages.create(model=MODEL_SONNET, max_tokens=256, system=sys_prompt,
                                     messages=[{"role": "user", "content": f"{ctx}\n\nQuestion: {question}"}])
        return text_of(msg), modeled_latency(key, msg.model, msg.usage.output_tokens)

    before = [ask(index, f"before-{i}") for i in range(3)]   # last week: index == source
    after = [ask(index, f"after-{i}") for i in range(3)]     # today: source changed, index did not
    r.probe("config diff (model id, prompt version) week-over-week", f"model {MODEL_SONNET} both weeks; prompt hash unchanged -> not a model/prompt change")
    p50_b = sorted(x[1] for x in before)[1]
    p50_a = sorted(x[1] for x in after)[1]
    r.probe("latency and errors before vs after refresh", f"p50 {fmt(p50_b, 0)} ms vs {fmt(p50_a, 0)} ms, errors 0 vs 0 -> infra and model are healthy")
    ans = after[0][0]
    r.probe("generation check: is the answer faithful to what was retrieved?",
            f"answer '{ans}' is {pct(grounded_fraction(ans, ' '.join(v['text'] for v in index.values())))} grounded in the retrieved chunks "
            "-> the model is NOT hallucinating")
    retrieved = retrieve(question, {k: v["text"] for k, v in index.items()}, 2)
    stale = [k for k, _ in retrieved if index[k]["hash"] != source[k]["hash"]]
    detail = ", ".join(f"{k}: index v{index[k]['version']} {index[k]['hash']} vs source v{source[k]['version']} {source[k]['hash']}" for k, _ in retrieved)
    r.probe("retrieval check: compare retrieved chunk hashes/versions with the source of truth",
            f"{len(stale)}/{len(retrieved)} retrieved chunks are stale ({detail})")
    reindexed = {k: dict(v) for k, v in source.items()}  # incremental re-embed of changed docs
    fixed, _ = ask(reindexed, "fixed")
    r.conclude("data pipeline", "index not rebuilt after refresh", "the refresh updated the source store but the re-indexing job did not run; the model faithfully "
               "answered from stale chunks", "re-index changed docs (content-hash diff), alert on ingest/index job failures, monitor "
               "index-vs-source version drift, add regression cases for updated docs", f"after re-index: '{fixed}'")


def scenario_model_mismatch(client: Any) -> None:
    r = Report(4, "Model mismatch on multi-step reasoning")
    r.line("SYMPTOM", "after a cost-cutting switch to a smaller model, prorated refund totals are often wrong;")
    r.line("", "simple billing questions are still fine")
    r.hypotheses(["task exceeds the smaller model's reasoning capability", "prompt regression", "bad test data"])
    sys_prompt = "You are a billing calculator. Show only the final line 'Net charge: <amount> dollars'."
    rows = []
    task_cost: dict[tuple[str, str], float] = {}
    for model in (MODEL_HAIKU, MODEL_SONNET, MODEL_OPUS):
        correct = {"simple": 0, "complex": 0}
        cost = 0.0
        for task in BILLING_TASKS:
            msg = client.messages.create(model=model, max_tokens=1024, system=sys_prompt, output_config={"effort": "medium"},
                                         messages=[{"role": "user", "content": task["text"]}])
            m = re.search(r"(-?\d+(?:\.\d+)?)", text_of(msg))
            ok = m is not None and abs(float(m.group(1)) - task["answer"]) < 0.01
            correct[task["kind"]] += 1 if ok else 0
            task_cost[(model, task["id"])] = cost_usd(msg.model, msg.usage)
            cost += task_cost[(model, task["id"])]
        n_ok = correct["simple"] + correct["complex"]
        rows.append((model, correct["simple"], correct["complex"], cost, n_ok))
    for model, s, c, cost, n_ok in rows:
        r.probe(f"same 8 tasks on {model}", f"simple {s}/3, multi-step {c}/5, total cost ${fmt(cost, 6)}, "
                f"cost per correct answer ${fmt(cost / n_ok, 6)}")
    routed = total(task_cost[(MODEL_HAIKU if t["kind"] == "simple" else MODEL_OPUS, t["id"])] for t in BILLING_TASKS)
    r.conclude("model", "small model on multi-step math", "capability mismatch: the same prompt is correct on larger models, so the prompt is not the problem",
               "route by complexity (small model for lookups, larger model for multi-step math) or use the larger model at lower "
               "effort; compute arithmetic in code/tools; always re-run the eval before downgrading a model",
               f"routed (simple->Haiku, multi-step->Opus): 8/8 correct for ${fmt(routed, 6)} vs ${fmt(rows[2][3], 6)} all-Opus; "
               "a wrong refund costs far more than the tokens saved")


def scenario_truncation(client: Any) -> None:
    r = Report(5, "Truncation (stop_reason = max_tokens)")
    r.line("SYMPTOM", "some incident reports are cut off mid-sentence and fail JSON parsing; short incidents are fine")
    r.hypotheses(["max_tokens too low for long outputs", "model produces invalid JSON", "network cut the stream"])
    sys_prompt = "Write an incident report as compact JSON with keys title, impact, timeline, root_cause, actions."

    def run(max_tokens: int) -> list[tuple[str, str, int, bool]]:
        out = []
        for inc in INCIDENTS:
            msg = client.messages.create(model=MODEL_SONNET, max_tokens=max_tokens, system=sys_prompt,
                                         messages=[{"role": "user", "content": f"Incident {inc['id']}: {inc['title']}, {inc['events']} timeline events."}])
            try:
                json.loads(text_of(msg))
                valid = True
            except ValueError:
                valid = False
            out.append((inc["id"], msg.stop_reason or "", msg.usage.output_tokens, valid))
        return out

    before = run(150)
    hist: dict[str, int] = {}
    for _, stop, _, _ in before:
        hist[stop] = hist.get(stop, 0) + 1
    r.probe("stop_reason histogram (max_tokens=150)", ", ".join(f"{k}={v}" for k, v in hist.items()))
    r.probe("correlate stop_reason with parse failures", "; ".join(
        f"{i}: {stop} out={out} json={'ok' if ok else 'INVALID'}" for i, stop, out, ok in before))
    after = run(2048)
    r.conclude("config", "max_tokens below output length", "max_tokens (copied from a chat feature) is below the length of long reports; every invalid JSON "
               "has stop_reason=max_tokens and output_tokens == max_tokens",
               "raise max_tokens as a backstop, ask for concise output, stream long generations, and treat stop_reason "
               "'max_tokens' as a failure (never parse it; retry with a higher cap)",
               f"max_tokens=2048: {sum(1 for x in after if x[1] == 'end_turn')}/6 end_turn, {sum(1 for x in after if x[3])}/6 valid JSON")


def scenario_context(client: Any) -> None:
    r = Report(6, "Context overload / lost-in-the-middle")
    r.line("SYMPTOM", "as the policy KB grew, the HR assistant says 'not found' for facts that exist; cost per call rising")
    r.hypotheses(["too much context: relevant fact buried mid-prompt", "fact missing from KB", "retrieval bug"])
    sys_prompt = "You are the HR policy assistant. Answer from the documents and cite the policy id."
    budget = 4000

    def build(pos: int, query_first: bool, retrieval: bool) -> str:
        docs = [(f"P-{i:03d}", filler_doc(i, t)) for i, t in enumerate(TOPICS)]
        docs.insert(pos, ("P-NEEDLE", NEEDLE))
        if retrieval:
            docs = retrieve(VACATION_Q, dict(docs), 3)
        body = "\n".join(f'<doc id="{d}">\n{t}\n</doc>' for d, t in docs)
        return f"Question: {VACATION_Q}\n\n{body}" if query_first else f"{body}\n\nQuestion: {VACATION_Q}"

    def found(content: str) -> bool:
        msg = client.messages.create(model=MODEL_SONNET, max_tokens=256, system=sys_prompt, messages=[{"role": "user", "content": content}])
        return "7" in text_of(msg)

    n_docs = len(TOPICS) + 1
    full = client.messages.count_tokens(model=MODEL_SONNET, system=sys_prompt, messages=[{"role": "user", "content": build(30, True, False)}])
    r.probe("count_tokens on the production request", f"{n_docs} docs -> {full.input_tokens} input tokens vs route budget {budget} "
            f"-> {'OVER budget' if full.input_tokens > budget else 'within budget'}")
    positions = [0, 15, 30, 45, n_docs - 1]
    sweep = [found(build(p, True, False)) for p in positions]
    r.probe("needle position sweep (fact moved through the context, question first)",
            "  ".join(f"pos {p}: {'found' if ok else 'MISSED'}" for p, ok in zip(positions, sweep)))
    sweep_last = [found(build(p, False, False)) for p in positions]
    r.probe("same sweep with documents first, question last",
            "  ".join(f"pos {p}: {'found' if ok else 'MISSED'}" for p, ok in zip(positions, sweep_last)))
    sweep_rag = [found(build(p, False, True)) for p in positions]
    small = client.messages.count_tokens(model=MODEL_SONNET, system=sys_prompt, messages=[{"role": "user", "content": build(30, False, True)}])
    r.conclude("context", "fact buried in oversized context", "context stuffing: the fact is present but buried mid-context (U-shaped recall), with the question "
               "placed before 60 documents", "retrieve the top-k relevant docs instead of everything, put long documents first "
               "and the question last, and gate requests with count_tokens",
               f"top-3 retrieval + question last: found at {sum(sweep_rag)}/5 positions; {small.input_tokens} tokens "
               f"({pct(1 - small.input_tokens / full.input_tokens)} fewer)")


def main() -> None:
    banner("D4.4 - Diagnosing Prompt Failure, Hallucination, Model Mismatch")
    section("Diagnostic decision tree (symptom -> first probe -> likely root cause)")
    tree = [
        ("output format varies between runs", "sample N times, measure variance", "underspecified prompt"),
        ("confident facts with no source", "self-consistency + groundedness", "missing grounding"),
        ("wrong after content change, latency OK", "retrieved chunk version vs source", "stale index / ingestion"),
        ("fails only on complex tasks", "same inputs across model tiers", "model mismatch"),
        ("cut off / invalid JSON on long outputs", "stop_reason histogram", "max_tokens truncation"),
        ("misses facts that are in the KB", "count_tokens + position sweep", "context overload"),
    ]
    print(f"  {'SYMPTOM':<40}{'FIRST PROBE':<36}LIKELY ROOT CAUSE")
    for s, p, c in tree:
        print(f"  {s:<40}{p:<36}{c}")
    print("  Rule: change ONE thing at a time, and probe the cheapest layer that could explain the symptom first")
    print("  (config diff, stop_reason, latency/errors) before blaming the model.")
    client = get_client(responder)
    scenario_format(client)
    scenario_grounding(client)
    scenario_stale_index(client)
    scenario_model_mismatch(client)
    scenario_truncation(client)
    scenario_context(client)
    section("Diagnosis summary")
    print(f"  {'scenario':<48}{'layer':<15}root cause")
    for title, cat, root in SUMMARY:
        print(f"  {title:<48}{cat:<15}{root}")


if __name__ == "__main__":
    main()
