"""
D2.4 - Context Window Optimization and Token Management
=======================================================

An IT-helpdesk copilot turn that naively stuffs everything into the prompt
(12 KB articles with export boilerplate, a 22-message history, a verbose
telemetry dump) is brought under a context budget step by step:

  1. Pre-flight: client.messages.count_tokens on the naive request + per-part breakdown.
  2. Context budget allocator (water-filling with spill-over) across
     retrieved docs / history / tool results, after reserving system + output.
  3. Lossless-ish cleanup: strip doc boilerplate, trim the verbose tool result.
  4. Relevance filtering: BM25 ranks KB articles; keep the relevant ones that fit.
  5. History strategies compared on tokens AND fact recall:
     sliding window | summarize older turns (Haiku) | pinned facts + window.
  6. Assemble, verify with count_tokens, and call the model: naive vs
     sliding-window vs optimized answers (lost facts show up in the answer).
  7. Server-side alternatives (live-only request code): context editing
     (clear_tool_uses_20250919) and compaction (compact_20260112) vs memory.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    PRICING,
    MockRequest,
    banner,
    cost_usd,
    get_client,
    is_live,
    section,
    text_of,
)

DATA = Path(__file__).resolve().parents[1] / "data"
KB: list[dict[str, str]] = json.loads((DATA / "kb_docs.json").read_text(encoding="utf-8"))["docs"]
CONVO: dict[str, Any] = json.loads((DATA / "conversation.json").read_text(encoding="utf-8"))
TELEMETRY: dict[str, Any] = json.loads((DATA / "telemetry_tool_result.json").read_text(encoding="utf-8"))

MODEL = MODEL_SONNET
CONTEXT_BUDGET = 2600   # tokens we allow per request for this route (cost/latency/quality target)
OUTPUT_RESERVE = 700    # = max_tokens; reserve it up front so the answer is never squeezed out
SAFETY_MARGIN = 0.9     # local estimates pack to 90% of a bucket; count_tokens verifies the final request


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
class TokenCounter:
    """count_tokens is authoritative (model-specific tokenizer); a local chars/4
    estimate is only for inner packing loops. Never use tiktoken for Claude."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls = 0

    def request(self, messages: list[dict[str, Any]], system: str | None = None) -> int:
        self.calls += 1
        kwargs: dict[str, Any] = {"model": MODEL, "messages": messages}
        if system:
            kwargs["system"] = system
        return self.client.messages.count_tokens(**kwargs).input_tokens

    def text(self, text: str) -> int:
        """Tokens a piece of text adds to a request (minus the fixed message overhead).
        The baseline uses "." because the API rejects empty text content."""
        return self.request([{"role": "user", "content": text}]) - self.request([{"role": "user", "content": "."}])


def est(text: str) -> int:
    return len(text) // 4 + 1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_docs(docs: list[dict[str, str]]) -> str:
    inner = "\n".join(f'<doc id="{d["id"]}" title="{d["title"]}">\n{d["body"]}\n</doc>' for d in docs)
    return f"<kb>\n{inner}\n</kb>"


def render_telemetry(obj: Any, pretty: bool) -> str:
    body = json.dumps(obj, indent=2) if pretty else json.dumps(obj, separators=(",", ":"))
    return f"<telemetry>\n{body}\n</telemetry>"


def final_turn(question: str, docs_block: str = "", telemetry_block: str = "", pinned: str = "") -> dict[str, Any]:
    parts = [p for p in (pinned, docs_block, telemetry_block) if p]
    parts.append(f"<question>{question}</question>")
    return {"role": "user", "content": "\n\n".join(parts)}


# ---------------------------------------------------------------------------
# 2. Budget allocator: weighted water-filling with spill-over
# ---------------------------------------------------------------------------
@dataclass
class Bucket:
    name: str
    demand: int
    weight: float


def allocate(available: int, buckets: list[Bucket]) -> dict[str, int]:
    """Give each bucket a weight-proportional share, capped at its demand; whatever a
    satisfied bucket doesn't need spills over to the others (repeat until stable)."""
    alloc = {b.name: 0 for b in buckets}
    active = [b for b in buckets if b.demand > 0]
    remaining = available
    while active and remaining > 0:
        total_w = sum(b.weight for b in active)
        satisfied = []
        for b in active:
            share = int(remaining * b.weight / total_w)
            if alloc[b.name] + share >= b.demand:
                satisfied.append(b)
        if not satisfied:  # nobody can be fully served: split what's left by weight and stop
            for b in active:
                alloc[b.name] += int(remaining * b.weight / total_w)
            break
        for b in satisfied:
            remaining -= b.demand - alloc[b.name]
            alloc[b.name] = b.demand
        active = [b for b in active if b not in satisfied]
    return alloc


# ---------------------------------------------------------------------------
# 3. Cleanup: boilerplate + tool-result trimming
# ---------------------------------------------------------------------------
BOILERPLATE = [
    re.compile(r"^ORBIT IT KNOWLEDGE BASE.*$"),
    re.compile(r"^Document owner:.*$"),
    re.compile(r"^-{4,}$"),
    re.compile(r"^This document is uncontrolled.*$"),
    re.compile(r"^\(c\) \d{4} Orbit.*$"),
]


def strip_boilerplate(body: str) -> str:
    return "\n".join(line for line in body.split("\n") if not any(p.match(line) for p in BOILERPLATE)).strip()


def trim_tool_result(result: dict[str, Any], keep_sources: set[str], min_level: str, fields: list[str], max_items: int) -> dict[str, Any]:
    """Project to the fields the model needs, keep only relevant/severe events, cap the
    count, and say what was dropped so the model knows the data is partial."""
    rank = {"Information": 0, "Warning": 1, "Error": 2}
    events = result["events"]
    kept = [e for e in events if e["source"] in keep_sources or rank[e["level"]] >= rank[min_level]]
    kept = kept[-max_items:]  # most recent events matter most
    return {
        "device": result["device"],
        "collected_at": result["collected_at"],
        "summary": {
            "total_events": len(events),
            "returned": len(kept),
            "errors_by_message": dict(Counter(e["message"] for e in events if e["level"] == "Error")),
        },
        "events": [{k: e[k] for k in fields} for e in kept],
    }


# ---------------------------------------------------------------------------
# 4. Relevance: BM25 over KB articles
# ---------------------------------------------------------------------------
STOPWORDS = set("a an and are as at be but by can do does for from has have i if in is it its me my of on or so "
                "that the this to was what when with you your should try next".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower()) if w not in STOPWORDS and len(w) > 1]


def bm25(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    toks = [tokenize(t) for t in texts]
    n = len(toks)
    avgdl = sum(len(t) for t in toks) / n
    df = Counter(w for t in toks for w in set(t))
    q_terms = sorted(set(tokenize(query)))
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q_terms:
            if w in tf:
                idf = math.log((n - df[w] + 0.5) / (df[w] + 0.5) + 1)
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(s)
    return scores


# ---------------------------------------------------------------------------
# 5. History strategies
# ---------------------------------------------------------------------------
FACT_PATTERNS: list[tuple[str, str]] = [
    ("asset_tag", r"\bORB-\d{5}\b"),
    ("ticket", r"\bINC-\d{5}\b"),
    ("os_version", r"\bWindows 11 \d{2}H\d\b"),
    ("vpn_client", r"\bGlobalProtect \d+\.\d+\.\d+\b"),
    ("already_tried", r"\balready tried ([^.,]+)"),
    ("deadline", r"\b((?:fly|flying|travel(?:ling)?) to [A-Z][a-z]+[^.,]*)"),
]


def extract_facts(turns: list[dict[str, str]]) -> dict[str, str]:
    """Deterministic fact pinning: the latest match per fact wins. Cheap, auditable,
    and immune to a summarizer deciding a detail is unimportant."""
    facts: dict[str, str] = {}
    for t in turns:
        for name, pat in FACT_PATTERNS:
            m = re.search(pat, t["content"])
            if m:
                facts[name] = m.group(1) if m.groups() else m.group(0)
    return facts


def pinned_block(facts: dict[str, str]) -> str:
    return "<pinned_facts>\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items()) + "\n</pinned_facts>"


def sliding_window(turns: list[dict[str, str]], budget: int) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    used = 0
    for t in reversed(turns):
        cost = est(t["content"]) + 8  # + per-message overhead
        if used + cost > budget * SAFETY_MARGIN:
            break
        kept.insert(0, t)
        used += cost
    while kept and kept[0]["role"] != "user":  # a conversation must start with a user turn
        kept.pop(0)
    return kept


def recall(text: str, facts: list[str]) -> list[str]:
    low = text.lower()
    return [f for f in facts if f.lower() in low]


SUMMARIZER_SYSTEM = (
    "You are a conversation summarizer for an IT helpdesk. Summarize the transcript in <transcript> into a "
    "compact running summary (max 80 words). Preserve identifiers verbatim: asset tags, ticket numbers, versions."
)


def summarize_older(client: Any, turns: list[dict[str, str]]) -> tuple[str, Any]:
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    msg = client.messages.create(
        model=MODEL_HAIKU, max_tokens=300, system=SUMMARIZER_SYSTEM, output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}],
    )
    return text_of(msg), msg


# ---------------------------------------------------------------------------
# Offline responder
# ---------------------------------------------------------------------------
SUMMARY_TEXT = (
    "Dana (Finance), laptop ORB-55812 (ThinkPad X1 Carbon Gen 11). Outlook shared calendar fixed via repair "
    "(KB-106). VPN drops about every 10 minutes on home and hotel Wi-Fi, also during Teams calls while plugged in; "
    "ticket INC-48213 open since yesterday with no callback. Windows 11 24H2, GlobalProtect 6.2.1. Expense tool "
    "slowness is a side effect of the VPN drops."
)


def responder(req: MockRequest) -> str:
    if "conversation summarizer" in req.system_text:
        return SUMMARY_TEXT  # note: it silently dropped "already tried reinstalling"
    text = req.all_text
    has = lambda s: s in text  # noqa: E731
    device = "ORB-55812 on Windows 11 24H2" if has("ORB-55812") and has("24H2") else "your laptop"
    version = "GlobalProtect 6.2.1 to 6.2.4+" if has("GlobalProtect 6.2.1") else "GlobalProtect to 6.2.4+"
    steps = []
    if has("KB-101"):
        steps.append(f"1. Upgrade {version} from Company Portal (KB-101). {device} matches the known "
                     "24H2 bug, and telemetry shows ROUTE_TABLE_RESET right after each DHCP renewal.")
    else:
        steps.append("1. Check your VPN client version against the approved list.")
    if has("INC-48213"):
        steps.append("2. If 6.2.4 is not listed yet, IT can push it via Intune in about 15 minutes; I have added this to INC-48213.")
    else:
        steps.append("2. If 6.2.4 is not listed yet, IT can push it via Intune; I will open a new ticket for that.")
    if has("reinstall"):
        steps.append("3. Skip another same-version reinstall - it cannot fix this bug (KB-110).")
    else:
        steps.append("3. Try reinstalling GlobalProtect from Company Portal (KB-110).")
    if has("KB-104"):
        steps.append("4. MTU (KB-104) is unlikely to be the main cause because you also drop on home Wi-Fi; 6.2.4 handles MTU anyway.")
    trip = " before your Singapore trip on Monday" if has("Singapore") else ""
    steps.append(f"5. Re-test on home Wi-Fi for 30 minutes after the upgrade{trip}.")
    return "\n".join(steps)


# ---------------------------------------------------------------------------
# 7. Server-side context management (live-only)
# ---------------------------------------------------------------------------
CONTEXT_EDITING_SNIPPET = '''\
resp = client.beta.messages.create(
    betas=["context-management-2025-06-27"],
    model="claude-opus-5-5", max_tokens=1024, tools=tools, messages=messages,
    context_management={"edits": [{
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 30000},       # start clearing above this
        "keep": {"type": "tool_uses", "value": 3},                 # keep the 3 most recent tool results
        "clear_at_least": {"type": "input_tokens", "value": 5000},  # clear in big batches (cache-friendly)
        "exclude_tools": ["get_user_profile"],                     # never clear these
    }]},
)
print(resp.context_management)  # which edits were applied, tokens cleared'''

COMPACTION_SNIPPET = '''\
resp = client.beta.messages.create(
    betas=["compact-2026-01-12"],
    model="claude-opus-5-5", max_tokens=16000, messages=messages,
    context_management={"edits": [{
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": 150000},      # default trigger
        "instructions": "Preserve asset tags, ticket ids, versions and deadlines verbatim.",
    }]},
)
messages.append({"role": "assistant", "content": resp.content})  # keep the compaction block!'''


def run_server_side(client: Any) -> None:
    short = [{"role": "user", "content": "My VPN drops every ten minutes on Windows 11 24H2. Ideas?"}]
    tool = {"name": "get_device_telemetry", "description": "Fetch recent device event logs.",
            "input_schema": {"type": "object", "properties": {"device": {"type": "string"}}, "required": ["device"]}}
    r1 = client.beta.messages.create(
        betas=["context-management-2025-06-27"], model=MODEL_OPUS, max_tokens=1024, tools=[tool], messages=short,
        context_management={"edits": [{
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 30000},
            "keep": {"type": "tool_uses", "value": 3},
            "clear_at_least": {"type": "input_tokens", "value": 5000},
            "exclude_tools": ["get_user_profile"],
        }]},
    )
    print(f"  live context editing -> stop_reason={r1.stop_reason} context_management={r1.context_management}")
    r2 = client.beta.messages.create(
        betas=["compact-2026-01-12"], model=MODEL_OPUS, max_tokens=4000, messages=short,
        context_management={"edits": [{
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": 150000},
            "instructions": "Preserve asset tags, ticket ids, versions and deadlines verbatim.",
        }]},
    )
    print(f"  live compaction      -> stop_reason={r2.stop_reason} block types={[b.type for b in r2.content]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))


def main() -> None:
    banner("D2.4 - Context Window Optimization and Token Management")
    client = get_client(responder)
    tc = TokenCounter(client)
    system: str = CONVO["system"]
    turns: list[dict[str, str]] = CONVO["turns"]
    question: str = CONVO["question"]
    critical: list[str] = CONVO["critical_facts"]

    # ---------------------------------------------------------------- 1
    section("1. Pre-flight: count_tokens on the naive request")
    raw_docs = render_docs(KB)
    raw_tel = render_telemetry(TELEMETRY, pretty=True)
    naive_msgs = turns + [final_turn(question, raw_docs, raw_tel)]
    naive_total = tc.request(naive_msgs, system)
    parts = {
        "system prompt": tc.text(system),
        "KB articles (12, raw)": tc.text(raw_docs),
        "history (22 msgs)": tc.request(turns + [final_turn(question)]) - tc.request([final_turn(question)]),
        "telemetry (raw JSON)": tc.text(raw_tel),
        "question": tc.text(question),
    }
    for name, n in parts.items():
        print(f"  {name:<24} {n:>6,} tok  {'#' * max(1, n // 100)}")
    input_budget = CONTEXT_BUDGET - OUTPUT_RESERVE
    print(f"  {'TOTAL (count_tokens)':<24} {naive_total:>6,} tok   budget {CONTEXT_BUDGET:,} = {input_budget:,} input + {OUTPUT_RESERVE} output reserve")
    print(f"  -> {'OVER' if naive_total > input_budget else 'within'} budget by {abs(naive_total - input_budget):,} tokens. "
          "A 1M window would accept it, but every token costs money, latency and attention.")

    # ---------------------------------------------------------------- 2
    section("2. Budget allocator (raw demands)")
    fixed = parts["system prompt"] + parts["question"] + 30  # + framing tags
    available = input_budget - fixed
    weights = {"docs": 0.5, "history": 0.3, "tools": 0.2}
    raw_buckets = [Bucket("docs", parts["KB articles (12, raw)"], weights["docs"]),
                   Bucket("history", parts["history (22 msgs)"], weights["history"]),
                   Bucket("tools", parts["telemetry (raw JSON)"], weights["tools"])]
    alloc_raw = allocate(available, raw_buckets)
    print(f"  available for variable context = {input_budget:,} - fixed {fixed} (system+question+framing) = {available:,}")
    table(["bucket", "weight", "demand", "allocated", "fits?"],
          [[b.name, f"{b.weight:.1f}", f"{b.demand:,}", f"{alloc_raw[b.name]:,}", "yes" if alloc_raw[b.name] >= b.demand else "NO"]
           for b in raw_buckets])
    print("  Everything is squeezed: clean the inputs before deciding what to drop.")

    # ---------------------------------------------------------------- 3
    section("3. Cleanup: strip boilerplate, trim the verbose tool result")
    clean_kb = [{**d, "body": strip_boilerplate(d["body"])} for d in KB]
    clean_docs_tok = tc.text(render_docs(clean_kb))
    print(f"  KB boilerplate stripped     : {parts['KB articles (12, raw)']:>5,} -> {clean_docs_tok:>5,} tok")
    keep_fields = ["ts", "source", "level", "message"]
    trimmed = trim_tool_result(TELEMETRY, keep_sources={"vpn"}, min_level="Warning", fields=keep_fields, max_items=6)
    tel_block = render_telemetry(trimmed, pretty=False)
    tel_tok = tc.text(tel_block)
    print(f"  telemetry trimmed           : {parts['telemetry (raw JSON)']:>5,} -> {tel_tok:>5,} tok "
          f"({trimmed['summary']['returned']}/{trimmed['summary']['total_events']} events, "
          f"{len(keep_fields)} of {len(TELEMETRY['events'][0])} fields, compact JSON + summary)")
    buckets = [Bucket("docs", clean_docs_tok, weights["docs"]),
               Bucket("history", parts["history (22 msgs)"], weights["history"]),
               Bucket("tools", tel_tok, weights["tools"])]
    alloc = allocate(available, buckets)
    print("  Re-allocation on cleaned demands (tools' unused share spills over to docs/history):")
    table(["bucket", "demand", "allocated", "fits?"],
          [[b.name, f"{b.demand:,}", f"{alloc[b.name]:,}", "yes" if alloc[b.name] >= b.demand else "NO"] for b in buckets])

    # ---------------------------------------------------------------- 4
    section("4. Relevance filtering: BM25 rank, keep what is relevant AND fits")
    facts = extract_facts(turns)
    query = question + " " + " ".join(facts.values())
    scores = bm25(query, [d["title"] + "\n" + d["body"] for d in clean_kb])
    ranked = sorted(zip(clean_kb, scores), key=lambda p: (-p[1], p[0]["id"]))
    top = ranked[0][1]
    kept_docs: list[dict[str, str]] = []
    used = 0
    rows = []
    for d, s in ranked:
        cost = est(render_docs([d]))
        if s < 0.35 * top:
            verdict = "drop (low relevance)"
        elif used + cost > alloc["docs"] * SAFETY_MARGIN:
            verdict = "drop (over budget)"
        else:
            kept_docs.append(d)
            used += cost
            verdict = "KEEP"
        rows.append([d["id"], f"{s:5.2f}", f"~{cost}", verdict, d["title"][:52]])
    table(["doc", "bm25", "tok", "decision", "title"], rows)
    docs_block = render_docs(kept_docs)
    docs_tok = tc.text(docs_block)
    print(f"  KB context: {parts['KB articles (12, raw)']:,} raw -> {clean_docs_tok:,} clean -> {docs_tok:,} tok "
          f"({len(kept_docs)} of {len(KB)} articles)")

    # ---------------------------------------------------------------- 5
    section("5. History strategies: tokens vs fact recall")
    budget_h = alloc["history"]
    q_only = tc.request([final_turn(question)])
    strategies: dict[str, tuple[list[dict[str, Any]], str, float]] = {}
    full_tok = parts["history (22 msgs)"]
    strategies["full history"] = (list(turns), "", 0.0)
    window = sliding_window(turns, budget_h)
    strategies["sliding window"] = (window, "", 0.0)
    keep_last = 4
    summary, smsg = summarize_older(client, turns[:-keep_last])
    summary_msgs = [{"role": "user", "content": f"<conversation_summary>\n{summary}\n</conversation_summary>"}] + turns[-keep_last:]
    strategies["summary + last 4"] = (summary_msgs, "", cost_usd(MODEL_HAIKU, smsg.usage))
    pin = pinned_block(facts)
    pin_window = sliding_window(turns, budget_h - est(pin) - 8)
    strategies["pinned facts + window"] = (pin_window, pin, 0.0)
    rows = []
    results: dict[str, tuple[int, list[str]]] = {}
    for name, (msgs, pinned, extra) in strategies.items():
        tok = full_tok if name == "full history" else tc.request(msgs + [final_turn(question, pinned=pinned)]) - q_only
        context_text = pinned + "\n" + "\n".join(str(m["content"]) for m in msgs)
        kept = recall(context_text, critical)
        lost = [f for f in critical if f not in kept]
        results[name] = (tok, lost)
        rows.append([name, f"{tok:,}", "yes" if tok <= budget_h else "NO", f"{len(kept)}/{len(critical)}",
                     ", ".join(lost) or "-", f"${extra:.6f}" if extra else "-"])
    table(["strategy", "tokens", f"fits {budget_h}?", "facts", "lost", "extra call"], rows)
    print("  pinned facts:")
    for line in pin.splitlines()[1:-1]:
        print("    " + line)
    viable = [(n, t, lost) for n, (t, lost) in results.items() if t <= budget_h]
    best = min(viable, key=lambda v: (len(v[2]), v[1]))
    print(f"  -> choose '{best[0]}': highest recall within budget. The summarizer silently dropped a detail;")
    print("     sliding windows forget the oldest turns - often where identifiers were given.")

    # ---------------------------------------------------------------- 6
    section("6. Assemble, verify with count_tokens, call the model")
    msgs_best, pinned_best, _ = strategies[best[0]]
    optimized = msgs_best + [final_turn(question, docs_block, tel_block, pinned_best)]
    opt_total = tc.request(optimized, system)
    print(f"  optimized request: {opt_total:,} tok (budget {input_budget:,}) -> {'OK' if opt_total <= input_budget else 'STILL OVER'}")
    variants = [
        ("naive (everything)", naive_msgs),
        ("sliding window only", window + [final_turn(question, docs_block, tel_block)]),
        ("optimized", optimized),
    ]
    rows = []
    answers: dict[str, str] = {}
    for name, msgs in variants:
        m = client.messages.create(model=MODEL, max_tokens=OUTPUT_RESERVE, system=system, messages=msgs,
                                   output_config={"effort": "medium"})
        if m.stop_reason == "max_tokens":
            print(f"  {name}: answer truncated - the output reserve was too small")
        answers[name] = text_of(m)
        rows.append([name, f"{m.usage.input_tokens:,}", f"{m.usage.output_tokens}", f"${cost_usd(MODEL, m.usage):.5f}",
                     f"{len(recall(text_of(m), critical))}/{len(critical)}"])
    table(["variant", "input tok", "output tok", "cost", "critical facts in answer"], rows)
    for name in ("sliding window only", "optimized"):
        print(f"  answer ({name}):")
        for line in answers[name].splitlines():
            print("    | " + line)
    missing = [f for f in critical if f not in recall(answers["sliding window only"], critical)]
    print(f"  Window-only answer is missing {', '.join(missing) or 'nothing'} -> it offers to open a DUPLICATE ticket.")
    per_req = (naive_total - opt_total) * PRICING[MODEL]["input"] / 1e6
    print(f"  Saved {naive_total - opt_total:,} input tok/request ({1 - opt_total / naive_total:.0%}) = "
          f"${per_req * 100_000:,.2f} per 100k requests on Sonnet 5.5 - with a better answer than the window-only variant.")
    print(f"  ({tc.calls} count_tokens calls: free to call, but rate-limited - estimate locally, verify once per request.)")

    # ---------------------------------------------------------------- 7
    section("7. Server-side context management (live-only)")
    print("  Context editing - clears stale tool results/thinking in place, no summary:")
    for line in CONTEXT_EDITING_SNIPPET.splitlines():
        print("    " + line)
    print("  Compaction - summarizes earlier context into a compaction block near the limit:")
    for line in COMPACTION_SNIPPET.splitlines():
        print("    " + line)
    table(["mechanism", "what happens", "best for", "watch out"], [
        ["client-side (this lab)", "you trim/filter/summarize/pin", "every request; full control", "you own the logic + tests"],
        ["context editing", "server clears old tool_use/results", "long tool loops, stale results", "cache invalidates at edit point"],
        ["compaction", "server summarizes older turns", "chats nearing the window", "pass compaction block back"],
        ["memory tool", "model reads/writes /memories files", "facts across SESSIONS", "you implement storage"],
    ])
    if is_live():
        run_server_side(client)
    else:
        print("  [live-only] Set ANTHROPIC_API_KEY to send both requests (betas context-management-2025-06-27,")
        print("  compact-2026-01-12); the offline mock has no server-side context management.")

    section("Takeaways")
    print("  * Count before you send (count_tokens) and reserve output tokens up front.")
    print("  * Budget by bucket; let unused share spill over; clean inputs before cutting them.")
    print("  * Retrieve less but better: relevance-filter docs, trim tool results to needed fields.")
    print("  * Long histories: pin critical facts + window or summary; measure recall, not just tokens.")
    print("  * Server-side: context editing clears, compaction summarizes, memory persists across sessions.")


if __name__ == "__main__":
    main()
