"""
D3.8 - Progressive Discovery vs Monolithic Context

An enterprise assistant with 60 tools, 20 knowledge docs and 5 skills replays the same scripted
6-request session under four context strategies, and we measure what each request really costs:

  monolithic : every tool schema + every document in every request (prefix cached)
  swap       : small core + search_tools; the client REBUILDS `tools` per request from what was found
  append     : small STABLE core (search_tools, call_tool, load_doc); discovered schemas are appended
               to the conversation, never swapped into `tools` - the prefix stays byte-identical
  skills     : Agent-Skills-style metadata (name + description) in the system prompt; load_skill
               reveals SKILL.md + the schemas it needs, on demand

Metrics per strategy: API round trips, prompt tokens per call, cache writes/reads (from the mock's
prompt-caching simulation), cost with and without caching, and tool-selection ambiguity.
The server-side tool search request shape (tool_search_tool_bm25 + defer_loading) is shown live-only.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_SONNET,
    PRICING,
    MockRequest,
    banner,
    cost_usd,
    get_client,
    is_live,
    section,
    text_block,
    text_of,
    tool_use_block,
)

DATA = Path(__file__).resolve().parents[1] / "data"
MODEL = MODEL_SONNET  # high-volume enterprise assistant tier
CATALOG: list[dict[str, Any]] = json.loads((DATA / "tool_catalog.json").read_text(encoding="utf-8"))["tools"]
API_TOOLS = [{k: t[k] for k in ("name", "description", "input_schema")} for t in CATALOG]  # wire format
TOOL_BY_NAME = {t["name"]: t for t in API_TOOLS}
DOCS: list[dict[str, str]] = json.loads((DATA / "knowledge_docs.json").read_text(encoding="utf-8"))["docs"]
DOC_BY_ID = {d["id"]: d for d in DOCS}
SESSION = json.loads((DATA / "session.json").read_text(encoding="utf-8"))
REQUESTS: list[dict[str, Any]] = SESSION["requests"]


def compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


# =============================================================================
# 1. Skills: progressive disclosure in three levels (metadata -> SKILL.md -> bundled files)
# =============================================================================
@dataclass
class Skill:
    name: str
    description: str
    tools: list[str]
    docs: list[str]
    body: str
    files: dict[str, str]


def load_skills() -> list[Skill]:
    skills = []
    for d in sorted((DATA / "skills").iterdir()):
        raw = (d / "SKILL.md").read_text(encoding="utf-8").replace("\r\n", "\n")
        _, front, body = raw.split("---\n", 2)
        meta = {}
        for line in front.strip().split("\n"):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
        files = {f.name: f.read_text(encoding="utf-8") for f in sorted(d.iterdir()) if f.name != "SKILL.md"}
        skills.append(Skill(meta["name"], meta["description"], [t.strip() for t in meta["tools"].split(",")],
                            [x.strip() for x in meta["docs"].split(",")], body.strip(), files))
    return skills


SKILLS = load_skills()
SKILL_BY_NAME = {s.name: s for s in SKILLS}


# =============================================================================
# 2. Local tool search (BM25 over name + description) - same tokenizer/formula as the TS port
# =============================================================================
STOPWORDS = frozenset("a an and are as at be by can do does for from has have how i if in is it its of on or "
                      "the their this to use when with you your".split())


def tokenize(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOPWORDS:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class ToolSearch:
    def __init__(self, tools: list[dict[str, Any]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = {t["name"]: tokenize(t["name"] + " " + t["description"]) for t in tools}
        self.df: dict[str, int] = {}
        total = 0
        for toks in self.docs.values():
            total += len(toks)
            for t in dict.fromkeys(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = total / len(self.docs)

    def scores(self, query: str) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(tokenize(query)))
        n = len(self.docs)
        out = []
        for name, toks in self.docs.items():
            s = 0.0
            for t in terms:
                f = toks.count(t)
                if f == 0:
                    continue
                df = self.df[t]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl))
            if s > 0:
                out.append((name, s))
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    def search(self, query: str, k: int = 3) -> list[str]:
        return [name for name, _ in self.scores(query)[:k]]


SEARCH = ToolSearch(API_TOOLS)


# =============================================================================
# 3. Prompts and meta-tools for each strategy
# =============================================================================
BASE_SYSTEM = """You are Northwind's enterprise assistant. You help employees with HR, payroll, expenses, procurement, IT, facilities, sales (CRM), legal and analytics requests by calling internal tools and reading internal policy documents.

Operating rules:
- Act only through tools. Never invent balances, booking IDs, request numbers, approvals or policy terms; if a tool or document does not give you the fact, say so and explain how the employee can get it.
- Prefer the most specific tool for the job. Several tools look alike (for example HR balances vs payroll accruals); read descriptions carefully and pick the one that answers the exact question asked.
- Resolve relative dates against today's date, which is 2026-10-11, and state absolute dates in answers.
- Before any action that changes state (bookings, requests, resets), make sure all required inputs are known. If something essential is missing, ask one short clarifying question instead of guessing.
- Policy answers must cite the document ID they come from, for example (DOC-04).
- Keep answers to one to three sentences unless the employee asks for detail. Plain business English, no internal jargon, no emojis.
- Never reveal secrets, credentials or other employees' personal data beyond what the requester is entitled to see; for payroll and HR data, only the employee, their manager or HR may see the details.
- Security incidents (lost devices, phishing) must be reported to the Security Operations Center within one hour; tell the employee this whenever it is relevant.
- If a request falls outside these domains, say so politely and suggest the right team.
- When a tool returns an error, read it, correct your input or choose a different approach; do not repeat the identical failing call.
- Money: state currency and amounts exactly as tools return them; do not round totals that drive approvals.
- People: refer to employees by full name as given; never guess an employee ID or username.
- Approvals: explain who must approve and why (threshold, policy) whenever you create a request that needs approval.
- Accessibility: if the employee mentions an accessibility need, offer the accessible option first (rooms, parking, formats).
- Escalation: for legal, medical, harassment or safety concerns, do not try to resolve them yourself; point to HR, Legal or Security.
- Audit: every state-changing action you take is logged with the employee's identity; never perform actions on behalf of someone else unless the request clearly comes from their manager or an administrator."""


def system_for(mode: str) -> str:
    if mode == "monolithic":
        docs = "\n\n".join(f"[{d['id']}] {d['title']}\n{d['body']}" for d in DOCS)
        return f"{BASE_SYSTEM}\n\n# Knowledge base (all documents)\n{docs}"
    if mode == "skills":
        meta = "\n".join(f"- {s.name}: {s.description}" for s in SKILLS)
        return f"{BASE_SYSTEM}\n\n# Skills (call load_skill to get full instructions before acting)\n{meta}"
    index = "\n".join(f"{d['id']}  {d['title']} - {d['summary']}" for d in DOCS)
    return f"{BASE_SYSTEM}\n\n# Document index (call load_doc for the full text)\n{index}"


SEARCH_TOOLS = {"name": "search_tools",
                "description": "Search the catalog of 60 enterprise tools by keywords; returns the best matching tools. "
                               "Call this before using any capability you do not already have.",
                "input_schema": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "What you need to do, in keywords"},
                    "k": {"type": "integer", "description": "Maximum results (default 3)"}}, "required": ["query"]}}
LOAD_DOC = {"name": "load_doc",
            "description": "Load the full text of a knowledge document by ID from the document index. Use for policy questions.",
            "input_schema": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]}}
CALL_TOOL = {"name": "call_tool",
             "description": "Invoke a catalog tool whose schema was disclosed earlier in this conversation (by search_tools "
                            "or load_skill). Pass its name and arguments matching that schema.",
             "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                              "required": ["name", "arguments"]}}
LOAD_SKILL = {"name": "load_skill",
              "description": "Load a skill's full instructions (SKILL.md) plus the schemas of the tools it uses. Call it "
                             "before acting on a request that matches a skill description.",
              "input_schema": {"type": "object", "properties": {"name": {"type": "string", "enum": [s.name for s in SKILLS]}},
                               "required": ["name"]}}
CORE: dict[str, list[dict[str, Any]]] = {
    "monolithic": API_TOOLS,
    "swap": [SEARCH_TOOLS, LOAD_DOC],
    "append": [SEARCH_TOOLS, CALL_TOOL, LOAD_DOC],
    "skills": [LOAD_SKILL, CALL_TOOL, LOAD_DOC],
}
STRATEGY_LABEL = {"monolithic": "monolithic", "swap": "progressive-swap", "append": "progressive-append", "skills": "skills-style"}


# =============================================================================
# 4. Fake enterprise backends + the harness that runs one strategy over the session
# =============================================================================
def backend(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    schema = TOOL_BY_NAME[name]["input_schema"]
    missing = [r for r in schema.get("required", []) if r not in args]
    if missing:
        return f"missing required argument(s) for {name}: {', '.join(missing)}", True
    for r in REQUESTS:
        if r.get("tool") == name and r.get("args") == args:
            return compact(r["result"]), False
        if r.get("distractor") == name and r.get("args") == args:
            return compact(r["distractor_result"]), False
    return compact({"ok": True, "tool": name}), False


@dataclass
class CallStat:
    request: str
    prompt: int
    cache_write: int
    cache_read: int
    output: int
    cost: float
    visible_tools: int  # tool definitions the model can see when choosing (tools array + disclosed schemas)

    @property
    def cache_event(self) -> str:
        return "W" if self.cache_write else ("R" if self.cache_read else "-")


@dataclass
class Runner:
    client: Any
    mode: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    disclosed: list[str] = field(default_factory=list)  # append/skills: schemas living in the conversation
    request_tools: list[str] = field(default_factory=list)  # swap: tools rebuilt for the current request
    calls: list[CallStat] = field(default_factory=list)
    skills_loaded: list[str] = field(default_factory=list)
    docs_loaded: list[str] = field(default_factory=list)

    def tools(self) -> list[dict[str, Any]]:
        if self.mode == "swap":
            return CORE["swap"] + [TOOL_BY_NAME[n] for n in self.request_tools]
        return CORE[self.mode]

    def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        if name == "search_tools":
            hits = SEARCH.search(args.get("query", ""), int(args.get("k", 3)))
            if self.mode == "swap":  # schemas go into `tools` (prefix changes!), result lists names only
                self.request_tools.extend(h for h in hits if h not in self.request_tools)
                return compact([{"name": h, "description": TOOL_BY_NAME[h]["description"]} for h in hits]), False
            self.disclosed.extend(h for h in hits if h not in self.disclosed)  # schemas appended to conversation
            return compact([TOOL_BY_NAME[h] for h in hits]), False
        if name == "load_doc":
            doc = DOC_BY_ID.get(args.get("doc_id", ""))
            if doc is None:
                return f"unknown document {args.get('doc_id')}", True
            self.docs_loaded.append(doc["id"])
            return f"[{doc['id']}] {doc['title']}\n{doc['body']}", False
        if name == "load_skill":
            skill = SKILL_BY_NAME.get(args.get("name", ""))
            if skill is None:
                return f"unknown skill {args.get('name')}", True
            self.skills_loaded.append(skill.name)
            self.disclosed.extend(t for t in skill.tools if t not in self.disclosed)
            files = ", ".join(skill.files) if skill.files else "none"
            return (f"{skill.body}\n\nBundled files (read on demand): {files}\n"
                    f"Tool schemas: {compact([TOOL_BY_NAME[t] for t in skill.tools])}"), False
        if name == "call_tool":
            target = args.get("name", "")
            if target not in self.disclosed:
                return f"tool '{target}' has not been disclosed yet - call search_tools or load_skill first", True
            return backend(target, args.get("arguments") or {})
        if name not in {t["name"] for t in self.tools()}:
            return f"tool '{name}' is not available in this request", True
        return backend(name, args)

    def run_request(self, req: dict[str, Any]) -> str:
        self.messages.append({"role": "user", "content": req["user"]})
        if self.mode == "swap":
            self.request_tools = []  # rebuild the tool set from scratch for every request
        for _ in range(8):
            resp = self.client.messages.create(
                model=MODEL, max_tokens=1024,
                system=[{"type": "text", "text": system_for(self.mode), "cache_control": {"type": "ephemeral"}}],
                tools=self.tools(), messages=self.messages)
            u = resp.usage
            visible = len(self.tools()) + (len(self.disclosed) if self.mode in ("append", "skills") else 0)
            self.calls.append(CallStat(req["id"], u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0),
                                       u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0,
                                       u.output_tokens, cost_usd(resp.model, u), visible))
            self.messages.append({"role": "assistant", "content": resp.content})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not calls:
                return text_of(resp).strip()
            results = []
            for c in calls:
                out, is_error = self.execute(c.name, c.input)
                r: dict[str, Any] = {"type": "tool_result", "tool_use_id": c.id, "content": out}
                if is_error:
                    r["is_error"] = True
                results.append(r)
            self.messages.append({"role": "user", "content": results})
        return "(max turns)"


# =============================================================================
# 5. Offline responder: plays the assistant for every strategy
# =============================================================================
def _conversation(req: MockRequest) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any], str, bool]], list[tuple[str, dict[str, Any], str, bool]]]:
    """Return (current scripted request, tool calls this turn, tool calls in the whole conversation)."""
    uses: dict[str, tuple[str, dict[str, Any]]] = {}
    all_calls: list[tuple[str, dict[str, Any], str, bool]] = []
    turn_start = 0
    current = ""
    for m in req.messages:
        c = m.get("content")
        if m.get("role") == "user" and isinstance(c, str):
            current, turn_start = c, len(all_calls)
            continue
        for b in c if isinstance(c, list) else []:
            if b.get("type") == "tool_use":
                uses[b["id"]] = (b["name"], b["input"])
            elif b.get("type") == "tool_result" and b["tool_use_id"] in uses:
                name, inp = uses[b["tool_use_id"]]
                content = b.get("content")
                text = content if isinstance(content, str) else "".join(x.get("text", "") for x in content or [])
                all_calls.append((name, inp, text, bool(b.get("is_error"))))
    entry = next(r for r in REQUESTS if r["user"] == current)
    return entry, all_calls[turn_start:], all_calls


def _reply(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    # Language-neutral output-token estimate so the Python and TS ports report identical usage.
    return {"content": blocks, "usage": {"output_tokens": max(1, len(compact(blocks)) // 4)}}


def _answer(entry: dict[str, Any]) -> dict[str, Any]:
    result = entry.get("result") or {}
    return _reply([text_block(re.sub(r"\{(\w+)\}", lambda m: str(result[m.group(1)]), entry["answer"]))])


def responder(req: MockRequest) -> Any:
    entry, turn, allc = _conversation(req)
    names = req.tool_names
    turn_names = [c[0] for c in turn]
    tool, doc = entry.get("tool"), entry.get("doc")
    called_target = any(c[0] == "call_tool" and c[1].get("name") == tool for c in turn) or tool in turn_names
    loaded_docs = [c[1].get("doc_id") for c in allc if c[0] == "load_doc"]
    if "load_skill" in names:
        mode = "skills"
    elif "call_tool" in names:
        mode = "append"
    elif "search_tools" in names:
        mode = "swap"
    else:
        mode = "monolithic"

    if mode == "monolithic":
        if tool and not called_target:
            distractor = entry.get("distractor")
            if distractor and distractor not in turn_names:  # simulated pick of a look-alike tool
                return _reply([tool_use_block(distractor, entry["args"])])
            return _reply([tool_use_block(tool, entry["args"])])
        return _answer(entry)  # docs are already in the system prompt

    if mode == "swap":
        if tool and tool not in names:
            blocks = [tool_use_block("search_tools", {"query": entry["search"]})]
            if doc and "load_doc" not in turn_names:
                blocks.append(tool_use_block("load_doc", {"doc_id": doc}))
            return _reply(blocks)
        if tool and not called_target:
            return _reply([tool_use_block(tool, entry["args"])])
        if not tool and doc and "load_doc" not in turn_names:
            return _reply([tool_use_block("load_doc", {"doc_id": doc})])
        return _answer(entry)

    disclosed: list[str] = []
    for c in allc:
        if c[0] == "search_tools":
            disclosed.extend(t["name"] for t in json.loads(c[2]))
        elif c[0] == "load_skill" and not c[3]:
            disclosed.extend(SKILL_BY_NAME[c[1]["name"]].tools)
    blocks = []
    if mode == "skills" and entry["skill"] not in [c[1].get("name") for c in allc if c[0] == "load_skill"]:
        return _reply([tool_use_block("load_skill", {"name": entry["skill"]})])
    if mode == "append" and tool and tool not in disclosed:
        blocks.append(tool_use_block("search_tools", {"query": entry["search"]}))
        if doc and doc not in loaded_docs:
            blocks.append(tool_use_block("load_doc", {"doc_id": doc}))
        return _reply(blocks)
    if doc and doc not in loaded_docs:
        blocks.append(tool_use_block("load_doc", {"doc_id": doc}))
    if tool and not called_target:
        blocks.append(tool_use_block("call_tool", {"name": tool, "arguments": entry["args"]}))
    return _reply(blocks) if blocks else _answer(entry)


# =============================================================================
# Demo
# =============================================================================
TOOL_SEARCH_SNIPPET = """client.messages.create(
    model="claude-sonnet-5-5", max_tokens=1024,
    tools=[
        {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
        LOAD_DOC,                                         # small always-loaded core
        *[{**t, "defer_loading": True} for t in CATALOG],  # 60 tools known, NOT loaded
    ],
    messages=[{"role": "user", "content": "Reset VPN access for user jdoe."}],
)"""


def count(client: Any, system: str | None = None, tools: list[dict[str, Any]] | None = None) -> int:
    kwargs: dict[str, Any] = {"model": MODEL, "messages": [{"role": "user", "content": "x"}]}
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    return client.messages.count_tokens(**kwargs).input_tokens


def main() -> None:
    banner("D3.8 - Progressive Discovery vs Monolithic Context")
    client = get_client(responder)

    section("1. What could go into context")
    base = count(client)
    sizes = [
        ("60 tool schemas", count(client, tools=API_TOOLS) - base),
        ("20 documents, full text", count(client, system="\n\n".join(f"[{d['id']}] {d['title']}\n{d['body']}" for d in DOCS)) - base),
        ("document index (id + title + summary)", count(client, system="\n".join(f"{d['id']}  {d['title']} - {d['summary']}" for d in DOCS)) - base),
        ("5 skill metadata lines", count(client, system="\n".join(f"- {s.name}: {s.description}" for s in SKILLS)) - base),
        ("5 SKILL.md bodies", count(client, system="\n\n".join(s.body for s in SKILLS)) - base),
        ("core meta-tools (search_tools, call_tool, load_doc)", count(client, tools=CORE["append"]) - base),
        ("base system prompt", count(client, system=BASE_SYSTEM) - base),
    ]
    for label, n in sizes:
        print(f"  {label:<52} ~{n:>6,} tokens  (count_tokens)")
    domains = sorted({t["domain"] for t in CATALOG})
    print(f"  {len(API_TOOLS)} tools across {len(domains)} domains: {', '.join(domains)}")

    section("2. Scripted session x 4 strategies (" + MODEL + ")")
    print("Same 6 requests, same conversation, four ways of building context. Cells: API calls / prompt tokens.")
    runners: dict[str, Runner] = {}
    finals: dict[str, list[str]] = {}
    for mode in ("monolithic", "swap", "append", "skills"):
        r = Runner(client, mode)
        finals[mode] = [r.run_request(q) for q in REQUESTS]
        runners[mode] = r
    print(f"  {'req':<5}" + "".join(f"{STRATEGY_LABEL[m]:>22}" for m in runners))
    for q in REQUESTS:
        cells = []
        for r in runners.values():
            cs = [c for c in r.calls if c.request == q["id"]]
            tok = 0
            for c in cs:
                tok += c.prompt
            cells.append(f"{len(cs)} / {tok:,}")
        print(f"  {q['id']:<5}" + "".join(f"{c:>22}" for c in cells))
    same = all(finals[m] == finals["monolithic"] for m in finals)
    print(f"  answers identical across strategies: {'yes' if same else 'no'}")
    print(f"  e.g. R4 -> {finals['append'][3]}")

    section("3. Totals: tokens, cost, round trips")
    p = PRICING[MODEL]
    print(f"  {'strategy':<20}{'calls':>6}{'prompt tok':>12}{'avg/call':>10}{'cached':>8}{'writes':>8}"
          f"{'cost':>10}{'no-cache cost':>15}")
    for mode, r in runners.items():
        prompt = writes = reads = out = 0
        cost = 0.0
        for c in r.calls:
            prompt += c.prompt
            writes += c.cache_write
            reads += c.cache_read
            out += c.output
            cost += c.cost
        no_cache = (prompt * p["input"] + out * p["output"]) / 1e6
        print(f"  {STRATEGY_LABEL[mode]:<20}{len(r.calls):>6}{prompt:>12,}{prompt // len(r.calls):>10,}"
              f"{100 * reads / prompt:>7.0f}%{writes:>8,}{'$' + format(cost, '.4f'):>10}{'$' + format(no_cache, '.4f'):>15}")
    mono = runners["monolithic"]
    mono_prompt = 0
    for c in mono.calls:
        mono_prompt += c.prompt
    projected = mono_prompt // len(mono.calls) + 4 * sizes[0][1]
    print(f"  Scale check: with 5x the tools (~300, e.g. several MCP servers) monolithic grows to ~{projected:,} tokens")
    print("  per call; progressive per-call context stays roughly flat because only matches are disclosed.")
    print("  Prompt caching makes the monolithic prefix cheap to RE-read, but it still occupies the context")
    print("  window on every call (crowding out conversation and retrieved data), raises time-to-first-token,")
    print("  and every cache miss (TTL expiry, any prefix edit) re-bills the whole thing.")
    print("  (Offline mock: only the tools+system prefix is cached; with message caching the conversation")
    print("  tail would be cheaper too - the per-call context size is the robust signal here.)")

    section("4. Cache behaviour per call (W = cache write, R = cache read)")
    for mode, r in runners.items():
        groups = []
        for q in REQUESTS:
            groups.append(" ".join(c.cache_event for c in r.calls if c.request == q["id"]))
        print(f"  {STRATEGY_LABEL[mode]:<20}" + " | ".join(groups))
    print("  swap rebuilds `tools` per request: tools render first, so every new tool set is a new prefix")
    print("  (a cache WRITE) - the cache only helps when an identical set happens to recur.")
    print("  append keeps `tools` + system byte-identical and adds discovered schemas to the conversation,")
    print("  so after the first call every request reads the cached prefix. Server-side tool search works")
    print("  the same way: discovered definitions are appended inline, not swapped into the prefix.")

    section("5. Tool-selection pressure")
    print("  Tool definitions visible when the model chooses (max per request; append/skills count disclosed schemas):")
    print(f"  {'req':<5}" + "".join(f"{STRATEGY_LABEL[m]:>20}" for m in runners) + "   search_tools top-3")
    for q in REQUESTS:
        vis = [max(c.visible_tools for c in r.calls if c.request == q["id"]) for r in runners.values()]
        top = ", ".join(SEARCH.search(q["search"])) if q.get("search") else "(docs only)"
        print(f"  {q['id']:<5}" + "".join(f"{v:>20}" for v in vis) + f"   {top}")
    print("  Look-alikes such as hr_get_pto_balance vs payroll_get_leave_accrual ('leave days for an employee')")
    print("  compete only when both are in view. In the monolithic run R1 picked the payroll tool first and")
    print("  paid an extra round trip (simulated, but a typical large-catalog error - measure it with evals).")
    print("  Note the flip side: append/skills accumulate disclosed schemas in history (trim per D2.4).")

    section("6. Skills: three levels of progressive disclosure")
    sk = runners["skills"]
    meta_tokens, body_tokens = sizes[3][1], sizes[4][1]
    print(f"  level 1  metadata for {len(SKILLS)} skills, always in the system prompt : ~{meta_tokens:,} tokens")
    print(f"  level 2  SKILL.md bodies loaded on demand              : {len(sk.skills_loaded)} loads "
          f"({', '.join(dict.fromkeys(sk.skills_loaded))}) of ~{body_tokens:,} tokens total")
    never = [f"{s.name}/{f}" for s in SKILLS for f in s.files]
    print(f"  level 3  bundled files, read only when needed           : {', '.join(never)} -> never loaded (0 tokens)")
    print("  On the Claude API, Agent Skills run in the code-execution container (see D2.5); the disclosure")
    print("  pattern is the same: descriptions decide WHEN a skill loads, so write them like tool descriptions.")

    section("7. Server-side tool search (live-only)")
    print("The production version of progressive discovery: declare all tools, defer most, let Claude search.")
    for line in TOOL_SEARCH_SNIPPET.splitlines():
        print("    " + line)
    if is_live():
        resp = client.messages.create(
            model=MODEL, max_tokens=1024,
            tools=[{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}, LOAD_DOC,
                   *[{**t, "defer_loading": True} for t in API_TOOLS]],
            messages=[{"role": "user", "content": "Reset VPN access for user jdoe."}])
        print(f"  live: stop_reason={resp.stop_reason}, blocks={[b.type for b in resp.content]}, "
              f"input_tokens={resp.usage.input_tokens}")
    else:
        print("  [live-only] the mock cannot run server tools. Never defer ALL tools (keep a small core loaded);")
        print("  discovered definitions are appended inline, so the cached prefix survives. Regex variant:")
        print("  tool_search_tool_regex_20251119. Rule of thumb: worth it once schemas exceed roughly 10K tokens.")

    section("8. Recommendation")
    rows = [
        ("Few tools (< ~15), small stable docs, one domain", "Monolithic + prompt caching"),
        ("Latency-critical, 1-2 calls per request", "Monolithic (or preloaded core set)"),
        ("Dozens-hundreds of tools / many MCP servers", "Server-side tool search, defer_loading"),
        ("Large per-task instructions and playbooks", "Skills (metadata -> SKILL.md -> files)"),
        ("Large or changing knowledge base", "load_doc / RAG on demand (D3.5)"),
        ("Dynamic tool sets mid-session", "Append-only (tool search / tool_addition),"),
        ("", "never swap the tools array"),
    ]
    for situation, choice in rows:
        print(f"  {situation:<50} {choice}")


if __name__ == "__main__":
    main()
