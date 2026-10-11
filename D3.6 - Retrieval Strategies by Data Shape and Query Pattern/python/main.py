"""
D3.6 - Retrieval Strategies by Data Shape and Query Pattern

One assistant, five data shapes, many query patterns - and no single retrieval method that
serves them all. This lab builds a query router and one working retriever per strategy:

  data shapes  : structured table (orders.csv), semi-structured JSON (catalog.json),
                 unstructured prose (policies.md), source code (data/code), knowledge graph
  router       : regex pre-router for unambiguous IDs -> Claude Haiku classifier with
                 structured output -> deterministic fallback rules on low confidence / errors
  strategies   : key lookup, SQL aggregation (sqlite3), attribute/metadata filter, BM25,
                 vector (local stand-in embeddings), hybrid RRF, code-aware BM25,
                 graph traversal, and an agentic tool-use loop for cross-source multi-hop

Retrieval code is real and deterministic; Claude calls go through the shared client
(offline mock by default, live with ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anthropic
import pydantic
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_HAIKU,
    MODEL_OPUS,
    MockRequest,
    banner,
    get_client,
    section,
    text_block,
    text_of,
    tool_use_block,
)

DATA = Path(__file__).resolve().parents[1] / "data"
QUERIES = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))
AS_OF: str = QUERIES["as_of"]  # fixed "today" so warranty math is deterministic


# =============================================================================
# 1. Load the five data shapes
# =============================================================================
def load_orders() -> list[dict[str, Any]]:
    rows = []
    with open(DATA / "orders.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["quantity"] = int(r["quantity"])
            r["unit_cents"] = round(float(r["unit_price"]) * 100)  # money as integer cents: exact sums
            rows.append(r)
    return rows


ORDERS = load_orders()
ORDERS_BY_ID = {o["order_id"]: o for o in ORDERS}
CATALOG: list[dict[str, Any]] = json.loads((DATA / "catalog.json").read_text(encoding="utf-8"))["products"]
CATALOG_BY_SKU = {p["sku"]: p for p in CATALOG}
GRAPH_DATA = json.loads((DATA / "graph.json").read_text(encoding="utf-8"))


@dataclass
class TextDoc:
    doc_id: str
    kind: str  # "policy" | "product" | "order" (metadata used for scoping)
    title: str
    text: str


def load_text_docs() -> list[TextDoc]:
    docs: list[TextDoc] = []
    body = (DATA / "policies.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    for block in body.split("\n## ")[1:]:
        title, _, text = block.partition("\n")
        docs.append(TextDoc(f"policy:{title.strip().lower().replace(' ', '-')}", "policy", title.strip(), text.strip()))
    for p in CATALOG:
        docs.append(TextDoc(f"product:{p['sku']}", "product", p["name"], f"{p['name']}. {p['description']}"))
    return docs


@dataclass
class CodeChunk:
    file: str
    start: int  # 1-based line numbers
    end: int
    symbol: str
    text: str


TOP_LEVEL = re.compile(r"^(def |async def |class |export function |function |export class |CREATE )")
SYMBOL = re.compile(r"(?:def|function|class|VIEW)\s+(\w+)")


def chunk_code(file: str, text: str) -> list[CodeChunk]:
    """Code-aware chunking: one chunk per top-level definition (leading comments attached);
    module-level constants/imports become a 'module header' chunk. Never split a function."""
    lines = text.replace("\r\n", "\n").rstrip("\n").split("\n")
    starts: list[int] = []
    for i, line in enumerate(lines):
        if TOP_LEVEL.match(line):
            j = i
            while j > 0 and lines[j - 1].strip().startswith(("//", "--", "#")):
                j -= 1
            starts.append(j)
    if not starts or starts[0] > 0:
        starts.insert(0, 0)
    chunks: list[CodeChunk] = []
    for n, s in enumerate(starts):
        e = (starts[n + 1] if n + 1 < len(starts) else len(lines)) - 1
        while e > s and not lines[e].strip():
            e -= 1
        seg = "\n".join(lines[s : e + 1])
        if not seg.strip():
            continue
        def_line = next((ln for ln in lines[s : e + 1] if TOP_LEVEL.match(ln)), None)
        m = SYMBOL.search(def_line) if def_line else None
        chunks.append(CodeChunk(file, s + 1, e + 1, m.group(1) if m else "(module header)", seg))
    return chunks


def load_code() -> list[CodeChunk]:
    out: list[CodeChunk] = []
    for p in sorted((DATA / "code").iterdir()):
        out.extend(chunk_code(p.name, p.read_text(encoding="utf-8")))
    return out


# =============================================================================
# 2. Text retrieval primitives (identical in the TypeScript port)
# =============================================================================
STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does don for from had has have how i if in into is it its
    just me my of on or our should so than that the their them then there these they this to too up us was we
    were what when where which while who why will with would you your t s anything good""".split()
)


def stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def code_tokenize(text: str) -> list[str]:
    """Identifier-aware: calculateShippingCost -> calculate shipping cost; RESTOCKING_FEE -> restocking fee."""
    return tokenize(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " "))


CONCEPT_GROUPS: dict[str, list[str]] = {
    "return": ["return", "send", "back", "exchange"],
    "open": ["open", "opened", "unsealed", "used"],
    "dislike": ["like", "dislike", "unhappy"],
    "charge": ["charge", "charging", "charger", "battery", "power"],
    "broken": ["broken", "stopped", "dead", "defect", "defective", "faulty", "failure", "dropout"],
    "warranty": ["warranty", "guarantee", "claim", "repair", "replace"],
    "water": ["water", "waterproof", "rain", "wet", "sweat", "splash"],
    "sport": ["sport", "running", "run", "gym", "workout", "cycling", "hiking", "outdoor"],
    "ship": ["ship", "shipping", "delivery", "deliver", "shipped", "courier"],
    "abroad": ["international", "abroad", "overseas"],
    "price": ["price", "cost", "fee", "surcharge", "usd"],
    "headphone": ["headphone", "headset", "earphone"],
    "earbud": ["earbud", "bud"],
    "quiet": ["noise", "cancelling", "anc", "quiet"],
}
CONCEPTS: dict[str, str] = {tok: canon for canon, toks in CONCEPT_GROUPS.items() for tok in toks}


def embed_features(text: str) -> list[str]:
    """Concept-mapped tokens; any token containing a digit collapses to '<num>' - this mimics how
    dense embeddings blur exact numbers, dates and IDs (the reason they fail at lookups and filters)."""
    out = []
    for t in tokenize(text):
        if t in CONCEPTS:
            out.append(CONCEPTS[t])
        elif re.search(r"\d", t):
            out.append("<num>")
        else:
            out.append(t)
    return out


def fnv1a(s: str) -> int:
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


EMBED_DIM = 256


class Embedder:
    """Deterministic local stand-in for an embedding model (signed feature hashing + frozen idf).
    Anthropic has no embeddings endpoint; production would call a partner model such as Voyage AI."""

    def __init__(self, corpus: list[str]) -> None:
        df: dict[str, int] = {}
        for text in corpus:
            for f in dict.fromkeys(embed_features(text)):
                df[f] = df.get(f, 0) + 1
        n = len(corpus)
        self.idf = {f: math.log((n + 1) / (c + 1)) + 1.0 for f, c in df.items()}
        self.default_idf = math.log(n + 1) + 1.0

    def embed(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for f in embed_features(text):
            counts[f] = counts.get(f, 0) + 1
        vec = [0.0] * EMBED_DIM
        for f, tf in counts.items():
            h = fnv1a("d36:" + f)
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            vec[h % EMBED_DIM] += sign * (1.0 + math.log(tf)) * self.idf.get(f, self.default_idf)
        sq = 0.0
        for x in vec:
            sq += x * x
        norm = math.sqrt(sq)
        return [x / norm for x in vec] if norm > 0 else vec


def dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


class BM25:
    def __init__(self, docs: dict[str, str], tokenizer: Any = tokenize, k1: float = 1.2, b: float = 0.75) -> None:
        self.tokenizer, self.k1, self.b = tokenizer, k1, b
        self.tf: dict[str, dict[str, int]] = {}
        self.len: dict[str, int] = {}
        self.df: dict[str, int] = {}
        for did, text in docs.items():
            toks = tokenizer(text)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.tf[did], self.len[did] = tf, len(toks)
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        total = 0
        for v in self.len.values():
            total += v
        self.avgdl = total / max(1, len(self.len))

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(self.tokenizer(query)))
        n = len(self.tf)
        scored = []
        for did, tf in self.tf.items():
            if allowed is not None and did not in allowed:
                continue
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                df = self.df[t]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[did] / self.avgdl))
            if s > 0:
                scored.append((did, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


class VectorIndex:
    def __init__(self, docs: dict[str, str], embedder: Embedder) -> None:
        self.embedder = embedder
        self.vecs = {did: embedder.embed(text) for did, text in docs.items()}

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        qv = self.embedder.embed(query)
        scored = [(did, dot(qv, v)) for did, v in self.vecs.items() if allowed is None or did in allowed]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60, top: int = 10) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (did, _) in enumerate(ranking, start=1):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top]


TEXT_DOCS = load_text_docs()
TEXT_BY_ID = {d.doc_id: d for d in TEXT_DOCS}
TEXT_EMBEDDER = Embedder([d.text for d in TEXT_DOCS])
TEXT_BM25 = BM25({d.doc_id: d.text for d in TEXT_DOCS})
TEXT_VEC = VectorIndex({d.doc_id: d.text for d in TEXT_DOCS}, TEXT_EMBEDDER)
CODE_CHUNKS = load_code()
CODE_BM25 = BM25({f"{c.file}:{c.start}": c.text for c in CODE_CHUNKS}, tokenizer=code_tokenize)


def scope(source: str) -> set[str] | None:
    """Metadata scoping: the router's data_source becomes a pre-filter on doc kind."""
    kind = {"policies": "policy", "catalog": "product"}.get(source)
    return None if kind is None else {d.doc_id for d in TEXT_DOCS if d.kind == kind}


def text_search(query: str, mode: str, source: str, k: int = 2) -> list[tuple[str, float]]:
    allowed = scope(source)
    if mode == "bm25":
        return TEXT_BM25.search(query, k, allowed)
    if mode == "vector":
        return TEXT_VEC.search(query, k, allowed)
    return rrf([TEXT_BM25.search(query, 20, allowed), TEXT_VEC.search(query, 20, allowed)], top=k)


# =============================================================================
# 3. Structured strategies: key lookup, SQL aggregation, attribute filter
# =============================================================================
ID_RE = re.compile(r"\b(ORD-\d{5}|[A-Z]{2}-\d{3})\b")


def key_lookup(query: str) -> list[str]:
    out = []
    for key in ID_RE.findall(query):
        if key in ORDERS_BY_ID:
            o = ORDERS_BY_ID[key]
            out.append(f"{key}: status={o['status']} placed={o['order_date']} customer={o['customer_name']} "
                       f"item={o['sku']} x{o['quantity']}")
        elif key in CATALOG_BY_SKU:
            p = CATALOG_BY_SKU[key]
            out.append(f"{key}: {p['name']} ${p['price']:.2f}")
        else:
            out.append(f"{key}: not found")
    return out


GROUP_COLUMNS = {"region": "region", "customer": "customer_name", "product": "sku", "sku": "sku", "status": "status"}


def plan_aggregation(query: str) -> dict[str, Any]:
    """Turn the question into a VALIDATED plan (allowlisted columns, bound parameters). The model or
    parser never writes raw SQL - that is what keeps text-to-SQL safe."""
    q = query.lower()
    m = re.search(r"\bby (region|customer|product|sku|status)\b", q)
    group = GROUP_COLUMNS[m.group(1)] if m else "region"
    qm = re.search(r"\bq([1-4]) (\d{4})\b", q)
    if qm:
        quarter, year = int(qm.group(1)), qm.group(2)
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        last_day = {3: 31, 6: 30, 9: 30, 12: 31}[end_month]
        date_from, date_to = f"{year}-{start_month:02d}-01", f"{year}-{end_month:02d}-{last_day}"
    else:
        date_from, date_to = "0000-01-01", "9999-12-31"
    return {"group_by": group, "date_from": date_from, "date_to": date_to, "exclude_status": ["cancelled", "returned"]}


def plan_to_sql(plan: dict[str, Any]) -> tuple[str, list[Any]]:
    col = plan["group_by"]
    if col not in GROUP_COLUMNS.values():
        raise ValueError(f"column not allowed: {col}")
    sql = (f"SELECT {col} AS grp, SUM(quantity * unit_cents) AS revenue_cents, COUNT(*) AS n_orders "
           "FROM orders WHERE order_date BETWEEN ? AND ? AND status NOT IN (?, ?) "
           "GROUP BY grp ORDER BY revenue_cents DESC, grp ASC")
    return sql, [plan["date_from"], plan["date_to"], *plan["exclude_status"]]


def run_aggregation(plan: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Python executes the plan with sqlite3 (stdlib); the TS port runs the same plan in memory."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE orders (order_id TEXT, order_date TEXT, customer_name TEXT, region TEXT, "
                 "sku TEXT, quantity INTEGER, unit_cents INTEGER, status TEXT)")
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)",
                     [(o["order_id"], o["order_date"], o["customer_name"], o["region"], o["sku"],
                       o["quantity"], o["unit_cents"], o["status"]) for o in ORDERS])
    sql, params = plan_to_sql(plan)
    rows = [(str(r[0]), int(r[1]), int(r[2])) for r in conn.execute(sql, params)]
    conn.close()
    return rows


def plan_filter(query: str) -> dict[str, Any]:
    q = query.lower()
    plan: dict[str, Any] = {}
    for cat in ("headphones", "earbuds", "speakers", "accessories"):
        if cat in q or cat[:-1] in q:
            plan["category"] = cat
            break
    for flag in ("wireless", "waterproof"):
        if flag in q:
            plan[flag] = True
    if "noise cancelling" in q or "anc" in q.split():
        plan["noise_cancelling"] = True
    m = re.search(r"(?:under|below|less than) \$?(\d+(?:\.\d+)?)", q)
    if m:
        price = float(m.group(1))
        plan["max_price"] = int(price) if price.is_integer() else price
    m = re.search(r"at least (\d+) hours", q)
    if m:
        plan["min_battery_h"] = int(m.group(1))
    return plan


def run_filter(plan: dict[str, Any]) -> list[dict[str, Any]]:
    def ok(p: dict[str, Any]) -> bool:
        a = p["attributes"]
        if "category" in plan and p["category"] != plan["category"]:
            return False
        for flag in ("wireless", "waterproof", "noise_cancelling"):
            if plan.get(flag) and not a[flag]:
                return False
        if "max_price" in plan and p["price"] > plan["max_price"]:
            return False
        if "min_battery_h" in plan and (a["battery_h"] is None or a["battery_h"] < plan["min_battery_h"]):
            return False
        return True

    return sorted([p for p in CATALOG if ok(p)], key=lambda p: (p["price"], p["sku"]))


# =============================================================================
# 4. Graph traversal
# =============================================================================
class Graph:
    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.out: dict[str, list[str]] = {}
        self.inc: dict[str, list[str]] = {}
        self.edge_count = 0

    def node(self, nid: str, name: str) -> None:
        self.names.setdefault(nid, name)

    def edge(self, src: str, rel: str, dst: str) -> None:
        lst = self.out.setdefault(f"{src}|{rel}", [])
        if dst not in lst:
            lst.append(dst)
            self.inc.setdefault(f"{dst}|{rel}", []).append(src)
            self.edge_count += 1

    def neighbors(self, nid: str, rel: str, direction: str) -> list[str]:
        return (self.out if direction == "out" else self.inc).get(f"{nid}|{rel}", [])

    def traverse(self, start: str, steps: list[tuple[str, str]]) -> list[list[str]]:
        """Follow a typed path pattern hop by hop, keeping every path (so results are explainable)."""
        paths = [[start]]
        for rel, direction in steps:
            paths = [p + [nb] for p in paths for nb in self.neighbors(p[-1], rel, direction)]
        return paths


def build_graph() -> Graph:
    g = Graph()
    for s in GRAPH_DATA["suppliers"]:
        g.node(f"supplier:{s['id']}", s["name"])
        g.node(f"country:{s['country']}", s["country"])
        g.edge(f"supplier:{s['id']}", "LOCATED_IN", f"country:{s['country']}")
    for p in CATALOG:
        g.node(f"product:{p['sku']}", p["sku"])
        g.edge(f"product:{p['sku']}", "SUPPLIED_BY", f"supplier:{p['supplier_id']}")
    for o in ORDERS:
        if o["status"] in ("delivered", "in_transit"):
            g.node(f"customer:{o['customer_id']}", o["customer_name"])
            g.node(f"product:{o['sku']}", o["sku"])
            g.edge(f"customer:{o['customer_id']}", "PURCHASED", f"product:{o['sku']}")
    for e in GRAPH_DATA["edges"]:
        g.edge(e["src"], e["rel"], e["dst"])
    return g


GRAPH = build_graph()


def graph_query(query: str) -> tuple[str, list[str]]:
    m = re.search(r"suppliers? (?:based|located) in ([A-Za-z]+)", query)
    country = m.group(1) if m else "Vietnam"
    pattern = [("LOCATED_IN", "in"), ("SUPPLIED_BY", "in"), ("PURCHASED", "in")]
    paths = GRAPH.traverse(f"country:{country}", pattern)
    via: dict[str, list[str]] = {}
    suppliers: list[str] = []
    for _country, supplier, product, customer in paths:
        if GRAPH.names[supplier] not in suppliers:
            suppliers.append(GRAPH.names[supplier])
        lst = via.setdefault(GRAPH.names[customer], [])
        if GRAPH.names[product] not in lst:
            lst.append(GRAPH.names[product])
    lines = [f"{name:<15} via {', '.join(via[name])}" for name in sorted(via)]
    pattern_text = "(country)<-LOCATED_IN-(supplier)<-SUPPLIED_BY-(product)<-PURCHASED-(customer)"
    return f"{pattern_text}  [{country}: suppliers {', '.join(suppliers)}; {len(paths)} paths]", lines


# =============================================================================
# 5. Agentic iterative retrieval (tool use) for cross-source multi-hop questions
# =============================================================================
AGENT_SYSTEM = (
    "You are Acme Audio's support analyst agent. Answer questions that need facts from several systems "
    "by calling tools step by step: orders database, product catalog, and policy search. Verify each fact "
    f"with a tool before using it. Today is {AS_OF}. Answer in two or three sentences and name the order."
)
AGENT_TOOLS: list[dict[str, Any]] = [
    {"name": "query_orders",
     "description": "Look up orders in the order database. Use it for a customer's orders, order dates or status. "
                    "Filter by customer_name and/or sku. Returns a JSON list of orders.",
     "input_schema": {"type": "object", "properties": {
         "customer_name": {"type": "string", "description": "Full customer name, e.g. 'Dana Lee'"},
         "sku": {"type": "string", "description": "Product SKU, e.g. 'HP-200'"}}}},
    {"name": "get_product",
     "description": "Get catalog details for ONE SKU: name, category, price, warranty_months and attributes. "
                    "Call it once per SKU you need (parallel calls are fine).",
     "input_schema": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}},
    {"name": "search_policies",
     "description": "Search customer policy text (returns, warranty, shipping, price match). Returns the best sections.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


def run_agent_tool(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Execute one tool call. Failures come back as (message, is_error=True) - never raise into the loop."""
    if name == "query_orders":
        rows = [o for o in ORDERS
                if (not args.get("customer_name") or o["customer_name"].lower() == args["customer_name"].lower())
                and (not args.get("sku") or o["sku"] == args["sku"])]
        return json.dumps([{k: o[k] for k in ("order_id", "order_date", "sku", "quantity", "status")} for o in rows]), False
    if name == "get_product":
        p = CATALOG_BY_SKU.get(args.get("sku", ""))
        if p is None:
            return f"SKU {args.get('sku')} not found in the catalog (discontinued or invalid).", True
        return json.dumps({k: p[k] for k in ("sku", "name", "category", "price", "warranty_months")}), False
    if name == "search_policies":
        hits = text_search(args.get("query", ""), "hybrid", "policies", k=1)
        return "\n".join(f"[{TEXT_BY_ID[d].title}] {TEXT_BY_ID[d].text}" for d, _ in hits), False
    return f"unknown tool {name}", True


def add_months(iso: str, months: int) -> str:
    y, m, d = (int(x) for x in iso.split("-"))
    idx = m - 1 + months
    y2, m2 = y + idx // 12, idx % 12 + 1
    dim = [31, 29 if (y2 % 4 == 0 and (y2 % 100 != 0 or y2 % 400 == 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return f"{y2:04d}-{m2:02d}-{min(d, dim[m2 - 1]):02d}"


def agentic_loop(client: anthropic.Anthropic, question: str, max_turns: int = 6) -> str:
    """Manual tool loop: check stop_reason every turn, send ALL tool results of a turn back in ONE user
    message, and report failures with is_error instead of crashing."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    for turn in range(1, max_turns + 1):
        resp = client.messages.create(model=MODEL_OPUS, max_tokens=2048, system=AGENT_SYSTEM,
                                      tools=AGENT_TOOLS, messages=messages)
        calls = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not calls:
            print(f"    turn {turn}: stop_reason={resp.stop_reason} -> final answer")
            return text_of(resp).strip()
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for c in calls:
            out, is_error = run_agent_tool(c.name, c.input)
            shown = out.replace("\n", " ")
            print(f"    turn {turn}: {c.name}({json.dumps(c.input, sort_keys=True)}) -> "
                  f"{'ERROR ' if is_error else ''}{shown[:70]}{'...' if len(shown) > 70 else ''}")
            result: dict[str, Any] = {"type": "tool_result", "tool_use_id": c.id, "content": out}
            if is_error:
                result["is_error"] = True
            results.append(result)
        messages.append({"role": "user", "content": results})
    return "(stopped: max turns reached)"


# =============================================================================
# 6. Router: regex pre-router -> Haiku structured output -> fallback rules
# =============================================================================
Pattern = Literal["exact_lookup", "aggregation", "filtered_search", "keyword", "conceptual", "mixed",
                  "multi_hop", "code_search"]
Source = Literal["orders", "catalog", "policies", "code", "graph", "multiple", "all"]


class RouteDecision(BaseModel):
    pattern: Pattern
    data_source: Source
    confidence: float
    reason: str


ROUTER_SYSTEM = """You are a query router for Acme Audio's assistant. Classify the user's query; do not answer it.

pattern:
- exact_lookup: fetch one record by its identifier (order ID, SKU)
- aggregation: totals, counts, averages, rankings over many records
- filtered_search: find items matching attribute constraints (price, battery, category, flags)
- keyword: a specific named term or phrase that must appear verbatim in documents
- conceptual: a natural-language question about meaning or policy, paraphrase-heavy
- mixed: conceptual question that also contains specific key terms
- multi_hop: needs facts chained across relations or across several systems
- code_search: where/how something is implemented in source code

data_source: orders (table), catalog (product JSON), policies (prose), code, graph (supplier/customer
relations), multiple (several systems), all (unsure - search everything).
Give a confidence between 0 and 1 and a short reason."""


def haiku_classify(query: str) -> dict[str, Any]:
    """What the offline mock 'Haiku' answers. Live, the real model classifies from ROUTER_SYSTEM."""
    q = query.lower()
    if re.search(r"\bin the code\b|\bfunction\b|\bcodebase\b|\bimplemented\b", q):
        return {"pattern": "code_search", "data_source": "code", "confidence": 0.95, "reason": "asks where logic lives in source code"}
    if re.search(r"\b(total|revenue|average|sum|how many)\b", q):
        return {"pattern": "aggregation", "data_source": "orders", "confidence": 0.94, "reason": "aggregate over many order rows"}
    if re.search(r"\bwhich customers\b.*\bsuppliers?\b", q):
        return {"pattern": "multi_hop", "data_source": "graph", "confidence": 0.91, "reason": "customer-product-supplier-country chain"}
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+'s\b", query) and re.search(r"warranty|return", q):
        return {"pattern": "multi_hop", "data_source": "multiple", "confidence": 0.88, "reason": "needs orders + catalog + policy"}
    if re.search(r"\$\d+|\bunder \d|\bat least\b", q):
        return {"pattern": "filtered_search", "data_source": "catalog", "confidence": 0.93, "reason": "numeric attribute constraints"}
    if re.search(r"^what is the [a-z ]+ (fee|rate|window|threshold)\??$", q):
        return {"pattern": "keyword", "data_source": "policies", "confidence": 0.86, "reason": "asks for a named policy term"}
    if "warranty" in q and re.search(r"stopped|broken|dead|not working", q):
        return {"pattern": "mixed", "data_source": "policies", "confidence": 0.83, "reason": "policy question with specific symptoms"}
    if re.search(r"^(can|may|do|does|how) i\b|\bcan i\b", q):
        return {"pattern": "conceptual", "data_source": "policies", "confidence": 0.9, "reason": "paraphrased policy question"}
    return {"pattern": "conceptual", "data_source": "catalog", "confidence": 0.41, "reason": "vague; products or policy unclear"}


FALLBACK_RULES: list[tuple[str, str, str]] = [
    (r"\b(total|sum|average|avg|count|how many|revenue)\b", "aggregation", "orders"),
    (r"\b(under|below|less than|at least|more than)\b.*\d", "filtered_search", "catalog"),
    (r"\b(code|function|implemented|module)\b", "code_search", "code"),
    (r"\bwhich (customers|suppliers|products)\b", "multi_hop", "graph"),
]


@dataclass
class Route:
    pattern: str
    source: str
    confidence: float
    via: str
    reason: str


class Router:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client
        self.llm_calls = 0
        self.llm_input_tokens = 0

    def route(self, query: str) -> Route:
        # 1) Cheap deterministic pre-router: unambiguous identifiers never need an LLM.
        if ORDER_ID_ONLY.search(query) and len(query.split()) <= 10:
            return Route("exact_lookup", "orders", 1.0, "rule:id-regex", "order ID present")
        # 2) LLM classifier with structured output (schema-validated by the SDK).
        try:
            msg = self.client.messages.parse(
                model=MODEL_HAIKU, max_tokens=1024, output_config={"effort": "low"},
                system=ROUTER_SYSTEM, messages=[{"role": "user", "content": query}],
                output_format=RouteDecision,
            )
            self.llm_calls += 1
            self.llm_input_tokens += msg.usage.input_tokens
            d = msg.parsed_output
            if d is None:
                raise ValueError("no parsed output")
        except (anthropic.APIError, pydantic.ValidationError, ValueError) as exc:
            return self.fallback(query, f"fallback(error: {type(exc).__name__})")
        # 3) Low confidence -> deterministic rules (or ask a clarifying question in a chat UI).
        if d.confidence < 0.6:
            return self.fallback(query, f"fallback(llm confidence {d.confidence:.2f})")
        return Route(d.pattern, d.data_source, d.confidence, "llm:haiku", d.reason)

    @staticmethod
    def fallback(query: str, via: str) -> Route:
        for rx, pattern, source in FALLBACK_RULES:
            if re.search(rx, query.lower()):
                return Route(pattern, source, 0.5, via, f"rule {rx}")
        return Route("mixed", "all", 0.5, via, "default: hybrid over all text")


ORDER_ID_ONLY = re.compile(r"\bORD-\d{5}\b")


def pick_strategy(r: Route) -> str:
    if r.pattern == "exact_lookup":
        return "key_lookup"
    if r.pattern == "aggregation":
        return "sql_aggregation"
    if r.pattern == "filtered_search":
        return "attribute_filter"
    if r.pattern == "keyword":
        return "bm25"
    if r.pattern == "conceptual":
        return "vector"
    if r.pattern == "mixed":
        return "hybrid"
    if r.pattern == "code_search":
        return "code_bm25"
    if r.pattern == "multi_hop":
        return "graph_traversal" if r.source == "graph" else "agentic_loop"
    return "hybrid"


# =============================================================================
# 7. Offline responder (router classifications + agent turns)
# =============================================================================
def _tool_history(req: MockRequest) -> list[tuple[str, dict[str, Any], str, bool]]:
    uses: dict[str, tuple[str, dict[str, Any]]] = {}
    out = []
    for m in req.messages:
        if not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                uses[b["id"]] = (b["name"], b["input"])
            elif b.get("type") == "tool_result" and b["tool_use_id"] in uses:
                name, inp = uses[b["tool_use_id"]]
                content = b.get("content")
                text = content if isinstance(content, str) else "".join(x.get("text", "") for x in content or [])
                out.append((name, inp, text, bool(b.get("is_error"))))
    return out


def responder(req: MockRequest) -> Any:
    if req.system_text.startswith("You are a query router"):
        return json.dumps(haiku_classify(req.last_user_text), separators=(",", ":"))
    if req.system_text.startswith("You are Acme Audio's support analyst"):
        question = req.messages[0]["content"]
        history = _tool_history(req)
        called = [h[0] for h in history]
        if "query_orders" not in called:
            m = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)'s", question)
            name = m.group(1) if m else ""
            return [text_block("I'll start with the customer's order history."),
                    tool_use_block("query_orders", {"customer_name": name})]
        orders = json.loads(next(h[2] for h in history if h[0] == "query_orders"))
        if "get_product" not in called:
            skus = list(dict.fromkeys(o["sku"] for o in orders))
            return [text_block("Checking the category and warranty of each product ordered.")] + \
                   [tool_use_block("get_product", {"sku": s}) for s in skus]
        if "search_policies" not in called:
            return [tool_use_block("search_policies", {"query": "warranty period start date proof of purchase"})]
        products = {json.loads(h[2])["sku"]: json.loads(h[2]) for h in history if h[0] == "get_product" and not h[3]}
        missing = [h[1]["sku"] for h in history if h[0] == "get_product" and h[3]]
        heads = [o for o in orders if products.get(o["sku"], {}).get("category") == "headphones"]
        latest = max(heads, key=lambda o: o["order_date"])
        p = products[latest["sku"]]
        end = add_months(latest["order_date"], p["warranty_months"])
        status = "is still under warranty" if AS_OF <= end else "is no longer under warranty"
        newer = [o for o in orders if o["order_date"] > latest["order_date"] and o["sku"] in products]
        note = f" Her newer order {newer[-1]['order_id']} is a {products[newer[-1]['sku']]['category'][:-1]}, not headphones." if newer else ""
        miss = f" ({', '.join(missing)} is no longer in the catalog, so it was ignored.)" if missing else ""
        who = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)'s", question)
        return (f"{who.group(1) if who else 'The customer'}'s most recent headphone order is {latest['order_id']} ({p['name']}, {p['sku']}) placed on "
                f"{latest['order_date']}. It has a {p['warranty_months']}-month warranty starting on the order date, "
                f"so coverage runs until {end} and it {status} as of {AS_OF}; a claim needs the order number as "
                f"proof of purchase.{note}{miss}")
    return "OK"


# =============================================================================
# Demo
# =============================================================================
STRATEGY_WHY = {
    "key_lookup": "an ID is a primary key: O(1) exact fetch, no ranking, no LLM",
    "sql_aggregation": "totals need ALL matching rows - top-k retrieval can never sum correctly",
    "attribute_filter": "numeric/boolean constraints are predicates, not similarity",
    "bm25": "a named term must match verbatim; lexical scoring is precise and explainable",
    "vector": "paraphrase with no shared keywords: semantic similarity",
    "hybrid": "meaning + specific terms: fuse lexical and semantic ranks (RRF)",
    "code_bm25": "identifiers split into words; chunks are whole functions",
    "graph_traversal": "explicit relations: follow typed edges hop by hop",
    "agentic_loop": "facts in different systems + reasoning: let Claude iterate with tools",
}


def print_matrix() -> None:
    rows = [
        ("Structured table", "key lookup", "SQL group-by", "SQL WHERE", "-", "-", "SQL joins / agent"),
        ("Semi-structured JSON", "key lookup", "SQL/JSON agg", "metadata filter", "BM25 on fields", "vector on desc", "graph / agent"),
        ("Unstructured prose", "-", "-", "metadata filter", "BM25", "vector / hybrid", "agentic multi-query"),
        ("Source code", "symbol index", "-", "path/lang filter", "code BM25", "code embeddings", "call-graph walk"),
        ("Knowledge graph", "node lookup", "graph agg", "property filter", "-", "-", "graph traversal"),
    ]
    head = ("data shape \\ pattern", "exact ID", "aggregation", "attr filter", "keyword", "conceptual", "multi-hop")
    widths = [22, 14, 14, 18, 16, 17, 20]
    print(("  " + "".join(h.ljust(w) for h, w in zip(head, widths))).rstrip())
    print("  " + "-" * (sum(widths) - 2))
    for r in rows:
        print(("  " + "".join(c.ljust(w) for c, w in zip(r, widths))).rstrip())


def run_query(client: anthropic.Anthropic, router: Router, qid: str, query: str) -> Route:
    print(f"\n[{qid}] \"{query}\"")
    r = router.route(query)
    strategy = pick_strategy(r)
    print(f"  route   : {r.pattern} / {r.source}  via {r.via} (conf {r.confidence:.2f}) -> {strategy}")
    print(f"  why     : {STRATEGY_WHY[strategy]}")
    if strategy == "key_lookup":
        for line in key_lookup(query):
            print(f"  result  : {line}")
    elif strategy == "sql_aggregation":
        plan = plan_aggregation(query)
        sql, params = plan_to_sql(plan)
        print(f"  plan    : {json.dumps(plan, sort_keys=True)}")
        print(f"  sql     : {sql}")
        print(f"  params  : {params}")
        for grp, cents, n in run_aggregation(plan):
            print(f"  result  : {grp:<14} ${cents / 100:>9.2f}  ({n} orders)")
    elif strategy == "attribute_filter":
        plan = plan_filter(query)
        print(f"  plan    : {json.dumps(plan, sort_keys=True)}")
        for p in run_filter(plan):
            print(f"  result  : {p['sku']} {p['name']:<30} ${p['price']:>7.2f}  battery={p['attributes']['battery_h']}h")
    elif strategy in ("bm25", "vector", "hybrid"):
        scope_note = r.source if r.source in ("policies", "catalog") else "all text"
        print(f"  scope   : {scope_note} (metadata pre-filter on doc kind)")
        for did, s in text_search(query, strategy, r.source):
            d = TEXT_BY_ID[did]
            print(f"  result  : {did:<26} {s:.3f}  \"{d.text[:62]}...\"")
    elif strategy == "code_bm25":
        for key, s in CODE_BM25.search(query, 2):
            c = next(c for c in CODE_CHUNKS if f"{c.file}:{c.start}" == key)
            print(f"  result  : {c.file}:{c.start}-{c.end}  {c.symbol}  ({s:.3f})")
        print(f"  tokens  : 'calculateShippingCost' -> {code_tokenize('calculateShippingCost')}")
    elif strategy == "graph_traversal":
        header, lines = graph_query(query)
        print(f"  pattern : {header}")
        for line in lines:
            print(f"  result  : {line}")
    elif strategy == "agentic_loop":
        print("  trace   :")
        print(f"  answer  : {agentic_loop(client, query)}")
    return r


def main() -> None:
    banner("D3.6 - Retrieval Strategies by Data Shape and Query Pattern")
    client = get_client(responder)

    section("1. Data shapes")
    print(f"  structured table     orders.csv     {len(ORDERS)} rows (order_id, date, customer, region, sku, qty, price, status)")
    print(f"  semi-structured JSON catalog.json   {len(CATALOG)} products with nested attributes")
    print(f"  unstructured prose   policies.md    {sum(1 for d in TEXT_DOCS if d.kind == 'policy')} sections (+ product descriptions as text docs)")
    files = len({c.file for c in CODE_CHUNKS})
    print(f"  source code          data/code      {files} files -> {len(CODE_CHUNKS)} definition-level chunks")
    print(f"  knowledge graph      graph.json     {len(GRAPH.names)} nodes, {GRAPH.edge_count} typed edges (joined from catalog + orders)")

    section("2. Decision matrix: data shape x query pattern -> strategy")
    print_matrix()

    section("3. Route and retrieve")
    print("Router = regex pre-router (IDs) -> Claude Haiku with structured output -> fallback rules.")
    router = Router(client)
    routes: list[tuple[dict[str, Any], Route]] = []
    for q in QUERIES["queries"]:
        routes.append((q, run_query(client, router, q["id"], q["query"])))

    section("4. Anti-pattern: one vector index for everything")
    print("Serialize every order row, product and policy into text and run pure vector search:")
    naive_docs: dict[str, str] = {
        o["order_id"]: (f"Order {o['order_id']} placed {o['order_date']} by {o['customer_name']} in {o['region']}: "
                        f"{o['quantity']} x {o['sku']} at USD {o['unit_price']}, status {o['status']}.") for o in ORDERS}
    naive_docs.update({d.doc_id: d.text for d in TEXT_DOCS})
    naive = VectorIndex(naive_docs, Embedder(list(naive_docs.values())))
    q1, q2, q3 = (QUERIES["queries"][i]["query"] for i in range(3))
    top1 = [d for d, _ in naive.search(q1, 3)]
    print(f"  {q1}\n    top-3: {', '.join(top1)} -> {'correct' if top1[0] == 'ORD-10447' else 'WRONG record'}: "
          "embeddings blur IDs and numbers")
    q3_orders = [o for o in ORDERS if "2026-07-01" <= o["order_date"] <= "2026-09-30" and o["status"] not in ("cancelled", "returned")]
    top2 = [d for d, _ in naive.search(q2, 5)]
    got = sum(1 for d in top2 if d in {o["order_id"] for o in q3_orders})
    print(f"  {q2}\n    top-5 holds {got} of the {len(q3_orders)} Q3 orders -> any total computed from it is wrong")
    plan = plan_filter(q3)
    top3 = [d for d, _ in naive.search(q3, 5) if d.startswith("product:")]
    good = {f"product:{p['sku']}" for p in run_filter(plan)}
    print(f"  {q3}\n    products in top-5: {', '.join(top3)} -> {sum(1 for d in top3 if d not in good)} violate the "
          "price/battery constraints")

    section("5. Router report")
    print(f"  {'id':<5}{'expected':<17}{'routed':<17}{'via':<32}ok")
    correct = 0
    for q, r in routes:
        ok = r.pattern in q["expected"]
        correct += ok
        print(f"  {q['id']:<5}{'/'.join(q['expected']):<17}{r.pattern:<17}{r.via:<32}{'yes' if ok else 'NO'}")
    print(f"  accuracy {correct}/{len(routes)} | Haiku router calls: {router.llm_calls} "
          f"(avg {router.llm_input_tokens // max(1, router.llm_calls)} input tokens) | "
          f"regex short-circuits: {sum(1 for _, r in routes if r.via.startswith('rule'))} | "
          f"fallbacks: {sum(1 for _, r in routes if r.via.startswith('fallback'))}")
    print("  Routing on Haiku costs a fraction of a cent per query and keeps the expensive model for the")
    print("  one question that truly needs agentic reasoning (Q9). Track router accuracy as its own eval.")

    section("Takeaways")
    print("  * Match the retriever to the data shape AND the query pattern; no single index serves all.")
    print("  * Aggregations and filters belong to databases (validated plans, bound parameters), not to RAG.")
    print("  * Regex for unambiguous IDs, a small model for fuzzy intent, deterministic fallback for doubt.")
    print("  * Graphs answer explicit relationship chains; agentic loops handle cross-system reasoning.")


if __name__ == "__main__":
    main()
