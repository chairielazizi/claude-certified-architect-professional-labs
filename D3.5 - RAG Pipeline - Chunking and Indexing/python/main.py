"""
D3.5 - RAG Pipeline: Chunking and Indexing

A complete, inspectable RAG pipeline over a small policy-handbook corpus:

  1. Load documents + metadata (doc_id, doc_type, version, updated_at).
  2. Compare four chunking strategies: fixed-size with overlap, sentence-aware,
     markdown-header recursive, and contextual chunking (Anthropic's Contextual
     Retrieval: Claude writes a short situating context per chunk; the full
     document sits in a cached prompt prefix so each extra chunk is cheap).
  3. Build a BM25 lexical index + a local vector index (deterministic stand-in
     embeddings - Anthropic has no embeddings endpoint; production systems use a
     partner model such as Voyage AI) and fuse them with reciprocal rank fusion.
  4. Evaluate retrieval (recall@3, MRR@10) per strategy on a labeled query set.
  5. Answer with Claude grounded on retrieved chunks via search_result blocks
     with citations enabled.
  6. Refresh the index incrementally by content hash, and show how a stale index
     or an embedding-model mismatch produces wrong answers.

Everything except the Claude calls is real, working code - that's where the
architecture lives. Claude calls go through the shared client (offline mock by
default, live with ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, replace
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
    get_client,
    section,
    text_block,
    text_of,
)

DATA = Path(__file__).resolve().parents[1] / "data"


# =============================================================================
# 1. Documents + metadata
# =============================================================================
@dataclass
class Doc:
    doc_id: str
    title: str
    doc_type: str
    version: str
    updated_at: str
    owner: str
    summary: str
    body: str  # markdown without front matter
    raw: str  # full text incl. front matter (what Claude sees when writing chunk context)


def parse_doc(raw: str) -> Doc:
    """Parse a markdown file with a simple `key: value` front-matter block."""
    raw = raw.replace("\r\n", "\n").strip()
    lines = raw.split("\n")
    if lines[0].strip() != "---":
        raise ValueError("document is missing front matter")
    end = lines.index("---", 1)
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    body = "\n".join(lines[end + 1 :]).strip()
    return Doc(
        doc_id=meta["doc_id"], title=meta["title"], doc_type=meta["doc_type"], version=meta["version"],
        updated_at=meta["updated_at"], owner=meta["owner"], summary=meta["summary"], body=body, raw=raw,
    )


def load_corpus() -> list[Doc]:
    return [parse_doc(p.read_text(encoding="utf-8")) for p in sorted((DATA / "corpus").glob("*.md"))]


# =============================================================================
# 2. Text processing shared by every index (identical in the TypeScript port,
#    so both languages produce the same rankings)
# =============================================================================
STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from had has have how i if in into is it its
    me my of on or our over per should so than that the their them then there these they this to too up us
    was we were what when where which while who why will with would you your""".split()
)


def stem(tok: str) -> str:
    """Tiny suffix stripper. Real systems use Snowball/Porter or the embedding model's own tokenizer."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


# The stand-in embedding model's "learned semantics": paraphrases map to one concept,
# which is what lets vector search match "new baby" to "parental leave" when BM25 can't.
CONCEPT_GROUPS: dict[str, list[str]] = {
    "pto": ["pto", "vacation", "timeoff"],
    "carryover": ["carry", "carryover", "rollover", "roll"],
    "parental": ["parental", "maternity", "paternity", "baby", "newborn", "adoption", "adoptive", "birth"],
    "reimburse": ["reimburse", "reimbursement", "reimbursed", "reimbursable", "repay", "repaid", "refund"],
    "expense": ["expense", "expensify", "cost", "claim"],
    "deadline": ["deadline", "within", "late", "due"],
    "meal": ["meal", "food", "dinner", "lunch", "breakfast", "diem", "snack"],
    "flight": ["flight", "fly", "flying", "airfare", "plane"],
    "device": ["laptop", "computer", "notebook", "device", "phone"],
    "theft": ["stolen", "lost", "lose", "theft", "missing"],
    "mfa": ["mfa", "multi", "factor", "2fa", "authentication"],
    "password": ["password", "passphrase", "credential"],
    "remote": ["remote", "home", "wfh", "telework", "hybrid"],
    "furniture": ["desk", "chair", "monitor", "ergonomic", "equipment", "furniture"],
    "money": ["money", "stipend", "usd", "dollar", "amount"],
    "approve": ["approve", "approval", "approver", "authorize", "sign", "signoff"],
    "purchase": ["purchase", "buy", "buying", "procurement", "order"],
    "vendor": ["vendor", "supplier", "seller"],
    "hour": ["hour", "online", "available", "availability", "reachable"],
    "abroad": ["abroad", "country", "international", "overseas"],
}
CONCEPTS: dict[str, str] = {tok: canon for canon, toks in CONCEPT_GROUPS.items() for tok in toks}


def concept_tokens(text: str) -> list[str]:
    return [CONCEPTS.get(t, t) for t in tokenize(text)]


def fnv1a(s: str) -> int:
    """32-bit FNV-1a: tiny, fast, and bit-identical across languages (unlike hash())."""
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


EMBED_DIM = 256
EMBED_SEEDS = {"hash-embed-v1": "v1-7f3a", "hash-embed-v2": "v2-c91e"}


class Embedder:
    """Deterministic local stand-in for an embedding model (offline, no dependencies).

    Signed feature hashing of concept-mapped tokens, weighted by sublinear tf x a
    frozen idf table, then L2-normalised. The (seed, dim, concept map, idf table)
    tuple IS the "model": change any of it and old vectors become incomparable,
    exactly like swapping embedding model versions in production.
    """

    def __init__(self, model: str, idf: dict[str, float], default_idf: float) -> None:
        self.model = model
        self.seed = EMBED_SEEDS[model]
        self.idf = idf
        self.default_idf = default_idf

    def embed(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for feat in concept_tokens(text):
            counts[feat] = counts.get(feat, 0) + 1
        vec = [0.0] * EMBED_DIM
        for feat, tf in counts.items():
            h = fnv1a(self.seed + ":" + feat)
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            vec[h % EMBED_DIM] += sign * (1.0 + math.log(tf)) * self.idf.get(feat, self.default_idf)
        sq = 0.0
        for x in vec:  # explicit loops (not sum()) keep float results identical to the TS port
            sq += x * x
        norm = math.sqrt(sq)
        return [x / norm for x in vec] if norm > 0 else vec


def fit_idf(docs: list[Doc]) -> tuple[dict[str, float], float]:
    """'Train' the stand-in model once on the initial corpus; the table is then frozen."""
    df: dict[str, int] = {}
    for d in docs:
        for feat in dict.fromkeys(concept_tokens(d.body)):
            df[feat] = df.get(feat, 0) + 1
    n = len(docs)
    return {f: math.log((n + 1) / (c + 1)) + 1.0 for f, c in df.items()}, math.log(n + 1) + 1.0


def dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


# =============================================================================
# 3. Chunking strategies
# =============================================================================
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str  # breadcrumb, e.g. "Travel and Expense Policy > Receipts"
    text: str
    version: str
    updated_at: str
    doc_type: str
    strategy: str
    context: str = ""  # contextual retrieval: Claude-written situating context

    @property
    def index_text(self) -> str:
        """What gets embedded / BM25-indexed: context is prepended, never stored instead of the chunk."""
        return f"{self.context}\n{self.text}" if self.context else self.text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def make_chunk(doc: Doc, chunk_id: str, section_path: str, text: str, strategy: str) -> Chunk:
    return Chunk(chunk_id, doc.doc_id, doc.title, section_path, text, doc.version, doc.updated_at, doc.doc_type, strategy)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(paragraph.strip()) if s.strip()]


def blocks_of(body: str) -> list[tuple[str, int, str]]:
    """Split markdown into ("h", level, text) heading blocks and ("p", 0, text) paragraph blocks."""
    out: list[tuple[str, int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(("p", 0, " ".join(buf)))
            buf.clear()

    for line in body.split("\n"):
        s = line.strip()
        m = HEADING.match(s)
        if m:
            flush()
            out.append(("h", len(m.group(1)), m.group(2).strip()))
        elif not s:
            flush()
        else:
            buf.append(s)
    flush()
    return out


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def chunk_fixed(doc: Doc, size: int = 60, overlap: int = 15) -> list[Chunk]:
    """Naive baseline: fixed word windows with overlap. Blind to sentences and headings."""
    words = doc.body.split()
    chunks: list[Chunk] = []
    i = 0
    while i < len(words):
        piece = " ".join(words[i : i + size])
        chunks.append(make_chunk(doc, f"{doc.doc_id}#fixed-{len(chunks):02d}", "(unknown)", piece, "fixed"))
        if i + size >= len(words):
            break
        i += size - overlap
    return chunks


def chunk_sentences(doc: Doc, max_words: int = 70) -> list[Chunk]:
    """Sentence-aware packing: never cuts a sentence, but ignores section boundaries."""
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_words = 0
    heading = doc.title
    cur_section = heading

    def emit() -> None:
        nonlocal cur, cur_words
        if cur:
            cid = f"{doc.doc_id}#sent-{len(chunks):02d}"
            chunks.append(make_chunk(doc, cid, f"{doc.title} > {cur_section}", " ".join(cur), "sentence"))
        cur, cur_words = [], 0

    for kind, _level, text in blocks_of(doc.body):
        if kind == "h":
            heading = text
        for sent in [text] if kind == "h" else split_sentences(text):
            n = len(sent.split())
            if cur and cur_words + n > max_words:
                emit()
            if not cur:
                cur_section = heading
            cur.append(sent)
            cur_words += n
    emit()
    return chunks


def split_recursive(paras: list[str], max_words: int) -> list[str]:
    """Recursive splitter: whole section -> paragraphs -> sentences, packing greedily to max_words."""
    total = 0
    for p in paras:
        total += len(p.split())
    if total <= max_words:
        return ["\n".join(paras)]
    units: list[str] = []
    for p in paras:
        units.extend([p] if len(p.split()) <= max_words else split_sentences(p))
    out: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for u in units:
        n = len(u.split())
        if cur and cur_n + n > max_words:
            out.append("\n".join(cur))
            cur, cur_n = [], 0
        cur.append(u)
        cur_n += n
    if cur:
        out.append("\n".join(cur))
    return out


def chunk_markdown(doc: Doc, max_words: int = 110) -> list[Chunk]:
    """Structure-aware: one chunk per heading section (split further only if too long).

    Chunk IDs derive from the section slug, so they stay STABLE across document
    versions - the property that makes incremental re-indexing by content hash work.
    """
    sections: list[tuple[list[str], list[str]]] = []
    path: list[tuple[int, str]] = []
    for kind, level, text in blocks_of(doc.body):
        if kind == "h":
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, text))
            sections.append(([t for _, t in path], []))
        else:
            if not sections:
                sections.append(([doc.title], []))
            sections[-1][1].append(text)
    chunks: list[Chunk] = []
    for sec_path, paras in sections:
        if not paras:
            continue
        heading = sec_path[-1]
        pieces = split_recursive(paras, max_words)
        for i, piece in enumerate(pieces):
            suffix = f"-{i + 1}" if len(pieces) > 1 else ""
            cid = f"{doc.doc_id}#{slugify(heading)}{suffix}"
            chunks.append(make_chunk(doc, cid, " > ".join(sec_path), f"## {heading}\n{piece}", "markdown"))
    return chunks


# =============================================================================
# 4. Indexes: BM25 + vector + hybrid (RRF), with metadata pre-filtering
# =============================================================================
class BM25Index:
    """Okapi BM25 with incremental add/remove (df and avg length kept up to date)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf: dict[str, dict[str, int]] = {}
        self.lengths: dict[str, int] = {}
        self.df: dict[str, int] = {}
        self.total_len = 0

    def add(self, cid: str, text: str) -> None:
        if cid in self.tf:
            self.remove(cid)
        toks = tokenize(text)
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        self.tf[cid] = tf
        self.lengths[cid] = len(toks)
        self.total_len += len(toks)
        for t in tf:
            self.df[t] = self.df.get(t, 0) + 1

    def remove(self, cid: str) -> None:
        tf = self.tf.pop(cid, None)
        if tf is None:
            return
        self.total_len -= self.lengths.pop(cid)
        for t in tf:
            self.df[t] -= 1
            if self.df[t] == 0:
                del self.df[t]

    def search(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(tokenize(query)))
        n = len(self.tf)
        if n == 0:
            return []
        avgdl = self.total_len / n
        scored: list[tuple[str, float]] = []
        for cid, tf in self.tf.items():
            if allowed is not None and cid not in allowed:
                continue
            dl = self.lengths[cid]
            s = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                df = self.df[t]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if s > 0:
                scored.append((cid, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


class EmbeddingModelMismatch(RuntimeError):
    pass


class VectorIndex:
    """Brute-force cosine index. Fine for thousands of chunks; use ANN (HNSW/IVF) beyond that."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.vectors: dict[str, list[float]] = {}

    def upsert(self, cid: str, text: str) -> None:
        self.vectors[cid] = self.embedder.embed(text)

    def remove(self, cid: str) -> None:
        self.vectors.pop(cid, None)

    def search(self, query: str, k: int, allowed: set[str] | None = None,
               query_embedder: Embedder | None = None, enforce_model: bool = True) -> list[tuple[str, float]]:
        enc = query_embedder or self.embedder
        if enforce_model and enc.model != self.embedder.model:
            raise EmbeddingModelMismatch(f"index built with {self.embedder.model}, query encoder is {enc.model}")
        qv = enc.embed(query)
        scored = [(cid, dot(qv, v)) for cid, v in self.vectors.items() if allowed is None or cid in allowed]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60, top: int = 10) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score = sum(1 / (k + rank)). Rank-based, so BM25 and cosine
    scores never need to be put on a common scale."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused[:top]


class RagIndex:
    """Chunk store + BM25 + vector index + a manifest (content hashes, embedding model)."""

    def __init__(self, embedder: Embedder) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.hashes: dict[str, str] = {}
        self.bm25 = BM25Index()
        self.vec = VectorIndex(embedder)
        self.embed_calls = 0

    def upsert(self, ch: Chunk) -> None:
        self.chunks[ch.chunk_id] = ch
        self.hashes[ch.chunk_id] = text_hash(ch.text)
        self.bm25.add(ch.chunk_id, ch.index_text)
        self.vec.upsert(ch.chunk_id, ch.index_text)
        self.embed_calls += 1

    def remove(self, cid: str) -> None:
        self.chunks.pop(cid, None)
        self.hashes.pop(cid, None)
        self.bm25.remove(cid)
        self.vec.remove(cid)

    def allowed(self, flt: dict[str, str] | None) -> set[str] | None:
        """Metadata PRE-filter: restrict candidates before ranking (post-filtering top-k can return nothing)."""
        if not flt:
            return None
        return {cid for cid, ch in self.chunks.items() if all(getattr(ch, key) == val for key, val in flt.items())}

    def search(self, query: str, mode: str = "hybrid", k: int = 10, flt: dict[str, str] | None = None,
               query_embedder: Embedder | None = None, enforce_model: bool = True) -> list[tuple[str, float]]:
        allowed = self.allowed(flt)
        if mode == "bm25":
            return self.bm25.search(query, k, allowed)
        if mode == "vector":
            return self.vec.search(query, k, allowed, query_embedder, enforce_model)
        depth = 20  # retrieve deeper from each retriever, then fuse
        return rrf([self.bm25.search(query, depth, allowed),
                    self.vec.search(query, depth, allowed, query_embedder, enforce_model)], top=k)


def build_index(chunks: list[Chunk], embedder: Embedder) -> RagIndex:
    idx = RagIndex(embedder)
    for ch in chunks:
        idx.upsert(ch)
    return idx


# =============================================================================
# 5. Evaluation
# =============================================================================
def norm_ws(s: str) -> str:
    return " ".join(s.lower().split())


def is_relevant(ch: Chunk, q: dict[str, Any], evidence_key: str = "evidence") -> bool:
    return ch.doc_id == q["doc_id"] and norm_ws(q.get(evidence_key, q["evidence"])) in norm_ws(ch.text)


def evaluate(idx: RagIndex, queries: list[dict[str, Any]], mode: str, k: int = 3,
             query_embedder: Embedder | None = None, evidence_key: str = "evidence") -> tuple[float, float, list[str]]:
    """recall@k = share of queries with an evidence chunk in the top k (single-answer queries);
    MRR@10 = mean reciprocal rank of the first evidence chunk in the top 10."""
    hits, rr, misses = 0, 0.0, []
    for q in queries:
        ranked = idx.search(q["query"], mode, 10, query_embedder=query_embedder, enforce_model=False)
        rank = 0
        for i, (cid, _) in enumerate(ranked, start=1):
            if is_relevant(idx.chunks[cid], q, evidence_key):
                rank = i
                break
        if 0 < rank <= k:
            hits += 1
        else:
            misses.append(q["id"])
        if rank:
            rr += 1.0 / rank
    return hits / len(queries), rr / len(queries), misses


# =============================================================================
# 6. Claude calls: contextual chunking + grounded answers (and their offline responder)
# =============================================================================
CONTEXT_SYSTEM = """You write short retrieval context for chunks of Northwind Systems internal policy documents.

Purpose: the context you write is prepended to the chunk before it is embedded and added to a BM25 keyword index. A chunk on its own often lacks the words a user will search for (for example "Submit within 60 days" never says "expense report" or "travel"), so your context must restore that missing information.

Guidelines:
- Name the document (title) and the section the chunk comes from.
- State the topic of the chunk in plain words, using the terms an employee would type into a search box.
- Mention the entities the chunk depends on but does not name (the policy area, the system or form involved).
- One or two sentences, at most 60 words. No preamble such as "Here is the context".
- Do not include version numbers, effective dates or owners: they change on every revision, they belong in metadata fields, and putting them into indexed text would force every chunk to be re-embedded on each revision.
- Never add facts that are not in the document.

Example
Document: Laptop Refresh Standard. Chunk: "Requests are approved automatically after 36 months."
Context: This chunk is from Northwind's Laptop Refresh Standard, section on eligibility; it explains when an employee's company laptop can be replaced with a new one."""
CONTEXT_PROMPT = (
    "Here is the chunk we want to situate within the whole document\n<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall document for the purposes "
    "of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
)
ANSWER_SYSTEM = (
    "You are Northwind's policy assistant. Answer ONLY from the search results provided in the user turn and "
    "cite them. If the search results do not contain the answer, say you could not find it - never guess. "
    "Keep answers to one or two sentences."
)


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    j = text.find(end, i + len(start))
    return text[i + len(start) : j] if i >= 0 and j >= 0 else ""


def _reply(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap mock content with a language-neutral output-token estimate (same number in the TS port)."""
    return {"content": blocks, "usage": {"output_tokens": max(1, len(json.dumps(blocks, separators=(",", ":"))) // 4)}}


def responder(req: MockRequest) -> Any:
    """Offline stand-in for Claude. It computes its reply FROM the request (like the real model would),
    so the demo reacts to whatever the pipeline actually retrieved."""
    if "<document>" in req.all_text and "<chunk>" in req.all_text:
        doc = parse_doc(_between(req.all_text, "<document>\n", "\n</document>"))
        chunk = _between(req.all_text, "<chunk>\n", "\n</chunk>")
        heading = next((ln[3:].strip() for ln in chunk.split("\n") if ln.startswith("## ")), "overview")
        summary = doc.summary[0].lower() + doc.summary[1:]
        return _reply([text_block(f"This chunk is from Northwind's {doc.title}, section \"{heading}\". The policy covers {summary}")])
    if req.system_text.startswith("You are Northwind's policy assistant"):
        content = req.messages[-1]["content"]
        results = [b for b in content if b.get("type") == "search_result"]
        question = " ".join(b["text"] for b in content if b.get("type") == "text")
        q_feats = set(concept_tokens(question))
        best: tuple[int, int, int] | None = None  # (overlap, result index, block index)
        for ri, r in enumerate(results):
            for bi, blk in enumerate(r["content"]):
                overlap = len(q_feats & set(concept_tokens(blk["text"])))
                if best is None or overlap > best[0]:
                    best = (overlap, ri, bi)
        if best is None or best[0] < 2:
            return _reply([text_block("I could not find this in the provided policy excerpts, so I won't guess. Please ask the People Team.")])
        _, ri, bi = best
        r = results[ri]
        sentence = r["content"][bi]["text"]
        citation = {"type": "search_result_location", "source": r["source"], "title": r["title"],
                    "cited_text": sentence, "search_result_index": ri, "start_block_index": bi, "end_block_index": bi + 1}
        return _reply([text_block(f"According to the {r['title']}: "), {"type": "text", "text": sentence, "citations": [citation]}])
    return "OK"


@dataclass
class UsageTotals:
    calls: int = 0
    input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0
    cost: float = 0.0

    def add(self, msg: Any) -> None:
        u = msg.usage
        self.calls += 1
        self.input += u.input_tokens
        self.cache_write += u.cache_creation_input_tokens or 0
        self.cache_read += u.cache_read_input_tokens or 0
        self.output += u.output_tokens
        self.cost += cost_usd(msg.model, u)


def contextualize(client: Any, doc: Doc, chunks: list[Chunk], totals: UsageTotals) -> bool:
    """Contextual Retrieval: one cheap Haiku call per chunk. The whole document is the first user
    block with cache_control, so chunk 2..n of the same document read it from cache.
    Returns True if this document's prefix was cached (it may be below the minimum cacheable size)."""
    cached = False
    for ch in chunks:
        msg = client.messages.create(
            model=MODEL_HAIKU,
            max_tokens=512,
            output_config={"effort": "low"},
            system=CONTEXT_SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"<document>\n{doc.raw}\n</document>", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": CONTEXT_PROMPT.format(chunk=ch.text)},
            ]}],
        )
        ch.context = text_of(msg).strip()
        totals.add(msg)
        cached = cached or bool(msg.usage.cache_creation_input_tokens or msg.usage.cache_read_input_tokens)
    return cached


def citable_units(text: str) -> list[str]:
    """Split a chunk into sentence-level text blocks so citations can point at exact sentences."""
    units: list[str] = []
    for line in text.split("\n"):
        if line.startswith("## "):
            units.append(line[3:].strip())
        elif line.strip():
            units.extend(split_sentences(line))
    return units


def answer(client: Any, idx: RagIndex, question: str, flt: dict[str, str] | None = None, k: int = 3) -> Any:
    hits = idx.search(question, "hybrid", k, flt=flt)
    content: list[dict[str, Any]] = []
    for cid, _ in hits:
        ch = idx.chunks[cid]
        content.append({
            "type": "search_result",
            "source": f"kb://{cid}@v{ch.version}",
            "title": f"{ch.title} > {ch.section.split(' > ')[-1]}",
            "content": [{"type": "text", "text": u} for u in citable_units(ch.text)],
            "citations": {"enabled": True},
        })
    content.append({"type": "text", "text": question})
    return client.messages.create(model=MODEL_OPUS, max_tokens=1024, system=ANSWER_SYSTEM,
                                  messages=[{"role": "user", "content": content}])


SOURCE_RE = re.compile(r"^kb://([^#]+)#.*@v(.+)$")


def print_answer(msg: Any, registry: dict[str, Doc]) -> None:
    """Print the answer + citations, then run a freshness guard: every cited version must match the
    source-of-truth registry, otherwise the answer came from a stale index."""
    print(f"  A: {text_of(msg).strip()}")
    n = 0
    for block in msg.content:
        for c in getattr(block, "citations", None) or []:
            if getattr(c, "type", "") != "search_result_location":
                continue
            n += 1
            print(f"     [{n}] {c.source}")
            m = SOURCE_RE.match(c.source)
            if m and m.group(1) in registry and registry[m.group(1)].version != m.group(2):
                print(f"     !! STALE: cites v{m.group(2)} but the source of truth is v{registry[m.group(1)].version}"
                      " -> answer is out of date; re-index required")
    if n == 0:
        print("     (no citations - treated as 'not found')")


# =============================================================================
# 7. Index refresh by content hash
# =============================================================================
def refresh_document(idx: RagIndex, new_doc: Doc, client: Any, totals: UsageTotals) -> dict[str, int]:
    """Incremental re-index of one document. Unchanged chunks (same stable id + same content hash)
    only get a metadata update - no Claude context call, no re-embedding."""
    stats = {"unchanged": 0, "changed": 0, "added": 0, "removed": 0}
    new_chunks = chunk_markdown(new_doc)
    new_ids = {c.chunk_id for c in new_chunks}
    to_embed: list[Chunk] = []
    for ch in new_chunks:
        if ch.chunk_id in idx.chunks and idx.hashes[ch.chunk_id] == text_hash(ch.text):
            stats["unchanged"] += 1
            old = idx.chunks[ch.chunk_id]
            idx.chunks[ch.chunk_id] = replace(old, version=new_doc.version, updated_at=new_doc.updated_at)
        else:
            stats["changed" if ch.chunk_id in idx.chunks else "added"] += 1
            to_embed.append(ch)
    contextualize(client, new_doc, to_embed, totals)
    for ch in to_embed:
        idx.upsert(ch)
    for cid in [c for c, ch in idx.chunks.items() if ch.doc_id == new_doc.doc_id and c not in new_ids]:
        idx.remove(cid)
        stats["removed"] += 1
    return stats


# =============================================================================
# Demo
# =============================================================================
def main() -> None:
    banner("D3.5 - RAG Pipeline: Chunking and Indexing")
    client = get_client(responder)
    docs = load_corpus()
    registry = {d.doc_id: d for d in docs}  # source of truth for current versions
    queries = json.loads((DATA / "queries.json").read_text(encoding="utf-8"))["queries"]
    idf, default_idf = fit_idf(docs)
    embedder = Embedder("hash-embed-v1", idf, default_idf)

    # -------------------------------------------------------------------------
    section("1. Corpus and metadata")
    print("Every chunk inherits document metadata so it can be filtered, cited and freshness-checked.")
    print(f"  {'doc_id':<26}{'type':<8}{'version':<9}{'updated_at':<12}{'words':>6}")
    for d in docs:
        print(f"  {d.doc_id:<26}{d.doc_type:<8}{d.version:<9}{d.updated_at:<12}{len(d.body.split()):>6}")

    # -------------------------------------------------------------------------
    section("2. Chunking strategies")
    strategies: dict[str, list[Chunk]] = {
        "fixed": [c for d in docs for c in chunk_fixed(d)],
        "sentence": [c for d in docs for c in chunk_sentences(d)],
        "markdown": [c for d in docs for c in chunk_markdown(d)],
    }
    descr = {
        "fixed": "60-word windows, 15-word overlap",
        "sentence": "pack whole sentences to <=70 words",
        "markdown": "one chunk per heading section, split >110 words",
        "contextual": "markdown chunks + Claude-written context",
    }
    print(f"  {'strategy':<11}{'chunks':>7}{'avg':>6}{'min':>5}{'max':>5}  evidence intact  rule")
    for name, chunks in strategies.items():
        lens = [len(c.text.split()) for c in chunks]
        total = 0
        for n in lens:
            total += n
        intact = sum(1 for q in queries if any(is_relevant(c, q) for c in chunks))
        print(f"  {name:<11}{len(chunks):>7}{total / len(lens):>6.0f}{min(lens):>5}{max(lens):>5}"
              f"  {intact:>2}/{len(queries)} queries    {descr[name]}")
    no_overlap = [c for d in docs for c in chunk_fixed(d, overlap=0)]
    intact0 = sum(1 for q in queries if any(is_relevant(c, q) for c in no_overlap))
    print("  'evidence intact' = some chunk contains the whole answer phrase; if none does, no retriever")
    print(f"  can find it. Fixed windows WITHOUT overlap keep it intact for only {intact0}/{len(queries)} queries -")
    print("  overlap is insurance against boundary cuts, paid for with duplicate tokens in the index.")
    sample = next(c for c in strategies["fixed"] if "within 60 days" in c.text)
    print(f"\n  Fixed-size chunk {sample.chunk_id} (starts/ends mid-sentence, no section):")
    print(f"    \"{sample.text[:80]} ... {sample.text[-50:]}\"")
    md = next(c for c in strategies["markdown"] if c.chunk_id.endswith("#submitting-expense-reports"))
    print(f"  Markdown chunk {md.chunk_id}")
    print(f"    section='{md.section}'")
    print(f"    \"{md.text[:110].replace(chr(10), ' | ')} ...\"")

    # -------------------------------------------------------------------------
    section("3. Contextual chunking (Contextual Retrieval) with prompt caching")
    print("For each markdown chunk, Claude Haiku writes 1-2 sentences situating it in its document.")
    print("The full document is a cache_control block, so only the first chunk per document pays full price.")
    ctx_chunks = [replace(c, strategy="contextual") for c in strategies["markdown"]]
    totals = UsageTotals()
    uncached_docs: list[str] = []
    for d in docs:
        if not contextualize(client, d, [c for c in ctx_chunks if c.doc_id == d.doc_id], totals):
            uncached_docs.append(d.doc_id)
    strategies["contextual"] = ctx_chunks
    p = PRICING[MODEL_HAIKU]
    no_cache = ((totals.input + totals.cache_write + totals.cache_read) * p["input"] + totals.output * p["output"]) / 1e6
    print(f"  chunks contextualized : {totals.calls} ({MODEL_HAIKU}, one call each)")
    print(f"  input tokens uncached : {totals.input}")
    print(f"  cache writes / reads  : {totals.cache_write} / {totals.cache_read}")
    print(f"  output tokens         : {totals.output}")
    print(f"  cost with caching     : ${totals.cost:.6f}   (without caching: ${no_cache:.6f})")
    if uncached_docs:
        print(f"  not cached            : {', '.join(uncached_docs)} - prefix below the minimum cacheable size")
        print("                          (no error, just no cache tokens - check usage, don't assume)")
    ex = next(c for c in ctx_chunks if c.chunk_id.endswith("#submitting-expense-reports"))
    print(f"  example context for {ex.chunk_id}:\n    \"{ex.context}\"")

    # -------------------------------------------------------------------------
    section("4. Indexes: BM25 + vector + hybrid (reciprocal rank fusion)")
    indexes = {name: build_index(chunks, embedder) for name, chunks in strategies.items()}
    prod = indexes["contextual"]
    print(f"  embedding stand-in : {embedder.model}, dim={EMBED_DIM} (deterministic local model; production would use")
    print("                       a partner embedding model such as Voyage AI - Anthropic has no embeddings API)")
    print(f"  BM25 vocabulary    : {len(prod.bm25.df)} terms over {len(prod.chunks)} contextual chunks")
    q = "How many weeks of paid leave do I get for a new baby?"
    print(f"\n  Query: \"{q}\"")
    for mode in ("bm25", "vector", "hybrid"):
        ranked = prod.search(q, mode, 3)
        shown = ", ".join(f"{cid.split('#')[1]}({s:.3f})" for cid, s in ranked)
        print(f"    {mode:<7}-> {shown}")
    print("  BM25 needs shared words; the vector stand-in maps 'baby' ~ 'parental'. RRF fuses ranks, not scores.")

    # -------------------------------------------------------------------------
    section("5. Retrieval evaluation: recall@3 and MRR@10")
    print(f"  {len(queries)} labeled queries; relevant = chunk from the right doc containing the evidence phrase.")
    print(f"  {'strategy':<11}| {'BM25 R@3':>8} {'MRR':>6} | {'Vector R@3':>10} {'MRR':>6} | {'Hybrid R@3':>10} {'MRR':>6}")
    hybrid_misses: dict[str, list[str]] = {}
    for name, idx in indexes.items():
        row = []
        for mode in ("bm25", "vector", "hybrid"):
            r3, mrr, misses = evaluate(idx, queries, mode)
            row.append((r3, mrr))
            if mode == "hybrid":
                hybrid_misses[name] = misses
        print(f"  {name:<11}| {row[0][0]:>8.2f} {row[0][1]:>6.3f} | {row[1][0]:>10.2f} {row[1][1]:>6.3f} |"
              f" {row[2][0]:>10.2f} {row[2][1]:>6.3f}")
    for name, misses in hybrid_misses.items():
        print(f"  hybrid misses ({name}): {', '.join(misses) if misses else 'none'}")
    print("  Read it like an architect: structure-aware chunks + context + hybrid fusion each remove a")
    print("  class of retrieval failure. Measure on YOUR labeled queries before choosing.")

    # -------------------------------------------------------------------------
    section("6. Metadata filtering: conflicting sources")
    q4 = "How much unused vacation can I carry over into next year?"
    print(f"  Query: \"{q4}\"")
    for label, flt in (("no filter", None), ("filter doc_type=policy", {"doc_type": "policy"})):
        ranked = prod.search(q4, "hybrid", 3, flt=flt)
        print(f"    {label:<23}-> " + ", ".join(
            f"{cid} (v{prod.chunks[cid].version}, {prod.chunks[cid].updated_at})" for cid, _ in ranked[:2]))
    print("  The 2025 FAQ phrases the question exactly like users do, so it wins on similarity - but it is")
    print("  stale (10 days vs 5 in policy v2.1). Authority/recency metadata, not similarity, must break the tie.")

    # -------------------------------------------------------------------------
    section("7. Grounded answers with citations (search_result blocks)")
    print("Retrieved chunks are passed as search_result content blocks with citations enabled, so")
    print("every claim maps back to a source + sentence. The system prompt forbids answering beyond them.")
    for question, flt in (
        ("How many days do I have to submit my expense report after a trip?", None),
        (q4, {"doc_type": "policy"}),
        ("What is Northwind's stock ticker symbol?", None),
    ):
        shown_filter = ", ".join(f"{k}={v}" for k, v in flt.items()) if flt else ""
        print(f"\n  Q: {question}" + (f"   [filter {shown_filter}]" if flt else ""))
        print_answer(answer(client, prod, question, flt), registry)

    # -------------------------------------------------------------------------
    section("8. Index refresh: stale index, then incremental re-index by content hash")
    new_doc = parse_doc((DATA / "updates" / "travel-expense-policy.md").read_text(encoding="utf-8"))
    registry[new_doc.doc_id] = new_doc  # the source system now says v4.0
    print(f"  Finance publishes {new_doc.doc_id} v{new_doc.version} ({new_doc.updated_at}): deadline 60 -> 30 days,")
    print("  per diem rates raised. The index has NOT been refreshed yet:")
    q1 = "How many days do I have to submit my expense report after a trip?"
    print(f"\n  Q: {q1}")
    print_answer(answer(client, prod, q1), registry)
    print("\n  Claude answered faithfully - from stale context. Grounding does not fix a stale index.")
    refresh_totals = UsageTotals()
    embeds_before = prod.embed_calls
    stats = refresh_document(prod, new_doc, client, refresh_totals)
    print(f"\n  Incremental refresh of {new_doc.doc_id}: " + " ".join(f"{k}={v}" for k, v in stats.items()))
    print(f"    re-embedded {prod.embed_calls - embeds_before} chunk(s), {refresh_totals.calls} context call(s)"
          f" (${refresh_totals.cost:.6f}); unchanged chunks only got a metadata (version) update")
    print(f"\n  Q: {q1}")
    print_answer(answer(client, prod, q1), registry)

    # -------------------------------------------------------------------------
    section("9. Embedding-model mismatch")
    print("  Someone upgrades the query encoder to hash-embed-v2 but the index still holds v1 vectors.")
    v2 = Embedder("hash-embed-v2", idf, default_idf)
    for label, enc in (("query encoder = v1 (matches index)", embedder), ("query encoder = v2 (MISMATCH)", v2)):
        rv, mv, _ = evaluate(prod, queries, "vector", query_embedder=enc, evidence_key="evidence_v4")
        rh, mh, _ = evaluate(prod, queries, "hybrid", query_embedder=enc, evidence_key="evidence_v4")
        print(f"    {label:<36} vector R@3={rv:.2f} MRR={mv:.3f} | hybrid R@3={rh:.2f} MRR={mh:.3f}")
    print("  Vectors from different models live in different spaces: similarity becomes noise, with no error.")
    print("  Hybrid hides part of the damage because BM25 still works - which makes it harder to notice.")
    try:
        prod.search(q1, "hybrid", 3, query_embedder=v2)
    except EmbeddingModelMismatch as exc:
        print(f"  Guard: the index manifest pins the model -> EmbeddingModelMismatch: {exc}")
        print("  Fix: re-embed everything with the new model into a NEW index, evaluate, then swap (blue/green).")

    # -------------------------------------------------------------------------
    section("Takeaways")
    print("  * Chunk on structure (headings, sentences); keep chunk IDs stable and metadata attached.")
    print("  * Contextual Retrieval prepends Claude-written context to each chunk; prompt caching of the")
    print("    full document makes it cheap. Keep volatile values (versions, dates) out of embedded text.")
    print("  * Hybrid (BM25 + vectors, fused with RRF) covers exact terms AND paraphrases.")
    print("  * Evaluate retrieval separately from generation (recall@k, MRR) on labeled queries.")
    print("  * Freshness is an indexing problem: content-hash incremental refresh, version-aware citations,")
    print("    and a pinned embedding model per index.")


if __name__ == "__main__":
    main()
