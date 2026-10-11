/**
 * D3.5 - RAG Pipeline: Chunking and Indexing (TypeScript port, functionally identical to python/main.py)
 *
 * 1. Load documents + metadata (doc_id, doc_type, version, updated_at).
 * 2. Compare chunking: fixed-size + overlap, sentence-aware, markdown-header recursive, and
 *    contextual chunking (Anthropic's Contextual Retrieval with the full doc in a cached prefix).
 * 3. BM25 + local vector index (deterministic stand-in embeddings - Anthropic has no embeddings
 *    endpoint; production uses a partner model such as Voyage AI) fused with reciprocal rank fusion.
 * 4. Evaluate recall@3 / MRR@10 per strategy on a labeled query set.
 * 5. Grounded answers via search_result blocks with citations.
 * 6. Incremental refresh by content hash; stale-index and embedding-mismatch failure modes.
 *
 * Tokenizer, hashing and scoring formulas match the Python version exactly, so rankings match.
 */

import type Anthropic from "@anthropic-ai/sdk";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  MODEL_HAIKU,
  MODEL_OPUS,
  PRICING,
  banner,
  costUsd,
  getClient,
  section,
  textBlock,
  textOf,
} from "../../shared/typescript/claudeClient.ts";
import type { Json, MockRequest, ResponderResult } from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");

// =============================================================================
// 1. Documents + metadata
// =============================================================================
interface Doc {
  doc_id: string;
  title: string;
  doc_type: string;
  version: string;
  updated_at: string;
  owner: string;
  summary: string;
  body: string; // markdown without front matter
  raw: string; // full text incl. front matter (what Claude sees when writing chunk context)
}

function parseDoc(rawIn: string): Doc {
  const raw = rawIn.replace(/\r\n/g, "\n").trim();
  const lines = raw.split("\n");
  if (lines[0]!.trim() !== "---") throw new Error("document is missing front matter");
  const end = lines.indexOf("---", 1);
  const meta: Record<string, string> = {};
  for (const line of lines.slice(1, end)) {
    const i = line.indexOf(":");
    const key = i >= 0 ? line.slice(0, i) : line;
    meta[key.trim()] = i >= 0 ? line.slice(i + 1).trim() : "";
  }
  return {
    doc_id: meta.doc_id!, title: meta.title!, doc_type: meta.doc_type!, version: meta.version!,
    updated_at: meta.updated_at!, owner: meta.owner!, summary: meta.summary!,
    body: lines.slice(end + 1).join("\n").trim(), raw,
  };
}

function loadCorpus(): Doc[] {
  const dir = path.join(DATA, "corpus");
  return readdirSync(dir)
    .filter((f) => f.endsWith(".md"))
    .sort()
    .map((f) => parseDoc(readFileSync(path.join(dir, f), "utf8")));
}

// =============================================================================
// 2. Text processing shared by every index (identical to the Python version)
// =============================================================================
const STOPWORDS = new Set(
  `a an and are as at be been but by can could do does for from had has have how i if in into is it its
  me my of on or our over per should so than that the their them then there these they this to too up us
  was we were what when where which while who why will with would you your`.split(/\s+/).filter(Boolean),
);

/** Tiny suffix stripper. Real systems use Snowball/Porter or the embedding model's own tokenizer. */
function stem(tok: string): string {
  if (tok.length > 4 && tok.endsWith("ies")) return tok.slice(0, -3) + "y";
  if (tok.length > 3 && tok.endsWith("s") && !tok.endsWith("ss")) return tok.slice(0, -1);
  return tok;
}

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => !STOPWORDS.has(t)).map(stem);
}

const words = (s: string): string[] => s.split(/\s+/).filter((w) => w.length > 0);

// The stand-in embedding model's "learned semantics": paraphrases map to one concept.
const CONCEPT_GROUPS: Record<string, string[]> = {
  pto: ["pto", "vacation", "timeoff"],
  carryover: ["carry", "carryover", "rollover", "roll"],
  parental: ["parental", "maternity", "paternity", "baby", "newborn", "adoption", "adoptive", "birth"],
  reimburse: ["reimburse", "reimbursement", "reimbursed", "reimbursable", "repay", "repaid", "refund"],
  expense: ["expense", "expensify", "cost", "claim"],
  deadline: ["deadline", "within", "late", "due"],
  meal: ["meal", "food", "dinner", "lunch", "breakfast", "diem", "snack"],
  flight: ["flight", "fly", "flying", "airfare", "plane"],
  device: ["laptop", "computer", "notebook", "device", "phone"],
  theft: ["stolen", "lost", "lose", "theft", "missing"],
  mfa: ["mfa", "multi", "factor", "2fa", "authentication"],
  password: ["password", "passphrase", "credential"],
  remote: ["remote", "home", "wfh", "telework", "hybrid"],
  furniture: ["desk", "chair", "monitor", "ergonomic", "equipment", "furniture"],
  money: ["money", "stipend", "usd", "dollar", "amount"],
  approve: ["approve", "approval", "approver", "authorize", "sign", "signoff"],
  purchase: ["purchase", "buy", "buying", "procurement", "order"],
  vendor: ["vendor", "supplier", "seller"],
  hour: ["hour", "online", "available", "availability", "reachable"],
  abroad: ["abroad", "country", "international", "overseas"],
};
const CONCEPTS = new Map<string, string>();
for (const [canon, toks] of Object.entries(CONCEPT_GROUPS)) for (const t of toks) CONCEPTS.set(t, canon);

function conceptTokens(text: string): string[] {
  return tokenize(text).map((t) => CONCEPTS.get(t) ?? t);
}

/** 32-bit FNV-1a: tiny, fast, and bit-identical across languages. */
function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (const b of Buffer.from(s, "utf8")) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

const EMBED_DIM = 256;
const EMBED_SEEDS: Record<string, string> = { "hash-embed-v1": "v1-7f3a", "hash-embed-v2": "v2-c91e" };

/**
 * Deterministic local stand-in for an embedding model: signed feature hashing of concept-mapped
 * tokens, sublinear tf x frozen idf, L2-normalised. (seed, dim, concept map, idf) IS the "model".
 */
class Embedder {
  readonly model: string;
  readonly seed: string;
  readonly idf: Map<string, number>;
  readonly defaultIdf: number;

  constructor(model: string, idf: Map<string, number>, defaultIdf: number) {
    this.model = model;
    this.seed = EMBED_SEEDS[model]!;
    this.idf = idf;
    this.defaultIdf = defaultIdf;
  }

  embed(text: string): number[] {
    const counts = new Map<string, number>(); // Map keeps insertion order (objects reorder numeric keys)
    for (const feat of conceptTokens(text)) counts.set(feat, (counts.get(feat) ?? 0) + 1);
    const vec = new Array<number>(EMBED_DIM).fill(0);
    for (const [feat, tf] of counts) {
      const h = fnv1a(this.seed + ":" + feat);
      const sign = ((h >>> 16) & 1) === 0 ? 1.0 : -1.0;
      vec[h % EMBED_DIM] += sign * (1.0 + Math.log(tf)) * (this.idf.get(feat) ?? this.defaultIdf);
    }
    let sq = 0.0;
    for (const x of vec) sq += x * x;
    const norm = Math.sqrt(sq);
    return norm > 0 ? vec.map((x) => x / norm) : vec;
  }
}

/** 'Train' the stand-in model once on the initial corpus; the table is then frozen. */
function fitIdf(docs: Doc[]): [Map<string, number>, number] {
  const df = new Map<string, number>();
  for (const d of docs) for (const feat of new Set(conceptTokens(d.body))) df.set(feat, (df.get(feat) ?? 0) + 1);
  const n = docs.length;
  const idf = new Map<string, number>();
  for (const [f, c] of df) idf.set(f, Math.log((n + 1) / (c + 1)) + 1.0);
  return [idf, Math.log(n + 1) + 1.0];
}

function dot(a: number[], b: number[]): number {
  let s = 0.0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

// =============================================================================
// 3. Chunking strategies
// =============================================================================
interface Chunk {
  chunk_id: string;
  doc_id: string;
  title: string;
  section: string; // breadcrumb, e.g. "Travel and Expense Policy > Receipts"
  text: string;
  version: string;
  updated_at: string;
  doc_type: string;
  strategy: string;
  context: string; // contextual retrieval: Claude-written situating context
}

/** What gets embedded / BM25-indexed: context is prepended, never stored instead of the chunk. */
const indexText = (ch: Chunk): string => (ch.context ? `${ch.context}\n${ch.text}` : ch.text);
const textHash = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex").slice(0, 12);

function makeChunk(doc: Doc, chunkId: string, sectionPath: string, text: string, strategy: string): Chunk {
  return {
    chunk_id: chunkId, doc_id: doc.doc_id, title: doc.title, section: sectionPath, text,
    version: doc.version, updated_at: doc.updated_at, doc_type: doc.doc_type, strategy, context: "",
  };
}

const SENT_SPLIT = /(?<=[.!?])\s+/;
const HEADING = /^(#{1,6})\s+(.*)$/;

function splitSentences(paragraph: string): string[] {
  return paragraph.trim().split(SENT_SPLIT).map((s) => s.trim()).filter((s) => s.length > 0);
}

type Block = ["h" | "p", number, string];

/** Split markdown into ["h", level, text] heading blocks and ["p", 0, text] paragraph blocks. */
function blocksOf(body: string): Block[] {
  const out: Block[] = [];
  let buf: string[] = [];
  const flush = () => {
    if (buf.length) out.push(["p", 0, buf.join(" ")]);
    buf = [];
  };
  for (const line of body.split("\n")) {
    const s = line.trim();
    const m = HEADING.exec(s);
    if (m) {
      flush();
      out.push(["h", m[1]!.length, m[2]!.trim()]);
    } else if (!s) flush();
    else buf.push(s);
  }
  flush();
  return out;
}

const slugify = (s: string): string => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");

/** Naive baseline: fixed word windows with overlap. Blind to sentences and headings. */
function chunkFixed(doc: Doc, size = 60, overlap = 15): Chunk[] {
  const ws = words(doc.body);
  const chunks: Chunk[] = [];
  let i = 0;
  while (i < ws.length) {
    const piece = ws.slice(i, i + size).join(" ");
    chunks.push(makeChunk(doc, `${doc.doc_id}#fixed-${String(chunks.length).padStart(2, "0")}`, "(unknown)", piece, "fixed"));
    if (i + size >= ws.length) break;
    i += size - overlap;
  }
  return chunks;
}

/** Sentence-aware packing: never cuts a sentence, but ignores section boundaries. */
function chunkSentences(doc: Doc, maxWords = 70): Chunk[] {
  const chunks: Chunk[] = [];
  let cur: string[] = [];
  let curWords = 0;
  let heading = doc.title;
  let curSection = heading;
  const emit = () => {
    if (cur.length) {
      const cid = `${doc.doc_id}#sent-${String(chunks.length).padStart(2, "0")}`;
      chunks.push(makeChunk(doc, cid, `${doc.title} > ${curSection}`, cur.join(" "), "sentence"));
    }
    cur = [];
    curWords = 0;
  };
  for (const [kind, , text] of blocksOf(doc.body)) {
    if (kind === "h") heading = text;
    for (const sent of kind === "h" ? [text] : splitSentences(text)) {
      const n = words(sent).length;
      if (cur.length && curWords + n > maxWords) emit();
      if (!cur.length) curSection = heading;
      cur.push(sent);
      curWords += n;
    }
  }
  emit();
  return chunks;
}

/** Recursive splitter: whole section -> paragraphs -> sentences, packing greedily to maxWords. */
function splitRecursive(paras: string[], maxWords: number): string[] {
  let total = 0;
  for (const p of paras) total += words(p).length;
  if (total <= maxWords) return [paras.join("\n")];
  const units: string[] = [];
  for (const p of paras) units.push(...(words(p).length <= maxWords ? [p] : splitSentences(p)));
  const out: string[] = [];
  let cur: string[] = [];
  let curN = 0;
  for (const u of units) {
    const n = words(u).length;
    if (cur.length && curN + n > maxWords) {
      out.push(cur.join("\n"));
      cur = [];
      curN = 0;
    }
    cur.push(u);
    curN += n;
  }
  if (cur.length) out.push(cur.join("\n"));
  return out;
}

/**
 * Structure-aware: one chunk per heading section (split further only if too long). Chunk IDs derive
 * from the section slug, so they stay STABLE across versions - what makes hash-based refresh work.
 */
function chunkMarkdown(doc: Doc, maxWords = 110): Chunk[] {
  const sections: [string[], string[]][] = [];
  const pathStack: [number, string][] = [];
  for (const [kind, level, text] of blocksOf(doc.body)) {
    if (kind === "h") {
      while (pathStack.length && pathStack[pathStack.length - 1]![0] >= level) pathStack.pop();
      pathStack.push([level, text]);
      sections.push([pathStack.map(([, t]) => t), []]);
    } else {
      if (!sections.length) sections.push([[doc.title], []]);
      sections[sections.length - 1]![1].push(text);
    }
  }
  const chunks: Chunk[] = [];
  for (const [secPath, paras] of sections) {
    if (!paras.length) continue;
    const heading = secPath[secPath.length - 1]!;
    const pieces = splitRecursive(paras, maxWords);
    pieces.forEach((piece, i) => {
      const suffix = pieces.length > 1 ? `-${i + 1}` : "";
      chunks.push(makeChunk(doc, `${doc.doc_id}#${slugify(heading)}${suffix}`, secPath.join(" > "), `## ${heading}\n${piece}`, "markdown"));
    });
  }
  return chunks;
}

// =============================================================================
// 4. Indexes: BM25 + vector + hybrid (RRF), with metadata pre-filtering
// =============================================================================
type Ranked = [string, number][];
const byScore = (a: [string, number], b: [string, number]): number =>
  b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);

/** Okapi BM25 with incremental add/remove (df and avg length kept up to date). */
class BM25Index {
  readonly k1: number;
  readonly b: number;
  readonly tf = new Map<string, Map<string, number>>();
  readonly lengths = new Map<string, number>();
  readonly df = new Map<string, number>();
  totalLen = 0;

  constructor(k1 = 1.2, b = 0.75) {
    this.k1 = k1;
    this.b = b;
  }

  add(cid: string, text: string): void {
    if (this.tf.has(cid)) this.remove(cid);
    const toks = tokenize(text);
    const tf = new Map<string, number>();
    for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
    this.tf.set(cid, tf);
    this.lengths.set(cid, toks.length);
    this.totalLen += toks.length;
    for (const t of tf.keys()) this.df.set(t, (this.df.get(t) ?? 0) + 1);
  }

  remove(cid: string): void {
    const tf = this.tf.get(cid);
    if (!tf) return;
    this.tf.delete(cid);
    this.totalLen -= this.lengths.get(cid)!;
    this.lengths.delete(cid);
    for (const t of tf.keys()) {
      const left = this.df.get(t)! - 1;
      if (left === 0) this.df.delete(t);
      else this.df.set(t, left);
    }
  }

  search(query: string, k: number, allowed: Set<string> | null = null): Ranked {
    const terms = [...new Set(tokenize(query))];
    const n = this.tf.size;
    if (n === 0) return [];
    const avgdl = this.totalLen / n;
    const scored: Ranked = [];
    for (const [cid, tf] of this.tf) {
      if (allowed !== null && !allowed.has(cid)) continue;
      const dl = this.lengths.get(cid)!;
      let s = 0.0;
      for (const t of terms) {
        const f = tf.get(t) ?? 0;
        if (f === 0) continue;
        const df = this.df.get(t)!;
        const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
        s += (idf * f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * dl) / avgdl));
      }
      if (s > 0) scored.push([cid, s]);
    }
    return scored.sort(byScore).slice(0, k);
  }
}

class EmbeddingModelMismatch extends Error {}

/** Brute-force cosine index. Fine for thousands of chunks; use ANN (HNSW/IVF) beyond that. */
class VectorIndex {
  readonly embedder: Embedder;
  readonly vectors = new Map<string, number[]>();

  constructor(embedder: Embedder) {
    this.embedder = embedder;
  }

  upsert(cid: string, text: string): void {
    this.vectors.set(cid, this.embedder.embed(text));
  }

  remove(cid: string): void {
    this.vectors.delete(cid);
  }

  search(query: string, k: number, allowed: Set<string> | null = null, queryEmbedder: Embedder | null = null, enforceModel = true): Ranked {
    const enc = queryEmbedder ?? this.embedder;
    if (enforceModel && enc.model !== this.embedder.model) {
      throw new EmbeddingModelMismatch(`index built with ${this.embedder.model}, query encoder is ${enc.model}`);
    }
    const qv = enc.embed(query);
    const scored: Ranked = [];
    for (const [cid, v] of this.vectors) if (allowed === null || allowed.has(cid)) scored.push([cid, dot(qv, v)]);
    return scored.sort(byScore).slice(0, k);
  }
}

/** Reciprocal rank fusion: score = sum(1 / (k + rank)). Rank-based, so scores never need a common scale. */
function rrf(rankings: Ranked[], k = 60, top = 10): Ranked {
  const scores = new Map<string, number>();
  for (const ranking of rankings) {
    ranking.forEach(([cid], i) => scores.set(cid, (scores.get(cid) ?? 0.0) + 1.0 / (k + i + 1)));
  }
  return [...scores.entries()].sort(byScore).slice(0, top);
}

type Filter = Partial<Record<keyof Chunk, string>>;

/** Chunk store + BM25 + vector index + a manifest (content hashes, embedding model). */
class RagIndex {
  readonly chunks = new Map<string, Chunk>();
  readonly hashes = new Map<string, string>();
  readonly bm25 = new BM25Index();
  readonly vec: VectorIndex;
  embedCalls = 0;

  constructor(embedder: Embedder) {
    this.vec = new VectorIndex(embedder);
  }

  upsert(ch: Chunk): void {
    this.chunks.set(ch.chunk_id, ch);
    this.hashes.set(ch.chunk_id, textHash(ch.text));
    this.bm25.add(ch.chunk_id, indexText(ch));
    this.vec.upsert(ch.chunk_id, indexText(ch));
    this.embedCalls += 1;
  }

  remove(cid: string): void {
    this.chunks.delete(cid);
    this.hashes.delete(cid);
    this.bm25.remove(cid);
    this.vec.remove(cid);
  }

  /** Metadata PRE-filter: restrict candidates before ranking (post-filtering top-k can return nothing). */
  allowed(flt: Filter | null): Set<string> | null {
    if (!flt || Object.keys(flt).length === 0) return null;
    const out = new Set<string>();
    for (const [cid, ch] of this.chunks) {
      if (Object.entries(flt).every(([key, val]) => ch[key as keyof Chunk] === val)) out.add(cid);
    }
    return out;
  }

  search(query: string, mode = "hybrid", k = 10, flt: Filter | null = null, queryEmbedder: Embedder | null = null, enforceModel = true): Ranked {
    const allowed = this.allowed(flt);
    if (mode === "bm25") return this.bm25.search(query, k, allowed);
    if (mode === "vector") return this.vec.search(query, k, allowed, queryEmbedder, enforceModel);
    const depth = 20; // retrieve deeper from each retriever, then fuse
    return rrf([this.bm25.search(query, depth, allowed), this.vec.search(query, depth, allowed, queryEmbedder, enforceModel)], 60, k);
  }
}

function buildIndex(chunks: Chunk[], embedder: Embedder): RagIndex {
  const idx = new RagIndex(embedder);
  for (const ch of chunks) idx.upsert(ch);
  return idx;
}

// =============================================================================
// 5. Evaluation
// =============================================================================
interface Query {
  id: string;
  query: string;
  doc_id: string;
  evidence: string;
  evidence_v4?: string;
}
type EvidenceKey = "evidence" | "evidence_v4";

const normWs = (s: string): string => words(s.toLowerCase()).join(" ");

function isRelevant(ch: Chunk, q: Query, evidenceKey: EvidenceKey = "evidence"): boolean {
  return ch.doc_id === q.doc_id && normWs(ch.text).includes(normWs(q[evidenceKey] ?? q.evidence));
}

/** recall@k = share of queries with an evidence chunk in the top k; MRR@10 = mean 1/rank of the first one. */
function evaluate(idx: RagIndex, queries: Query[], mode: string, k = 3, queryEmbedder: Embedder | null = null,
  evidenceKey: EvidenceKey = "evidence"): [number, number, string[]] {
  let hits = 0;
  let rr = 0.0;
  const misses: string[] = [];
  for (const q of queries) {
    const ranked = idx.search(q.query, mode, 10, null, queryEmbedder, false);
    let rank = 0;
    for (let i = 0; i < ranked.length; i++) {
      if (isRelevant(idx.chunks.get(ranked[i]![0])!, q, evidenceKey)) {
        rank = i + 1;
        break;
      }
    }
    if (rank > 0 && rank <= k) hits += 1;
    else misses.push(q.id);
    if (rank) rr += 1.0 / rank;
  }
  return [hits / queries.length, rr / queries.length, misses];
}

// =============================================================================
// 6. Claude calls: contextual chunking + grounded answers (and their offline responder)
// =============================================================================
const CONTEXT_SYSTEM = `You write short retrieval context for chunks of Northwind Systems internal policy documents.

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
Context: This chunk is from Northwind's Laptop Refresh Standard, section on eligibility; it explains when an employee's company laptop can be replaced with a new one.`;

const contextPrompt = (chunk: string): string =>
  `Here is the chunk we want to situate within the whole document\n<chunk>\n${chunk}\n</chunk>\n` +
  "Please give a short succinct context to situate this chunk within the overall document for the purposes " +
  "of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.";

const ANSWER_SYSTEM =
  "You are Northwind's policy assistant. Answer ONLY from the search results provided in the user turn and " +
  "cite them. If the search results do not contain the answer, say you could not find it - never guess. " +
  "Keep answers to one or two sentences.";

function between(text: string, start: string, end: string): string {
  const i = text.indexOf(start);
  const j = text.indexOf(end, i + start.length);
  return i >= 0 && j >= 0 ? text.slice(i + start.length, j) : "";
}

/** Wrap mock content with a language-neutral output-token estimate (same number in the Python port). */
const reply = (blocks: Json[]): ResponderResult => ({ content: blocks, usage: { output_tokens: Math.max(1, Math.floor(JSON.stringify(blocks).length / 4)) } });

/** Offline stand-in for Claude: it computes its reply FROM the request, so the demo reacts to retrieval. */
function responder(req: MockRequest): ResponderResult {
  const all = req.allText;
  if (all.includes("<document>") && all.includes("<chunk>")) {
    const doc = parseDoc(between(all, "<document>\n", "\n</document>"));
    const chunk = between(all, "<chunk>\n", "\n</chunk>");
    const headingLine = chunk.split("\n").find((ln) => ln.startsWith("## "));
    const heading = headingLine ? headingLine.slice(3).trim() : "overview";
    const summary = doc.summary[0]!.toLowerCase() + doc.summary.slice(1);
    return reply([textBlock(`This chunk is from Northwind's ${doc.title}, section "${heading}". The policy covers ${summary}`)]);
  }
  if (req.systemText.startsWith("You are Northwind's policy assistant")) {
    const content = req.messages[req.messages.length - 1]!.content as Json[];
    const results = content.filter((b) => b.type === "search_result");
    const question = content.filter((b) => b.type === "text").map((b) => String(b.text)).join(" ");
    const qFeats = new Set(conceptTokens(question));
    let best: [number, number, number] | null = null; // [overlap, result index, block index]
    results.forEach((r, ri) => {
      (r.content as Json[]).forEach((blk, bi) => {
        const overlap = [...new Set(conceptTokens(String(blk.text)))].filter((f) => qFeats.has(f)).length;
        if (best === null || overlap > best[0]) best = [overlap, ri, bi];
      });
    });
    const found = best as [number, number, number] | null;
    if (found === null || found[0] < 2) {
      return reply([textBlock("I could not find this in the provided policy excerpts, so I won't guess. Please ask the People Team.")]);
    }
    const [, ri, bi] = found;
    const r = results[ri]!;
    const sentence = String((r.content as Json[])[bi]!.text);
    const citation = {
      type: "search_result_location", source: r.source, title: r.title, cited_text: sentence,
      search_result_index: ri, start_block_index: bi, end_block_index: bi + 1,
    };
    return reply([textBlock(`According to the ${r.title}: `), { type: "text", text: sentence, citations: [citation] }]);
  }
  return "OK";
}

class UsageTotals {
  calls = 0;
  input = 0;
  cacheWrite = 0;
  cacheRead = 0;
  output = 0;
  cost = 0.0;

  add(msg: Anthropic.Message): void {
    const u = msg.usage;
    this.calls += 1;
    this.input += u.input_tokens;
    this.cacheWrite += u.cache_creation_input_tokens ?? 0;
    this.cacheRead += u.cache_read_input_tokens ?? 0;
    this.output += u.output_tokens;
    this.cost += costUsd(msg.model, u);
  }
}

/**
 * Contextual Retrieval: one cheap Haiku call per chunk. The whole document is the first user block
 * with cache_control, so chunks 2..n of the same document read it from cache.
 * Returns true if this document's prefix was cached (it may be below the minimum cacheable size).
 */
async function contextualize(client: Anthropic, doc: Doc, chunks: Chunk[], totals: UsageTotals): Promise<boolean> {
  let cached = false;
  for (const ch of chunks) {
    const msg = await client.messages.create({
      model: MODEL_HAIKU,
      max_tokens: 512,
      output_config: { effort: "low" },
      system: CONTEXT_SYSTEM,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: `<document>\n${doc.raw}\n</document>`, cache_control: { type: "ephemeral" } },
          { type: "text", text: contextPrompt(ch.text) },
        ],
      }],
    });
    ch.context = textOf(msg).trim();
    totals.add(msg);
    cached = cached || Boolean(msg.usage.cache_creation_input_tokens || msg.usage.cache_read_input_tokens);
  }
  return cached;
}

/** Split a chunk into sentence-level text blocks so citations can point at exact sentences. */
function citableUnits(text: string): string[] {
  const units: string[] = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("## ")) units.push(line.slice(3).trim());
    else if (line.trim()) units.push(...splitSentences(line));
  }
  return units;
}

async function answer(client: Anthropic, idx: RagIndex, question: string, flt: Filter | null = null, k = 3): Promise<Anthropic.Message> {
  const hits = idx.search(question, "hybrid", k, flt);
  const content: Array<Anthropic.SearchResultBlockParam | Anthropic.TextBlockParam> = [];
  for (const [cid] of hits) {
    const ch = idx.chunks.get(cid)!;
    const parts = ch.section.split(" > ");
    content.push({
      type: "search_result",
      source: `kb://${cid}@v${ch.version}`,
      title: `${ch.title} > ${parts[parts.length - 1]}`,
      content: citableUnits(ch.text).map((u) => ({ type: "text" as const, text: u })),
      citations: { enabled: true },
    });
  }
  content.push({ type: "text", text: question });
  return client.messages.create({ model: MODEL_OPUS, max_tokens: 1024, system: ANSWER_SYSTEM, messages: [{ role: "user", content }] });
}

const SOURCE_RE = /^kb:\/\/([^#]+)#.*@v(.+)$/;

/** Print answer + citations, then a freshness guard: cited versions must match the source of truth. */
function printAnswer(msg: Anthropic.Message, registry: Map<string, Doc>): void {
  console.log(`  A: ${textOf(msg).trim()}`);
  let n = 0;
  for (const block of msg.content) {
    if (block.type !== "text") continue;
    for (const c of block.citations ?? []) {
      if (c.type !== "search_result_location") continue;
      n += 1;
      console.log(`     [${n}] ${c.source}`);
      const m = SOURCE_RE.exec(c.source);
      const current = m ? registry.get(m[1]!) : undefined;
      if (m && current && current.version !== m[2]) {
        console.log(`     !! STALE: cites v${m[2]} but the source of truth is v${current.version} -> answer is out of date; re-index required`);
      }
    }
  }
  if (n === 0) console.log("     (no citations - treated as 'not found')");
}

// =============================================================================
// 7. Index refresh by content hash
// =============================================================================
/** Incremental re-index of one document. Unchanged chunks only get a metadata update. */
async function refreshDocument(idx: RagIndex, newDoc: Doc, client: Anthropic, totals: UsageTotals): Promise<Record<string, number>> {
  const stats: Record<string, number> = { unchanged: 0, changed: 0, added: 0, removed: 0 };
  const newChunks = chunkMarkdown(newDoc);
  const newIds = new Set(newChunks.map((c) => c.chunk_id));
  const toEmbed: Chunk[] = [];
  for (const ch of newChunks) {
    if (idx.chunks.has(ch.chunk_id) && idx.hashes.get(ch.chunk_id) === textHash(ch.text)) {
      stats.unchanged! += 1;
      const old = idx.chunks.get(ch.chunk_id)!;
      idx.chunks.set(ch.chunk_id, { ...old, version: newDoc.version, updated_at: newDoc.updated_at });
    } else {
      stats[idx.chunks.has(ch.chunk_id) ? "changed" : "added"]! += 1;
      toEmbed.push(ch);
    }
  }
  await contextualize(client, newDoc, toEmbed, totals);
  for (const ch of toEmbed) idx.upsert(ch);
  const stale = [...idx.chunks.entries()].filter(([c, ch]) => ch.doc_id === newDoc.doc_id && !newIds.has(c)).map(([c]) => c);
  for (const cid of stale) {
    idx.remove(cid);
    stats.removed! += 1;
  }
  return stats;
}

// =============================================================================
// Demo
// =============================================================================
banner("D3.5 - RAG Pipeline: Chunking and Indexing");
const client = getClient(responder);
const docs = loadCorpus();
const registry = new Map(docs.map((d) => [d.doc_id, d] as [string, Doc])); // source of truth for current versions
const queries = (JSON.parse(readFileSync(path.join(DATA, "queries.json"), "utf8")) as { queries: Query[] }).queries;
const [idf, defaultIdf] = fitIdf(docs);
const embedder = new Embedder("hash-embed-v1", idf, defaultIdf);

// -----------------------------------------------------------------------------
section("1. Corpus and metadata");
console.log("Every chunk inherits document metadata so it can be filtered, cited and freshness-checked.");
console.log(`  ${"doc_id".padEnd(26)}${"type".padEnd(8)}${"version".padEnd(9)}${"updated_at".padEnd(12)}${"words".padStart(6)}`);
for (const d of docs) {
  console.log(`  ${d.doc_id.padEnd(26)}${d.doc_type.padEnd(8)}${d.version.padEnd(9)}${d.updated_at.padEnd(12)}${String(words(d.body).length).padStart(6)}`);
}

// -----------------------------------------------------------------------------
section("2. Chunking strategies");
const strategies = new Map<string, Chunk[]>([
  ["fixed", docs.flatMap((d) => chunkFixed(d))],
  ["sentence", docs.flatMap((d) => chunkSentences(d))],
  ["markdown", docs.flatMap((d) => chunkMarkdown(d))],
]);
const descr: Record<string, string> = {
  fixed: "60-word windows, 15-word overlap",
  sentence: "pack whole sentences to <=70 words",
  markdown: "one chunk per heading section, split >110 words",
  contextual: "markdown chunks + Claude-written context",
};
console.log(`  ${"strategy".padEnd(11)}${"chunks".padStart(7)}${"avg".padStart(6)}${"min".padStart(5)}${"max".padStart(5)}  evidence intact  rule`);
for (const [name, chunks] of strategies) {
  const lens = chunks.map((c) => words(c.text).length);
  let total = 0;
  for (const n of lens) total += n;
  const intact = queries.filter((q) => chunks.some((c) => isRelevant(c, q))).length;
  console.log(
    `  ${name.padEnd(11)}${String(chunks.length).padStart(7)}${(total / lens.length).toFixed(0).padStart(6)}` +
      `${String(Math.min(...lens)).padStart(5)}${String(Math.max(...lens)).padStart(5)}` +
      `  ${String(intact).padStart(2)}/${queries.length} queries    ${descr[name]}`,
  );
}
const noOverlap = docs.flatMap((d) => chunkFixed(d, 60, 0));
const intact0 = queries.filter((q) => noOverlap.some((c) => isRelevant(c, q))).length;
console.log("  'evidence intact' = some chunk contains the whole answer phrase; if none does, no retriever");
console.log(`  can find it. Fixed windows WITHOUT overlap keep it intact for only ${intact0}/${queries.length} queries -`);
console.log("  overlap is insurance against boundary cuts, paid for with duplicate tokens in the index.");
const sample = strategies.get("fixed")!.find((c) => c.text.includes("within 60 days"))!;
console.log(`\n  Fixed-size chunk ${sample.chunk_id} (starts/ends mid-sentence, no section):`);
console.log(`    "${sample.text.slice(0, 80)} ... ${sample.text.slice(-50)}"`);
const md = strategies.get("markdown")!.find((c) => c.chunk_id.endsWith("#submitting-expense-reports"))!;
console.log(`  Markdown chunk ${md.chunk_id}`);
console.log(`    section='${md.section}'`);
console.log(`    "${md.text.slice(0, 110).replace(/\n/g, " | ")} ..."`);

// -----------------------------------------------------------------------------
section("3. Contextual chunking (Contextual Retrieval) with prompt caching");
console.log("For each markdown chunk, Claude Haiku writes 1-2 sentences situating it in its document.");
console.log("The full document is a cache_control block, so only the first chunk per document pays full price.");
const ctxChunks: Chunk[] = strategies.get("markdown")!.map((c) => ({ ...c, strategy: "contextual" }));
const totals = new UsageTotals();
const uncachedDocs: string[] = [];
for (const d of docs) {
  if (!(await contextualize(client, d, ctxChunks.filter((c) => c.doc_id === d.doc_id), totals))) uncachedDocs.push(d.doc_id);
}
strategies.set("contextual", ctxChunks);
const p = PRICING[MODEL_HAIKU]!;
const noCache = ((totals.input + totals.cacheWrite + totals.cacheRead) * p.input + totals.output * p.output) / 1e6;
console.log(`  chunks contextualized : ${totals.calls} (${MODEL_HAIKU}, one call each)`);
console.log(`  input tokens uncached : ${totals.input}`);
console.log(`  cache writes / reads  : ${totals.cacheWrite} / ${totals.cacheRead}`);
console.log(`  output tokens         : ${totals.output}`);
console.log(`  cost with caching     : $${totals.cost.toFixed(6)}   (without caching: $${noCache.toFixed(6)})`);
if (uncachedDocs.length) {
  console.log(`  not cached            : ${uncachedDocs.join(", ")} - prefix below the minimum cacheable size`);
  console.log("                          (no error, just no cache tokens - check usage, don't assume)");
}
const ex = ctxChunks.find((c) => c.chunk_id.endsWith("#submitting-expense-reports"))!;
console.log(`  example context for ${ex.chunk_id}:\n    "${ex.context}"`);

// -----------------------------------------------------------------------------
section("4. Indexes: BM25 + vector + hybrid (reciprocal rank fusion)");
const indexes = new Map<string, RagIndex>([...strategies].map(([name, chunks]) => [name, buildIndex(chunks, embedder)] as [string, RagIndex]));
const prod = indexes.get("contextual")!;
console.log(`  embedding stand-in : ${embedder.model}, dim=${EMBED_DIM} (deterministic local model; production would use`);
console.log("                       a partner embedding model such as Voyage AI - Anthropic has no embeddings API)");
console.log(`  BM25 vocabulary    : ${prod.bm25.df.size} terms over ${prod.chunks.size} contextual chunks`);
const babyQ = "How many weeks of paid leave do I get for a new baby?";
console.log(`\n  Query: "${babyQ}"`);
for (const mode of ["bm25", "vector", "hybrid"]) {
  const shown = prod.search(babyQ, mode, 3).map(([cid, s]) => `${cid.split("#")[1]}(${s.toFixed(3)})`).join(", ");
  console.log(`    ${mode.padEnd(7)}-> ${shown}`);
}
console.log("  BM25 needs shared words; the vector stand-in maps 'baby' ~ 'parental'. RRF fuses ranks, not scores.");

// -----------------------------------------------------------------------------
section("5. Retrieval evaluation: recall@3 and MRR@10");
console.log(`  ${queries.length} labeled queries; relevant = chunk from the right doc containing the evidence phrase.`);
console.log(`  ${"strategy".padEnd(11)}| ${"BM25 R@3".padStart(8)} ${"MRR".padStart(6)} | ${"Vector R@3".padStart(10)} ${"MRR".padStart(6)} | ${"Hybrid R@3".padStart(10)} ${"MRR".padStart(6)}`);
const hybridMisses = new Map<string, string[]>();
for (const [name, idx] of indexes) {
  const row: [number, number][] = [];
  for (const mode of ["bm25", "vector", "hybrid"]) {
    const [r3, mrr, misses] = evaluate(idx, queries, mode);
    row.push([r3, mrr]);
    if (mode === "hybrid") hybridMisses.set(name, misses);
  }
  console.log(
    `  ${name.padEnd(11)}| ${row[0]![0].toFixed(2).padStart(8)} ${row[0]![1].toFixed(3).padStart(6)} |` +
      ` ${row[1]![0].toFixed(2).padStart(10)} ${row[1]![1].toFixed(3).padStart(6)} |` +
      ` ${row[2]![0].toFixed(2).padStart(10)} ${row[2]![1].toFixed(3).padStart(6)}`,
  );
}
for (const [name, misses] of hybridMisses) console.log(`  hybrid misses (${name}): ${misses.length ? misses.join(", ") : "none"}`);
console.log("  Read it like an architect: structure-aware chunks + context + hybrid fusion each remove a");
console.log("  class of retrieval failure. Measure on YOUR labeled queries before choosing.");

// -----------------------------------------------------------------------------
section("6. Metadata filtering: conflicting sources");
const q4 = "How much unused vacation can I carry over into next year?";
console.log(`  Query: "${q4}"`);
for (const [label, flt] of [["no filter", null], ["filter doc_type=policy", { doc_type: "policy" }]] as [string, Filter | null][]) {
  const ranked = prod.search(q4, "hybrid", 3, flt);
  const shown = ranked.slice(0, 2).map(([cid]) => {
    const ch = prod.chunks.get(cid)!;
    return `${cid} (v${ch.version}, ${ch.updated_at})`;
  });
  console.log(`    ${label.padEnd(23)}-> ${shown.join(", ")}`);
}
console.log("  The 2025 FAQ phrases the question exactly like users do, so it wins on similarity - but it is");
console.log("  stale (10 days vs 5 in policy v2.1). Authority/recency metadata, not similarity, must break the tie.");

// -----------------------------------------------------------------------------
section("7. Grounded answers with citations (search_result blocks)");
console.log("Retrieved chunks are passed as search_result content blocks with citations enabled, so");
console.log("every claim maps back to a source + sentence. The system prompt forbids answering beyond them.");
const askList: [string, Filter | null][] = [
  ["How many days do I have to submit my expense report after a trip?", null],
  [q4, { doc_type: "policy" }],
  ["What is Northwind's stock ticker symbol?", null],
];
for (const [question, flt] of askList) {
  const shownFilter = flt ? Object.entries(flt).map(([k, v]) => `${k}=${v}`).join(", ") : "";
  console.log(`\n  Q: ${question}` + (flt ? `   [filter ${shownFilter}]` : ""));
  printAnswer(await answer(client, prod, question, flt), registry);
}

// -----------------------------------------------------------------------------
section("8. Index refresh: stale index, then incremental re-index by content hash");
const newDoc = parseDoc(readFileSync(path.join(DATA, "updates", "travel-expense-policy.md"), "utf8"));
registry.set(newDoc.doc_id, newDoc); // the source system now says v4.0
console.log(`  Finance publishes ${newDoc.doc_id} v${newDoc.version} (${newDoc.updated_at}): deadline 60 -> 30 days,`);
console.log("  per diem rates raised. The index has NOT been refreshed yet:");
const q1 = "How many days do I have to submit my expense report after a trip?";
console.log(`\n  Q: ${q1}`);
printAnswer(await answer(client, prod, q1), registry);
console.log("\n  Claude answered faithfully - from stale context. Grounding does not fix a stale index.");
const refreshTotals = new UsageTotals();
const embedsBefore = prod.embedCalls;
const stats = await refreshDocument(prod, newDoc, client, refreshTotals);
console.log(`\n  Incremental refresh of ${newDoc.doc_id}: ` + Object.entries(stats).map(([k, v]) => `${k}=${v}`).join(" "));
console.log(
  `    re-embedded ${prod.embedCalls - embedsBefore} chunk(s), ${refreshTotals.calls} context call(s)` +
    ` ($${refreshTotals.cost.toFixed(6)}); unchanged chunks only got a metadata (version) update`,
);
console.log(`\n  Q: ${q1}`);
printAnswer(await answer(client, prod, q1), registry);

// -----------------------------------------------------------------------------
section("9. Embedding-model mismatch");
console.log("  Someone upgrades the query encoder to hash-embed-v2 but the index still holds v1 vectors.");
const v2 = new Embedder("hash-embed-v2", idf, defaultIdf);
for (const [label, enc] of [["query encoder = v1 (matches index)", embedder], ["query encoder = v2 (MISMATCH)", v2]] as [string, Embedder][]) {
  const [rv, mv] = evaluate(prod, queries, "vector", 3, enc, "evidence_v4");
  const [rh, mh] = evaluate(prod, queries, "hybrid", 3, enc, "evidence_v4");
  console.log(`    ${label.padEnd(36)} vector R@3=${rv.toFixed(2)} MRR=${mv.toFixed(3)} | hybrid R@3=${rh.toFixed(2)} MRR=${mh.toFixed(3)}`);
}
console.log("  Vectors from different models live in different spaces: similarity becomes noise, with no error.");
console.log("  Hybrid hides part of the damage because BM25 still works - which makes it harder to notice.");
try {
  prod.search(q1, "hybrid", 3, null, v2);
} catch (exc) {
  if (!(exc instanceof EmbeddingModelMismatch)) throw exc;
  console.log(`  Guard: the index manifest pins the model -> EmbeddingModelMismatch: ${exc.message}`);
  console.log("  Fix: re-embed everything with the new model into a NEW index, evaluate, then swap (blue/green).");
}

// -----------------------------------------------------------------------------
section("Takeaways");
console.log("  * Chunk on structure (headings, sentences); keep chunk IDs stable and metadata attached.");
console.log("  * Contextual Retrieval prepends Claude-written context to each chunk; prompt caching of the");
console.log("    full document makes it cheap. Keep volatile values (versions, dates) out of embedded text.");
console.log("  * Hybrid (BM25 + vectors, fused with RRF) covers exact terms AND paraphrases.");
console.log("  * Evaluate retrieval separately from generation (recall@k, MRR) on labeled queries.");
console.log("  * Freshness is an indexing problem: content-hash incremental refresh, version-aware citations,");
console.log("    and a pinned embedding model per index.");
