# D3.5 - RAG Pipeline - Chunking and Indexing

> **Exam objective:** Design a RAG pipeline with appropriate chunking and indexing strategies

## Why it matters

Most "the model hallucinated" incidents in enterprise RAG are really **retrieval** failures: the
right fact was never in a chunk, was split across two chunks, was outranked by a stale duplicate, or
lived in an index built with a different embedding model. Claude answers faithfully from whatever
context you hand it. The architect's job is the pipeline in front of the model: how documents are
split, enriched, indexed, filtered, refreshed, and measured. Those decisions set the ceiling on
answer quality, and no prompt can raise it.

## Key concepts

### The pipeline, stage by stage

| Stage | Decision | Typical choice on the Claude platform |
|---|---|---|
| Ingest / parse | Keep structure (headings, tables, code blocks) | Markdown/HTML-aware parsers; keep doc metadata |
| Chunk | Unit of retrieval | Structure-aware, about 200-800 tokens, 10-20% overlap for fixed windows |
| Enrich | What else goes into the index | Metadata fields (doc_id, section, version, updated_at, ACL) + **Contextual Retrieval** context |
| Embed + index | Lexical, dense, or both | BM25 + a partner embedding model (e.g. Voyage AI) in a vector store; **Anthropic has no embeddings API** |
| Retrieve | How candidates are found | Hybrid (BM25 + vectors) fused with **RRF**, metadata **pre-filters**, optional reranker |
| Generate | How the answer is grounded | Retrieved chunks as `search_result` (or `document`) blocks with `citations: {enabled: true}` |
| Evaluate | Retrieval and generation scored separately | recall@k / MRR / nDCG for retrieval; faithfulness and correctness for answers |
| Refresh | Keeping the index true to the sources | Event-driven incremental re-index by content hash; blue/green rebuild on model change |

### RAG or just long context?

Anthropic's Contextual Retrieval guidance: if the whole knowledge base fits in the prompt (it
suggests under ~200K tokens, roughly 500 pages), skip retrieval and send everything with **prompt
caching**. RAG pays off when the corpus is large, changes often, needs per-user access control, or
cost/latency per question must stay flat as the corpus grows.

| Signal | Long context + caching | RAG |
|---|---|---|
| Corpus size | Fits in the window with headroom | Larger than the window, or growing |
| Update rate | Rare (cache is rebuilt on change) | Frequent (incremental index updates) |
| Access control | Same content for every user | Per-user/role filtering (security trimming) |
| Cost per question | Cache reads; scales with corpus size | Scales with top-k, not corpus size |
| Failure mode | Lost-in-the-middle on huge prompts | Retrieval misses, staleness |

### Chunking strategies

| Strategy | How | Strengths | Weaknesses | Use for |
|---|---|---|---|---|
| Fixed-size + overlap | N tokens/words, sliding overlap | Simple, predictable size | Cuts sentences and facts; no section metadata | Unstructured text, baselines |
| Sentence / paragraph | Pack whole sentences up to a limit | Never splits a sentence | Ignores section boundaries; topics bleed together | Prose without headings, transcripts |
| Structure-aware (recursive) | Split on headings, then paragraphs, then sentences | Coherent topics; section breadcrumbs; stable IDs | Uneven sizes; needs clean structure | Policies, docs, wikis, contracts |
| Code-aware | Split on functions/classes (AST) | Whole units of meaning | Language-specific parsers | Source code (see D3.6) |
| Semantic | Break where adjacent-sentence embedding similarity drops | Topic-coherent without markup | Costly to compute, harder to debug | Long unstructured documents |
| Parent-child (small-to-big) | Retrieve small chunks, return the parent section | Precise matching + full context for generation | Two-level index | Long sections, dense manuals |
| **Contextual chunking** | An LLM prepends chunk-specific context before indexing | Restores missing terms such as company, product, or period | One LLM call per chunk (use caching) | Chunks that are ambiguous on their own |

Chunk size trade-off: **small** chunks give precise embeddings but lose surrounding context;
**large** chunks keep context but dilute the embedding and cost more tokens per retrieved result.
Tune size, overlap and top-k together, against an eval set.

### Contextual Retrieval (Anthropic)

For each chunk, a small model (Haiku class) receives the **full document** plus the chunk and writes
50-100 tokens of situating context, for example "This chunk is from the Travel and Expense Policy,
section on submitting expense reports...". The context is **prepended to the chunk before both
embedding and BM25 indexing** (contextual embeddings + contextual BM25). Because the document is the
same for every chunk, you mark it with `cache_control`, so each extra chunk reads the document from
the cache. Anthropic's engineering post reports that contextual embeddings plus contextual BM25 cut
top-20 retrieval failures by **49%**, and by **67%** when reranking is added.

Design rules the lab enforces:
- Keep **volatile values** (version numbers, dates, owners) out of the context text. Store them in
  metadata. Otherwise every document revision changes the text of every chunk and forces a full
  re-embed.
- Keep the original chunk. The context is an addition for retrieval, not a replacement.
- Check `usage.cache_read_input_tokens`. A prefix below the model's minimum cacheable size is
  **silently not cached** (in the lab, the short FAQ).

### Indexing

| Index | Finds | Misses | Notes |
|---|---|---|---|
| **BM25** (inverted index) | Exact terms, IDs, codes ("Form PRC-7"), rare words | Paraphrases, synonyms | k1 about 1.2, b about 0.75; cheap, explainable, incremental |
| **Dense vectors** | Paraphrases, concepts ("new baby" -> parental leave) | Exact codes, negations, rare tokens | Approximate nearest neighbour indexes (HNSW, IVF) at scale; model-specific vector space |
| **Hybrid + RRF** | Both | - | `score = sum(1/(k + rank))`, k about 60; rank-based, so no score normalisation needed |
| **Reranker** | Better ordering of the top 20-100 candidates | - | Cross-encoder reranker, or Claude on a small candidate set; adds latency |

**Metadata filtering.** Pre-filtering (restrict the candidate set, then rank) guarantees k results
that satisfy the filter. Post-filtering (rank, then drop) can leave you with nothing. Use metadata
for **authority and recency** (policy beats FAQ, newest version wins), tenancy, and ACLs (D3.2). Do
not count on similarity to resolve conflicting sources: the stale FAQ in the lab wins on similarity
because it uses the users' own phrasing.

**Embedding model pinning.** Query and document vectors must come from the **same model and
version**. A mismatch raises no error; similarity simply turns into noise. Record the model in the
index manifest, refuse mismatched queries, and treat a model change as a **full re-embed into a new
index**, followed by evaluation and a blue/green swap.

### Freshness and refresh

- **Stable chunk IDs** (doc_id + section slug), not positional IDs, so an edit in one section does
  not renumber every later chunk.
- **Content hash** per chunk. Unchanged hash: metadata-only update. Changed hash: regenerate the
  context and re-embed. Missing from the new version: delete (tombstone).
- Trigger refresh from **change events** (webhooks or CDC from the document system), not from a
  nightly full rebuild, when freshness matters.
- **Freshness guard at answer time.** Compare the version a citation points to with the source of
  truth, and flag or refuse stale answers.

### Grounded generation on Claude

- Pass retrieved chunks as **`search_result` content blocks** (`source`, `title`, `content: [text
  blocks]`, `citations: {enabled: true}`). Claude returns `search_result_location` citations that
  point to exact text blocks. `document` blocks with citations work for whole documents.
- Splitting chunks into sentence-level text blocks makes citations precise to the sentence.
- Citations **cannot** be combined with structured outputs (`output_config.format`); the API
  returns a 400.
- Tell the model to answer only from the provided sources and to say "not found" otherwise. Then
  *test* the not-found path (the lab's stock-ticker question).

## Architecture

```mermaid
flowchart LR
  subgraph Ingest
    D[Markdown docs + front matter] --> P[parse_doc]
    P --> C1[fixed] & C2[sentence] & C3[markdown recursive]
    C3 --> CX[Contextualize: Claude Haiku<br/>full doc in cached prefix]
  end
  CX --> IDX
  subgraph IDX[Index per strategy]
    B[BM25 inverted index]
    V[Vector index<br/>hash-embed-v1 stand-in]
    M[(chunk store + manifest:<br/>content hashes, metadata, model id)]
  end
  Q[User question] --> F{metadata pre-filter}
  F --> B & V --> R[RRF fusion] --> G[Claude Opus<br/>search_result blocks + citations]
  G --> FG[Freshness guard:<br/>cited version vs registry]
  U[Doc v4.0 published] --> H[hash diff per stable chunk id] -->|changed only| CX
  E[queries.json labels] --> EV[recall@3 / MRR@10<br/>per strategy x retriever]
  IDX --> EV
```

```
docs ──► chunkers ──► (contextualize w/ cached doc) ──► BM25 + vectors + manifest
                                                              │
question ──► pre-filter ──► BM25 top20 ┐                      │
                        └─► vector top20 ┴─► RRF ─► top-k ─► Claude (search_result + citations)
                                                              │
doc update ──► chunk ──► sha256 per stable id ──► re-embed changed only ──► freshness guard OK
```

## What the code demonstrates

| Step | Printed section | Code (`python/main.py`, same names in camelCase in `typescript/main.ts`) |
|---|---|---|
| 1 | Corpus and metadata | `parse_doc`, `load_corpus` (front matter: doc_id, doc_type, version, updated_at) |
| 2 | Chunking strategies | `chunk_fixed`, `chunk_sentences`, `chunk_markdown` + `split_recursive`; "evidence intact" check; no-overlap comparison |
| 3 | Contextual chunking with caching | `contextualize`: Haiku, `effort: low`, `cache_control` on the document block; usage totals with and without cache |
| 4 | Indexes | `BM25Index` (incremental add/remove), `Embedder` (signed feature hashing, frozen idf), `VectorIndex`, `rrf`, `RagIndex` |
| 5 | Retrieval evaluation | `evaluate`: recall@3 and MRR@10 for 4 strategies x 3 retrievers on 14 labeled queries |
| 6 | Metadata filtering | `RagIndex.allowed` pre-filter: stale FAQ vs authoritative policy |
| 7 | Grounded answers | `answer` builds `search_result` blocks; `print_answer` prints citations; the not-found path |
| 8 | Index refresh | Stale answer flagged by the freshness guard, then `refresh_document` (hash diff, 2 of 9 chunks re-embedded) |
| 9 | Embedding mismatch | v1 index queried with a v2 encoder: vector recall collapses; `EmbeddingModelMismatch` guard |

The embedding model is a **deterministic local stand-in** (feature hashing plus a small synonym
map) so the lab runs offline. It is not a real embedding model. In production, use a partner
embedding model. Tokenizer, hash function and formulas are identical in both languages, so the
rankings match exactly.

## Run it

From the repository root:

```bash
.venv/Scripts/python "D3.5 - RAG Pipeline - Chunking and Indexing/python/main.py"
node "D3.5 - RAG Pipeline - Chunking and Indexing/typescript/main.ts"
```

Offline by default: the Claude calls (contextualization, answers) go through the shared mock, and
the responder computes its reply from the request it receives. Set `ANTHROPIC_API_KEY` to run the
same code against the real API.

## Exam tips & traps

- **"The answer is outdated even though the document was updated"** is an indexing problem
  (stale index, missing refresh trigger). Fix the pipeline with change-driven incremental
  re-indexing and version metadata. Changing the model, the prompt or top-k does not fix it.
- **Changing the embedding model means re-embedding everything.** Vectors from different models
  (or versions or dimensions) are not comparable. Mixing them degrades quality silently, with no
  error.
- **Exact identifiers (SKUs, error codes, form numbers) are missed by vector search.** The answer
  is hybrid retrieval with BM25. **Paraphrases are missed by BM25.** The answer is dense or hybrid
  retrieval, plus Contextual Retrieval.
- **Anthropic does not offer an embeddings model.** An option such as "use Claude's embeddings
  endpoint" is a distractor. The documented path is a partner model (e.g. Voyage AI).
- **A small, stable knowledge base** (under about 200K tokens) usually does not need RAG. Put it in
  the prompt and cache it.
- **Conflicting sources** are resolved with metadata (authority, recency, version filters), not by
  hoping similarity ranks the right one first.
- **Contextual Retrieval cost** stays low because of **prompt caching** of the full document across
  that document's chunks. Volatile metadata stays out of the embedded text.
- **Measure retrieval separately** (recall@k, MRR) before tuning generation. If the evidence is not
  in the top k, no prompt will recover it.

### Practice questions

**Q1.** A benefits assistant built on Claude answers "60 days" for the expense deadline, but HR
changed the policy to 30 days last week. Traces show the retrieved chunk contains "60 days" and the
answer cites it correctly. What is the BEST corrective action?

- A. Switch the generator from Sonnet to Opus to reduce hallucinations
- B. Add "always use the latest policy" to the system prompt
- C. Add change-triggered incremental re-indexing with stable chunk IDs and content hashes, plus a version check on citations
- D. Increase top-k from 5 to 20 so newer chunks are also retrieved

**Answer: C.** The model is faithful to stale context, so this is not hallucination (A) and not a
prompt problem (B). Raising top-k (D) cannot surface a chunk that was never indexed. Fixing
freshness in the indexing pipeline, and detecting staleness at answer time, addresses the root
cause.

**Q2.** Engineering runbooks are indexed as 512-token fixed windows with no overlap and dense
vectors only. Searches for error codes such as `ERR-4471` return unrelated runbooks, and
paraphrased questions often retrieve chunks that start mid-procedure. Which change addresses
**both** problems with the least operational risk?

- A. Double the embedding dimension and re-embed
- B. Switch to heading-aware recursive chunking with section metadata, and use hybrid BM25 + vector retrieval fused with RRF
- C. Increase the chunk size to 4,096 tokens so procedures are never split
- D. Fine-tune Claude on the runbooks and remove retrieval

**Answer: B.** Structure-aware chunks keep procedures whole and carry section context. BM25 in a
hybrid setup catches exact codes. A larger dimension (A) does not help exact-token matching. Huge
chunks (C) dilute the embeddings and inflate token cost. Fine-tuning (D) does not keep up with
changing runbooks and gives up citations.

## Further reading

- Anthropic engineering blog: *Introducing Contextual Retrieval*
- Claude docs: *Embeddings* (Voyage AI partner models)
- Claude docs: *Search results* content blocks and *Citations*
- Claude docs: *Prompt caching* (minimum cacheable prefix, cache_control placement)
- Claude docs: *Long context prompting tips* (put documents first, the question last)
- Anthropic cookbook: Contextual Retrieval and RAG evaluation notebooks
- Related labs: D2.4 (context window and token management), D2.5 (caching), D3.6 (retrieval by data shape), D4.2 (evaluation datasets)
