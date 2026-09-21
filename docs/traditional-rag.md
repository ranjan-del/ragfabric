# Traditional RAG

> Status: **shipped in v0.1.0** (Phase 3). `TraditionalRAGStrategy`
> (`packages/core/src/ragfabric_core/strategies/traditional.py`) replaced the v1 in memory pipeline
> behind the unchanged `RetrieverStrategy` interface. Retrieval quality (whether an answer is correct,
> complete or faithful to its sources) is **not measured**; Phase 8 measures it.

## What it does

```
question -> embed -> vector search (top k, access filter inside the query) -> similarity threshold
         -> rerank -> cut to top k -> context budget -> LLM -> verify citation contract -> answer + citations
```

The question and every chunk are turned into vectors by the same embedding model, L2 normalised so
cosine distance and dot product agree across PostgreSQL and SQLite. Chunks whose vectors point in a
similar direction to the question's vector are retrieved with the caller's access filter applied
**inside** the store query (ADR 0003), filtered by a minimum similarity, optionally re-scored by a
stronger model, fit to a token budget, assembled into a prompt, and the LLM answers from that prompt
with numbered citations, which are then checked against a mechanical citation contract before the run
is recorded.

## The pipeline order, and why it is fixed

The order above (embed, store query with the access filter inside it, threshold, rerank, cut to
`top_k`, context budget) is deliberate and asserted by the strategy's own tests, not incidental:

- **The threshold runs before the rerank.** `similarity_threshold` is expressed against the retrieval
  score (cosine similarity from the vector store). Comparing that same configured number to a
  reranker's score, which is a different, unrelated scale, would make the configured threshold
  meaningless: a value tuned for cosine similarity has no defined meaning against an LLM's or a cross
  encoder's own scoring.
- **The store is asked for more candidates than `top_k`** (`top_k * candidate_multiplier`, 3 by
  default), so the reranker has a genuinely larger pool to improve on; the cut down to `top_k` happens
  **after** reranking, so with the `none` reranker the behaviour is identical to asking the store for
  `top_k` directly.
- **The context budget drops whole chunks from the tail, never splits one.** A chunk's citation offsets
  (`char_start`, `char_end`) are checked against the text actually stored for that chunk. Splitting a
  chunk to fit a token budget would leave its stored character span pointing at text the answer no
  longer has access to, which breaks citation rendering; dropping the whole chunk keeps every kept
  chunk's span exact.

## Why it is needed

A language model knows nothing about your documents and will guess when asked. Retrieval supplies the
relevant passages so the model reads instead of guessing, and citations let a human check.

## How it works internally

### Embeddings

```
text -> tokenizer -> embedding model -> vector (for example 1536 floats) -> vector store
```

An embedding model maps text to a point in a high dimensional space where meaning is geometry: "annual
leave" and "yearly holiday entitlement" land close together even though they share no words. Vectors are
usually L2 normalised so the dot product equals cosine similarity, which turns "how similar" into one
matrix multiplication.

### Chunking

Documents are split into passages of a few hundred tokens with overlap. Too small and a passage loses the
context needed to answer; too large and one vector has to represent several topics, which blurs it.
Overlap stops an answer being cut in half at a boundary. Page numbers and character spans travel with
every chunk so a citation can point at a place, not just a file.

### Vector search

A vector store indexes chunk vectors for approximate nearest neighbour search (HNSW in pgvector and
Chroma). `top_k` controls how many candidates come back, though the strategy asks the store for
`top_k * 3` by default so a configured reranker has a genuinely larger pool to work from. The store
always returns its nearest neighbours, even when the nearest neighbour is unrelated, which is why the
next step exists. `PgVectorStore` and `ChromaVectorStore` both apply the caller's access filter
**inside** the query itself, so a chunk the caller may not read is never ranked and never reaches this
strategy at all.

### Similarity threshold

Candidates below `similarity_threshold` are dropped. This is what separates a retriever from a random
passage generator: with no floor, an unanswerable question still produces confident quotes from whatever
ranked first. The threshold runs before reranking, because it is expressed against the retrieval score;
see "The pipeline order" above.

### Reranking

The embedding model compares the question and each chunk independently. A cross encoder or an LLM
reranker reads them together and scores the pair, which is more accurate and much slower. Typical
pattern: retrieve 20 to 50 by vector, rerank, keep 5. See
[concepts/reranking.md](concepts/reranking.md) for why this trade works and when it does not pay.

### Context construction and citations

Kept chunks are numbered `[1]`, `[2]`, ... and placed in the prompt with an instruction to answer only
from them and cite by number. `POST /api/ask` streams the answer over SSE by default: `retrieval`,
then a `token` event per generated piece, then `citations`, then `done`. A response is checked against
a mechanical citation contract before it is recorded:

1. **Marker validity**: every `[n]` refers to a chunk that was actually retrieved.
2. **Quote fidelity**: a quoted span of eight or more characters, or any quoted span containing a
   digit, must appear verbatim in the chunk it is attributed to. A shorter, non-numeric quote is not
   checked; that is a deliberately named gap, not an oversight, because verifying it strictly would
   trigger retries for phrasing that carries no factual risk.
3. **Grounding**: an answer with evidence available must carry at least one citation.

The contract does **not**, and cannot, verify that a paraphrase is faithful to its source. That is a
semantic judgement, not a mechanical one, and asserting it here would be exactly the kind of fabricated
guarantee ADR 0004 forbids. Phase 8 is where faithfulness gets measured, with a question set and a
scoring harness.

If the streamed answer fails the contract, the client has already rendered tokens it cannot unsend, so
the answer is regenerated (the generator gets two attempts) and the correction is announced with a
`superseded` SSE event carrying the corrected text, rather than silently recording a different answer
from the one the user saw. This is a real, observed behaviour, not a hypothetical: on the first
question this project ever answered end to end, a local `llama3.2:3b` model's first streamed pass
carried no `[n]` marker at all, the contract caught it, and the CLI printed
`notice: the streamed answer above failed the citation contract and was corrected` followed by the
repaired, cited answer. If the generator still cannot produce a compliant answer after its allowed
attempts, the run falls back to the extractive generator (verbatim quotes only, no paraphrase, from
the v1 path) rather than shipping an answer that failed the contract.

## Configuration exposed

| Parameter | Effect | Typical |
|---|---|---|
| `top_k` | Candidates returned to the caller (the store is asked for `top_k * 3`) | 5 to 20 |
| `similarity_threshold` | Minimum cosine to keep a chunk | 0.2 to 0.4 depending on model |
| `chunk_size`, `chunk_overlap` | Ingestion time chunking | 400 to 800 tokens, 10 to 20 percent overlap |
| `reranker.kind` | none, llm (needs no extra), cross_encoder (needs `ragfabric[rerank]`) | none for speed, llm or cross_encoder for quality |
| `max_context_tokens` | Prompt budget; whole chunks are dropped from the tail to fit it, never split | model dependent |
| `vector_store.kind` | pgvector (default) or chroma | pgvector needs one less service |

`top_k`, `similarity_threshold` and `reranker.kind` (as `rerank`) can also be overridden per request on
`POST /api/ask`; `chunk_size` and `chunk_overlap` can be overridden per upload; `max_context_tokens` and
`vector_store.kind` are deployment wide only.

## Alternatives and trade offs

| Alternative | When it is better | Cost |
|---|---|---|
| Vectorless (BM25) | Exact identifiers, names, codes | Misses paraphrase |
| Hybrid fusion (vector plus BM25) | Most corpora, strong default | Two indexes to maintain |
| Long context stuffing | Tiny corpus, few queries | Cost per query grows with corpus |
| Fine tuning | Style or format, not facts | Expensive, stale immediately |

## Where it fails

- The right chunk exists but is not in the top k (retrieval miss). Fix with hybrid search, a better
  embedding model, larger k with reranking, or query rewriting.
- Exact tokens: an invoice number or product code is one token among many in a vector; BM25 wins.
- Tables and numbers: embeddings capture topic, not figures.
- Multi hop: the answer needs facts from three documents combined; a single retrieval cannot plan.

## Metrics to watch

Hit rate and MRR on the evaluation set, share of queries where all candidates fell below the threshold
(the "I don't know" rate), latency split between embedding call, store query, rerank and generation.
