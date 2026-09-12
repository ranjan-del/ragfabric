# Traditional RAG

> Status: concept complete. Implementation ships in **v0.1.0**. The v1 code on `main` already runs a
> version of this with an offline hashing embedder; v0.1.0 replaces the embedder and store behind the
> same interfaces.

## What it does

```
question -> embed -> vector search (top k) -> similarity threshold -> optional rerank -> context -> LLM -> answer + citations
```

The question and every chunk are turned into vectors by the same embedding model. Chunks whose vectors
point in a similar direction to the question's vector are retrieved, filtered by a minimum similarity,
optionally re-scored by a stronger model, assembled into a prompt, and the LLM answers from that prompt
with numbered citations.

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
Chroma). `top_k` controls how many candidates come back. The store always returns its nearest
neighbours, even when the nearest neighbour is unrelated, which is why the next step exists.

### Similarity threshold

Candidates below `similarity_threshold` are dropped. This is what separates a retriever from a random
passage generator: with no floor, an unanswerable question still produces confident quotes from whatever
ranked first.

### Reranking

The embedding model compares the question and each chunk independently. A cross encoder or an LLM
reranker reads them together and scores the pair, which is more accurate and much slower. Typical
pattern: retrieve 20 to 50 by vector, rerank, keep 5.

### Context construction and citations

Kept chunks are numbered `[1]`, `[2]`, ... and placed in the prompt with an instruction to answer only
from them and cite by number. The response is parsed for markers; each marker maps back to a chunk,
which maps back to a document and page. A deterministic check confirms every marker refers to a chunk
that was actually in the context.

## Configuration exposed

| Parameter | Effect | Typical |
|---|---|---|
| `top_k` | Candidates fetched from the store | 5 to 20 |
| `similarity_threshold` | Minimum cosine to keep a chunk | 0.2 to 0.4 depending on model |
| `chunk_size`, `chunk_overlap` | Ingestion time chunking | 400 to 800 tokens, 10 to 20 percent overlap |
| `rerank` | none, llm, cross_encoder | none for speed, llm for quality |
| `max_context_tokens` | Prompt budget | model dependent |

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
