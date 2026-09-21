# Embeddings, concept notes

> Status: Phase 3 (shipped). This is a learning document: why embeddings and the vector store are
> shaped the way they are, not an API reference. For the code, see
> `packages/core/src/ragfabric_core/embeddings/`, `providers/`, `tokens.py` and `stores/`.

## From text to a stored vector

```
text -> tokeniser -> embedding model -> vector (e.g. 768 floats for nomic-embed-text) -> normalise -> vector store
```

A tokeniser splits text into the units the model was trained on, sub-word pieces more often than whole
words. The embedding model reads those tokens and produces one fixed length vector: a point in a high
dimensional space where, if the model is any good, meaning is geometry. "Annual leave" and "yearly
holiday entitlement" land close together even though they share no tokens at all. The vector is then
L2 normalised (see below) and written to the vector store alongside the chunk's id, its collection and
document ids for the access filter, and the name of the model that produced it.

The same model embeds both a chunk at ingestion time and a question at query time. That is not
incidental: two vectors from two different models occupy different, unrelated spaces, and comparing
them produces a number that looks like a similarity score but means nothing (see "Why one model per
deployment" below).

## Cosine similarity, dot product, and why normalising makes them the same number

Two ways to compare vectors show up in this codebase:

- **Cosine similarity** measures the angle between two vectors, ignoring their length. It answers "do
  these point in the same direction", which is what "similar meaning" is supposed to mean.
- **Dot product** (element-wise multiply, then sum) is cheaper to compute and is what a plain matrix
  multiplication gives you for free, but it is sensitive to vector length: a longer vector wins even
  if its direction is a worse match.

The two become the **same number** exactly when every vector involved has length 1. That is what L2
normalisation does: divide a vector by its own magnitude, `v / ||v||`. `embeddings/normalise.py` does
this once, at write time, for every stored vector (and for every query vector before it is compared).

This is not a performance nicety; it fixed a real, observed problem. pgvector's cosine distance
operator and the SQLite development fallback's dot product ranking would otherwise disagree with each
other on which chunk ranks first for the same corpus and the same question, purely because of a
dialect difference in which operation each backend runs fastest. Normalising once, centrally, at the
one place a vector is written, means every store, every dialect and the reindexer score identically,
and a stored score is directly comparable across queries and across models (once they are all
normalised the same way).

## Why one embedding model per deployment (ADR 0006)

`chunk_embeddings.embedding` is a `vector(768)` column, a fixed dimension, with an HNSW index built on
top of it (`vector_cosine_ops`), added by migration 0004. Two facts forced this:

1. **pgvector cannot build an HNSW or IVFFlat index on a column with no declared dimension.** Without
   an index, every query is a sequential scan over the whole corpus; the index needs to know, once and
   for all, how long a vector in that column is.
2. **Vectors from two different models are not comparable**, even if they happen to have the same
   dimension. Ranking a `nomic-embed-text` vector against a `text-embedding-3-small` vector produces a
   confident-looking number, not an error, and that is worse than a crash: nothing downstream would
   notice the ranking was meaningless.

So one deployment pins one model and one dimension. Every vector query additionally filters on the
active embedding model's name (`stores/access_sql.py` and the store implementations), so even a row
left behind by a previous model can never silently enter a ranking. Serving two embedding models at
once, so they could be compared against each other, is explicitly an evaluation concern for Phase 8,
where it can be done against two separate indexes with a measured result, rather than inside one
ranking with no way to tell which model actually won.

## What reindexing costs

Changing `embeddings.model` (or `embeddings.provider`) does not update anything by itself: the stored
vectors were produced by the old model and mean nothing compared against a query embedded with the
new one. `ragfabric reindex` re-embeds every chunk in the corpus under the newly active model, which
means:

- **One embedding call per chunk** (batched where the provider supports it), so the cost scales
  linearly with corpus size, not with how much has changed since the last reindex. There is no
  incremental "only re-embed what changed" mode; the whole corpus is re-embedded every time.
- **A real cost or a real wait**, depending on the provider: a paid API meters this by the token, a
  local Ollama model pays it in wall-clock time instead.
- **A dimension change needs a migration first**, not just a reindex, if the new model's output length
  differs from the pinned `embeddings.dim` (768 by default). The column and its HNSW index are fixed at
  build time; pgvector cannot store a vector of a different length in that column.

See [configuration.md](../configuration.md#changing-the-embedding-model) for the exact runbook.
