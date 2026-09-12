# Vectorless RAG

> Status: concept complete. Implementation ships in **v0.1.0**.

## What it does

```
question -> lexical search (BM25 and PostgreSQL full text) -> fusion -> context -> LLM -> answer + citations
```

No embeddings anywhere. Retrieval ranks chunks by how well their words match the question's words, using
algorithms that weight rare terms higher and stop long documents from winning by volume.

## Why it is needed

Vector retrieval blurs exact tokens. "HR-POL-2026-07", "Priya Raman", "kubectl rollout undo" are precise
strings whose meaning is their spelling. Lexical retrieval matches them exactly, needs no embedding
model, costs nothing to index, and is fully explainable: you can see which terms matched.

## How it works internally

### BM25

For a query with terms q1..qn and a chunk d:

```
score(d) = sum over qi of  IDF(qi) * ( tf(qi, d) * (k1 + 1) ) / ( tf(qi, d) + k1 * (1 - b + b * |d| / avgdl) )
IDF(qi)  = ln( (N - n(qi) + 0.5) / (n(qi) + 0.5) + 1 )
```

| Symbol | Meaning | Effect |
|---|---|---|
| `tf(qi, d)` | Times term appears in the chunk | More occurrences, higher score, with saturation |
| `n(qi)`, `N` | Chunks containing the term, total chunks | Rare terms weigh more (IDF) |
| `k1` (about 1.2 to 2.0) | Saturation | Stops the tenth occurrence counting as much as the first |
| `b` (about 0.75) | Length normalisation | Stops long chunks winning just by containing more words |
| `|d|`, `avgdl` | Chunk length, average chunk length | Used by `b` |

This is why it is not "keyword search" in the naive sense. Saturation and length normalisation are what
make BM25 competitive with vector retrieval on many benchmarks.

### PostgreSQL full text search

`to_tsvector` stems words (running, runs, ran become run), removes stop words and stores positions.
`to_tsquery` and `phraseto_tsquery` support boolean and phrase queries. `ts_rank_cd` scores by cover
density, rewarding query terms that occur close together. It runs inside the database with a GIN index,
so there is nothing extra to operate.

### Fusion

BM25 and full text ranks are combined with reciprocal rank fusion:

```
rrf(d) = sum over rankers of 1 / (k + rank_r(d)),  k = 60
```

RRF needs no score calibration between systems, which is why it is also the standard way to fuse lexical
with vector results in hybrid search.

### Boosting

Quoted phrases are matched as phrases. Tokens that look like identifiers (mixed letters, digits and
dashes) get a boost. Metadata filters (collection, document type, date range) apply before ranking.

## Where it performs well

Exact terminology, names, IDs, error codes, technical terms, known phrases, legal clause numbers,
low cost retrieval at scale, and any corpus where users search the way they would search a wiki.

## Where it performs poorly

Synonyms and paraphrase ("how many holidays" versus "leave entitlement"), misspellings, cross lingual
queries, and questions phrased very differently from the source text. It matches tokens, not meaning.

## Alternatives

| Alternative | Trade off |
|---|---|
| Vector retrieval | Meaning over spelling; needs an embedding model and store |
| Hybrid fusion | Best of both, two indexes |
| Learned sparse retrieval (SPLADE) | Lexical form with learned term weights; needs a model |
| OpenSearch or Elasticsearch | Same algorithms, more operations |

## Metrics to watch

Hit rate on the exact match category of the evaluation set, and the gap to Traditional on the
paraphrase heavy categories. That gap is the argument for the router.
