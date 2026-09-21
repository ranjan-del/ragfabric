# ADR 0007: BM25 computed in SQL, over statistics maintained on write

Status: accepted
Date: 2026-09-21
Supersedes: none
Related: ADR 0003 (access control inside retrieval), ADR 0006 (pinned embedding model and dimension)

## Context

Phase 4 adds a retrieval strategy that finds the right chunk without embeddings. That
requires real BM25, not a proxy for it. PostgreSQL already ships `ts_rank_cd`, and Phase 2
already stores a `tsvector` per chunk in `chunk_search`, but `ts_rank_cd` is not BM25. It
has no inverse document frequency at all, so a term present in almost every chunk counts
as much as a term present in three. It is also called here without a normalisation
argument, so PostgreSQL's default applies and no document length normalisation is done,
which leaves a long chunk that mentions a term once competing on equal footing with a
short one about nothing else.

BM25 needs three quantities the schema did not hold:

- `tf(t, D)`, how often term `t` occurs in chunk `D`
- `df(t)`, how many chunks contain term `t`, and `N`, the corpus chunk count
- `|D|` and `avgdl`, the length of this chunk and the mean chunk length

The question was where those live and where the arithmetic runs.

## Decision

BM25 is computed in SQL, next to the data, over statistics maintained incrementally on
write.

- `tf(t, D)` is read out of the `tsvector` that `chunk_search` already stores. No
  row-per-term-per-chunk table is introduced.
- `df(t)` lives in a new `term_stats` table, one row per term.
- `N` and the running total behind `avgdl` live in `corpus_stats`, a single row with
  `id = 1` holding `n_chunks` and `sum_len`.
- `|D|` is a new `chunk_search.doc_len` column.
- `stores/term_stats.py` maintains all of it on every index and every delete, applying the
  delta in SQL so concurrent writers cannot lose an update.
- The formula is the Lucene variant, with `k1 = 1.2` and `b = 0.75`:

      score(D, Q) = sum over t in Q of  IDF(t) * ( tf(t,D) * (k1 + 1) )
                                       / ( tf(t,D) + k1 * (1 - b + b * |D| / avgdl) )

      IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

  The `+ 1` inside the logarithm is not cosmetic. Robertson's original IDF goes negative
  for a term present in more than half the corpus, which lets a chunk containing a common
  query term score below a chunk containing none of them. The `+ 1` floors IDF at zero, so
  a common term contributes nothing rather than actively subtracting.

`corpus_stats` stores running totals rather than a precomputed average because a mean
cannot be updated incrementally without drifting, while a running sum stays exact. `avgdl`
is derived as `sum_len / n_chunks` at query time.

The access filter stays inside the ranking query, per ADR 0003. It is a predicate in the
same SELECT that computes the score, not a pass over results the database already ranked.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| `rank_bm25` in the Python process | Real BM25, but it holds the whole corpus in process memory. Every API worker keeps its own copy, the copies drift apart as documents change, and the ceiling becomes RAM rather than disk. Kept as an explicitly capped option in `stores/bm25_memory.py` for small corpora, where those costs do not bite. |
| `pg_search` (ParadeDB), native BM25 | pgvector is available on essentially every managed PostgreSQL; `pg_search` is not. Depending on it would force self hosters onto a custom PostgreSQL image and lock managed PostgreSQL users out of the product entirely. Portability of the deployment target outranks the convenience of not writing the formula. |
| A `chunk_terms` table, one row per term per chunk | The obvious way to hold `tf`, and the reason it was rejected is size: at a million chunks that table runs to hundreds of millions of rows, with an index larger than the corpus it describes. The `tsvector` already encodes exactly this and is already written. |
| Recompute `df` and `avgdl` per query | Correct and always fresh, but it is a full scan of the corpus on every single query, which is the cost BM25 exists to avoid. |
| Recompute the statistics on a schedule | Cheaper per query, but it makes ranking a function of when the job last ran. A document ingested after the last run is ranked against statistics that do not know it exists. |
| `ts_rank_cd` alone | Already present and free, but it is not BM25. No length normalisation and no IDF, which are the two things that make BM25 worth having. It is kept as the second ranking in the fusion (ADR 0008) rather than as the ranking. |

## Consequences

- **Corpora indexed before this phase must be re-indexed.** Rows written before Phase 4
  have `doc_len = 0`. Those rows are excluded from BM25 results rather than scored,
  because a zero length would zero the `b * |D| / avgdl` term, shrink the denominator, and
  rank every legacy row above everything else. Excluding them is visible in the results;
  scoring them would be silently wrong. `ragfabric reindex --lexical-only` brings them
  forward without re-embedding anything.
- Every write path now carries a statistics update. Ingest and delete cost slightly more
  so that query time costs much less.
- `term_stats` grows with the vocabulary of the corpus, not with its size, so it stays
  small relative to the chunk table.
- The ranking is only as good as the tokenisation feeding the `tsvector`. Changing the
  text search configuration changes what a term is, and therefore invalidates `term_stats`
  and `doc_len` together. That is a re-index, the same as changing the embedding model is
  under ADR 0006.
- BM25 is computed on the PostgreSQL path. The SQLite development path runs the same code
  through a portable fallback, so behaviour is testable without a database server, but the
  performance claim applies to PostgreSQL only.
