# ADR 0006: One pinned embedding model and dimension per deployment

Status: accepted
Date: 2026-09-18
Supersedes: none
Related: ADR 0003 (access control inside retrieval)

## Context

Phase 2 wrote `chunk_embeddings` rows with a `model` name and a `dim` per row, and left
the vector column unconstrained so one migration could serve both PostgreSQL and SQLite.
That was right for a table nothing queried. Phase 3 queries it, and two facts now bite:

1. pgvector cannot build an HNSW or IVFFlat index on a `vector` column with no declared
   dimension. Without an index every query is a sequential scan over the whole corpus.
2. Vectors from two different models occupy different spaces. Ranking them against each
   other produces confident nonsense, not an error.

## Decision

A deployment pins one embedding model and one dimension.

- The vector column is `vector(768)`, fixed by migration 0004. 768 is `nomic-embed-text`,
  the default no key model.
- An HNSW index with `vector_cosine_ops` is created on that column.
- Every vector query filters on the active model name, so a row written by another model
  can never enter a ranking even if it is present.
- Vectors are stored L2 normalised, so cosine distance and dot product agree and a score
  means the same thing on PostgreSQL and on the SQLite development path.
- Changing the embedding model requires `ragfabric reindex`. Changing to a model with a
  different dimension additionally requires a one line migration. Both are documented in
  `docs/configuration.md`.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| Leave the column unconstrained and accept sequential scans | Honest but slow, and it makes the product's own benchmark numbers a function of corpus size rather than of the retrieval design. An adopter with 100k chunks would measure our scan, not our ranking. |
| Store a dimension per row and build one partial index per dimension | pgvector indexes cannot be built on an unconstrained column at all, partial or otherwise, so this does not exist as an option. |
| One table per embedding model | Supports two models at once, at the cost of a dynamic table name in every query, a migration per model, and an access predicate duplicated per table. The benefit is an evaluation feature, and Phase 8 can get it with a second index rather than a second schema. |
| Keep both models' rows and disambiguate only at query time | The model filter alone does make ranking correct, and this task ships it. It is not sufficient: the column still cannot be indexed, and stale rows grow the table forever with no process to remove them. |
| Read the dimension from `ragfabric.yaml` inside the migration | Makes `alembic upgrade head` produce different schemas on different machines from the same revision, which breaks the drift test, CI reproducibility and any support conversation about "what does your schema look like". |

## Consequences

- Migration 0004 deletes `chunk_embeddings` rows whose dimension is not 768, because the
  column type change cannot succeed otherwise. Those rows are derived data, rebuilt by
  `ragfabric reindex` from `chunks`, which is the source of truth. No user content is lost.
- A deployment cannot serve two embedding models at once. That is deliberate. Comparing
  embedding models is an evaluation concern and belongs to Phase 8, where it can be done
  against separate indexes with measured results rather than silently inside one ranking.
- The pinned dimension is a constant in the migration, not a value read from configuration,
  so `alembic upgrade head` produces the same schema on every machine.
