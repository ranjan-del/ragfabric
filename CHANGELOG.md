# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

Release plan (see [ROADMAP.md](ROADMAP.md) for the phases inside each release):

| Version | Theme | Contents |
|---|---|---|
| v0.1.0 | Initial RAG engine | Monorepo, interfaces, providers, ingestion, access control, Traditional and Vectorless RAG, CLI, Python SDK, console v1 |
| v0.2.0 | Agentic retrieval | Bounded agent loop with a repair policy, budgets and traces |
| v0.3.0 | Graph retrieval | Neo4j knowledge graph build and Graph RAG strategy |
| v0.4.0 | Adaptive router | AUTO and MANUAL modes, RouterDecision, fallbacks |
| v0.5.0 | Evaluation framework | Corpus, question set, metrics, `make eval`, dashboards, generated benchmarks |
| v1.0.0 | Production release | Reference UI with Compare and Trace, TypeScript SDK, connectors, hardening, docs site, deployment guides |

## [0.2.0] - unreleased

Agentic retrieval. An agent that decomposes a question, judges its own evidence, and when it comes
up short chooses a repair move and tries again, stopping honestly when it is not making progress.

### Added
- **Agentic RAG (Phase 5).** `AgenticRAGStrategy`, selectable as `strategy: "agentic"` on
  `POST /api/ask` and `POST /api/search/query`, as `ragfabric ask --strategy agentic`, and through
  the Python SDK. `POST /api/search/hybrid` continues to accept only `traditional`, because a
  hybrid is one vector ranking fused with one lexical ranking and an agent cannot stand in for
  either leg.
- A **sub-question ledger**: the question is decomposed, and each sub-question carries its own
  status, the tool answering it, and the history of what has already been tried on it. Evidence is
  judged per sub-question, so the loop chases only what is still missing instead of re-retrieving
  everything on every pass.
- A **repair policy with six moves**: `broaden`, `narrow`, `switch_strategy`, `decompose`,
  `fetch_document`, and `abandon`, which requires a reason. A move is never repeated on an
  unchanged sub-question, because an agent that can only alternate between broaden and narrow
  oscillates until its budget dies. The move is chosen from a fixed enum, so a small local model
  cannot invent an action that does not exist.
- **Progress detection.** Pooled evidence is keyed by chunk id, and an iteration that adds no new
  ids has not progressed however many rows the stores returned. This is the failure the textbook
  agentic design has no answer to.
- **Three termination conditions**, each recorded at the branch that decides it, never inferred
  afterwards: `resolved`, `budget` and `no_progress`.
- Three agent tools over the strategies that already exist: `semantic_search`, `lexical_search` and
  `fetch_document`, which returns a document whole and in reading order. Every tool passes the
  caller's `AccessFilter` through unchanged, and `fetch_document` applies the same predicate inside
  its SQL because it names a document directly instead of searching for one.
- **Validated JSON contracts** for every model decision, tolerant of fenced code blocks and
  surrounding prose because small models emit both, and strict about content. A malformed response
  is a reported violation, never a crash and never a silent success.
- `RetrievalResult.sub_questions`, a per sub-question report giving each one's final status and,
  for anything unanswered, the reason: budget, no evidence, or abandoned after N moves. This
  replaces a single best-effort boolean, and defaults to empty so no other strategy changes.
- Agentic generation reuses the Phase 3 citation contract unchanged, applied claim by claim. An
  unsupported claim is **removed** and the removal recorded, rather than triggering more retrieval,
  which is usually not the fix.
- Typed agent configuration under `strategies.agentic`: `max_iterations`, `max_llm_calls`,
  per-node call caps, `max_cost_usd`, `max_latency_ms`, the enabled tool list, and
  `assess_strictness`. Strict validation, so a misspelled key is an error rather than a silently
  ignored setting.
- ADR 0009 (a plain state machine rather than LangGraph) and ADR 0010 (the repair policy and the
  three termination conditions). New `docs/concepts/agentic-loops.md` and
  `docs/learning/agentic-first-run.md`.

### Changed
- **LangGraph is not used, and the roadmap promise is withdrawn.** A spike built the identical loop
  both ways. The plain state machine was 93 lines against 107, took zero new dependencies against
  38 (including a SaaS telemetry client), and kept the stop reason at the branch that decided it,
  where the framework version lost it silently. Issue #6 requires a trace of every node, so the
  framework worked against a stated requirement. See ADR 0009.

### Fixed
- Four configured agent limits were typed, validated and printed back by `ragfabric config
  validate` while nothing read any of them. `Budget` was never constructed outside its own class
  definition, so the enforced global call cap was the dataclass default of 8 while
  `ragfabric.example.yaml` advertised 12, and `max_cost_usd`, `max_latency_ms` and
  `assess_strictness` had no readers at all. All four now bind.

### Notes
- **Retrieval quality is still not measured.** No accuracy, recall or latency figure exists for any
  strategy in this project. Phase 8 builds the evaluation framework; until then there is no number
  to quote for whether agentic retrieval is better than traditional or vectorless retrieval at
  anything.
- **`dated_sources` will be empty in a live deployment.** Nothing in the data model carries a
  document effective date, and `Document.created_at` measures upload time rather than when a policy
  took effect, so substituting it would be a fabricated number. This needs an ingestion-side field
  before the dated-source reporting does anything.
- Semantic conflict detection is deliberately not shipped. Deciding that two passages genuinely
  contradict is a hard inference task that cannot be tuned without measurement, and a confident
  "these sources disagree" that is wrong is worse than silence. Deferred to Phase 8.
- The latency cap is checked between nodes and cannot interrupt a model call in flight, so a run
  can overshoot `max_latency_ms` by the duration of the node that was running. Read it as "stop
  starting new work after N", not "return within N".
- Generation happens above the loop, reusing the Phase 3 path. There is no generate node inside the
  agent.

## [0.1.0] - 2026-09-22

### Added
- Monorepo: `packages/core` (engine), `packages/server` (API), `packages/cli` (the `ragfabric` command, published as `ragfabric`), `apps/assistant` (UI). uv workspace on Python 3.13.
- Core interfaces: `RetrieverStrategy` and `RetrievalResult`, `LLMProvider`, `EmbeddingProvider`, `VectorStore`, `LexicalStore`, `GraphStore`, `Cache`, `AuthProvider`, `Connector`; `Principal` and `AccessFilter`.
- Providers: OpenAI, Anthropic, Ollama (OpenAI compatible), offline test doubles. Provider registry driven by `ragfabric.yaml`.
- `ragfabric.yaml` configuration with strict validation; `ragfabric config validate`.
- Pricing configuration (`pricing.yaml`, dated and sourced) and cost estimation that reports unknown models instead of guessing.
- Migration 0002: groups, grants, overrides, API keys, audit log, conversations, messages, retrieval runs, sources, evaluation runs and results, entities, relationships.
- Docker Compose profiles: lite (PostgreSQL 18 with pgvector, Redis, API, UI) and full (adds Chroma 1.5.9 and Neo4j 2026.08.1).
- CI: lint (ruff, import-linter), tests, migrations against PostgreSQL 18, frontend on Node 24.
- `docs/` with getting started, architecture, one document per retrieval strategy, routing, evaluation,
  configuration, providers and troubleshooting. Documents for unshipped features are marked as design
  documents and say which release delivers them.
- Phase 1 implementation plan at docs/plans/2026-09-13-phase-1-architecture.md
- `ragfabric_core.runtime` with `get_config`, `reset_config` and `get_session_factory`.
- Ingestion cleaning (`ingest/clean.py`): hyphenation repair, whitespace normalisation, removal of
  headers and footers repeated on three or more pages, heading detection for numbered, ALL CAPS and
  markdown headings, and `document_type_for`.
- Chunker attaches `section` to every chunk; `Document.document_type`, `Document.storage_path` and
  `Chunk.section` columns added by migration 0003.
- Retained originals: uploaded files kept under a configurable `uploads_dir`, at
  `<uploads_dir>/<document id>/<safe name>` with the name sanitised and capped at 120 characters;
  `GET /api/documents/{id}/download`; files removed when a document is deleted.
- `PgVectorStore` (pgvector cosine distance on PostgreSQL, a NumPy fallback on SQLite) and
  `PostgresLexicalStore` (`tsvector`, `plainto_tsquery`, `ts_rank_cd`, a GIN index, token overlap
  fallback on SQLite); migration 0003 also creates `chunk_embeddings` and `chunk_search`. Both stores
  apply the `AccessFilter` inside the query via `stores/access_sql.access_clause`.
- `MemoryCache` and `RedisCache`, plus `stores/registry` builders.
- `queue/` (`Job`, `JobQueue`, `MemoryJobQueue`, `RedisJobQueue` over one Redis list with RPUSH and
  BLPOP, `build_queue`) and `workers/` (`index_document`, an `extract_graph` placeholder, `Worker`
  with `run_once` and `run_forever`, `default_handlers`); `ingest/indexing.schedule_indexing`.
- `auth/service.py` (groups, members, grants with upsert, revoke, overrides) and
  `auth/policy.compute_access_filter`, in order: a principal with a missing or inactive API key sees
  nothing; admins are unrestricted; a collection is readable on a group grant, ownership, or when it
  carries no grants at all (open by default); a document is readable when owned, read overridden, or
  collection-less; a deny override always wins; API key collection scopes narrow everything.
  `AccessFilter` gains `denied_document_ids`.
- `auth/api_keys.py`: `rf_`-prefixed, 256 bit random keys, SHA-256 stored, a 12 character visible
  prefix, plaintext shown once, active and expiry checks; `auth/ratelimit.check_rate_limit`, a fixed
  one minute window on the `Cache`.
- Server `deps.get_principal` (X-API-Key header or `rf_` bearer token, else JWT), `get_access_filter`,
  `get_cache`; admin API under `/api/admin`: groups, group members, grants, overrides and API keys
  (create, list, revoke).
- The v1 in memory store, `Retriever` and `HybridRetriever` take `access` and filter before ranking;
  `LegacyHybridStrategy` passes it through. `/api/search/query` writes a `RetrievalRun`, `Source` rows
  and an `AuditLog(action="query")` with `sources_returned` and `sources_filtered`; `/semantic` and
  `/hybrid` write `AuditLog(action="search")`. `GET /api/runs/{id}` (owner or admin).
- CLI: `init`, `ingest PATH [--collection] [--recursive] [--owner]`, `users create|list|set-role|deactivate`,
  `groups create|add-member|list`, `grants add|list`, `keys create|list|revoke`, `worker [--once]`.
- `telemetry/tracing.py` (`start_trace`, `trace`, `configure_otel`, `otel_enabled`) with spans `parse`,
  `clean`, `chunk`, `persist_chunks`, `schedule_indexing`, `embed`, `vector_upsert`, `lexical_index`,
  `semantic_search`, `hybrid_search`, `answer`, stored on `RetrievalRun.trace`; OTLP HTTP export when
  `telemetry.otlp_endpoint` is set, configured by the API at startup.
- Compose: `worker` service under the `workers` and `full` profiles; a CI migrations job that runs the
  PostgreSQL store integration tests; `.env.example` gains `RAGFABRIC_TEST_DATABASE_URL`.
- `TraditionalRAGStrategy`: embed the question, query the configured vector store with the access filter
  inside the query, drop candidates below `similarity_threshold`, optional rerank, cut to `top_k`, fit
  a context budget that drops whole chunks rather than splitting one.
- `embeddings/normalise.py`: L2 normalisation applied once at write time, so cosine distance and dot
  product agree on PostgreSQL and SQLite and a stored score is comparable across models.
- Migration 0004 pins `chunk_embeddings.embedding` to `vector(768)` and builds an HNSW index with
  `vector_cosine_ops`; every vector query filters on the active embedding model name (ADR 0006).
- `rerank/`: `Reranker` interface with `none`, `llm` (`LlmReranker`, capped candidate count against
  silent prompt truncation by some model runtimes) and `cross_encoder` (`ragfabric[rerank]` extra,
  lazily loaded).
- `tokens.py`: exact token counts via `tiktoken` where a tokeniser is known, a labelled four
  characters per token estimate otherwise, and `fit_to_budget`.
- `generate/cited.py` and `generate/contract.py`: LLM generated answers with numbered citations, and a
  mechanical citation contract that checks marker validity, quote fidelity for quotes of eight or more
  characters or any quote containing a digit, and grounding. It does not and cannot verify that a
  paraphrase is faithful to its source; that is Phase 8's job.
- `POST /api/ask`: manual mode, SSE streaming (`retrieval`, `token`, `citations`, `done`, and
  `superseded` when a streamed answer fails the citation contract and is regenerated).
- `POST /api/documents/{id}/move`: move a document to another collection (owner or admin), updating
  `chunks.collection_id`, `chunk_embeddings.collection_id` and `chunk_search.collection_id` in the same
  transaction so the SQL-side access filter and the SQL-side index never disagree about which
  collection a document belongs to. This does not touch a Chroma vector store's own metadata copy of
  `collection_id`, so on a `vector_store.kind: chroma` deployment the move is refused (409) instead.
- `ragfabric reconcile`: retries documents stuck in `indexing` past a grace period after a crashed
  fan out. Known limitation, documented in its own help text: a merely slow document, not a crashed
  one, can be re-run while a worker is still processing it; the safe procedure is to stop the worker(s)
  first.
- `ragfabric reindex`: batched re-embedding of the whole corpus under the active model, with bounded
  memory and de-duplicated progress reporting.
- `ragfabric ask` and `packages/sdk-python` (`ragfabric_sdk`): a Python client that talks HTTP only and
  never imports `ragfabric_core`. `ragfabric ask --json` and `--no-stream` are the safe choice for a
  machine consumer, because streamed stdout can contain a stale, superseded draft ahead of the
  corrected answer.
- `stores/chroma_store.py`: Chroma as a second `VectorStore`, with the access predicate inside the
  query, selected with `vector_store.kind: chroma` and `CHROMA_URL`.
- **Vectorless retrieval (Phase 4).** `VectorlessRAGStrategy` answers without calling an embedding
  model at any point. Select it with `strategy: "vectorless"` on `POST /api/ask`,
  `POST /api/search/query` and `POST /api/search/semantic`, or `ragfabric ask --strategy vectorless`.
  `POST /api/search/hybrid` accepts only `traditional`, because a lexical-only strategy cannot stand
  in for the vector leg of a hybrid.
- Migration 0007 adds `term_stats` (document frequency per term), `corpus_stats` (one row holding
  `n_chunks` and `sum_len` as running totals, not a precomputed average, because a mean cannot be
  updated incrementally without drifting) and `chunk_search.doc_len`.
- BM25 computed in SQL (`stores/bm25_sql.py`), using the Lucene IDF variant so a term present in more
  than half the corpus contributes zero rather than a negative score. Term frequency is read from the
  `tsvector` `chunk_search` already stored since Phase 2, so no row-per-term-per-chunk table exists.
  Statistics are maintained incrementally on every index and delete by `stores/term_stats.py`.
  Recorded as ADR 0007.
- Reciprocal rank fusion (`stores/fusion.py`) combines the BM25 and `ts_rank_cd` rankings on rank
  position rather than score, because the two scores share no scale and any normaliser would have to
  be refitted per query. Ties break on `chunk_id`, so identical inputs give an identical order.
  `top_k` is applied after fusion, not per store. Recorded as ADR 0008.
- Exact phrase and identifier boosting. A boost multiplies the score of a chunk a store already
  returned and can never introduce a new one, because introducing one would bypass the access filter
  that ADR 0003 requires to run inside the store query.
- Optional in-process BM25 (`stores/bm25_memory.py`) for small corpora, behind an enforced cap and the
  `rank_bm25` extra. It skips cleanly when the extra is absent.
- Console v1 at `/console`, admin guarded: users, groups, collections, grants, API keys and provider
  configuration, so access is managed without anyone editing the database by hand. Built on a new
  token scale and nine standalone UI primitives with light and dark themes.
- Admin API completing the gaps the console needed: `POST /api/admin/users`,
  `DELETE /api/admin/users/{id}`, `PUT`/`DELETE /api/admin/groups/{id}`,
  `GET /api/admin/groups/{id}/members`, `PUT /api/admin/collections/{id}`, and
  `GET`/`PUT /api/admin/providers` with `POST /api/admin/providers/test`. No endpoint or schema on
  this surface can carry a secret; provider key presence is reported as a boolean, never a masked
  value.
- `docs/concepts/lexical-vs-vector.md` and `docs/learning/lexical-vs-semantic.md`: where lexical
  retrieval beats vectors, and equally where it loses.

### Changed
- **Lexical search requires a re-index after upgrading to Phase 4.** `chunk_search` now stores a
  per-chunk `doc_len` and, off PostgreSQL, a term list that preserves repetition, because BM25 needs
  term frequency and the previous representation was a set that destroyed it. Rows written before
  this change carry `doc_len = 0` and are **excluded from BM25 ranking rather than scored**, so
  results from an un-reindexed corpus will be incomplete rather than merely stale. Run
  `ragfabric reindex --lexical-only` to bring an existing corpus forward: it rebuilds the lexical
  index and the term statistics with no embedding calls and no vector writes.
- Frontend upgraded from Angular 17 to Angular 22 with TypeScript 6 and Tailwind 4.
- PostgreSQL driver psycopg2 to psycopg 3; connection URLs use `postgresql+psycopg://`.
- The v1 hybrid pipeline is available as `LegacyHybridStrategy` behind the strategy interface until Phase 3 replaces it.
- The Angular `ng update` schematics added `changeDetection: ChangeDetectionStrategy.Eager` to the page
  components, and `provideZoneChangeDetection()` with `withXhr()` to the application bootstrap, to preserve
  v1 behaviour under Angular 22's new defaults.
- Relicensed from MIT to Apache License 2.0 (ADR 0005).
- README rewritten for RagFabric; roadmap, design document and ADRs 0001 to 0005 added.
- Repository governance: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CODEOWNERS, issue and PR templates, Dependabot.
- `main` protected: pull request and green CI required, no force pushes, linear history.
- v1 vector store and pipeline `zip()` calls now pass `strict=True`, which turns a silent length mismatch into an error.
- `md` accepted as a text format for ingestion.
- Pipeline chunk size and overlap now come from `ragfabric.yaml` (defaults 600 and 80) rather than
  being fixed in code.
- Document status gains `indexing`, alongside `processing`, `ready` and `failed`.
- `/api/search/query`, `/api/search/semantic` and `/api/search/hybrid` accept API keys as well as JWTs.
- The documents list, get and download routes are access filtered; a document outside the caller's
  `AccessFilter` returns 404.
- **Behaviour change.** The default LLM and embedding provider is now Ollama, with `llama3.2:3b` and
  `nomic-embed-text`, rather than OpenAI. An upgrader who relies on the previous OpenAI default must set
  `llm.provider: openai` and `embeddings.provider: openai` explicitly.
- **Behaviour change.** `ragfabric serve` now binds to `127.0.0.1` by default instead of `0.0.0.0`. A
  deployment that means to expose the API on every network interface, for example inside a container
  behind its own network boundary, must pass `--host 0.0.0.0` explicitly.
- **Behaviour change.** The `openai` and `anthropic` SDKs are now optional extras (`ragfabric[openai]`,
  `ragfabric[anthropic]`) rather than base dependencies. Because Ollama is served through the OpenAI
  compatible client, an Ollama only deployment still needs the `openai` extra installed; installing
  `ragfabric` with none of `openai`, `anthropic` or `rerank` will not run the shipped default
  configuration.
- **Behaviour change.** Every timestamp column is now declared timezone aware (migration 0006). On
  PostgreSQL this is a real `TIMESTAMPTZ` column; SQLite accepts the declared type but continues to
  store and return naive values, so its actual runtime behaviour is unchanged.
- **Behaviour change.** The `limits.max_upload_mb`, `limits.allowed_types` and
  `limits.rate_limit_per_minute` keys in `ragfabric.yaml` are now enforced on upload and on API key
  rate limiting, rather than being read and ignored.
- Query the pgvector or Chroma index instead of the v1 in memory index; the v1 index no longer exists.
- Vector queries filter on the active embedding model name, so a row written by a previous model can
  never enter a ranking even if it is still present in the table.

### Removed
- The v1 in memory index (`store/vector_store.py`), which is no longer built at startup and no longer
  queried.
- The v1 retrievers (`retrieve/retriever.py`, `retrieve/hybrid.py`).
- `strategies/legacy.py` (`LegacyHybridStrategy`), replaced by `TraditionalRAGStrategy` behind the same
  `RetrieverStrategy` interface.

### Fixed
- Vector and lexical rows are now deleted when a document is deleted (directly, by an admin, or through
  a collection delete), rather than surviving as orphans; SQLite in particular never enforced the
  declared foreign key cascade.
- Denormalised `collection_id` on `chunks`, `chunk_embeddings` and `chunk_search` is updated inside the
  same transaction as a document move, so the SQL-side access filter and the SQL-side index never
  disagree about which collection a document belongs to. A Chroma vector store's own metadata copy of
  `collection_id` is not part of that transaction and is not updated by a move; on a
  `vector_store.kind: chroma` deployment the move endpoint refuses the operation (409) instead.
- The index fan out (vector store write plus lexical store write) is now atomic across both stores
  instead of two independent commits that could disagree after a crash between them.
- The collection grant upsert is now atomic.
- SQLite now enforces foreign keys (`PRAGMA foreign_keys=ON`) so declared `ON DELETE CASCADE`
  constraints actually apply in tests and in a SQLite deployment, rather than only on PostgreSQL.
- Reranker scores are guarded against `NaN` and infinity and clamped before being stored; a malformed
  or out of range score degrades to retrieval order instead of corrupting the ranking.
- The LLM reranker caps how many candidates it will ever send in one prompt, because some model
  runtimes silently truncate an over-long prompt and still return a valid-looking, correctly sized
  score array computed against passages the model never actually saw.
- Per request rerankers are now cached by kind instead of rebuilt on every request that overrides
  retrieval parameters.
- The citation contract now verifies short quoted spans that contain a digit, not only spans of eight
  or more characters, because a short quote carrying a number is a factual claim worth checking.
- `llm_calls` is reported as zero when the reranker never runs (an empty candidate list), rather than
  always attributing a call to a configured but unused reranker; the threshold now runs strictly before
  the rerank, since the threshold is expressed against the retrieval score and comparing it to a
  reranker's score would make the configured number meaningless.
- The originals download route now pins its `Content-Type` from the file's own extension and forces a
  download, rather than trusting the caller supplied upload `Content-Type`.

### Notes
- Retrieval quality, including whether an answer is correct, complete or faithful to its sources, is
  not measured this phase. The citation contract proves a marker points at a retrieved chunk and a
  quote is real; it proves nothing about whether a paraphrase is faithful. Phase 8 measures answer
  quality with a question set and a scoring harness.
- A real end to end run against a local `llama3.2:3b` model produced a correct, grounded, cited answer,
  but the model's first streamed attempt omitted the citation marker entirely; the citation contract
  caught it and the run was transparently corrected, emitting a `superseded` event. This is recorded as
  a real characteristic of the small model path, not hidden.
- A single deployment serves one embedding model and one vector dimension at a time (ADR 0006).
  Comparing embedding models is an evaluation concern and belongs to Phase 8.
- Retrieval is limited to a single collection per request (`collection_ids[0]`). Multi collection
  metadata filters did **not** ship in Phase 4: `AccessFilter.collection_ids` takes a list in core,
  but `SearchRequest` and `AskRequest` still expose a single `collection_id`, so the API surface is
  unchanged. Carried forward.
- Deleting a user does not delete what they produced. Group memberships and per-user document
  overrides cascade away; their API keys are deactivated and detached rather than deleted, so the
  keys authenticate nobody while their audit trail survives; audit log, retrieval run, conversation
  and query log rows are preserved and anonymised; collections and documents they owned are preserved
  and disowned for an admin to reassign. An admin cannot delete their own account, so the last
  administrator cannot lock everyone out. Deleting a group removes its memberships and grants and
  never touches the users themselves.
- Saving provider configuration from the console rewrites `ragfabric.yaml` as data, which does not
  preserve comments in that file. Providers are built once at application startup, so a save reports
  `restart_required` rather than implying running workers picked the change up.
- A collection with no grants is open to every signed in user; the first grant restricts it to its
  grantees.

## [1.0.0 of the v1 assistant] - 2026-08-16

The single strategy assistant that this project grows from. Kept on `main` and described in the README
under "What works today".

### Added
- Ingestion for PDF, DOCX, PPTX, TXT, CSV with page numbers and character spans
- Offline hashing embedder, in memory cosine index, semantic and hybrid retrieval
- Extractive answers with numbered citations and relevance floors
- JWT auth, two role RBAC, bootstrap admin, production safety rails
- Alembic migrations with drift test, CI against real PostgreSQL
- Angular app: login, dashboard, upload, ask, collections, analytics, admin
- 107 backend tests, 20 frontend tests
