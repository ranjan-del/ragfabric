# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

Release plan (see [ROADMAP.md](ROADMAP.md) for the phases inside each release):

| Version | Theme | Contents |
|---|---|---|
| v0.1.0 | Initial RAG engine | Monorepo, interfaces, providers, ingestion, access control, Traditional and Vectorless RAG, CLI, Python SDK, console v1 |
| v0.2.0 | Agentic retrieval | LangGraph agentic strategy with budgets and traces |
| v0.3.0 | Graph retrieval | Neo4j knowledge graph build and Graph RAG strategy |
| v0.4.0 | Adaptive router | AUTO and MANUAL modes, RouterDecision, fallbacks |
| v0.5.0 | Evaluation framework | Corpus, question set, metrics, `make eval`, dashboards, generated benchmarks |
| v1.0.0 | Production release | Reference UI with Compare and Trace, TypeScript SDK, connectors, hardening, docs site, deployment guides |

## [Unreleased]

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

### Changed
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
- Retrieval is limited to a single collection per request (`collection_ids[0]`); multi collection
  metadata filters are tracked for Phase 4.
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
