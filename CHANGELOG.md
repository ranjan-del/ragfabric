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

### Notes
- Queries still run through the v1 in memory index this phase; the pgvector and PostgreSQL full text
  indexes are written by ingestion but not yet queried by the API. Phase 3 switches retrieval over and
  retires the in memory index.
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
