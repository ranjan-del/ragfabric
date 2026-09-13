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
