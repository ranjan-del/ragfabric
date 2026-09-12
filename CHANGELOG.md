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

### Changed
- Relicensed from MIT to Apache License 2.0 (ADR 0005).
- README rewritten for RagFabric; roadmap, design document and ADRs 0001 to 0005 added.
- Repository governance: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CODEOWNERS, issue and PR templates, Dependabot.
- `main` protected: pull request and green CI required, no force pushes, linear history.

### Added
- `docs/` with getting started, architecture, one document per retrieval strategy, routing, evaluation,
  configuration, providers and troubleshooting. Documents for unshipped features are marked as design
  documents and say which release delivers them.

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
