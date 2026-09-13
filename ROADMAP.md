# Roadmap

RagFabric is built in ten phases grouped into six releases. Each phase is a GitHub issue under a
milestone; this file is the human readable summary and is updated when a phase closes.

| Release | Theme | Phases |
|---|---|---|
| v0.1.0 | Initial RAG engine | 1, 2, 3, 4 |
| v0.2.0 | Agentic retrieval | 5 |
| v0.3.0 | Graph retrieval | 6 |
| v0.4.0 | Adaptive router | 7 |
| v0.5.0 | Evaluation framework | 8 |
| v1.0.0 | Production release | 9, 10 |

Legend: `[x]` merged to main, `[~]` in progress, `[ ]` not started.

## What exists today (inherited v1, all on `main`)

- [x] Ingestion for PDF, DOCX, PPTX, TXT, CSV with page numbers and character spans surviving into citations
- [x] Chunking with overlap
- [x] Offline hashing embedder and in-memory cosine index (kept as the no-key test double)
- [x] Semantic and hybrid retrieval, extractive cited answers with relevance floors
- [x] JWT auth, two-role RBAC, bootstrap admin, production safety rails
- [x] Alembic migrations with drift test, CI against real PostgreSQL
- [x] Angular app: login, dashboard, upload, ask, collections, analytics, admin
- [x] 107 backend tests, 20 frontend tests

## v0.1.0 Initial RAG engine

### Phase 1: Architecture, monorepo, interfaces, Docker, database
- [x] Monorepo layout: `packages/{core,server,cli,sdk-python,sdk-typescript}`, `apps/{console,assistant}`, `deploy/`, `evaluation/`, `docs/`
- [x] `RetrieverStrategy`, `RetrievalResult`, `RetrievedChunk`, `RetrievalContext`
- [x] `LLMProvider`, `EmbeddingProvider`, `Reranker`, `VectorStore`, `LexicalStore`, `GraphStore`, `Cache`, `AuthProvider`, `Connector` interfaces
- [x] Providers: OpenAI, Anthropic, Ollama, offline test double
- [x] `ragfabric.yaml` configuration loader, pricing configuration, cost calculator
- [x] Migration 0002: conversations, messages, retrieval_runs, sources, evaluation_runs, evaluation_results, entities, relationships, groups, grants, api_keys, audit_log
- [x] Compose profiles: `lite` (Postgres with pgvector, Redis, API, UI) and `full` (adds Chroma, Neo4j)
- [x] Python 3.13 via `uv`, Tailwind in the Angular apps

### Phase 2: Shared ingestion and access control foundation
- [ ] Cleaning, section detection, document type metadata, retained originals
- [ ] Configurable `chunk_size` and `chunk_overlap`
- [ ] Index fan-out: vector, lexical, graph queue, as background workers on Redis
- [ ] Groups, collection grants, per-document overrides, scoped API keys, audit log
- [ ] Access filter applied inside every store query
- [ ] CLI: extend with ingest, users, keys (init, version, config validate, db, serve exist since Phase 1)
- [ ] OpenTelemetry spans on every ingestion and retrieval step

### Phase 3: Traditional RAG
- [ ] Embeddings to pgvector or Chroma, top-k, similarity threshold, metadata filter, optional reranking, context budget
- [ ] Numbered citations with a tested citation contract
- [ ] `POST /api/ask` manual mode, SSE streaming
- [ ] `ragfabric ask` and Python SDK

### Phase 4: Vectorless RAG
- [ ] BM25 and PostgreSQL full-text search with fusion, exact phrase and identifier boosting
- [ ] `docs/concepts/lexical-vs-vector.md`
- [ ] Console v1: users, groups, collections, grants, API keys, provider configuration
- [ ] **Release v0.1.0**

## v0.2.0 Agentic retrieval

### Phase 5: Agentic RAG (LangGraph)
- [ ] Typed state; analyze, plan, retrieve with tools, evaluate evidence, rewrite loop, generate, verify
- [ ] Step budget, failure handling, full trace
- [ ] **Release v0.2.0**

## v0.3.0 Graph retrieval

### Phase 6: Graph RAG (Neo4j)
- [ ] Entity and relationship extraction with source chunk links, entity resolution
- [ ] Neo4j schema and upserts, mirrored entities and relationships in Postgres
- [ ] Entity match, k-hop traversal, source chunk retrieval, multi-hop questions
- [ ] **Release v0.3.0**

## v0.4.0 Adaptive router

### Phase 7: Query Router
- [ ] `RouterDecision` with confidence and user-safe reasoning
- [ ] Feature signals plus classifier, AUTO and MANUAL modes
- [ ] Fallback chain recorded on every run
- [ ] **Release v0.4.0**

## v0.5.0 Evaluation framework

### Phase 8: Evaluation
- [ ] Shipped synthetic corpus and `questions.json` in eight categories, bring-your-own questions
- [ ] Retrieval metrics: precision, recall, hit rate, MRR
- [ ] Generation metrics: correctness, faithfulness, context relevance, citation correctness
- [ ] System metrics: latency split, calls, tokens, cost. Complexity score documented as an engineering assessment
- [ ] `make eval` persists runs and regenerates `docs/benchmarks/latest.md`
- [ ] Console dashboards: latency percentiles, cost per day, fallback rate, quality trend
- [ ] **Release v0.5.0**

## v1.0.0 Production release

### Phase 9: Assistant UI, TypeScript SDK
- [ ] Ask with router card and clickable citations, Compare (four strategies side by side), Trace, Evaluation pages
- [ ] `@ragfabric/sdk` generated from the OpenAPI spec

### Phase 10: Hardening, docs, deployment
- [ ] End-to-end tests, rate limiting, file validation, structured logging
- [ ] Connectors: watched folder, Google Drive
- [ ] Docs site, deployment guides, release automation (repository already renamed to `ragfabric`)
- [ ] **Release v1.0.0**, first production deployment

## Later
- OIDC and SAML, multi-tenant workspaces, more connectors (Slack, Jira, Confluence, S3), more stores
  (Qdrant, Weaviate, OpenSearch), community summaries for Graph RAG, Helm chart.
