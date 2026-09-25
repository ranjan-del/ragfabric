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
- [x] Monorepo layout: `packages/{core,server,cli}`, `apps/assistant`, `deploy/`, `docs/` (the SDK packages, `apps/console` and `evaluation/` are created by the phases that ship them)
- [x] `RetrieverStrategy`, `RetrievalResult`, `RetrievedChunk`, `RetrievalContext`
- [x] `LLMProvider`, `EmbeddingProvider`, `VectorStore`, `LexicalStore`, `Cache`, `AuthProvider`, `Connector` interfaces (the `Reranker` interface arrives with Traditional RAG in Phase 3). A `GraphStore`
      interface was added here for a Neo4j implementation and removed in Phase 6: nothing ever
      implemented it, and the graph that shipped is written directly against the relational schema
      it shares with every other store, not behind a swappable interface (ADR 0011)
- [x] Providers: OpenAI, Anthropic, Ollama, offline test double
- [x] `ragfabric.yaml` configuration loader, pricing configuration, cost calculator
- [x] Migration 0002: conversations, messages, retrieval_runs, sources, evaluation_runs, evaluation_results, entities, relationships, groups, grants, api_keys, audit_log
- [x] Compose profiles: `lite` (Postgres with pgvector, Redis, API, UI) and `full` (adds Chroma).
      A `full` profile Neo4j service was added here and removed in Phase 6, once the graph store
      decision landed on PostgreSQL instead ([ADR 0011](docs/adr/0011-postgres-recursive-cte-over-neo4j.md))
- [x] Python 3.13 via `uv`, Tailwind in the Angular apps

### Phase 2: Shared ingestion and access control foundation
- [x] Cleaning, section detection, document type metadata, retained originals
- [x] Configurable `chunk_size` and `chunk_overlap`
- [x] Index fan-out: vector, lexical, graph queue, as background workers on Redis
- [x] Groups, collection grants, per-document overrides, scoped API keys, audit log
- [x] Access filter applied inside every store query
- [x] CLI: add init, ingest, users, groups, grants, keys, worker (version, config validate, db upgrade/downgrade and serve exist since Phase 1)
- [x] OpenTelemetry spans on every ingestion and retrieval step

### Phase 3: Traditional RAG
- [x] Query the pgvector index populated since Phase 2 and retire the in memory index
- [x] Embeddings to pgvector or Chroma, top-k, similarity threshold, metadata filter, optional reranking, context budget
- [x] Numbered citations with a tested citation contract
- [x] `POST /api/ask` manual mode, SSE streaming
- [x] `ragfabric ask` and Python SDK

### Phase 4: Vectorless RAG
- [x] BM25 and PostgreSQL full-text search with fusion, exact phrase and identifier boosting
- [x] `docs/concepts/lexical-vs-vector.md`
- [x] Console v1: users, groups, collections, grants, API keys, provider configuration
- [ ] Multi collection metadata filters (single collection only since Phase 3). Not shipped in
      Phase 4: `AccessFilter.collection_ids` takes a list in core, but `SearchRequest` and
      `AskRequest` still expose a single `collection_id`, so the API surface is unchanged.
- [ ] **Release v0.1.0**

## v0.2.0 Agentic retrieval

### Phase 5: Agentic RAG (plain Python state machine, not LangGraph)

LangGraph was promised here and is not used. A spike built the identical loop both ways and
measured it: the framework version was longer, added 38 packages, could not carry the model and
the tools through its state, and lost the reason the loop stopped. Recorded as
[ADR 0009](docs/adr/0009-plain-state-machine-over-langgraph.md).

Phase 5 merged to `main` as
[f8bb012](https://github.com/ranjan-del/ragfabric/commit/f8bb012) (#39). The list below was written
while Phase 5 was still on its own branch and said `[x]` meant merged into that branch, not main;
every item that was checked then, plus the four that were still open, are now on `main`, verified
against it directly rather than left as a stale snapshot.

- [x] Typed `AgentState` with the sub-question ledger: per sub-question status, tool, attempt
      history and abandon reason, plus budget accounting that refuses a call rather than
      recording it after the fact
- [x] Validated JSON contracts for every model decision, no native tool calling, a malformed
      response handled as a typed violation instead of an exception
- [x] Three tools over the strategies that already exist: `semantic_search`, `lexical_search`
      and `fetch_document`, the last with the access predicate inside the SQL (ADR 0003)
- [x] Nodes: `plan` (decompose and route in one call), `retrieve` (open sub-questions only,
      evidence pooled by chunk id), `assess` (strict per sub-question rubric), `repair`
- [x] Repair policy with six distinct moves, never repeating a move on an unchanged
      sub-question, abandoning with a reason when the set is exhausted
      ([ADR 0010](docs/adr/0010-repair-policy-and-termination.md))
- [x] Three termination conditions, `resolved`, `budget` and `no_progress`, each assigned at the
      branch that decides it, with a `TraceSpan` per node and real counters (ADR 0004)
- [x] `AgenticRAGStrategy` registered as `agentic`, with typed configuration and per-node call
      caps
- [x] Every branch driven offline by scripted JSON model doubles, with no network and no database
- [x] `docs/concepts/agentic-loops.md`, ADR 0009, ADR 0010, and `docs/agentic-rag.md` rewritten
      to describe what shipped
- [x] Generation over the pooled evidence, reusing the Phase 3 citation contract
      (`generate/cited.py`'s `apply_agentic_contract`), with unsupported claims removed rather
      than retried
- [x] Per sub-question report on `RetrievalResult` (`sub_questions`), replacing a best effort
      boolean
- [x] `agentic` on `POST /api/ask`, `POST /api/search/query`, `ragfabric ask --strategy` and the
      Python SDK
- [x] `strategies.agentic.max_cost_usd`, `max_latency_ms`, `max_llm_calls` and
      `assess_strictness` wired from `AgenticConfig` into the strategy build
      (`strategies/registry_defaults.py`) and read by the loop
- [x] An end to end run against a local model, written down in
      `docs/learning/agentic-first-run.md`
- [ ] **Release v0.2.0** (publishing is gated; `CHANGELOG.md`'s `[0.2.0]` section is still headed
      `unreleased`)

## v0.3.0 Graph retrieval

### Phase 6: Graph RAG (PostgreSQL recursive CTEs, not Neo4j)

Neo4j was named here and is not used. A spike built the identical traversal both ways and measured
it: PostgreSQL could put the access filter inside the walk itself and Neo4j could not, without
duplicating the access model or filtering after the fact, which ADR 0003 forbids. Recorded as
[ADR 0011](docs/adr/0011-postgres-recursive-cte-over-neo4j.md).

Phase 6 has landed on its own branch and is not yet merged to main; `[x]` below means merged into
that branch.

- [x] `entities` and `relationships` gain nullable `confidence` and `extraction_model`;
      `entity_sources` / `relationship_sources` link tables (ruling R1) become the single source of
      truth for provenance and access, replacing the JSON `source_chunk_ids` columns; `entity_merges`
      records every resolution decision (migration 0008)
- [x] Extraction contracts (`graph/contracts.py`): fixed `EntityType` and `RelationType` enums, the
      `INVERSES` direction table, `normalise()`, and the `Subgraph` shape every later task consumes
- [x] Extraction behind a confidence floor, with discarded items counted rather than silently
      dropped ([ADR 0012](docs/adr/0012-extraction-confidence-and-graph-citations.md))
- [x] Incremental extraction: an unchanged chunk is skipped by its content hash, and a changed
      chunk's stale entities and edges are cleaned up before the new text is stored
- [x] Entity resolution in three stages (exact, alias, embedding tie-break), every merge recorded
      with its evidence and reversible; unmerge is globally last-in, first-out (ruling R30)
- [x] Directed, access-checked traversal (`graph/traverse.py`): the access predicate sits inside the
      recursive term, applied to node matching and to edge walking alike, and a relation type absent
      from `INVERSES` is never walked backwards
- [x] Node budget and relation-type guidance on the walk, with truncation reported rather than
      silent
- [x] `GraphRAGStrategy`, registered as `graph`, with the three honest empty cases
      (`no_graph_coverage`, `no_entity_matched`, `no_walkable_edges`)
- [x] The graph citation contract: a relationship claim must cite its edge and a chunk that backs it,
      additive to the unchanged Phase 3 contract
- [x] Graph extraction wired into ingestion behind typed, strict `graph_store` configuration; every
      setting read by something
- [x] `graph` on `POST /api/ask`, `POST /api/search/query`, `ragfabric ask --strategy` and the
      Python SDK; `POST /api/search/hybrid` continues to refuse it; `ragfabric graph merges
      list|show|undo` for inspecting and reversing a merge
- [~] A real extraction run against a local model, with what it actually got right, missed and
      invented written down (Task 13, landing alongside this document)
- [x] `docs/concepts/knowledge-graphs.md`, ADR 0011, ADR 0012, and `docs/graph-rag.md` rewritten to
      describe what shipped; the `GraphStore` protocol removed, nothing having ever implemented it
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
- [ ] Cross encoder reranking measured against llm and none
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
