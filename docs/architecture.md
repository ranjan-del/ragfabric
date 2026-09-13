# Architecture

> Status: design approved 2026-09-13, implemented progressively from v0.1.0. The v1 code on `main`
> maps onto this design as described in the migration section.

## Layers

```mermaid
flowchart LR
    subgraph apps
        A1[assistant UI]
        A2[console UI]
    end
    subgraph clients
        C1[CLI]
        C2[Python SDK]
        C3[TypeScript SDK]
    end
    apps --> C3
    C1 & C2 & C3 --> S[server: FastAPI]
    S --> CORE[core: engine]
    CORE --> I[interfaces]
    I --> P[providers: OpenAI, Anthropic, Ollama]
    I --> ST[stores: pgvector, Chroma, Postgres FTS, Neo4j, Redis]
```

Dependency direction is one way. `core` never imports from `server`; `server` never imports from an
app; apps never call the API without the SDK. CI enforces this with import linting.

## Packages

| Package | Responsibility | Depends on |
|---|---|---|
| `packages/core` | Strategies, router, ingestion, evaluation, access policy, tracing hooks, all interfaces | nothing in the repo |
| `packages/server` | FastAPI app: auth, REST, SSE streaming, admin endpoints, trace and metrics endpoints | core |
| `packages/cli` | `ragfabric` command | core, and the server over HTTP when running |
| `packages/sdk-python` | Typed client generated from the OpenAPI spec | nothing |
| `packages/sdk-typescript` | Same, published as `@ragfabric/sdk` | nothing |
| `apps/console` | Admin: users, groups, collections, grants, API keys, providers, dashboards | sdk-typescript |
| `apps/assistant` | Reference end user UI: Ask, Compare, Trace, Sources | sdk-typescript |

## Request flow

```mermaid
sequenceDiagram
    participant U as Client
    participant S as server
    participant R as Router
    participant St as Strategy
    participant Store as Stores
    participant L as LLMProvider
    U->>S: POST /api/ask {question, mode, strategy?}
    S->>S: authenticate, resolve principal, compute permitted documents
    S->>R: route(question) if mode=auto
    R-->>S: RouterDecision
    S->>St: retrieve(question, RetrievalContext{principal, filter, params, budget})
    St->>Store: query with access filter
    Store-->>St: permitted chunks
    St-->>S: RetrievalResult
    S->>L: generate(prompt with numbered passages)
    L-->>S: answer with [n] markers
    S->>S: citation check, metrics, trace, audit log
    S-->>U: answer, sources, metrics, router decision, trace id (SSE streamed)
```

## Core types

```python
class RetrieverStrategy(Protocol):
    name: StrategyName
    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult: ...

class RetrievalContext(BaseModel):
    principal: Principal
    access_filter: AccessFilter          # permitted document ids, computed once per request
    collection_ids: list[str] | None
    params: StrategyParams               # top_k, similarity_threshold, ...
    budget: Budget                       # max llm calls, max latency, max cost

class RetrievalResult(BaseModel):
    strategy: StrategyName
    chunks: list[RetrievedChunk]
    retrieval_calls: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    trace: list[TraceSpan]
    fallback_from: StrategyName | None = None
```

Generation, citation checking, metrics and evaluation never inspect `strategy`. That is what makes the
four strategies comparable.

## Data model

| Table | Purpose |
|---|---|
| users, groups, group_members | Identity and grouping |
| collections, collection_grants, document_overrides | Access control |
| api_keys | Hashed keys with scopes and rate limits |
| documents, document_chunks | Content with page, section, span, document_type |
| entities, relationships | Mirror of the graph for the console; Neo4j is the query engine |
| conversations, messages | Chat history |
| retrieval_runs, sources | One row per ask with metrics; sources returned and sources filtered |
| evaluation_runs, evaluation_results | Benchmark runs and per question results |
| audit_log | Who asked what, which strategy, what was filtered |

Vector data lives in pgvector or Chroma, lexical data in a `tsvector` column or an in process BM25
index rebuilt from `document_chunks`, graph data in Neo4j.

## Ingestion

```
file -> parser -> cleaning -> metadata -> chunker -> document_chunks
     -> fan out (Redis queue, background workers): embed+vector index | lexical index | graph extraction
```

Workers are stateless and horizontally scalable. The API never blocks on indexing.

## Observability

One trace per request. Spans: router, each retrieval call, each LLM call, generation, citation check,
each agent node, each graph traversal. Spans carry latency; LLM spans carry tokens and cost. Stored in
PostgreSQL for the built in Trace page and dashboards; exported over OTLP when configured.

## Migration from the v1 layout

The move from the v1 `backend/` and `frontend/` layout to the monorepo happened in Phase 1. The v1
ingestion, retrieval, generation and store code now lives in `packages/core`; the v1 API, security and
dependency wiring in `packages/server`; the v1 Angular app in `apps/assistant`. The hashing embedder and
extractive generator are kept as offline test doubles. `packages/core/src/ragfabric_core/` today:

```
ragfabric_core/
  auth/            principal.py (Principal, AccessFilter), base.py (AuthProvider)
  connectors/      base.py (SourceDocument, Connector)
  db/              migrate.py, session.py
  generate/        answer.py, llm.py
  ingest/          chunk.py, embed.py, parser.py, pipeline.py
  migrations/      alembic.ini, env.py, script.py.mako, versions/0001_initial_schema.py, versions/0002_platform_tables.py
  models/          base.py, user.py, document.py, access.py, runs.py, evaluation.py, graph.py
  providers/       base.py, offline.py, openai_compat.py, anthropic_provider.py, registry.py
  retrieve/        hybrid.py, retriever.py (v1 pipeline, unchanged)
  store/           vector_store.py (v1 store, unchanged)
  stores/          base.py (VectorStore, LexicalStore, GraphStore, Cache)
  strategies/      base.py (StrategyName, RetrievedChunk, TraceSpan, StrategyParams, Budget,
                   RetrievalContext, RetrievalResult, RetrieverStrategy, StrategyRegistry),
                   contract.py, legacy.py (LegacyHybridStrategy)
  testing/         fixtures.py
  config.py        environment Settings
  config_file.py   ragfabric.yaml loader, strict validation
  pricing.py       pricing.yaml    dated, sourced cost data
  security.py
  __init__.py
```
