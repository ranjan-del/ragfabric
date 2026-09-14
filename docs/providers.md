# Providers and stores

> Status: interface matrix for **v0.1.0** onward. "Planned" means designed, not shipped.

Every external dependency sits behind an interface in `packages/core`. An implementation is one module
plus tests; adding one never touches the strategies.

## LLM providers (`LLMProvider`)

| Provider | Status | Notes |
|---|---|---|
| OpenAI | Phase 1 (shipped) | Default. Also serves any OpenAI compatible endpoint via `base_url` |
| Anthropic | Phase 1 (shipped) | Claude models. Temperature is not forwarded to Anthropic models in Phase 1: current Claude models reject it while adaptive thinking is active |
| Ollama | Phase 1 (shipped) | Local models, no key, the free trial path; served through the OpenAI compatible provider |
| Gemini, Azure OpenAI, Bedrock | later | Contributions welcome once the interface is stable |

The interface exposes `complete(messages, *, model=None, max_tokens=1024, temperature=0.0,
json_schema=None) -> Completion`. There is no `tools` parameter and no `stream(...)` method in
Phase 1; streaming arrives in Phase 3. Token counts come from the provider response, never
estimated when the provider reports them.

### Default models

| Provider | LLM | Embedding |
|---|---|---|
| OpenAI | `gpt-5.4-mini` | `text-embedding-3-small` |
| Anthropic | `claude-sonnet-5` | (uses OpenAI or Ollama for embeddings) |
| Ollama | `llama3.2` | `nomic-embed-text` |

## Embedding providers (`EmbeddingProvider`)

| Provider | Status | Notes |
|---|---|---|
| OpenAI `text-embedding-3-*` | Phase 1 (shipped) | Default |
| Ollama (`nomic-embed-text` and others) | Phase 1 (shipped) | Local |
| Offline hashing | Phase 1 (shipped) | Test double from v1, deterministic, no network. Not for production use |
| Voyage, Cohere, sentence transformers | later | |

Changing the embedding model requires re-indexing; the CLI `ingest` command (Phase 2) re-runs the
ingestion pipeline, including embedding, over a path.

## Rerankers (`Reranker`)

The `Reranker` interface is not part of Phase 1; it arrives with Traditional RAG in Phase 3.

| Kind | Status |
|---|---|
| none | Phase 3 |
| LLM rerank | Phase 3 |
| Cross encoder (local), Cohere rerank | later |

## Vector stores (`VectorStore`)

Interface: Phase 1 (shipped).

| Store | Status | Notes |
|---|---|---|
| PostgreSQL pgvector (`PgVectorStore`) | Phase 2 (shipped, fed by ingestion; queried from Phase 3) | Cosine distance; a NumPy fallback on SQLite. Lite profile default, one less service |
| Chroma | v0.1.0 (planned, Phase 3) | Full profile |
| Qdrant, Weaviate, Milvus | later | |

All stores accept an access filter and metadata filters and apply them before ranking; the pgvector and
PostgreSQL full text stores apply the filter inside the SQL query via `stores/access_sql.access_clause`.
The `memory` vector store kind in `ragfabric.yaml` maps to `PgVectorStore` for now, since its SQLite
fallback path covers the no-Postgres case; a true in memory implementation is not planned separately.

## Lexical stores (`LexicalStore`)

Interface: Phase 1 (shipped).

| Store | Status | Notes |
|---|---|---|
| PostgreSQL full text (`PostgresLexicalStore`: `tsvector`, `plainto_tsquery`, `ts_rank_cd`, GIN index) | Phase 2 (shipped, fed by ingestion; queried from Phase 3) | Token overlap fallback on SQLite |
| In process BM25 (`rank_bm25`), rebuilt from the database | v0.1.0 (planned, Phase 4) | |
| OpenSearch | later | |

## Graph stores (`GraphStore`)

Interface: Phase 1 (shipped).

| Store | Status |
|---|---|
| Neo4j 5 | v0.3.0 (planned, Phase 6) |
| none | v0.1.0 (Graph RAG disabled) |
| Memgraph, PostgreSQL adjacency tables | later |

## Cache (`Cache`)

Interface: Phase 1 (shipped).

| Store | Status |
|---|---|
| Redis (`RedisCache`) | Phase 2 (shipped) |
| In memory (`MemoryCache`) | Phase 2 (shipped) |

Used for the rate limiter (`auth/ratelimit.check_rate_limit`, a fixed one minute window) today, and for
an embedding cache and answer cache later. The Redis backed job queue (`queue/redis_queue.py`) is a
separate `JobQueue` interface, not `Cache`.

## Auth (`AuthProvider`)

Interface: Phase 1 (shipped).

| Kind | Status |
|---|---|
| Local users with JWT | Phase 1 (shipped, v1 auth) |
| API keys (`auth/api_keys.py`) | Phase 2 (shipped) |

OIDC and SAML planned for the production release.

## Connectors (`Connector`)

Interface: Phase 1 (shipped).

| Connector | Status |
|---|---|
| Upload | Phase 1 (shipped, v1 upload) |
| Watched folder | v1.0.0 |
| Google Drive | v1.0.0 |
| Slack, Jira, Confluence, S3 | later |

## Writing a provider

1. Implement the interface in `packages/core/src/ragfabric_core/providers/<name>.py`.
2. Register it in the provider registry with its configuration schema.
3. Add unit tests with recorded responses and one integration test marked `integration`.
4. Document it here with its status.
