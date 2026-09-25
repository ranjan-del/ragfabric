# Providers and stores

> Status: interface matrix for **v0.1.0** onward. "Planned" means designed, not shipped.

Every external dependency sits behind an interface in `packages/core`. An implementation is one module
plus tests; adding one never touches the strategies.

## LLM providers (`LLMProvider`)

| Provider | Status | Notes |
|---|---|---|
| Ollama | Phase 1 (shipped) | **Default** since Phase 3 (`llama3.2:3b`). Local, no key, the free path; served through the OpenAI compatible provider, so `ragfabric[openai]` is still an installed extra for an Ollama-only deployment |
| OpenAI | Phase 1 (shipped) | Also serves any OpenAI compatible endpoint via `base_url`; optional extra `ragfabric[openai]` |
| Anthropic | Phase 1 (shipped) | Claude models; optional extra `ragfabric[anthropic]`. Temperature is not forwarded to Anthropic models: current Claude models reject it while adaptive thinking is active |
| Gemini, Azure OpenAI, Bedrock | later | Contributions welcome once the interface is stable |

The interface exposes `complete(messages, *, model=None, max_tokens=1024, temperature=0.0,
json_schema=None) -> Completion` and, since Phase 3, `stream(messages, *, max_tokens) -> Iterator[str]`
for `POST /api/ask`'s SSE path. There is no `tools` parameter. Token counts come from the provider
response, never estimated when the provider reports them.

### Default models

| Provider | LLM | Embedding |
|---|---|---|
| Ollama | `llama3.2:3b` (default provider and model) | `nomic-embed-text` (default provider and model, 768 dimensions, pinned by migration 0004) |
| OpenAI | `gpt-5.4-mini` | `text-embedding-3-small` |
| Anthropic | `claude-sonnet-5` | (uses OpenAI or Ollama for embeddings) |

## Embedding providers (`EmbeddingProvider`)

| Provider | Status | Notes |
|---|---|---|
| Ollama (`nomic-embed-text` and others) | Phase 1 (shipped) | **Default** since Phase 3. Local |
| OpenAI `text-embedding-3-*` | Phase 1 (shipped) | |
| Offline hashing | Phase 1 (shipped) | Test double from v1, deterministic, no network. Not for production use |
| Voyage, Cohere, sentence transformers | later | |

Changing the embedding model requires re-indexing: edit `embeddings.provider` / `embeddings.model` in
`ragfabric.yaml`, then run `ragfabric reindex`, which re-embeds the whole corpus under the newly active
model. If the new model's dimension differs from the pinned `embeddings.dim` (768 by default), a
migration is needed first; see [configuration.md](configuration.md) and ADR 0006.

## Rerankers (`Reranker`)

The `Reranker` interface shipped with Traditional RAG in Phase 3, along with the strategy that uses it.

| Kind | Status | Notes |
|---|---|---|
| none | Phase 3 (shipped) | Default; no reranking |
| llm | Phase 3 (shipped) | Asks the configured `LLMProvider` to score each candidate; capped candidate count, because some model runtimes silently truncate an over-long prompt and still return a valid-looking, correctly sized score array computed against passages the model never actually saw |
| cross_encoder | Phase 3 (shipped, extra `ragfabric[rerank]`) | Reads the question and one passage together and scores the pair directly; needs `sentence-transformers`, loaded lazily on the first rerank call, not at process start |
| Cohere rerank | later | |

## Vector stores (`VectorStore`)

Interface: Phase 1 (shipped).

| Store | Status | Notes |
|---|---|---|
| PostgreSQL pgvector (`PgVectorStore`) | Phase 3 (shipped, queried) | Cosine distance on a fixed 768 dimension column with an HNSW index (migration 0004); a NumPy fallback on SQLite. Lite profile default, one less service |
| Chroma | Phase 3 (shipped, queried) | Full profile; `vector_store.kind: chroma` and `CHROMA_URL` |
| Qdrant, Weaviate, Milvus | later | |

All stores accept an access filter and metadata filters and apply them before ranking, with the filter
passed **inside** the store query, not applied afterward (ADR 0003); the pgvector, Chroma and PostgreSQL
full text stores apply the filter inside the query. Every vector query also filters on the active
embedding model name, so a row written by a previous model can never enter a ranking even if it is
still present in the table (ADR 0006). The `memory` vector store kind in `ragfabric.yaml` maps to
`PgVectorStore` for now, since its SQLite fallback path covers the no-Postgres case; a true in memory
implementation is not planned separately.

## Lexical stores (`LexicalStore`)

Interface: Phase 1 (shipped).

| Store | Status | Notes |
|---|---|---|
| PostgreSQL full text (`PostgresLexicalStore`: `tsvector`, `plainto_tsquery`, `ts_rank_cd`, GIN index) | Phase 2 (shipped, fed by ingestion; queried by `POST /api/search/hybrid` since Phase 3) | Token overlap fallback on SQLite. Fusion, exact phrase and identifier boosting for a dedicated Vectorless strategy are Phase 4 |
| In process BM25 (`rank_bm25`), rebuilt from the database | v0.1.0 (planned, Phase 4) | |
| OpenSearch | later | |

## Graph store

There is no separate graph store interface. A `GraphStore` protocol was added in Phase 1 for a
planned Neo4j implementation; Phase 6 removed it, because nothing ever implemented it and the graph
that shipped is written directly against the relational schema it shares with every other store, not
behind a swappable interface ([ADR 0011](adr/0011-postgres-recursive-cte-over-neo4j.md)).

| Backend | Status |
|---|---|
| PostgreSQL, recursive CTEs (`graph/traverse.py`), behind `graph_store.kind: postgres` | v0.3.0 (Phase 6, shipped on the phase branch) |
| disabled (`graph_store.enabled: false`, the default) | v0.1.0 onward (Graph RAG makes no call and needs no provider) |

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
