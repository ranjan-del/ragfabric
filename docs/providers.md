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

The interface exposes `complete(messages, tools?, response_schema?) -> Completion` with token counts,
and `stream(...)` for SSE. Token counts come from the provider response, never estimated when the
provider reports them.

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

Changing the embedding model requires re-indexing; the CLI warns and offers `ragfabric reindex`.

## Rerankers (`Reranker`)

The `Reranker` interface is not part of Phase 1; it arrives with Traditional RAG in Phase 3.

| Kind | Status |
|---|---|
| none | Phase 3 |
| LLM rerank | Phase 3 |
| Cross encoder (local), Cohere rerank | later |

## Vector stores (`VectorStore`)

| Store | Status | Notes |
|---|---|---|
| PostgreSQL pgvector | v0.1.0 | Lite profile default, one less service |
| Chroma | v0.1.0 | Full profile |
| Qdrant, Weaviate, Milvus | later | |

All stores accept an access filter and metadata filters and apply them before ranking.

## Lexical stores (`LexicalStore`)

| Store | Status |
|---|---|
| PostgreSQL full text (`tsvector`, GIN) | v0.1.0 |
| In process BM25 (`rank_bm25`), rebuilt from the database | v0.1.0 |
| OpenSearch | later |

## Graph stores (`GraphStore`)

| Store | Status |
|---|---|
| Neo4j 5 | v0.3.0 |
| none | v0.1.0 (Graph RAG disabled) |
| Memgraph, PostgreSQL adjacency tables | later |

## Cache (`Cache`)

Redis (default) or in memory. Used for embedding cache, answer cache, rate limiting and the ingestion
queue.

## Auth (`AuthProvider`)

Local users with JWT and hashed API keys in v0.1.0. OIDC and SAML planned for the production release.

## Connectors (`Connector`)

| Connector | Status |
|---|---|
| Upload | v0.1.0 |
| Watched folder | v1.0.0 |
| Google Drive | v1.0.0 |
| Slack, Jira, Confluence, S3 | later |

## Writing a provider

1. Implement the interface in `packages/core/ragfabric/providers/<name>.py`.
2. Register it in the provider registry with its configuration schema.
3. Add unit tests with recorded responses and one integration test marked `integration`.
4. Document it here with its status.
