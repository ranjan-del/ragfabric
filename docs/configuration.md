# Configuration

> Status: the file, its validation and `ragfabric config validate` shipped in Phase 1. As of Phase 3
> every top level key, including `llm`, `reranker`, `strategies` and `limits`, is read through
> `ragfabric_core.runtime.get_config` and takes effect.

## Principles

- One file, `ragfabric.yaml`, for everything that is not a secret.
- Secrets only in environment variables, never in the file, never in the repository.
- Every setting has a safe default so a copy of `ragfabric.example.yaml` produces a working configuration.
- `ragfabric config validate` reports the active implementation for each interface and fails on
  placeholders in production.

## ragfabric.yaml

`ragfabric.example.yaml`, copied to `ragfabric.yaml` and edited:

```yaml
# ragfabric.example.yaml
# Copy to ragfabric.yaml and edit. Every key is optional; defaults are shown.
# Secrets never go here: OPENAI_API_KEY, ANTHROPIC_API_KEY, DATABASE_URL, JWT_SECRET,
# NEO4J_PASSWORD come from the environment (.env.example lists them).

llm:
  provider: ollama            # openai | anthropic | ollama | offline
  model: llama3.2:3b          # openai default: gpt-5.4-mini, anthropic default: claude-sonnet-5
  base_url: http://localhost:11434/v1   # ollama or any OpenAI compatible endpoint

embeddings:
  provider: ollama            # openai | ollama | offline
  model: nomic-embed-text     # 768 dimensions, pinned by migration 0004
  dim: 768                    # must match embeddings.dim in the active migration; see "Changing the
                               # embedding model" below before changing this
  base_url: http://localhost:11434/v1

reranker:
  kind: none                  # none | llm | cross_encoder; cross_encoder needs the ragfabric[rerank] extra
                               # (pulls in sentence-transformers and torch)

vector_store:
  kind: pgvector               # pgvector (lite profile) | chroma (full profile, needs CHROMA_URL) | memory (tests)
lexical_store:
  kind: postgres_fts          # postgres_fts | bm25
graph_store:
  kind: neo4j
  enabled: false              # true needs the full compose profile
cache:
  kind: redis                 # redis | memory

ingestion:
  chunk_size: 600
  chunk_overlap: 80
  retain_originals: true
  uploads_dir: data/uploads    # where retained originals are stored (a volume in compose)
  indexing: inline             # inline | queue (queue needs Redis and `ragfabric worker`)

strategies:
  traditional: { top_k: 8, similarity_threshold: 0.25, rerank: none, max_context_tokens: 6000 }
  vectorless:  { top_k: 8, phrase_boost: 2.0, identifier_boost: 3.0 }
  agentic:
    max_iterations: 4          # retrieve, assess and repair passes before the loop stops
    max_llm_calls: 12          # model calls for the whole run, across every node
    per_node_llm_calls:        # per node caps, so one runaway node cannot spend the lot
      plan: 2
      assess: 6
      repair: 6
      generate: 2
    max_cost_usd: 0.10         # spend ceiling for one run, in US dollars
    max_latency_ms: 30000      # wall clock ceiling for one run, in milliseconds
    tools: [semantic_search, lexical_search, fetch_document]   # which tools the planner may choose
    assess_strictness: strict   # strict: answerable from the retrieved text. lenient: topical is enough
  graph:       { max_hops: 2, max_nodes: 200 }

router:
  mode: auto                  # auto | manual
  min_confidence: 0.6
  classifier_model: null      # defaults to llm.model

limits:                       # enforced, not just read: upload rejects an over-size or wrong-type file,
  max_upload_mb: 50            # the rate limiter uses rate_limit_per_minute as its window
  allowed_types: [pdf, docx, pptx, txt, csv, md]
  rate_limit_per_minute: 60

telemetry:
  otlp_endpoint: null         # off by default; e.g. http://otel-collector:4318
```

## Environment variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `RAGFABRIC_CONFIG` | Path to `ragfabric.yaml`; the api container mounts it at `/app/ragfabric.yaml` |
| `RAGFABRIC_TEST_DATABASE_URL` | PostgreSQL connection string used only by the `PgVectorStore` and `PostgresLexicalStore` integration tests; unset, those tests skip |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Graph store, full profile only |
| `CHROMA_URL` | Vector store when `kind: chroma`. Note: `docker-compose.yml` publishes Chroma on host port 8001 (`127.0.0.1:8001:8000`), matching `.env.example`, not 8000 |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Provider keys, only the ones you use |
| `JWT_SECRET` | At least 32 characters; production refuses placeholders |
| `FIRST_ADMIN_EMAIL`, `FIRST_ADMIN_PASSWORD` | Bootstrap admin; production refuses the shipped defaults |
| `ENVIRONMENT` | `development` or `production` |

## Optional dependencies (extras)

Installed with `uv sync --extra <name>` or `pip install 'ragfabric[<name>]'`.

| Extra | Needed for | Notes |
|---|---|---|
| `openai` | `llm.provider: openai` or `ollama`, `embeddings.provider: openai` or `ollama` | **Ollama is served through the same OpenAI compatible client** (`providers/openai_compat.py`), because Ollama exposes an OpenAI compatible chat and embeddings API at `/v1`. An Ollama-only deployment, including the shipped default configuration, still needs this extra installed; only a deployment using none of OpenAI, Azure, vLLM, LM Studio or Ollama can skip it. Forgetting it fails at process start with `ProviderError: ollama: openai is not installed` |
| `anthropic` | `llm.provider: anthropic` | The `anthropic` SDK |
| `chroma` | `vector_store.kind: chroma` | The `chromadb` client |
| `rerank` | `reranker.kind: cross_encoder` | `sentence-transformers`, which pulls in `torch`; the model itself is downloaded lazily, on the first rerank call, not at process start |

## Changing the embedding model

Changing `embeddings.model` (or `embeddings.provider`) does not re-embed anything by itself; the stored
vectors were written by the old model and a query embedded with the new model would be compared
against them meaninglessly.

1. Edit `embeddings.provider` / `embeddings.model` in `ragfabric.yaml`.
2. Run `ragfabric reindex`, which re-embeds every chunk under the newly active model. Every vector query
   already filters on the active model name, so nothing is served from the old model's vectors while
   the reindex runs; they are simply ignored until `reindex` overwrites or supersedes them.
3. If the new model's dimension differs from `embeddings.dim` (768, pinned by migration 0004 for the
   default `nomic-embed-text`), step 2 alone is not enough: the `chunk_embeddings.embedding` column and
   its HNSW index are fixed at a dimension by the migration, and pgvector cannot store a vector of a
   different length in that column. A new migration that changes the column's declared dimension (and
   rebuilds the HNSW index) is required first. See ADR 0006 for why one deployment pins one model and
   one dimension rather than serving two at once.

## Pricing

`pricing.yaml` holds per model input, output and embedding prices per million tokens. Cost shown anywhere
in the system is `input_tokens * input_price + output_tokens * output_price` plus embedding cost, labelled
"estimate from configured pricing". Update the file when prices change; nothing is hard coded. Every entry
in `pricing.yaml` carries `as_of` and `source`; a model without an entry reports `known=false` and no
number.

## Compose profiles

| Profile | Services | Strategies available |
|---|---|---|
| `lite` | postgres (pgvector), redis, api, ui | Traditional, Vectorless, Agentic |
| `full` | lite plus chroma, neo4j | all four |
| `workers` | lite plus worker | same as lite; needed when `ingestion.indexing: queue` |

`workers` is the only profile that starts the worker; `full` adds chroma and neo4j but does not imply
`workers`. The `worker` service runs `ragfabric worker`, draining the Redis queue that
`ingestion.indexing: queue` schedules ingestion jobs onto, so it requires that setting. In `inline`
mode (the default) no worker is needed; the API indexes a document as part of the ingest call, and
starting the worker against an inline config gives it nothing to drain.

## Exposing the stack

`docker-compose.yml` publishes every port on loopback (`127.0.0.1`) by default, so the stack is
reachable only from the machine it runs on. To expose it on a network:

1. Change the bind address on the ports you need in `docker-compose.yml` (for example
   `"0.0.0.0:8000:8000"`).
2. Set `ENVIRONMENT=production` in `.env`.
3. Set a real `JWT_SECRET` and your own `FIRST_ADMIN_EMAIL` / `FIRST_ADMIN_PASSWORD` in `.env`,
   because the development defaults seed `admin@example.com`.

## Precedence

Environment variables override `ragfabric.yaml`, which overrides defaults. Per request parameters (for
example `top_k` on `/api/ask`) override all three for that request only.
