# Configuration

> Status: implemented in Phase 1; strategy blocks are read by the phases that ship them.

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
  provider: openai            # openai | anthropic | ollama | offline
  model: gpt-5.4-mini         # anthropic default: claude-sonnet-5, ollama default: llama3.2
  base_url: null              # ollama or any OpenAI compatible endpoint, e.g. http://localhost:11434/v1

embeddings:
  provider: openai            # openai | ollama | offline
  model: text-embedding-3-small
  dim: null                   # inferred for known models

reranker:
  kind: none                  # none | llm | cross_encoder

vector_store:
  kind: pgvector              # pgvector (lite profile) | chroma (full profile) | memory (tests)
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

strategies:
  traditional: { top_k: 8, similarity_threshold: 0.25, rerank: none, max_context_tokens: 6000 }
  vectorless:  { top_k: 8, phrase_boost: 2.0, identifier_boost: 3.0 }
  agentic:     { max_iterations: 4, max_cost_usd: 0.10, max_latency_ms: 30000 }
  graph:       { max_hops: 2, max_nodes: 200 }

router:
  mode: auto                  # auto | manual
  min_confidence: 0.6
  classifier_model: null      # defaults to llm.model

limits:
  max_upload_mb: 50
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
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Graph store, full profile only |
| `CHROMA_URL` | Vector store when `kind: chroma` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Provider keys, only the ones you use |
| `JWT_SECRET` | At least 32 characters; production refuses placeholders |
| `FIRST_ADMIN_EMAIL`, `FIRST_ADMIN_PASSWORD` | Bootstrap admin; production refuses the shipped defaults |
| `ENVIRONMENT` | `development` or `production` |

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

## Precedence

Environment variables override `ragfabric.yaml`, which overrides defaults. Per request parameters (for
example `top_k` on `/api/ask`) override all three for that request only.
