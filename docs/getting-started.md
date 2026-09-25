# Getting started

> Status: the compose profiles, the CLI and `ragfabric.yaml` exist since Phase 1. Ingestion and access
> control shipped in Phase 2. Traditional RAG (real embeddings, vector search, cited answers, `ragfabric
> ask`) shipped in Phase 3. Vectorless RAG arrives in Phase 4.

## Prerequisites

| Need | Version | Notes |
|---|---|---|
| Docker and Docker Compose | Docker 24+, Compose v2 | Runs PostgreSQL, Redis, and in the full profile Chroma. The knowledge graph lives in PostgreSQL too (ADR 0011), no extra service needed |
| Python | 3.13 | `uv` is the recommended tool: `uv sync` |
| Node | 24 | For the Angular apps |
| Ollama | any recent build | The shipped default, no key needed. Install it separately; RagFabric only calls its API |

No paid account is required for the default configuration. Ollama serves both the LLM and the
embedding model locally, over an OpenAI compatible API, which is why `ragfabric[openai]` is still an
installed extra even for an Ollama only setup; see [configuration.md](configuration.md).

## Pull the default models

```bash
ollama pull nomic-embed-text   # embeddings, 768 dimensions, pinned by migration 0004
ollama pull llama3.2:3b        # the default local LLM
```

## Run RagFabric

```bash
git clone https://github.com/ranjan-del/ragfabric.git
cd ragfabric
cp .env.example .env && cp ragfabric.example.yaml ragfabric.yaml
docker compose --profile lite up -d postgres redis   # PostgreSQL with pgvector, Redis
docker compose --profile full up --build             # adds Chroma, and the API and UI
```

Backend on `http://localhost:8000` (OpenAPI at `/docs`), frontend on `http://localhost:4200`.

## The real flow, from an empty database

This is the exact sequence that was run and captured end to end against a fresh database, no API key,
no paid account, using the CLI directly against the containers above rather than the full compose
build (the containerized `api`/`worker` images currently fail to start under the shipped Ollama
default because they omit the `openai` extra; see [troubleshooting.md](troubleshooting.md)).

```bash
docker compose --profile lite up -d postgres redis
export DATABASE_URL="postgresql+psycopg://ragfabric:ragfabric@127.0.0.1:5432/ragfabric"
uv run ragfabric db upgrade
uv run ragfabric users create --email admin@example.com --password 'ChangeMe123!' --role admin
mkdir -p /tmp/rf-demo
printf 'Employees receive 24 days of annual leave per year. Leave accrues monthly and unused days expire in March.\n' > /tmp/rf-demo/handbook.txt
uv run ragfabric ingest /tmp/rf-demo --collection handbook
uv run ragfabric serve --host 127.0.0.1 --port 8000 &
```

`ragfabric serve` defaults to `127.0.0.1` only; pass `--host 0.0.0.0` explicitly to bind every
interface. Login is form encoded, not JSON, with a `username` field (the route uses
`OAuth2PasswordRequestForm`):

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/auth/login \
  -H 'content-type: application/x-www-form-urlencoded' \
  -d 'username=admin@example.com&password=ChangeMe123!' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
```

Then ask a question with the real access token:

```bash
uv run ragfabric ask "how much annual leave do employees get" --token "$TOKEN"
```

The real output from that run, verbatim:

```
Employees receive 24 days of annual leave per year.
notice: the streamed answer above failed the citation contract and was corrected; the corrected answer follows.


According to the provided passage [1], employees receive 24 days of annual leave per year.
sources:
  [1] handbook.txt p1
run 1 in 23272ms
```

The `llama3.2:3b` model's first streamed pass carried no `[1]` marker at all; the citation contract
caught it and the CLI printed the corrected, cited answer transparently, as the `superseded` SSE event
is designed to do. This is the real, repeatable behaviour of a small local model on the very first
question this project ever answered, not a rare edge case: the same thing happened again on a second
run of the identical question. For a script or another program, prefer `ragfabric ask --json` or
`--no-stream`, because streamed stdout can otherwise show a stale, superseded draft ahead of the
corrected text.

## First ingest, first user, first key

The real flow, in order: create the configuration, upgrade the database, create an admin, ingest a
folder into a collection, then mint an API key.

```bash
ragfabric init                                                    # writes .env and ragfabric.yaml from the examples
ragfabric db upgrade                                              # applies migrations, including 0004 (pinned dimension, HNSW index)
ragfabric users create --email you@example.com --password ... --role admin
ragfabric ingest ./docs --collection handbook                     # cleans, chunks, embeds, retains the originals, writes both indexes
ragfabric keys create --name ci --user you@example.com            # prints the plaintext key once; only its hash is stored
ragfabric ask "your question" --token "$TOKEN"                    # or --api-key
```

`ragfabric groups create`, `ragfabric grants add` and `ragfabric groups add-member` layer group based
access on top of a collection once it needs to stop being open by default.

## Using Chroma instead of pgvector

Chroma's container port is 8001 on the host (`127.0.0.1:8001:8000` in `docker-compose.yml`), not 8000:

```bash
docker compose --profile full up -d chroma
export CHROMA_URL="http://localhost:8001"
# set vector_store.kind: chroma in ragfabric.yaml, then:
uv run ragfabric reindex --yes
```

`ragfabric reindex` re-embeds every chunk under the active model and reports the count re-embedded;
cross check it against `curl http://localhost:8001/api/v2/.../collections/<id>/count` if you want a
second source. Restart `ragfabric serve` after changing `vector_store.kind`; it reads configuration
once at startup and does not hot reload `ragfabric.yaml`.

## Queued indexing

By default (`ingestion.indexing: inline` in `ragfabric.yaml`) `ragfabric ingest` embeds and indexes a
document before returning. To move that work onto background workers over Redis, set:

```yaml
ingestion:
  indexing: queue
```

and start the worker alongside the rest of the stack:

```bash
docker compose --profile workers up --build
```

or, without Docker, `ragfabric worker`. A queued document moves through `processing`, `indexing`, then
`ready` (or `failed`); `ragfabric worker --once` drains the queue once and exits, which is what CI uses.

## What to read next

- [architecture.md](architecture.md) for how the pieces fit
- One strategy document to understand what retrieval actually does: start with [traditional-rag.md](traditional-rag.md)
- [configuration.md](configuration.md) to swap providers and stores
