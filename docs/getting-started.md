# Getting started

> Status: the compose profiles, the CLI and `ragfabric.yaml` exist since Phase 1. Traditional and
> Vectorless RAG arrive in Phases 3 and 4.

## Prerequisites

| Need | Version | Notes |
|---|---|---|
| Docker and Docker Compose | Docker 24+, Compose v2 | Runs PostgreSQL, Redis, and in the full profile Chroma and Neo4j |
| Python | 3.13 | `uv` is the recommended tool: `uv sync` |
| Node | 24 | For the Angular apps |
| An LLM provider | any of OpenAI, Anthropic, Ollama | Ollama needs no key; quality depends on the local model |

## Run RagFabric

The compose stack runs the v1 assistant today.

```bash
git clone https://github.com/ranjan-del/ragfabric.git
cd ragfabric
cp .env.example .env && cp ragfabric.example.yaml ragfabric.yaml
docker compose up --build                  # lite: PostgreSQL with pgvector, Redis, API, UI
docker compose --profile full up --build   # adds Chroma and Neo4j
docker compose --profile workers up --build # adds the worker, needed for ingestion.indexing: queue
```

Backend on `http://localhost:8000` (OpenAPI at `/docs`), frontend on `http://localhost:4200`. Sign in with
the bootstrap admin from `.env.example`, upload a PDF, ask a question, and inspect the citations.
No API key is needed: the v1 path uses the offline hashing embedder and extractive answers.

Without Docker:

```bash
uv sync
uv run ragfabric db upgrade
uv run ragfabric serve --reload
cd apps/assistant && npm ci && npm start
```

The CLI that exists today:

```bash
ragfabric version
ragfabric config validate
ragfabric db upgrade
ragfabric serve
```

`ragfabric ingest`, `users` and `keys` are Phase 2. `ragfabric ask` is Phase 3, once the Traditional
strategy ships.

## What to read next

- [architecture.md](architecture.md) for how the pieces fit
- One strategy document to understand what retrieval actually does: start with [traditional-rag.md](traditional-rag.md)
- [configuration.md](configuration.md) to swap providers and stores
