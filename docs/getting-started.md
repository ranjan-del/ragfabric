# Getting started

> Status: the **v1 assistant** path below works today. The **RagFabric** path (compose profiles, CLI,
> `ragfabric.yaml`) ships in v0.1.0 and is described so the shape is agreed before the code lands.

## Prerequisites

| Need | Version | Notes |
|---|---|---|
| Docker and Docker Compose | Docker 24+, Compose v2 | Runs PostgreSQL, Redis, and in the full profile Chroma and Neo4j |
| Python | 3.12 | `uv` is the recommended tool: `uv venv --python 3.12` |
| Node | 20+ | For the Angular apps |
| An LLM provider | any of OpenAI, Anthropic, Ollama | Ollama needs no key; quality depends on the local model |

## Run the v1 assistant today

```bash
git clone https://github.com/ranjan-del/ragfabric.git
cd ragfabric
docker compose up --build
```

Backend on `http://localhost:8000` (OpenAPI at `/docs`), frontend on `http://localhost:4200`. Sign in with
the bootstrap admin from `backend/.env.example`, upload a PDF, ask a question, and inspect the citations.
No API key is needed: the v1 path uses the offline hashing embedder and extractive answers.

Without Docker:

```bash
cd backend && uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
uvicorn app.main:app --reload
cd ../frontend && npm ci && npm start
```

## Run RagFabric (from v0.1.0)

```bash
cp .env.example .env                       # set OPENAI_API_KEY or ANTHROPIC_API_KEY, or configure Ollama
docker compose --profile lite up --build   # PostgreSQL with pgvector, Redis, API, UI
# docker compose --profile full up --build # adds Chroma and Neo4j
```

First steps with the CLI:

```bash
ragfabric init                   # writes ragfabric.yaml, validates providers and stores
ragfabric users create --email you@example.com --role admin
ragfabric ingest ./docs/policies --collection hr-policies
ragfabric ask "How many days of casual leave do interns get?" --strategy traditional
ragfabric ask "Show me policy HR-POL-2026-07" --strategy vectorless
```

Every answer prints its citations and a metrics line: latency, LLM calls, retrieval calls, tokens and
estimated cost.

## What to read next

- [architecture.md](architecture.md) for how the pieces fit
- One strategy document to understand what retrieval actually does: start with [traditional-rag.md](traditional-rag.md)
- [configuration.md](configuration.md) to swap providers and stores
