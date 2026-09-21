# Troubleshooting

This document grows with each release. Entries state the symptom, the likely cause and how to confirm
it before changing anything.

## Installation

| Symptom | Likely cause | Confirm and fix |
|---|---|---|
| `docker compose up` fails on `pgvector` | Image tag or platform mismatch | `docker compose logs postgres`; use the `pgvector/pgvector:pg16` image pinned in the compose file |
| Backend exits with "JWT_SECRET is still a placeholder" | `ENVIRONMENT=production` with the default secret | Set a real secret: `python -c "import secrets; print(secrets.token_urlsafe(64))"` |
| No admin user after first start in production | The shipped bootstrap admin is refused in production by design | Set `FIRST_ADMIN_EMAIL` and `FIRST_ADMIN_PASSWORD` to your own values |
| `ModuleNotFoundError` on Python 3.14 | Dependencies target 3.12 | `uv venv --python 3.12` |
| Neo4j healthcheck never passes | Memory limits or password not set | `docker compose logs neo4j`; set `NEO4J_PASSWORD`; give Docker at least 4 GB |
| The `api` or `worker` container exits at startup with `ProviderError: ollama: openai is not installed. Install it with: uv pip install 'ragfabric[openai]'` | Ollama is the shipped default provider and is served through the OpenAI compatible client, but `deploy/docker/api.Dockerfile`'s `uv sync --frozen --no-dev --all-packages` does not pull in the `openai` extra by default | Known issue with the default containerized build under the shipped configuration, not something you misconfigured; until the Dockerfile is fixed, run the CLI and `ragfabric serve` / `ragfabric worker` directly on the host (which does have the extra in a normal `uv sync`), or rebuild the image with the `openai` extra added explicitly |

## Ingestion

| Symptom | Likely cause | Confirm and fix |
|---|---|---|
| PDF ingests with zero chunks | Scanned PDF without a text layer | Open the file and try to select text; run OCR before upload (OCR is not built in) |
| Citations show page 1 for everything | Parser did not preserve pages for that format | Check `document_chunks.page`; open an issue with the file type |
| Embedding step slow or rate limited | Provider limits | Check the worker logs; embeddings are cached in Redis, re-runs are cheap |
| Re-uploading a document duplicates results | Not the same document id | Use `ragfabric ingest --replace` or the console's re-ingest action |

## Retrieval

| Symptom | Likely cause | Confirm and fix |
|---|---|---|
| "I don't have enough information" on an answerable question | Similarity threshold too high for the embedding model, or the chunk is restricted for this user | Run in MANUAL mode with Vectorless; check the Trace page for the filtered count; lower `similarity_threshold` |
| Exact ID query returns unrelated passages | Traditional selected | Force Vectorless; if AUTO chose wrong, file the question in the evaluation set |
| Graph RAG returns nothing | Entities in the question did not match any node | Check the entities view in the console; review resolution merges |
| Agentic RAG very slow | Loop ran to the budget | Trace page shows iterations; lower `max_iterations` or tighten `evaluate_evidence` |
| A vector query fails with a dimension error from pgvector (something like "expected 768 dimensions, not N") | `embeddings.model` or `embeddings.dim` was changed in `ragfabric.yaml` without re-embedding the corpus; `chunk_embeddings.embedding` is fixed at one dimension by migration 0004 | Run `uv run ragfabric reindex`. If the new model's dimension differs from the pinned one, a new migration is needed first; see [configuration.md](configuration.md#changing-the-embedding-model) |
| `ProviderError: ollama: ... model 'nomic-embed-text' not found` (or the equivalent from `ollama list`) | The default embedding model was never pulled into the local Ollama install | `ollama pull nomic-embed-text`. This is a one time step per machine, not something RagFabric can do for you |
| Retrieval or ingestion fails with a connection error from the Chroma client | `vector_store.kind: chroma` but the Chroma container is not running, or `CHROMA_URL` points at the wrong port | `docker compose --profile full up -d chroma`; `curl http://localhost:8001/api/v2/heartbeat` (the compose file publishes Chroma on host port 8001, not 8000); set `CHROMA_URL=http://localhost:8001` |
| `ProviderError: cross_encoder: sentence-transformers is not installed. Install the extra with 'ragfabric[rerank]' or set reranker.kind to none or llm.` | `reranker.kind: cross_encoder` without the extra installed | `uv pip install 'ragfabric[rerank]'`, or set `reranker.kind` to `none` or `llm` if you cannot install `torch` in this environment |
| An answer is shorter than expected, has no citations at all, and every clause is a verbatim quote | The citation contract rejected the LLM's answer twice in a row and the run fell back to the extractive generator | Expected, not a bug: `generate_cited_answer` retries once with the specific violation quoted back to the model, and falls back to extraction on a second failure so a non-compliant answer is never shipped. Smaller local models (see `llama3.2:3b` in the getting started evidence) hit this more often; a stronger model or a lower `max_context_tokens` (fewer, more focused passages) can reduce it |

## Access control

| Symptom | Likely cause | Confirm and fix |
|---|---|---|
| A user sees no documents | No group grant on the collection | Console, Collections, Grants; the audit log shows `filtered_count` |
| API key returns 403 | Key scoped to other collections or strategies | Console, API keys, scopes |

## Cost and latency

| Symptom | Likely cause | Confirm and fix |
|---|---|---|
| Estimated cost looks wrong | `pricing.yaml` out of date | Update prices; cost is always labelled an estimate |
| Latency spikes | Reranker or Agentic on simple questions | Dashboards show latency per strategy; adjust router `min_confidence` |

## Getting help

Search existing issues, then open one with the bug template and the relevant `docker compose logs`
excerpt. Never paste API keys or documents you cannot share.
