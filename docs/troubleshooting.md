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
