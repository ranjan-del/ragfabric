# RagFabric documentation

These pages are also built into a documentation site with Material for MkDocs (`mkdocs.yml`;
`uv sync --group docs && uv run mkdocs serve`). Plans and design documents are not published.

| Document | What it covers | Status |
|---|---|---|
| [getting-started.md](getting-started.md) | Install, first ingest, first question | Phase 3: the real end to end flow with Traditional RAG; Vectorless RAG lands in Phase 4 |
| [architecture.md](architecture.md) | Layers, packages, request flow, data model | Phase 3: Traditional RAG request flow, the citation contract, the module layout |
| [concepts/ingestion.md](concepts/ingestion.md) | Cleaning, chunking trade offs, metadata, inline vs queued indexing | Phase 2 |
| [concepts/access-control.md](concepts/access-control.md) | Principals, groups, grants, overrides, API keys, audit | Phase 2 |
| [concepts/embeddings.md](concepts/embeddings.md) | Text to vector to store, cosine versus dot product, one model per deployment, reindex cost | Phase 3 |
| [concepts/reranking.md](concepts/reranking.md) | Cheap ranking plus expensive rerank, cross encoder versus bi encoder, when it does not pay | Phase 3 |
| [concepts/lexical-vs-vector.md](concepts/lexical-vs-vector.md) | BM25 term by term, ts_rank_cd versus BM25, reciprocal rank fusion, where lexical beats vectors and where it loses | Phase 4 |
| [concepts/agentic-loops.md](concepts/agentic-loops.md) | State machine agents versus retry loops, the sub-question ledger, why progress is new chunk ids, why stopping is the hard part | Phase 5 |
| [concepts/knowledge-graphs.md](concepts/knowledge-graphs.md) | Entities and relationships, why identity is a (name, type) pair, direction and invertibility, traversal bounds, where extraction goes wrong | Phase 6 |
| [learning/lexical-vs-semantic.md](learning/lexical-vs-semantic.md) | The same ground with no code or schema: the mental model and why the two approaches fail in differently shaped ways | Phase 4 |
| [learning/agentic-first-run.md](learning/agentic-first-run.md) | A record of one real agent run against a local model, including what the model did badly. Not a benchmark | Phase 5 |
| [learning/graph-extraction-first-run.md](learning/graph-extraction-first-run.md) | A record of one real extraction run against a local model: what it got right, missed and invented. Not a benchmark | Phase 6 |
| [traditional-rag.md](traditional-rag.md) | Embeddings, vector search, thresholds, reranking, citation contract | Shipped in v0.1.0 (Phase 3) |
| [vectorless-rag.md](vectorless-rag.md) | BM25, PostgreSQL full text, fusion | Concept complete; implementation v0.1.0 |
| [agentic-rag.md](agentic-rag.md) | Plain state machine, sub-question ledger, six repair moves, three stop conditions, budgets | Phase 5: core merged for v0.2.0, not yet exposed on the API, CLI or SDK |
| [graph-rag.md](graph-rag.md) | Extraction behind a confidence floor, reversible entity resolution, directed access-checked traversal on PostgreSQL, the graph citation contract | Phase 6: core merged for v0.3.0, landed on the phase branch |
| [routing.md](routing.md) | RouterDecision, signals, fallbacks | Concept complete; implementation v0.4.0 |
| [evaluation.md](evaluation.md) | Question set, metrics, `make eval`, complexity score | Concept complete; implementation v0.5.0 |
| [configuration.md](configuration.md) | `ragfabric.yaml`, environment, pricing | Phase 3: every top level key, including `llm`, `reranker` and `limits`, is read by the runtime |
| [providers.md](providers.md) | LLM, embedding, store and connector implementations | Phase 3: pgvector and Chroma queried, rerankers shipped, Ollama the default provider |
| [connectors.md](connectors.md) | Watched folder and Google Drive connectors: setup, what is synced, what is deleted | Phase 10: folder verified for real, Drive tested with fake HTTP only |
| [deployment.md](deployment.md) | Choosing a route; the production rails every route needs | Phase 10 |
| [deploy/single-vm.md](deploy/single-vm.md) | One machine with Docker Compose and Caddy for HTTPS | Phase 10: run on a workstation, not yet on a public server |
| [deploy/render.md](deploy/render.md), [deploy/railway.md](deploy/railway.md) | Render Blueprint and Railway setup | Phase 10: written from the providers' documentation, not deployed |
| [deploy/static-ui.md](deploy/static-ui.md) | The web app on a static host that can proxy `/api` | Phase 10: not deployed |
| [operations.md](operations.md) | Logs, request ids, rate limits, upload validation, backups, upgrades | Phase 10 |
| [releasing.md](releasing.md) | Version script, release workflow, switching on PyPI | Phase 10: no release has used it yet |
| [learning/hardening-first-run.md](learning/hardening-first-run.md) | A record of the first real run of the production stack, including what went wrong | Phase 10 |
| [licensing.md](licensing.md) | Apache 2.0 in plain words, why not MIT or copyleft, the CLA | Current |
| [troubleshooting.md](troubleshooting.md) | Common failures and how to diagnose them | Grows with each release |
| [design/](design/) | Full design document | Approved 2026-09-13 |
| [adr/](adr/) | Architecture decision records | Accepted |
| benchmarks/ | Generated by `make eval`, never hand written | Empty until v0.5.0 |

Documents describing features that have not shipped carry a status line at the top. They are written
before the code on purpose: the explanation is the specification, and the implementation must match it
or the document is corrected in the same pull request.

## Flagship standard documents

Added for the flagship documentation standard (see the README section "Project documentation"). Each is a stub until filled in.

| Document | Status |
|---|---|
| [benchmarks/README.md](benchmarks/README.md) | partial |
| [cost.md](cost.md) | to be written |
| [failure-cases.md](failure-cases.md) | partial |
| [trade-offs.md](trade-offs.md) | partial |
