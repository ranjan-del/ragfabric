# RagFabric documentation

These pages are also built into a documentation site with Material for MkDocs (`mkdocs.yml`;
`uv sync --group docs && uv run mkdocs serve`). Plans and design documents are not published.

| Document | What it covers | Status |
|---|---|---|
| [how-it-works.md](how-it-works.md) | The tool, its pages and the benchmark numbers in plain words | Current for `main` |
| [setup.md](setup.md) | Connect a local or hosted model, run it, host it and use it | Current for `main`; each step carries its state |
| [getting-started.md](getting-started.md) | Install, first ingest, first question | Released (v0.4.0 flow); the quickstart verified from a clean folder on 25 October 2026 |
| [architecture.md](architecture.md) | Layers, packages, request flow, data model | Current for `main` |
| [concepts/ingestion.md](concepts/ingestion.md) | Cleaning, chunking trade offs, metadata, inline vs queued indexing | Released (v0.1.0) |
| [concepts/access-control.md](concepts/access-control.md) | Principals, groups, grants, overrides, API keys, audit | Released (v0.1.0) |
| [concepts/embeddings.md](concepts/embeddings.md) | Text to vector to store, cosine versus dot product, one model per deployment, reindex cost | Released (v0.1.0) |
| [concepts/reranking.md](concepts/reranking.md) | Cheap ranking plus expensive rerank, cross encoder versus bi encoder, when it does not pay | Released (v0.1.0) |
| [concepts/lexical-vs-vector.md](concepts/lexical-vs-vector.md) | BM25 term by term, ts_rank_cd versus BM25, reciprocal rank fusion, where lexical beats vectors and where it loses | Released (v0.1.0) |
| [concepts/agentic-loops.md](concepts/agentic-loops.md) | State machine agents versus retry loops, the sub-question ledger, why progress is new chunk ids, why stopping is the hard part | Released (v0.2.0) |
| [concepts/knowledge-graphs.md](concepts/knowledge-graphs.md) | Entities and relationships, why identity is a (name, type) pair, direction and invertibility, traversal bounds, where extraction goes wrong | Released (v0.3.0) |
| [learning/lexical-vs-semantic.md](learning/lexical-vs-semantic.md) | The same ground with no code or schema: the mental model and why the two approaches fail in differently shaped ways | Learning note |
| [learning/agentic-first-run.md](learning/agentic-first-run.md) | A record of one real agent run against a local model, including what the model did badly. Not a benchmark | Learning note (v0.2.0) |
| [learning/graph-extraction-first-run.md](learning/graph-extraction-first-run.md) | A record of one real extraction run against a local model: what it got right, missed and invented. Not a benchmark | Learning note (v0.3.0) |
| [learning/assistant-first-run.md](learning/assistant-first-run.md) | A record of one real session driving Ask, Compare and Trace across all four strategies through the SDK and the UI, including what went wrong. Not a benchmark | Learning note (Phase 9) |
| [traditional-rag.md](traditional-rag.md) | Embeddings, vector search, thresholds, reranking, citation contract | Released (v0.1.0) |
| [vectorless-rag.md](vectorless-rag.md) | BM25, PostgreSQL full text, fusion | Released (v0.1.0) |
| [agentic-rag.md](agentic-rag.md) | Plain state machine, sub-question ledger, six repair moves, three stop conditions, budgets | Released (v0.2.0); measured below Traditional on a 3B model, fixes planned |
| [graph-rag.md](graph-rag.md) | Extraction behind a confidence floor, reversible entity resolution, directed access-checked traversal on PostgreSQL, the graph citation contract | Released (v0.3.0) |
| [routing.md](routing.md) | RouterDecision, signals, fallbacks | Released (v0.4.0) |
| [evaluation.md](evaluation.md) | Question set, metrics, judges, `ragfabric eval`, `make eval`, the evaluation API | Implemented, validation pending: on `main` for v0.5.0, not tagged |
| [complexity.md](complexity.md) | The 1 to 5 engineering complexity assessment per strategy. An assessment, not a measurement | Implemented, validation pending (v0.5.0); an assessment, not a measurement |
| [learning/evaluation-first-run.md](learning/evaluation-first-run.md) | A record of the first full evaluation run against a local model. Not a benchmark | Learning note (Phase 8) |
| [assistant.md](assistant.md) | The Ask, Compare, Trace and Evaluation pages, what each number means and where it comes from | Implemented, validation pending: on `main` for v1.0.0 |
| [configuration.md](configuration.md) | `ragfabric.yaml`, environment, pricing | Current for `main` |
| [providers.md](providers.md) | LLM, embedding, store and connector implementations | Current for `main`; each provider carries its own state |
| [connectors.md](connectors.md) | Watched folder and Google Drive connectors: setup, what is synced, what is deleted | Implemented, validation pending: the folder connector verified for real, Drive tested with fake HTTP only |
| [deployment.md](deployment.md) | Choosing a route; the production rails every route needs | Implemented, validation pending: guides only, nothing deployed by the project |
| [deploy/single-vm.md](deploy/single-vm.md) | One machine with Docker Compose and Caddy for HTTPS | Implemented, validation pending: run on a workstation, not yet on a public server |
| [deploy/render.md](deploy/render.md), [deploy/railway.md](deploy/railway.md) | Render Blueprint and Railway setup | Implemented, validation pending: written from the providers' documentation, not deployed |
| [deploy/static-ui.md](deploy/static-ui.md) | The web app on a static host that can proxy `/api` | Implemented, validation pending: not deployed |
| [operations.md](operations.md) | Logs, request ids, rate limits, upload validation, backups, upgrades | Implemented, validation pending: on `main` for v1.0.0 |
| [releasing.md](releasing.md) | Version script, release workflow, switching on PyPI | Implemented, validation pending: no release has used it yet |
| [learning/hardening-first-run.md](learning/hardening-first-run.md) | A record of the first real run of the production stack, including what went wrong | Learning note (Phase 10) |
| [licensing.md](licensing.md) | Apache 2.0 in plain words, why not MIT or copyleft, the CLA | Current |
| [troubleshooting.md](troubleshooting.md) | Common failures and how to diagnose them | Grows with each release |
| [design/](design/) | Full design document | Approved 2026-09-13 |
| [adr/](adr/) | Architecture decision records | Accepted |
| [benchmarks/](benchmarks/) | `latest.md` is generated by `make eval`; the comparison pages are filled from stored runs | Implemented, validation pending: Phase 11 paused at a checkpoint on 25 October 2026 |

States follow the repository's three: Released (in a tagged release and exercised end to end),
Implemented, validation pending (on `main`, with a release, a live run or quality evidence missing) and
Planned. Documents describing features that have not shipped carry a status line at the top. They are written
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
