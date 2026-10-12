# RagFabric

**Self hosted, measurement first retrieval augmented generation.** Run your documents through four
retrieval architectures, measure accuracy, latency and cost on your own corpus, then serve the
winner behind a query router. Bring your own model, database and hosting.

> Status: pre-release, built in the open phase by phase. Each page carries a status line saying what
> has shipped and what has not. The [roadmap](https://github.com/ranjan-del/ragfabric/blob/main/ROADMAP.md)
> is the record of what exists. No number on this site comes from anything but a recorded run
> ([ADR 0004](adr/0004-measurement-first-no-fabricated-numbers.md)).

## Start here

| If you want to | Read |
|---|---|
| Run it on a laptop and ask a first question | [Quick start](getting-started.md) |
| Understand why there is more than one way to retrieve | [Concepts](concepts/ingestion.md), then [Trade-offs](trade-offs.md) |
| Pick a strategy | [Traditional](traditional-rag.md), [Vectorless](vectorless-rag.md), [Agentic](agentic-rag.md), [Graph](graph-rag.md), and the [Router](routing.md) that chooses between them |
| Keep a folder or a Google Drive in sync | [Connectors](connectors.md) |
| Run it for other people | [Deployment](deployment.md), then [Operations](operations.md) |
| Look something up | [Configuration](configuration.md), [Providers](providers.md), [Troubleshooting](troubleshooting.md) |
| See how it is built and why | [Architecture](architecture.md) and the architecture decision records, from [ADR 0001](adr/0001-monorepo-with-layered-packages.md) |
| Help | [Contributing](contributing.md) |

## The four strategies

| Strategy | Retrieves by | Good at |
|---|---|---|
| Traditional | Embedding the question and searching a vector index | Paraphrased questions over prose |
| Vectorless | BM25 and PostgreSQL full text, fused | Exact terms, codes, identifiers; no embedding cost |
| Agentic | Decomposing the question and repairing the parts that fail | Multi-part questions |
| Graph | Walking a knowledge graph extracted from the documents | Questions about relationships |

Every strategy returns the same result shape with the same citation contract: every claim in an
answer points at a passage, and the passage is checked to support it.

## Source

The code, issues and discussions are at
[github.com/ranjan-del/ragfabric](https://github.com/ranjan-del/ragfabric), under the
[Apache License 2.0](licensing.md).
