# Complexity score

> **An engineering assessment, not a measurement.** These scores are the maintainers' judgement of
> what each strategy costs to run and maintain, written down so the reasoning can be argued with. They
> are not produced by `ragfabric eval` and must never be shown next to measured numbers without this
> label (ADR 0004). Assessed 2026-10-10 against v0.5.0 (unreleased).

## Scale

Each axis is scored 1 (least) to 5 (most):

| Axis | 1 means | 5 means |
|---|---|---|
| Infrastructure | Runs on what any RagFabric install already has | Needs extra services or heavy index builds |
| Components | One retrieval call and one generation | Many moving parts that must agree |
| Operations | Nothing to tune or watch | Budgets, thresholds and model choice need ongoing attention |
| Debugging | A wrong answer is explained by one ranked list | A wrong answer needs a multi-step trace to explain |
| Maintenance | Changes rarely break it | Model or prompt changes regularly move its behaviour |

## Scores

| Strategy | Infrastructure | Components | Operations | Debugging | Maintenance | Total (of 25) |
|---|---|---|---|---|---|---|
| Vectorless | 1 | 2 | 1 | 1 | 1 | 6 |
| Traditional | 2 | 2 | 2 | 2 | 2 | 10 |
| Graph | 4 | 4 | 3 | 4 | 4 | 19 |
| Agentic | 2 | 5 | 4 | 5 | 4 | 20 |
| Auto (router) | 4 | 5 | 3 | 3 | 4 | 19 |

## Reasoning

- **Vectorless**: BM25 and PostgreSQL full text over the chunk table. No embeddings, no model at
  retrieval time. A wrong ranking is explained by the terms; the fusion weights and boosts rarely
  change.
- **Traditional**: needs an embedding model and the vector index, and the embedding model and
  dimension are pinned (ADR 0006), so changing it means re-embedding. A reranker adds one more part.
- **Graph**: extraction costs one model call per chunk at ingest, entity resolution has thresholds,
  and the walk has hop and node budgets. A wrong answer may come from extraction, resolution or the
  walk, which is why every edge keeps its source chunk. A model change changes the graph.
- **Agentic**: a state machine with planning, assessment, repair and budgets (ADRs 0009, 0010). Few
  services, but the most behaviour that a model decides, so the most to trace when it goes wrong.
- **Auto**: inherits the infrastructure of every strategy it may pick, plus signals and a classifier.
  The decision itself is one sentence and easy to inspect, so debugging is lower than the agent's.

The measured side (latency, calls, tokens, estimated cost per strategy) is in
[benchmarks/latest.md](benchmarks/latest.md).
