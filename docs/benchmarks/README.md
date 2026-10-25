# Benchmarks

> Status: Phase 11, paused at a checkpoint on 25 October 2026. The basic tiers are measured; the larger tiers, model sweep, FinanceBench, leak test and graph tiers are not run yet. Part of the flagship documentation standard: every flagship
> repository documents README, Architecture, Design decisions, Benchmarks, Failure cases,
> Evaluation, Trade-offs, Deployment, Cost and Future work. See the README section
> "Project documentation".

## What this document will cover

What is measured, on which workloads and hardware, and the recorded results, each linked to the run that produced it.

## Where the results are

[latest.md](latest.md) is written by `make eval` (or `ragfabric eval run --report`) from stored runs,
with the commit, models, judge and date that produced it. How the numbers are computed:
[docs/evaluation.md](../evaluation.md). What the first run looked like and what it taught:
[docs/learning/evaluation-first-run.md](../learning/evaluation-first-run.md).

[comparison.md](comparison.md) (Phase 11) compares RagFabric's strategies with LlamaIndex and
LangChain on public datasets, with the method, datasets and licences, setup, and results; the
detailed tables are on the pages listed under it in the navigation. Its tables are filled from stored runs by
`benchmarks/summarise.py --fill`.

No number appears here unless it comes from a recorded run.

## What is measured so far

| Batch | What it measures | State |
|---|---|---|
| `p11-mh100q50-l32` | MultiHop-RAG, 100 documents, 50 questions, every RagFabric strategy, LlamaIndex and LangChain, `llama3.2:3b` | Measured |
| `p11-scifact-100`, `p11-fiqa-100`, `p11-scifact-1000` | BEIR retrieval only, RagFabric strategies (the LlamaIndex and LangChain retrieval runs are not run) | Measured |
| `ext-claude-2026-10-25` | The shipped questions answered by two cloud models from RagFabric's exported prompts | Measured |
| `p11-fiqa-1000` | BEIR retrieval only | Part way, resumes |
| Full BEIR tiers, NFCorpus, FinanceBench, model sweep (`qwen2.5:1.5b`, `llama3.1:8b`), leak test, graph tiers, the competitor BEIR runs | | Not run |

## Resuming the runs

Every batch is named and resumable: rerunning a step with the same batch name continues from its
last stored answer (`ragfabric eval run --resume`), and a finished batch is not run again.

1. Start PostgreSQL with pgvector and Ollama, then `source benchmarks/env.sh` with `DATABASE_URL`
   pointing at the database that holds the stored batches.
2. Run the remaining tiers with `benchmarks/run_tier.sh` and `benchmarks/run_competitor_tier.sh`,
   using the batch names in [comparison.md](comparison.md#samples). Run them on the commit the earlier
   batches used, or say on the results pages which batches ran on a later commit.
3. Refill the pages: `uv run python benchmarks/summarise.py --fill docs/benchmarks/comparison.md docs/benchmarks/results/*.md`.

The full FiQA tier embeds 57,638 documents per tool and takes a day or more on a laptop; it can be
left out without losing a tier size, because the full SciFact tier (5,183 documents) already
covers the largest size measured.
