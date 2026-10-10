# Evaluation

> Status: shipped in **v0.5.0** (Phase 8, unreleased on this branch). Design and decisions:
> [docs/plans/2026-10-10-phase-8-evaluation.md](plans/2026-10-10-phase-8-evaluation.md) and
> [ADR 0015](adr/0015-evaluation-targets-and-judges.md). No number appears anywhere in this repository
> unless `ragfabric eval` produced it (ADR 0004).

## Why

Every claim about which strategy is better must be measured on a defined corpus with a defined question
set, reproducibly. Otherwise the router is guesswork and the README is opinion.

## Quick start

```bash
ragfabric eval corpus                                   # load the shipped corpus (once)
ragfabric eval run --report docs/benchmarks/latest.md   # all strategies and the router
make eval                                               # the same two steps from a checkout
```

`make eval STRATEGY=graph` runs one target; `make eval QUESTIONS=./my.json COLLECTION=handbook` runs
your own questions over your own collection.

## The benchmark dataset

The shipped corpus is a fictitious company, Brightwater Analytics, in eight short Markdown documents:
two versions of a leave policy (2025 and 2026), an organisation chart, notes for two projects, an IT
runbook with error codes, a product catalogue with codes and prices, and a travel and expenses
policy. It ships inside the `ragfabric-core` package
(`ragfabric_core/evaluation/data/`), so a `pip install` user runs exactly the set a checkout does.

The question set has 24 questions, three in each of eight categories:

| Category | What it tests |
|---|---|
| `simple_factual` | One fact from one passage |
| `multi_document` | Facts from two passages or documents |
| `multi_hop` | A chain: a project's lead, then who that person reports to |
| `comparison` | The same rule in the 2025 and 2026 policies |
| `relationship` | Reporting lines in the organisation chart |
| `exact_match` | Error and product codes such as `BW-7731` and `BWA-200` |
| `ambiguous` | Questions such as "What is the limit?" that match several documents |
| `complex_reasoning` | Arithmetic over retrieved facts: a discounted price, a carry forward |

Each question follows this schema:

```json
{
  "id": "q-010",
  "question": "How did the casual leave allowance change between the 2025 and 2026 leave policies?",
  "expected_answer": "Casual leave rose from 8 days in 2025 to 10 days in 2026.",
  "expected_sources": [
    {"document": "leave-policy-2025.md", "evidence": "8 days of casual leave"},
    {"document": "leave-policy-2026.md", "evidence": "10 days of casual leave"}
  ],
  "question_type": "comparison",
  "difficulty": "medium"
}
```

`expected_sources` names a document and, optionally, a short evidence phrase from it. Chunk ids would
change with every re-ingest and every `chunk_size`, and Markdown has no pages; a document plus a
phrase survives both. A test checks that every evidence phrase appears in its shipped document.

### Bring your own questions

Write a file with the same schema, `{"version": 1, "name": "...", "questions": [...]}`, ingest your
documents into a collection, then:

```bash
ragfabric eval run --questions ./my-questions.json --collection handbook
```

The file is validated on load and every bad field is named. `--category` limits a run to some
categories.

## Metrics

### Retrieval (deterministic)

A retrieved passage is **relevant** to an expected source when it comes from that source's document
and, if the source names `evidence`, contains it (case and whitespace insensitive).

| Metric | Definition |
|---|---|
| Precision | Relevant retrieved passages over retrieved passages |
| Recall | Expected sources matched by at least one retrieved passage, over expected sources |
| Hit rate | 1 if any retrieved passage is relevant, else 0, averaged |
| MRR | Mean of 1 / rank of the first relevant passage (0 when none) |

A question with no expected sources has no retrieval score (`null`), never 0 or 1.

### Generation

| Metric | How it is judged |
|---|---|
| Correctness | Judge: does the answer state the facts in `expected_answer`, 0 to 1 |
| Faithfulness | Judge: share of the answer's claims the retrieved context supports, 0 to 1 |
| Context relevance | Judge: share of the retrieved passages needed to answer, 0 to 1 |
| Citation correctness | Deterministic, see below |

**Two judges.** `evaluation.judge: llm` asks a model three fixed rubric questions at temperature 0 and
expects `{"score": 0..1, "reason": "..."}`; a reply it cannot read becomes `null` with the reason
stored, never a guessed score. The prompt version (`judge-v1`) and model are stored with every run.
`lexical` is deterministic word overlap (token F1 for correctness, supported-sentence share for
faithfulness, passages sharing a word with the expected answer for context relevance); it keeps CI
and offline runs reproducible and every report names it. `auto` (the default) is `llm` unless the
provider is offline. A judge is an estimate, and an LLM judge marking its own model's answers has an
obvious bias: use a different `evaluation.judge_model` when you can.

**Citation correctness** is the ground truth for citations. An answer passes when the Phase 3 citation
contract holds (every `[n]` points at a passage that was in context, every quote is found in a cited
passage, an answer with evidence carries a marker) **and** every sentence with a marker shares at
least half of its content words with the passages it cites. A "could not find" answer passes only
when nothing was retrieved.

### System

Per question: total, retrieval and generation latency; LLM, retrieval and embedding calls; input and
output tokens; estimated cost from `pricing.yaml` (labelled an estimate, `null` when the model is not
priced). Per run: p50 and p95 latency (nearest rank), totals, judge calls, fallbacks, and which
strategy answered.

### Router

When a batch includes `auto` and at least one fixed strategy, the report lists, per question, which
strategy `auto` chose and whether its correctness matched the best fixed strategy on that question.
There is no hand written "right strategy" label: that would encode the author's opinion of which
strategy should win, which is what the evaluation exists to test.

### Reranking

`traditional+rerank=none`, `traditional+rerank=llm` and `traditional+rerank=cross_encoder` are targets
like any other, so the rerankers are compared on the same questions. The cross encoder target is
skipped, with the reason recorded, when the `rerank` extra is not installed.

### Complexity score

An engineering assessment, not a measurement: see [complexity.md](complexity.md).

## What a run does

Targets run one at a time, questions in file order. Each answer goes through the same retrieve and
generate code the API uses. The run retrieves as an unrestricted system principal scoped to the
evaluation collection at `evaluation.top_k`, the same depth for every target. Each target gets one
`evaluation_runs` row (a batch is the rows sharing a name) and each question one `evaluation_results`
row, committed as it is scored. A question that fails records its error and the run continues. A
target that cannot run here (agentic or graph with the offline provider, graph with
`graph_store.enabled: false`, a missing reranker extra) gets a run row saying why and no zeros.
Evaluation answers are not written to `retrieval_runs`, so they never distort the production
dashboards.

## Reading results

| Command or route | What |
|---|---|
| `ragfabric eval list` / `show RUN_ID` | Stored runs and per question rows (`--json` for scripts) |
| `ragfabric eval report [BATCH] --out FILE` | Re-render a stored batch as Markdown |
| `GET /api/eval/runs`, `GET /api/eval/runs/{id}` | The same, for the console (admin only) |
| `GET /api/eval/dashboard?days=30` | Latency percentiles per strategy, cost per day, calls per strategy and fallback rate from production `retrieval_runs`, and the quality trend from evaluation runs |

The dashboard pages that draw these numbers ship with the Evaluation page in Phase 9; Phase 8 ships
the data.

`docs/benchmarks/latest.md` is written by `--report` and never edited by hand. The interesting output
is not a winner. It is the shape: where Vectorless beats Traditional (exact match), where Agentic earns
its cost (comparison, complex reasoning), where Graph is the only one that gets multi hop right, and
how often the router would have chosen as well as the best fixed strategy. Twenty four questions means
one question moves a category mean by a third: read the shape, not the decimals.

## Configuration

```yaml
evaluation:
  collection: ragfabric-eval  # where `eval corpus` ingests and `eval run` retrieves
  judge: auto                 # auto | llm | lexical
  judge_model: null           # null uses llm.model
  top_k: 5                    # retrieval depth for every target
```

## For Phase 11

Metrics read a `TargetAnswer` (answer text, ranked passages with document names, counts, latency),
never RagFabric internals. A RagFabric strategy is one `EvalTarget`; an adapter for another RAG tool
that returns the same shape gets every metric, the store and the report unchanged.
