# Evaluation

> Status: concept complete. Implementation ships in **v0.5.0**. Until then `docs/benchmarks/` stays
> empty. No number appears anywhere in this repository unless `make eval` produced it.

## Why

Every claim about which strategy is better must be measured on a defined corpus with a defined question
set, reproducibly. Otherwise the router is guesswork and the README is opinion.

## The benchmark dataset

`evaluation/corpus/` ships a synthetic company: HR policies for two years, an org chart, project notes,
IT runbooks, a product catalogue with codes. `evaluation/questions.json` holds test cases:

```json
{
  "id": "q-014",
  "question": "Compare the 2025 and 2026 leave policies and list what changed.",
  "expected_answer": "Casual leave rose from 8 to 10 days; carry forward capped at 5 days from 2026.",
  "expected_sources": ["leave-policy-2025.pdf#p2", "leave-policy-2026.pdf#p2"],
  "question_type": "comparison",
  "difficulty": "hard"
}
```

Categories: simple factual, multi document, multi hop, comparison, relationship, exact match, ambiguous,
complex reasoning. Users add their own questions over their own corpus with the same schema.

## Metrics

### Retrieval

| Metric | Definition |
|---|---|
| Precision | Retrieved chunks that are expected sources, over retrieved chunks |
| Recall | Expected sources that were retrieved, over expected sources |
| Hit rate | 1 if any expected source was retrieved, else 0, averaged |
| MRR | Mean of 1 / rank of the first expected source |

### Generation

| Metric | How it is judged |
|---|---|
| Answer correctness | LLM judge with a fixed rubric compares to `expected_answer`, returns 0 to 1 |
| Faithfulness | Judge checks every claim is supported by the retrieved context |
| Context relevance | Judge scores how much of the context was needed |
| Citation correctness | Deterministic: every marker points at a chunk that was in context, and the cited chunk supports the sentence |

The judge model, prompt and version are recorded with every run. The deterministic citation check is the
ground truth for citations; the judge is an estimate and is labelled as one.

### System

Total, retrieval and generation latency; LLM calls; retrieval calls; input and output tokens; estimated
cost from `pricing.yaml`.

### Complexity score

An engineering assessment, not a measurement, on a documented 1 to 5 scale across infrastructure,
components, operations, debugging and maintenance. Recorded in `docs/complexity.md` with the reasoning
per strategy and labelled as an assessment wherever it is shown.

## Running it

```bash
make eval                         # all strategies, full question set
make eval STRATEGY=graph          # one strategy
make eval QUESTIONS=./my.json     # your questions
```

Each run persists to `evaluation_runs` and `evaluation_results`, and regenerates
`docs/benchmarks/latest.md` with the commit hash, models, date and per category tables. The README's
benchmark section links there and is never edited by hand.

## Reading the results

The interesting output is not a winner. It is the shape: where Vectorless beats Traditional (exact match),
where Agentic earns its cost (comparison, complex reasoning), where Graph is the only one that gets multi
hop right, and how often the router would have chosen correctly. That shape is the answer to "which
architecture should we run".
