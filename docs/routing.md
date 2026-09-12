# Query routing

> Status: concept complete. Implementation ships in **v0.4.0**.

## What it does

In AUTO mode the router reads the question and chooses one of Traditional, Vectorless, Agentic or Graph.
In MANUAL mode the caller names the strategy and the router is bypassed, which is how the four are
compared fairly on the same question.

## The decision object

```json
{
  "selected_strategy": "graph_rag",
  "confidence": 0.81,
  "reasoning": "The question asks about reporting lines and the policies that apply to them, which are relationships across documents.",
  "query_type": "relationship",
  "estimated_complexity": "medium",
  "expected_cost_level": "medium",
  "expected_latency_level": "medium"
}
```

`reasoning` is one user safe sentence. Hidden chain of thought is never exposed.

## How it works internally

Two stages, cheap first.

| Stage | Input | Output |
|---|---|---|
| Signals | Regex and heuristics: identifiers, quoted phrases, named entity count, comparison and aggregation words, question length, presence of dates | A feature vector and a provisional strategy with confidence |
| Classifier | A small LLM call with the question and the signals, constrained to a JSON schema | `query_type`, refined strategy, confidence, reasoning |

If the signals are decisive (an identifier pattern and nothing else, for example) the classifier is
skipped to save a call. Confidence is calibrated against the evaluation set: the router is itself
evaluated by whether its choice produced the best score per question.

## Default mapping

| Query type | Default strategy | Signals |
|---|---|---|
| Simple factual | Traditional | Short, one concept, no identifiers |
| Exact match | Vectorless | Identifier pattern, quoted phrase, proper nouns without relationships |
| Comparison, multi document, complex reasoning, ambiguous | Agentic | Comparison words, multiple constraints, vague phrasing |
| Relationship, multi hop | Graph | Multiple entities joined by relational verbs |

The mapping is a starting point, not a hard rule. The classifier can override it and the evaluation
harness measures whether it should.

## Fallbacks

| Trigger | Action |
|---|---|
| Selected strategy returns no evidence above its floor | Run Traditional |
| Graph finds no matching entities | Run Traditional |
| Vectorless finds no term matches | Run Traditional |
| Agentic exhausts its budget with insufficient evidence | Run Traditional |
| Router confidence below `min_confidence` | Run Traditional and Vectorless, fuse |

Every fallback is recorded on the run as `fallback_from`, shown in the UI and counted in the dashboards.
A high fallback rate for a strategy is a signal that the router or the strategy needs work.

## Why not only rules

Rules are transparent but brittle: "compare" appears in simple questions too. Why not only an LLM: cost
and latency on every question, and no explanation of the decision. The two stage design gets most
questions routed for free and spends a small call only where the signals disagree.

## Trade offs

A wrong route costs either accuracy (Traditional on a multi hop question) or money (Agentic on a lookup).
The evaluation framework quantifies both so the thresholds can be tuned with numbers.
