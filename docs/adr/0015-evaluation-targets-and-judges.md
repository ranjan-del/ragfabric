# ADR 0015: Evaluation targets, ground truth by document and evidence, two judges

Status: accepted
Date: 2026-10-10
Supersedes: none
Related: ADR 0002 (common retrieval result), ADR 0004 (measurement first, no fabricated numbers)

## Context

Phase 8 builds the evaluation harness that every later claim in this repository rests on, and Phase
11 points the same harness at other RAG tools. Three choices shape everything downstream: what the
metrics read, what counts as a correct source, and who judges an answer.

## Decision

1. **Metrics read a `TargetAnswer` and nothing else.** A target answers a question with text, ranked
   passages carrying document names, counts and latency. A RagFabric strategy is wrapped as a
   `StrategyTarget` that retrieves and generates through the same code the API uses; a Phase 11
   adapter for another tool returns the same shape.
2. **Ground truth is a document plus an optional evidence phrase,** not a chunk id or page. Relevance
   is "from that document and containing that phrase".
3. **Two judges, always named.** A fixed-rubric LLM judge (temperature 0, JSON out, prompt version
   recorded) for correctness, faithfulness and context relevance, and a deterministic lexical judge
   for offline and CI runs. Citation correctness is never judged by a model: it is the Phase 3
   contract plus a per-sentence word overlap check.
4. **Unmeasured is `null`.** A question without expected sources, a judge reply that cannot be read,
   or a target that cannot run in the deployment produces `null` or a skipped run with a reason, never
   a zero.

## Consequences

- Rechunking or re-ingesting does not invalidate a question set, and users can write ground truth by
  reading their documents rather than querying chunk ids.
- A metric that only RagFabric could compute cannot be added without changing the `TargetAnswer`
  shape, which keeps Phase 11's comparison fair.
- LLM-judge numbers carry the judge model's bias, and on a laptop the judge is often the answering
  model. Reports name the judge so a reader can discount it.
- An evidence phrase that splits across two chunks matches neither. Phrases are kept short for this
  reason.

## Rejected

- Ragas, DeepEval or TruLens as the metric engine: their own prompts and model calls would be measured,
  not a rubric this repository controls; Phase 11 may run them as comparisons.
- A hand written "correct strategy" label for router accuracy: it encodes the author's opinion. The
  router is compared with what the fixed strategies actually scored on the same question.
