# What the first evaluation run looked like, learning notes

> Status: Phase 8 (on branch `feat/phase-8-evaluation`). A record of **one** run of the shipped
> question set through every strategy, the router and the LLM reranker, against a local model. See
> [routing-first-run.md](routing-first-run.md), [agentic-first-run.md](agentic-first-run.md) and
> [graph-extraction-first-run.md](graph-extraction-first-run.md) for the same exercise on earlier
> phases. The generated tables are in [../benchmarks/latest.md](../benchmarks/latest.md).

## This is a record, not a benchmark

Per ADR 0004, everything below is what happened on the shipped 8-document corpus and 24 questions,
with `llama3.1:8b` for every model call (answers, routing, the agent, graph extraction, the reranker
**and the judge**) and `nomic-embed-text` for embeddings, on one laptop, on 2026-10-10. It is a single
run of a non-deterministic system. Twenty four questions means one question moves a category mean by
a third. Nothing here may be quoted as a measurement of any strategy in general, and nothing here was
used to tune a strategy, a prompt, a threshold or a question. The questions were written and committed
before the run (commit `119d354`) and were not edited afterwards.

| Setting | Value |
|---|---|
| Batch | `first-run-2026-10-10` (runs 1 to 7) |
| Code | commit `3c7851f`; the report was re-rendered from the stored rows by `fe94511`, which only added the refusal count |
| Database | PostgreSQL 18 with pgvector (`ragfabric_eval_p8`), `graph_store.enabled: true` |
| `evaluation.top_k` | 5 |
| Judge | LLM rubric `judge-v1`, the same `llama3.1:8b`, so the judge marks its own model's answers |
| Wall time | corpus load with graph extraction 14 minutes; the batch 76 minutes |

## What the harness itself did

Everything the offline tests promise held on a real run. No question raised an error. No judge reply
was unreadable: all 345 judge calls returned a score between 0 and 1 in JSON. The cross encoder target
was skipped with its reason (`sentence-transformers` is not installed) and produced no rows. Every
answer was priced from `pricing.yaml`, at zero for local Ollama, and the report labels it an estimate.

One thing the run exposed in the harness's own reporting, fixed during the run as decision D22: the
largest single cause of low scores was the model **refusing** ("I could not find an answer to that in
the documents provided") on questions whose sources had been retrieved. Without a count, that reads
as a retrieval failure. Refusals are now counted per run and shown next to the hit rate. The metric
definitions were not changed.

## Findings

### 1. Retrieval was not the bottleneck; generation was

Traditional, vectorless, auto and agentic all retrieved a relevant passage for 23 or 24 of the 24
questions (hit rate 0.96 to 1.00), yet correctness sat between 0.58 and 0.67. The gap is mostly
refusals: traditional refused 8 times and vectorless 9, most of them on questions it had a hit for.
Question q-001 ("How many days of casual leave does an employee get in 2026?") is the clearest case:
the 2026 policy was ranked first, the 2025 policy second, and the model answered that it could not
find the answer. A plausible reading is that two conflicting numbers in context make an 8B model
retreat to the no-evidence sentence. That is a hypothesis from one run, not a measured cause.

### 2. Vectorless matched or beat traditional on this corpus

Vectorless had the best hit rate (1.00) and MRR (0.91), the highest precision of the strategies that
answer everything (0.37), and the lowest median latency (4970 ms against 7995 ms), with no embedding
call. On exact-match questions (error and product codes) both reached hit rate 1.00, and the router
sent all three to vectorless. The corpus is small and written in plain sentences, which favours word
matching; this says nothing about a large or paraphrased corpus.

### 3. Graph answered only relationship questions, as designed, and refused the rest

Graph's hit rate was 0.17: it found evidence for the three relationship questions and one other, and
refused 21 of 24. That is the shape the design expects (the graph walks named entities), but the
extraction explains how narrow it is here. From 15 chunks the extractor stored 35 entities and 25
relationships. It got every reporting line in the organisation chart right, and it never extracted
Project Atlas or Project Beacon as entities, so no project question could be walked. It also invented
an entity named "rain" located in "new columnar warehouse", turned "employee" and "manager" into
person entities, and stored "Brightwater Analytics REPORTS_TO Maya Okafor". Graph's citation
correctness (0.88) is high for an unflattering reason: a refusal with nothing retrieved counts as
correct.

### 4. The router matched the best fixed strategy on 20 of 24 questions

`auto` sent 16 questions to traditional, 4 to vectorless, 3 to the agent and 1 to graph, and fell
back from graph to traditional 4 times (questions whose wording looked relational, such as "Which team
owns Project Atlas", where graph had nothing to walk). Its correctness (0.63) was second only to the
agent's, at a fifth of the agent's median latency (7202 ms against 35839 ms). On q-022 it chose
vectorless because the classifier read "price" as an exact-word search, and the arithmetic failed
there as it did everywhere.

### 5. The agent was the most correct and the slowest

The agent had the best correctness (0.67) and citation correctness (0.71), the fewest refusals (6),
and was perfect on comparison questions, at a median of 35.8 seconds and 75 model calls for 24
questions (traditional: 24). It was worse than the single-shot strategies on exact-match questions
(correctness 0.33, against 0.93 to 0.97 for traditional, vectorless and auto): it refused q-016 ("What does error BW-7731 mean?") outright,
a question every single-shot strategy answered.

### 6. The LLM reranker made things worse

`traditional+rerank=llm` dropped the hit rate from 0.96 to 0.54 and correctness from 0.60 to 0.32,
and tripled latency. The reranker asks for one score per passage for up to twelve passages of up to
1200 characters each. The reranker's own docstring warns that Ollama silently truncates a prompt
beyond its context window and the model then returns well-formed scores for a passage list it only
partly saw. This run did not check whether that happened, so the cause is open; the result is that,
with this model, reranking by LLM should be off. The cross encoder comparison the roadmap asks for
is still unmeasured, because the extra was not installed.

### 7. Nothing answered the ambiguous or arithmetic questions well

Ambiguous questions scored 0.27 correctness on every target except graph (0.00), and complex reasoning (a discounted price, a
leave carry forward, a three month overrun cost) at most 0.33. The ambiguous questions' expected
answers ask the system to name the ambiguity, which no prompt asks it to do; that is a property of
the question set as much as of the strategies, and it is recorded here rather than fixed by editing
the questions.

## What this does not tell you

- Whether any of this holds with a larger model, a different judge, or a corpus that is not written
  for the questions.
- Whether the judge is right. It is the answering model marking itself. The deterministic numbers
  (hit rate, MRR, precision, recall, citation correctness, refusals, latency, calls) do not depend on
  it.
- Anything about cost beyond "local inference is priced at zero in `pricing.yaml`".

## What to try next

- Run again with a different judge model (`evaluation.judge_model`), and with a second answering
  model, before reading anything into the correctness column.
- Install the `rerank` extra and measure the cross encoder.
- Investigate the refusal-with-a-hit pattern on q-001 (two conflicting policy years in context) and
  the LLM reranker's prompt size against Ollama's context window.
