# MultiHop-RAG results

> Generated tables: `uv run python benchmarks/summarise.py --fill docs/benchmarks/results/*.md`.
> **TBD** means no stored run yet; **n/a** means not measured (ADR 0004). Method, datasets, licences
> and setup: [comparison.md](../comparison.md).

Licence as stated by the source: ODC-BY. 100 documents, 50 questions, seed 11, `top_k` 5,
answering model `llama3.2:3b`, judge `llama3.1:8b` (correctness only).

Contention (D19): TBD.

## All targets

<!-- summarise kind="batch" batch="p11-mh100q50-l32" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,correctness,citation,refusals,fallbacks,p50_ms,p95_ms,llm_model,judge,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | correctness | citation | refusals | fallbacks | p50_ms | p95_ms | llm_model | judge | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| agentic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

## Hit rate by question type

<!-- summarise kind="category" batch="p11-mh100q50-l32" metric="hit" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" -->
| category | traditional | vectorless | traditional+rerank=cross_encoder | auto | agentic | llamaindex | langchain |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

## Correctness by question type

<!-- summarise kind="category" batch="p11-mh100q50-l32" metric="correctness" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" -->
| category | traditional | vectorless | traditional+rerank=cross_encoder | auto | agentic | llamaindex | langchain |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

Ambiguous questions have no relevant document by design, so their retrieval metrics are n/a and a
refusal is the expected answer.

## Graph tier, 20 documents, 20 questions

<!-- summarise kind="batch" batch="p11-mh20q20-graph" targets="graph,traditional" columns="target,status,questions,errors,hit,mrr,recall,precision,correctness,citation,refusals,fallbacks,p50_ms,p95_ms,llm_model,judge,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | correctness | citation | refusals | fallbacks | p50_ms | p95_ms | llm_model | judge | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graph | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->
