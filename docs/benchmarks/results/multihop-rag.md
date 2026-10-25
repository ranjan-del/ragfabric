# MultiHop-RAG results

> Generated tables: `uv run python benchmarks/summarise.py --fill docs/benchmarks/results/*.md`.
> **TBD** means no stored run yet; **n/a** means not measured (ADR 0004). Method, datasets, licences
> and setup: [comparison.md](../comparison.md).
> Phase 11 paused at a checkpoint on 25 October 2026: a table still reading TBD was not run before
> it, and resumes from the same batch ([benchmarks overview](../README.md#resuming-the-runs)).

Licence as stated by the source: ODC-BY. 100 documents, 50 questions, seed 11, `top_k` 5,
answering model `llama3.2:3b`, judge `llama3.1:8b` (correctness only).

Contention (D19): partly. On 25 October 2026 another job shared the same Ollama server during part of the RagFabric targets' run; LlamaIndex and LangChain ran alone afterwards. Latency is indicative only.

## All targets

<!-- summarise kind="batch" batch="p11-mh100q50-l32" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,correctness,citation,refusals,fallbacks,p50_ms,p95_ms,llm_model,judge,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | correctness | citation | refusals | fallbacks | p50_ms | p95_ms | llm_model | judge | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | finished | 50 | 0 | 0.71 | 0.59 | 0.38 | 0.19 | 0.66 | 0.30 | 21 | 0 | 5195 | 14367 | llama3.2:3b | llm:llama3.1:8b | d7838b4 | 2026-10-25 | 20 |
| vectorless | finished | 50 | 0 | 0.78 | 0.65 | 0.46 | 0.22 | 0.65 | 0.38 | 21 | 0 | 4219 | 13161 | llama3.2:3b | llm:llama3.1:8b | d7838b4 | 2026-10-25 | 21 |
| traditional+rerank=cross_encoder | finished | 50 | 0 | 0.83 | 0.67 | 0.41 | 0.20 | 0.52 | 0.26 | 26 | 0 | 4305 | 11405 | llama3.2:3b | llm:llama3.1:8b | d7838b4 | 2026-10-25 | 22 |
| auto | finished | 50 | 0 | 0.73 | 0.57 | 0.42 | 0.19 | 0.66 | 0.36 | 22 | 0 | 6001 | 57134 | llama3.2:3b | llm:llama3.1:8b | d7838b4 | 2026-10-25 | 23 |
| agentic | finished | 50 | 0 | 0.66 | 0.49 | 0.40 | 0.12 | 0.53 | 0.22 | 27 | 0 | 40702 | 94760 | llama3.2:3b | llm:llama3.1:8b | d7838b4 | 2026-10-25 | 24 |
| llamaindex | finished | 50 | 0 | 0.71 | 0.59 | 0.38 | 0.19 | 0.63 | n/a | 0 | 0 | 5317 | 19898 | llama3.2:3b | llm:llama3.1:8b | d7838b4 | 2026-10-25 | 43 |
| langchain | finished | 50 | 0 | 0.71 | 0.59 | 0.38 | 0.19 | 0.72 | n/a | 0 | 0 | 5360 | 7153 | llama3.2:3b | llm:llama3.1:8b | d7838b4 | 2026-10-25 | 44 |
<!-- /summarise -->

## Hit rate by question type

<!-- summarise kind="category" batch="p11-mh100q50-l32" metric="hit" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" -->
| category | traditional | vectorless | traditional+rerank=cross_encoder | auto | agentic | llamaindex | langchain |
|---|---|---|---|---|---|---|---|
| ambiguous | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| comparison | 0.67 | 0.87 | 0.93 | 0.67 | 0.60 | 0.67 | 0.67 |
| multi_document | 0.50 | 0.60 | 0.70 | 0.50 | 0.50 | 0.50 | 0.50 |
| multi_hop | 0.88 | 0.81 | 0.81 | 0.94 | 0.81 | 0.88 | 0.88 |
<!-- /summarise -->

## Correctness by question type

<!-- summarise kind="category" batch="p11-mh100q50-l32" metric="correctness" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" -->
| category | traditional | vectorless | traditional+rerank=cross_encoder | auto | agentic | llamaindex | langchain |
|---|---|---|---|---|---|---|---|
| ambiguous | 0.89 | 0.89 | 0.89 | 0.89 | 1.00 | 0.28 | 1.00 |
| comparison | 0.57 | 0.40 | 0.27 | 0.43 | 0.47 | 0.69 | 0.83 |
| multi_document | 0.35 | 0.48 | 0.50 | 0.48 | 0.20 | 0.61 | 0.53 |
| multi_hop | 0.82 | 0.84 | 0.56 | 0.86 | 0.53 | 0.80 | 0.57 |
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
