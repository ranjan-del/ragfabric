# BEIR results (retrieval only)

> Generated tables: `uv run python benchmarks/summarise.py --fill docs/benchmarks/results/*.md`.
> **TBD** means no stored run yet; **n/a** means not measured (ADR 0004). Method, datasets, licences
> and setup: [comparison.md](../comparison.md).
> Phase 11 paused at a checkpoint on 25 October 2026: a table still reading TBD was not run before
> it, and resumes from the same batch ([benchmarks overview](../README.md#resuming-the-runs)).

Retrieval only: BEIR has relevance judgements and no reference answers, so nothing is generated or
judged (D6). Licences as stated by the sources: SciFact CC BY 4.0 (allenai/scifact; the BeIR card
states CC BY-SA 4.0); NFCorpus academic use only, **results pending licence confirmation**; FiQA-2018
non-commercial use only.

Contention (D19): none known for the 100 and 1,000 document tiers (25 October 2026). Retrieval only runs make few model calls, so latency is mostly search time.

## SciFact

### 100 documents

<!-- summarise kind="batch" batch="p11-scifact-100" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | finished | 63 | 0 | 0.94 | 0.82 | 0.94 | 0.49 | 39 | 45 | d7838b4 | 2026-10-25 | 25 |
| vectorless | finished | 63 | 0 | 0.89 | 0.82 | 0.87 | 0.45 | 20 | 31 | d7838b4 | 2026-10-25 | 26 |
| traditional+rerank=cross_encoder | finished | 63 | 0 | 0.92 | 0.83 | 0.92 | 0.49 | 135 | 150 | d7838b4 | 2026-10-25 | 27 |
| auto | finished | 63 | 0 | 0.90 | 0.79 | 0.90 | 0.43 | 35 | 26508 | d7838b4 | 2026-10-25 | 28 |
| agentic | finished | 63 | 0 | 0.89 | 0.71 | 0.89 | 0.34 | 21598 | 41798 | d7838b4 | 2026-10-25 | 29 |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### 1000 documents

<!-- summarise kind="batch" batch="p11-scifact-1000" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | finished | 300 | 0 | 0.88 | 0.81 | 0.87 | 0.46 | 34 | 39 | d7838b4 | 2026-10-25 | 35 |
| vectorless | finished | 300 | 0 | 0.83 | 0.74 | 0.81 | 0.39 | 47 | 54 | d7838b4 | 2026-10-25 | 36 |
| traditional+rerank=cross_encoder | finished | 300 | 0 | 0.88 | 0.81 | 0.86 | 0.45 | 130 | 150 | d7838b4 | 2026-10-25 | 37 |
| auto | finished | 300 | 0 | 0.85 | 0.78 | 0.84 | 0.42 | 48 | 21212 | d7838b4 | 2026-10-25 | 38 |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### full corpus

<!-- summarise kind="batch" batch="p11-scifact-full" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

## NFCorpus

### 100 documents

<!-- summarise kind="batch" batch="p11-nfcorpus-100" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| agentic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### 1000 documents

<!-- summarise kind="batch" batch="p11-nfcorpus-1000" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### full corpus

<!-- summarise kind="batch" batch="p11-nfcorpus-full" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

## FiQA-2018

### 100 documents

<!-- summarise kind="batch" batch="p11-fiqa-100" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | finished | 20 | 0 | 1.00 | 0.94 | 0.92 | 0.62 | 37 | 50 | d7838b4 | 2026-10-25 | 30 |
| vectorless | finished | 20 | 0 | 0.85 | 0.80 | 0.67 | 0.47 | 26 | 41 | d7838b4 | 2026-10-25 | 31 |
| traditional+rerank=cross_encoder | finished | 20 | 0 | 0.95 | 0.90 | 0.86 | 0.60 | 129 | 228 | d7838b4 | 2026-10-25 | 32 |
| auto | finished | 20 | 0 | 1.00 | 0.91 | 0.87 | 0.57 | 48 | 19789 | d7838b4 | 2026-10-25 | 33 |
| agentic | finished | 20 | 0 | 0.90 | 0.80 | 0.75 | 0.46 | 13981 | 40938 | d7838b4 | 2026-10-25 | 34 |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### 1000 documents

<!-- summarise kind="batch" batch="p11-fiqa-1000" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | finished | 184 | 0 | 0.83 | 0.71 | 0.63 | 0.41 | 33 | 37 | d7838b4 | 2026-10-25 | 39 |
| vectorless | finished | 184 | 0 | 0.78 | 0.63 | 0.55 | 0.32 | 29 | 33 | d7838b4 | 2026-10-25 | 40 |
| traditional+rerank=cross_encoder | finished | 184 | 0 | 0.82 | 0.74 | 0.62 | 0.40 | 116 | 130 | d7838b4 | 2026-10-25 | 41 |
| auto | running | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | d7838b4 | 2026-10-25 | 42 |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### full corpus

<!-- summarise kind="batch" batch="p11-fiqa-full" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

## Graph tier, SciFact 100 documents

<!-- summarise kind="batch" batch="p11-scifact-100-graph" targets="graph" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| graph | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

Agentic is run on the 100 document tiers only; graph only on the graph tiers, because extraction
costs one model call per chunk at ingest (D11).
