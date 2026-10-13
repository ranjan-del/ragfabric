# BEIR results (retrieval only)

> Generated tables: `uv run python benchmarks/summarise.py --fill docs/benchmarks/results/*.md`.
> **TBD** means no stored run yet; **n/a** means not measured (ADR 0004). Method, datasets, licences
> and setup: [comparison.md](../comparison.md).

Retrieval only: BEIR has relevance judgements and no reference answers, so nothing is generated or
judged (D6). Licences as stated by the sources: SciFact CC BY 4.0 (allenai/scifact; the BeIR card
states CC BY-SA 4.0); NFCorpus academic use only, **results pending licence confirmation**; FiQA-2018
non-commercial use only.

Contention (D19): TBD.

## 

### 100 documents

<!-- summarise kind="batch" batch="p11-scifact SciFact-100" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
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

<!-- summarise kind="batch" batch="p11-scifact SciFact-1000" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
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

<!-- summarise kind="batch" batch="p11-scifact SciFact-full" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

## 

### 100 documents

<!-- summarise kind="batch" batch="p11-nfcorpus NFCorpus-100" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
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

<!-- summarise kind="batch" batch="p11-nfcorpus NFCorpus-1000" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
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

<!-- summarise kind="batch" batch="p11-nfcorpus NFCorpus-full" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
| target | status | questions | errors | hit | mrr | recall | precision | p50_ms | p95_ms | commit | date | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

## 

### 100 documents

<!-- summarise kind="batch" batch="p11-fiqa FiQA-2018-100" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
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

<!-- summarise kind="batch" batch="p11-fiqa FiQA-2018-1000" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
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

<!-- summarise kind="batch" batch="p11-fiqa FiQA-2018-full" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,p50_ms,p95_ms,commit,date,run" -->
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
