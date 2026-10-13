# FinanceBench results

> Generated tables: `uv run python benchmarks/summarise.py --fill docs/benchmarks/results/*.md`.
> **TBD** means no stored run yet; **n/a** means not measured (ADR 0004). Method, datasets, licences
> and setup: [comparison.md](../comparison.md).

Licence as stated by the source: CC BY-NC 4.0 (Hugging Face card `PatronusAI/financebench`; the
GitHub repository has no licence file). **Non-commercial use only.** Open sample, 10 filings and the
questions on them, the original SEC filing PDFs. Expected sources are whole filings (the dataset's
evidence is a page), so retrieval metrics are at document level.

Contention (D19): TBD.

## All targets

<!-- summarise kind="batch" batch="p11-fb10-l32" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,status,questions,errors,hit,mrr,recall,precision,correctness,citation,refusals,fallbacks,p50_ms,p95_ms,llm_model,judge,commit,date,run" -->
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

## Correctness by question type

<!-- summarise kind="category" batch="p11-fb10-l32" metric="correctness" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" -->
| category | traditional | vectorless | traditional+rerank=cross_encoder | auto | agentic | llamaindex | langchain |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->
