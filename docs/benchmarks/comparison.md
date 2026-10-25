# Public benchmarks and comparison

> Status: Phase 11, paused at a checkpoint on 25 October 2026. MultiHop-RAG (100 documents) and the
> SciFact and FiQA 100, and SciFact 1,000, document tiers are measured; a table still reading TBD was not run before the
> checkpoint ([resuming](README.md#resuming-the-runs)). Tables are filled from stored runs by
> `benchmarks/summarise.py --fill`; a cell reading **TBD** has no stored run behind it yet, and a
> cell reading **n/a** was not measured (ADR 0004). No number on this page is written by hand.
> Every result is one run on one laptop, on the sample, models and date named with it. It is not
> a general claim about any tool, and the samples are too small for statistical significance.

## What was compared

RagFabric's strategies and router, and two widely used Python RAG frameworks, run through the same
Phase 8 harness on the same chunks, embedding model, answering model, judge and hardware.

| Target | What it is |
|---|---|
| `traditional` | Vector search over pgvector, cited generation |
| `vectorless` | BM25 computed in SQL fused with PostgreSQL full text ranking, cited generation |
| `traditional+rerank=cross_encoder` | Vector search, then a local cross encoder reranks |
| `auto` | The router picks a strategy per question |
| `agentic` | The bounded agent loop (plan, retrieve, assess, repair) |
| `graph` | Graph RAG; only on the graph tiers, because extraction costs one model call per chunk |
| `llamaindex` | LlamaIndex vector index over RagFabric's chunks, its default question answering prompt |
| `langchain` | LangChain in-memory vector store over RagFabric's chunks, the prompt of LangChain's own RAG tutorial (it ships no single default) |

A feature comparison with the wider market, from published documentation rather than runs, is in
[landscape.md](landscape.md).

Haystack, graph tools (Microsoft GraphRAG, LightRAG) and RAG platforms (RAGFlow, R2R) were not run:
graph tools need an extraction pass this laptop cannot afford beyond the smallest tier, and
platforms need their own servers and storage, which breaks "same hardware, same store" (design D13).

## Method

- **One harness.** Every target returns a `TargetAnswer` to the Phase 8 runner, which scores it
  with the same metrics and stores it in `evaluation_runs` and `evaluation_results` under a named
  batch. A batch can be resumed; the summary is recomputed from every stored row.
- **Same chunks.** The comparison frameworks index RagFabric's own chunks, read from the
  collection, with the same embedding model through Ollama, and retrieve the same `top_k`. Chunking
  is removed as a variable (D14). They answer with their own default prompt, which is what a user
  of that tool gets. The LlamaIndex adapter keeps the document name in the node metadata LlamaIndex
  shows the model; RagFabric's prompt numbers passages without document names.
- **Same context window.** Every model call, for every system, runs with `num_ctx` 8192 through
  Ollama's native chat endpoint (D8). Without it Ollama 0.34 silently cut long prompts to about
  2,048 tokens from the front, which would have measured truncation instead of retrieval.
- **Retrieval only on BEIR.** SciFact, NFCorpus and FiQA have relevance judgements and no reference
  answers, so they measure retrieval only: no generation, no judge (D6).
- **Correctness only on answer sets.** MultiHop-RAG and FinanceBench answers are judged for
  correctness against the reference answer; faithfulness and context relevance are not measured
  there and show as n/a (D7).
- **One judge.** `llama3.1:8b` judges every run, so the model sweep changes one thing, the answering
  model (D9).
- **Citations.** A target that does not emit `[n]` markers declares it, and its citation
  correctness is n/a rather than 0 (D16). The comparison frameworks do not emit markers.
- **Refusals** are answers that say the documents do not hold the answer. They are counted next to
  the hit rate, so a refusal on a retrieved source is not read as a retrieval miss.
- **Access leaks.** One collection sits behind a grant the probing principal does not have; the
  probes are answerable only from it. Per target the test counts forbidden chunks retrieved (any is
  a leak) and result slots left unfilled although the open collection could fill them (D17).
- **Fix wave.** Every loss or bug found is recorded, fixed test first where it can be, and the
  affected batch is rerun. Both numbers are published, with the commit each ran on (D18).

### Metrics

| Metric | Meaning |
|---|---|
| hit | Share of questions with at least one relevant document in the top `k` |
| mrr | Mean reciprocal rank of the first relevant result |
| recall | Share of a question's relevant documents (or evidence phrases) retrieved |
| correctness | Judge score against the reference answer, 0 to 1 |
| citation | Share of answers whose `[n]` markers point at chunks holding the evidence |
| refusals | Answers that declined (count, not share) |
| p50_ms, p95_ms | Latency percentiles per question, answer included where generated |

## Datasets

Dataset content is downloaded by `ragfabric eval import` into a local cache and never committed or
redistributed. Licences are recorded as each source states them; RagFabric does not grant any
right to the data.

| Dataset | Source | Licence as stated by the source | Used for |
|---|---|---|---|
| SciFact | BEIR zip (UKP Darmstadt) | CC BY 4.0 per the allenai/scifact LICENSE.md, abstracts ODC-By 1.0 (S2ORC); the BeIR Hugging Face card states CC BY-SA 4.0 | Retrieval |
| NFCorpus | BEIR zip (UKP Darmstadt) | Original site (Heidelberg University, StatNLP): free to use for academic purposes; other uses fall under the NutritionFacts.org terms. **Academic or non-commercial use only. Results pending licence confirmation** | Retrieval |
| FiQA-2018 | BEIR zip (UKP Darmstadt) | FiQA 2018 challenge site: available for non-commercial use only; no named licence. **Non-commercial use only** | Retrieval |
| MultiHop-RAG | Hugging Face `yixuantt/MultiHopRAG` | ODC-BY (dataset README and Hugging Face card) | Retrieval and answers |
| FinanceBench (open sample) | GitHub `patronus-ai/financebench` | CC BY-NC 4.0 per the Hugging Face card `PatronusAI/financebench`; the GitHub repository has no licence file. **Non-commercial use only** | Retrieval and answers |

### Samples

`--limit` counts documents. Questions are drawn with seed 11; a question is taken when all of its
relevant documents fit into half of the document budget, and distractors fill the rest. The same
arguments always give the same files.

| Tier | Documents | Questions | Batch |
|---|---|---|---|
| MultiHop-RAG | 100 | 50 | `p11-mh100q50-l32` |
| SciFact, NFCorpus, FiQA | 100 | every sampled question | `p11-scifact-100`, `p11-nfcorpus-100`, `p11-fiqa-100` |
| SciFact, NFCorpus, FiQA | 1,000 | every sampled question | `p11-scifact-1000`, `p11-nfcorpus-1000`, `p11-fiqa-1000` |
| SciFact, NFCorpus | full | every test question | `p11-scifact-full`, `p11-nfcorpus-full` |
| FiQA | full | every test question | `p11-fiqa-full` |
| FinanceBench | 10 filings | questions on those filings | `p11-fb10-l32` |
| Graph tiers | SciFact 100; MultiHop-RAG 20 | every sampled; 20 | `p11-scifact-100-graph`, `p11-mh20q20-graph` |

## Setup

| Item | Value |
|---|---|
| Machine | Apple M1, 8 cores, 16 GB, macOS |
| Model server | Ollama 0.34.0, native `/api/chat`, `num_ctx` 8192 |
| Answering model | `llama3.2:3b` (sweep: `qwen2.5:1.5b`, `llama3.1:8b`) |
| Judge | `llama3.1:8b`, prompt `judge-v1`, correctness only |
| Embeddings | `nomic-embed-text`, 768 dimensions, through Ollama |
| Store | PostgreSQL with pgvector, one throwaway database for every target |
| Chunking | 600 characters, 80 overlap (RagFabric's defaults); the frameworks index the same chunks |
| `top_k` | 5 |
| LlamaIndex | `llama-index-core` 0.14.25, `llama-index-llms-ollama` 0.11.0, `llama-index-embeddings-ollama` 0.10.0 |
| LangChain | `langchain-core` 1.6.9, `langchain-ollama` 1.1.0 |
| RagFabric code | Each batch records the commit it ran on (shown in the tables) |

**Hardware and model notes.** Local inference on this laptop is slow, so samples were sized from a
timing pass of five questions per target (D12). Ollama was shared with other work during some
batches; latency figures are only comparable inside one batch, and each results page says whether
the batch ran under contention (D19). A 3B answering model refuses more readily and follows the
citation format less reliably than a larger one; the model sweep shows how far the results depend
on that choice.

## Results

Reproduce any table: `source benchmarks/env.sh`, run the batch with `benchmarks/run_tier.sh` and
`benchmarks/run_competitor_tier.sh`, then
`uv run python benchmarks/summarise.py --fill docs/benchmarks/comparison.md docs/benchmarks/results/*.md`.
Detailed pages: [MultiHop-RAG](results/multihop-rag.md), [BEIR](results/beir.md),
[FinanceBench](results/financebench.md), [model sweep](results/model-sweep.md),
[access leak test](results/access-leak.md), [fix wave](results/fix-wave.md).

### MultiHop-RAG, 100 documents, 50 questions, `llama3.2:3b`

<!-- summarise kind="batch" batch="p11-mh100q50-l32" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,questions,errors,hit,mrr,recall,correctness,citation,refusals,p50_ms,p95_ms,commit,run" -->
| target | questions | errors | hit | mrr | recall | correctness | citation | refusals | p50_ms | p95_ms | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | 50 | 0 | 0.71 | 0.59 | 0.38 | 0.66 | 0.30 | 21 | 5195 | 14367 | d7838b4 | 20 |
| vectorless | 50 | 0 | 0.78 | 0.65 | 0.46 | 0.65 | 0.38 | 21 | 4219 | 13161 | d7838b4 | 21 |
| traditional+rerank=cross_encoder | 50 | 0 | 0.83 | 0.67 | 0.41 | 0.52 | 0.26 | 26 | 4305 | 11405 | d7838b4 | 22 |
| auto | 50 | 0 | 0.73 | 0.57 | 0.42 | 0.66 | 0.36 | 22 | 6001 | 57134 | d7838b4 | 23 |
| agentic | 50 | 0 | 0.66 | 0.49 | 0.40 | 0.53 | 0.22 | 27 | 40702 | 94760 | d7838b4 | 24 |
| llamaindex | 50 | 0 | 0.71 | 0.59 | 0.38 | 0.63 | n/a | 0 | 5317 | 19898 | d7838b4 | 43 |
| langchain | 50 | 0 | 0.71 | 0.59 | 0.38 | 0.72 | n/a | 0 | 5360 | 7153 | d7838b4 | 44 |
<!-- /summarise -->

### SciFact, 100 documents, retrieval only

<!-- summarise kind="batch" batch="p11-scifact-100" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,questions,errors,hit,mrr,recall,p50_ms,p95_ms,commit,run" -->
| target | questions | errors | hit | mrr | recall | p50_ms | p95_ms | commit | run |
|---|---|---|---|---|---|---|---|---|---|
| traditional | 63 | 0 | 0.94 | 0.82 | 0.94 | 39 | 45 | d7838b4 | 25 |
| vectorless | 63 | 0 | 0.89 | 0.82 | 0.87 | 20 | 31 | d7838b4 | 26 |
| traditional+rerank=cross_encoder | 63 | 0 | 0.92 | 0.83 | 0.92 | 135 | 150 | d7838b4 | 27 |
| auto | 63 | 0 | 0.90 | 0.79 | 0.90 | 35 | 26508 | d7838b4 | 28 |
| agentic | 63 | 0 | 0.89 | 0.71 | 0.89 | 21598 | 41798 | d7838b4 | 29 |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### NFCorpus, 100 documents, retrieval only (pending licence confirmation)

<!-- summarise kind="batch" batch="p11-nfcorpus-100" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,questions,errors,hit,mrr,recall,p50_ms,p95_ms,commit,run" -->
| target | questions | errors | hit | mrr | recall | p50_ms | p95_ms | commit | run |
|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| agentic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### FiQA-2018, 100 documents, retrieval only (non-commercial data)

<!-- summarise kind="batch" batch="p11-fiqa-100" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,questions,errors,hit,mrr,recall,p50_ms,p95_ms,commit,run" -->
| target | questions | errors | hit | mrr | recall | p50_ms | p95_ms | commit | run |
|---|---|---|---|---|---|---|---|---|---|
| traditional | 20 | 0 | 1.00 | 0.94 | 0.92 | 37 | 50 | d7838b4 | 30 |
| vectorless | 20 | 0 | 0.85 | 0.80 | 0.67 | 26 | 41 | d7838b4 | 31 |
| traditional+rerank=cross_encoder | 20 | 0 | 0.95 | 0.90 | 0.86 | 129 | 228 | d7838b4 | 32 |
| auto | 20 | 0 | 1.00 | 0.91 | 0.87 | 48 | 19789 | d7838b4 | 33 |
| agentic | 20 | 0 | 0.90 | 0.80 | 0.75 | 13981 | 40938 | d7838b4 | 34 |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### FinanceBench, 10 filings, `llama3.2:3b` (non-commercial data)

<!-- summarise kind="batch" batch="p11-fb10-l32" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,questions,errors,hit,mrr,recall,correctness,citation,refusals,p50_ms,p95_ms,commit,run" -->
| target | questions | errors | hit | mrr | recall | correctness | citation | refusals | p50_ms | p95_ms | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| agentic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

### Access leak test

<!-- summarise kind="leak" file="leak-test.json" targets="ragfabric:traditional,ragfabric:vectorless,ragfabric:auto,ragfabric:agentic,llamaindex:default,llamaindex:metadata-filter,langchain:default,langchain:metadata-filter" -->
| target | probes_leaking | forbidden_chunks | unfilled_slots |
|---|---|---|---|
| ragfabric:traditional | TBD | TBD | TBD |
| ragfabric:vectorless | TBD | TBD | TBD |
| ragfabric:auto | TBD | TBD | TBD |
| ragfabric:agentic | TBD | TBD | TBD |
| llamaindex:default | TBD | TBD | TBD |
| llamaindex:metadata-filter | TBD | TBD | TBD |
| langchain:default | TBD | TBD | TBD |
| langchain:metadata-filter | TBD | TBD | TBD |
<!-- /summarise -->

## What the results say

Read from the tables above, at the 25 October 2026 checkpoint. Each point is one run on a small
sample, not a general claim.

- **Retrieval is equal by construction where the chunks are shared.** LlamaIndex and LangChain
  search RagFabric's own chunks with the same embedding model, so their hit rate and MRR match
  `traditional` exactly. The differences between the three are in the answer step.
- **RagFabric lost on correctness to LangChain on MultiHop-RAG.** LangChain's tutorial prompt
  scored higher than every RagFabric strategy, and LlamaIndex scored a little below `traditional`.
  The main cause is visible in the refusals column: RagFabric's citation contract refuses an answer
  it cannot cite, and with `llama3.2:3b` that happened on many questions, while the two frameworks
  never refused and give no citations to check. RagFabric trades some correctness on a small model
  for answers whose sources can be verified.
- **Vectorless and the cross encoder find more than vector search.** On MultiHop-RAG both beat
  `traditional` on hit rate and MRR. The reranker's better retrieval did not turn into better
  answers: its correctness and citation scores fell, which is recorded for the fix wave.
- **Vector search is strongest on the BEIR samples.** On SciFact and FiQA `traditional` leads or
  ties on hit rate, and drops only a little from 100 to 1,000 SciFact documents.
- **Agentic is the slowest and, on this model, the weakest.** It scores below `traditional` on
  correctness, citations and hit rate at several times the latency; the cause and the planned fixes
  are in [agentic-rag.md](../agentic-rag.md#measured-behaviour-and-next-steps).
- **The answering model matters more than the strategy.** On the shipped questions, the same
  retrieval answered by two cloud models scored well above the local `llama3.1:8b`
  ([model sweep](results/model-sweep.md#cloud-models-on-the-shipped-question-set)).

## Limits

- One run per configuration; no repeated runs and no confidence intervals.
- Small samples (50 questions on MultiHop-RAG, 10 filings on FinanceBench).
- One laptop, sometimes shared; latency is indicative only.
- The judge is a local 8B model scoring correctness only; its scores are an estimate.
- BEIR measures retrieval only, so generation quality on those sets is not measured.
