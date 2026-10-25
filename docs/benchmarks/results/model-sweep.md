# Model sweep, MultiHop-RAG 100 documents, 50 questions

> Generated tables: `uv run python benchmarks/summarise.py --fill docs/benchmarks/results/*.md`.
> **TBD** means no stored run yet; **n/a** means not measured (ADR 0004). Method, datasets, licences
> and setup: [comparison.md](../comparison.md).
> Phase 11 paused at a checkpoint on 25 October 2026: a table still reading TBD was not run before
> it, and resumes from the same batch ([benchmarks overview](../README.md#resuming-the-runs)).

The answering model changes; the judge (`llama3.1:8b`), chunks, embeddings and questions stay the
same (D9). Contention (D19): as on the MultiHop-RAG page for `llama3.2:3b`; the other two models were not run before the checkpoint.

## `llama3.2:3b`

<!-- summarise kind="batch" batch="p11-mh100q50-l32" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,questions,errors,hit,mrr,correctness,citation,refusals,p50_ms,p95_ms,llm_model,commit,run" -->
| target | questions | errors | hit | mrr | correctness | citation | refusals | p50_ms | p95_ms | llm_model | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | 50 | 0 | 0.71 | 0.59 | 0.66 | 0.30 | 21 | 5195 | 14367 | llama3.2:3b | d7838b4 | 20 |
| vectorless | 50 | 0 | 0.78 | 0.65 | 0.65 | 0.38 | 21 | 4219 | 13161 | llama3.2:3b | d7838b4 | 21 |
| traditional+rerank=cross_encoder | 50 | 0 | 0.83 | 0.67 | 0.52 | 0.26 | 26 | 4305 | 11405 | llama3.2:3b | d7838b4 | 22 |
| auto | 50 | 0 | 0.73 | 0.57 | 0.66 | 0.36 | 22 | 6001 | 57134 | llama3.2:3b | d7838b4 | 23 |
| agentic | 50 | 0 | 0.66 | 0.49 | 0.53 | 0.22 | 27 | 40702 | 94760 | llama3.2:3b | d7838b4 | 24 |
| llamaindex | 50 | 0 | 0.71 | 0.59 | 0.63 | n/a | 0 | 5317 | 19898 | llama3.2:3b | d7838b4 | 43 |
| langchain | 50 | 0 | 0.71 | 0.59 | 0.72 | n/a | 0 | 5360 | 7153 | llama3.2:3b | d7838b4 | 44 |
<!-- /summarise -->

## `qwen2.5:1.5b`

<!-- summarise kind="batch" batch="p11-mh100q50-qwen2.5-1.5b" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,questions,errors,hit,mrr,correctness,citation,refusals,p50_ms,p95_ms,llm_model,commit,run" -->
| target | questions | errors | hit | mrr | correctness | citation | refusals | p50_ms | p95_ms | llm_model | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| agentic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

## `llama3.1:8b`

<!-- summarise kind="batch" batch="p11-mh100q50-llama3.1-8b" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic,llamaindex,langchain" columns="target,questions,errors,hit,mrr,correctness,citation,refusals,p50_ms,p95_ms,llm_model,commit,run" -->
| target | questions | errors | hit | mrr | correctness | citation | refusals | p50_ms | p95_ms | llm_model | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| agentic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| llamaindex | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| langchain | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->


## Cloud models on the shipped question set

A separate check on how far answer quality depends on the answering model. The 24 shipped questions
were retrieved by RagFabric (traditional and vectorless, `top_k` 5) and exported with
`ragfabric eval export-prompts`; two cloud models wrote the answers from those exact prompts, and
`ragfabric eval import-answers` scored them with the same judge (`llama3.1:8b`, correctness). The
retrieval is identical across the rows, so only the answering model differs. The shipped questions
were written alongside the system, so they favour it; MultiHop-RAG above is the fairer test.

<!-- summarise kind="batch" batch="ext-claude-2026-10-25" targets="external:claude-sonnet-5-5-claude-code-subagent-alias-sonnet-25-october-2026,external:claude-haiku-5-5-claude-code-subagent-alias-haiku-25-october-2026,external:claude-sonnet-5-5-vectorless-claude-code-subagent-alias-sonnet-25-october-2026,external:claude-haiku-5-5-vectorless-claude-code-subagent-alias-haiku-25-october-2026" columns="target,questions,errors,hit,mrr,correctness,citation,refusals,commit,run" -->
| target | questions | errors | hit | mrr | correctness | citation | refusals | commit | run |
|---|---|---|---|---|---|---|---|---|---|
| external:claude-sonnet-5-5-claude-code-subagent-alias-sonnet-25-october-2026 | 24 | 0 | 0.96 | 0.88 | 0.97 | 0.92 | 0 | 343335c | 7 |
| external:claude-haiku-5-5-claude-code-subagent-alias-haiku-25-october-2026 | 24 | 0 | 0.96 | 0.88 | 0.91 | 0.92 | 0 | 343335c | 8 |
| external:claude-sonnet-5-5-vectorless-claude-code-subagent-alias-sonnet-25-october-2026 | 24 | 0 | 1.00 | 0.89 | 0.94 | 0.92 | 0 | 343335c | 9 |
| external:claude-haiku-5-5-vectorless-claude-code-subagent-alias-haiku-25-october-2026 | 24 | 0 | 1.00 | 0.89 | 0.93 | 1.00 | 0 | 343335c | 10 |

Answers below were written outside RagFabric and scored by it. They are not a controlled run and not a model tested through its API:

| target | answered_by | answer_source | prompt_hash | export_date |
|---|---|---|---|---|
| external:claude-sonnet-5-5-claude-code-subagent-alias-sonnet-25-october-2026 | Claude Sonnet 5.5, Claude Code subagent alias sonnet, 25 October 2026 | external answers file | 1a6235b5b08d | 2026-10-25 |
| external:claude-haiku-5-5-claude-code-subagent-alias-haiku-25-october-2026 | Claude Haiku 5.5, Claude Code subagent alias haiku, 25 October 2026 | external answers file | 1a6235b5b08d | 2026-10-25 |
| external:claude-sonnet-5-5-vectorless-claude-code-subagent-alias-sonnet-25-october-2026 | Claude Sonnet 5.5, vectorless, Claude Code subagent alias sonnet, 25 October 2026 | external answers file | 5f29d56d8c56 | 2026-10-25 |
| external:claude-haiku-5-5-vectorless-claude-code-subagent-alias-haiku-25-october-2026 | Claude Haiku 5.5, vectorless, Claude Code subagent alias haiku, 25 October 2026 | external answers file | 5f29d56d8c56 | 2026-10-25 |
<!-- /summarise -->

The same questions answered by the local `llama3.1:8b`, for comparison:

<!-- summarise kind="batch" batch="h2-pr2-after" targets="traditional,vectorless" columns="target,questions,errors,hit,mrr,correctness,citation,refusals,p50_ms,llm_model,commit,run" -->
| target | questions | errors | hit | mrr | correctness | citation | refusals | p50_ms | llm_model | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | 24 | 0 | 0.96 | 0.88 | 0.83 | 0.92 | 2 | 8468 | llama3.1:8b | 471dc2a | 1 |
| vectorless | 24 | 0 | 1.00 | 0.91 | 0.87 | 0.92 | 1 | 6052 | llama3.1:8b | 471dc2a | 2 |
<!-- /summarise -->
