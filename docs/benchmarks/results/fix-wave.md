# Fix wave

> Generated tables: `uv run python benchmarks/summarise.py --fill docs/benchmarks/results/*.md`.
> **TBD** means no stored run yet; **n/a** means not measured (ADR 0004). Method, datasets, licences
> and setup: [comparison.md](../comparison.md).
> Phase 11 paused at a checkpoint on 25 October 2026: a table still reading TBD was not run before
> it, and resumes from the same batch ([benchmarks overview](../README.md#resuming-the-runs)).

Each loss or bug the benchmarks exposed is fixed test first where it can be, and the affected batch
is rerun on the fixed commit. Both numbers stay published (D18). The first batches ran on the frozen
commit `ec9d4b8`; the commit of every run is in its table.

| Finding | Effect | Fix |
|---|---|---|
| Ollama's OpenAI compatible endpoint ignored `num_ctx` and cut long prompts from the front | Lost system prompt and top ranked passages on long prompts | Native `/api/chat` with `num_ctx` on every call (`e9ba6dc`, before the benchmarks ran) |
| Graph extraction dropped every entity in a chunk when one relationship was malformed | People missing from the graph | Entities and relationships validated one by one (`9473b32`) |
| Agentic stopped before its first retrieval when planning outran `max_latency_ms` | "No evidence" answers on a busy model | Spend caps checked from the second iteration (`9473b32`) |
| Streamed answers reported 0 output tokens | A wrong token count shown as a measurement | Providers report stream usage; unknown is null (`9473b32`) |
| The citation retry never showed the model the answer it had to repair | Right but badly cited answers became refusals | Rejected answer resent as the model's turn with a repair instruction (`3f32905`) |
| A small model answered with citation markers alone, such as `[1]`, and the answer was accepted | An empty answer shown as the whole answer, including the quickstart sample | An answer must hold words besides its markers; single shot generation retries, then falls back to the extractive answer, and agentic refuses (`b18811d`, #97) |
| The router's classifier wrote invented facts in its reasoning, which the Ask page showed above the cited answer | Unsourced claims next to a cited answer | The classifier is asked why its strategy fits and never to answer; the page labels the sentence "Why this strategy:" (`cf16bf0`, #101). The prompt change moves some routing decisions |
| Agentic spent most of its time searching and hit `max_latency_ms` (30 s) on 27 of 50 MultiHop-RAG questions with `llama3.2:3b`; it then answered from an evidence pool of 11.6 passages on average against 5 for traditional | More refusals (27 against 21) and lower correctness than traditional, with almost the same recall | Open. Measured in runs 20 and 24 (`d7838b4`); planned fixes are in [agentic-rag.md](../../agentic-rag.md#measured-behaviour-and-next-steps) |

## Agentic, MultiHop-RAG 100 / 50, before and after

<!-- summarise kind="batch" batch="p11-mh100q50-l32" targets="agentic" columns="target,questions,errors,hit,mrr,correctness,refusals,p50_ms,p95_ms,commit,run" -->
| target | questions | errors | hit | mrr | correctness | refusals | p50_ms | p95_ms | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|
| agentic | 50 | 0 | 0.66 | 0.49 | 0.53 | 27 | 40702 | 94760 | d7838b4 | 24 |
| traditional | 50 | 0 | 0.71 | 0.59 | 0.66 | 21 | 5195 | 14367 | d7838b4 | 20 |
| vectorless | 50 | 0 | 0.78 | 0.65 | 0.65 | 21 | 4219 | 13161 | d7838b4 | 21 |
| traditional+rerank=cross_encoder | 50 | 0 | 0.83 | 0.67 | 0.52 | 26 | 4305 | 11405 | d7838b4 | 22 |
| auto | 50 | 0 | 0.73 | 0.57 | 0.66 | 22 | 6001 | 57134 | d7838b4 | 23 |
| llamaindex | 50 | 0 | 0.71 | 0.59 | 0.63 | 0 | 5317 | 19898 | d7838b4 | 43 |
| langchain | 50 | 0 | 0.71 | 0.59 | 0.72 | 0 | 5360 | 7153 | d7838b4 | 44 |
<!-- /summarise -->

<!-- summarise kind="batch" batch="p11-mh100q50-l32-fix" targets="agentic" columns="target,questions,errors,hit,mrr,correctness,refusals,p50_ms,p95_ms,commit,run" -->
| target | questions | errors | hit | mrr | correctness | refusals | p50_ms | p95_ms | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|
| agentic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->

## Citation retry, MultiHop-RAG 100 / 50, rerun

Batch name provisional until the Task 8 rerun starts.

<!-- summarise kind="batch" batch="p11-mh100q50-l32-t8" targets="traditional,vectorless,traditional+rerank=cross_encoder,auto,agentic" columns="target,questions,errors,hit,mrr,correctness,citation,refusals,p50_ms,p95_ms,commit,run" -->
| target | questions | errors | hit | mrr | correctness | citation | refusals | p50_ms | p95_ms | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| traditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vectorless | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| traditional+rerank=cross_encoder | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| auto | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| agentic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
<!-- /summarise -->
