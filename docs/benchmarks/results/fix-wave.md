# Fix wave

> Generated tables: `uv run python benchmarks/summarise.py --fill docs/benchmarks/results/*.md`.
> **TBD** means no stored run yet; **n/a** means not measured (ADR 0004). Method, datasets, licences
> and setup: [comparison.md](../comparison.md).

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

## Agentic, MultiHop-RAG 100 / 50, before and after

<!-- summarise kind="batch" batch="p11-mh100q50-l32" targets="agentic" columns="target,questions,errors,hit,mrr,correctness,refusals,p50_ms,p95_ms,commit,run" -->
| target | questions | errors | hit | mrr | correctness | refusals | p50_ms | p95_ms | commit | run |
|---|---|---|---|---|---|---|---|---|---|---|
| agentic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
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
