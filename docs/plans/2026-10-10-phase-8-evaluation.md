# Phase 8: the evaluation framework. Design

> Status: design for Phase 8 (issue #9, release target v0.5.0). The task plan follows in the second
> half of this file once it is written. Written 2026-10-10 while the owner was away, under a
> pre-approval to design, record every decision with its reason, plan and build on a feature branch.
> Every decision below is open to the owner's review before the branch is merged.

**Goal:** RagFabric can answer "which strategy, on this corpus, for these questions, and at what cost"
with numbers it produced itself: a shipped synthetic corpus and question set, retrieval and
generation metrics, one runner over all four strategies and the router, results persisted and
written to `docs/benchmarks/latest.md`, and the data the dashboards read.

**Issue:** https://github.com/ranjan-del/ragfabric/issues/9

**Predecessor:** `docs/plans/2026-10-01-phase-7b-terminal-experience.md` (merged as #54) and the
v0.4.0 release (#61).

**Successor that reuses this:** Phase 11 (outward comparison against other RAG tools). The harness is
shaped so that phase adds adapters, not a second harness.

---

## What exists today

| Piece | State on main (b7632c5) |
|---|---|
| `evaluation_runs`, `evaluation_results` tables | Created by migration 0002, mapped in `models/evaluation.py`, never written |
| `docs/evaluation.md` | Concept: question schema, eight categories, metric definitions, `make eval` |
| `retrieval_runs` | One row per ask with latency split, calls and tokens; `estimated_cost_usd` and `llm_model` are always written as null |
| `pricing.py` | `estimate_cost` over `pricing.yaml`, unknown models give `usd=None` |
| Generation | `generate_cited_answer`, `generate_agentic_answer`, `generate_graph_answer` in core; the choice between them (`_generate`) lives in the server's `search.py` |
| Citation contract | `assert_citation_contract` in `generate/contract.py` (markers in range, quotes verified, cited when evidence exists) |
| Strategies | traditional, vectorless, agentic, graph, and `auto` (the router) behind one `StrategyRegistry` built by `default_registry` |
| Makefile | None at the repository root |

---

## Design

### Units

| Unit | File | What it does |
|---|---|---|
| Dataset | `ragfabric_core/evaluation/dataset.py` | `EvalQuestion`, `QuestionSet` (pydantic, strict), `load_questions(path)`, `shipped_questions()`, `shipped_corpus_paths()` |
| Shipped data | `ragfabric_core/evaluation/data/corpus/*.md`, `data/questions.json` | The synthetic company corpus and 24 questions, three per category |
| Target | `ragfabric_core/evaluation/target.py` | `EvalTarget` protocol and `TargetAnswer`, the only shape metrics read |
| RagFabric target | `ragfabric_core/evaluation/strategy_target.py` | `StrategyTarget`: one RagFabric strategy (or a rerank variant) answering in process through the same retrieve and generate code as the API |
| Generation dispatch | `ragfabric_core/generate/dispatch.py` | `Generated`, `generate_for_result`, `uses_graph_path`, `cited_llm_calls`, moved from the server's `search.py` (which re-exports them) |
| Retrieval metrics | `ragfabric_core/evaluation/metrics.py` | precision, recall, hit, reciprocal rank, from ranked contexts and expected sources |
| Citation check | `ragfabric_core/evaluation/metrics.py` | deterministic citation correctness |
| Judges | `ragfabric_core/evaluation/judge.py` | `Judge` protocol; `LLMJudge` (fixed rubric, JSON out) and `LexicalJudge` (deterministic, offline) for correctness, faithfulness and context relevance |
| Runner | `ragfabric_core/evaluation/runner.py` | Runs every question through every target, scores, aggregates, persists |
| Persistence | `ragfabric_core/evaluation/store.py` | Writes and reads `evaluation_runs` and `evaluation_results` |
| Report | `ragfabric_core/evaluation/report.py` | Renders a batch to Markdown (`docs/benchmarks/latest.md`) |
| Dashboard data | `ragfabric_core/evaluation/dashboard.py` | Latency percentiles, cost per day, calls per strategy, fallback rate, quality trend |
| CLI | `ragfabric_cli/commands/eval.py` | `ragfabric eval corpus`, `eval run`, `eval list`, `eval show`, `eval report` |
| API | `ragfabric_server/api/routes/evaluation.py` | `GET /api/eval/runs`, `GET /api/eval/runs/{id}`, `GET /api/eval/dashboard` |
| Make | `Makefile` | `make eval`, `make eval STRATEGY=graph`, `make eval QUESTIONS=./my.json` |
| Docs | `docs/evaluation.md`, `docs/complexity.md`, `docs/benchmarks/latest.md`, `docs/learning/evaluation-first-run.md`, ADR 0015 | |

### The shipped corpus

A fictitious company, Brightwater Analytics, in eight short Markdown documents shipped inside the
core package so a `pip install` user has them:

| Document | Carries |
|---|---|
| `leave-policy-2025.md`, `leave-policy-2026.md` | Two versions of one policy with deliberate differences (comparison) |
| `org-chart.md` | People, teams, who reports to whom (relationship, multi hop) |
| `project-atlas.md`, `project-beacon.md` | Project owners, budgets, dependencies between projects (multi document, multi hop) |
| `it-runbook.md` | Error codes such as `BW-7731` and their fixes (exact match) |
| `product-catalogue.md` | Product codes and prices (exact match, complex reasoning) |
| `travel-and-expenses.md` | Limits that share words with the leave policy (ambiguous) |

Every fact a question needs is written once, in plain sentences, so the expected answer is not a
matter of opinion. The corpus was written for the questions and is not a sample of real data.

### The question schema

Unchanged in spirit from `docs/evaluation.md`, with one change to `expected_sources` (decision D3):

```json
{
  "id": "q-014",
  "question": "Compare the 2025 and 2026 leave policies: how did casual leave change?",
  "expected_answer": "Casual leave rose from 8 days in 2025 to 10 days in 2026.",
  "expected_sources": [
    {"document": "leave-policy-2025.md", "evidence": "8 days of casual leave"},
    {"document": "leave-policy-2026.md", "evidence": "10 days of casual leave"}
  ],
  "question_type": "comparison",
  "difficulty": "medium"
}
```

`question_type` is one of `simple_factual`, `multi_document`, `multi_hop`, `comparison`,
`relationship`, `exact_match`, `ambiguous`, `complex_reasoning`. `evidence` is optional. The file
has a top level `{"version": 1, "name": "...", "questions": [...]}`. Bring-your-own question sets use
the same schema and are validated on load, with every bad field named.

### Metrics

**Retrieval** (deterministic, per question, over the ranked contexts the target retrieved):

| Metric | Definition |
|---|---|
| Relevant chunk | A retrieved chunk whose document is an expected source's document and, when that source names `evidence`, whose text contains the evidence (case and whitespace insensitive) |
| Precision | Relevant retrieved chunks over retrieved chunks |
| Recall | Expected sources matched by at least one retrieved chunk over expected sources |
| Hit | 1 when any retrieved chunk is relevant |
| Reciprocal rank | 1 over the rank of the first relevant chunk, 0 when none |

A question with no expected sources has `None` for all four, never 0 or 1.

**Generation:**

| Metric | How |
|---|---|
| Correctness | Judge: does the answer state what `expected_answer` states, 0 to 1 |
| Faithfulness | Judge: share of the answer's claims supported by the retrieved context, 0 to 1 |
| Context relevance | Judge: share of the retrieved context that bears on the question, 0 to 1 |
| Citation correctness | Deterministic: the Phase 3 citation contract holds (markers in range, quotes found, cited when evidence was given) **and** every sentence with a marker shares at least half of its content words with the chunks it cites. A no-evidence answer with no markers counts as correct only when nothing was retrieved |

**System** (per question, from the target's own counts): total, retrieval and generation latency;
LLM calls, retrieval calls, embedding calls; input and output tokens; estimated cost from
`pricing.yaml`, labelled an estimate, `None` when the model is not priced.

**Router:** when a batch includes `auto` and at least one fixed strategy, the report says, per
question, which strategy `auto` chose and whether its correctness matched the best fixed strategy on
that question, and counts how often it did (decision D9).

### The runner

```
for each target in the batch:            one evaluation_runs row per target
    for each question (file order):
        answer = target.answer(question)  in-process, timed, errors caught per question
        score retrieval, citations        deterministic
        score generation                   judge
        write one evaluation_results row
    write the target's summary (means, per category, percentiles, skipped)
render docs/benchmarks/latest.md when --report is given
```

A failure on one question is recorded on that row (`details.error`) and scores `None`; the run
continues. A target that cannot run in this deployment (agentic or graph with the offline provider,
graph with `graph_store.enabled: false`) is recorded as skipped with the reason and produces no
zeros.

### The CLI

| Command | What it does |
|---|---|
| `ragfabric eval corpus [--collection NAME]` | Ingests the shipped corpus into the evaluation collection (default `ragfabric-eval`); idempotent, skips a document already ingested with the same content |
| `ragfabric eval run [--strategy S]... [--questions PATH] [--collection NAME] [--category C]... [--judge auto\|llm\|lexical] [--report PATH] [--json]` | Runs a batch. No `--strategy` means all four plus `auto`. `--strategy traditional+rerank=llm` and `traditional+rerank=cross_encoder` are rerank variants |
| `ragfabric eval list [--json]` | Recent runs, one line each |
| `ragfabric eval show RUN_ID [--json]` | One run's summary and per question rows |
| `ragfabric eval report BATCH [--out PATH]` | Re-renders the Markdown report from stored rows |

`make eval` runs `ragfabric eval corpus` then `ragfabric eval run --report docs/benchmarks/latest.md`.

### The API

| Route | Returns | Who |
|---|---|---|
| `GET /api/eval/runs?limit=` | Recent runs with summaries | Admin |
| `GET /api/eval/runs/{id}` | One run with its per question results | Admin |
| `GET /api/eval/dashboard?days=30` | Latency percentiles per strategy, cost per day, calls per strategy, fallback rate (all from `retrieval_runs`), quality trend (from `evaluation_runs`) | Admin |

### Configuration

A new `evaluation` section in `ragfabric.yaml`, every key read by something (a test asserts it):

```yaml
evaluation:
  collection: ragfabric-eval   # where `eval corpus` ingests and `eval run` retrieves
  judge: auto                  # auto | llm | lexical; auto means llm unless the provider is offline
  judge_model: null            # null means the configured llm.model
  top_k: 5                     # retrieval depth for every target, so targets compare fairly
```

---

## Decisions and reasons

| # | Decision | Reason |
|---|---|---|
| D1 | The harness lives in core (`ragfabric_core/evaluation/`), the CLI and the API only render it | The import contracts already forbid core importing upward; the CLI, the API and tests then share one implementation |
| D2 | The corpus and questions ship as package data inside `ragfabric_core`, not a top level `evaluation/` folder | `make eval` and `pip install ragfabric` users must run the same set; a repository folder is invisible to a pip install (the Phase 7b quickstart samples were packaged for the same reason) |
| D3 | `expected_sources` are `{document, evidence}` pairs, not chunk ids or page anchors | Chunk ids change with `chunk_size` and with every re-ingest, and Markdown has no pages. A document name plus a short evidence phrase survives rechunking and still distinguishes the right passage from the wrong one in the same document |
| D4 | Metrics read a `TargetAnswer`, never RagFabric internals; RagFabric strategies are wrapped as `StrategyTarget` | This is the seam Phase 11 needs: an adapter for LlamaIndex or LightRAG returns the same `TargetAnswer` and every metric, the store and the report work unchanged. It also forbids a metric that only RagFabric could score |
| D5 | Generation in the evaluation goes through the same `generate_for_result` the API uses, moved from the server to core (`generate/dispatch.py`); the server re-exports the old names | An evaluation of a different code path than the one users get would measure nothing. Moving rather than copying keeps one implementation; re-exporting keeps every existing import and test working |
| D6 | Two judges: `LLMJudge` with a fixed rubric (temperature 0, JSON output, prompt version recorded) and a deterministic `LexicalJudge`; `judge: auto` picks the LLM judge unless the provider is offline | The LLM judge is the estimate the docs promise. The lexical judge keeps CI and offline runs deterministic and is labelled as lexical in every report, so it is never mistaken for the LLM rubric (ADR 0004) |
| D7 | Citation correctness is deterministic and stricter than the API's contract (adds the per sentence word overlap) | The docs name it "the ground truth for citations"; a judge cannot be the ground truth. The overlap rule checks support without a model |
| D8 | No migration. A batch is the set of `evaluation_runs` rows sharing a `name` (the batch label); run metadata (corpus and question hashes, judge, prompt version, ragfabric version, top_k, skipped reason) goes in `summary` | The Phase 1 tables already have every per question column the metrics need. A new column is not worth a migration in a release whose subject is measurement |
| D9 | Router quality is measured as agreement with the best fixed strategy per question, not against a hand written "correct strategy" label | A label would encode the author's opinion of which strategy should win, which is exactly what the evaluation exists to test. Comparing `auto` with what the fixed strategies actually scored is a measurement |
| D10 | Evaluation answers are not written to `retrieval_runs` | Those rows feed the production dashboards (latency, cost per day, fallback rate). Mixing a batch of benchmark questions into them would distort exactly the numbers an operator watches |
| D11 | The ask paths start writing `llm_model` and `estimated_cost_usd` on `retrieval_runs` | The roadmap's "cost per day" dashboard reads that column, and it is null today. Cost is computed at write time from `pricing.yaml` with the model that answered, so a later price or model change does not rewrite history |
| D12 | Evaluation retrieves as an unrestricted system principal scoped to the evaluation collection | It measures retrieval and generation quality, not access control. The access leak test belongs to Phase 11, as the roadmap says |
| D13 | Targets run sequentially, questions in file order, one target at a time | Ollama serves one model at a time on a laptop; concurrency would measure contention. Order makes a rerun comparable |
| D14 | Every target gets the same `top_k` (from `evaluation.top_k`) | Precision and recall are only comparable at equal depth |
| D15 | Reranking is compared by running `traditional+rerank=none`, `+rerank=llm` and `+rerank=cross_encoder` as separate targets; the cross encoder target is skipped with a reason when its optional extra is not installed | The roadmap asks for that comparison; making it a target variant reuses the runner and report rather than adding a mode |
| D16 | The API is read only; runs are started from the CLI or `make eval` | A full batch takes minutes to hours on a local model and there is no job runner for it; a synchronous HTTP route would time out, and a background job system is a Phase 10 hardening concern |
| D17 | The console dashboards ship as data (the `/api/eval/dashboard` route) in Phase 8; the pages that draw them are built with the Evaluation page in Phase 9 | Phase 9 already owns the Evaluation page and the assistant UI rework. Drawing the same numbers twice in two phases would mean two UI passes over one data shape; this phase's time goes to measuring correctly. The roadmap and issue #9 are updated to say so |
| D18 | Twenty four shipped questions, three per category | Enough to show a shape per category; small enough that a full local batch over five targets with an LLM judge finishes in an evening on a laptop. Users who want more bring their own |
| D19 | `make eval` writes `docs/benchmarks/latest.md` and the file is committed only from a real run, with commit, models, date, judge and the "one run, not a benchmark" label | ADR 0004 |
| D20 | Version stays 0.4.0 on this branch; CHANGELOG gains `[0.5.0] - unreleased` | Tagging and publishing need the owner's approval |
| D21 | Issue #55's remaining item is left to the separate v0.4.0 loose ends branch | Another agent owns it; two branches editing the same lines would conflict |

### Rejected

| Option | Why not |
|---|---|
| Ragas, DeepEval or TruLens as the metric engine | A large dependency tree and their own LLM calls and prompts; the harness would then measure their judge, not a rubric this repository controls and records. Phase 11 may run them as comparisons |
| Evaluating over HTTP through the SDK | Needs a running server for `make eval`, and the server adds nothing to measure that the in-process path does not already run |
| Chunk id ground truth | See D3 |
| Writing eval answers into `retrieval_runs` with a flag | Every dashboard query would need the filter forever (D10) |
| A background job route for `POST /api/eval/runs` | See D16 |

---

## Out of scope

- The dashboard and evaluation pages in the assistant UI (Phase 9, D17).
- Starting a run over the API (D16).
- Comparisons with other RAG tools, public datasets and the access leak test (Phase 11).
- Statistical significance, repeated runs and confidence intervals. One batch is one run; the report
  says so.
- Tuning any strategy, router threshold or prompt against the shipped questions. A result that looks
  wrong is written down in `docs/learning/evaluation-first-run.md`, never fixed by editing a question.
- Tagging v0.5.0, the PyPI publish and the Discussions post (owner approval).
