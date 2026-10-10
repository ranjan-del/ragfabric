# Phase 11: public benchmarks and a fix wave. Design

> Status: design and task plan for Phase 11 (issue #52, release target v1.0.0), implemented on
> branch `feat/phase-11-benchmarks`. Written 2026-10-10 under a pre-approval to design, record every
> decision with its reason, plan and build on a feature branch. Every decision below is open to the
> owner's review before the branch is merged.
>
> **This branch is stacked on Phase 8.** It starts from `origin/feat/phase-8-evaluation` at
> `3cc177a`, because it needs the evaluation harness, which is finished but not merged. After Phase 8
> merges into `main`, this branch must be rebased onto `main` (`git rebase --onto origin/main 3cc177a`)
> before its own pull request is opened.

**Goal:** point the Phase 8 harness outward. Import public datasets into the Phase 8 question and
corpus format, run every RagFabric strategy and the router on them at several corpus sizes and on
several local models, run two widely used RAG frameworks through the same harness on the same
models, chunks and hardware, test access control for leaks, then turn every loss or bug into an
issue draft, fix what can be fixed test first, and rerun.

**Issue:** https://github.com/ranjan-del/ragfabric/issues/52

**Predecessor:** `docs/plans/2026-10-10-phase-8-evaluation.md`. Its decision D4 is the contract this
phase builds on: other tools are added as targets that return a `TargetAnswer`, not as a second
harness.

---

## What exists on the base (3cc177a)

| Piece | State |
|---|---|
| `ragfabric_core.evaluation` | Dataset schema, `EvalTarget` and `TargetAnswer`, `StrategyTarget`, retrieval metrics, citation check, LLM and lexical judges, runner, store, report |
| `ragfabric eval corpus` | Loads only the shipped corpus |
| `ragfabric eval run` | Runs a batch; a batch is named; there is no way to continue an interrupted run |
| First run | 24 questions, `llama3.1:8b` for everything, 76 minutes; findings in `docs/learning/evaluation-first-run.md` |

---

## Findings made while designing (they shape the decisions)

| # | Finding | Evidence |
|---|---|---|
| F1 | Ollama 0.34, reached through its OpenAI compatible `/v1` endpoint, silently truncates every prompt to about 2,048 tokens and keeps the **end**, so the system prompt and the first ranked passages are the part that is lost. `options.num_ctx` or `num_ctx` in the request body are ignored on that endpoint | A 6,400 token prompt with a fact in its first line: `usage.prompt_tokens` 2050 and a wrong answer on `qwen2.5:1.5b`; the same prompt on a Modelfile copy with `PARAMETER num_ctx 8192`: 6444 tokens and the right answer |
| F2 | Local inference on this laptop is slow: `llama3.1:8b` took 41 s for a 3,200 token prompt and 116 output tokens | Measured 2026-10-10, Apple M1, 16 GB, while another agent was also using Ollama |

F1 is the most plausible explanation for the LLM reranker's collapse in the Phase 8 run (twelve
passages of up to 1,200 characters is well beyond 2,048 tokens) and a candidate cause of the refusals
with a hit. It is treated as a fix wave item, and it also decides how every benchmark here is run
(D8).

---

## Design

### Units

| Unit | File | What it does |
|---|---|---|
| Importer core | `ragfabric_core/evaluation/datasets/base.py` | `DatasetSpec`, download with checksum and cache, `ImportedDataset` writer (corpus files, `questions.json`, `manifest.json`), deterministic sampling |
| BEIR | `.../datasets/beir.py` | SciFact, NFCorpus, FiQA-2018 from the BEIR zips |
| MultiHop-RAG | `.../datasets/multihop.py` | News corpus and multi hop queries |
| FinanceBench | `.../datasets/financebench.py` | The open sample's questions and their SEC filing PDFs |
| Registry | `.../datasets/__init__.py` | `DATASETS`, `import_dataset(name, out, limit, questions, seed)` |
| CLI | `ragfabric_cli/commands/eval.py` | `eval import`, `eval datasets`, `eval corpus --path`, `eval run --resume --retrieval-only --judge-metric` |
| Runner changes | `.../evaluation/runner.py` | Resume a named batch; retrieval only runs; judge metric selection |
| Comparison adapters | `benchmarks/adapters/*.py` | LlamaIndex and LangChain targets returning `TargetAnswer` |
| Leak test | `benchmarks/leak_test.py` | Forbidden collection probes for every target |
| Bench scripts | `benchmarks/run_*.sh`, `benchmarks/configs/*.yaml`, `benchmarks/summarise.py` | Reproduce every published number |
| Results | `docs/benchmarks/comparison.md`, `docs/benchmarks/results/*.md` | Method, versions, dates, hardware, results including losses |

### `ragfabric eval import`

```
ragfabric eval datasets                        list importable datasets with source and licence
ragfabric eval import scifact --limit 100      download (cached), sample, convert
ragfabric eval import multihop-rag --limit 100 --questions 50 --seed 11 --out ./bench/mh-100
ragfabric eval corpus --path ./bench/mh-100/corpus --collection mh-100
ragfabric eval run --questions ./bench/mh-100/questions.json --collection mh-100 --batch mh-100-a
```

The output directory holds `corpus/` (one file per document), `questions.json` (Phase 8 schema,
validated by `load_questions` before it is written) and `manifest.json` (source URLs, the SHA-256 of
every downloaded file, the licence as stated by the source, sample size, seed, counts, and what was
dropped and why).

### Sampling (`--limit`, `--questions`, `--seed`)

`--limit N` is always a number of **documents**. Questions are shuffled with `random.Random(seed)`
over their sorted ids; a question is taken when all of its relevant documents fit into half of the
document budget, and its relevant documents join the corpus; the rest of the budget is filled with
distractor documents drawn with the same generator. `--questions M` stops after M questions. No
`--limit` means the whole corpus and every question. The same arguments always give the same files.

### Mapping each dataset to the Phase 8 schema

| Dataset | Document file | Question | `expected_sources` | `expected_answer` | `question_type` |
|---|---|---|---|---|---|
| SciFact, NFCorpus, FiQA (BEIR) | `<doc id>.txt`, title then text | The BEIR query | Every qrels document with score above 0, document only | Empty: BEIR has relevance judgements, not answers | `simple_factual` |
| MultiHop-RAG | `<slug of title>.txt`, title, source, date, body | `query` | One per evidence item: the article, with an evidence phrase of at most 72 characters taken from the start of the verbatim `fact` | `answer` | inference to `multi_hop`, comparison to `comparison`, temporal to `multi_document`, null to `ambiguous` with no sources |
| FinanceBench open sample | `<doc_name>.pdf`, the original filing | `question` | The filing, document only | `answer` | `complex_reasoning` when the question needs calculation, `simple_factual` otherwise |

The 72 character cap is deliberate: RagFabric chunks overlap by 80 characters, so a phrase of at most
80 characters is always wholly inside at least one chunk, and a relevant chunk cannot fail to match
just because it was cut at a boundary. A MultiHop-RAG fact has a median length of 161 characters, so
the whole fact would miss about one chunk in six for a reason that has nothing to do with retrieval.

---

## Decisions and reasons

| # | Decision | Reason |
|---|---|---|
| D1 | Stack on `feat/phase-8-evaluation` (3cc177a); rebase onto `main` after Phase 8 merges | The harness is the instrument this phase uses; waiting for the merge would stall the phase |
| D2 | The importer lives in core and uses the standard library only (`urllib`, `zipfile`, `json`) | `pip install ragfabric` users must be able to run `eval import`; no new runtime dependency, no `datasets` or `pyarrow` |
| D3 | Downloads go to a cache (`~/.cache/ragfabric/datasets`, override `--cache-dir`), converted output to `--out`; nothing is written into the repository | Dataset content is never committed (licences, size). The repository holds the importer, small format fixtures written by hand, and results |
| D4 | Sources are the original publishers' files: the BEIR zips (UKP Darmstadt), MultiHop-RAG on Hugging Face (`yixuantt/MultiHopRAG`, the GitHub repository no longer carries the data), FinanceBench on GitHub (`patronus-ai/financebench`) | The files the papers describe, at stable URLs, without an extra library to read Parquet |
| D5 | Each dataset's licence is recorded as its source states it, in the manifest, in `eval datasets` and in the results | Three of the five are non-commercial (NFCorpus, FiQA, FinanceBench). A user must see that before using the data; the repository redistributes none of it |
| D6 | BEIR sets measure retrieval only (`--retrieval-only`: no generation, no judge) | BEIR has relevance judgements and no reference answers, so correctness cannot be judged; generating answers nobody can score would cost hours of laptop time for nothing |
| D7 | Answers are judged on MultiHop-RAG and FinanceBench, for correctness only (`--judge-metric correctness`); faithfulness and context relevance are left unmeasured there | The judge's context-bearing rubrics cost two long calls per answer (F2); correctness reads only the question, the reference and the answer. Unmeasured is `None`, never a number (ADR 0004) |
| D8 | Every model call in every run, for every system, goes to a Modelfile copy of the model with `num_ctx 8192` (`rfb-<model>-8k`); the competitors are given the same window | F1: without it a long prompt loses its beginning silently, and a benchmark would measure Ollama's truncation instead of retrieval. The baseline effect of F1 is measured separately, as a fix wave item |
| D9 | One judge for every run: `llama3.1:8b` (8k copy), set through `evaluation.judge_model` | The model sweep must change one thing, the answering model. A judge that changed with it would confound the comparison |
| D10 | A named batch can be resumed (`eval run --resume`): each target continues its own run in that batch and skips questions already stored | Runs take hours and the laptop may sleep; Phase 8 batches are named, so the name is the resume key. The summary is recomputed from every stored row |
| D11 | Tiers: about 100 documents, about 1,000, the full set (SciFact 5,183) and FiQA (57,638) for scale; graph extraction only on the 100 document tiers | Graph extraction costs one model call per chunk at ingest; at F2's speed a 1,000 document tier is a day of extraction. Larger tiers record graph as skipped with that reason, not as zeros |
| D12 | Sample sizes are fixed after a timing pass on 5 questions per target, and stated with their limits in the results | Planning from a guess would either overrun the time or under-use it |
| D13 | Comparison targets: LlamaIndex and LangChain; Haystack if time allows; graph tools (Microsoft GraphRAG, LightRAG) and platforms (RAGFlow, R2R) deferred | LlamaIndex and LangChain are the two most used Python RAG frameworks. Graph tools need an extraction pass per chunk like ours, which this laptop cannot afford beyond the smallest tier; platforms need their own servers and storage, which breaks "same hardware, same store" |
| D14 | Competitors index RagFabric's own chunks (read from the collection) with the same embedding model through Ollama, retrieve the same `top_k`, and answer with their own default question answering prompt | Same chunks removes chunking as a variable; the default prompt is what a user of that tool gets. Their configuration and versions are published with the results |
| D15 | Adapters live in `benchmarks/adapters/` and run with `uv run --with <pinned versions>`; nothing is added to the workspace's dependencies or CI | The frameworks are large; CI's three required checks must not grow. Their unit tests use duck typed fakes and do not import the frameworks |
| D16 | A target that does not emit `[n]` markers declares it (`TargetAnswer.cites = False`) and its citation correctness is `None`, not 0 | Scoring a tool on a contract it never claimed would be a manufactured loss |
| D17 | The access leak test puts one collection behind a grant the probing principal does not have, asks questions answered only there, and counts forbidden chunks retrieved and allowed slots left unfilled, per target | Phase 8 D12 deferred it here. Leaks are what matter, but a post-filter that leaks nothing can still starve the answer, so both are counted |
| D18 | Every loss or bug becomes an issue draft in the private notes (`phase-11-sdd/issues/`), not a filed issue; fixes are test first; affected benchmarks are rerun and both numbers are published | The owner files issues. A fix without a rerun is a claim, not a measurement |
| D19 | Contention is recorded: each batch notes whether another job was using Ollama during it | Latency figures on a shared laptop mean nothing without that note |

### Rejected

| Option | Why not |
|---|---|
| Hugging Face `datasets` as the loader | A large dependency with its own cache; the original files are a zip or a JSON away |
| Committing sampled corpora as fixtures | Licences (three are non-commercial) and size. Format tests use hand written miniatures |
| Judging BEIR answers against the relevant abstract | That would be a made-up reference answer; BEIR does not provide one |
| Running competitors with their own chunkers | Differences would then mix chunking with retrieval and generation; one variable at a time |
| Raising Ollama's server-wide context (`OLLAMA_CONTEXT_LENGTH`) | Needs a server restart that would interrupt another agent's work on the same machine; per model Modelfile copies change nothing for anyone else |

---

## Out of scope

- Filing issues, opening the pull request, tagging v1.0.0, Discussions posts (owner).
- HotpotQA (MultiHop-RAG covers multi hop with a corpus built for RAG; HotpotQA's Wikipedia corpus
  is far beyond this laptop).
- Statistical significance. Every result is one run; samples are small and the results say so.

---

# Implementation Plan

> Executed inline, one task at a time, test first, `ruff` and `lint-imports` clean, one commit per
> task, pushed after every task. Progress, timings and the resume point are kept in the private notes.

## Global Constraints

- Python 3.13, line length 100, `ruff check packages benchmarks`, `ruff format` on changed files.
- import-linter stays at 3 kept, 0 broken.
- Unit tests offline and deterministic: importer tests read hand written miniature files through a
  fake fetcher, never the network.
- ADR 0004: unmeasured is `None`; every published number names its run, models, sample and date.
- No em dashes; no organisation names; no AI attribution.

## Task 1: Importer core and BEIR
Failing tests: miniature BEIR zip (3 queries, 8 documents) converts to a valid question set; sampling
is deterministic for a seed and different for another; `--limit` counts documents, relevant
documents are always included, qrels outside the sample are dropped; manifest records licence and
checksums; a checksum mismatch on a cached file re-downloads.

## Task 2: MultiHop-RAG and FinanceBench
Failing tests: miniature MultiHop-RAG converts with the type mapping, evidence phrase at most 72
characters and present in the document, null queries with no sources; FinanceBench miniature with a
tiny generated PDF converts, the question set names the PDF.

## Task 3: CLI `eval import`, `eval datasets`, `eval corpus --path`

## Task 4: Runner: `--resume`, `--retrieval-only`, `--judge-metric`, `cites`

## Task 5: Timing pass and tiered RagFabric runs (background, resumable)

## Task 6: Model sweep

## Task 7: LlamaIndex and LangChain adapters, comparison runs, leak test

## Task 8: Fix wave: issue drafts, test-first fixes, reruns

## Task 9: `docs/benchmarks/comparison.md`, roadmap, changelog, docs
