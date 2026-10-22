# Evaluation

> Status: shipped in **v0.5.0** (Phase 8, unreleased on this branch). Design and decisions:
> [docs/plans/2026-10-10-phase-8-evaluation.md](plans/2026-10-10-phase-8-evaluation.md) and
> [ADR 0015](adr/0015-evaluation-targets-and-judges.md). No number appears anywhere in this repository
> unless `ragfabric eval` produced it (ADR 0004).

## Why

Every claim about which strategy is better must be measured on a defined corpus with a defined question
set, reproducibly. Otherwise the router is guesswork and the README is opinion.

## Quick start

```bash
ragfabric eval corpus                                   # load the shipped corpus (once)
ragfabric eval run --report docs/benchmarks/latest.md   # all strategies and the router
make eval                                               # the same two steps from a checkout
```

`make eval STRATEGY=graph` runs one target; `make eval QUESTIONS=./my.json COLLECTION=handbook` runs
your own questions over your own collection.

## The benchmark dataset

The shipped corpus is a fictitious company, Brightwater Analytics, in eight short Markdown documents:
two versions of a leave policy (2025 and 2026), an organisation chart, notes for two projects, an IT
runbook with error codes, a product catalogue with codes and prices, and a travel and expenses
policy. It ships inside the `ragfabric-core` package
(`ragfabric_core/evaluation/data/`), so a `pip install` user runs exactly the set a checkout does.

The question set has 24 questions, three in each of eight categories:

| Category | What it tests |
|---|---|
| `simple_factual` | One fact from one passage |
| `multi_document` | Facts from two passages or documents |
| `multi_hop` | A chain: a project's lead, then who that person reports to |
| `comparison` | The same rule in the 2025 and 2026 policies |
| `relationship` | Reporting lines in the organisation chart |
| `exact_match` | Error and product codes such as `BW-7731` and `BWA-200` |
| `ambiguous` | Questions such as "What is the limit?" that match several documents |
| `complex_reasoning` | Arithmetic over retrieved facts: a discounted price, a carry forward |

Each question follows this schema:

```json
{
  "id": "q-010",
  "question": "How did the casual leave allowance change between the 2025 and 2026 leave policies?",
  "expected_answer": "Casual leave rose from 8 days in 2025 to 10 days in 2026.",
  "expected_sources": [
    {"document": "leave-policy-2025.md", "evidence": "8 days of casual leave"},
    {"document": "leave-policy-2026.md", "evidence": "10 days of casual leave"}
  ],
  "question_type": "comparison",
  "difficulty": "medium"
}
```

`expected_sources` names a document and, optionally, a short evidence phrase from it. Chunk ids would
change with every re-ingest and every `chunk_size`, and Markdown has no pages; a document plus a
phrase survives both. A test checks that every evidence phrase appears in its shipped document.

### Bring your own questions

Write a file with the same schema, `{"version": 1, "name": "...", "questions": [...]}`, ingest your
documents into a collection, then:

```bash
ragfabric eval run --questions ./my-questions.json --collection handbook
```

The file is validated on load and every bad field is named. `--category` limits a run to some
categories.

### Public datasets

`ragfabric eval import` converts a public dataset into the same format: a `corpus/` folder, a
`questions.json` and a `manifest.json` (sources, the SHA-256 of every downloaded file, the licence as
the source states it, the sampling arguments, counts, and what was dropped and why).

```bash
ragfabric eval datasets                                   # name, source and licence of each
ragfabric eval import scifact --limit 100                 # writes ./bench/scifact-100
ragfabric eval corpus --path ./bench/scifact-100/corpus --collection scifact-100
ragfabric eval run --questions ./bench/scifact-100/questions.json --collection scifact-100
```

| Dataset | What it measures here | Licence as stated by the source |
|---|---|---|
| `scifact`, `nfcorpus`, `fiqa` (BEIR) | Retrieval only: BEIR has relevance judgements, not answers | SciFact CC BY 4.0; NFCorpus academic use; FiQA non-commercial |
| `multihop-rag` | Retrieval with evidence phrases, and answers | ODC-BY |
| `financebench` (open sample) | Retrieval of the right filing, and answers | CC BY-NC 4.0 |

`--limit N` counts documents. Questions are shuffled with the seed; a question is taken when its
relevant documents fit into half of the budget, and the rest is filled with distractors, so the same
arguments always give the same files. Downloads are cached in `~/.cache/ragfabric/datasets`
(`--cache-dir`) and checked against a pinned checksum; nothing is written into the repository, and
RagFabric redistributes none of the data. Check each licence before you use a dataset.

### Long runs

| Option | What it does |
|---|---|
| `--batch NAME --resume` | Continue a named batch: each target keeps the questions already stored, retries the ones stored with an error, answers the rest, and its summary is recomputed from every stored row. Without `--resume`, an existing batch name is refused |
| `--retrieval-only` | Retrieve and score retrieval only; nothing is generated or judged, and answer metrics are `None` (for sets with relevance judgements but no answers, such as BEIR) |
| `--judge-metric correctness` | Repeatable; ask the judge only for these rubrics. The others are `None`, not 0 |

A target that never emits `[n]` markers (another tool's default prompt) sets `cites = False` on its
answers, and its citation correctness is `None` rather than 0.

## Metrics

### Retrieval (deterministic)

A retrieved passage is **relevant** to an expected source when it comes from that source's document
and, if the source names `evidence`, contains it (case and whitespace insensitive).

| Metric | Definition |
|---|---|
| Precision | Relevant retrieved passages over retrieved passages |
| Recall | Expected sources matched by at least one retrieved passage, over expected sources |
| Hit rate | 1 if any retrieved passage is relevant, else 0, averaged |
| MRR | Mean of 1 / rank of the first relevant passage (0 when none) |

A question with no expected sources has no retrieval score (`null`), never 0 or 1.

### Generation

| Metric | How it is judged |
|---|---|
| Correctness | Judge: does the answer state the facts in `expected_answer`, 0 to 1 |
| Faithfulness | Judge: share of the answer's claims the retrieved context supports, 0 to 1 |
| Context relevance | Judge: share of the retrieved passages needed to answer, 0 to 1 |
| Citation correctness | Deterministic, see below |

**Two judges.** `evaluation.judge: llm` asks a model three fixed rubric questions at temperature 0.
Correctness and context relevance expect `{"score": 0..1, "reason": "..."}`. Faithfulness does not ask
for a number: the model lists the answer's claims with a true or false verdict for each, and the share
supported is computed in code (a small model asked for the number gave 0 to answers its own reason
called supported, see [Judge agreement](#judge-agreement)). A reply it cannot read becomes `null` with
the reason stored, never a guessed score. The prompt version (`judge-v2`) and model are stored with
every run; runs stored before 22 October 2026 used `judge-v1` and their faithfulness is not comparable.
`lexical` is deterministic word overlap (token F1 for correctness, supported-sentence share for
faithfulness, passages sharing a word with the expected answer for context relevance); it keeps CI
and offline runs reproducible and every report names it. `auto` (the default) is `llm` unless the
provider is offline. A judge is an estimate, and an LLM judge marking its own model's answers has an
obvious bias: use a different `evaluation.judge_model` when you can.

**Citation correctness** is the ground truth for citations. An answer passes when the Phase 3 citation
contract holds (every `[n]` points at a passage that was in context, every quote is found in a cited
passage, an answer with evidence carries a marker) **and** every sentence with a marker shares at
least half of its content words with the passages it cites, **and** no quote, code or number-with-unit
in it is cited to a passage that lacks it while another retrieved passage has it (added 22 October 2026;
runs stored before that were scored without this condition). A "could not find" answer passes only
when nothing was retrieved.

### Judge agreement

The faithfulness judge is itself measured. `packages/core/tests/fixtures/faithfulness_labels.json`
holds 73 answers to the shipped questions, each labelled by hand against the passages the answer was
given (every claim stated by the passages, or following from them by correct arithmetic), with a
one-line reason. They come from the eval harness: the traditional and vectorless targets with
`llama3.1:8b`, and traditional with `llama3.2:3b` and `qwen2.5:1.5b` (answers that refused were left
out, since the judge is never asked about them). 69 are labelled supported and 4 not. A judge score of
0.5 or more counts as "supported".

| Judge, `llama3.1:8b`, 22 October 2026, 73 labelled answers | `judge-v1` | `judge-v2` |
|---|---|---|
| Agrees with the label | 61 of 73 | 68 of 72 (one reply was cut off and is unmeasured) |
| Supported answers scored 0 | 10 | 1 |
| Supported answers scored 1.0 | 44 of 69 | 54 of 68 |
| Unsupported answers scored below 0.5 | 2 of 4 | 2 of 4 |

`judge-v1` asked for `{"score", "reason"}` with the score first, so the model committed to a number
before it had written a reason, and gave 0 to answers whose claims the context stated (for example
"10 days of casual leave in 2026" scored 0 against a passage saying so). It also sometimes scored how
much of the context the answer covered instead of how much of the answer the context supports.
`judge-v2` has the model list each claim with a true or false verdict and computes the share in code. Only
4 answers are labelled unsupported, and the judge misses two of them (an arithmetic slip and an invented
launch date), so the set measures false zeros well and sensitivity poorly. A reason-first schema with the
same rubric was tried and agreed with 48 of 73, which is why it was not shipped.

### System

Per question: total, retrieval and generation latency; LLM, retrieval and embedding calls; input and
output tokens; estimated cost from `pricing.yaml` (labelled an estimate, `null` when the model is not
priced). Per run: p50 and p95 latency (nearest rank), totals, judge calls, fallbacks, and which
strategy answered.

**Refusals** are answers that say the documents do not contain the answer. They are counted per run
and shown in the report next to the hit rate: a refusal on a question whose sources were retrieved is a
generation failure, not a retrieval one. A refusal scores 0 on correctness and fails citation
correctness when passages were retrieved, and its faithfulness is not judged (it makes no claim).

### Router

When a batch includes `auto` and at least one fixed strategy, the report lists, per question, which
strategy `auto` chose and whether its correctness matched the best fixed strategy on that question.
There is no hand written "right strategy" label: that would encode the author's opinion of which
strategy should win, which is what the evaluation exists to test.

### Reranking

`traditional+rerank=none`, `traditional+rerank=llm` and `traditional+rerank=cross_encoder` are targets
like any other, so the rerankers are compared on the same questions. The cross encoder target is
skipped, with the reason recorded, when the `rerank` extra is not installed.

### Complexity score

An engineering assessment, not a measurement: see [complexity.md](complexity.md).

## What a run does

Targets run one at a time, questions in file order. Each answer goes through the same retrieve and
generate code the API uses. The run retrieves as an unrestricted system principal scoped to the
evaluation collection at `evaluation.top_k`, the same depth for every target. Each target gets one
`evaluation_runs` row (a batch is the rows sharing a name) and each question one `evaluation_results`
row, committed as it is scored. A question that fails records its error and the run continues. A
target that cannot run here (agentic or graph with the offline provider, graph with
`graph_store.enabled: false`, a missing reranker extra) gets a run row saying why and no zeros.
Evaluation answers are not written to `retrieval_runs`, so they never distort the production
dashboards.

## Reading results

| Command or route | What |
|---|---|
| `ragfabric eval list` / `show RUN_ID` | Stored runs and per question rows (`--json` for scripts) |
| `ragfabric eval report [BATCH] --out FILE` | Re-render a stored batch as Markdown |
| `GET /api/eval/runs`, `GET /api/eval/runs/{id}` | The same, for the console (admin only) |
| `GET /api/eval/dashboard?days=30` | Latency percentiles per strategy, cost per day, calls per strategy and fallback rate from production `retrieval_runs`, and the quality trend from evaluation runs |

The dashboard pages that draw these numbers ship with the Evaluation page in Phase 9; Phase 8 ships
the data.

`docs/benchmarks/latest.md` is written by `--report` and never edited by hand. The interesting output
is not a winner. It is the shape: where Vectorless beats Traditional (exact match), where Agentic earns
its cost (comparison, complex reasoning), where Graph is the only one that gets multi hop right, and
how often the router would have chosen as well as the best fixed strategy. Twenty four questions means
one question moves a category mean by a third: read the shape, not the decimals.

## Scoring answers from outside RagFabric

A model that RagFabric does not call, for example a hosted model run as a subagent inside a coding
session with no API key, can still be benchmarked. RagFabric retrieves, the outside model answers the
exact prompts a local model would get, and RagFabric scores the answers with the same metrics and the
same judge. No code in RagFabric calls a hosted model API for this.

```bash
# 1. Retrieve only. No generation, no judge, no model call.
ragfabric eval export-prompts --questions ./bench/mh-100/questions.json \
  --collection mh-100 --strategy traditional --top-k 5 --out prompts.jsonl

# 2. Give prompts.jsonl to the answering model, in batches (see benchmarks/external/ANSWER_INSTRUCTIONS.md),
#    and save its output as answers.jsonl.

# 3. Score the answers and store them.
ragfabric eval import-answers --prompts prompts.jsonl --answers answers.jsonl \
  --label "hosted model, alias A" --batch ext-2026-10 --judge lexical
ragfabric eval report ext-2026-10
```

`--strategy` is `traditional` (default) or `vectorless`; `--top-k` defaults to `evaluation.top_k`.
Without `--questions` the shipped 24 questions are used. `--judge` and `--judge-metric` work as in
`eval run`; use the same judge as the local runs you compare with.

### File formats

`prompts.jsonl` starts with one header line, then one line per question:

```json
{"type": "header", "format": "ragfabric-external-prompts-v1", "corpus": "own corpus", "collection": "mh-100", "top_k": 5, "strategy": "traditional", "exported_at": "2026-10-22", "commit": "abc1234", "ragfabric_version": "0.5.0", "question_set": "mh-100", "questions_hash": "0123456789ab"}
{"type": "question", "id": "q-001", "question": "...", "passages": [{"number": 1, "chunk_id": 41, "document": "leave-policy-2026.md", "text": "...", "score": 0.82}], "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Passages:\n[1] ...\n\nQuestion: ..."}], "ground_truth": {"expected_answer": "...", "expected_sources": [{"document": "leave-policy-2026.md", "evidence": "10 days"}], "question_type": "simple_factual", "difficulty": "easy"}, "retrieval": {"strategy_used": "traditional", "retrieval_calls": 1, "embedding_calls": 1}}
```

`messages` is built by the same function the generator uses (`build_prompt` and the system prompt),
so an outside model sees exactly what a local model sees. `passages` and `ground_truth` are what
import needs to score hit rate, recall, precision and MRR without retrieving again. The file holds the
question set's expected answers, so give an answering model the `messages` only, never the whole line.

`answers.jsonl` has one line per question:

```json
{"id": "q-001", "answer": "Casual leave is 10 days [1].", "model": "alias A as reported", "input_tokens": 812, "output_tokens": 41}
```

`id` and `answer` are required. `model`, `input_tokens` and `output_tokens` are optional; when they
are absent they are stored as null, never zero, and the summary adds up only the values reported.
Blank lines are ignored; every other line must be a record.

### What import checks

Import runs `run_batch`, the code a normal run uses: retrieval metrics from the exported passages,
citation correctness (which `[n]` markers resolve to the exported passages and whether long or numeric
quoted spans appear in a cited passage), refusal detection by the same function that counts
refusals, and the configured judge for correctness, faithfulness and context relevance. The run is
stored in `evaluation_runs` and `evaluation_results` under `--batch` as the target
`external:<slug of the label>`. Re-importing the same label into the same batch is refused.

Each problem below is counted and named, stored on the run (`summary.meta.import_errors`), printed
by the command, shown in the report, and makes the command exit 1. A question with a problem gets an
error row with null scores; nothing is skipped silently and nothing is scored as zero.

| Error | Meaning |
|---|---|
| `unreadable_line` | The line is not a JSON object |
| `invalid_record` | `id` or `answer` missing, empty or the wrong type, or a token count that is not a non-negative integer |
| `unknown_id` | The id is not in the prompts file |
| `duplicate_id` | The id was answered more than once; none of its answers is used |
| `missing_answer` | A question in the prompts file has no usable answer |

A prompts file that cannot be read (no header, a bad line, a repeated question id) stops the import
before anything is stored.

### Provenance

Every stored external run carries, in its metadata: `answered_by` (the label), `answer_source`
(`external answers file`), `prompt_file_sha256`, `export_date` and the export commit, plus any
`reported_models`. `eval report` prints an "Answers from outside RagFabric" table with these fields
above the numbers and marks the target `(external answers)`; `eval show` prints them; and the batch
tables on the benchmark pages (`benchmarks/summarise.py`) add the same table to any batch that holds
an external run, whatever columns the page asked for. An external run therefore cannot be read as a
RagFabric API run. Quote it with its label and date, for example "answers written by hosted model,
alias A, on 22 October 2026, scored by RagFabric", never as "tested on <model> via the API".

### Limits

- An answer written by a coding-session subagent is not a controlled run. The answerer has tools and
  may read source files or search, whatever it was told.
- The answerer is a coding agent, not a plain model call: its system prompt, hidden instructions and
  sampling settings are not those of a direct API request.
- The model version is whatever the alias resolved to on the day. It is recorded only as the label
  and any model name the answerer reported, and it can change between batches.
- Sample sizes are small. Ten to twenty questions per call and a few dozen per run mean one question
  moves a mean by several points; read the shape, not the decimals.
- The same judge scores local and external answers, so judge bias applies to both, and the lexical
  judge measures word overlap only.
- The instruction text and the batch size are part of the experiment. Keep the raw outputs.

## Configuration

```yaml
evaluation:
  collection: ragfabric-eval  # where `eval corpus` ingests and `eval run` retrieves
  judge: auto                 # auto | llm | lexical
  judge_model: null           # null uses llm.model
  top_k: 5                    # retrieval depth for every target
```

## For Phase 11

Metrics read a `TargetAnswer` (answer text, ranked passages with document names, counts, latency),
never RagFabric internals. A RagFabric strategy is one `EvalTarget`; an adapter for another RAG tool
that returns the same shape gets every metric, the store and the report unchanged.
