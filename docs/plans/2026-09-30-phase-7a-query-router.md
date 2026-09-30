# Phase 7a: Query Router and the agent's reach to all four strategies. Implementation Plan

**Goal:** Choose the right strategy for a question without the caller naming one, say why in one
safe sentence, fall back honestly when the choice finds nothing, and let the agent send each
sub-question to the strategy that suits it, Graph included.

**Architecture:** A fifth registry entry, `auto`, runs a two-stage router (free signals first, one
classifier call only when the signals are not decisive), then the chosen strategy, then at most one
fallback. The same signals module checks the agent planner's tool choice per sub-question and wins
when it is decisive. A new `graph_search` tool gives the agent the graph.

**Tech Stack:** Python 3.13 (uv), SQLAlchemy 2, PostgreSQL 18, FastAPI, Pydantic, Typer. **No new
runtime dependencies. No migration:** `retrieval_runs` has carried `selected_strategy`,
`fallback_from`, `router_confidence` and `router_reasoning` since migration 0002.

**Issue:** https://github.com/ranjan-del/ragfabric/issues/8

**Predecessor:** `docs/plans/2026-09-23-phase-6-graph-rag.md` (Graph RAG, merged as PR #40)

**Successor:** Phase 7b, the terminal experience, planned separately. v0.4.0 is tagged only after
both land.

---

## Why this shape, and what was rejected

Recorded here because the reasoning is the deliverable, not just the code.

### The evidence that the router is needed

Two real runs against local models, both in `docs/learning/`:

| Run | What the model did | What it shows |
|---|---|---|
| `agentic-first-run.md`, `llama3.1:8b` | Sent a paraphrase sub-question, with no identifier in it, to `lexical_search` | An 8B planner chooses tools poorly a meaningful fraction of the time |
| `graph-extraction-first-run.md` | Extraction and the citation contract held, but the graph is sparse on a small corpus | Graph must be chosen only when the question is about relationships, with an honest fallback when it has nothing |

And one gap found in the code on 2026-09-30: the agent's tools are `semantic_search`,
`lexical_search` and `fetch_document` only. The agent shipped in Phase 5, before Graph RAG existed
in Phase 6, so no sub-question can ever reach the graph, and `switch_strategy` only swaps semantic
and lexical.

### Router as a strategy, not a layer (to be ADR 0013)

| | **`auto` as a strategy (chosen)** | Router as a separate layer | Agent as the router |
|---|---|---|---|
| Entry points | API, CLI and SDK accept `strategy: auto` with no new wiring | Every entry point wired and kept in sync separately | None new |
| Result shape | One `RetrievalResult`, ADR 0002 unchanged | A second wrapper shape | One |
| Cost on a simple lookup | Zero model calls when signals are decisive | Same | Several model calls on every question |

The agent-as-router option defeats the reason the router exists: most questions do not need an
agent, and the router's job is to spend the agent's cost only where it pays.

### Signals override the planner only when decisive (to be ADR 0014)

Three ways to combine the router with the planner were considered.

| Option | Why not chosen, or chosen |
|---|---|
| Signals as a hint in the plan prompt | Rejected. A weak model can ignore a hint, which is exactly the failure the first run recorded |
| The router decides every tool, the planner only decomposes | Rejected. Up to one extra model call per sub-question, on every agentic run |
| **The planner proposes, decisive signals override, the override is traced** | **Chosen.** No extra model call. The worst case is the planner's own choice, which is today's behaviour |

### A signal carries no confidence number

A signal is a rule that fired, not a measurement. Giving it a confidence of 0.9 would be a number
nobody measured, which ADR 0004 forbids. So a signals decision carries `decisive: true` and
`confidence: null`. Only the classifier reports a confidence, and it is recorded as what the model
said, **uncalibrated until Phase 8** measures it against the evaluation set.

### One change from `docs/routing.md`

`docs/routing.md` says "agentic exhausts its budget with insufficient evidence, run Traditional".
This phase falls back **only when the agent returns zero usable evidence**. An agent that ran out
of budget with some sub-questions answered keeps them, and already reports the unanswered ones as
open with a reason. Discarding answered evidence to run a simpler search would make the answer
worse. `docs/routing.md` is rewritten to match in Task 13.

---

## Design

### Units

| Unit | File | What it does | Depends on |
|---|---|---|---|
| Signals | `router/signals.py` | Pure function: question in, `Signals` out. Then `propose(signals) -> Proposal(strategy, decisive, reasons)` | Nothing. No I/O, no model |
| Classifier | `router/classifier.py` | One model call under a JSON contract: `query_type`, strategy, confidence, one-sentence reasoning | `LLMProvider`, `json_contract.py` |
| Decision | `router/decision.py` | `RouterDecision`, the fields in `docs/routing.md` plus `source` | Nothing |
| Fallback policy | `router/fallback.py` | The fallback table as data | `StrategyName` |
| Auto strategy | `strategies/auto.py` | Route, run, fall back at most once, record | The registry |
| Graph tool | `agent/tools.py` | `GraphSearchTool` over `GraphRAGStrategy` | `GraphRAGStrategy` |
| Tool check | `agent/nodes.py` | `propose()` per sub-question between `plan` and `retrieve` | `router/signals.py` |

### Signals

| Signal | Detected by | Points to |
|---|---|---|
| Identifier | `identifiers()` in `stores/boosting.py`, the same function the Phase 4 identifier boost uses, so there is one definition (underscored names such as `ERR_QUOTA_4419`, mixed letters and digits, camelCase, versions such as `v2.3.1`) | Vectorless |
| Quoted phrase | `phrases()` in `stores/boosting.py`, straight or curly double quotes | Vectorless |
| Relational phrase | A verb phrase mapped to a configured relation type (`reports to` to `REPORTS_TO`, `owns` to `OWNS`), with two or more entity mentions | Graph |
| Comparison or aggregation | `compare`, `difference between`, `versus`, `how many`, `total`, with two or more entity mentions or constraints | Agentic |
| Plain | Short, one concept, none of the above | Traditional |

**Decisive** means exactly one strategy's signal fired, or a plain short question fired nothing.
Conflicting signals, or a long or vague question, are not decisive and go to the classifier.

### Decision

```json
{
  "selected_strategy": "graph",
  "source": "signals",
  "decisive": true,
  "confidence": null,
  "reasoning": "The question asks who reports to whom, which is a relationship between people.",
  "query_type": "relationship",
  "estimated_complexity": "medium",
  "expected_cost_level": "medium",
  "expected_latency_level": "medium",
  "fused": false
}
```

`source` is one of `signals`, `classifier`, `signals_fallback` (the classifier failed and the
signals' own proposal was used). `reasoning` is one sentence of at most 200 characters, from a
fixed template for signals and from the classifier otherwise.

### Flow for `strategy: auto`

| Step | What happens | Model calls |
|---|---|---|
| 1 | `extract_signals()` then `propose()` | 0 |
| 2 | Decisive: decision made, `source: signals` | 0 |
| 3 | Not decisive: classifier, `source: classifier` | 1 |
| 4 | Classifier confidence below `router.min_confidence`: run Traditional and Vectorless and fuse with the existing `rrf()` in `stores/fusion.py`, `fused: true` | 0 extra |
| 5 | Run the chosen strategy with the caller's own `RetrievalContext` | per strategy |
| 6 | Empty: at most one fallback, per the table below | per strategy |
| 7 | Record `RetrievalResult.router`, `fallback_from`, and the four `retrieval_runs` columns. Counters summed across every run made; every trace kept | 0 |

### Fallbacks: one step, never a chain

| Trigger | Action |
|---|---|
| Graph returns `no_graph_coverage`, `no_entity_matched` or `no_walkable_edges` | Traditional |
| Vectorless finds no term match | Traditional |
| Agentic returns zero usable evidence | Traditional |
| Traditional was chosen and found nothing | None. An honest empty result |
| The fallback also finds nothing | Stop. Empty, with both attempts in the trace |

### The agent

| Step | Change |
|---|---|
| `plan` | Unchanged. The model decomposes and proposes a tool per sub-question |
| Tool check (new) | `propose()` per sub-question. Decisive and different from the planner: the signals win, and `tool_override{from, to, reason}` is traced |
| `retrieve` | Can call `graph_search` |
| `repair`, `switch_strategy` | Chooses among all three search tools, never one already tried on that sub-question, the signals' next best first |
| Generation | When pooled evidence includes graph edges, the graph citation contract (ADR 0012) applies to relationship claims, on top of the unchanged Phase 3 contract |

`graph_search` returns the chunks that back each edge, pooled by chunk id like every other tool,
and the edges, merged into `RetrievalResult.subgraph`.

### Shared types

| Type | Change |
|---|---|
| `StrategyName` | Gains `AUTO = "auto"` |
| `RetrievalResult` | Gains `router: RouterDecision \| None`. `strategy` stays the strategy that actually ran, never `auto` |
| Request schemas, CLI, SDK | `strategy` defaults to unset. The server resolves it from `router.mode`: `auto` gives `auto`, `manual` gives `traditional`. Naming a strategy always bypasses the router. A changed SDK default, so it goes in the CHANGELOG |

### Errors and safety

| Situation | Behaviour |
|---|---|
| Classifier raises or times out | `source: signals_fallback`, the signals' proposal is used. Routing trouble never fails a request |
| Classifier returns malformed JSON | A typed violation, treated as below `min_confidence`, so step 4 fuses |
| `graph_store.enabled: false` | The router never selects Graph and the agent is not given `graph_search`. A graph signal resolves to Traditional with the reason `graph disabled` |
| Access | `auto`, every fallback and `graph_search` pass the caller's own `RetrievalContext` unchanged. Nothing rebuilds one |
| What the classifier sees | The question and the signals only, **never chunk text**, so its reasoning cannot disclose content from any document |
| Cost | The classifier call counts in `llm_calls`, tokens and cost. The router has its own `TraceSpan` |
| Guards | `auto` is never a routing target, never a fallback target and never an agent tool |

### Configuration

`RouterConfig` already exists and nothing reads it. After this phase every key is read:

| Key | Read by |
|---|---|
| `router.mode` | Default strategy resolution |
| `router.min_confidence` | Step 4 |
| `router.classifier_model` | The classifier; `null` uses `llm.model` |

A test fails if any `RouterConfig` field is not read, because Phase 5 shipped four typed limits that
nothing enforced while `ragfabric config validate` echoed them back.

### Out of scope, on purpose

| Item | Where it goes |
|---|---|
| Calibrating classifier confidence and thresholds | Phase 8, against labelled data |
| The full terminal experience, rich `ask` output included | Phase 7b. This phase prints one line: the strategy that ran and why |
| Router card in the assistant UI | Phase 9 |
| Recovering partial credit from a graph extraction response | Noted by the Phase 6 run, not a routing change |

### Testing

| Area | Tests |
|---|---|
| Signals | Table-driven, one fixture per query type, plus conflicting and empty cases |
| Classifier | Scripted JSON doubles: valid, malformed, low confidence, provider error |
| Auto | Fake strategies driving every fallback row, the fuse, the guards, counters summed |
| Agent | Override applied and traced; not applied when not decisive; `switch_strategy` over three tools; graph evidence and the ADR 0012 contract |
| PostgreSQL | `graph_search` access test, and a leak test that runs a denied document through the whole `auto` path including fallbacks, with `RAGFABRIC_TEST_DATABASE_URL` set |
| Config | Every `RouterConfig` field is read |
| Real run | `llama3.1:8b` on the Phase 5 question and a graph question, written up in `docs/learning/routing-first-run.md`. One run, not a benchmark |

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.13**, line length 100, `ruff` clean (`E`, `F`, `I`, `UP`, `B`), `ruff format` clean.
- **Run `ruff format packages/`, never `ruff format .`**
- **import-linter contracts stay at 3 kept, 0 broken.**
- **No new runtime dependencies.** If a task appears to need one, stop and say so.
- **ADR 0003 holds.** Every path, fallbacks included, runs with the caller's own filter.
- **ADR 0004 holds.** A signal has no confidence number. A classifier confidence is what the model
  reported, labelled uncalibrated.
- **ADR 0002 holds.** `auto` returns `RetrievalResult`.
- **SQLite remains the test default.** PostgreSQL-only tests are skipped via the
  `RAGFABRIC_TEST_DATABASE_URL` guard, never by silently passing. The baseline is measured on this
  branch before Task 1 with that variable set; the last recorded figure, at the Phase 6 merge, was
  1238 passed and 9 skipped.
- **No AI attribution anywhere.** Commits authored `Ranjan G <ranjan.g@ispf.ngo>`, no trailers, no
  assistant references in code, comments, docs or PR bodies.
- **No em dashes** anywhere.
- **Tests must be able to fail.**
- **One implementer at a time in a worktree.**
- **Check `pg_isready` first when agents stall.** OrbStack's engine wedged after sleep in Phase 6.

---

## Tasks

To be written once this design is approved.
