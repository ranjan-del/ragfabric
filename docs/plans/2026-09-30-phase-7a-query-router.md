# Phase 7a: Query Router and the agent's reach to all four strategies. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

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
| Relational phrase | A verb phrase mapped to a configured relation type (`reports to` to `REPORTS_TO`, `owns` to `OWNS`), with at least one entity mention ("Who does Ravi Sharma report to?" names one) | Graph |
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
and the edges, merged into `RetrievalResult.subgraph`. `GraphRAGStrategy.retrieve` makes one model
call to match the question's entities, so every `graph_search` call is charged to the agent's
budget through `state.spend` like any node, and a refused spend stops the run with `budget`.

**Graph is unavailable to a request that sets `document_id` or `format`.** The graph walk applies
access and collection scope only, and `/api/ask` already refuses such a request with a 422 when the
caller names `graph`. Under `auto` the router leaves Graph out of the candidates, and the agent
leaves `graph_search` out of that run's tools, rather than letting a walk return chunks from
documents the caller asked to exclude.

**A fallback result carries none of the failed attempt's `sub_questions` or `subgraph`.**
Generation chooses its path from those fields (`api/routes/search.py:_generate`), so a Traditional
fallback that kept an agent's reports would be answered on the agentic path. The failed attempt is
kept in the trace instead.

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

## Review Focus

Inputs and conditions the design implies but a happy-path test would miss, most likely first. Each
line has a test in the task that owns the code.

1. **A request with `document_id` or `format` under `auto`.** Expected: Graph is never selected and
   never used as a fallback, and the agent is not given `graph_search` for that run (Tasks 5, 7).
2. **An `auto` fallback after the agent.** Expected: the Traditional result carries no
   `sub_questions` and no `subgraph`, so the answer is generated on the Traditional path (Task 4).
3. **A caller budget of zero model calls under `auto`.** Expected: the classifier is not called,
   the decision is `signals_fallback`, and the chosen strategy receives a budget with the
   classifier's call already deducted when one was made (Task 5).
4. **A classifier that names a strategy this deployment cannot serve** (Graph disabled, or Graph
   unavailable under a filter). Expected: a contract violation, treated as low confidence, so the
   request is fused rather than routed to an unavailable strategy (Task 3).
5. **A question that is only an identifier, a quoted phrase, or empty after trimming.** Expected:
   the first two are decisive Vectorless; an empty question is refused by the request schema before
   routing and, if one reaches `extract_signals`, it proposes Traditional without raising (Task 2).

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/core/src/ragfabric_core/router/__init__.py` | Empty on purpose, so importing `router.decision` from `strategies/base.py` cannot pull `router.signals` into an import cycle |
| `packages/core/src/ragfabric_core/router/decision.py` | `RouterDecision`, `QueryType`, `Level`, the per strategy engineering levels |
| `packages/core/src/ragfabric_core/router/signals.py` | `Signals`, `extract_signals`, `Proposal`, `propose`, `TOOL_FOR_STRATEGY` |
| `packages/core/src/ragfabric_core/router/classifier.py` | `ClassifierReply`, `ClassifierOutcome`, `classify`, `one_sentence` |
| `packages/core/src/ragfabric_core/router/fallback.py` | `fallback_for`, `empty_reason`, `combine`, `fuse` |
| `packages/core/src/ragfabric_core/router/mode.py` | `resolve_requested` |
| `packages/core/src/ragfabric_core/strategies/auto.py` | `AutoStrategy` |
| `packages/core/src/ragfabric_core/strategies/base.py` | `StrategyName.AUTO`, `RetrievalResult.router` |
| `packages/core/src/ragfabric_core/strategies/contract.py` | Contract widened for `auto` only |
| `packages/core/src/ragfabric_core/strategies/registry_defaults.py` | Registers `auto` and `graph_search` |
| `packages/core/src/ragfabric_core/agent/tools.py` | `GraphSearchTool`, `GraphToolRun` |
| `packages/core/src/ragfabric_core/agent/nodes.py` | Graph runs in `retrieve`, `check_tools` after `plan` |
| `packages/core/src/ragfabric_core/agent/loop.py` | Tool check wired, `AgentRun.subgraph`, retrieve under the budget guard |
| `packages/core/src/ragfabric_core/agent/policy.py` | `switch_strategy` over three tools |
| `packages/core/src/ragfabric_core/strategies/agentic.py` | Per request tool set, subgraph on the result |
| `packages/server/src/ragfabric_server/...` | Schemas, default resolution, run row, graph evidence generation path |
| `packages/cli/src/ragfabric_cli/commands/ask.py`, `packages/sdk-python/...` | `auto`, unset default, one line of routing output |
| `docs/adr/0013-router-as-a-strategy.md`, `docs/adr/0014-signals-override-the-planner.md` | The two decisions |
| `docs/concepts/routing-as-classification.md`, `docs/routing.md`, `docs/agentic-rag.md` | Concepts and what shipped |
| `docs/learning/routing-first-run.md` | The real run |

Test commands run from the worktree root. `uv run pytest packages/core/tests/<file> -v` for one file.
The full suite with PostgreSQL:

```bash
export RAGFABRIC_TEST_DATABASE_URL="postgresql+psycopg://ragfabric:ragfabric@localhost:5432/ragfabric_test"
uv run pytest -q
```

---

## Task 1: Decision types, `auto` in the strategy vocabulary, and the widened contract

**Files:**
- Create: `packages/core/src/ragfabric_core/router/__init__.py`, `router/decision.py`
- Modify: `strategies/base.py` (`StrategyName`, `RetrievalResult`), `strategies/contract.py`
- Test: `packages/core/tests/test_router_decision.py`

Foundation. Both tracks compile against it, so the controller lands it before either starts.

**Interfaces:**
- Produces: `RouterDecision`, `QueryType`, `Level`, `ENGINEERING_LEVELS: dict[str, tuple[Level, Level, Level]]`,
  `StrategyName.AUTO`, `RetrievalResult.router: RouterDecision | None = None`.

- [ ] **Step 0: Measure the baseline** with PostgreSQL up (`pg_isready -h localhost`) and the variable set. Write the passed and skipped counts into the ledger; every later "no regressions" claim is against this number.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_router_decision.py
import pytest
from pydantic import ValidationError

from ragfabric_core.router.decision import ENGINEERING_LEVELS, RouterDecision
from ragfabric_core.strategies.base import RetrievalResult, StrategyName
from ragfabric_core.strategies.contract import assert_strategy_contract


def decision(**overrides) -> RouterDecision:
    fields = {
        "selected_strategy": "graph",
        "source": "signals",
        "decisive": True,
        "confidence": None,
        "reasoning": "The question asks who reports to whom.",
        "query_type": "relationship",
        "estimated_complexity": "medium",
        "expected_cost_level": "medium",
        "expected_latency_level": "medium",
    }
    return RouterDecision(**(fields | overrides))


def test_a_signals_decision_carries_no_confidence_number():
    assert decision().confidence is None


def test_auto_is_never_a_selected_strategy():
    with pytest.raises(ValidationError):
        decision(selected_strategy="auto")


def test_reasoning_longer_than_one_short_sentence_is_refused():
    with pytest.raises(ValidationError):
        decision(reasoning="x" * 201)


def test_every_servable_strategy_has_engineering_levels_and_auto_does_not():
    assert set(ENGINEERING_LEVELS) == {"traditional", "vectorless", "agentic", "graph"}


def test_auto_is_in_the_vocabulary():
    assert StrategyName("auto") is StrategyName.AUTO


def test_router_defaults_to_none_so_the_four_strategies_are_unaffected():
    result = RetrievalResult(
        strategy=StrategyName.TRADITIONAL, chunks=[], retrieval_calls=0, llm_calls=0,
        input_tokens=0, output_tokens=0, latency_ms=0,
    )
    assert result.router is None


class _Auto:
    name = StrategyName.AUTO

    def __init__(self, router):
        self._router = router

    def retrieve(self, query, ctx):
        return RetrievalResult(
            strategy=StrategyName.TRADITIONAL, chunks=[], retrieval_calls=0, llm_calls=0,
            input_tokens=0, output_tokens=0, latency_ms=0, router=self._router,
        )


def test_the_contract_accepts_auto_reporting_the_strategy_that_ran(make_ctx):
    assert_strategy_contract(_Auto(decision(selected_strategy="traditional")), "q", make_ctx())


def test_the_contract_refuses_auto_without_a_decision(make_ctx):
    with pytest.raises(AssertionError):
        assert_strategy_contract(_Auto(None), "q", make_ctx())
```

`make_ctx` is the helper in `test_strategy_interface.py`. Move it into `packages/core/tests/conftest.py` as a fixture returning the function, and update that file to use the fixture, in this task.

- [ ] **Step 2: Run them, confirm they fail** with `ModuleNotFoundError: ragfabric_core.router`.

Run: `uv run pytest packages/core/tests/test_router_decision.py -v`

- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/router/__init__.py
"""Query routing. Deliberately empty: see router/decision.py for why."""
```

```python
# packages/core/src/ragfabric_core/router/decision.py
"""What the router decided, and who decided it.

Imports nothing from the rest of core, because ``strategies/base.py`` imports
this module to type ``RetrievalResult.router``. Strategy names are therefore
spelled as literals here rather than taken from ``StrategyName``.

``confidence`` is ``None`` for a signals decision. A rule that fired is not a
measurement, and a number attached to it would be one nobody measured
(ADR 0004). A classifier confidence is what the model reported and is
uncalibrated until Phase 8 measures it against labelled questions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Servable = Literal["traditional", "vectorless", "agentic", "graph"]
QueryType = Literal[
    "simple_factual", "exact_match", "relationship", "multi_hop",
    "comparison", "aggregation", "compound", "ambiguous",
]
Level = Literal["low", "medium", "high"]
Source = Literal["signals", "classifier", "signals_fallback"]

# Engineering assessments of each strategy's shape, not measurements: one
# embedding call and one query is low, a model call per step is high.
# (complexity, cost, latency)
ENGINEERING_LEVELS: dict[str, tuple[Level, Level, Level]] = {
    "traditional": ("low", "low", "low"),
    "vectorless": ("low", "low", "low"),
    "graph": ("medium", "medium", "medium"),
    "agentic": ("high", "high", "high"),
}

MAX_REASONING_CHARS = 200


class RouterDecision(BaseModel):
    selected_strategy: Servable
    source: Source
    decisive: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=MAX_REASONING_CHARS)
    query_type: QueryType
    estimated_complexity: Level
    expected_cost_level: Level
    expected_latency_level: Level
    fused: bool = False


def levels_for(strategy: Servable) -> dict[str, Level]:
    complexity, cost, latency = ENGINEERING_LEVELS[strategy]
    return {
        "estimated_complexity": complexity,
        "expected_cost_level": cost,
        "expected_latency_level": latency,
    }
```

In `strategies/base.py`: add `AUTO = "auto"` to `StrategyName`, import `RouterDecision` from
`ragfabric_core.router.decision`, and add to `RetrievalResult` after `subgraph`:

```python
    # The routing decision, set only by the auto strategy. None for every
    # strategy a caller names directly, the same additive move as subgraph.
    router: RouterDecision | None = None
```

In `strategies/contract.py`, replace the `result.strategy == strategy.name` assertion with:

```python
    if strategy.name == StrategyName.AUTO:
        assert result.router is not None, "auto must say what it decided"
        assert result.strategy != StrategyName.AUTO, "auto must report the strategy that ran"
        assert str(result.strategy) == result.router.selected_strategy or result.fallback_from, (
            "auto reported a strategy it neither selected nor fell back to"
        )
    else:
        assert result.strategy == strategy.name, (
            f"strategy reported {result.strategy!r}, expected {strategy.name!r}"
        )
```

- [ ] **Step 4: Run the new tests and the whole core suite**, confirm the new tests pass and the baseline is otherwise unchanged. `StrategyName` gaining a member must not break any exhaustive match: search `packages/` for `StrategyName` and fix any that enumerate members.
- [ ] **Step 5: Commit** `feat: add the router decision type and the auto strategy name`

---

## Task 2: Signals

**Files:**
- Create: `packages/core/src/ragfabric_core/router/signals.py`
- Test: `packages/core/tests/test_router_signals.py`

Also landed by the controller before the tracks fork: Track A's router and Track B's agent check both call `propose()`.

**Interfaces:**
- Consumes: `identifiers()`, `phrases()` from `stores/boosting.py`; `StrategyName`.
- Produces:
  - `extract_signals(question: str, *, relation_types: Collection[str]) -> Signals`
  - `propose(signals: Signals, *, available: Collection[StrategyName]) -> Proposal`
  - `Proposal(strategy: StrategyName, decisive: bool, reasons: tuple[str, ...], query_type: QueryType, ranking: tuple[StrategyName, ...])`
  - `TOOL_FOR_STRATEGY = {TRADITIONAL: "semantic_search", VECTORLESS: "lexical_search", GRAPH: "graph_search"}`
  - `DEFAULT_RELATION_PHRASES: dict[str, tuple[str, ...]]`

- [ ] **Step 1: Write the failing tests**, table driven.

```python
# packages/core/tests/test_router_signals.py
import pytest

from ragfabric_core.router.signals import extract_signals, propose
from ragfabric_core.strategies.base import StrategyName as S

RELATIONS = ["REPORTS_TO", "MEMBER_OF", "BELONGS_TO", "OWNS", "WORKS_ON", "LOCATED_IN",
             "AUTHORED", "MENTIONS", "RELATED_TO"]
ALL = {S.TRADITIONAL, S.VECTORLESS, S.AGENTIC, S.GRAPH}


def run(question, available=ALL):
    return propose(extract_signals(question, relation_types=RELATIONS), available=available)


@pytest.mark.parametrize(
    ("question", "strategy", "query_type"),
    [
        ("What does ERR_QUOTA_4419 mean?", S.VECTORLESS, "exact_match"),
        ('Where is "annual leave carry forward" defined?', S.VECTORLESS, "exact_match"),
        ("ERR_QUOTA_4419", S.VECTORLESS, "exact_match"),
        ("Who does Ravi Sharma report to?", S.GRAPH, "relationship"),
        ("Which team owns Billing?", S.GRAPH, "relationship"),
        ("Compare the leave policy for Pune and Delhi", S.AGENTIC, "comparison"),
        ("How many offices does the company have?", S.AGENTIC, "aggregation"),
        ("What is our refund policy?", S.TRADITIONAL, "simple_factual"),
    ],
)
def test_a_single_signal_is_decisive(question, strategy, query_type):
    proposal = run(question)
    assert (proposal.strategy, proposal.decisive, proposal.query_type) == (
        strategy, True, query_type,
    )


def test_conflicting_signals_are_not_decisive():
    proposal = run("Compare ERR_QUOTA_4419 with ERR_QUOTA_4420 for Billing")
    assert not proposal.decisive


def test_a_long_question_with_no_signal_is_not_decisive():
    question = " ".join(["policy"] * 20) + "?"
    assert not run(question).decisive


def test_a_relation_phrase_without_an_entity_is_not_a_graph_signal():
    assert run("who do people report to?").strategy is not S.GRAPH


def test_an_unconfigured_relation_type_is_not_a_graph_signal():
    signals = extract_signals("Who does Ravi Sharma report to?", relation_types=["OWNS"])
    assert propose(signals, available=ALL).strategy is S.TRADITIONAL


def test_graph_unavailable_resolves_to_traditional_and_says_why():
    proposal = run("Who does Ravi Sharma report to?", available=ALL - {S.GRAPH})
    assert proposal.strategy is S.TRADITIONAL
    assert any("graph" in reason for reason in proposal.reasons)
    assert S.GRAPH not in proposal.ranking


def test_the_ranking_puts_the_proposal_first_and_lists_every_available_strategy_once():
    proposal = run("What does ERR_QUOTA_4419 mean?")
    assert proposal.ranking[0] is S.VECTORLESS
    assert sorted(proposal.ranking) == sorted(ALL)


@pytest.mark.parametrize("question", ["", "   ", "?"])
def test_an_empty_question_proposes_traditional_without_raising(question):
    assert run(question).strategy is S.TRADITIONAL


def test_extraction_is_pure():
    first = extract_signals("Who owns Billing?", relation_types=RELATIONS)
    second = extract_signals("Who owns Billing?", relation_types=RELATIONS)
    assert first == second
```

- [ ] **Step 2: Run them, confirm they fail** with `ModuleNotFoundError`.
- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/router/signals.py
"""Cheap, deterministic features of a question, and the strategy they point to.

No I/O and no model. Everything here is a rule that either fired or did not,
which is why a proposal says ``decisive`` and never carries a confidence
number (ADR 0004).
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from ragfabric_core.router.decision import QueryType
from ragfabric_core.stores.boosting import identifiers, is_identifier, phrases
from ragfabric_core.strategies.base import StrategyName

PLAIN_MAX_WORDS = 15

# Relation types a question can name in plain words. MENTIONS and RELATED_TO
# are left out on purpose: "related to" appears in questions that are not
# about the graph at all, and a signal that fires on everything is noise.
DEFAULT_RELATION_PHRASES: dict[str, tuple[str, ...]] = {
    "REPORTS_TO": ("report to", "reports to", "reporting to", "manager of", "managed by"),
    "MEMBER_OF": ("member of", "members of", "part of the team", "on the team"),
    "BELONGS_TO": ("belong to", "belongs to"),
    "OWNS": ("own", "owns", "owner of", "owned by"),
    "WORKS_ON": ("work on", "works on", "working on"),
    "LOCATED_IN": ("located in", "based in", "based out of"),
    "AUTHORED": ("wrote", "author of", "authored"),
}
_COMPARISON = ("compare", "comparison", "difference between", "differences between",
               "versus", " vs ", " vs.", "differ from")
_AGGREGATION = ("how many", "total", "sum of", "count of", "list all", "list every")
_NOT_ENTITIES = frozenset(
    "who what which when where why how does do did is are was were can could should "
    "would will the a an list compare show tell give find i we our my".split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

TOOL_FOR_STRATEGY: dict[StrategyName, str] = {
    StrategyName.TRADITIONAL: "semantic_search",
    StrategyName.VECTORLESS: "lexical_search",
    StrategyName.GRAPH: "graph_search",
}
_DEFAULT_ORDER = (StrategyName.TRADITIONAL, StrategyName.VECTORLESS,
                  StrategyName.GRAPH, StrategyName.AGENTIC)


@dataclass(frozen=True)
class Signals:
    identifiers: tuple[str, ...]
    phrases: tuple[str, ...]
    entities: tuple[str, ...]
    relations: tuple[str, ...]
    comparison: bool
    aggregation: bool
    compound: bool
    words: int


@dataclass(frozen=True)
class Proposal:
    strategy: StrategyName
    decisive: bool
    reasons: tuple[str, ...]
    query_type: QueryType
    ranking: tuple[StrategyName, ...]


def entity_mentions(question: str) -> tuple[str, ...]:
    """Runs of capitalised words that are not question words or identifiers."""
    mentions: list[str] = []
    run: list[str] = []
    for word in _WORD.findall(question):
        if word[0].isupper() and word.lower() not in _NOT_ENTITIES and not is_identifier(word):
            run.append(word)
            continue
        if run:
            mentions.append(" ".join(run))
            run = []
    if run:
        mentions.append(" ".join(run))
    return tuple(mentions)


def extract_signals(question: str, *, relation_types: Collection[str]) -> Signals:
    text = f" {' '.join(question.lower().split())} "
    configured = {name.upper() for name in relation_types}
    relations = tuple(
        sorted(
            name for name, spoken in DEFAULT_RELATION_PHRASES.items()
            if name in configured and any(f" {phrase} " in text for phrase in spoken)
        )
    )
    return Signals(
        identifiers=tuple(sorted(identifiers(question))),
        phrases=tuple(phrases(question)),
        entities=entity_mentions(question),
        relations=relations,
        comparison=any(marker in text for marker in _COMPARISON),
        aggregation=any(marker in text for marker in _AGGREGATION),
        compound=question.count("?") > 1,
        words=len(question.split()),
    )


def _fired(signals: Signals) -> list[tuple[StrategyName, QueryType, str]]:
    fired: list[tuple[StrategyName, QueryType, str]] = []
    if signals.identifiers or signals.phrases:
        fired.append((StrategyName.VECTORLESS, "exact_match",
                      "The question names an exact identifier or quoted phrase."))
    if signals.relations and signals.entities:
        fired.append((StrategyName.GRAPH, "relationship",
                      "The question asks how named things are related."))
    if signals.comparison:
        fired.append((StrategyName.AGENTIC, "comparison",
                      "The question compares several things."))
    elif signals.aggregation:
        fired.append((StrategyName.AGENTIC, "aggregation",
                      "The question asks for a count or a total across sources."))
    elif signals.compound:
        fired.append((StrategyName.AGENTIC, "compound",
                      "The question asks several things at once."))
    return fired


def propose(signals: Signals, *, available: Collection[StrategyName]) -> Proposal:
    fired = _fired(signals)
    reasons: list[str] = []
    usable = [item for item in fired if item[0] in available]
    if len(usable) < len(fired):
        reasons.append("The graph is not available for this request, so it was not considered.")

    if len(usable) == 1:
        strategy, query_type, reason = usable[0]
        decisive = True
        reasons.insert(0, reason)
    elif not usable and (fired or signals.words <= PLAIN_MAX_WORDS):
        strategy, query_type, decisive = StrategyName.TRADITIONAL, "simple_factual", True
        reasons.insert(0, "A short question about one concept.")
    else:
        # Conflicting signals or a long question: the classifier decides. The
        # provisional choice is the most capable strategy that fired, for the
        # case where the classifier cannot be reached.
        strategy = next((s for s, _, _ in usable if s is StrategyName.AGENTIC),
                        usable[0][0] if usable else StrategyName.TRADITIONAL)
        query_type, decisive = "ambiguous", False
        reasons.insert(0, "The signals disagree or the question is long and open.")

    ranking = [strategy] + [s for s, _, _ in usable if s is not strategy]
    ranking += [s for s in _DEFAULT_ORDER if s in available and s not in ranking]
    return Proposal(strategy=strategy, decisive=decisive, reasons=tuple(reasons),
                    query_type=query_type, ranking=tuple(ranking))
```

Check `"Which team owns Billing?"`: "Which" is a question word, "Billing" is an entity, "owns" hits OWNS. Check `"Who does Ravi Sharma report to?"`: entity "Ravi Sharma", phrase "report to". If a table row fails, fix the rule, not the fixture, and write why in the ledger.

- [ ] **Step 4: Run tests, confirm they pass.** Then break `_fired` on purpose (drop the `signals.entities` condition), confirm `test_a_relation_phrase_without_an_entity_is_not_a_graph_signal` fails, and restore it.
- [ ] **Step 5: Commit** `feat: add deterministic routing signals and proposals`

---

## Task 3: The classifier

**Files:**
- Create: `packages/core/src/ragfabric_core/router/classifier.py`
- Test: `packages/core/tests/test_router_classifier.py`

**Interfaces:**
- Consumes: `Signals`, `Proposal` (Task 2); `parse_contract`, `ContractViolation` (`json_contract.py`); `LLMProvider`, `Message`.
- Produces: `classify(question, signals, *, llm, available, model=None) -> ClassifierOutcome`;
  `ClassifierOutcome(reply: ClassifierReply | None, violation: ContractViolation | None, error: str | None, input_tokens: int, output_tokens: int, provider: str, model: str, llm_calls: int, latency_ms: int)`;
  `one_sentence(text: str, limit: int = 200) -> str`.

- [ ] **Step 1: Write the failing tests** with `RecordingLLM` from `tests/agent_doubles.py`.

```python
# packages/core/tests/test_router_classifier.py
import json

from agent_doubles import RecordingLLM

from ragfabric_core.router.classifier import classify, one_sentence
from ragfabric_core.router.signals import extract_signals
from ragfabric_core.strategies.base import StrategyName as S

ALL = [S.TRADITIONAL, S.VECTORLESS, S.AGENTIC, S.GRAPH]


def reply(**fields):
    body = {"query_type": "multi_hop", "strategy": "agentic", "confidence": 0.7,
            "reasoning": "It needs facts from two documents."} | fields
    return json.dumps(body)


def signals(q="Tell me everything about how onboarding and payroll interact for contractors"):
    return extract_signals(q, relation_types=["OWNS"])


def test_a_valid_reply_is_returned_with_its_real_token_counts():
    llm = RecordingLLM(reply())
    outcome = classify("q", signals(), llm=llm, available=ALL)
    assert outcome.reply.strategy == "agentic" and outcome.llm_calls == 1
    assert outcome.provider == llm.name


def test_the_prompt_carries_the_question_and_signals_and_no_document_text():
    llm = RecordingLLM(reply())
    classify("What is onboarding?", signals(), llm=llm, available=ALL)
    prompt = "\n".join(m.content for m in llm.prompts[0])
    assert "What is onboarding?" in prompt and "chunk" not in prompt.lower()


def test_malformed_json_is_a_violation_not_an_exception():
    outcome = classify("q", signals(), llm=RecordingLLM("I think agentic"), available=ALL)
    assert outcome.reply is None and outcome.violation is not None and outcome.llm_calls == 1


def test_a_strategy_this_request_cannot_serve_is_a_violation():
    outcome = classify("q", signals(), llm=RecordingLLM(reply(strategy="graph")),
                       available=[S.TRADITIONAL, S.VECTORLESS, S.AGENTIC])
    assert outcome.reply is None and "graph" in outcome.violation.error


def test_a_provider_error_is_reported_not_raised():
    class Broken(RecordingLLM):
        def complete(self, *args, **kwargs):
            raise TimeoutError("model did not answer")

    outcome = classify("q", signals(), llm=Broken(), available=ALL)
    assert outcome.error and outcome.reply is None and outcome.llm_calls == 1


def test_one_sentence_keeps_the_first_sentence_and_the_length_cap():
    assert one_sentence("First part. Second part.") == "First part."
    assert len(one_sentence("x" * 500)) <= 200
```

`llm_calls` is 1 on the provider error path because the call was made and failed; ADR 0004 counts what happened.

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/router/classifier.py
"""The router's second stage: one model call, only when the signals are not decisive.

The model sees the question and the signals and nothing else. It never sees a
chunk, so the one sentence of reasoning it returns cannot repeat anything from
a document, including one the caller is not allowed to read.
"""

from __future__ import annotations

import re
import time
from collections.abc import Collection
from dataclasses import asdict

from pydantic import BaseModel, Field

from ragfabric_core.json_contract import ContractViolation, parse_contract
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.router.decision import MAX_REASONING_CHARS, QueryType, Servable
from ragfabric_core.router.signals import Signals
from ragfabric_core.strategies.base import StrategyName

CONTRACT = "router_classifier"
SYSTEM = (
    "You route questions to a retrieval strategy. "
    "You reply with one JSON object and nothing else."
)
GUIDE = """Strategies:
- traditional: meaning based search. Simple factual questions, paraphrases.
- vectorless: exact word search. Identifiers, codes, quoted phrases, names.
- graph: relationships between named people, teams, projects and places.
- agentic: comparisons, several constraints, several documents, vague questions.

Reply as {"query_type": one of simple_factual, exact_match, relationship, multi_hop,
comparison, aggregation, compound, ambiguous, "strategy": one of the allowed strategies,
"confidence": a number from 0 to 1, "reasoning": one short sentence for the user}."""
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


class ClassifierReply(BaseModel):
    query_type: QueryType
    strategy: Servable
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class ClassifierOutcome(BaseModel):
    reply: ClassifierReply | None = None
    violation: ContractViolation | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""
    llm_calls: int = 0
    latency_ms: int = 0


def one_sentence(text: str, limit: int = MAX_REASONING_CHARS) -> str:
    first = _SENTENCE_END.split(" ".join(text.split()), maxsplit=1)[0]
    return first if len(first) <= limit else first[: limit - 3].rstrip() + "..."


def build_prompt(question: str, signals: Signals, available: Collection[StrategyName]) -> str:
    allowed = ", ".join(str(name) for name in available if name is not StrategyName.AUTO)
    return (f"Question: {question}\n\nSignals: {asdict(signals)}\n\n"
            f"Allowed strategies: {allowed}\n\n{GUIDE}")


def classify(
    question: str,
    signals: Signals,
    *,
    llm: LLMProvider,
    available: Collection[StrategyName],
    model: str | None = None,
) -> ClassifierOutcome:
    started = time.perf_counter()
    messages = [Message(role="system", content=SYSTEM),
                Message(role="user", content=build_prompt(question, signals, available))]
    try:
        completion = llm.complete(messages, model=model, max_tokens=200, temperature=0.0)
    except Exception as exc:  # noqa: BLE001  routing trouble never fails a request
        return ClassifierOutcome(error=f"{type(exc).__name__}: {exc}", llm_calls=1,
                                 provider=llm.name, model=model or llm.default_model,
                                 latency_ms=int((time.perf_counter() - started) * 1000))
    common = {
        "input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens,
        "provider": completion.provider, "model": completion.model, "llm_calls": 1,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
    parsed = parse_contract(completion.text, CONTRACT, ClassifierReply)
    if isinstance(parsed, ContractViolation):
        return ClassifierOutcome(violation=parsed, **common)
    if StrategyName(parsed.strategy) not in available:
        violation = ContractViolation(
            contract=CONTRACT, raw=completion.text,
            error=f"strategy {parsed.strategy} is not available for this request",
        )
        return ClassifierOutcome(violation=violation, **common)
    parsed.reasoning = one_sentence(parsed.reasoning)
    return ClassifierOutcome(reply=parsed, **common)
```

Confirm `RecordingLLM`'s `Completion` carries `provider`; the test asserts it. If `ruff` rejects the blanket `except`, keep the `noqa` with its reason: a provider can raise anything its SDK raises, and the design says routing trouble never fails a request.

- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add the router classifier with a validated JSON contract`

---

## Task 4: Fallback, combination and fusion

**Files:**
- Create: `packages/core/src/ragfabric_core/router/fallback.py`
- Test: `packages/core/tests/test_router_fallback.py`

**Interfaces:**
- Consumes: `RetrievalResult`, `StrategyName`, `rrf()` from `stores/fusion.py`.
- Produces: `fallback_for(chosen: StrategyName, result: RetrievalResult) -> StrategyName | None`;
  `empty_reason(result: RetrievalResult) -> str`;
  `combine(first: RetrievalResult, second: RetrievalResult, *, fallback_from: StrategyName) -> RetrievalResult`;
  `fuse(traditional: RetrievalResult, vectorless: RetrievalResult, *, top_k: int) -> RetrievalResult`.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_router_fallback.py
import pytest
from agent_doubles import chunk

from ragfabric_core.graph.contracts import EmptyReason, Subgraph
from ragfabric_core.router.fallback import combine, empty_reason, fallback_for, fuse
from ragfabric_core.strategies.base import RetrievalResult, StrategyName as S, SubQuestionReport


def result(strategy, chunks=(), **extra):
    fields = {"retrieval_calls": 1, "llm_calls": 1, "embedding_calls": 1, "input_tokens": 10,
              "output_tokens": 5, "latency_ms": 100}
    return RetrievalResult(strategy=strategy, chunks=list(chunks), **(fields | extra))


@pytest.mark.parametrize("chosen", [S.GRAPH, S.VECTORLESS, S.AGENTIC])
def test_an_empty_result_falls_back_to_traditional(chosen):
    assert fallback_for(chosen, result(chosen)) is S.TRADITIONAL


def test_traditional_never_falls_back():
    assert fallback_for(S.TRADITIONAL, result(S.TRADITIONAL)) is None


def test_a_result_with_evidence_never_falls_back():
    assert fallback_for(S.AGENTIC, result(S.AGENTIC, [chunk(1)])) is None


def test_the_graph_reason_is_reported_when_there_is_one():
    empty = Subgraph(nodes=[], edges=[], truncated=False,
                     empty_reason=EmptyReason.NO_ENTITY_MATCHED)
    assert empty_reason(result(S.GRAPH, subgraph=empty)) == "no_entity_matched"


def test_combine_sums_every_counter_and_keeps_both_traces():
    merged = combine(result(S.AGENTIC), result(S.TRADITIONAL, [chunk(1)]), fallback_from=S.AGENTIC)
    assert (merged.retrieval_calls, merged.llm_calls, merged.embedding_calls) == (2, 2, 2)
    assert (merged.input_tokens, merged.output_tokens, merged.latency_ms) == (20, 10, 200)
    assert merged.strategy is S.TRADITIONAL and merged.fallback_from is S.AGENTIC


def test_combine_drops_the_failed_attempts_sub_questions_and_subgraph():
    failed = result(S.AGENTIC, sub_questions=[SubQuestionReport(text="a", status="open",
                                                                 reason="budget")])
    merged = combine(failed, result(S.TRADITIONAL, [chunk(1)]), fallback_from=S.AGENTIC)
    assert merged.sub_questions == [] and merged.subgraph is None


def test_fuse_only_reorders_chunks_the_strategies_returned():
    fused = fuse(result(S.TRADITIONAL, [chunk(1), chunk(2)]),
                 result(S.VECTORLESS, [chunk(2), chunk(3)]), top_k=5)
    assert {c.chunk_id for c in fused.chunks} == {1, 2, 3}
    assert fused.chunks[0].chunk_id == 2
    assert fused.strategy is S.TRADITIONAL


def test_fuse_cuts_to_top_k_after_fusion():
    fused = fuse(result(S.TRADITIONAL, [chunk(i) for i in range(10)]),
                 result(S.VECTORLESS, []), top_k=3)
    assert len(fused.chunks) == 3
```

Check the real import location of `EmptyReason` and `Subgraph` (`graph/contracts.py`) and the `EmptyReason` member names before running; adjust the import, not the assertion.

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/router/fallback.py
"""What auto does when the strategy it chose finds nothing.

One step, never a chain. A fallback that found nothing either is reported as
empty, with both attempts in the trace, rather than trying a third strategy
the caller never asked for.
"""

from __future__ import annotations

from ragfabric_core.stores.fusion import rrf
from ragfabric_core.strategies.base import RetrievalResult, StrategyName

NO_EVIDENCE = "no evidence"
_COUNTERS = ("retrieval_calls", "llm_calls", "embedding_calls", "input_tokens",
             "output_tokens", "latency_ms")


def fallback_for(chosen: StrategyName, result: RetrievalResult) -> StrategyName | None:
    if result.chunks or chosen is StrategyName.TRADITIONAL:
        return None
    return StrategyName.TRADITIONAL


def empty_reason(result: RetrievalResult) -> str:
    if result.subgraph is not None and result.subgraph.empty_reason is not None:
        return result.subgraph.empty_reason.value
    return NO_EVIDENCE


def _summed(first: RetrievalResult, second: RetrievalResult) -> dict[str, int]:
    return {name: getattr(first, name) + getattr(second, name) for name in _COUNTERS}


def combine(
    first: RetrievalResult, second: RetrievalResult, *, fallback_from: StrategyName
) -> RetrievalResult:
    """The fallback's evidence, with what both attempts cost.

    ``sub_questions`` and ``subgraph`` come from the fallback only. Generation
    picks its path from those fields, so keeping the failed agent's reports
    would answer a Traditional result on the agentic path.
    """
    return second.model_copy(update={
        **_summed(first, second),
        "trace": [*first.trace, *second.trace],
        "fallback_from": fallback_from,
    })


def fuse(
    traditional: RetrievalResult, vectorless: RetrievalResult, *, top_k: int
) -> RetrievalResult:
    """Traditional and Vectorless fused with RRF (ADR 0008), cut to top_k after fusion."""
    chunks = rrf([traditional.chunks, vectorless.chunks])[:top_k]
    return traditional.model_copy(update={
        **_summed(traditional, vectorless),
        "chunks": chunks,
        "trace": [*traditional.trace, *vectorless.trace],
    })
```

- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add the one-step fallback and the low-confidence fusion`

---

## Task 5: `AutoStrategy`

**Files:**
- Create: `packages/core/src/ragfabric_core/strategies/auto.py`
- Test: `packages/core/tests/test_strategy_auto.py`

**Interfaces:**
- Consumes: Tasks 1 to 4; `StrategyRegistry`.
- Produces: `AutoStrategy(*, registry: StrategyRegistry, llm: LLMProvider | None, min_confidence: float, classifier_model: str | None, graph_enabled: bool, relation_types: Sequence[str])`
  with read-only properties `min_confidence`, `classifier_model`, `graph_enabled`, and `available(ctx) -> list[StrategyName]`.

- [ ] **Step 1: Write the failing tests** with fake strategies that record the context they were given.

```python
# packages/core/tests/test_strategy_auto.py
import json

import pytest
from agent_doubles import RecordingLLM, chunk, ctx

from ragfabric_core.strategies.auto import AutoStrategy
from ragfabric_core.strategies.base import (
    RetrievalResult, StrategyName as S, StrategyParams, StrategyRegistry,
)
from ragfabric_core.strategies.contract import assert_strategy_contract

RELATIONS = ["REPORTS_TO", "OWNS"]


class Fake:
    def __init__(self, name, chunks=()):
        self.name, self._chunks, self.seen = name, list(chunks), []

    def retrieve(self, query, ctx):
        self.seen.append(ctx)
        return RetrievalResult(strategy=self.name, chunks=self._chunks, retrieval_calls=1,
                               llm_calls=0, input_tokens=0, output_tokens=0, latency_ms=1)


def build(*, found=None, llm=None, graph_enabled=True, min_confidence=0.6):
    found = found or {}
    fakes = {name: Fake(name, found.get(name, [chunk(9)]))
             for name in (S.TRADITIONAL, S.VECTORLESS, S.AGENTIC, S.GRAPH)}
    registry = StrategyRegistry()
    for fake in fakes.values():
        registry.register(fake)
    auto = AutoStrategy(registry=registry, llm=llm, min_confidence=min_confidence,
                        classifier_model=None, graph_enabled=graph_enabled,
                        relation_types=RELATIONS)
    registry.register(auto)
    return auto, fakes


def classifier(strategy="agentic", confidence=0.9):
    return RecordingLLM(json.dumps({"query_type": "multi_hop", "strategy": strategy,
                                    "confidence": confidence, "reasoning": "Two documents."}))


def test_a_decisive_question_is_routed_with_no_model_call():
    llm = classifier()
    auto, fakes = build(llm=llm)
    result = assert_strategy_contract(auto, "What does ERR_QUOTA_4419 mean?", ctx())
    assert result.strategy is S.VECTORLESS and result.router.source == "signals"
    assert llm.calls == 0 and result.llm_calls == 0


def test_an_undecided_question_is_classified_and_the_call_is_counted():
    llm = classifier()
    auto, fakes = build(llm=llm)
    question = " ".join(["onboarding"] * 20) + "?"
    result = auto.retrieve(question, ctx())
    assert result.strategy is S.AGENTIC and result.router.source == "classifier"
    assert result.llm_calls == 1 and result.router.confidence == 0.9


def test_the_chosen_strategy_gets_a_budget_with_the_classifier_call_deducted():
    auto, fakes = build(llm=classifier())
    auto.retrieve(" ".join(["onboarding"] * 20) + "?", ctx(max_llm_calls=5))
    assert fakes[S.AGENTIC].seen[0].budget.max_llm_calls == 4


def test_a_zero_call_budget_skips_the_classifier():
    llm = classifier()
    auto, _ = build(llm=llm)
    result = auto.retrieve(" ".join(["onboarding"] * 20) + "?", ctx(max_llm_calls=0))
    assert llm.calls == 0 and result.router.source == "signals_fallback"


def test_low_confidence_fuses_traditional_and_vectorless():
    auto, fakes = build(llm=classifier(confidence=0.2),
                        found={S.TRADITIONAL: [chunk(1)], S.VECTORLESS: [chunk(2)]})
    result = auto.retrieve(" ".join(["onboarding"] * 20) + "?", ctx())
    assert result.router.fused and {c.chunk_id for c in result.chunks} == {1, 2}


def test_an_empty_choice_falls_back_once_and_records_it():
    auto, fakes = build(found={S.VECTORLESS: []})
    result = auto.retrieve("What does ERR_QUOTA_4419 mean?", ctx())
    assert result.strategy is S.TRADITIONAL and result.fallback_from is S.VECTORLESS
    assert len(fakes[S.TRADITIONAL].seen) == 1


def test_a_fallback_that_also_finds_nothing_stops():
    auto, fakes = build(found={S.VECTORLESS: [], S.TRADITIONAL: []})
    result = auto.retrieve("What does ERR_QUOTA_4419 mean?", ctx())
    assert result.chunks == [] and len(fakes[S.TRADITIONAL].seen) == 1


@pytest.mark.parametrize("filters", [{"document_id": 3}, {"format": "pdf"}])
def test_graph_is_never_used_under_a_filter_the_walk_cannot_apply(filters):
    auto, fakes = build()
    filtered = ctx().model_copy(update={"params": StrategyParams(metadata_filters=filters)})
    result = auto.retrieve("Who does Ravi Sharma report to?", filtered)
    assert fakes[S.GRAPH].seen == [] and result.strategy is not S.GRAPH


def test_graph_disabled_is_never_selected():
    auto, fakes = build(graph_enabled=False)
    auto.retrieve("Who does Ravi Sharma report to?", ctx())
    assert fakes[S.GRAPH].seen == []


def test_every_strategy_receives_the_callers_own_access_filter():
    auto, fakes = build(found={S.VECTORLESS: []})
    caller = ctx()
    auto.retrieve("What does ERR_QUOTA_4419 mean?", caller)
    for fake in fakes.values():
        for seen in fake.seen:
            assert seen.access_filter == caller.access_filter
            assert seen.principal == caller.principal


def test_auto_is_never_a_routing_target():
    auto, _ = build()
    assert S.AUTO not in auto.available(ctx())
```

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/strategies/auto.py
"""The auto strategy: route, run, fall back at most once, and say what happened.

Registered like the four strategies it routes between, so every entry point
accepts ``strategy: auto`` without its own wiring (ADR 0013). It returns the
result of the strategy that actually ran, with the decision attached, and
never reports itself as the strategy (ADR 0002).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ragfabric_core.providers.base import LLMProvider
from ragfabric_core.router.classifier import ClassifierOutcome, classify, one_sentence
from ragfabric_core.router.decision import RouterDecision, levels_for
from ragfabric_core.router.fallback import combine, empty_reason, fallback_for, fuse
from ragfabric_core.router.signals import Proposal, extract_signals, propose
from ragfabric_core.strategies.base import (
    RetrievalContext, RetrievalResult, StrategyName, StrategyRegistry, TraceSpan,
)

# Request filters the graph walk cannot apply. /api/ask refuses them with a 422
# when a caller names graph; under auto the graph is left out instead.
_GRAPH_BLIND_FILTERS = ("document_id", "format")


class AutoStrategy:
    name = StrategyName.AUTO

    def __init__(
        self,
        *,
        registry: StrategyRegistry,
        llm: LLMProvider | None,
        min_confidence: float,
        classifier_model: str | None,
        graph_enabled: bool,
        relation_types: Sequence[str],
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._min_confidence = min_confidence
        self._classifier_model = classifier_model
        self._graph_enabled = graph_enabled
        self._relation_types = list(relation_types)

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    @property
    def classifier_model(self) -> str | None:
        return self._classifier_model

    @property
    def graph_enabled(self) -> bool:
        return self._graph_enabled

    def available(self, ctx: RetrievalContext) -> list[StrategyName]:
        names = [n for n in self._registry.names() if n is not StrategyName.AUTO]
        filters = ctx.params.metadata_filters
        if not self._graph_enabled or any(key in filters for key in _GRAPH_BLIND_FILTERS):
            names = [n for n in names if n is not StrategyName.GRAPH]
        return names

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        started = time.perf_counter()
        available = self.available(ctx)
        proposal = propose(extract_signals(query, relation_types=self._relation_types),
                           available=available)
        outcome: ClassifierOutcome | None = None

        if proposal.decisive:
            decision = _from_proposal(proposal, source="signals")
        elif self._llm is None or ctx.budget.max_llm_calls < 1:
            decision = _from_proposal(proposal, source="signals_fallback")
        else:
            outcome = classify(query, _signals_of(query, self._relation_types), llm=self._llm,
                               available=available, model=self._classifier_model)
            decision = self._from_classifier(proposal, outcome)

        inner = _deduct_calls(ctx, outcome.llm_calls if outcome else 0)
        chosen = StrategyName(decision.selected_strategy)
        if decision.fused:
            result = fuse(self._run(StrategyName.TRADITIONAL, query, inner),
                          self._run(StrategyName.VECTORLESS, query, inner),
                          top_k=ctx.params.top_k)
        else:
            result = self._run(chosen, query, inner)

        next_strategy = None if decision.fused else fallback_for(chosen, result)
        reason = empty_reason(result) if next_strategy else ""
        if next_strategy is not None:
            result = combine(result, self._run(next_strategy, query, inner),
                             fallback_from=chosen)

        span = TraceSpan(
            name="router", started_ms=0,
            # Routing's own time only. Clamped because the two clocks are read
            # at different moments and TraceSpan refuses a negative duration.
            duration_ms=max(0, int((time.perf_counter() - started) * 1000) - result.latency_ms),
            attributes={"selected": decision.selected_strategy, "source": decision.source,
                        "decisive": decision.decisive, "fused": decision.fused,
                        "fallback_from": str(chosen) if next_strategy else None,
                        "fallback_reason": reason or None,
                        "classifier_violation": outcome.violation.error
                        if outcome and outcome.violation else None,
                        "classifier_error": outcome.error if outcome else None},
        )
        extra = outcome or ClassifierOutcome()
        return result.model_copy(update={
            "router": decision,
            "trace": [span, *result.trace],
            "llm_calls": result.llm_calls + extra.llm_calls,
            "input_tokens": result.input_tokens + extra.input_tokens,
            "output_tokens": result.output_tokens + extra.output_tokens,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        })

    def _run(self, name: StrategyName, query: str, ctx: RetrievalContext) -> RetrievalResult:
        if name is StrategyName.AUTO:
            raise RuntimeError("auto cannot route to itself")
        return self._registry.get(name).retrieve(query, ctx)

    def _from_classifier(self, proposal: Proposal, outcome: ClassifierOutcome) -> RouterDecision:
        if outcome.error is not None:
            return _from_proposal(proposal, source="signals_fallback")
        reply = outcome.reply
        if reply is None or reply.confidence < self._min_confidence:
            return RouterDecision(
                selected_strategy="traditional", source="classifier", decisive=False,
                confidence=reply.confidence if reply else None,
                reasoning="The question was unclear, so meaning and exact word search were combined.",
                query_type=reply.query_type if reply else "ambiguous", fused=True,
                **levels_for("traditional"),
            )
        return RouterDecision(
            selected_strategy=reply.strategy, source="classifier", decisive=False,
            confidence=reply.confidence, reasoning=one_sentence(reply.reasoning),
            query_type=reply.query_type, **levels_for(reply.strategy),
        )


def _signals_of(query: str, relation_types: Sequence[str]):
    return extract_signals(query, relation_types=relation_types)


def _from_proposal(proposal: Proposal, *, source: str) -> RouterDecision:
    strategy = str(proposal.strategy)
    return RouterDecision(
        selected_strategy=strategy, source=source, decisive=proposal.decisive, confidence=None,
        reasoning=one_sentence(proposal.reasons[0]), query_type=proposal.query_type,
        **levels_for(strategy),
    )


def _deduct_calls(ctx: RetrievalContext, calls: int) -> RetrievalContext:
    """The caller's budget less what routing spent, so llm_calls stays within it.

    ``model_copy`` keeps the principal and the access filter untouched (ADR 0003).
    """
    if calls == 0:
        return ctx
    budget = ctx.budget.model_copy(
        update={"max_llm_calls": max(0, ctx.budget.max_llm_calls - calls)}
    )
    return ctx.model_copy(update={"budget": budget})
```

Remove `_signals_of` if review finds it redundant with the first `extract_signals` call: keep the `Signals` object from the first call in a local instead, and pass it to both `propose` and `classify`. The shape above is the minimum; the simplification is expected.

- [ ] **Step 4: Run tests, confirm they pass.** Break `available()` on purpose (drop the filter check) and confirm the filter test fails; restore it.
- [ ] **Step 5: Commit** `feat: add the auto strategy with routing, fusion and one-step fallback`

---

## Task 6: Registry wiring, default resolution, and every router setting read

**Files:**
- Create: `packages/core/src/ragfabric_core/router/mode.py`
- Modify: `strategies/registry_defaults.py`
- Test: `packages/core/tests/test_router_wiring.py`

**Interfaces:**
- Consumes: `AutoStrategy` (Task 5), `RouterConfig`, `RagFabricConfig`.
- Produces: `resolve_requested(requested: str | None, router: RouterConfig) -> StrategyName`; `default_registry` registering `auto`.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_router_wiring.py
import pytest

from ragfabric_core.config_file import RagFabricConfig, RouterConfig
from ragfabric_core.router.mode import resolve_requested
from ragfabric_core.strategies.base import StrategyName as S
from ragfabric_core.strategies.registry_defaults import default_registry


def test_unset_resolves_from_the_router_mode():
    assert resolve_requested(None, RouterConfig(mode="auto")) is S.AUTO
    assert resolve_requested(None, RouterConfig(mode="manual")) is S.TRADITIONAL


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_a_named_strategy_always_bypasses_the_router(mode):
    assert resolve_requested("graph", RouterConfig(mode=mode)) is S.GRAPH


def test_every_router_setting_reaches_something(session_factory):
    cfg = RagFabricConfig.model_validate({
        "llm": {"provider": "offline"}, "embeddings": {"provider": "offline", "dim": 768},
        "router": {"mode": "manual", "min_confidence": 0.33, "classifier_model": "tiny"},
        "graph_store": {"enabled": True},
    })
    auto = default_registry(cfg, session_factory).get(S.AUTO)
    checks = {
        "mode": lambda: resolve_requested(None, cfg.router) is S.TRADITIONAL,
        "min_confidence": lambda: auto.min_confidence == 0.33,
        "classifier_model": lambda: auto.classifier_model == "tiny",
    }
    assert set(checks) == set(RouterConfig.model_fields), "a RouterConfig field is not checked"
    assert all(check() for check in checks.values())


def test_classifier_model_null_uses_the_llm_model(session_factory):
    cfg = RagFabricConfig.model_validate({"llm": {"provider": "offline", "model": "m1"},
                                          "embeddings": {"provider": "offline", "dim": 768}})
    assert default_registry(cfg, session_factory).get(S.AUTO).classifier_model == "m1"
```

Use the `session_factory` fixture the existing registry tests use (`test_registry_agentic_wiring.py`); copy its setup rather than inventing one. Check the offline embedding config keys against `test_graph_config_wiring.py`.

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/router/mode.py
"""Which strategy serves a request that did not name one."""

from __future__ import annotations

from ragfabric_core.config_file import RouterConfig
from ragfabric_core.strategies.base import StrategyName


def resolve_requested(requested: str | None, router: RouterConfig) -> StrategyName:
    if requested:
        return StrategyName(requested)
    return StrategyName.AUTO if router.mode == "auto" else StrategyName.TRADITIONAL
```

In `default_registry`, after the four `register` calls:

```python
    registry.register(
        AutoStrategy(
            registry=registry,
            llm=llm if llm is not None else build_llm_provider(cfg.llm),
            min_confidence=cfg.router.min_confidence,
            classifier_model=cfg.router.classifier_model or cfg.llm.model,
            graph_enabled=cfg.graph_store.enabled,
            relation_types=cfg.graph_store.relation_types,
        )
    )
```

Building the provider makes no call, the same reasoning as `_build_agentic`.

- [ ] **Step 4: Run tests, confirm they pass**, then the full core suite.
- [ ] **Step 5: Commit** `feat: register the auto strategy and resolve the default from router.mode`

---

## Task 7: `graph_search`, its evidence through the loop, and the per request tool set

**Files:**
- Modify: `agent/tools.py`, `agent/nodes.py` (`retrieve`, `RetrieveOutcome`), `agent/loop.py` (`AgentRun.subgraph`, retrieve inside the budget guard), `strategies/agentic.py`, `strategies/registry_defaults.py`
- Create: `packages/core/src/ragfabric_core/graph/merge.py` (`merge_subgraphs`)
- Test: `packages/core/tests/test_agent_graph_tool.py`, and a PostgreSQL test appended to `test_graph_integration.py`

**Interfaces:**
- Consumes: `GraphRAGStrategy`, `Subgraph`, `AgentState.spend`, `NodeName.RETRIEVE`.
- Produces: `GraphSearchTool(strategy: GraphRAGStrategy, llm: LLMProvider)` with `name = "graph_search"`, `run(query, ctx) -> list[RetrievedChunk]`, `run_graph(query, ctx) -> GraphToolRun`;
  `GraphToolRun(chunks: list[RetrievedChunk], subgraph: Subgraph | None, llm_calls: int, input_tokens: int, output_tokens: int)`;
  `RetrieveOutcome.subgraphs: list[Subgraph]`; `AgentRun.subgraph: Subgraph | None`;
  `merge_subgraphs(parts: Sequence[Subgraph]) -> Subgraph | None`;
  `tools_for_request(tools: ToolRegistry, ctx: RetrievalContext) -> ToolRegistry`.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_agent_graph_tool.py
from agent_doubles import FakeTool, RecordingLLM, chunk, ctx

from ragfabric_core.agent.nodes import retrieve
from ragfabric_core.agent.state import AgentState, SubQuestion
from ragfabric_core.agent.tools import GraphToolRun, tools_for_request
from ragfabric_core.graph.merge import merge_subgraphs
from ragfabric_core.strategies.base import StrategyParams


class FakeGraphTool:
    name = "graph_search"
    description = "relationships"

    def __init__(self, run, provider="recording", model="recording"):
        self._run, self.seen, self.provider, self.model = run, [], provider, model

    def run(self, query, ctx):
        return self.run_graph(query, ctx).chunks

    def run_graph(self, query, ctx):
        self.seen.append(ctx)
        return self._run


def test_a_graph_call_is_charged_to_the_agents_budget(small_subgraph):
    tool = FakeGraphTool(GraphToolRun(chunks=[chunk(1)], subgraph=small_subgraph, llm_calls=1,
                                      input_tokens=7, output_tokens=3))
    state = AgentState(question="q", max_llm_calls=5)
    state.sub_questions = [SubQuestion(text="Who owns Billing?", tool="graph_search")]
    outcome = retrieve(state, tools={"graph_search": tool}, ctx=ctx())
    assert state.llm_calls == 1 and outcome.input_tokens == 7
    assert outcome.subgraphs == [small_subgraph]


def test_a_graph_call_over_budget_raises_budget_exceeded(small_subgraph):
    import pytest

    from ragfabric_core.agent.state import BudgetExceeded

    tool = FakeGraphTool(GraphToolRun(chunks=[], subgraph=None, llm_calls=1,
                                      input_tokens=0, output_tokens=0))
    state = AgentState(question="q", max_llm_calls=0)
    state.sub_questions = [SubQuestion(text="Who owns Billing?", tool="graph_search")]
    with pytest.raises(BudgetExceeded):
        retrieve(state, tools={"graph_search": tool}, ctx=ctx())


def test_the_graph_tool_is_dropped_under_a_filter_the_walk_cannot_apply():
    tools = {"semantic_search": FakeTool("semantic_search"),
             "graph_search": FakeGraphTool(None)}
    filtered = ctx().model_copy(
        update={"params": StrategyParams(metadata_filters={"document_id": 4})})
    assert set(tools_for_request(tools, filtered)) == {"semantic_search"}
    assert set(tools_for_request(tools, ctx())) == {"semantic_search", "graph_search"}


def test_merging_subgraphs_unions_by_id_and_keeps_truncation(small_subgraph):
    merged = merge_subgraphs([small_subgraph, small_subgraph])
    assert [n.id for n in merged.nodes] == [n.id for n in small_subgraph.nodes]
    assert merge_subgraphs([]) is None
```

Add a `small_subgraph` fixture to `packages/core/tests/conftest.py` (and the same in `packages/server/tests/conftest.py` for Task 10): three `GraphNode`s, `Ravi Sharma` (person), `Platform Team` (team) and `Billing` (product), and one `GraphEdge`, `Ravi Sharma MEMBER_OF Platform Team`, sourced from chunk 1, built with the real constructors from `graph/contracts.py` (read their required fields; do not guess them). `FakeTool`'s constructor signature is in `agent_doubles.py`; match it.

The PostgreSQL test, appended to `test_graph_integration.py` beside the existing access tests and reusing their seeded fixture: a restricted caller runs the agent with a scripted plan that sends a relationship sub-question to `graph_search`, and neither `run.chunks` nor `run.subgraph` contains any chunk, node or edge sourced only from the denied collection.

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**

In `agent/tools.py`:

```python
class GraphToolRun(BaseModel):
    """One graph_search call: its chunks, its walk, and the model call it cost."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks: list[RetrievedChunk]
    subgraph: Subgraph | None
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class GraphSearchTool:
    """The graph strategy as an agent tool.

    Handed the caller's own ``RetrievalContext``, like every other tool, so the
    access predicate stays inside the walk (ADR 0003). The strategy makes one
    model call to match the question's entities, which is why this tool reports
    its calls and tokens and the retrieve node charges them to the budget.
    """

    name = "graph_search"
    description = (
        "Relationships between named people, teams, organisations, projects, products, "
        "policies and places: who reports to whom, who owns what, what belongs to what. "
        "Blind to passages that name no entities."
    )

    def __init__(self, strategy: GraphRAGStrategy, llm: LLMProvider) -> None:
        self._strategy = strategy
        self.provider = llm.name
        self.model = llm.default_model

    def run(self, query: str, ctx: RetrievalContext) -> list[RetrievedChunk]:
        return self.run_graph(query, ctx).chunks

    def run_graph(self, query: str, ctx: RetrievalContext) -> GraphToolRun:
        result = self._strategy.retrieve(query, ctx)
        return GraphToolRun(chunks=result.chunks, subgraph=result.subgraph,
                            llm_calls=result.llm_calls, input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens)


_GRAPH_BLIND_FILTERS = ("document_id", "format")


def tools_for_request(tools: ToolRegistry, ctx: RetrievalContext) -> ToolRegistry:
    """This request's tools: graph_search is left out under a filter the walk cannot apply."""
    if any(key in ctx.params.metadata_filters for key in _GRAPH_BLIND_FILTERS):
        return {name: tool for name, tool in tools.items() if name != "graph_search"}
    return tools
```

In `agent/nodes.py`, `retrieve`: before calling a tool, branch on `hasattr(tool, "run_graph")`:

```python
        if hasattr(tool, "run_graph"):
            # One model call to match entities, charged before it is made.
            state.spend(NodeName.RETRIEVE, llm_calls=1)
            graph_run = tool.run_graph(query, _with_override(ctx, override))
            chunks = graph_run.chunks
            input_tokens += graph_run.input_tokens
            output_tokens += graph_run.output_tokens
            if graph_run.subgraph is not None and graph_run.subgraph.edges:
                subgraphs.append(graph_run.subgraph)
            provider, model = tool.provider, tool.model
        else:
            chunks = tool.run(query, _with_override(ctx, override))
```

Initialise `input_tokens = output_tokens = 0`, `subgraphs: list[Subgraph] = []`, `provider = model = ""` before the loop, and pass `input_tokens`, `output_tokens`, `provider`, `model` and `subgraphs` into `RetrieveOutcome`. Add `subgraphs: list[Subgraph] = Field(default_factory=list)` to `RetrieveOutcome`. Confirm `state.spend` accepts `NodeName.RETRIEVE` with no per node cap configured for it; if it requires a cap entry, treat a missing entry as no per node cap, and add a test for that.

In `agent/loop.py`, wrap the `retrieve(...)` call in the same `try: ... except BudgetExceeded as exc: stop, detail = STOP_BUDGET, str(exc); break` the `assess` call uses. Collect `retrieved.subgraphs` across iterations and set `AgentRun.subgraph = merge_subgraphs(collected)` (new field, default `None`).

`graph/merge.py`:

```python
"""Several walks' sub-graphs as one, for an agent that walked more than once."""

from __future__ import annotations

from collections.abc import Sequence

from ragfabric_core.graph.contracts import Subgraph


def merge_subgraphs(parts: Sequence[Subgraph]) -> Subgraph | None:
    with_edges = [part for part in parts if part.edges]
    if not with_edges:
        return None
    nodes = {node.id: node for part in with_edges for node in part.nodes}
    edges = {edge.id: edge for part in with_edges for edge in part.edges}
    return Subgraph(nodes=list(nodes.values()), edges=list(edges.values()),
                    truncated=any(part.truncated for part in with_edges), empty_reason=None)
```

In `strategies/agentic.py`, `retrieve`: pass `tools=tools_for_request(self._tools, ctx)` to `run_agent`, and `subgraph=run.subgraph` into the `RetrievalResult`.

In `registry_defaults.py`, `_build_agentic`: build the graph strategy once in `default_registry`, pass it into `_build_agentic`, and add `GraphSearchTool(graph, llm_provider)` to the tool list **only when `cfg.graph_store.enabled`**. Add `graph_search` to the allowed values of `AgenticConfig.tools` and leave the default list unchanged, so a deployment opts in by listing it; a test asserts that with graph enabled and `graph_search` listed the tool is present, and with graph disabled listing it raises the existing unknown tool error.

- [ ] **Step 4: Run tests, confirm they pass**, including the PostgreSQL access test with the variable set.
- [ ] **Step 5: Commit** `feat: give the agent a graph_search tool charged to its budget`

---

## Task 8: The tool check after `plan`

**Files:**
- Modify: `agent/nodes.py` (new `check_tools`), `agent/loop.py`
- Test: `packages/core/tests/test_agent_tool_check.py`

**Interfaces:**
- Consumes: `extract_signals`, `propose`, `TOOL_FOR_STRATEGY` (Task 2).
- Produces: `check_tools(state: AgentState, *, tools: ToolRegistry, relation_types: Sequence[str], origin: float) -> NodeOutcome`; the span is named `tool_check` and carries `overrides` (a count) and `detail` (a string such as `"1:semantic_search->lexical_search(identifier)"`).

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_agent_tool_check.py
import time

from agent_doubles import FakeTool

from ragfabric_core.agent.nodes import check_tools
from ragfabric_core.agent.state import AgentState, SubQuestion

TOOLS = {name: FakeTool(name) for name in ("semantic_search", "lexical_search", "graph_search")}
RELATIONS = ["REPORTS_TO", "OWNS"]


def state_with(*pairs):
    state = AgentState(question="q", max_llm_calls=5)
    state.sub_questions = [SubQuestion(text=text, tool=tool) for text, tool in pairs]
    return state


def test_a_decisive_signal_overrides_the_planner_and_is_traced():
    state = state_with(("What does ERR_QUOTA_4419 mean?", "semantic_search"))
    outcome = check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "lexical_search"
    assert outcome.span.attributes["overrides"] == 1
    assert "semantic_search->lexical_search" in outcome.span.attributes["detail"]


def test_a_relationship_sub_question_goes_to_the_graph():
    state = state_with(("Who does Ravi Sharma report to?", "semantic_search"))
    check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "graph_search"


def test_an_undecided_sub_question_keeps_the_planners_tool():
    long_text = " ".join(["onboarding"] * 20) + "?"
    state = state_with((long_text, "lexical_search"))
    check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "lexical_search"


def test_no_override_to_a_tool_this_run_does_not_have():
    tools = {k: v for k, v in TOOLS.items() if k != "graph_search"}
    state = state_with(("Who does Ravi Sharma report to?", "lexical_search"))
    check_tools(state, tools=tools, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool != "graph_search"


def test_fetch_document_is_never_overridden():
    state = state_with(("document 12", "fetch_document"))
    check_tools(state, tools=TOOLS | {"fetch_document": FakeTool("fetch_document")},
                relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "fetch_document"


def test_the_check_makes_no_model_call():
    state = state_with(("What does ERR_QUOTA_4419 mean?", "semantic_search"))
    check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.llm_calls == 0
```

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement** `check_tools` in `agent/nodes.py`:

```python
def check_tools(
    state: AgentState,
    *,
    tools: ToolRegistry,
    relation_types: Sequence[str],
    origin: float | None = None,
) -> NodeOutcome:
    """Correct the planner's tool choice where the signals are decisive (ADR 0014).

    No model call. The planner's choice stands unless a rule fired
    unambiguously for a different tool this run actually has. ``fetch_document``
    is left alone: it names a document, and no signal knows better.
    """
    begin = time.perf_counter()
    origin = begin if origin is None else origin
    available = [s for s, tool in TOOL_FOR_STRATEGY.items() if tool in tools]
    changed: list[str] = []
    for index, sq in enumerate(state.sub_questions):
        if sq.tool == "fetch_document":
            continue
        proposal = propose(extract_signals(sq.text, relation_types=relation_types),
                           available=available)
        wanted = TOOL_FOR_STRATEGY.get(proposal.strategy)
        if proposal.decisive and wanted and wanted in tools and wanted != sq.tool:
            changed.append(f"{index}:{sq.tool}->{wanted}({proposal.query_type})")
            sq.tool = wanted
    return NodeOutcome(span=_span("tool_check", origin=origin, begin=begin,
                                  overrides=len(changed), detail=";".join(changed) or None))
```

`available` excludes `AGENTIC`, since the agent cannot hand a sub-question to another agent; a comparison sub-question therefore never overrides. In `agent/loop.py`, right after `record(plan(...))` succeeds, call `record(check_tools(state, tools=tools, relation_types=relation_types, origin=origin))`. Thread `relation_types` into `run_agent` as a keyword argument with default `()`, and from `AgenticRAGStrategy` (new constructor argument, from `cfg.graph_store.relation_types` in `_build_agentic`). With `()` no graph signal fires, so existing callers are unaffected.

- [ ] **Step 4: Run tests, confirm they pass**, then `test_agent_*.py` in full: every scripted-double test from Phase 5 must still pass. A test that now fails because the check overrode a scripted tool choice is a real finding; record it in the ledger and change the fixture's question only if its text was never meant to carry that signal.
- [ ] **Step 5: Commit** `feat: let decisive signals override the planner's tool choice`

---

## Task 9: `switch_strategy` across three search tools

**Files:**
- Modify: `agent/policy.py` (`apply_move`, `_PAIRED_TOOLS` replaced), `agent/loop.py` (pass the tool names and the relation types through `_repair_open_sub_questions`)
- Test: add to `packages/core/tests/test_agent_policy.py`

**Interfaces:**
- Consumes: `propose`, `TOOL_FOR_STRATEGY` (Task 2).
- Produces: `apply_move(..., available_tools: Collection[str] = ("semantic_search", "lexical_search"), relation_types: Sequence[str] = ())`.

`SWITCH_STRATEGY` is already limited to once per sub-question by `has_tried`, so "never a tool already tried" reduces to "never the current tool", chosen from the signals' ranking.

- [ ] **Step 1: Write the failing tests**

```python
def test_switch_goes_to_the_graph_when_the_signals_rank_it_next():
    sq = SubQuestion(text="Who does Ravi Sharma report to?", tool="lexical_search")
    outcome = apply_move(sq, RepairMove.SWITCH_STRATEGY, working=RetrievalOverride(query=sq.text),
                         available_tools=("semantic_search", "lexical_search", "graph_search"),
                         relation_types=("REPORTS_TO",))
    assert sq.tool == "graph_search" and "lexical_search to graph_search" in outcome.note


def test_switch_never_returns_the_current_tool():
    sq = SubQuestion(text="What does ERR_QUOTA_4419 mean?", tool="lexical_search")
    apply_move(sq, RepairMove.SWITCH_STRATEGY, working=RetrievalOverride(query=sq.text),
               available_tools=("semantic_search", "lexical_search", "graph_search"))
    assert sq.tool != "lexical_search"


def test_switch_without_graph_keeps_the_phase_5_pairing():
    sq = SubQuestion(text="anything", tool="semantic_search")
    apply_move(sq, RepairMove.SWITCH_STRATEGY, working=RetrievalOverride(query=sq.text))
    assert sq.tool == "lexical_search"
```

- [ ] **Step 2: Run them, confirm they fail** with a `TypeError` on the new keyword.
- [ ] **Step 3: Implement.** Replace the `SWITCH_STRATEGY` branch:

```python
    if move is RepairMove.SWITCH_STRATEGY:
        previous = sub_question.tool
        sub_question.tool = _next_tool(sub_question.text, previous, available_tools,
                                       relation_types)
        return RepairOutcome(move=move, working=working,
                             note=f"switched from {previous} to {sub_question.tool}")
```

```python
_SEARCH_TOOLS = ("semantic_search", "lexical_search", "graph_search")


def _next_tool(
    text: str, current: str, available: Collection[str], relation_types: Sequence[str]
) -> str:
    """The signals' best ranked search tool that is not the one that just failed."""
    strategies = [s for s, tool in TOOL_FOR_STRATEGY.items() if tool in available]
    ranking = propose(extract_signals(text, relation_types=relation_types),
                      available=strategies).ranking
    for strategy in ranking:
        tool = TOOL_FOR_STRATEGY.get(strategy)
        if tool and tool in available and tool != current:
            return tool
    return next((t for t in _SEARCH_TOOLS if t in available and t != current), "semantic_search")
```

The third test pins the Phase 5 behaviour: with only semantic and lexical available, a semantic sub-question switches to lexical and a lexical one to semantic. Check the ranking produces that for a plain question (Traditional first, Vectorless second); if not, the fallback line does.

Thread `available_tools=tuple(tools)` and `relation_types` from `run_agent` into `_repair_open_sub_questions` and on to `apply_move`.

- [ ] **Step 4: Run tests, confirm they pass**, then all `test_agent_*.py`.
- [ ] **Step 5: Commit** `feat: switch_strategy chooses among all three search tools`

---

## Task 10: Answers from agent graph evidence use the graph citation contract

**Files:**
- Modify: `packages/server/src/ragfabric_server/api/routes/search.py` (`_generate`), `api/routes/ask.py` (streaming branch)
- Test: `packages/server/tests/test_agentic_graph_generation.py`

**Interfaces:**
- Produces: `uses_graph_path(result: RetrievalResult) -> bool` in `search.py`, imported by `ask.py`.

- [ ] **Step 1: Write the failing tests**, following the scripted-provider style of the existing server tests for the graph path (find them with `grep -l generate_graph_answer packages/server/tests`).

```python
# packages/server/tests/test_agentic_graph_generation.py
from agent_doubles import RecordingLLM, chunk

from ragfabric_core.strategies.base import RetrievalResult, StrategyName, SubQuestionReport
from ragfabric_server.api.routes.search import _generate, uses_graph_path


def agentic_result(*, chunks, subgraph):
    return RetrievalResult(
        strategy=StrategyName.AGENTIC, chunks=chunks, retrieval_calls=1, llm_calls=2,
        input_tokens=0, output_tokens=0, latency_ms=1, subgraph=subgraph,
        sub_questions=[SubQuestionReport(text="Who is Ravi Sharma's team?", status="answered",
                                         chunk_ids=[c.chunk_id for c in chunks])],
    )


def test_an_agentic_result_with_edges_is_answered_on_the_graph_path(small_subgraph):
    assert uses_graph_path(agentic_result(chunks=[chunk(1)], subgraph=small_subgraph))


def test_an_agentic_result_without_edges_stays_on_the_agentic_path():
    assert not uses_graph_path(agentic_result(chunks=[chunk(1)], subgraph=None))


def test_an_unbacked_relationship_claim_in_an_agentic_answer_is_dropped(small_subgraph):
    evidence = [chunk(1, text="Ravi Sharma is a member of the Platform Team.")]
    llm = RecordingLLM("Ravi Sharma is a member of the Platform Team [1]. "
                       "Ravi Sharma owns Billing [1].")
    generated = _generate("Tell me about Ravi Sharma", agentic_result(
        chunks=evidence, subgraph=small_subgraph), llm)
    assert "owns Billing" not in generated.text
    assert generated.dropped_relationship_claims
```

`small_subgraph` (the Task 7 fixture) holds three nodes, `Ravi Sharma` (person), `Platform Team` (team) and `Billing` (product), and one edge, `Ravi Sharma MEMBER_OF Platform Team`, sourced from chunk 1. "Ravi Sharma owns Billing" names two sub-graph entities with no edge between them, so ADR 0012 must drop it. `agent_doubles` lives in `packages/core/tests`; if the server test package cannot import it, copy `RecordingLLM` and `chunk` into `packages/server/tests/conftest.py` rather than adding a cross-package import. Check the citation marker form `generate_graph_answer` expects (`[1]` for chunks, `[E k]` for edges) against `graph/citations.py` before running, and adjust the scripted answer, not the assertion.

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement** in `search.py`:

```python
def uses_graph_path(result: RetrievalResult) -> bool:
    """A result generated against its walk: the graph strategy's, or an agent's that walked one."""
    return result.strategy == StrategyName.GRAPH or (
        result.subgraph is not None and bool(result.subgraph.edges)
    )
```

Replace `if result.strategy == StrategyName.GRAPH:` with `if uses_graph_path(result):` in `_generate`, and `elif result.strategy == StrategyName.GRAPH:` with `elif uses_graph_path(result):` in `ask.py`'s streaming branch. The graph path already applies the Phase 3 contract to chunk claims and ADR 0012 to relationship claims, so nothing else changes.

- [ ] **Step 4: Run the server suite, confirm the new tests pass and nothing else changed.**
- [ ] **Step 5: Commit** `feat: answer agent graph evidence under the graph citation contract`

---

## Task 11: API surface, run rows and responses

**Files:**
- Modify: `packages/server/src/ragfabric_server/schemas/ask.py`, `schemas/search.py`, `api/routes/ask.py`, `api/routes/search.py`
- Test: `packages/server/tests/test_auto_routes.py`

- [ ] **Step 1: Write the failing tests**

| Test | Asserts |
|---|---|
| `test_auto_is_accepted_on_ask_and_query` | `strategy: "auto"` returns 200 on `POST /api/ask` (both `stream` values) and `POST /api/search/query` |
| `test_an_unset_strategy_resolves_from_router_mode` | With `router.mode: auto` the run row has `requested_strategy` `null` stored as `"auto"` and `mode` `"auto"`; with `manual` it is `"traditional"` and `"manual"` |
| `test_the_run_row_records_the_strategy_that_ran` | `selected_strategy` is the real strategy, and `router_confidence`, `router_reasoning`, `fallback_from` are filled from the result |
| `test_the_response_carries_the_decision` | `AnswerResponse.strategy`, `router`, `fallback_from` present; `router` is `null` for a named strategy |
| `test_hybrid_still_refuses_auto` | `POST /api/search/hybrid` with `auto` is a 422 |
| `test_graph_named_with_a_document_filter_is_still_422` | Unchanged Phase 6 behaviour |
| `test_auto_with_a_document_filter_never_walks_the_graph` | A scripted graph signal question with `document_id` set: `selected_strategy` is not `graph` |
| `test_access_stats_are_measured_on_the_strategy_that_ran` | The audit counts come from `registry.get(result.strategy)`, not from the `auto` object |

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**
  - Schemas: `strategy: str | None = Field(default=None, pattern="^(auto|traditional|vectorless|agentic|graph)$")` on `AskRequest` and the query request; the `Literal` on the search request gains `"auto"` and becomes `... | None = None`. `AnswerResponse` gains `strategy: str | None = None`, `router: RouterDecisionOut | None = None`, `fallback_from: str | None = None`, where `RouterDecisionOut` mirrors `RouterDecision`.
  - Each route resolves `name = resolve_requested(payload.strategy, get_config().router)` once at the top and uses `name` everywhere it used `payload.strategy`, including `_refuse_unapplied_filters` and `_strategy_for`.
  - `_record` and its search twin write `mode="auto" if name is StrategyName.AUTO else "manual"`, `requested_strategy=str(name)`, `selected_strategy=str(result.strategy)`, `fallback_from=str(result.fallback_from) if result.fallback_from else None`, `router_confidence=result.router.confidence if result.router else None`, `router_reasoning=result.router.reasoning if result.router else None`.
  - `_access_stats` is called with `registry.get(result.strategy)` when `name is StrategyName.AUTO`, so the count is measured on the index that actually answered (ADR 0004).
  - The streaming `retrieval` event gains `router` and `fallback_from`.
- [ ] **Step 4: Run the server suite with PostgreSQL**, confirm it passes.
- [ ] **Step 5: Commit** `feat: accept auto on the API and record the routing decision`

---

## Task 12: CLI and SDK

**Files:**
- Modify: `packages/cli/src/ragfabric_cli/commands/ask.py`, `packages/sdk-python/src/ragfabric_sdk/client.py`, `sdk-python/src/ragfabric_sdk/models.py`
- Test: `packages/cli/tests/test_ask_auto.py`, `packages/sdk-python/tests/test_client_auto.py`

- [ ] **Step 1: Write the failing tests**

| Test | Asserts |
|---|---|
| SDK `test_ask_omits_strategy_when_unset` | The request body has no `strategy` key when the caller passes none |
| SDK `test_answer_parses_router_and_tolerates_an_old_server` | `Answer.router` parses; a response without it parses with `router is None` |
| CLI `test_auto_is_a_choice` | `--strategy auto` is accepted by Typer |
| CLI `test_the_default_sends_no_strategy` | No `--strategy` sends none, so the server's `router.mode` decides |
| CLI `test_one_line_names_the_strategy_and_why` | Output contains `Strategy: graph (signals). The question asks how named things are related.`, and `fell back from vectorless` when `fallback_from` is set |

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** SDK: `strategy: str | None = None` on `ask`, `ask_stream` and the query method, included in the body only when not `None`; `Answer` gains `strategy`, `router`, `fallback_from`, all optional so an older server parses. CLI: `auto = "auto"` in `Strategy`, the option defaults to `None`, the help text says the server's `router.mode` decides when it is omitted, and after the answer one line is printed:

```python
def _routing_line(answer) -> str | None:
    if answer.router is None:
        return None
    line = f"Strategy: {answer.strategy} ({answer.router['source']}). {answer.router['reasoning']}"
    if answer.fallback_from:
        line += f" Fell back from {answer.fallback_from}, which found nothing."
    return line
```

Adapt the attribute access to however `Answer.router` is typed (a model or a dict). The richer panel is Phase 7b.
- [ ] **Step 4: Run the CLI and SDK suites.** CHANGELOG `[0.4.0] - unreleased` gains a **Changed** entry: the SDK and CLI no longer default to `traditional`; an unset strategy is resolved by the server's `router.mode`, which defaults to `auto`.
- [ ] **Step 5: Commit** `feat: auto in the CLI and SDK, with the default left to the server`

---

## Task 13: Documentation and ADRs

**Files:**
- Create: `docs/adr/0013-router-as-a-strategy.md`, `docs/adr/0014-signals-override-the-planner.md`, `docs/concepts/routing-as-classification.md`
- Modify: `docs/routing.md` (rewritten to what shipped), `docs/agentic-rag.md` (tools table gains `graph_search`, the "no graph tool yet" paragraph removed, the tool check and the three-way switch described), `README.md` (router section), `ROADMAP.md` (Phase 7 router and agent items ticked, `[~]` on the phase until 7b), `CHANGELOG.md`, `ragfabric.example.yaml` (comments on `router.*` say what reads each key; `graph_search` shown as an opt-in tool)

- [ ] **Step 1:** ADR 0013 records the three options and the table from this plan's "Router as a strategy" section. ADR 0014 records the three ways to combine signals and planner, and why the hint-only option was rejected, citing `docs/learning/agentic-first-run.md`.
- [ ] **Step 2:** `docs/routing.md` states the decisive rule, the `signals`, `classifier` and `signals_fallback` sources, the fusion, the fallback table as built, and the one deliberate change from the concept (agentic falls back only on zero evidence), with the reason.
- [ ] **Step 3:** `docs/concepts/routing-as-classification.md` explains, in tables, routing as classification, why a rule has no confidence, the cost of a wrong route in each direction, and what calibration in Phase 8 will change.
- [ ] **Step 4:** Grep every changed doc for em dashes and for a quality claim with no run behind it; fix both.
- [ ] **Step 5: Commit** `docs: the router, ADRs 0013 and 0014, and routing as classification`

---

## Task 14: A real routing run against Ollama

**Files:**
- Create: `packages/core/tests/test_router_integration.py` (gated by `RAGFABRIC_TEST_OLLAMA=1` and the PostgreSQL URL, the same guard as `test_agentic_integration.py`), `docs/learning/routing-first-run.md`

- [ ] **Step 1:** Write the gated test. Corpus: the Phase 5 fixture plus the Phase 6 graph fixture, ingested with `graph_store.enabled: true`. Questions: the Phase 5 compound question verbatim; "Who does Ravi Sharma report to?"; an identifier question; a long vague question that must reach the classifier. The test asserts only what must hold whatever the model does: every result satisfies the strategy contract, no denied chunk appears, every `router.reasoning` is at most 200 characters, and `llm_calls` equals the calls the doubles would count.
- [ ] **Step 2:** Run it with `llama3.1:8b`. Record, per question: the decision and its source, the agent's planned tools, every `tool_check` override, every `switch_strategy`, fallbacks, termination, counters and latency.
- [ ] **Step 3:** Write `docs/learning/routing-first-run.md` in the shape of the two earlier first-run records: "This is a record, not a benchmark", what happened, findings including the unflattering ones, and what the run does not tell you. The comparison that matters: did the tool check correct the paraphrase-to-lexical mistake the Phase 5 run recorded?
- [ ] **Step 4:** If the model does something the design did not expect, write it down. Do not tune the fixture until it looks good (ADR 0004).
- [ ] **Step 5: Commit** `docs: record the first real routing run`

---

## Task 15: Whole-branch verification and the pull request

- [ ] **Step 1:** Full suite with PostgreSQL. Record passed and skipped against the Task 1 baseline; every new skip is named with its reason.
- [ ] **Step 2:** `uv run ruff check packages/`, `uv run ruff format --check packages/`, `uv run lint-imports` (3 kept, 0 broken), the frontend build and specs unchanged.
- [ ] **Step 3:** Grep all commit messages on the branch and the diff for em dashes and assistant references; fix any.
- [ ] **Step 4:** Whole-branch review against this plan's Self-Review list, then one fix wave.
- [ ] **Step 5:** Push `feat/phase-7a-router` and open a PR into `main` that says it is Phase 7a of 7 and that v0.4.0 is tagged after 7b. **Stop there.** Merging, tagging and publishing each need explicit approval.

---

## Parallel execution

Tasks 1 and 2 land first, by the controller, because both tracks compile against them: Phase 5 proved what happens when one track owns a shared interface.

| Track | Tasks | Area | Worktree and branch |
|---|---|---|---|
| A, the router | 3, 4, 5, 6 | classifier, fallback, `auto`, wiring | `~/AI/ragfabric-wt/phase-7a-router`, `feat/phase-7a-track-router` |
| B, the agent | 7, 8, 9 | `graph_search`, tool check, switch | `~/AI/ragfabric-wt/phase-7a-agent`, `feat/phase-7a-track-agent` |

Both merge into `feat/phase-7a-router`. Tasks 10 to 15 then run sequentially, one implementer at a time. The two tracks touch `registry_defaults.py` in different functions (`default_registry` in A, `_build_agentic` in B); expect a small conflict there and resolve it by hand.

**Expect the merge to break.** The merged-branch test run is the only one that proves anything.

---

## Self-Review

Before the phase is called done:

- Can any path under `auto`, fallbacks and fusion included, return a chunk from a document the caller cannot read? Prove by test.
- Can `auto` or the agent walk the graph for a request that set `document_id` or `format`?
- Does any `RouterDecision` from signals carry a confidence number?
- Does a fallback result carry the failed attempt's `sub_questions` or `subgraph`?
- Is the classifier ever shown chunk text?
- Can `auto` spend more model calls than the caller's budget allows?
- Is every `RouterConfig` field read by something?
- Does any existing Phase 5 or Phase 6 test pass only because a fixture was changed to avoid the new tool check?
- Is any quality claim made anywhere without a measured run behind it?
