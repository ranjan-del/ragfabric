# ADR 0013: The router is a strategy, not a layer

Status: accepted
Date: 2026-09-30
Supersedes: none
Related: ADR 0002 (common retrieval result), ADR 0003 (access control inside retrieval),
ADR 0004 (measurement first, no fabricated numbers), ADR 0008 (RRF over score fusion)

## Context

`docs/routing.md` described a router that chooses between Traditional, Vectorless, Agentic and
Graph, but not where it sits. Three places were possible, and the choice decides how many entry
points must know about routing, what shape a routed answer has, and what a simple lookup costs.

The evidence that routing is needed comes from two real runs against local models
(`docs/learning/agentic-first-run.md`, `docs/learning/graph-extraction-first-run.md`): a small
planner sent a paraphrase question to exact-word search, and the graph is sparse on a small corpus,
so choosing it blindly returns nothing. Neither run measured retrieval quality, and this ADR does
not claim any.

## Decision

`auto` is a fifth entry in the strategy registry, `AutoStrategy` in `strategies/auto.py`. It runs a
two-stage router, then the strategy it chose, then at most one fallback, and returns an ordinary
`RetrievalResult` with a `router` decision attached. `RetrievalResult.strategy` stays the strategy
that actually ran and is never `auto`.

| | **`auto` as a strategy (chosen)** | Router as a separate layer | Agent as the router |
|---|---|---|---|
| Entry points | API, CLI and SDK accept `strategy: auto` with no new wiring | Every entry point wired and kept in sync separately | None new |
| Result shape | One `RetrievalResult`, ADR 0002 unchanged | A second wrapper shape | One |
| Cost on a simple lookup | Zero model calls when signals are decisive | Same | Several model calls on every question |

The agent-as-router option defeats the reason a router exists. Most questions do not need an agent,
and the router's job is to spend the agent's cost only where it pays.

### Rules that follow from it

| Rule | Why |
|---|---|
| Naming a strategy always bypasses the router. An unset strategy is resolved by the server from `router.mode` (`auto` gives `auto`, `manual` gives `traditional`) | Comparing the strategies fairly needs a way to ask for exactly one |
| `auto` is never a routing target, never a fallback target and never an agent tool | A router that can route to itself can loop |
| `auto` passes the caller's own `RetrievalContext` to everything it runs, fallbacks included | ADR 0003. Nothing rebuilds a context |
| Graph is left out of the candidates when `graph_store.enabled` is false, or when the request sets `document_id` or `format` | The graph walk applies access and collection scope only, so it cannot honour those filters. `/api/ask` already refuses them with a 422 when a caller names `graph` |
| The classifier sees the question and the signals, never chunk text | Its one sentence of reasoning cannot repeat content from a document the caller may not read |
| A failed classifier call does not fail the request | The signals' own proposal is used and the decision says `source: signals_fallback` |
| Every path stays within the caller's `max_llm_calls` | The classifier call is deducted before the chosen strategy runs, a fallback gets only what is left after the first attempt, and on the fused path Vectorless gets what Traditional left |
| A fallback result carries none of the failed attempt's `sub_questions` or `subgraph`; the failed attempt stays in the trace | Generation picks its path from those fields. A Traditional fallback that kept an agent's reports would be answered on the agentic path |
| Counters are summed across every run made and every trace is kept | The cost shown is the cost incurred (ADR 0004) |

### Fallbacks, one step and never a chain

| Trigger | Action |
|---|---|
| Graph returns `no_graph_coverage`, `no_entity_matched` or `no_walkable_edges` | Traditional |
| Vectorless finds no term match | Traditional |
| Agentic returns zero usable evidence | Traditional |
| Traditional was chosen and found nothing | None. An honest empty result |
| The fallback also finds nothing | Stop. Empty, with both attempts in the trace |

The one deliberate change from the earlier concept: agentic falls back only when it returns zero
usable evidence, not when it exhausts its budget. An agent that ran out of budget with some
sub-questions answered keeps them and already reports the unanswered ones as open with a reason.
Discarding answered evidence to run a simpler search would make the answer worse.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| Router as a separate layer in front of the strategies | Every entry point would need wiring and keeping in sync, and a second result wrapper would break the single shape ADR 0002 gives every consumer |
| The agent as the router | Several model calls on every question, including the simple ones the router exists to keep cheap |

## Consequences

- Every client that accepts a strategy name accepts `auto` without new code, and every consumer of
  `RetrievalResult` keeps working. The decision travels in an added `router` field.
- The SDK and CLI default changes from `traditional` to unset, so the server decides. This is a
  behaviour change and is recorded in the CHANGELOG.
- A routed request is at most one classifier call, one strategy run and one fallback, so its cost
  is bounded by the caller's budget on every path.
- Whether the router picks the best strategy is a measurement, and none exists yet. Phase 8
  measures it (ADR 0004), and the first real run is written up in `docs/learning/`.
