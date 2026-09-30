# Query routing

> Status: **shipped in core for v0.4.0 (Phase 7a)**, release not yet cut. The terminal
> experience around it (rich `ask` output, `doctor`, `quickstart`) is Phase 7b. Whether the router
> chooses well is **not measured**: the first real run is written up in `docs/learning/`, and
> Phase 8 measures it properly (ADR 0004).

See [concepts/routing-as-classification.md](concepts/routing-as-classification.md) for the mental
model, [ADR 0013](adr/0013-router-as-a-strategy.md) for where the router sits and
[ADR 0014](adr/0014-signals-override-the-planner.md) for how it checks the agent's tool choices.

## What it does

`strategy: auto` reads the question and chooses one of Traditional, Vectorless, Graph or Agentic.
Naming a strategy bypasses the router, which is how the four are compared fairly on the same
question. `auto` is a fifth entry in the strategy registry (`strategies/auto.py`), so the API,
CLI and SDK accept it with no separate wiring, and it returns an ordinary `RetrievalResult`.

| Caller sends | What runs |
|---|---|
| `strategy: auto` | The router, then the chosen strategy, then at most one fallback |
| A named strategy | That strategy only. The router is bypassed |
| Nothing | The server resolves it from `router.mode`: `auto` gives `auto`, `manual` gives `traditional`. This holds on `/api/ask`, `/api/search/query` and `/api/search/semantic`. `/api/search/hybrid` does not route: unset means `traditional` there whatever `router.mode` says, and `auto` is a 422 |

## The decision object

```json
{
  "selected_strategy": "graph",
  "source": "signals",
  "decisive": true,
  "confidence": null,
  "reasoning": "The question asks how named things are related.",
  "query_type": "relationship",
  "estimated_complexity": "medium",
  "expected_cost_level": "medium",
  "expected_latency_level": "medium",
  "fused": false
}
```

`reasoning` is one sentence of at most 200 characters. It comes from a fixed template for signals
and from the classifier otherwise. Hidden chain of thought is never exposed. The three level
fields are engineering assessments of each strategy's shape (one embedding call is low, a model
call per step is high), not measurements.

| `source` | Meaning | `confidence` |
|---|---|---|
| `signals` | A rule fired, or a short plain question fired nothing. `decisive` is true | `null` |
| `classifier` | The signals were not decisive and one model call chose | What the model reported, uncalibrated until Phase 8 |
| `signals_fallback` | The classifier could not be used, so the signals' own proposal stood. `decisive` is false | `null` |

## How it works internally

Two stages, cheap first.

| Stage | Input | Output | Model calls |
|---|---|---|---|
| Signals (`router/signals.py`) | The question only. Pure function, no I/O | A `Proposal`: strategy, `decisive`, `from_rule`, reasons, a ranking | 0 |
| Classifier (`router/classifier.py`) | The question and the signals, never chunk text | `query_type`, strategy, confidence, one sentence of reasoning | 1 |

### The signals

| Signal | Detected by | Points to |
|---|---|---|
| Identifier | `identifiers()` in `stores/boosting.py`, the same function the identifier boost uses | Vectorless |
| Quoted phrase | `phrases()` in `stores/boosting.py` | Vectorless |
| Relation phrase | A spoken form of a configured relation type ("reports to", "owns") together with at least one named entity | Graph |
| Comparison, aggregation or several questions | Words such as compare, versus, how many, total, or more than one question mark | Agentic |
| None of these, and 15 words or fewer | The default | The first available of Traditional, Vectorless, Graph, Agentic |

### The decisive rule

A decision is **decisive** when exactly one usable signal fired (`from_rule` is true), or when
nothing usable fired and either the question is short or a signal fired that could not be used
(`from_rule` is false, because that is a default and not a rule). Conflicting usable signals, or a
long question with no signal at all, are not decisive and go to the classifier. A signal that
points at a strategy unavailable for this request is noted in the reasons and not counted.

### What the router does when

| Situation | Result |
|---|---|
| Decisive | `source: signals`, no model call |
| Not decisive, the classifier answers with confidence at or above `router.min_confidence` | `source: classifier`, that strategy |
| Not decisive, confidence below `router.min_confidence`, or the reply breaks the contract | Traditional and Vectorless run and are fused with `rrf()` (ADR 0008), `fused: true` |
| The classifier raises or times out, or the request has no budget for a call | `source: signals_fallback`, the signals' proposal is used |

A contract violation includes malformed JSON, a strategy not available for this request, and a
reasoning that is empty after trimming. All of them fuse rather than fail the request.

### Which strategies are candidates

| Condition | Effect |
|---|---|
| `graph_store.enabled` is false | Graph is never selected. A graph signal resolves to another strategy |
| The request sets `document_id` or `format` | Graph is never selected and never a fallback. The walk applies access and collection scope only |
| Otherwise | All four |

`auto` itself is never a routing target, a fallback target or an agent tool.

## Fallbacks as built

One step, never a chain. Every fallback is recorded on the run as `fallback_from`.

| Trigger | Action |
|---|---|
| Any strategy other than Traditional returns no chunks | Run Traditional. For Graph the recorded reason is one of `no_graph_coverage`, `no_entity_matched` or `no_walkable_edges`; for Vectorless no term match; for Agentic zero usable evidence |
| Traditional was chosen and found nothing | None. An honest empty result |
| The fallback also finds nothing | Stop. Empty, with both attempts in the trace |
| Classifier confidence below `router.min_confidence` | Not a fallback. Traditional and Vectorless are fused up front |

A fallback result carries none of the failed attempt's `sub_questions` or `subgraph`. Generation
picks its path from those fields, so a Traditional fallback that kept an agent's reports would be
answered on the agentic path. The failed attempt stays in the trace.

### The one deliberate change from the concept

The earlier concept said an agent that "exhausts its budget with insufficient evidence" falls back
to Traditional. As built, agentic falls back **only when it returns zero usable evidence**. An
agent that ran out of budget with some sub-questions answered keeps them, and already reports the
unanswered ones as open with a reason. Discarding answered evidence to run a simpler search would
make the answer worse.

## The budget

`auto` never exceeds the caller's `max_llm_calls` on any path.

| Step | Budget rule |
|---|---|
| The classifier call | Deducted before the chosen strategy runs |
| A fallback | Gets only what is left after the first attempt |
| The fused path | Vectorless gets what Traditional left |
| Traditional with no calls left | Skips an LLM reranker when its call budget is zero |
| Graph with no calls left | The graph strategy skips its entity-matching call when no call is left, returns nothing, and `auto` falls back to Traditional |
| Counters | Summed across every run made. Every trace is kept |

## Safety

| Concern | Behaviour |
|---|---|
| Access | `auto`, every fallback and `graph_search` pass the caller's principal and access filter untouched (ADR 0003). The context is a copy with a reduced budget and nothing else changed |
| What the classifier sees | The question and the signals only, never chunk text |
| Routing trouble | Never fails a request. A failed classifier becomes `signals_fallback` |
| Cost | The classifier call counts in `llm_calls`, tokens and cost. The router has its own `router` span |

## What is recorded

| Where | What |
|---|---|
| `RetrievalResult.router` | The decision above. `RetrievalResult.strategy` is the strategy that ran, never `auto` |
| `RetrievalResult.fallback_from` | The strategy that found nothing, when a fallback ran |
| `/api/ask` and `/api/search/query` responses | `strategy`, `router` and `fallback_from`. The ask stream's `retrieval` event carries them too |
| `/api/search/semantic` response | `strategy` only |
| The `retrieval_runs` row | The requested strategy, the strategy that ran, `fallback_from`, router confidence and reasoning (no migration, the columns exist since 0002) |
| CLI | One line, `Strategy: <ran> (<source>). <reasoning>`, plus a fallback note |

## Configuration

| Key | Read by |
|---|---|
| `router.mode` | Default strategy resolution when the caller names none |
| `router.min_confidence` | The fuse threshold. A classifier confidence below it fuses Traditional and Vectorless |
| `router.classifier_model` | The classifier. `null` uses `llm.model` |

A test fails if any `RouterConfig` field is not read.

## The agent

The same signals module checks the agent's tool choices and gives it a third search tool. See
[agentic-rag.md](agentic-rag.md) and [ADR 0014](adr/0014-signals-override-the-planner.md).

## Why not only rules

Rules are transparent but brittle: "compare" appears in simple questions too. Why not only an LLM:
a model call on every question, and no explanation of the decision. The two-stage design routes a
question with no model call when a rule is decisive and spends one call where the signals
disagree.

## Trade offs

A wrong route costs either accuracy (Traditional on a multi-part question) or money (Agentic on a
lookup). The evaluation framework in Phase 8 is what will quantify both, so the thresholds can be
tuned with numbers. Until then `router.min_confidence` is an untuned default.
