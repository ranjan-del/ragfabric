# Routing as classification, concept notes

> Status: Phase 7a (shipped in core). This is a learning document: what choosing a strategy
> actually is, why a rule carries no confidence, and what a wrong choice costs, not an API
> reference. For the code, see `packages/core/src/ragfabric_core/router/signals.py`,
> `classifier.py`, `decision.py`, `fallback.py` and `strategies/auto.py`. For the behaviour as a
> whole, see [routing.md](../routing.md). The decisions are
> [ADR 0013](../adr/0013-router-as-a-strategy.md) and
> [ADR 0014](../adr/0014-signals-override-the-planner.md).

## Choosing a strategy is classifying a question

| Piece | In machine learning | In the router |
|---|---|---|
| Input | A feature vector | The question, plus the signals read from it |
| Classes | Labels | `traditional`, `vectorless`, `graph`, `agentic` |
| Stage one | A cheap model | Signals: rules that fire or do not |
| Stage two | A stronger model, used when the first is unsure | The classifier: one model call |
| Output | A label and a score | A strategy, a reason, and a confidence only when a model gave one |
| Ground truth | A labelled test set | None yet. Phase 8 builds it |

The difference from most classification is that the label is not the goal. A question is not
"really" a graph question. The right class is whichever strategy finds the evidence, so a router
can only be judged by what happened after it chose, which is why it cannot be graded without a
labelled question set.

## The two stages

| Stage | Cost | Decides | Fails by |
|---|---|---|---|
| Signals | No model call, no I/O | When exactly one rule fired, or nothing fired on a short plain question | Being wrong about a question whose wording hides its real shape |
| Classifier | One model call | When rules conflict or the question is long and open | Being unsure, malformed, or unreachable |

| Signal | Points to |
|---|---|
| An identifier or a quoted phrase | Vectorless |
| A relation phrase with a named entity | Graph |
| Comparison, aggregation or several questions in one | Agentic |
| Short, one concept, none of the above | Traditional |

When the classifier is unsure (its confidence is below `router.min_confidence`, or its reply
breaks the contract), the router stops guessing and runs Traditional and Vectorless, then fuses
them with RRF ([ADR 0008](../adr/0008-rrf-over-score-fusion.md)). When the classifier cannot be
reached, the signals' own proposal is used.

## Why a rule has no confidence

| Quantity | What it is | Can it be reported as a number? |
|---|---|---|
| A signal | A rule that fired or did not | No. It is a yes, not a measurement |
| A classifier confidence | What the model said about its own answer | Yes, as reported, and labelled uncalibrated |
| A calibrated confidence | A score checked against outcomes | Not until Phase 8 |

A rule that matches an error-code pattern might be right most of the time, but "most" was never
counted. Attaching 0.9 to it would put a figure in the decision that nobody measured, which
[ADR 0004](../adr/0004-measurement-first-no-fabricated-numbers.md) forbids. So a signals decision
says `confidence: null`, with `decisive: true` when it came from the signals (a `signals_fallback`
decision is not decisive). Even the classifier's number is only a model's self-report, and models
are often more sure than they are right. That is why the threshold `router.min_confidence` is an
untuned starting value.

## The cost of a wrong route

| Route taken | What the question needed | What the error costs | How the system limits it |
|---|---|---|---|
| Traditional | An exact identifier | The exact code can be missed, because a near-identical code looks almost the same to vector search | The identifier rule sends these to Vectorless before any model is asked |
| Vectorless | A paraphrase match | The passage in other words is not found, so the result is empty | An empty result falls back to Traditional |
| Graph | No entities in the corpus | An empty walk, plus one wasted entity-matching call | Any empty result, including the graph reasons `no_graph_coverage`, `no_entity_matched` and `no_walkable_edges`, falls back to Traditional |
| Agentic | A simple lookup | Several model calls, more latency, same evidence | Only a comparison, aggregation or compound question fires the agentic rule. The classifier can also choose it, and nothing limits that beyond its confidence threshold |
| Traditional | A multi-part comparison | One retrieval answers part of it | Not caught by a fallback, since some evidence was returned |

| Direction of error | Looks like | Recoverable by the router |
|---|---|---|
| Too cheap a strategy | Empty or partial evidence | Only when the result is empty |
| Too costly a strategy | The right evidence, at higher cost and latency | No, the spend has already happened |

The asymmetry is the main design fact. An empty result announces itself and can be retried. A
partial result does not, so the router cannot tell a good answer from a half answer. That is why
the fallback rule is written on emptiness only.

## One step, never a chain

| Choice | Reason |
|---|---|
| At most one fallback | A chain multiplies cost and makes the trace hard to read |
| The fallback is always Traditional | It is the cheapest and the broadest, so it is the safe last word |
| Agentic falls back only on zero usable evidence | An agent that answered some sub-questions keeps them, and already reports the open ones |
| Every attempt stays in the trace and every counter is summed | The cost shown is the cost incurred |

## What calibration in Phase 8 will change

| Today | After Phase 8 |
|---|---|
| Signals have no confidence | Unchanged. A rule is still a rule, but its hit rate can be reported beside it |
| Classifier confidence is a model's self-report | Checked against labelled questions, so a stated 0.8 can be compared with how often such a choice was right |
| `router.min_confidence` is an untuned default | Set from the evaluation data |
| No measure of whether the router picked well | Per question, whether the chosen strategy scored as well as the best of the four |
| The fallback rate is recorded per run but not analysed | A high rate for one strategy becomes a signal that the router or the strategy needs work |
| Nothing is claimed about quality | Figures appear only when a run produced them (ADR 0004) |
