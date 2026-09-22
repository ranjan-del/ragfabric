# Agentic loops, concept notes

> Status: Phase 5 (shipped in core). This is a learning document: what a state machine agent
> actually is, what makes one different from a retry loop, and why stopping is the hard part,
> not an API reference. For the code, see
> `packages/core/src/ragfabric_core/agent/state.py`, `agent/nodes.py`, `agent/loop.py`,
> `agent/policy.py`, `agent/contracts.py` and `agent/tools.py`. For the strategy as a whole,
> see [agentic-rag.md](../agentic-rag.md). The two decisions behind the design are
> [ADR 0009](../adr/0009-plain-state-machine-over-langgraph.md) and
> [ADR 0010](../adr/0010-repair-policy-and-termination.md).

## What an agent is here, and what it is not

"Agent" is used for two very different things, and the difference is worth being blunt about.

| | What it decides | What bounds it | How you test it |
|---|---|---|---|
| **Open-ended agent** | Which tool to call, with what arguments, and when to stop, all from free-form model output | A turn limit, and hope | With difficulty. The action space is whatever the model emits |
| **State machine agent** (this one) | Which sub-questions to ask, which tool answers each, whether the evidence is good enough, and which of six repair moves to make | A fixed set of nodes, a fixed set of moves, an iteration cap, per-node and global call caps | Exhaustively. Every branch is reachable from a scripted model double with no network and no database |

The second is smaller and that is the point. The control flow is written in Python and does
not change between runs. What changes is the content of the decisions: how the question is
split, which tool each part goes to, whether a part is answered, and which repair fires. The
model supplies judgement, and the code supplies the shape.

The model's judgement arrives as JSON validated against a schema, never as a native tool call
and never as free text. That choice is argued in ADR 0009, and it has one consequence worth
stating here: **the repair move comes from a fixed six-value enum**, so a small local model
cannot invent an action that has no defined effect. The worst it can do is name a valid move
that fits the situation poorly, and the next iteration can correct that.

## The parts

| Part | In this codebase | What it holds |
|---|---|---|
| **State** | `AgentState` | The question, the sub-question ledger, the evidence pool keyed by chunk id, the call counters and the caps |
| **Node** | A plain function in `agent/nodes.py`, plus `repair` in `agent/loop.py` | One step. It reads state, mutates it, returns a `TraceSpan`, and decides nothing about what runs next |
| **Branch** | `run_agent` in `agent/loop.py` | All the control flow, in one readable `while` loop |
| **Tool** | `agent/tools.py` | One retrieval capability, wrapping a strategy that already exists |
| **Trace** | `list[TraceSpan]` on the result | What actually happened, in order, appended by the code that did it |

Nothing here is a framework concept. A node is a function, an edge is an `if`, and the trace
is a list. ADR 0009 records why.

## The sub-question ledger, and why it is the whole design

The single most consequential choice is that evidence is judged **per sub-question**, not for
the question as a whole.

Judge globally and decomposition buys nothing. The loop can tell that the answer is not
complete, but not which part is missing, so on the next pass it re-retrieves everything,
including the parts it already has. Judge per sub-question and each part carries its own
status, its own tool and its own history of what has been tried on it. The next iteration
retrieves only for the parts still open, which is what makes a second iteration cheap enough
to be worth having.

```
question
  |
  +-- sub-question 1   tool: lexical_search    status: answered
  +-- sub-question 2   tool: semantic_search   status: open
  |      attempts: broaden  "retry limit payments API"  -> no passage states the number
  +-- sub-question 3   tool: fetch_document    status: abandoned
         reason: abandoned after trying broaden, switch_strategy: the effective date
```

The attempt history is not a list of queries already used. It is what was tried, what was
sent, and what the assessment said was missing. "Try something different" is a uniqueness
check; keeping the failure is what lets the next choice be better than the last one rather
than merely unlike it.

## Progress is new chunk ids, not rows returned

An agent that re-fetches the same evidence and calls it an iteration will burn its entire
budget going nowhere, and every pass will look like work from the outside: calls were made,
chunks came back, the model produced a verdict.

The evidence pool is a dict keyed by chunk id. Adding to it returns only the ids that were not
already there, and that set is the progress signal.

| Definition of progress | What it reports on a tool that returns the same three chunks every call |
|---|---|
| The tools returned something | Progress, every time, forever |
| The number of chunks grew | No progress, but only because the pool happens to be a set. It breaks the moment anything is counted before deduplication |
| The pool gained a chunk id it did not hold | No progress, which is the truth |

The third is what the code uses. It is the only definition that cannot be satisfied by doing
the same thing again.

## Why stopping is the hard part

Retrieving is easy to start and hard to stop. Three failure modes sit either side of the right
answer.

| Failure | What it looks like | What prevents it here |
|---|---|---|
| Stopping too early | A confident, incomplete answer. The agent decided the evidence was sufficient because it was on the right topic | A pessimistic assessment rubric: a sub-question is answered only if the answer can be read out of the retrieved text. Related is not answered, implied is not answered, unsure is not answered |
| Stopping too late | The full budget spent, the same evidence, a worse latency and a real bill | The `no_progress` stop, and the rule that a repair move is never repeated on an unchanged sub-question |
| Stopping for an unrecorded reason | A short answer that looks like a complete one | Three named stop reasons, each assigned at the branch that decides it |

The third failure is the quiet one. An answer that stopped early because the budget ran out
and an answer that stopped because the corpus does not contain the fact are the same text on
the screen. They are completely different facts about the world, and only one of them is worth
re-asking. That is why the run reports which of the three fired:

| Stop reason | Meaning |
|---|---|
| `resolved` | Every sub-question ended answered or abandoned. Not the same as "fully answered": an abandoned part carries its reason |
| `budget` | An iteration cap, a global call cap or a per-node cap stopped it. There may well be more to find |
| `no_progress` | An iteration added nothing the pool did not hold. More iterations cannot help |

`no_progress` is the one the textbook agentic design does not have, and it is the one that
fires most often on a real corpus that simply does not contain the answer.

There is one exemption. `no_progress` is not checked on the first iteration, because an empty
first retrieval is exactly what the repair moves exist to fix: the plan may have picked the
wrong tool, or written a query more specific than the corpus. Stopping there would reduce the
agent to a single-shot retriever with extra model calls.

## Repair is a choice between moves, not a rewrite

The one-move design, rewrite the query and try again, is a retry loop with a vocabulary. Every
failure gets the same response, so a failure that a different wording cannot fix is never
fixed.

| The failure | What a rewrite does | What is actually needed |
|---|---|---|
| An error code missed by semantic search | Rewords the code | Search lexically instead |
| A sub-question that is still two questions | Rewords both at once | Split it |
| An answer spread across a whole policy | Sharpens the search for fragments | Read the document whole |
| A corpus that does not contain the fact | Produces attempt four of four | Say so, and stop |

Six moves cover those: `broaden`, `narrow`, `switch_strategy`, `decompose`, `fetch_document`
and `abandon`. Three of them do not touch the query text at all. ADR 0010 gives the full
policy, including the order the code falls back to when the model's proposal is unusable, and
the rule that a move is never repeated on a sub-question that has not changed.

## What this costs

A single-shot strategy makes one model call to answer. This one makes a call to plan, a call
to assess per iteration, and a call per failed sub-question per iteration to choose a repair.
Those are bounded by the iteration cap, the global call cap and the per-node caps, all of
which are configuration with defaults in `ragfabric.example.yaml`, and all of which are
refused by `AgentState.spend` before the call is made rather than noticed after.

It is more expensive and slower per question, by construction. Whether it retrieves better is
a measurement, and there is no measurement yet. Per
[ADR 0004](../adr/0004-measurement-first-no-fabricated-numbers.md) no figure for it appears
anywhere in this repository until the evaluation framework in Phase 8 produces one.
