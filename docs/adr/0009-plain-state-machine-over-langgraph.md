# ADR 0009: A plain Python state machine for the agent, not LangGraph

Status: accepted
Date: 2026-09-22
Supersedes: none
Related: ADR 0002 (one common retrieval result), ADR 0003 (access control inside retrieval), ADR 0004 (measurement first, no fabricated numbers), ADR 0007 (BM25 computed in SQL), ADR 0010 (repair policy and termination)

## Context

Phase 5 adds a retrieval agent: a bounded loop that decomposes a question, judges its own
evidence per part, and when it comes up short chooses a repair move and tries again. The
roadmap named LangGraph for it. So did issue #6, and so did the release plan row in the
changelog. The framework was written into the plan before anyone had built the loop.

Issue #6 also states two requirements that turn out to be the deciding ones:

- **a trace of every node**, so a caller can see the path a run actually took
- **honest failure handling**, so a run that stops short says which of its limits stopped
  it, rather than returning a short answer that looks like a complete one

Before committing to the dependency, a spike built the identical loop both ways, with the
same nodes, the same state shape and the same termination rules, and the two were measured
against each other.

| Criterion | Plain Python | LangGraph |
|---|---|---|
| Lines for the same loop | 93 | 107 |
| New dependencies | 0 | 38 packages, including `langsmith` (a SaaS telemetry client), `requests`, `websockets`, `zstandard` |
| Injecting the LLM and the retriever | Function arguments | Not possible through state: LangGraph strips keys that are not declared in the state schema. It needs closures or `config` |
| Recording why the loop stopped | Set at the branch point, beside the decision | A conditional edge returns only a node name. The reason has to be written somewhere else, and in the spike it silently was not: the run returned `stop_reason=None` |

The first row is close enough to be noise. The second is a cost. The last row is the one
that decided it.

## Decision

The agent is a plain Python state machine. No graph framework, and no new runtime
dependency of any kind.

- `agent/state.py` holds `AgentState`, a Pydantic model carrying the question, the
  sub-question ledger, the evidence pool and the budget counters. Nodes receive it and
  mutate it. Nothing strips a field, because nothing is declaring a reduced schema.
- `agent/nodes.py` holds `plan`, `retrieve` and `assess` as plain functions. Each takes the
  state and its collaborators as ordinary arguments, does one thing, and returns an outcome
  carrying a `TraceSpan`. No node decides whether to continue.
- `agent/loop.py` holds `run_agent`, which is the branching, plus the `repair` step, which
  lives there because it is the one step that needs the repair policy.
- `agent/policy.py` holds the repair move selection, and `agent/tools.py` holds the three
  tools. Both are ordinary modules with ordinary imports.

Two properties follow directly from the shape, and they are the reason for it:

**The stop reason is assigned at the branch that decides it.** `run_agent` sets `stop` and
`detail` at the point the condition is found true, carries them to the end, and writes them
into the finalize span and onto the state. There is no function anywhere that inspects a
finished run and works out why it ended. That is exactly what the framework version could
not do, because the return value of a conditional edge is a routing decision and not a
record.

**The model and the tools are arguments.** `run_agent(question, llm=..., tools=..., ctx=...)`
takes them by keyword. The caller's `RetrievalContext`, filter included, is passed through to
every tool call untouched, which is what ADR 0003 requires of an agent that pools evidence
across iterations. Threading that through a framework's `config` channel would have made the
access filter a thing to remember to carry rather than a parameter that cannot be dropped.

There is a second decision inside this one. **Every decision the agent delegates to a model
comes back as JSON validated against a schema in `agent/contracts.py`, and there is no native
tool calling.** `LLMProvider` exposes `complete` and `stream` and nothing else; building a
tool-calling surface would have been a second large subsystem inside this phase, and one that
cannot be exercised without paid API credit this project does not have. The JSON contract
works on every provider including a 3B local model, and it lets the scripted test double
drive every branch offline, which issue #6 asks for directly. The repair move in particular
is parsed into a fixed six-value enum, so a weak model cannot invent an action: the worst it
can do is name a valid move that fits the situation poorly, and the next iteration can correct
that.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| LangGraph, as the roadmap promised | Measured in the spike against the same loop. It cost 38 packages, could not carry the model and the tools through state, and lost the stop reason. The last of those is not a style complaint: issue #6 asks for honest failure handling, and a run that cannot say why it ended is the failure this phase exists to remove. This follows the precedent of ADR 0007, which rejected `pg_search` even though it gave BM25 for free, because a dependency that compromises the deployment story is not worth the code it saves. |
| LangGraph, with the stop reason written to a side channel | The workaround is real and it works: write the reason into the state before returning the node name from the conditional edge. It also means the reason is recorded next to the branch by convention rather than by construction, and the spike demonstrated what happens when the convention is not followed. The remaining 38 packages would then be buying 14 fewer lines. |
| LangGraph, with the LLM and tools bound by closures | Works, and moves the dependencies a node needs out of its signature into the enclosing scope, where nothing declares them and a test cannot substitute them without rebuilding the graph. The access filter is one of those dependencies. |
| A generic in-house graph abstraction, nodes plus edges plus a runner | The framework's shape without the framework's dependencies. Rejected because the abstraction would have exactly one user, and a bounded loop with one linear phase and one repair fan-out does not need a general edge resolver to express it. `run_agent` is a `while` loop with four `break` statements and it reads as one. |
| Native provider tool calling instead of JSON contracts | It is the better interface where it exists. It does not exist on `LLMProvider`, it is not uniformly available across the providers this project targets, it degrades badly on small local models, and it cannot be tested here without paid credit. A malformed JSON response is a `ContractViolation` the loop handles; a provider-specific tool-call failure is a second error surface per provider. |
| Free-text repair moves rather than a fixed enum | More expressive, and unimplementable: a move the policy has never heard of has no defined effect on the sub-question, so it would either be dropped silently or crash the request. The enum makes the invalid case impossible rather than handled. |
| Keep LangGraph anyway, because the roadmap said so | The roadmap is a plan, not a measurement. Per ADR 0004 the spike's numbers outrank the intention, and the roadmap, issue #6 and the changelog release-plan row are corrected instead. |

## Consequences

- **Zero new runtime dependencies for the whole of Phase 5.** The agent runs on what Phases
  1 to 4 already installed. No telemetry client is linked into the process by transitive
  dependency, which matters for a self-hosted deployment that has not agreed to one.
- **The trace is a list this code appends to.** Every node returns a `TraceSpan`, `run_agent`
  records each one in order, and the finalize span carries the stop reason, its detail, the
  iteration count, the size of the evidence pool and how many sub-questions ended answered or
  abandoned. Nothing is reconstructed.
- **Durable, resumable, human-in-the-loop execution is not available**, and that is the bulk
  of what LangGraph exists for. A run here is a single synchronous call capped at four
  iterations. If a future phase needs to pause a run for human input and resume it hours
  later, this decision is the one to revisit, and the loop is small enough to be rewritten
  rather than adapted.
- **Concurrency is not free.** The loop retrieves for open sub-questions one after another.
  A graph runner would have offered parallel node execution as a configuration flag. Here it
  would be written, and it has not been, because the retrieval calls hit the same database
  session factory and parallelising them is a change to the store layer rather than to the
  loop.
- **The loop is the thing to read.** There is no framework documentation standing between an
  engineer and the control flow, and equally no framework convention keeping the control flow
  tidy. `run_agent` has to stay short enough to read in one sitting, and the node functions
  have to stay free of branching, or this decision starts costing what it saved.
- **No performance or quality claim is made here.** The spike measured lines of code and
  dependency count, which are facts about the two implementations. Whether the agent
  retrieves better than a single-shot strategy is a measurement, and it belongs to the
  evaluation framework in Phase 8 (ADR 0004).
