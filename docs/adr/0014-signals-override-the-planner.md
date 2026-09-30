# ADR 0014: Decisive signals override the agent's planner

Status: accepted
Date: 2026-09-30
Supersedes: none
Related: ADR 0004 (measurement first, no fabricated numbers), ADR 0009 (a plain Python state
machine), ADR 0010 (repair policy and termination), ADR 0013 (the router is a strategy)

## Context

The agent's planner is a model. It decomposes a question and proposes a tool for each sub-question,
and a small model does this poorly a meaningful fraction of the time. The recorded evidence is
`docs/learning/agentic-first-run.md`: with `llama3.1:8b`, the planner sent both sub-questions to
`lexical_search`, including a paraphrase question with no identifier in it, which is the case
`semantic_search` exists for. That is one run, not a rate.

The router already has a free, deterministic way to read a question: the signals module. The
question here is how the agent should combine its planner with that module.

## Decision

The planner proposes, decisive signals override, and every override is traced. A `tool_check` step
runs between `plan` and `retrieve`, costs no model call, and looks at each sub-question once.

| Option | Verdict |
|---|---|
| Signals as a hint in the plan prompt | Rejected. A weak model can ignore a hint, which is exactly the failure the first run recorded |
| The router decides every tool, the planner only decomposes | Rejected. Up to one extra model call per sub-question on every agentic run |
| **The planner proposes, decisive signals override, the override is traced** | **Chosen.** No extra model call. The worst case is the planner's own choice, which is what the agent did before this change |

### Two refinements, and why each exists

| Refinement | Rule | Why |
|---|---|---|
| **`from_rule`** | A signals proposal carries `from_rule`, true only when exactly one usable signal actually fired. The tool check overrides the planner only when the proposal is decisive and `from_rule` is true | A plain short question is also "decisive", but only because nothing fired and it was short. That is a default, not a rule, and a default must not overrule a planner that may have known better. Without this, every short sub-question would have been pushed to `semantic_search` whatever the planner chose |
| **`no_exact_terms`** | A sub-question the planner sent to `lexical_search` is moved to `semantic_search` only when it has no identifier, no quoted phrase and no entity mention. The trace says `no_exact_terms` | This is the one mistake the recorded run showed, corrected narrowly. Acronyms and product names such as SSO or Kubernetes are not identifiers, yet exact-word search is often right for them, so an entity mention keeps a lexical pick |

`fetch_document` is never overridden. It names a document by id and no signal knows better.

### How it is traced

The `tool_check` span records the number of overrides and, for each, the sub-question index, the
planner's tool, the tool the signals chose and the query type or the reason `no_exact_terms`. An
override is therefore always visible, and a run with no overrides says so.

### The same ranking drives the repair

`switch_strategy` chooses among all three search tools, never the one that just failed on that
sub-question, taking the signals' ranking in order. Switching away from `fetch_document` restores
the sub-question text as the query, because the working query was the bare document id.

## A signal has no confidence number

A signal is a rule that fired, not a measurement. Giving it a confidence of 0.9 would be a number
nobody measured, which ADR 0004 forbids. A signals decision carries `confidence: null`, and
`decisive: true` (a `signals_fallback` decision has `decisive: false`). Only the classifier reports a confidence, recorded as what the model said and
uncalibrated until Phase 8 measures it.

## Consequences

- No extra model call on any agentic run. The tool check is a function over the sub-question text.
- The override can be wrong. A rule such as "an identifier means lexical" is a heuristic, and it
  will misroute a question where an identifier is incidental. The trace shows each override so
  the cases can be found, and whether the overrides help is a measurement that does not exist yet.
- `graph_search` takes part only when the deployment lists it in `strategies.agentic.tools`, turns
  `graph_store.enabled` on, and the request sets neither `document_id` nor `format`. Otherwise
  the signals are asked only about the tools the run has.
- The narrow `no_exact_terms` rule leaves some paraphrase questions on a lexical tool, because it
  errs toward the planner's choice when a named thing is present.
