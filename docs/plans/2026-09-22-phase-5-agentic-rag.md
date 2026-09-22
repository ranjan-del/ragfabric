# Phase 5: Agentic RAG, release v0.2.0. Implementation Plan

**Goal:** A retrieval agent that decomposes a question, judges its own evidence, and when it comes up short chooses a repair move and tries again, stopping honestly when it is not making progress. Not a retry wrapper.

**Architecture:** A plain Python state machine over a typed `AgentState`. Every decision the model makes is returned as JSON against a strict schema the graph validates. The repair move is chosen from a fixed enum, so a weak local model cannot invent a nonsense action. Answer verification reuses the Phase 3 citation contract rather than adding a second LLM judgement.

**Tech Stack:** Python 3.13 (uv), SQLAlchemy 2, FastAPI, Pydantic. **No new runtime dependencies.**

**Issue:** https://github.com/ranjan-del/ragfabric/issues/6

**Predecessor:** `docs/plans/2026-09-21-phase-4-vectorless-rag.md` (Vectorless RAG, merged as PR #36, released v0.1.0)

---

## Why this shape, and what was rejected

Recorded here because the reasoning is the deliverable, not just the code.

### What was wrong with the committed concept design

`docs/agentic-rag.md` describes the textbook agentic RAG: analyze, plan_need, retrieve, evaluate_evidence, rewrite_query, generate, verify_answer, finalize. It is a reasonable skeleton and it is not enough. Six specific gaps, each of which this phase closes:

| Gap | Consequence | Closed by |
|---|---|---|
| `rewrite_query` is the only repair | The agent has exactly one response to every failure. That is a retry loop wearing an agent's clothes | Task 6, a repair policy with six distinct moves |
| Evidence judged globally, not per sub-question | Decomposition is pointless if the loop then re-retrieves everything. The agent cannot chase only what is still missing | Task 2, the sub-question ledger |
| Nothing detects lack of progress | An agent that re-fetches the same chunks and calls it an iteration will burn the entire budget going nowhere. Issue #6 names this: "why stopping is the hard part" | Task 5, novelty check on chunk ids |
| No memory of why an attempt failed | "A rewritten query must differ from all previous" is a uniqueness check, not learning. Different is not smarter | Task 2, attempt history per sub-question |
| `verify_answer` routes back to `rewrite_query` | If a claim is unsupported, more retrieval is usually the wrong fix. Removing the claim is the right one | Task 12, reuse the mechanical contract, drop unsupported claims |
| `best_effort` is a single boolean | It cannot say which part failed or why | Task 13, structured partial answers |

### Nodes merged or removed, deliberately

- **`analyze` and `plan_need` merge into one `plan` node.** Two LLM calls doing adjacent work (understand the question, choose tools per part). One call does both. Saves a call and a failure point on every query.
- **`rewrite_query` disappears as a node.** Rewriting is one move inside the repair policy. Promoting it to a node hardcodes the weakest strategy as the only one.
- **`verify_answer` is not a new LLM call.** `generate/contract.py` from Phase 3 already checks marker validity, quote fidelity for quotes of eight or more characters or any quote with a digit, and grounding. It already caught a real failure: a local `llama3.2:3b` streamed a correct answer with no citation marker and the contract repaired it. A second, LLM-based verifier would duplicate that and add a judgement that can itself be wrong.

Net: roughly 4 LLM calls per question instead of 6 or more, with strictly more capability.

### Conflict detection deferred, on purpose

Semantic contradiction between two passages is a hard inference task. On `llama3.1:8b` it produces false positives, and a confident "these sources disagree" that is wrong is worse than silence. It also cannot be tuned without measurement, which is Phase 8. Per ADR 0004, shipping an unmeasurable heuristic is exactly what this project does not do.

What ships instead is the honest metadata subset: when two chunks answering the same sub-question come from documents with different effective dates, both are surfaced with their dates in the generate step. That is a fact about metadata, not a judgement about meaning.

### No LangGraph, and why

The roadmap named LangGraph. A spike built the same loop both ways and measured it.

| Criterion | Plain Python | LangGraph |
|---|---|---|
| Lines for the identical loop | 93 | 107 |
| New dependencies | 0 | 38 packages, including `langsmith` (a SaaS telemetry client), `requests`, `websockets`, `zstandard` |
| Injecting the LLM and retriever | Function arguments | Impossible through state: LangGraph strips keys not declared in the schema. Requires closures or `config` |
| Recording why the loop stopped | Set at the branch point, next to the decision | A conditional edge returns only a node name. The reason must be written separately. The spike lost it silently, returning `stop_reason=None` |

The last row decided it. Issue #6 requires a trace of every node and honest failure handling, and the framework works against that. This also follows the precedent of ADR 0007, which rejected `pg_search` despite it giving BM25 for free, because a dependency that compromises the deployment story is not worth the code it saves. Most of what LangGraph exists for, durable resumable human-in-the-loop execution, is unused by a bounded loop capped at four iterations.

Recorded as ADR 0009. `ROADMAP.md`, issue #6 and the CHANGELOG release-plan row are updated to match, because they currently promise LangGraph.

### Why JSON contracts rather than native tool calling

`LLMProvider` today has `complete` and `stream` and no tool-calling surface. Adding real function calling means a second large subsystem inside this phase, and it cannot be exercised without paid API credit that this project does not currently have.

Instead every model decision returns JSON against a strict schema. This is not a workaround, it is better here for three reasons: it works on every provider including a 3B local model, the `ScriptedLLMProvider` can return canned JSON and drive every branch deterministically offline (issue #6 requires exactly this), and a malformed response is a validation error the graph handles rather than a silent misbehaviour.

**The repair move is chosen from a fixed enum.** A weak model cannot invent an action that does not exist. The worst case is a valid but suboptimal move, which the next iteration can correct. This single decision is what makes the design survive on `llama3.1:8b`.

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.13**, line length 100, `ruff` clean (`E`, `F`, `I`, `UP`, `B`), `ruff format` clean.
- **Run `ruff format packages/`, never `ruff format .`** The repo-root form rewrites Python code blocks inside markdown and silently mangled six documents in Phase 4.
- **import-linter contracts stay at 3 kept, 0 broken.**
- **No new runtime dependencies.** If a task appears to need one, stop and say so rather than adding it.
- **ADR 0003 holds: the access filter goes INSIDE the store query.** Every tool call the agent makes carries the caller's `AccessFilter`. An agent that retrieves across iterations must not accumulate evidence the principal cannot see.
- **ADR 0004 holds: never fabricate a number.** Unknown cost is `None`. Counters report what actually happened.
- **ADR 0002 holds: one result shape.** The agent returns `RetrievalResult`.
- **Citations reuse the Phase 3 contract unchanged** (`generate/contract.py`). Do not fork it, do not loosen it.
- **SQLite remains the test default.** PostgreSQL-only work is skipped via the `RAGFABRIC_TEST_DATABASE_URL` guard, never by silently passing. Baseline on this branch is **533 passed, 6 skipped** with that variable set.
- **No AI attribution anywhere.** Commits authored `Ranjan G <ranjan.g@ispf.ngo>`, no trailers, no assistant references in code, comments, docs or PR bodies.
- **No em dashes** anywhere.
- **Tests must be able to fail.** A test that passes against a deliberately broken implementation is not a test.

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/core/src/ragfabric_core/agent/state.py` | `AgentState`, `SubQuestion`, `SubQuestionStatus`, `RepairMove`, budget accounting |
| `packages/core/src/ragfabric_core/agent/contracts.py` | JSON schemas for plan, assess and repair, with validation and malformed-output handling |
| `packages/core/src/ragfabric_core/agent/tools.py` | `AgentTool` protocol and the concrete tools |
| `packages/core/src/ragfabric_core/agent/nodes.py` | `plan`, `retrieve`, `assess`, `repair`, `generate`, `finalize` |
| `packages/core/src/ragfabric_core/agent/loop.py` | The state machine, termination and trace assembly |
| `packages/core/src/ragfabric_core/agent/policy.py` | Repair move selection and application |
| `packages/core/src/ragfabric_core/strategies/agentic.py` | `AgenticRAGStrategy`, the `RetrieverStrategy` implementation |
| `docs/concepts/agentic-loops.md` | What a state machine agent is, and why stopping is the hard part |
| `docs/adr/0009-plain-state-machine-over-langgraph.md` | The framework decision |
| `docs/adr/0010-repair-policy-and-termination.md` | Why six moves, and the three stop conditions |

---

## Task 1: Agent state and the sub-question ledger

**Files:** Create `agent/state.py`, `packages/core/tests/test_agent_state.py`

This is the foundation both tracks build on. It lands before either starts.

`SubQuestionStatus` is `open`, `answered` or `abandoned`. `RepairMove` is `broaden`, `narrow`, `switch_strategy`, `decompose`, `fetch_document`, `abandon`. `SubQuestion` carries text, the tool chosen, status, an attempt history (move plus the query tried plus why it failed), and a reason when abandoned. `AgentState` carries the original question, the sub-question list, an evidence pool keyed by chunk id, `seen_chunk_ids`, per-node call counters and the budget.

Budget accounting is a method on the state, not scattered through nodes: `state.spend(node, llm_calls=1)` raises `BudgetExceeded` when a global or per-node cap would be crossed.

- [ ] **Step 1: Write the failing tests.** `test_a_sub_question_starts_open`, `test_abandoning_requires_a_reason`, `test_spend_raises_when_the_global_llm_cap_is_crossed`, `test_spend_raises_when_a_per_node_cap_is_crossed`, `test_evidence_pool_deduplicates_by_chunk_id`.
- [ ] **Step 2: Run them, confirm they fail** with `ImportError` on `AgentState`.
- [ ] **Step 3: Implement.** Pydantic models, mirroring the style of `strategies/base.py`.
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add agent state and the sub-question ledger`

---

## Task 2: The JSON contract layer

**Files:** Create `agent/contracts.py`, `packages/core/tests/test_agent_contracts.py`

Three schemas: `PlanResponse` (sub-questions with a tool each), `AssessResponse` (per sub-question answered plus what is missing), `RepairResponse` (one `RepairMove` plus an optional rewritten query).

A model returns text. This layer extracts the JSON (tolerating a fenced code block or leading prose, which small models emit constantly), validates it, and on failure returns a typed `ContractViolation` rather than raising into the loop. The caller decides: retry once with a corrective instruction, or take a safe default.

**A malformed response must never crash a request and must never be silently treated as success.**

- [ ] **Step 1: Write the failing tests**, including `test_json_inside_a_fenced_code_block_is_accepted`, `test_leading_prose_before_the_json_is_tolerated`, `test_an_unknown_repair_move_is_a_violation_not_a_crash`, `test_a_violation_is_reported_not_swallowed`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add validated JSON contracts for agent decisions`

---

## Task 3: Agent tools over the existing strategies

**Files:** Create `agent/tools.py`, `packages/core/tests/test_agent_tools.py`

An `AgentTool` has a name, a description the planner sees, and `run(query, ctx) -> list[RetrievedChunk]`. Three tools: `semantic_search` wrapping `TraditionalRAGStrategy`, `lexical_search` wrapping `VectorlessRAGStrategy`, and `fetch_document` returning a whole document's chunks in order.

Tools are thin. They do not re-implement retrieval, they call the strategies that already exist and are already tested.

**Every tool receives and honours the caller's `AccessFilter`.** `fetch_document` is the dangerous one: fetching a document by id must apply the same access predicate as a search, or the agent becomes a way to read documents the principal cannot search.

- [ ] **Step 1: Write the failing tests**, including `test_fetch_document_refuses_a_document_the_principal_cannot_read` and `test_each_tool_passes_the_access_filter_through`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add agent tools over the traditional and vectorless strategies`

---

## Task 4: The plan node

**Files:** Create `agent/nodes.py` (plan only), `packages/core/tests/test_agent_plan.py`

One LLM call decomposes the question into sub-questions and assigns a tool to each. A simple question yields exactly one sub-question: decomposition is not mandatory, and forcing it on "what is the retry limit" wastes calls.

The prompt states the available tools and when each fits: identifiers and error codes to `lexical_search`, conceptual or paraphrased questions to `semantic_search`, "the whole policy" to `fetch_document`.

- [ ] **Step 1: Write the failing tests**, including `test_a_simple_question_yields_one_sub_question`, `test_a_comparison_question_is_decomposed`, `test_an_identifier_question_is_routed_to_lexical_search`, `test_a_malformed_plan_falls_back_to_a_single_sub_question`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add the plan node with question decomposition`

---

## Task 5: Retrieve node, evidence pool and progress detection

**Files:** Modify `agent/nodes.py`, create `packages/core/tests/test_agent_progress.py`

Retrieve runs the tool for each open sub-question, adds chunks to the evidence pool keyed by chunk id, and records which chunk ids are new this iteration.

**Progress detection is the point of this task.** If an iteration adds zero new chunk ids, the loop is not progressing and must not simply try again. This is the failure mode the textbook design has no answer to.

- [ ] **Step 1: Write the failing tests**, including `test_an_iteration_that_adds_no_new_chunks_is_marked_no_progress`, `test_evidence_is_deduplicated_across_iterations`, `test_only_open_sub_questions_are_retrieved_for`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Mutation check.** Make the novelty check compare counts instead of ids and confirm `test_an_iteration_that_adds_no_new_chunks_is_marked_no_progress` fails. Revert.
- [ ] **Step 5: Commit** `feat: add the retrieve node with evidence pooling and progress detection`

---

## Task 6: The repair policy

**Files:** Create `agent/policy.py`, `packages/core/tests/test_agent_policy.py`

The heart of the phase. Six moves, each with a defined effect on the sub-question:

| Move | Effect | When it fits |
|---|---|---|
| `broaden` | Relax the query, drop qualifiers, raise `top_k` | Retrieval returned nothing |
| `narrow` | Add constraints from the question | Retrieval returned much, none of it on point |
| `switch_strategy` | Flip `semantic_search` to `lexical_search` or back | An identifier missed semantically, or a paraphrase missed lexically |
| `decompose` | Split this sub-question further | It is still compound |
| `fetch_document` | Pull a whole document already partially matched | Evidence is fragmentary and the right document is identified |
| `abandon` | Mark abandoned with a reason | Nothing is working, and saying so beats looping |

**A move must never be repeated for the same sub-question without something else having changed**, or the agent oscillates between broaden and narrow forever. The attempt history enforces this.

- [ ] **Step 1: Write the failing tests**, one per move, plus `test_the_same_move_is_not_repeated_on_an_unchanged_sub_question`, `test_switch_strategy_flips_the_tool`, `test_abandon_records_a_reason`, `test_exhausting_the_moves_abandons_rather_than_loops`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Mutation check.** Remove the repeat guard and confirm `test_the_same_move_is_not_repeated_on_an_unchanged_sub_question` fails. Revert.
- [ ] **Step 5: Commit** `feat: add the repair policy with six distinct recovery moves`

---

## Task 7: The assess node

**Files:** Modify `agent/nodes.py`, create `packages/core/tests/test_agent_assess.py`

One LLM call returns, per open sub-question, whether the evidence answers it and what is missing if not. A structured verdict, never prose.

The rubric is strict: evidence answers a sub-question only if the answer can be read out of the retrieved text. "Related to the topic" is not answered. A permissive assessor is how an agent ends up confidently wrong, and this is the node most worth being pessimistic in.

- [ ] **Step 1: Write the failing tests**, including `test_topically_related_evidence_is_not_counted_as_answered`, `test_a_missing_description_is_carried_into_the_repair`, `test_a_malformed_assessment_leaves_the_sub_question_open`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add the assess node with a strict evidence rubric`

---

## Task 8: The loop, termination and trace

**Files:** Create `agent/loop.py`, `packages/core/tests/test_agent_loop.py`

The state machine. Three stop conditions, and the result records which one fired:

1. **resolved**, every sub-question answered or abandoned
2. **budget**, a global or per-node cap reached
3. **no_progress**, an iteration added no new evidence

Every node appends a `TraceSpan` with its name, duration and the counters it changed. The stop reason is recorded where the branch is decided, not inferred afterwards.

- [ ] **Step 1: Write the failing tests**, including `test_the_loop_terminates_on_budget`, `test_the_loop_terminates_on_no_progress`, `test_the_loop_terminates_when_all_sub_questions_resolve`, `test_the_stop_reason_is_recorded`, `test_every_node_appends_a_trace_span`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Mutation check.** Remove the no-progress check and confirm the loop runs to the budget on a static retriever. Revert.
- [ ] **Step 5: Commit** `feat: add the agent loop with three explicit termination conditions`

---

## Task 9: AgenticRAGStrategy

**Files:** Create `strategies/agentic.py`, modify `strategies/registry_defaults.py`, create `packages/core/tests/test_strategy_agentic.py`

Implements `RetrieverStrategy`: `retrieve(query, ctx) -> RetrievalResult`. Runs the loop, returns the accumulated evidence as `chunks`, and reports real `llm_calls`, `retrieval_calls`, `embedding_calls`, tokens and `latency_ms`. Registered as `agentic` in `registry_defaults.py`.

`embedding_calls` is whatever the semantic tool actually spent, and zero when the agent used only lexical tools. Per ADR 0004 it is counted, never estimated.

- [ ] **Step 1: Write the failing tests**, including `test_counters_report_actual_calls`, `test_embedding_calls_is_zero_when_only_lexical_tools_ran`, `test_the_access_filter_reaches_every_tool_call`, `test_the_strategy_is_resolvable_from_the_registry`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement and register.**
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add AgenticRAGStrategy over the bounded agent loop`

---

## Task 10: Scripted JSON doubles and full branch coverage

**Files:** Modify `providers/offline.py`, create `packages/core/tests/test_agent_branches.py`

Extend `ScriptedLLMProvider` so a test can queue structured JSON responses per node. Then drive every branch offline and deterministically, which is what issue #6 asks for.

Branches covered: each of the six repair moves, budget exhaustion, no-progress stall, malformed JSON at each of the three contracts, a sub-question abandoned with a reason, and a fully resolved multi-sub-question run.

- [ ] **Step 1: Write the failing tests.**
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement the provider extension.**
- [ ] **Step 4: Confirm every branch is exercised** and that the suite runs with no network and no database.
- [ ] **Step 5: Commit** `test: drive every agent branch with scripted JSON doubles`

---

## Task 11: Configuration

**Files:** Modify `config_file.py`, `ragfabric.example.yaml`, create `packages/core/tests/test_config_agentic.py`

Typed settings: `max_iterations` (default 4), per-node LLM call caps, `max_cost_usd`, `max_latency_ms`, which tools are enabled, and the assess strictness. Strict validation, extra keys rejected.

- [ ] **Step 1: Write the failing tests**, including `test_an_unknown_agent_key_is_rejected` and `test_max_iterations_below_one_is_rejected`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**, and document every key with a one-line comment in `ragfabric.example.yaml`.
- [ ] **Step 4: Verify `ragfabric config validate` reports the agent settings.**
- [ ] **Step 5: Commit** `feat: type the agentic strategy configuration`

---

## Task 12: Generation, the citation contract and dropped claims

**Files:** Modify `generate/cited.py` as needed, create `packages/server/tests/test_agentic_generation.py`

Generation runs over the evidence pool and reuses the Phase 3 citation contract unchanged. When the contract finds an unsupported claim, the claim is **removed** and the removal is recorded. The agent does not loop back to retrieval for it, because more retrieval is usually not the fix.

Where two chunks answering one sub-question come from documents with different effective dates, both are presented with their dates. This is the whole of the conflict handling that ships in Phase 5.

- [ ] **Step 1: Write the failing tests**, including `test_an_unsupported_claim_is_removed_not_retried` and `test_two_dated_sources_for_one_sub_question_are_both_surfaced`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: generate agentic answers with dropped unsupported claims`

---

## Task 13: Structured partial answers

**Files:** Modify `strategies/base.py` (add an optional report field), `agent/loop.py`, tests

`RetrievalResult` gains `sub_questions: list[SubQuestionReport]`, defaulting to empty so no other strategy changes. Each report carries the sub-question, its final status, and for anything not answered the reason: budget, no evidence, or abandoned after N moves.

This replaces `best_effort` as a boolean. The caller can say exactly what was and was not answered.

- [ ] **Step 1: Write the failing tests**, including `test_an_unanswered_sub_question_reports_why` and `test_other_strategies_return_an_empty_report`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite** and confirm no Phase 3 or 4 strategy test regressed.
- [ ] **Step 5: Commit** `feat: report per sub-question outcomes instead of a best effort flag`

---

## Task 14: API, CLI and SDK surface

**Files:** Modify server schemas and routes, `packages/cli/src/ragfabric_cli/commands/ask.py`, the SDK

Add `agentic` to the strategy choices on `POST /api/ask` and `POST /api/search/query`, to `ragfabric ask --strategy agentic`, and to the SDK. `POST /api/search/hybrid` continues to accept only `traditional`.

The agent's trace and sub-question report are returned on the ask response.

- [ ] **Step 1: Read the current request and response schemas and write down their real fields.**
- [ ] **Step 2: Write the failing tests**, including `test_the_ask_response_carries_the_sub_question_report` and `test_hybrid_still_refuses_agentic`.
- [ ] **Step 3: Run them, confirm they fail.**
- [ ] **Step 4: Implement across route, CLI and SDK.**
- [ ] **Step 5: Run the full suite** and confirm no Phase 4 route test regressed.
- [ ] **Step 6: Commit** `feat: expose the agentic strategy over the API, CLI and SDK`

---

## Task 15: A real end to end run against Ollama

**Files:** Create `packages/core/tests/test_agentic_integration.py` (marked `integration`)

Skipped unless `RAGFABRIC_TEST_OLLAMA` is set. Ingest a small fixture corpus, ask a genuinely multi-part question, and record what actually happened: how many sub-questions, which tools, which repair moves fired, how it terminated.

**Per ADR 0004, whatever this run produces is reported as what happened on that corpus with that model. It is not a benchmark and must not be quoted as one.** If `llama3.1:8b` plans badly, that is a finding to write down, not to hide.

- [ ] **Step 1: Write the test and the fixture corpus.**
- [ ] **Step 2: Run it against real Ollama and record the actual trace.**
- [ ] **Step 3: Write the findings into the learning notes**, including anything the model did badly.
- [ ] **Step 4: Commit** `test: exercise the agent end to end against a local model`

---

## Task 16: Documentation and ADRs

**Files:** Create `docs/concepts/agentic-loops.md`, `docs/adr/0009-plain-state-machine-over-langgraph.md`, `docs/adr/0010-repair-policy-and-termination.md`; rewrite `docs/agentic-rag.md`; update `docs/README.md`, `ROADMAP.md`, issue #6

`docs/agentic-rag.md` currently describes the eight-node design that this phase deliberately did not build. It is rewritten to describe what shipped.

ADR 0009 records the LangGraph decision with the spike's real numbers. ADR 0010 records why six moves and three stop conditions.

**One implementer at a time in the worktree.** Two agents sharing a tree means one runs `git add -A` over the other's half-finished work. This is recorded in Phase 4's notes and nearly happened again.

- [ ] **Step 1: Write both ADRs.**
- [ ] **Step 2: Rewrite `docs/agentic-rag.md` to match what shipped.**
- [ ] **Step 3: Update `ROADMAP.md` and issue #6**, ticking Phase 5 only for what actually merged, and correcting the LangGraph promise.
- [ ] **Step 4: Re-read every changed doc against the code.**
- [ ] **Step 5: Commit** `docs: record the agent loop and framework decisions`

---

## Task 17: Release v0.2.0 preparation

**Files:** `CHANGELOG.md`, version fields across the packages

**Publishing is GATED.** Prepare in the working tree only. Tagging, pushing to main, publishing images and creating a release each need explicit per-action approval from the owner at the time. Do not read a standing permission out of this document.

- [ ] **Step 1: Write the `[0.2.0]` changelog section**, every entry traceable to a merged commit, heading dated `unreleased` until tag time.
- [ ] **Step 2: Set the version to `0.2.0`** across core, server, cli and sdk-python, and confirm `uv.lock` agrees.
- [ ] **Step 3: Verify the image still builds and runs** with the agent strategy reachable.
- [ ] **Step 4: Stop.** Report and wait for approval on the release steps.

---

## Parallel execution

Task 1 lands first; both tracks depend on it.

| Track | Tasks | Area |
|---|---|---|
| A, agent core | 4, 5, 6, 7, 8, 9 | plan, retrieve, repair policy, assess, loop, strategy |
| B, substrate | 2, 3, 10, 11 | contracts, tools, scripted doubles, config |

The two meet at the interfaces defined in Tasks 1, 2 and 3, which is why those are specified before the fork. Each track works in **its own worktree on its own branch**, merged when both land. Tasks 12 to 17 run sequentially afterwards, one implementer at a time.

---

## Self-Review

Before the phase is called done:

- Does the agent ever loop without adding evidence? It must not.
- Can a malformed model response crash a request? It must not.
- Does `fetch_document` honour the access filter? Verify by test, not by reading.
- Are all counters real, or is any of them defaulted to a plausible number?
- Does the sub-question report say why, for every unanswered part?
- Is any quality claim made anywhere without a measured run behind it?
