# What a real routing run looked like, learning notes

> Status: Phase 7a (in progress on this branch). A record of **one** run of `strategy: auto`
> against a local model, written down because the offline tests cannot tell you this. See
> [agentic-first-run.md](agentic-first-run.md) and
> [graph-extraction-first-run.md](graph-extraction-first-run.md) for the same exercise on
> Phases 5 and 6. The comparison that matters is with the first of them.

## This is a record, not a benchmark

Per ADR 0004, everything below is what happened on a fifteen-chunk fixture corpus, four
questions, `llama3.1:8b` for every model call and `nomic-embed-text` for embeddings, on one
machine, on 2026-09-30. It is a single run of a non-deterministic system. Nothing here may be
quoted as a measurement of the router, of routing accuracy, or of any model's ability to
route, and nothing here may be used to tune `router.min_confidence` (0.6, untuned) or any
signal rule. The corpus and the four questions were fixed before the run and were not edited
afterwards. Retrieval quality in this project is unmeasured until the Phase 8 evaluation
framework exists.

The offline suite proves every branch of the router by handing it JSON a model is told to
produce. This run asks a different question: when a real small model sits behind the
classifier and behind the agent, what does `auto` actually do.

The committed test is `packages/core/tests/test_router_integration.py`. It prints the whole
record as JSON between two marker lines, and writes it to the path in
`RAGFABRIC_ROUTING_RECORD` when that is set. It asserts only what must hold whatever the model
does: `auto` never reports itself as the strategy, every `router.reasoning` is at most 200
characters, no chunk of the denied document appears on any path, a fallback names a reason,
and `llm_calls` equals the number of model calls counted by a wrapper around the provider.
All of those held. One note on the run itself: the first execution of the test printed and
saved the record below and then stopped on a leftover assertion that referenced a module not
imported (a mistake in the test, not a finding about the router). The assertion was removed
and the committed test was then run a second time with Ollama and passed. Every routing decision,
planned tool, tool_check override, repair move, stop reason and call counter in that second run
was identical to the record below; only latencies differed (the compound question took 33102ms
instead of 44841ms, and stopped on the same latency budget). The tables below are from the first
execution and are unchanged. The second run shows the decisions repeated on this machine; it
is still one model and one corpus.

## The corpus and the questions

The Phase 5 fixture (nine chunks in three documents) and the Phase 6 graph fixture (six chunks
in three documents), in one collection, ingested and embedded through the real code path, then
run through real graph extraction and resolution with `graph_store.enabled: true`. The agent
was given four tools: `semantic_search`, `lexical_search`, `fetch_document` and `graph_search`.
As in Phase 5, `leave-policy.txt` was **denied** by the caller's access filter on every
question.

Extraction and resolution, from the same run: `entities_stored=31, relationships_stored=19,
entities_discarded=0, relationships_discarded=0, contract_violation=True, chunks_skipped=0`,
and twelve of fifteen extraction calls completed. Chunks 4, 6 and 15 did not: with
`chunks_skipped=0`, each of those responses violated the extraction contract and was rolled back. One
entity merge, by embedding. Extraction quality is not the subject of this record; it is
reported because the graph path below depends on it.

| Label | Question |
|---|---|
| phase5_compound | How many times is a failed indexing job retried before it is dead lettered, and how many days of unused annual leave carry forward? (Phase 5, verbatim) |
| relationship | Who does Ravi Sharma report to? |
| identifier | What does error RF-4312 mean? |
| long_vague | I have been wondering about how things generally work around here when somebody new joins the engineering group, and what kind of things they are expected to read and get familiar with during the first few days of settling in. |

## What happened, per question

| | phase5_compound | relationship | identifier | long_vague |
|---|---|---|---|---|
| Decision | agentic | graph | traditional | traditional |
| Source | signals | signals | signals | classifier |
| Decisive | true | true | true | false |
| Confidence | none | none | none | 0.8 (what the model reported, uncalibrated) |
| Query type | aggregation | relationship | simple_factual | simple_factual |
| Reasoning | The question asks for a count or a total across sources. | The question asks how named things are related. | A short question about one concept. | The question is asking for general information about the onboarding process, which is a straightforward factual question. |
| Strategy that ran | agentic | graph | traditional | traditional |
| Fallback | none | none | none | none |
| llm_calls | 3 | 1 | 0 | 1 |
| retrieval_calls | 2 | 5 | 1 | 1 |
| embedding_calls | 2 | 0 | 1 | 1 |
| input / output tokens | 1390 / 248 | 229 / 54 | 11 / 0 | 300 / 55 |
| latency_ms | 44841 | 6847 | 55 | 7067 |
| Chunks returned | 5 | 2 | 4 | 4 |

For `phase5_compound`, the agent's own record:

```
  0ms   +1ms      router     selected=agentic source=signals decisive=True
  4ms   +24236ms  plan       sub_questions=2  tools=lexical_search  fallback=False
24240ms +0ms      tool_check overrides=2
                             0:lexical_search->semantic_search(no_exact_terms)
                             1:lexical_search->semantic_search(no_exact_terms)
24241ms +139ms    retrieve   tool_calls=2  returned=8  new_chunks=5  progress=True
24381ms +12910ms  assess     judged=2  answered=1  unjudged=0  strictness=strict
37292ms +7547ms   repair     sub_question=1  proposed=decompose  move=narrow
44839ms +0ms      finalize   stop_reason=budget
                             max_latency_ms of 30000 reached after 44839ms
```

Sub-question 0 ("How many times is a failed indexing job retried before it is dead
lettered?") was answered, with chunk 2 (the runbook chunk that states the answer) in its
evidence. Sub-question 1 ("How many days of unused annual leave carry forward?") was left
open: "4 chunks were retrieved but none answered it; the run stopped on budget".
No `switch_strategy` move was proposed on any question, and no fallback fired on any question.

The full JSON record is in the task report; the values in this document are copied from it.

## The comparison: did the tool check correct the Phase 5 mistake

Yes, on the routing decision, and the record does not show that it mattered to the outcome.

In Phase 5, `llama3.1:8b` sent a paraphrase sub-question about annual leave to
`lexical_search`. In this run the planner did the same thing again: the plan span lists
`tools=lexical_search`, for both sub-questions. The tool check that now runs after the plan
changed both to `semantic_search`, with the detail
`no_exact_terms`: neither sub-question contained an identifier, a quoted phrase or a named
thing, so the check judged that exact-word search could only miss a paraphrase.

Two honest qualifications. First, both overrides came from the `no_exact_terms` correction,
not from a signal rule firing; that is the branch the design added for exactly this mistake,
and it is the branch that ran. Second, the outcome looks like Phase 5's. Sub-question 0 was
answered and sub-question 1 was not, the repair node again proposed `decompose` and the policy
again applied `narrow`, and the run again stopped on the latency budget. The leave answer lives
in a document the filter denies, so no tool choice could have answered it, and the pooled
chunks for it are the same kind of distractors as Phase 5 (chunks 8, 9, 7 and 1: two onboarding
chunks that mention leave without stating the number, the handbook index line, and a runbook
chunk about the job queue). This run has no counterfactual
(the same question with the check disabled), so it cannot say whether `semantic_search` found
the retry answer more reliably than `lexical_search` would have. The check corrected the
choice; whether the correction helped is not something one run can show.

## Findings, including the unflattering ones

**1. The identifier question did not reach the identifier rule.** "What does error RF-4312
mean?" was routed to `traditional` by the plain short-question default, with reasoning "A
short question about one concept", not to `vectorless` as the exact-match rule intends. The
cause is in the existing identifier detection, not in the router: `RF-4312` is letters, a
hyphen and digits, and none of the four documented patterns in
`ragfabric_core.stores.boosting` (`is_identifier`) match it (the `_DIGIT_AND_LETTER` pattern forbids a hyphen and
the version pattern needs a leading digit). Checked directly: the signals extractor reports
`identifiers=()` and `entities=('RF-4312',)`. The same code written `RF4312` is detected. The
router is correct given what it was told, and it inherits a gap in the Phase 4 boosting rule. The
traditional strategy did return the right chunk first (chunk 3, the runbook's RF-4312 line),
so this run does not show a wrong answer, only that the exact-match rule did not fire on the
most identifier-shaped question in the set. It is recorded, not fixed here.

**2. The planner repeated the Phase 5 mistake.** The 8B model again chose `lexical_search` for
paraphrase sub-questions, and `graph_search`, although listed, was never planned on the
compound question. The planner is no better than in Phase 5 on this point. What changed is
that there is now a second layer that notices.

**3. The classifier was reached by one question and called it simple.** The long vague
question matched no rule, so it went to the model, which chose `traditional` with reported
confidence 0.8, above the 0.6 floor, so nothing was fused. The model's own `query_type` was
`simple_factual`, which is an odd label for a question that was written to be vague. The 0.8
is what the model said; it is not calibrated and this single value says nothing about how
often it is right. The four chunks returned look topically related (two onboarding chunks, a
team chunk, a management chunk), and this run did not check them against an answer.

**4. The router's own cost is one model call, and only on one question.** Three of four
decisions came from signals with no model call. The classifier spent one call and about 7s on
`long_vague`. The graph strategy spent one call (question entity extraction, 6.8s of the
6.8s total) on `relationship`. The compound question spent three, all inside the agent.

**5. The graph path found the answer chunk but only matched one of two names.** The
relationship question returned chunk 10 ("Ravi Sharma ... reports to Meera Iyer") and chunk 12
(Anjali Sharma, who is not the subject). The walk matched one of two extracted mentions, found
2 nodes and 1 edge, and was not truncated. Three chunks (4, 6 and 15) failed extraction on a contract
violation, including chunk 15, which carries the second reporting hop, so
a two-hop version of this question would have been answered from a thinner graph than the
fixture intends. That is a Phase 6 finding that this run re-observes.

**6. The latency cap overshot again, by more.** `max_latency_ms` is 30000 and the compound run
stopped at 44839ms. As in Phase 5 the check runs between nodes, so the `assess` call (12.9s)
and the `repair` call (7.5s) carried the run past it. The `plan` call took 24.2s against
13.8s in Phase 5. The plan prompt now lists one more tool, but one run cannot say that is the
cause, and the machine's load was not controlled.

**7. Access control held on every path.** No chunk of `leave-policy.txt` appeared in any
result, including the graph and agent paths, and the agent again left the leave sub-question
open with a reason rather than answering from the loosely related onboarding chunks.

## What this run does not tell you

Whether `auto` picks the right strategy more often than a fixed default. Whether the
classifier's 0.8 means anything. Whether any of the four answers was correct. How often the
no_exact_terms correction helps, or whether it ever hurts (for example on a short lexical
question with an acronym the entity rule misses). Whether a fallback works on a real model:
none fired here, so the fallback paths are proven only by the offline tests. Whether a
different model plans or classifies better. Those need a labelled corpus and metrics, which is
Phase 8.
