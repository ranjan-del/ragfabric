# What a real agent run looked like, learning notes

> Status: Phase 5 (shipped). A record of **one** run of the agentic strategy against a local model,
> written down because the offline tests cannot tell you this. See
> [concepts/agentic-loops.md](../concepts/agentic-loops.md) for how the loop works.

## This is a record, not a benchmark

Per ADR 0004, everything below is what happened on a nine-chunk fixture corpus, with
`llama3.1:8b`, on one machine, on 2026-09-22. It is a single run of a non-deterministic system.
Nothing here may be quoted as a measurement of the agentic strategy, compared against another
strategy, or used to claim any model plans well or badly in general. Retrieval quality in this
project is unmeasured until the Phase 8 evaluation framework exists.

The offline suite proves every branch of the loop by feeding it JSON that a model is *told* to
produce. This run answers a different question: does a real small model produce that JSON at all,
is its plan sane, and does the loop still stop honestly when it is not.

## The question, and the trap in it

> How many times is a failed indexing job retried before it is dead lettered, and how many days of
> unused annual leave carry forward?

Two unrelated halves, deliberately. The retry answer is in a document the caller may read. The
leave answer is in `leave-policy.txt`, which the caller's access filter **denies**. So the honest
outcome is: answer the first half, and say plainly that the second could not be answered.

## What the agent did

```
llm_calls 3   retrieval_calls 2   embedding_calls 0   chunks pooled 5   latency 33008ms

  0ms   +13819ms  plan      sub_questions=2  tools=lexical_search  fallback=False
13820ms +81ms     retrieve  tool_calls=2  returned=6  new_chunks=5  progress=True
13902ms +11169ms  assess    judged=2  answered=1  unjudged=0  strictness=strict
25071ms +7935ms   repair    sub_question=1  proposed=decompose  move=narrow
33007ms +0ms      finalize  stop_reason=budget  max_latency_ms of 30000 reached after 33007ms
```

Sub-question 0 was answered. Sub-question 1 was left open with the reason recorded:

> 2 chunks were retrieved but none answered it; the run stopped on budget
> (max_latency_ms of 30000 reached after 33007ms)

## Five findings, including the unflattering ones

**1. The model routed both halves to `lexical_search`.** "How many days of unused annual leave
carry forward" is a paraphrase question with no identifier in it, and `semantic_search` is the tool
written for exactly that. The planner picked lexical anyway. This is a planning weakness of a small
model, and it is the single clearest argument for the Phase 7 router: the agent is only as good as
its tool choice, and an 8B model chooses poorly some of the time.

**2. The repair node proposed a move that made no sense, and the policy overrode it.** The model
asked to `decompose` a sub-question that was already atomic. The policy does not offer a move that
cannot do anything, so it applied `narrow` instead. This is the fixed-enum design working exactly as
intended: the worst a weak model can do is pick a valid move that is a poor fit, and the policy
catches the subset that are impossible rather than merely suboptimal.

**3. The latency cap fired, and it overshot.** The limit is 30000ms and the run stopped at 33007ms.
The check runs between nodes, not preemptively, so a cap can only be enforced after the node in
flight returns. Here the `repair` call took 7935ms and carried the run past the limit before anyone
could look. That is honest behaviour rather than a bug, but an operator setting `max_latency_ms`
should read it as "stop starting new work after N", not "return within N".

**4. Almost all the latency is model time.** `plan` took 13.8s and `assess` took 11.2s; `retrieve`
took 81ms. The agent's own bookkeeping is free, and the entire cost of agentic retrieval on a local
model is the model.

**5. Access control held, and the report said so.** `leave-policy.txt` was denied, the agent never
saw it, and sub-question 1 came back open with a reason rather than a confident wrong answer
assembled from the two loosely-related chunks it did retrieve. Those chunks (a handbook page
mentioning that retries and leave carry forward are common questions) are exactly the material a
permissive assessor would have accepted. The strict rubric refused them.

## What this run does not tell you

Whether the answer to sub-question 0 was correct. Whether a different model plans better. Whether
agentic retrieval beats traditional or vectorless retrieval on anything. All three need a labelled
corpus and metrics, which is Phase 8.
