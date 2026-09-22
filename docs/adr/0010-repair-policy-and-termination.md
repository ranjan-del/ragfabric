# ADR 0010: Six repair moves, and three reasons to stop

Status: accepted
Date: 2026-09-22
Supersedes: none
Related: ADR 0004 (measurement first, no fabricated numbers), ADR 0009 (a plain Python state machine for the agent)

## Context

The committed concept design for agentic RAG had one repair: when the evidence was
insufficient, a `rewrite_query` node produced a different query and the loop retrieved again.
Its only guard was that a rewritten query must differ from every query already tried.

That is a retry loop with a vocabulary. Every failure gets the same response, so a failure
that a different wording cannot fix is never fixed. A paraphrase that the lexical ranking
cannot see is not helped by another paraphrase. A sub-question that is still two questions is
not helped by rewording either of them. A question whose answer is spread across a whole
policy is not helped by a sharper search for fragments of it. And "must differ from all
previous" is a uniqueness check, not learning: different is not smarter.

The design also had no answer to an agent that is not getting anywhere. It stopped when the
evidence was judged sufficient, or when the budget ran out. An agent that re-fetches the same
three chunks on every pass satisfies neither condition early, so it spends the whole budget
to arrive where it was after the first iteration, and every iteration looks like work from the
outside.

## Decision

### Six moves, not one

`RepairMove` is a fixed enum of six values, and `agent/policy.py` defines what each one does
to the sub-question it is applied to.

| Move | What it changes | The failure it answers |
|---|---|---|
| `broaden` | Drops qualifier words and any trailing restriction from the working query, and widens the candidate set to the larger of double its current value and five more than it, capped at the `top_k` ceiling of 100 | Retrieval returned nothing. The query was more specific than the corpus |
| `narrow` | Appends what the assessment said was missing to the working query, and cuts the candidate set back to at most 5 | Retrieval returned plenty and none of it was on point |
| `switch_strategy` | Flips the sub-question's tool between `semantic_search` and `lexical_search`, and sends `fetch_document` back to `semantic_search` | An identifier was missed by meaning, or a paraphrase was missed by wording |
| `decompose` | Splits the sub-question on a conjunction into children with empty histories of their own | The sub-question is still compound |
| `fetch_document` | Points the sub-question at the best-scoring retrieved chunk's document and reads that document whole, in order | The evidence is fragmentary and the right document has been identified |
| `abandon` | Marks the sub-question abandoned with a reason | Nothing is working, and saying so beats another iteration that cannot help |

The six are not six wordings of one idea. Three of them (`switch_strategy`, `decompose`,
`fetch_document`) do not touch the query text at all: they change the tool, the shape of the
question, or what is being read. `rewrite_query` could not express any of them, which is the
argument for the set.

The model proposes a move and the policy decides whether the proposal is usable. A proposal
is refused only when it has already been tried on this sub-question, or when it cannot do
anything here: `decompose` when the text will not split, `fetch_document` when no retrieved
chunk names a document. It is never refused for being unexpected, because the enum already
made an invented move impossible (ADR 0009). When the proposal is refused or could not be
read at all, the policy falls back to a situational order, and there are two of them. When
nothing came back, `broaden` is first and `narrow` is next to last, because narrowing a query
that already returned nothing is how an agent talks itself into an empty result twice. When
evidence came back but none of it was on point, `narrow` is first and `broaden` is last.

`abandon` proposed by the model is honoured only once something has actually been tried.
A model that gives up on the first iteration would otherwise turn the agent into a plain
retriever that reports a gap it never looked for.

### A move is never repeated on an unchanged sub-question

`SubQuestion.attempts` records every repair already made on that sub-question: the move, the
query that failed, and what the assessment said was missing. `has_tried` reads it, and
`choose_move` will not offer a move that appears there.

Without that guard the agent alternates `broaden` and `narrow` until the budget is gone. Each
pass is a legitimate-looking decision, each produces a different query, and the pair
oscillates around the same neighbourhood forever. The uniqueness check in the old design does
not catch this, because a broadened query and a narrowed query genuinely are different
strings.

The history is per sub-question, so a move spent on one part of the question says nothing
about another. And a sub-question that has genuinely changed is not the same sub-question:
`decompose` replaces the parent with children that start with empty histories, which is the
one legitimate way a move becomes available again.

The attempt is recorded before the sub-question is changed, so the history holds the query
that actually failed rather than its replacement.

### Exhausting the moves abandons, with a reason

When every move has been tried or is inapplicable, `choose_move` returns `abandon`, and
`apply_move` requires a reason: either the model's own stated reason, or one assembled from
the attempt history, in the form `abandoned after trying broaden, switch_strategy: <what was
missing>`. `SubQuestion.abandon` raises on an empty reason, so an abandoned sub-question with
no explanation cannot be constructed.

The alternative is to keep looping until the budget stops it. That spends real money and real
latency to arrive at the same gap, and it arrives there with `budget` as the stop reason,
which tells the caller the agent ran out of room rather than out of ideas. Those are different
facts and the caller deserves the true one.

Abandoning is a move, not a failure of the design. A question with a part this corpus cannot
answer should end with that part marked, explained, and not paid for twice.

### Three stop conditions

`run_agent` can stop for exactly three reasons, and the result carries which.

| Stop reason | Fires when | What it tells the caller |
|---|---|---|
| `resolved` | Every sub-question is answered or abandoned | The loop finished its work. Not a synonym for success: an abandoned sub-question carries its reason, and the report is where that is read |
| `budget` | A cap was reached. Either the iteration cap, or `AgentState.spend` refused a call against the global LLM budget or a per-node cap | The agent ran out of room, not out of ideas. There may be more to find |
| `no_progress` | An iteration added no chunk id the evidence pool did not already hold | The repairs are no longer reaching anything new. More iterations cannot change that |

`no_progress` is the condition the textbook design lacks, and it is the one that matters
most in practice. Progress is defined as **new chunk ids**, never as a count of rows returned:
`AgentState.add_evidence` pools by chunk id and returns only the ids that were genuinely new,
and `RetrieveOutcome.made_progress` reads that set. A tool that returns the same three chunks
on every call returns something every time and teaches the agent nothing, so counting rows
would report progress on precisely the run that has none.

`budget` covers two mechanisms on purpose, because both are the same fact from the caller's
side: a limit stopped the run. The `stop_detail` string names which limit it was, so the
distinction is available without splitting the reason.

### The stop reason is assigned at the branch, never inferred

Each of the branches above sets the reason and its detail at the point the condition is found
true. `run_agent` carries them to the end, writes the reason onto the state and both into the
finalize trace span. No function anywhere looks at a finished run and works out why it ended.

The reason this is written down as a decision rather than left as a coding habit is in ADR
0009: the framework spike lost the stop reason exactly because the branch and the record were
in different places, and returned `stop_reason=None` on a run that had clearly stopped for a
reason. Inference after the fact is also wrong in a way that is hard to see. A finished run
where every sub-question is resolved and the iteration count happens to equal the cap is
indistinguishable, from the outside, from a run that was cut off at the cap.

### Two facts the plan did not anticipate

Both were found while building, and both are load bearing.

**A repaired query never rewrites the sub-question.** `SubQuestion.text` stays as the plan
wrote it for the life of the run. The broadened, narrowed or document-id query produced by a
repair lives in a separate `RetrievalOverride`, which the retrieve node uses in place of the
text and which also carries the repaired `top_k` and any document id. The obvious
implementation, overwriting `text`, destroys the record of what was actually asked, and that
record is what the per sub-question report shows the caller. A report saying "not answered:
retry limit payments" when the plan asked "what is the retry limit for the payments API" is a
worse answer than no report.

The same separation is why widening a search cannot widen access. The override's `top_k` is
applied with `model_copy` on the existing `RetrievalContext`, which carries the principal and
the access filter across untouched (ADR 0003), rather than by constructing a fresh context and
remembering to copy the filter into it.

**`no_progress` is exempt on the first iteration.** An empty or unchanged first retrieval is
exactly the situation the repair moves exist for: the plan may have chosen the wrong tool, or
written a query too specific for the corpus. Stopping there would deny the agent its first
move and reduce it to a single-shot retriever with extra LLM calls. From the second iteration
on, no new chunk id means the repairs are not reaching anything new, and another pass cannot
change that.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| One repair move, query rewriting, as the concept design had it | It is the only response to every failure, so any failure that a different wording cannot fix is never fixed. Three of the six moves here change the tool, the shape of the question, or what is read rather than the query text, and none of those is expressible as a rewrite. |
| Free-text repair instructions from the model | Maximally expressive and undefined: an instruction the policy has never seen has no defined effect on the sub-question. See ADR 0009 on why the move is a fixed enum. |
| Let the model repeat a move if it insists | The model does insist, and it is how an agent alternates broaden and narrow until the budget is gone. The guard is not a limit on the model's judgement so much as on its short memory of what it has already tried. |
| Allow a repeat when the query text differs | This is the old uniqueness check under a new name. A broadened query and a narrowed one differ as strings while the pair oscillates around the same place. |
| Keep looping instead of abandoning | Arrives at the same gap having spent the remaining budget, and reports `budget` as the reason, which tells the caller the agent ran out of room when it had actually run out of ideas. |
| Two stop conditions: resolved and budget | This is the concept design. It has no answer to a loop that is progressing nowhere, and that loop always terminates on `budget`, so the honest failure handling issue #6 asks for would report the wrong cause on the most common bad run. |
| Detect no progress by comparing the number of chunks returned | Cheaper and wrong. A tool returning the same three chunks every call returns three chunks every call. Counting rows reports progress on the run that has none, which is the run this check exists to catch. |
| Stop on no progress from the first iteration | Simpler, and it removes the agent's first opportunity to change tool or widen a query. An empty first retrieval is the normal case the repair policy was built for. |
| Infer the stop reason after the loop ends | The failure mode is in ADR 0009. It also cannot distinguish a run that resolved on the final permitted iteration from one that was cut off at the cap. |

## Consequences

- **A sub-question can end in one of three states**, and each carries its own explanation:
  answered, abandoned with a reason, or open when the run stopped on `budget` or
  `no_progress`. There is no single boolean anywhere saying the answer was best effort.
- **The repair budget is bounded by the move set.** Five of the six moves change a
  sub-question; the sixth ends it. So a sub-question can be repaired at most five times before
  the set is exhausted and it is abandoned, independently of the iteration cap and the LLM call
  caps. Three limits bound the same loop from three directions, and the stop detail says which
  one bound it.
- **`decompose` uses the abandoned status for the parent**, with the reason `decomposed into N
  sub-questions`. The parent has been replaced rather than given up on, and a reader of the
  report needs to see that difference in the reason text, not in the status.
- **`fetch_document` returns chunks with no score**, because nothing was ranked and a number
  there would be a fabricated relevance (ADR 0004). Those chunks therefore sort to the end of
  the pooled evidence, which is ordered by score with a missing score read as zero.
- **The policy's textual edits are mechanical, not model-driven.** Dropping qualifiers,
  cutting at a trailing preposition and splitting on a conjunction are list-and-string
  operations over fixed vocabularies in `agent/policy.py`, used when the model supplies no
  rewritten query. They are deliberately dull, they are unit tested, and they are the floor
  under a model that returns nothing usable. They are not natural language understanding and
  will handle an unusual phrasing badly.
- **None of this is a quality claim.** Whether six moves recover more questions than one, and
  whether `no_progress` fires on the runs it should, are measurements. They belong to the
  evaluation framework in Phase 8, and until then no figure for either appears anywhere in
  this repository (ADR 0004).
