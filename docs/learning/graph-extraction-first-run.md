# What a real graph extraction run looked like, learning notes

> Status: Phase 6 (in progress on this branch). A record of **one** run of graph extraction,
> resolution and traversal against a local model, written down because the offline tests
> cannot tell you this. See [../graph-rag.md](../graph-rag.md) for the concept and
> [agentic-first-run.md](agentic-first-run.md) for the same exercise on Phase 5's agent loop.

## This is a record, not a benchmark

Per ADR 0004, everything below is what happened on a six-chunk fixture corpus, with
`llama3.1:8b` for extraction and question answering and `nomic-embed-text` for entity
resolution, on one machine, on 2026-09-25. It is a single run of a non-deterministic system
(the model call itself uses temperature 0.0, which made the runs below reproduce closely, but
that is a property of this model and this prompt, not a guarantee). Nothing here may be
quoted as a measurement of extraction precision, resolution precision, or citation-contract
strictness in general, compared against another model, or used to tune the confidence floor
(0.5) or the similarity threshold (0.9) away from their documented defaults. Both are still
untuned, exactly as `GraphStoreConfig` says, until Phase 8 measures them.

There are two sources of findings below, and they are kept separate rather than folded into
one set of numbers:

* **Calibration runs.** Before writing the committed test, this corpus was run twice through
  the real pipeline from a scratch script, outside the test suite, to see what a real model
  would do before locking in the test's assertions. These runs are what first surfaced the
  contract violation, the direction error and the failed merges described below.
* **The committed test run.** `packages/core/tests/test_graph_integration.py`, run for real
  with `RAGFABRIC_TEST_OLLAMA=1`, is what actually ships. It reproduced every finding from the
  calibration runs exactly (same entities, same relationships, same merge, same dropped
  claim, same answer text), which is noted below wherever it matters. The numbers quoted
  below as evidence are the committed run's own output, captured with `-s`, not the earlier
  calibration runs.

## The corpus and the trap in it

Three documents, six short chunks, about people, a team, a project and an organisation:

```
profile.txt:      "Ravi Sharma is a software engineer on the Platform Team. Ravi Sharma
                    reports to Meera Iyer."
team.txt:         "R. Sharma presented the migration plan at the weekly Platform Team sync
                    meeting."
                   "Anjali Sharma is a product designer who works on Project Atlas. Anjali
                    Sharma is a different person from Ravi Sharma, despite sharing a
                    surname."
                   "The Platform Team belongs to the Engineering organisation."
                   "Project Atlas is owned by the Engineering organisation."
management.txt:   "Meera Iyer manages the Platform Team and reports to Kabir Rao, the head
                    of the Engineering organisation."
```

`profile.txt` holds exactly one fact and nothing else, so it can be denied on its own for the
access-filter proof without also hiding the rest of the corpus.

The ground truth, written down before any extraction ran (in the test file, ahead of the
fixture):

**Entities:** Ravi Sharma, Anjali Sharma, Meera Iyer, Kabir Rao (person); Platform Team
(team); Engineering (organisation); Project Atlas (project).

**Relationships:** a two-hop directed `REPORTS_TO` chain (Ravi Sharma to Meera Iyer to Kabir
Rao), three invertible relations (`MEMBER_OF`, `BELONGS_TO`, `OWNS`).

**The alias trap:** "Ravi Sharma" and "R. Sharma" name the same person and should resolve to
one entity.

**The surname trap:** "Ravi Sharma" and "Anjali Sharma" are two different people who happen to
share a surname and must not resolve to one entity.

The question asked of the graph: *"Who does Ravi Sharma report to, and who does that manager
report to in turn?"*: the two-hop chain, deliberately.

## What extraction actually produced

Committed test run counts: `entities_stored=12, relationships_stored=6, entities_discarded=0,
relationships_discarded=0, contract_violation=True, chunks_skipped=0,
extracted_chunk_ids=(1, 2, 3, 4, 5)`. Six chunks went in; five extraction calls completed and
were stored; one hit a contract violation and stored nothing at all. The calibration runs
produced the identical counts.

Entities actually stored (after resolution; `[7]`, the merged-away duplicate, is gone):

```
[1] Ravi Sharma                     person        confidence=1.0
[2] Meera Iyer                      person        confidence=1.0
[3] Platform Team                   team          confidence=1.0  aliases=['The Platform Team']
[4] R. Sharma                       person        confidence=1.0
[5] Anjali Sharma                   person        confidence=0.9
[6] Project Atlas                   project       confidence=1.0
[8] the Engineering organisation    organisation  confidence=1.0
[9] Engineering                     organisation  confidence=1.0
```

Relationships actually stored:

```
Ravi Sharma   --REPORTS_TO--> Meera Iyer                    confidence=1.0
R. Sharma     --WORKS_ON-->   Platform Team                 confidence=1.0
Anjali Sharma --WORKS_ON-->   Project Atlas                 confidence=0.9
Anjali Sharma --RELATED_TO--> Ravi Sharma                   confidence=0.8
Platform Team --BELONGS_TO--> the Engineering organisation  confidence=1.0
Project Atlas --OWNS-->       Engineering                   confidence=1.0
```

Matched against ground truth (both the calibration runs and the committed test run agree on
every line below):

* **Matched exactly (2):** `Ravi Sharma REPORTS_TO Meera Iyer`; `Anjali Sharma WORKS_ON
  Project Atlas`.
* **Direction error (1):** the corpus says "Project Atlas is owned by the Engineering
  organisation" (`Engineering OWNS Project Atlas`); the model stored `Project Atlas OWNS
  Engineering`, the two endpoints reversed. Entity fragmentation compounds this: the model
  named the owner "Engineering" here but "the Engineering organisation" in the chunk about
  the Platform Team (see below), so even the correct direction would not have matched a
  single entity.
* **Missed (3):** `Meera Iyer REPORTS_TO Kabir Rao` (the second hop of the chain, entirely
  lost, see the contract violation below), `Ravi Sharma MEMBER_OF Platform Team` (the model
  never turned "is a software engineer on the Platform Team" into a membership edge for the
  canonically named person), and `Platform Team BELONGS_TO Engineering` (stored instead
  against "the Engineering organisation", a different normalised name; see entity
  fragmentation below).
* **Invented (3):** `R. Sharma WORKS_ON Platform Team` (the text only says R. Sharma
  *presented at* a team sync meeting; attending or presenting at a meeting is not stated as
  working on the team, and the model attributed this to the unmerged alias entity rather than
  to "Ravi Sharma," who is the one the corpus actually describes as working on the team);
  `Anjali Sharma RELATED_TO Ravi Sharma` (the sentence explicitly says these are two
  *different* people who merely share a surname; the model turned that disambiguating
  sentence into a relationship between them anyway); `Platform Team BELONGS_TO the
  Engineering organisation` (this one is arguably correct in substance, just recorded against
  a fragment of the intended entity name, see below).

### The contract violation: one invented relation type sank a whole chunk

The one chunk that produced nothing was `management.txt`'s sentence: "Meera Iyer manages the
Platform Team and reports to Kabir Rao, the head of the Engineering organisation." Replaying
the exact extraction call for this sentence outside the test (a calibration step) shows why:

```json
{
  "relationships": [
    {"source": "Meera Iyer", "target": "Platform Team", "relation_type": "MANAGES", "confidence": 1.0},
    {"source": "Meera Iyer", "target": "Kabir Rao", "relation_type": "REPORTS_TO", "confidence": 1.0}
  ]
}
```

```
CONTRACT VIOLATION: 1 validation error for ExtractionResponse
relationships.0.relation_type
  Input should be 'REPORTS_TO', 'MEMBER_OF', 'BELONGS_TO', 'OWNS', 'WORKS_ON', 'LOCATED_IN',
  'AUTHORED', 'MENTIONS' or 'RELATED_TO' [type=enum, input_value='MANAGES', input_type=str]
```

`MANAGES` is not one of the nine fixed relation types, and the extraction contract validates
the whole response as one object: one bad relation type invalidates every entity and every
relationship the model found in that call, including the perfectly good `REPORTS_TO` fact in
the very same response and the `Kabir Rao` entity itself. `ExtractionReport.contract_violation`
only reports that *some* chunk in the batch violated the contract, as a single boolean OR'd
across the whole call; it does not say which chunk or how many. Finding the actual chunk and
the actual reason above took a manual replay outside the committed test, not something the
report alone would have told an operator. That gap is itself worth naming for whoever builds
an operator-facing view of this later.

The practical consequence: `Kabir Rao` never entered the graph, `Meera Iyer REPORTS_TO Kabir
Rao` never entered the graph, and the two-hop chain the fixture was built to exercise
collapsed to one hop before resolution or traversal even ran. This is exactly the kind of
finding ADR 0004 asks for recorded rather than tuned away: the fixture sentence was not
rewritten to avoid tripping the model into inventing `MANAGES`, because that would have hidden
a real, reproducible failure mode: a single relation compound in one sentence is enough for
this model to reach for a synonym outside the fixed vocabulary, and the fixed-vocabulary
contract's all-or-nothing validation turns that one bad guess into the loss of an entire
sentence's worth of otherwise-correct extraction.

### Entity fragmentation: the same real-world thing, named two ways

Two adjacent sentences in `team.txt`, both about the same real-world organisation, produced
two different entities. "The Platform Team belongs to the Engineering organisation" gave
`[8] the Engineering organisation`; the very next chunk, "Project Atlas is owned by the
Engineering organisation," gave `[9] Engineering` instead of reusing the name the model had
just used one chunk earlier for the identical phrase. `normalise()`
does not strip leading articles, so `"the engineering organisation"` and `"engineering"` are
genuinely different normalised names, and the exact and alias resolution stages never see them
as candidates for the same identity. Resolution's stage 3 (embedding similarity) was eligible
to compare them (both are `organisation` type, and there were exactly two), but the merge
list below shows only one merge, for a different pair. Whatever cosine similarity
`nomic-embed-text` assigned to the bare names `"Engineering"` and `"the Engineering
organisation"` (neither entity has a stored description, so the embedded text is just the
name), it fell under the 0.9 threshold. This is an embedding-quality finding, not asserted anywhere in
the test, and it directly explains why `Platform Team BELONGS_TO Engineering` shows as both
"invented" (stored against the wrong-named entity) and "missed" (never stored against the
right-named one) in the comparison above: the underlying fact is substantively correct, the
entity identity is the problem.

## What resolution did

`resolve_entities` made exactly one embedding call for the whole batch (`embedding_calls=1`)
and produced exactly one merge:

```
'The Platform Team' -> 'Platform Team'   method=embedding   similarity=0.9716   threshold=0.9
```

This one is correct and unsurprising: "The Platform Team" (from the start of "The Platform
Team belongs to the Engineering organisation") and "Platform Team" (from the other four
chunks) differ only by a leading article, and their embeddings were close enough to clear the
0.9 threshold comfortably.

**The alias pair did not merge.** "Ravi Sharma" (entity `[1]`) and "R. Sharma" (entity `[4]`)
remained two separate entities through both calibration runs and the committed test run. They
are, by construction, the same person under two surface names, both typed `person`, so they
were eligible for stage 3 comparison; whatever similarity `nomic-embed-text` computed between
`"Ravi Sharma"` and `"R. Sharma"` (both have no stored description, so the embedded text is
just the bare name) did not clear 0.9. This is precisely the trap this fixture was built to
expose, straight from the project's own graph-rag concept notes, which use "R. Sharma" /
"Ravi Sharma" as the canonical example of a resolvable alias, and on this model, with no
alias-stage help (nothing populates `Entity.aliases` at extraction time; only a later merge
appends one), name-embedding similarity alone was not enough.

**The surname pair correctly did not merge**, but not cleanly: `Ravi Sharma` and `Anjali
Sharma` stayed as two separate entities, which is the right outcome, but the model itself blurred
the line by inventing the `RELATED_TO` edge described above between them. Resolution kept them
apart; extraction still tied them together with an unsupported relationship.

## The real graph query and the citation contract

Question: *"Who does Ravi Sharma report to, and who does that manager report to in turn?"*
With the second hop missing from the graph (`Kabir Rao` never stored), the traversal from
`Ravi Sharma` (`max_hops=2`, the strategy's default) found only:

```
nodes: Ravi Sharma (depth 0), Meera Iyer (depth 1)
edges: Ravi Sharma --REPORTS_TO--> Meera Iyer  (source_chunk_ids=[1])
llm_calls=1  embedding_calls=0  retrieval_calls=5
```

`embedding_calls=0` on the query path, as ruling R19 requires: entity mentions in a question
match by name and alias only, never a fresh embedding call.

The generated answer:

> "I could not find an answer to that in the documents provided regarding who Meera Iyer
> reports to."

with one dropped relationship claim:

```
no_edge_cited: "Meera Iyer reports to [E 1] [1]."
```

Two things are worth naming here, both reproduced identically across every run:

1. **The model tried to answer the unanswerable half and dropped the answerable half in the
   same breath.** The corpus and the graph both plainly support "Ravi Sharma reports to Meera
   Iyer," a real edge with a real citation, but the generated text never states it. Instead the
   model produced one sentence trying to complete the chain past Meera Iyer, citing `[E 1]`
   (the only edge that exists, which connects Ravi Sharma and Meera Iyer, not Meera Iyer and
   anyone else) as if it supported "Meera Iyer reports to X." The graph citation contract
   correctly refused this: `[E 1]` does not join two entities the claim names in that
   direction, so the claim is dropped as `no_edge_cited` rather than let through as a
   fabricated third fact. This is exactly the invariant Task 10 built and this test asserts
   directly against the real model's output (every `[E k]` marker surviving into the kept
   answer must index a real edge in the traversed sub-graph), and it held.
2. **The near-miss on the refusal sentence.** The model's text is not the exact fixed
   `NO_EVIDENCE_ANSWER` string; it appended "regarding who Meera Iyer reports to" to it. Since
   nothing in that sentence carries a citation marker or a relationship claim, it survived the
   citation contract as an ordinary, technically uncited-but-also-unclaiming sentence; the
   test does not treat this as a contract failure, because it makes no factual claim to check,
   but it is a real instance of the model not reproducing an instructed exact string, which
   would matter to anything downstream that pattern-matches on the literal refusal sentence.

## Access control, over this run's real graph

`profile.txt` is the sole source of `Meera Iyer` and of the `Ravi Sharma REPORTS_TO Meera
Iyer` edge in this corpus. Traversing from `Ravi Sharma` with unrestricted access reaches
`['Ravi Sharma', 'Meera Iyer', 'Anjali Sharma', 'Project Atlas']` (Anjali Sharma and Project
Atlas by way of the invented `RELATED_TO` edge and the `WORKS_ON` edge, both real edges in the
stored graph, wrong or not); denying `profile.txt` in the access filter reaches
`['Ravi Sharma', 'Anjali Sharma', 'Project Atlas']`: Meera Iyer is gone. This is asserted
directly, on the real graph the model produced rather than a hand-built one, and it held: the
access filter (ADR 0003) worked correctly on real, model-written data, including data the
model itself entangled through an invented relationship.

## What this implies for Phase 7 and Phase 8

* **A single fixed-vocabulary contract violation costs more than the one bad field.** Because
  `ExtractionResponse` validates the whole chunk's response as one unit, a model reaching for
  one synonym outside the nine relation types (`MANAGES` here) throws away every other fact in
  that response, including ones the model got completely right. Phase 7's router and Phase 8's
  evaluation should both expect this: a chunk's relation count is not just "what the model
  found," it also reflects how compound the sentence was and how likely the model was to reach
  for a vocabulary word the schema does not have. A prompt or a retry strategy that recovers
  partial credit from an otherwise-good response is a real option to evaluate, not implemented
  here.
* **Name-only embedding similarity is not enough for alias resolution on this model.** The
  fixture's headline alias pair, the one the project's own concept docs use as the canonical
  example, did not merge. Phase 8's evaluation of the 0.9 similarity threshold should not
  assume that surface-form aliases without descriptions will cluster; a description-bearing
  embedding, a normalised initials rule, or a lower threshold for the alias stage specifically
  are all things worth measuring against labelled data, not guessing at from one run.
* **Entity naming is not stable across chunks for the same real-world thing.** "Engineering"
  and "the Engineering organisation" fragmenting into two entities is a second, independent
  way for a true fact to disappear from the graph (as opposed to being invented or reversed):
  the relationship was extracted correctly in substance and still failed to land on a single
  identity. Whether this is best fixed by a normalisation change (stripping leading
  articles), a resolution-stage change (comparing names more leniently before falling back to
  embeddings), or a prompt change (asking the model to use a consistent short name) needs
  labelled data to compare, which is Phase 8's job.
* **Direction correctness is not free, and the citation contract does not check it.** The
  `OWNS` edge came out backwards. Ruling R32/R33 already documents that direction and
  paraphrase faithfulness are not mechanically checked by the citation contract, by design,
  and this run gives a concrete, real instance of exactly that gap: a wrong-direction edge
  would be walkable and citable exactly as if it were correct. Phase 8's evaluation is where
  direction correctness needs to be measured, because nothing before it can catch this
  mechanically.
* **The citation contract's core promise held under real model output.** Despite an invented
  relationship, a reversed edge, and an incomplete chain, the one relationship claim the model
  tried to state past what the graph actually contained was refused, not hallucinated into the
  answer. That is the one finding here that generalises past this particular corpus: whatever
  else a small local model gets wrong at extraction time, ADR 0003's access filter and Task
  10's citation contract kept holding at generation time.

## What this run does not tell you

Whether `llama3.1:8b` extracts relationships correctly in general, whether 0.5 and 0.9 are the
right floor and threshold, whether a bigger or different model would avoid the `MANAGES`
mistake, or whether this corpus's specific failure modes are typical of real documents. All of
that needs a labelled corpus and metrics, which is Phase 8.
