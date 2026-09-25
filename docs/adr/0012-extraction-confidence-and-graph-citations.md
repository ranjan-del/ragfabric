# ADR 0012: An unverified edge is not a fact

Status: accepted
Date: 2026-09-24
Supersedes: none
Related: ADR 0004 (measurement first, no fabricated numbers), ADR 0003 (access control inside
retrieval), ADR 0011 (PostgreSQL recursive CTE over Neo4j)

## Context

The committed concept design for Graph RAG trusted extraction blindly: an LLM read a chunk,
reported entities and relationships, and they became rows in the graph with nothing to say how
sure the model was, or whether a later sentence supports the exact relationship a generated answer
narrates. That has two separate failure points, and this ADR is the decision that closes both.

**At write time**, an LLM emits an edge and it becomes indistinguishable from a real one, forever.
A hallucinated `REPORTS_TO` looks exactly like an observed one once it is a row, and nothing in the
concept design recorded how confident the model actually was, or let an operator see that a corpus
or a model was producing junk.

**At read time**, Phase 3's citation contract (`generate/contract.py`) verifies a claim against the
text of the chunks it cites. It cannot see a relationship claim, because "the platform team reports
to the CTO" can be a fair narration of two walked edges while appearing in no single chunk. A model
that narrates a traversal as prose will happily assert a relationship no source sentence ever
stated, and the edges it cites can be entirely real: the walk is correct and the claim is still
fabricated. Nothing built in Phase 3 or Phase 5 catches that, because it is not a chunk-grounding
problem, it is a graph-narration problem.

ADR 0004 already settled the general principle: an extraction confidence is what the model
reported, never a default that looks plausible, and unknown is `None`. This ADR is that principle
applied to two specific places extraction and traversal touch: what gets written, and what a
generated answer is allowed to say about it.

## Decision

### The confidence floor (Task 3)

One LLM call reads a chunk against `graph/contracts.ExtractionResponse` and returns entities and
relationships, each carrying the confidence the model reported for that item. Everything below
`graph_store.confidence_floor` (default 0.5, untuned until Phase 8 measures extraction precision
against it) is **discarded, not stored**, and the discard is counted
(`ExtractionReport.entities_discarded`, `relationships_discarded`) rather than silent. An edge whose
source or target entity itself fell below the floor is discarded too, even when the edge's own
confidence clears it, because an edge cannot point at a row that was never stored (ruling R9).

Without the floor, the graph would silently accumulate the model's guesses as facts. Reporting how
many were discarded is what lets an operator see that a corpus or a model is producing junk, which
a floor with no counter cannot do.

### Confidence is per source, and the row is the max of what is currently measured (ruling R20)

`entity_sources` and `relationship_sources` (migration 0008) carry a nullable `confidence` and
`extraction_model`, one row per chunk that contributed to that entity or relationship. The parent
`Entity.confidence` / `Relationship.confidence` is not written once and left alone; it is
**recomputed as the maximum over the row's current sources** every time a source is added, removed
(Task 4, a chunk changed; or its chunk deleted with its document or collection, or replaced by a
re-ingest, ruling R42) or moved (Task 5, a merge or unmerge; or a re-ingest carrying it to an
identical new chunk, ruling R41). `None` iff no surviving source has a measured confidence. What a
caller is shown is narrower still: an edge's confidence in a traversal result is the maximum over
the sources that caller may read (ruling R43), never the stored aggregate.

This is not the same guarantee as a floor at write time. R9's original single-max rule goes stale
the moment the chunk that supplied the maximum is re-extracted or its entity is merged away; keeping
a number a current source does not support is exactly the fabrication ADR 0004 forbids, just spread
across time instead of concentrated in one call. Recomputing from the sources that exist right now,
rather than trusting a value written earlier, is what keeps that promise as the graph changes under
ingestion and resolution.

### The graph citation contract (Task 10, ruling R32 amended by R33)

The Phase 3 contract stays exactly as it is and still applies to every chunk-grounded claim
unchanged; this contract is additive, for the class of claim Phase 3 cannot see.

The graph prompt numbers the retrieved passages `[n]`, as Phase 3 does, and numbers the traversed
sub-graph's edges `[E k]`, 1-based over `Subgraph.edges`, each rendered in its walked direction. A
claim is a **relationship claim** iff it names two or more distinct sub-graph node names or carries
an `[E k]` marker; a single mention of a name shared by two differently-typed nodes counts once,
not twice, so it does not manufacture a relationship claim out of one entity mentioned once (R33).

A relationship claim survives only if, in order, the first failure being the recorded reason:

1. **`edge_not_in_subgraph`**: every `[E k]` it cites resolves to an edge of the traversed
   sub-graph. A number outside it, or any marker when there is no sub-graph, fails here.
2. **`no_edge_cited`**: at least one cited edge joins two entities the claim names.
3. **`chunk_not_edge_source`**: every cited edge that joins two named entities has at least one
   cited `[n]` chunk among its `source_chunk_ids`.
4. **`chunk_does_not_name_endpoints`**: at least one such chunk's text names both of that edge's
   endpoints.
5. **`uncovered_entity`** (ruling R33, amending the original rule 5): every entity the claim names
   is an endpoint of at least one of those cited, chunk-backed edges. Naming a third entity a
   sentence's cited edges never touch is exactly the traversal-narration fabrication this contract
   exists to stop: "the platform team and engineering both report to the CTO", citing only the
   backed engineering-to-CTO edge, is dropped because no cited edge touches the platform team.

Every drop is recorded with its reason, the same discipline the floor applies at write time: a
mechanical check that only asserts what needs no semantic judgement (ADR 0004), never a judgement
call about whether the sentence is probably fine.

### What this contract does not check, stated rather than hidden

Five loosenesses are recorded in the `graph/citations.py` module docstring rather than silently
accepted as correct:

1. **Direction and paraphrase faithfulness are not checked.** A claim that cites a supported
   `OWNS` edge from A to B while saying B owns A passes.
2. **Rule 5 checks that every named entity is touched by a backed edge, not which pairs the
   sentence relates.** With backed A-B and B-C edges cited, "A reports to C" in a sentence that also
   names B passes, although no edge joins A to C directly.
3. **A cited edge that joins no two named entities is tolerated, not refused**, provided it
   resolves. Such a marker adds no support and is not treated as a failure.
4. **Names are matched only on the node's own name**, which is the spelling of the caller's best
   admitted source (ruling R40). A chunk that names an endpoint by another spelling, an alias or
   a pronoun does not count as naming it, so a true claim can be dropped; the rendered sub-graph
   carries no aliases (ADR 0003, ruling R6: no stored description, and no alias, is ever rendered to
   a caller), so there is nothing else to match against.
5. **A single-entity claim with no `[E k]` marker is not a relationship claim**, even if it relates
   that entity to something outside the sub-graph. It goes through the Phase 3 contract, chunk
   grounding only, the same as before this phase.

Deciding any of the first two needs a semantic reading of the sentence, which cannot be asserted
mechanically (ADR 0004); whether they matter in practice is a measurement for Phase 8, not a claim
made here.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| No confidence floor, store everything the model reports | The concept design's approach. A hallucinated edge becomes indistinguishable from an observed one the moment it is a row, and no operator signal exists to notice a corpus or model producing junk. |
| A single confidence on the parent row, written once | Goes stale the moment the source that supplied it is re-extracted (Task 4) or merged away (Task 5), leaving a number no current source supports, which is the exact fabrication ADR 0004 forbids. |
| Default a missing model confidence to a fixed value such as 0.5 | Exactly the fabricated number ADR 0004 names directly. A missing confidence is a contract violation for that item, not a value to substitute. |
| Trust the Phase 3 contract for relationship claims too | It verifies against chunk text and cannot see a relationship a chunk never states in one sentence but two edges jointly support. It would either wrongly drop a true, edge-supported relationship claim or, if loosened to let such claims through, stop catching narration fabrication entirely. |
| Require every named pair, not just every named entity, to have a directly cited edge between them | Stronger, and unimplementable without deciding which pairs a sentence actually asserts a relationship between, which is the semantic reading loosenesses 1 and 2 leave to Phase 8. Over-strict here drops a correct multi-hop narration ("A is in B, which reports to C") that this contract is meant to allow. |
| Silently drop a relationship claim without recording why | Matches the letter of "wrong claims do not reach the caller" but hides the failure rate from whoever is judging whether extraction and the prompt are any good, the same operator-visibility argument that justifies counting discarded extraction items. |

## Consequences

- **A stored confidence always means "the maximum a chunk currently linked to this row actually
  reported"**, never a number left over from a chunk that no longer sources it. That is a query
  (`_recompute_entity_confidence` / `_recompute_relationship_confidence`) run after every source
  change, not a value trusted at write time and left alone.
- **A relationship claim can be true and still dropped** (loosenesses 3 and 4 in particular): the
  contract is deliberately over-strict rather than over-lenient, matching Phase 3's citation
  contract precedent of refusing what it cannot verify rather than accepting what looks plausible.
- **The five loosenesses are a stated boundary of this phase's guarantee, not a defect found
  later.** Whether they matter on real questions, and whether direction or pair-faithfulness checks
  would be worth their false-drop cost, is measurement work for Phase 8; no number for either
  appears anywhere in this repository until then (ADR 0004).
- **Extraction quality and citation-contract precision are two different measurements**, and this
  ADR makes no claim about either. It only guarantees that what is unmeasured is discarded (the
  floor) or recorded as a mechanical failure (the contract), never silently promoted to a fact.
