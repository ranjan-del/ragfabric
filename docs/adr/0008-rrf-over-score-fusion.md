# ADR 0008: Fuse lexical rankings by rank, not by score

Status: accepted
Date: 2026-09-21
Supersedes: none
Related: ADR 0003 (access control inside retrieval), ADR 0007 (BM25 computed in SQL)

## Context

The vectorless strategy runs two lexical rankings over the same corpus and has to return
one ordered list.

- BM25 (ADR 0007) ranks by term statistics: rare terms, term frequency, document length.
- `ts_rank_cd` ranks by term proximity and coverage within the document, which BM25 does
  not model at all.

They are kept as two rankings rather than collapsed into one because they fail
differently. BM25 scores a chunk highly for containing a rare query term even if that term
sits alone in an unrelated sentence. `ts_rank_cd` rewards the query terms appearing close
together, which is often what the reader actually wanted, but it has no sense of which
terms were worth finding. A combination is more robust than either, provided the
combination itself does not introduce a new failure.

The question was how to combine them.

## Decision

The two rankings are combined by Reciprocal Rank Fusion, on rank position, never on score.

    rrf_score(d) = sum over rankings r of  weight_r / (k + rank_r(d))

`rank` is 1-based and `k` defaults to 60, the value from Cormack, Clarke and Buettcher's
original paper. `k` exists to stop the top rank from dominating: without it, first place
would be worth twice second place, and a single ranking that is confidently wrong would
drag the whole fusion with it.

Ties break on `chunk_id`, so identical inputs always produce an identical order.

The fused chunks carry the fused score, not whichever input score they arrived with. The
input scores are on incompatible scales, so keeping one of them would be quietly
misleading about how the order was reached.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| Weighted sum of the raw scores | BM25 is unbounded and corpus dependent; its numbers mean nothing without knowing the corpus and the query. `ts_rank_cd` is roughly 0 to 1. Adding them adds quantities with no common unit, and the weights would silently encode one corpus's score distribution. |
| Normalise both, then sum | Normalising requires knowing each ranking's distribution, and that distribution changes with every query. A normaliser fitted on one query's results is wrong for the next. |
| Min-max normalise within each result set | Worse than a global normaliser, because it makes the best hit a 1.0 whether it was excellent or merely the least bad. A query where every result is poor produces a confident looking 1.0 at the top. |
| Take BM25 and use `ts_rank_cd` only to break ties | Throws away the proximity signal in every case where it disagrees, which is exactly the case it exists for. |
| Train a learned combiner | Needs labelled relevance judgements for this corpus, which an adopter does not have on day one, and it would make the ranking unexplainable. Reconsider under Phase 8, where evaluation infrastructure exists and any claim can be measured rather than asserted. |

## Consequences

- Fusion is explainable. A chunk's position is a function of its positions in two lists,
  and both are inspectable. There is no fitted parameter beyond `k` and the per ranking
  weights, both of which are configuration with documented defaults.
- Absolute score magnitude is discarded. A chunk ranked first by a wide margin and a chunk
  ranked first by a hair contribute identically. That is the cost of not needing a
  normaliser, and it is accepted: the margin is exactly the quantity that is not comparable
  across the two rankings.
- **Fusion reorders; it never introduces.** Every ranking passed in came out of a store
  query that applied the access filter inside the SQL (ADR 0003), so a chunk absent from
  every ranking is a chunk this principal was not allowed to see. Adding one at the fusion
  step would route around the access filter entirely. The same rule constrains exact
  phrase and identifier boosting, which multiplies the score of a chunk already present and
  can never pull a new chunk in.
- `top_k` is applied after fusion, not per store. Cutting each ranking first would discard
  a chunk that placed modestly in both lists but would have fused to the top, which is
  precisely the agreement RRF exists to reward.
- No performance or quality figure is claimed here. Per ADR 0004, a comparison against a
  single ranking is a measurement, and it belongs to the evaluation framework in Phase 8.
