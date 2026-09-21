"""Reciprocal Rank Fusion: combine ranked lists that have no common scale.

    rrf_score(d) = sum over rankings r of  weight_r / (k + rank_r(d))

rank is 1-based, and k defaults to 60, the value from Cormack, Clarke and
Buettcher's original paper. k exists to stop the top rank from dominating:
without it, first place would be worth twice second place, and a list that is
confidently wrong would drag the whole fusion with it.

**Why ranks and not scores.** BM25 is unbounded and corpus dependent: its
numbers mean nothing without knowing the corpus and the query. ts_rank_cd is
roughly 0 to 1. Putting the two on a shared scale means normalising, and
normalising means knowing each one's distribution, which changes with every
query. A min-max over a single result set is worse still, because it makes the
best hit a 1.0 whether it was excellent or merely the least bad. Ranks need no
such assumption: second place is second place whatever the numbers were. That
is why fusion here is RRF and not a weighted sum of scores.

**What fusion may not do.** It reorders chunks the stores already returned. It
never introduces one. Every ranking passed in came out of a store query that
applied the access filter inside the SQL (ADR 0003), so a chunk that appears
in no ranking is a chunk this principal was not allowed to see, and inventing
it here would route around the filter entirely.
"""

from __future__ import annotations

from ragfabric_core.strategies.base import RetrievedChunk

DEFAULT_K = 60


def rrf(
    rankings: list[list[RetrievedChunk]],
    k: int = DEFAULT_K,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one, best first.

    ``weights`` is one multiplier per ranking, defaulting to 1.0 each. The
    returned chunks carry the fused score, not whichever input score they
    arrived with: the input scores are on incompatible scales, so keeping one
    of them would be quietly misleading about how the order was reached.

    Ties break on chunk_id, so the same inputs always give the same order.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(f"expected {len(rankings)} weights, got {len(weights)}")

    fused: dict[int, float] = {}
    first_seen: dict[int, RetrievedChunk] = {}
    for index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[index]
        for position, hit in enumerate(ranking, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + weight / (k + position)
            # A chunk can appear in several rankings. Keep the first copy so
            # the text and metadata come from a real retrieval, and replace
            # only the score below.
            first_seen.setdefault(hit.chunk_id, hit)

    order = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
    return [
        first_seen[chunk_id].model_copy(update={"score": fused[chunk_id]}) for chunk_id in order
    ]
