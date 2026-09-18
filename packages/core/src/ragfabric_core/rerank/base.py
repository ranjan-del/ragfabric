"""The Reranker interface.

A reranker re-scores a shortlist that a store already ranked cheaply. It exists
as its own interface because all four retrieval strategies share it and Phase 8
measures the same reranker across all of them; a per strategy flag would mean
four implementations of one idea.

Contract, asserted in the tests of every implementation: the output is a subset
of the input, no longer than top_k, with identity fields untouched. A reranker
reorders and rewrites score. It never invents, merges or edits a chunk.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from ragfabric_core.strategies.base import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


def all_finite(scores: list[float]) -> bool:
    """True unless any score is NaN or +/-infinity.

    A model can emit a syntactically valid, correct length, all-float score
    list that still contains NaN or infinity. Sorting on such a list places
    the offending chunk at an arbitrary position, which is indistinguishable
    from a real ranking to anything downstream. So a non-finite score must be
    treated exactly like a wrong type or a wrong length: it degrades the
    whole rerank to retrieval order, never a partial rescore.
    """
    return all(math.isfinite(score) for score in scores)


def rescore_and_sort(
    chunks: list[RetrievedChunk], scores: list[float], top_k: int
) -> list[RetrievedChunk]:
    """Sort chunks by their raw score, but store only a clamped view of it.

    Sorting happens on the raw score so a model's real ordering signal
    survives even when a score sits outside the documented [0, 1] band (for
    example 5.0 or -2.0): clamping before sorting would collapse distinct
    out-of-range scores together and destroy that signal. The value written
    back onto the chunk is clamped into [0, 1] because that is the band the
    contract and a later cross strategy evaluation phase both expect; the
    stored score is therefore a clamped view of the model's output, not a
    verbatim copy of it.
    """
    paired = list(zip(chunks, scores, strict=True))
    paired.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
    return [
        chunk.model_copy(update={"score": max(0.0, min(1.0, score))})
        for chunk, score in paired
    ][:top_k]
