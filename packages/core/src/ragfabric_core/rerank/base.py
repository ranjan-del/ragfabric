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

from typing import Protocol, runtime_checkable

from ragfabric_core.strategies.base import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...
