"""The reranker used when reranking is off. Truncation only, order preserved."""

from __future__ import annotations

from ragfabric_core.strategies.base import RetrievedChunk


class NoopReranker:
    name = "none"

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return list(chunks[:top_k])
