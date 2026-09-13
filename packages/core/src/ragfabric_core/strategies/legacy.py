"""The v1 hybrid pipeline behind the RetrieverStrategy interface.

This is deliberately thin. It exists so the interface is proven against real
retrieval code from day one and so the API can route through the registry
before Phase 3 replaces the internals with real embeddings and a real store.
The access filter is applied here, after the v1 retriever, because the v1
in-memory index predates the filter; Phase 3 stores take the filter natively.
"""

from __future__ import annotations

import time

from ragfabric_core.retrieve.hybrid import HybridRetriever
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    RetrievedChunk,
    StrategyName,
    TraceSpan,
)


class LegacyHybridStrategy:
    name = StrategyName.TRADITIONAL

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        started = time.perf_counter()
        collection_id = ctx.collection_ids[0] if ctx.collection_ids else None
        rows = HybridRetriever().retrieve(
            query, top_k=ctx.params.top_k, collection_id=collection_id
        )
        chunks = [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                collection_id=row.get("collection_id"),
                text=row["text"],
                page=row.get("page"),
                section=None,
                score=row.get("score"),
                char_start=row.get("char_start"),
                char_end=row.get("char_end"),
                metadata={
                    "filename": row.get("filename"),
                    "format": row.get("format"),
                    "hybrid_score": row.get("hybrid_score"),
                },
            )
            for row in rows
            if ctx.access_filter.allows(row["document_id"], row.get("collection_id"))
        ][: ctx.params.top_k]
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RetrievalResult(
            strategy=self.name,
            chunks=chunks,
            retrieval_calls=1,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=elapsed_ms,
            trace=[
                TraceSpan(
                    name="hybrid_search",
                    started_ms=0,
                    duration_ms=elapsed_ms,
                    attributes={"top_k": ctx.params.top_k, "returned": len(chunks)},
                )
            ],
        )
