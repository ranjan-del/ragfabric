"""Executable contract for RetrieverStrategy implementations.

Plugin authors call this from their own tests. It runs the strategy once and
asserts the invariants every downstream component relies on.
"""

from __future__ import annotations

from ragfabric_core.strategies.base import RetrievalContext, RetrievalResult, RetrieverStrategy


def assert_strategy_contract(
    strategy: RetrieverStrategy, query: str, ctx: RetrievalContext
) -> RetrievalResult:
    assert isinstance(strategy, RetrieverStrategy), "object does not implement RetrieverStrategy"
    result = strategy.retrieve(query, ctx)
    assert isinstance(result, RetrievalResult), "retrieve() must return a RetrievalResult"
    assert result.strategy == strategy.name, (
        f"result.strategy is {result.strategy!r} but the strategy is named {strategy.name!r}"
    )
    assert len(result.chunks) <= ctx.params.top_k, (
        f"returned {len(result.chunks)} chunks, more than top_k={ctx.params.top_k}"
    )
    for chunk in result.chunks:
        assert ctx.access_filter.allows(chunk.document_id, chunk.collection_id), (
            f"chunk {chunk.chunk_id} is outside the caller's access filter"
        )
        assert chunk.text.strip(), f"chunk {chunk.chunk_id} has empty text"
    assert result.llm_calls <= ctx.budget.max_llm_calls, "strategy exceeded max_llm_calls"
    return result
