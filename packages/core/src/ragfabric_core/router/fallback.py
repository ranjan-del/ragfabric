"""What auto does when the strategy it chose finds nothing.

One step, never a chain. A fallback that found nothing either is reported as
empty, with both attempts in the trace, rather than trying a third strategy
the caller never asked for.
"""

from __future__ import annotations

from ragfabric_core.stores.fusion import rrf
from ragfabric_core.strategies.base import RetrievalResult, StrategyName

NO_EVIDENCE = "no evidence"
_COUNTERS = (
    "retrieval_calls",
    "llm_calls",
    "embedding_calls",
    "input_tokens",
    "output_tokens",
    "latency_ms",
)


def fallback_for(chosen: StrategyName, result: RetrievalResult) -> StrategyName | None:
    if result.chunks or chosen is StrategyName.TRADITIONAL:
        return None
    return StrategyName.TRADITIONAL


def empty_reason(result: RetrievalResult) -> str:
    if result.subgraph is not None and result.subgraph.empty_reason is not None:
        return result.subgraph.empty_reason.value
    return NO_EVIDENCE


def _summed(first: RetrievalResult, second: RetrievalResult) -> dict[str, int]:
    return {name: getattr(first, name) + getattr(second, name) for name in _COUNTERS}


def combine(
    first: RetrievalResult, second: RetrievalResult, *, fallback_from: StrategyName
) -> RetrievalResult:
    """The fallback's evidence, with what both attempts cost.

    ``sub_questions`` and ``subgraph`` come from the fallback only. Generation
    picks its path from those fields, so keeping the failed agent's reports
    would answer a Traditional result on the agentic path.
    """
    return second.model_copy(
        update={
            **_summed(first, second),
            "trace": [*first.trace, *second.trace],
            "fallback_from": fallback_from,
        }
    )


def fuse(
    traditional: RetrievalResult, vectorless: RetrievalResult, *, top_k: int
) -> RetrievalResult:
    """Traditional and Vectorless fused with RRF (ADR 0008), cut to top_k after fusion."""
    chunks = rrf([traditional.chunks, vectorless.chunks])[:top_k]
    return traditional.model_copy(
        update={
            **_summed(traditional, vectorless),
            "chunks": chunks,
            "trace": [*traditional.trace, *vectorless.trace],
        }
    )
