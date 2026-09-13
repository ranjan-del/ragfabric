"""The strategy contract is the product: four implementations, one result shape."""

import pytest

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.strategies.base import (
    Budget,
    RetrievalContext,
    RetrievalResult,
    RetrievedChunk,
    RetrieverStrategy,
    StrategyName,
    StrategyParams,
    StrategyRegistry,
    TraceSpan,
)
from ragfabric_core.strategies.contract import assert_strategy_contract


def make_ctx(**overrides) -> RetrievalContext:
    values = dict(
        principal=Principal(
            user_id=1, email="u@example.com", role="user", group_ids=[], api_key_id=None
        ),
        access_filter=AccessFilter.unrestricted(),
        collection_ids=None,
        params=StrategyParams(),
        budget=Budget(),
    )
    values.update(overrides)
    return RetrievalContext(**values)


class StaticStrategy:
    """Minimal conforming implementation used to test the contract itself."""

    name = StrategyName.VECTORLESS

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        chunk = RetrievedChunk(
            chunk_id=1,
            document_id=1,
            collection_id=None,
            text=f"about {query}",
            page=1,
            section=None,
            score=0.9,
            char_start=0,
            char_end=10,
            metadata={},
        )
        return RetrievalResult(
            strategy=self.name,
            chunks=[chunk],
            retrieval_calls=1,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=3,
            trace=[
                TraceSpan(name="lexical_search", started_ms=0, duration_ms=3, attributes={"k": 1})
            ],
        )


def test_static_strategy_satisfies_the_runtime_protocol():
    assert isinstance(StaticStrategy(), RetrieverStrategy)


def test_contract_helper_returns_the_result_for_a_conforming_strategy():
    result = assert_strategy_contract(StaticStrategy(), "leave policy", make_ctx())
    assert result.strategy == StrategyName.VECTORLESS
    assert result.chunks[0].text == "about leave policy"


def test_contract_helper_rejects_a_result_whose_strategy_name_lies():
    class Liar(StaticStrategy):
        def retrieve(self, query, ctx):
            result = super().retrieve(query, ctx)
            return result.model_copy(update={"strategy": StrategyName.GRAPH})

    with pytest.raises(AssertionError, match="strategy"):
        assert_strategy_contract(Liar(), "q", make_ctx())


def test_contract_helper_rejects_more_chunks_than_top_k():
    class TooMany(StaticStrategy):
        def retrieve(self, query, ctx):
            result = super().retrieve(query, ctx)
            return result.model_copy(update={"chunks": result.chunks * 3})

    with pytest.raises(AssertionError, match="top_k"):
        assert_strategy_contract(TooMany(), "q", make_ctx(params=StrategyParams(top_k=2)))


def test_access_filter_unrestricted_allows_everything():
    f = AccessFilter.unrestricted()
    assert f.allows(document_id=42, collection_id=None)


def test_access_filter_restricts_by_document_and_collection():
    f = AccessFilter(document_ids=frozenset({1, 2}), collection_ids=frozenset({10}))
    assert f.allows(document_id=1, collection_id=None)
    assert f.allows(document_id=99, collection_id=10)
    assert not f.allows(document_id=99, collection_id=11)


def test_registry_round_trip_and_unknown_name():
    registry = StrategyRegistry()
    registry.register(StaticStrategy())
    assert registry.names() == [StrategyName.VECTORLESS]
    assert registry.get(StrategyName.VECTORLESS).name == StrategyName.VECTORLESS
    with pytest.raises(KeyError):
        registry.get(StrategyName.GRAPH)


def test_result_counts_cannot_be_negative():
    with pytest.raises(ValueError):
        RetrievalResult(
            strategy=StrategyName.TRADITIONAL,
            chunks=[],
            retrieval_calls=-1,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            trace=[],
        )


def test_legacy_hybrid_strategy_meets_the_contract(monkeypatch):
    """The v1 pipeline, wrapped, is the first real strategy behind the interface."""
    from ragfabric_core.strategies.legacy import LegacyHybridStrategy

    class FakeRetriever:
        def retrieve(self, query, top_k, collection_id=None, document_id=None, format=None):
            return [
                {
                    "chunk_id": 7,
                    "document_id": 3,
                    "collection_id": None,
                    "text": "leave is 12 days",
                    "page": 2,
                    "score": 0.42,
                    "char_start": 5,
                    "char_end": 21,
                    "filename": "hr.pdf",
                },
            ][:top_k]

    monkeypatch.setattr("ragfabric_core.strategies.legacy.HybridRetriever", lambda: FakeRetriever())
    strategy = LegacyHybridStrategy()
    assert strategy.name == StrategyName.TRADITIONAL
    result = assert_strategy_contract(strategy, "leave", make_ctx(params=StrategyParams(top_k=3)))
    assert result.retrieval_calls == 1 and result.llm_calls == 0
    assert result.chunks[0].metadata["filename"] == "hr.pdf"
    assert result.trace[0].name == "hybrid_search"
