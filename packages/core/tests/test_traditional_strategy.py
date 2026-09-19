from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.providers.offline import HashingEmbeddingProvider
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievedChunk,
    StrategyName,
    StrategyParams,
)
from ragfabric_core.strategies.contract import assert_strategy_contract
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy


class StubStore:
    name = "stub"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_top_k: int | None = None
        self.last_access: AccessFilter | None = None
        self.last_filters: dict | None = None

    def upsert(self, chunk_ids, vectors, payloads):  # pragma: no cover
        raise NotImplementedError

    def query(self, vector, top_k, access, filters=None):
        self.last_top_k = top_k
        self.last_access = access
        self.last_filters = filters
        return [c for c in self._chunks if access.allows(c.document_id, c.collection_id)][:top_k]

    def delete_document(self, document_id):  # pragma: no cover
        raise NotImplementedError

    def count(self):
        return len(self._chunks)


def chunk(cid: int, score: float, doc: int = 1, col: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=doc,
        collection_id=col,
        text=f"passage {cid} about leave policy",
        score=score,
    )


def ctx(top_k: int = 3, threshold: float = 0.0, access: AccessFilter | None = None):
    return RetrievalContext(
        principal=Principal(user_id=1, email="a@b.c", role="user"),
        access_filter=access or AccessFilter.unrestricted(),
        params=StrategyParams(top_k=top_k, similarity_threshold=threshold),
    )


def strategy(store, reranker=None, max_context_tokens: int = 6000):
    return TraditionalRAGStrategy(
        embedding_provider=HashingEmbeddingProvider(dim=16),
        vector_store=store,
        reranker=reranker,
        max_context_tokens=max_context_tokens,
    )


def test_it_satisfies_the_strategy_contract():
    store = StubStore([chunk(1, 0.9), chunk(2, 0.8), chunk(3, 0.7)])
    result = assert_strategy_contract(strategy(store), "how much leave do I get", ctx(top_k=2))
    assert result.strategy == StrategyName.TRADITIONAL
    assert len(result.chunks) == 2


def test_it_asks_the_store_for_more_candidates_than_top_k():
    store = StubStore([chunk(i, 1.0 - i / 100) for i in range(1, 30)])
    strategy(store).retrieve("q", ctx(top_k=5))
    assert store.last_top_k == 15, "candidates must be top_k * candidate_multiplier"


def test_it_passes_the_access_filter_into_the_store_rather_than_filtering_after():
    store = StubStore([chunk(1, 0.9, doc=1), chunk(2, 0.8, doc=2)])
    access = AccessFilter(document_ids=frozenset({2}), collection_ids=None)
    result = strategy(store).retrieve("q", ctx(access=access))
    assert store.last_access is access
    assert [c.chunk_id for c in result.chunks] == [2]


def test_it_drops_candidates_below_the_similarity_threshold():
    store = StubStore([chunk(1, 0.9), chunk(2, 0.2), chunk(3, 0.05)])
    result = strategy(store).retrieve("q", ctx(top_k=3, threshold=0.25))
    assert [c.chunk_id for c in result.chunks] == [1]


def test_it_returns_no_chunks_when_everything_is_below_the_threshold():
    store = StubStore([chunk(1, 0.1)])
    result = strategy(store).retrieve("q", ctx(threshold=0.5))
    assert result.chunks == []
    assert result.retrieval_calls == 1


def test_it_applies_the_reranker_after_the_threshold_and_before_the_cut():
    from ragfabric_core.rerank.noop import NoopReranker

    class Reverse(NoopReranker):
        name = "reverse"

        def rerank(self, query, chunks, top_k):
            return list(reversed(chunks))[:top_k]

    store = StubStore([chunk(1, 0.9), chunk(2, 0.8), chunk(3, 0.7)])
    result = strategy(store, reranker=Reverse()).retrieve("q", ctx(top_k=2))
    assert [c.chunk_id for c in result.chunks] == [3, 2]


def test_it_records_the_expected_trace_spans():
    store = StubStore([chunk(1, 0.9)])
    result = strategy(store).retrieve("q", ctx())
    assert [s.name for s in result.trace] == [
        "embed_query",
        "vector_search",
        "context_budget",
    ]


def test_it_records_a_rerank_span_when_a_real_reranker_is_configured():
    from ragfabric_core.rerank.noop import NoopReranker

    class Named(NoopReranker):
        name = "llm"

    store = StubStore([chunk(1, 0.9)])
    result = strategy(store, reranker=Named()).retrieve("q", ctx())
    assert "rerank" in [s.name for s in result.trace]


def test_it_enforces_the_context_budget_by_dropping_the_tail():
    store = StubStore([chunk(1, 0.9), chunk(2, 0.8), chunk(3, 0.7)])
    result = strategy(store, max_context_tokens=9).retrieve("q", ctx(top_k=3))
    assert len(result.chunks) < 3
    budget_span = [s for s in result.trace if s.name == "context_budget"][0]
    assert budget_span.attributes["tokens_used"] <= 9


def test_the_context_budget_span_records_the_token_count_method():
    store = StubStore([chunk(1, 0.9)])
    result = strategy(store).retrieve("q", ctx())
    budget_span = [s for s in result.trace if s.name == "context_budget"][0]
    assert budget_span.attributes["token_count_method"] == "estimate"


def test_metadata_filters_reach_the_store():
    store = StubStore([chunk(1, 0.9)])
    context = RetrievalContext(
        principal=Principal(user_id=1, email="a@b.c", role="user"),
        access_filter=AccessFilter.unrestricted(),
        collection_ids=[7],
        params=StrategyParams(top_k=3, metadata_filters={"format": "pdf"}),
    )
    strategy(store).retrieve("q", context)
    assert store.last_filters == {"collection_id": 7, "format": "pdf"}


def test_it_exposes_the_store_it_was_built_with():
    store = StubStore([chunk(1, 0.9)])
    built = strategy(store)
    assert built.store is store
