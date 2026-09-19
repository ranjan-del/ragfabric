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


def test_it_thresholds_using_the_retrieval_score_not_the_reranked_score():
    """Pins the ordering that is the substance of this task.

    A reranker here both reorders AND rescores, the way a real LLM or cross
    encoder reranker does. If the threshold ran after the rerank instead of
    before it, it would compare against the reranker's inverted score rather
    than the retrieval score, and a different chunk would survive: chunk 3
    (retrieval score 0.2, below the 0.5 threshold) would pass because its
    inverted rerank score is 0.8, while chunk 1 (retrieval score 0.9, well
    above threshold) would be dropped because its inverted score is 0.1. So
    this test fails under either ordering bug: threshold-after-rerank changes
    which chunk survives, not just their order.
    """
    from ragfabric_core.rerank.noop import NoopReranker

    class RescoringReverse(NoopReranker):
        name = "rescoring_reverse"

        def rerank(self, query, chunks, top_k):
            reversed_chunks = list(reversed(chunks))
            return [
                c.model_copy(update={"score": 1.0 - (c.score or 0.0)}) for c in reversed_chunks
            ][:top_k]

    store = StubStore([chunk(1, 0.9), chunk(2, 0.6), chunk(3, 0.2)])
    result = strategy(store, reranker=RescoringReverse()).retrieve(
        "q", ctx(top_k=3, threshold=0.5)
    )
    result_ids = [c.chunk_id for c in result.chunks]
    assert 3 not in result_ids, "chunk 3 is below the retrieval threshold and must be dropped"
    assert result_ids == [2, 1], "surviving chunks must be in the reranker's order"


def test_it_records_the_expected_trace_spans():
    store = StubStore([chunk(1, 0.9)])
    result = strategy(store).retrieve("q", ctx())
    assert [s.name for s in result.trace] == [
        "embed_query",
        "vector_search",
        "context_budget",
    ]
    started = [s.started_ms for s in result.trace]
    assert started == sorted(started), "spans must be recorded in the order they ran"
    assert all(s.duration_ms >= 0 for s in result.trace), "duration cannot be negative"


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


def test_a_caller_supplied_collection_id_filter_overrides_the_context_one():
    """Documents the precedence deliberately, since it is safe but was untested.

    This can only narrow, never widen, the result set: the AccessFilter (not
    this metadata dict) is what the store ANDs against its access predicate,
    so overriding this key cannot let a caller see a document or collection it
    was not already permitted to see.
    """
    store = StubStore([chunk(1, 0.9)])
    context = RetrievalContext(
        principal=Principal(user_id=1, email="a@b.c", role="user"),
        access_filter=AccessFilter.unrestricted(),
        collection_ids=[7],
        params=StrategyParams(top_k=3, metadata_filters={"collection_id": 99}),
    )
    strategy(store).retrieve("q", context)
    assert store.last_filters == {"collection_id": 99}


def test_it_exposes_the_store_it_was_built_with():
    store = StubStore([chunk(1, 0.9)])
    built = strategy(store)
    assert built.store is store


def test_llm_calls_is_zero_when_the_store_returns_no_candidates():
    """FINDING 1: LlmReranker.rerank() returns [] without calling the model on
    an empty candidate list. The name based guard alone cannot tell that
    apart from a real call, so llm_calls must also require candidates."""
    from ragfabric_core.rerank.noop import NoopReranker

    class Named(NoopReranker):
        name = "llm"

    store = StubStore([])
    result = strategy(store, reranker=Named()).retrieve("q", ctx())
    assert result.chunks == []
    assert result.llm_calls == 0


def test_llm_calls_is_zero_when_the_threshold_filters_everything_out():
    """FINDING 1, second path to the same bug: the threshold can also empty
    the candidate list before the reranker ever runs."""
    from ragfabric_core.rerank.noop import NoopReranker

    class Named(NoopReranker):
        name = "llm"

    store = StubStore([chunk(1, 0.1)])
    result = strategy(store, reranker=Named()).retrieve("q", ctx(threshold=0.5))
    assert result.chunks == []
    assert result.llm_calls == 0
