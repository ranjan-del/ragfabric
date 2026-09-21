"""VectorlessRAGStrategy: two lexical store queries, fused, boosted, cut.

The plan writes these tests against a ``strategy.retrieve(query, access=...,
top_k=...)`` signature and a ``strategy.ask(...)`` method. Neither exists:
RetrieverStrategy is ``retrieve(query, ctx: RetrievalContext)`` and generation
deliberately lives above every strategy so all four are generated and cited by
identical code (ADR 0002). The tests below keep the plan's names and intent on
the real interface, and the citation test drives the real Phase 3 generator
rather than an invented method.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.generate.cited import generate_cited_answer
from ragfabric_core.generate.contract import assert_citation_contract
from ragfabric_core.models import Base
from ragfabric_core.providers.offline import ScriptedLLMProvider
from ragfabric_core.strategies import registry_defaults
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievedChunk,
    StrategyName,
    StrategyParams,
)
from ragfabric_core.strategies.contract import assert_strategy_contract
from ragfabric_core.strategies.vectorless import VectorlessRAGStrategy

ALL = AccessFilter.unrestricted()

OPEN_COLLECTION = 1
SECRET_COLLECTION = 2


class StubLexicalStore:
    """A LexicalStore double that applies the access filter, as a real one must."""

    def __init__(self, name: str, chunks: list[RetrievedChunk]) -> None:
        self.name = name
        self._chunks = chunks
        self.last_top_k: int | None = None
        self.last_access: AccessFilter | None = None
        self.last_filters: dict | None = None

    def index(self, chunk_ids, texts, payloads):  # pragma: no cover
        raise NotImplementedError

    def delete_document(self, document_id):  # pragma: no cover
        raise NotImplementedError

    def search(self, query, top_k, access, filters=None):
        self.last_top_k = top_k
        self.last_access = access
        self.last_filters = filters
        permitted = [c for c in self._chunks if access.allows(c.document_id, c.collection_id)]
        return permitted[:top_k]


def chunk(
    chunk_id: int,
    score: float = 1.0,
    text: str | None = None,
    collection_id: int = OPEN_COLLECTION,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=100 + collection_id,
        collection_id=collection_id,
        text=text or f"passage {chunk_id} about the retry limit",
        score=score,
    )


def ctx(top_k: int = 3, access: AccessFilter | None = None, **params):
    return RetrievalContext(
        principal=Principal(user_id=1, email="a@b.c", role="user"),
        access_filter=access or ALL,
        params=StrategyParams(top_k=top_k, **params),
    )


def build(bm25_chunks, ts_chunks, **kwargs) -> VectorlessRAGStrategy:
    return VectorlessRAGStrategy(
        bm25_store=StubLexicalStore("bm25", bm25_chunks),
        ts_rank_store=StubLexicalStore("postgres_fts", ts_chunks),
        **kwargs,
    )


@pytest.fixture()
def strategy():
    """Each store returns five chunks, and the two sets are disjoint.

    Disjoint on purpose: if the cut to top_k were applied per store instead of
    after fusion, the fused list would still be longer than top_k, which is
    what test_top_k_is_applied_after_fusion_not_per_store detects.
    """
    return build([chunk(i) for i in range(1, 6)], [chunk(i) for i in range(6, 11)])


class SpyEmbeddings:
    model, dim = "spy", 8

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        raise AssertionError("the vectorless strategy must not embed anything")


@pytest.fixture()
def spy_embeddings(monkeypatch):
    """The embedding provider the registry would hand to any strategy that wants one.

    Wired through default_registry rather than into VectorlessRAGStrategy,
    because the strategy takes no embedding provider at all: the only way an
    embedding call could happen is if the assembly reached for one, so that is
    where the spy goes. Its embed() raises, so a call fails loudly rather than
    merely incrementing a counter nobody checks.
    """
    spy = SpyEmbeddings()
    monkeypatch.setattr(registry_defaults, "build_embedding_provider", lambda cfg: spy)
    return spy


@pytest.fixture()
def registered(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_vectorless_is_registered_by_the_default_registry(registered, spy_embeddings):
    registry = registry_defaults.default_registry(RagFabricConfig(), registered)
    assert StrategyName.VECTORLESS in registry.names()
    assert isinstance(registry.get("vectorless"), VectorlessRAGStrategy)


def test_vectorless_makes_no_embedding_call(registered, spy_embeddings):
    registry = registry_defaults.default_registry(RagFabricConfig(), registered)
    result = registry.get("vectorless").retrieve("anything", ctx(top_k=5))
    assert spy_embeddings.calls == 0
    # The structural half of the claim: there is no embedding provider to call.
    assert not hasattr(registry.get("vectorless"), "embedder")
    assert "embed" not in {span.name for span in result.trace}


def test_it_reports_zero_embedding_work_rather_than_omitting_it(strategy):
    """ADR 0004: the count is zero because nothing happened, not defaulted."""
    result = strategy.retrieve("the retry limit", ctx(top_k=3))
    assert result.llm_calls == 0
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.retrieval_calls == 2


def test_access_filtered_chunks_never_reach_the_ranker():
    restricted_access = AccessFilter(collection_ids=frozenset({OPEN_COLLECTION}))
    strategy = build(
        [chunk(1), chunk(2, collection_id=SECRET_COLLECTION)],
        [chunk(3, collection_id=SECRET_COLLECTION), chunk(4)],
    )
    hits = strategy.retrieve("quota", ctx(top_k=10, access=restricted_access)).chunks
    assert hits
    assert all(h.collection_id in restricted_access.collection_ids for h in hits)


def test_the_access_filter_is_handed_to_both_stores_not_applied_afterwards():
    restricted = AccessFilter(collection_ids=frozenset({OPEN_COLLECTION}))
    strategy = build([chunk(1)], [chunk(2)])
    strategy.retrieve("quota", ctx(top_k=5, access=restricted))
    assert strategy.bm25_store.last_access == restricted
    assert strategy.ts_rank_store.last_access == restricted


def test_top_k_is_applied_after_fusion_not_per_store(strategy):
    # each store returns 5; fused and cut to 3
    assert len(strategy.retrieve("quota", ctx(top_k=3)).chunks) == 3


def test_it_satisfies_the_strategy_contract(strategy):
    result = assert_strategy_contract(strategy, "the retry limit", ctx(top_k=2))
    assert result.strategy == StrategyName.VECTORLESS
    assert len(result.chunks) == 2


def test_identifier_boosting_reorders_the_fused_list():
    exact = chunk(9, score=0.1, text="raised as ERR_QUOTA_4419 by the exporter")
    scattered = chunk(1, score=5.0, text="an err about a quota, code 4419, somewhere")
    strategy = build([scattered, exact], [scattered, exact], identifier_boost=100.0)
    hits = strategy.retrieve("why ERR_QUOTA_4419", ctx(top_k=2)).chunks
    assert hits[0].chunk_id == 9


def test_it_records_the_expected_trace_spans(strategy):
    names = [span.name for span in strategy.retrieve("quota", ctx(top_k=3)).trace]
    assert names == ["bm25_search", "ts_rank_search", "fuse", "boost", "context_budget"]


def test_it_enforces_the_context_budget(strategy):
    result = strategy.retrieve("quota", ctx(top_k=5))
    tight = build(
        [chunk(i) for i in range(1, 6)],
        [chunk(i) for i in range(6, 11)],
        max_context_tokens=1,
    ).retrieve("quota", ctx(top_k=5))
    assert result.chunks
    assert tight.chunks == []


def test_metadata_filters_reach_both_stores(strategy):
    strategy.retrieve("quota", ctx(top_k=3, metadata_filters={"format": "pdf"}))
    assert strategy.bm25_store.last_filters == {"format": "pdf"}
    assert strategy.ts_rank_store.last_filters == {"format": "pdf"}


def test_an_answer_carries_citations_that_pass_the_phase_3_contract(strategy):
    result = strategy.retrieve("what is the retry limit for ERR_QUOTA_4419", ctx(top_k=3))
    llm = ScriptedLLMProvider(["The retry limit is described in passage one [1]."])
    answer = generate_cited_answer("what is the retry limit for ERR_QUOTA_4419", result.chunks, llm)
    assert answer.text
    assert_citation_contract(answer.text, result.chunks)  # real API, do not reimplement
