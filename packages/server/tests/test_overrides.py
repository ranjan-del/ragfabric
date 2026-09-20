"""Per-request retrieval overrides (search) and per-upload chunking overrides.

Field names verified against the real routes/schemas before writing these
assertions:

- ``DocumentOut`` (``schemas/document.py``) has no ``chunk_count`` field; the
  upload response exposes the chunk count as ``num_chunks`` (also asserted by
  ``test_api.py::test_upload_ingest_and_query_flow``). Every assertion below
  reads ``num_chunks``, not ``chunk_count``.
"""

from __future__ import annotations

from ragfabric_core.providers.offline import HashingEmbeddingProvider
from ragfabric_core.rerank.noop import NoopReranker
from ragfabric_core.strategies.base import StrategyName, StrategyRegistry
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy
from ragfabric_server.api.routes.search import _strategy_for


class _StubStore:
    """Bare enough to satisfy TraditionalRAGStrategy.__init__; never queried
    by this test, which only inspects object identity, never calls retrieve.
    """

    name = "stub"


def test_strategy_for_none_returns_the_shared_instance_unchanged():
    """The default (no rerank override) must be the exact registry object,
    not a copy, so behaviour for a plain request is unaffected by Task 12.
    """
    shared = TraditionalRAGStrategy(
        embedding_provider=HashingEmbeddingProvider(dim=16),
        vector_store=_StubStore(),
        reranker=None,
    )
    registry = StrategyRegistry()
    registry.register(shared)

    resolved = _strategy_for(None, registry, llm=None)

    assert resolved is shared


def test_strategy_for_rerank_builds_a_fresh_strategy_without_mutating_the_shared_one():
    """A named reranker must build a NEW strategy object around the shared
    store/embedder rather than mutate the registry's instance in place: two
    concurrent requests, one naming a reranker and one not, must never
    observe each other's choice through a shared mutable strategy.
    """
    original_reranker = NoopReranker()
    embedder = HashingEmbeddingProvider(dim=16)
    store = _StubStore()
    shared = TraditionalRAGStrategy(
        embedding_provider=embedder,
        vector_store=store,
        reranker=original_reranker,
    )
    registry = StrategyRegistry()
    registry.register(shared)

    # "none" is a deliberate rerank CHOICE ("no reranking, regardless of what
    # the shared strategy is configured with"), distinct from the field's
    # own default of None ("no override, use the shared strategy as is").
    # cross_encoder/llm are exercised over HTTP in the tests below; "none"
    # is used here so this unit test never needs sentence-transformers or a
    # network-shaped LLM double, only NoopReranker.
    resolved = _strategy_for("none", registry, llm=None)

    # A fresh object, not the shared one, ...
    assert resolved is not shared
    # ... built around the SAME shared store and embedding provider, per the
    # brief's requirement (never a private attribute reach-around: both are
    # read through the public store/embedder properties inside _strategy_for)
    assert resolved.store is store
    assert resolved.embedder is embedder
    # ... and the shared instance the registry hands to every OTHER request
    # still has its own original reranker: nothing about resolving this
    # request touched it.
    still_shared = registry.get(StrategyName.TRADITIONAL)
    assert still_shared is shared
    assert still_shared._reranker is original_reranker  # noqa: SLF001 (test-only introspection)


def test_similarity_threshold_in_the_request_filters_results(client, admin_token, ingested_doc):
    loose = client.post(
        "/api/search/semantic",
        json={"query": "entirely unrelated aardvark", "top_k": 10, "similarity_threshold": 0.0},
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()["results"]
    strict = client.post(
        "/api/search/semantic",
        json={"query": "entirely unrelated aardvark", "top_k": 10, "similarity_threshold": 0.99},
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()["results"]
    assert len(strict) <= len(loose)
    assert all(r["score"] >= 0.99 for r in strict)


def test_top_k_is_honoured_per_request(client, admin_token, ingested_doc):
    res = client.post(
        "/api/search/semantic",
        json={"query": "leave", "top_k": 1},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert len(res.json()["results"]) <= 1


def test_an_out_of_range_threshold_is_rejected_rather_than_clamped(client, admin_token):
    res = client.post(
        "/api/search/semantic",
        json={"query": "leave", "similarity_threshold": 1.5},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 422


def test_rerank_override_does_not_mutate_the_shared_strategy(client, admin_token, ingested_doc):
    """A request naming a reranker must not leak into a later request that
    names none: the shared registry instance is built once with no reranker
    override applied, so a later request behaves exactly as the first one
    did before any override was ever sent.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}
    baseline = client.post(
        "/api/search/semantic",
        json={"query": "leave", "top_k": 5},
        headers=headers,
    )
    assert baseline.status_code == 200, baseline.text

    overridden = client.post(
        "/api/search/semantic",
        json={"query": "leave", "top_k": 5, "rerank": "llm"},
        headers=headers,
    )
    assert overridden.status_code == 200, overridden.text

    after = client.post(
        "/api/search/semantic",
        json={"query": "leave", "top_k": 5},
        headers=headers,
    )
    assert after.status_code == 200, after.text
    assert after.json()["results"] == baseline.json()["results"]


def test_upload_accepts_a_chunk_size_override(client, admin_token, tmp_path):
    body = ("sentence one. " * 200).encode()
    res = client.post(
        "/api/documents/upload",
        files={"file": ("big.txt", body, "text/plain")},
        data={"chunk_size": "200", "chunk_overlap": "20"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code in (200, 201), res.text
    small = res.json()["num_chunks"]

    res = client.post(
        "/api/documents/upload",
        files={"file": ("big2.txt", body, "text/plain")},
        data={"chunk_size": "2000", "chunk_overlap": "20"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    large = res.json()["num_chunks"]
    assert small > large, "a smaller chunk size must produce more chunks"


def test_an_invalid_chunk_overlap_is_rejected(client, admin_token):
    res = client.post(
        "/api/documents/upload",
        files={"file": ("x.txt", b"hello", "text/plain")},
        data={"chunk_size": "100", "chunk_overlap": "500"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 422, "overlap larger than the chunk size must be refused"


def test_upload_without_overrides_keeps_the_configured_chunk_size(client, admin_token):
    """No chunk_size/chunk_overlap in the request: identical to pre-Task-12
    behaviour, using whatever ragfabric.yaml configures."""
    res = client.post(
        "/api/documents/upload",
        files={"file": ("plain.txt", b"hello world", "text/plain")},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 201, res.text
    assert res.json()["num_chunks"] >= 1
