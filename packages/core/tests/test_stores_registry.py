"""build_vector_store: kind switching, and the chroma extra's error path.

No network and no real chromadb client is touched here (see
test_stores_chroma.py for the integration tests against a live Chroma).
"""

from __future__ import annotations

import pytest

from ragfabric_core.config_file import VectorStoreConfig
from ragfabric_core.providers.base import ProviderError
from ragfabric_core.stores import registry
from ragfabric_core.stores.chroma_store import ChromaVectorStore
from ragfabric_core.stores.pgvector_store import PgVectorStore


def test_pgvector_and_memory_kinds_build_a_pgvector_store():
    sf = object()
    for kind in ("pgvector", "memory"):
        store = registry.build_vector_store(VectorStoreConfig(kind=kind), sf, embedding_model="m")
        assert isinstance(store, PgVectorStore)
        assert store.model == "m"


def test_chroma_kind_builds_a_chroma_store_from_chroma_url(monkeypatch):
    calls: dict = {}

    class FakeChromaModule:
        @staticmethod
        def HttpClient(host, port):
            calls["host"], calls["port"] = host, port
            return "fake-client"

    monkeypatch.setattr(registry, "_import_chromadb", lambda: FakeChromaModule)
    monkeypatch.setenv("CHROMA_URL", "http://chroma-host:9000")

    sf = object()
    store = registry.build_vector_store(VectorStoreConfig(kind="chroma"), sf, embedding_model="m")

    assert isinstance(store, ChromaVectorStore)
    assert store.model == "m"
    assert calls == {"host": "chroma-host", "port": 9000}


def test_chroma_url_defaults_to_localhost_8000(monkeypatch):
    calls: dict = {}

    class FakeChromaModule:
        @staticmethod
        def HttpClient(host, port):
            calls["host"], calls["port"] = host, port
            return "fake-client"

    monkeypatch.setattr(registry, "_import_chromadb", lambda: FakeChromaModule)
    monkeypatch.delenv("CHROMA_URL", raising=False)

    registry.build_vector_store(VectorStoreConfig(kind="chroma"), object())

    assert calls == {"host": "localhost", "port": 8000}


def test_chroma_kind_with_the_extra_missing_raises_a_clear_provider_error(monkeypatch):
    def _boom():
        raise ImportError("no module named chromadb")

    monkeypatch.setattr(registry, "_import_chromadb", _boom)

    with pytest.raises(ProviderError, match=r"ragfabric\[chroma\]"):
        registry.build_vector_store(VectorStoreConfig(kind="chroma"), object())
