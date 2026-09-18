"""Unit tests for ChromaVectorStore that need neither the network nor a real
chromadb client. See test_stores_chroma.py for the integration tests against
a live Chroma container.

Why this test exists: chroma_where deliberately maps an empty allow set to a
literal ``{"field": {"$in": []}}`` (ADR 0003: an empty allow set must match
nothing, never everything), but Chroma's server rejects an empty ``$in``
operand as invalid input rather than treating it as "never true" (confirmed
against a live container while implementing this task). ChromaVectorStore
must therefore recognise an unsatisfiable predicate itself and never send it
to Chroma at all, which this test verifies by asserting the fake collection's
``query`` is never called.
"""

from __future__ import annotations

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.stores.chroma_store import ChromaVectorStore


class _FakeCollection:
    def __init__(self) -> None:
        self.query_calls: list[dict] = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {"ids": [[]], "distances": [[]]}


class _FakeClient:
    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection

    def get_or_create_collection(self, name, metadata):
        return self._collection


def test_an_empty_allow_set_never_reaches_chroma_as_a_request():
    collection = _FakeCollection()
    store = ChromaVectorStore(_FakeClient(collection))

    access = AccessFilter(document_ids=frozenset(), collection_ids=None)
    hits = store.query([1.0, 0.0], top_k=5, access=access)

    assert hits == []
    assert collection.query_calls == [], (
        "an unsatisfiable access predicate must short-circuit before any request "
        "is sent to Chroma, which rejects an empty $in operand as invalid input"
    )


def test_an_empty_document_allow_set_ored_with_a_real_collection_allow_still_queries():
    collection = _FakeCollection()
    store = ChromaVectorStore(_FakeClient(collection))

    # document_ids is explicitly empty but collection_ids still allows something,
    # so the OR as a whole is satisfiable: the impossible branch must be dropped,
    # not treated as making the whole predicate impossible.
    access = AccessFilter(document_ids=frozenset(), collection_ids=frozenset({7}))
    store.query([1.0, 0.0], top_k=5, access=access)

    assert len(collection.query_calls) == 1
    where = collection.query_calls[0]["where"]
    assert where == {"collection_id": {"$in": [7]}}


def test_model_property_matches_the_constructor_argument():
    store = ChromaVectorStore(_FakeClient(_FakeCollection()), model="nomic-embed-text")
    assert store.model == "nomic-embed-text"
