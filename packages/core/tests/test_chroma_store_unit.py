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

import pytest

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.stores.chroma_store import ChromaVectorStore, _prune_unsatisfiable


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


def test_prune_unsatisfiable_keeps_every_part_of_a_satisfiable_and():
    """Pins the reachable $and path: chroma_where never builds an $and with
    fewer than two parts, and both are satisfiable here, so nothing is dropped
    and always_false stays False."""
    node = {"$and": [{"document_id": {"$in": [1, 2]}}, {"model": {"$eq": "m"}}]}

    pruned, always_false = _prune_unsatisfiable(node)

    assert always_false is False
    assert pruned == node


def test_prune_unsatisfiable_fails_loud_not_open_on_a_malformed_empty_and():
    """_prune_unsatisfiable is only ever fed well-formed input by chroma_where,
    which never builds an empty $and (it always has at least two parts), so an
    empty $and cannot arrive through any real caller. This pins what happens
    if that invariant were ever violated anyway, by a future caller composing
    a where document by hand: the function must crash with IndexError rather
    than silently treating the empty $and as "no restriction" and running an
    unfiltered query, which would be the dangerous, fail-open direction for
    access-control code.
    """
    with pytest.raises(IndexError):
        _prune_unsatisfiable({"$and": []})
