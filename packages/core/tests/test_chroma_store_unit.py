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


def _matches(where: dict | None, metadata: dict) -> bool:
    """A tiny interpreter for the where documents chroma_where/access_stats build.

    Only the operators those two producers actually emit (``$and``, ``$or``,
    ``$eq``, ``$in``, ``$nin``) need to be understood here.
    """
    if where is None:
        return True
    if "$and" in where:
        return all(_matches(clause, metadata) for clause in where["$and"])
    if "$or" in where:
        return any(_matches(clause, metadata) for clause in where["$or"])
    ((field, op_value),) = where.items()
    ((op, value),) = op_value.items()
    if op == "$eq":
        return metadata.get(field) == value
    if op == "$in":
        return metadata.get(field) in value
    if op == "$nin":
        return metadata.get(field) not in value
    raise AssertionError(f"unsupported operator {op!r} in a test where document")


class _FakeCollectionWithData(_FakeCollection):
    """A fake collection that actually stores rows, for ``get(where=...)``.

    ``_FakeCollection`` above only needs to record ``query`` calls; proving
    ``access_stats`` counts real, access-narrowed candidates needs a fake that
    can be queried back, not just inspected after the fact.
    """

    def __init__(self, rows: dict[int, dict]) -> None:
        super().__init__()
        self._rows = rows
        self.get_calls: list[dict] = []

    def get(self, where=None, include=None):
        self.get_calls.append({"where": where, "include": include})
        ids = [cid for cid, metadata in self._rows.items() if _matches(where, metadata)]
        return {"ids": ids}

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


def test_access_stats_reports_the_real_before_and_after_counts():
    """Prove access_stats measures the access-narrowed count, not an estimate.

    Two chunks exist: one in collection 100, one in collection 200. A
    restrictive access filter genuinely removes the second, so ``before`` and
    ``after`` must differ by exactly that one row, not merely satisfy
    ``before >= after``.
    """
    rows = {
        1: {"document_id": 10, "collection_id": 100, "model": "hashing-8"},
        2: {"document_id": 20, "collection_id": 200, "model": "hashing-8"},
    }
    collection = _FakeCollectionWithData(rows)
    store = ChromaVectorStore(_FakeClient(collection), model="hashing-8")

    restricted = AccessFilter(collection_ids=frozenset({100}))
    assert store.access_stats({}, restricted) == (2, 1)
    assert store.access_stats({}, AccessFilter.unrestricted()) == (2, 2)

    # A metadata filter narrows the candidate pool before access is applied:
    # only chunk 2 matches document_id 20, and it is not in the permitted
    # collection.
    assert store.access_stats({"document_id": 20}, restricted) == (1, 0)

    # A denied document wins over an otherwise unrestricted filter.
    denied = AccessFilter(denied_document_ids=frozenset({20}))
    assert store.access_stats({}, denied) == (2, 1)

    # A store pinned to a different model never sees these rows at all.
    other_model = ChromaVectorStore(_FakeClient(collection), model="nomic-embed-text")
    assert other_model.access_stats({}, AccessFilter.unrestricted()) == (0, 0)


def test_access_stats_resolves_a_format_filter_against_the_chunks_table():
    """format is not Chroma metadata (see module docstring), so a format
    filter must fall back to resolving the surviving ids against the chunks
    table rather than silently being ignored."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from ragfabric_core.models import Base
    from ragfabric_core.models.document import Chunk, Document

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    sf = sessionmaker(bind=engine, expire_on_commit=False)
    with sf() as db:
        doc_txt = Document(filename="a.txt", format="txt", status="ready")
        doc_pdf = Document(filename="b.pdf", format="pdf", status="ready")
        db.add_all([doc_txt, doc_pdf])
        db.flush()
        chunk_txt = Chunk(
            document_id=doc_txt.id,
            chunk_index=0,
            text="txt chunk",
            embedding=[],
        )
        chunk_pdf = Chunk(
            document_id=doc_pdf.id,
            chunk_index=0,
            text="pdf chunk",
            embedding=[],
        )
        db.add_all([chunk_txt, chunk_pdf])
        db.commit()
        txt_id, pdf_id, txt_doc, pdf_doc = chunk_txt.id, chunk_pdf.id, doc_txt.id, doc_pdf.id

    rows = {
        txt_id: {"document_id": txt_doc, "collection_id": -1, "model": "hashing-8"},
        pdf_id: {"document_id": pdf_doc, "collection_id": -1, "model": "hashing-8"},
    }
    collection = _FakeCollectionWithData(rows)
    store = ChromaVectorStore(_FakeClient(collection), model="hashing-8", session_factory=sf)

    assert store.access_stats({"format": "txt"}, AccessFilter.unrestricted()) == (1, 1)


def test_chroma_store_satisfies_the_vector_store_protocol():
    """VectorStore is runtime_checkable: a store missing any method (including
    access_stats, added in Task 13) fails isinstance rather than raising at
    call time. PgVectorStore already has this assertion
    (test_stores_sqlite.py::test_stores_satisfy_the_protocols); this is the
    matching one for ChromaVectorStore, which needs no live server, only a
    stub client, to construct.
    """
    from ragfabric_core.stores.base import VectorStore

    store = ChromaVectorStore(_FakeClient(_FakeCollection()))
    assert isinstance(store, VectorStore)
