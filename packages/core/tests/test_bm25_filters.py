"""Metadata filters on the BM25 path.

The filters go inside the SQL alongside the access predicate (ADR 0003). A
filter applied after the query would compute top_k over rows the caller asked
to exclude, and would hand back fewer than top_k rows while claiming the store
had nothing more to give.
"""

import os
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.stores.bm25_sql import Bm25Store

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")

ALL = AccessFilter.unrestricted()

# Ten chunks that all match "quota". Two of them live in the second
# collection, so a filter that runs before top_k returns exactly two and a
# filter that runs after top_k=5 returns at most one or two by luck.
MATCHING = [
    "The quota for exports is raised every quarter.",
    "A quota breach pages the on call engineer.",
    "Each tenant gets its own quota and its own budget.",
    "The quota resets at midnight in the tenant's own timezone.",
    "Quota accounting is eventually consistent across regions.",
    "Raising a quota requires an approval from the account owner.",
    "The quota dashboard lags the enforcement path by a minute.",
    "A soft quota warns, a hard quota rejects.",
    "Quota errors are retried with exponential backoff.",
    "The quota is expressed in requests per minute, not per second.",
]
IN_SECOND_COLLECTION = 2


@pytest.fixture(params=["sqlite", "postgresql"])
def corpus(request, tmp_path):
    if request.param == "postgresql":
        if not URL:
            pytest.skip("needs RAGFABRIC_TEST_DATABASE_URL")
        downgrade(URL)
        upgrade(URL)
        engine = create_engine(URL)
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'filters.db'}")
        Base.metadata.create_all(engine)

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        first, second = Collection(name="first"), Collection(name="second")
        db.add_all([first, second])
        db.flush()
        pdf_document = Document(
            filename="a.pdf", format="pdf", collection_id=first.id, status="ready"
        )
        txt_document = Document(
            filename="b.txt", format="txt", collection_id=second.id, status="ready"
        )
        db.add_all([pdf_document, txt_document])
        db.flush()

        placements = [
            (pdf_document, first) if i >= IN_SECOND_COLLECTION else (txt_document, second)
            for i in range(len(MATCHING))
        ]
        chunks = [
            Chunk(
                document_id=document.id,
                collection_id=collection.id,
                chunk_index=i,
                page=1,
                char_start=0,
                char_end=len(text),
                text=text,
                embedding=[],
            )
            for i, (text, (document, collection)) in enumerate(
                zip(MATCHING, placements, strict=True)
            )
        ]
        db.add_all(chunks)
        db.commit()
        ids = [c.id for c in chunks]
        payloads = [
            {"document_id": document.id, "collection_id": collection.id}
            for document, collection in placements
        ]
        second_collection_id = second.id
        pdf_document_id = pdf_document.id

    store = Bm25Store(factory)
    store.index(ids, MATCHING, payloads)
    return SimpleNamespace(
        store=store,
        second_collection_id=second_collection_id,
        pdf_document_id=pdf_document_id,
    )


@pytest.fixture()
def store(corpus):
    return corpus.store


def test_filters_are_applied_before_top_k_not_after(store, corpus):
    # 10 matching chunks, only 2 in the second collection
    hits = store.search(
        "quota", top_k=5, access=ALL, filters={"collection_id": corpus.second_collection_id}
    )
    assert len(hits) == IN_SECOND_COLLECTION
    assert all(h.collection_id == corpus.second_collection_id for h in hits)


def test_an_unknown_filter_key_is_rejected_not_ignored(store):
    with pytest.raises(ValueError):
        store.search("quota", top_k=5, access=ALL, filters={"nonsense": 1})


def test_the_document_id_filter_is_honoured(store, corpus):
    hits = store.search(
        "quota", top_k=10, access=ALL, filters={"document_id": corpus.pdf_document_id}
    )
    assert len(hits) == len(MATCHING) - IN_SECOND_COLLECTION
    assert all(h.document_id == corpus.pdf_document_id for h in hits)


def test_the_format_filter_is_honoured(store):
    hits = store.search("quota", top_k=10, access=ALL, filters={"format": "txt"})
    assert len(hits) == IN_SECOND_COLLECTION
    assert all(h.metadata["format"] == "txt" for h in hits)


def test_a_filter_and_the_access_predicate_both_apply(store, corpus):
    denied = AccessFilter(denied_document_ids=frozenset({corpus.pdf_document_id}))
    hits = store.search("quota", top_k=10, access=denied, filters={"format": "pdf"})
    assert hits == []
