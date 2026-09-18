"""Chroma integration. Skipped unless RAGFABRIC_TEST_CHROMA_URL is set.

There is no shared session_factory fixture in this repo (see test_stores_sqlite.py's
local `sf` fixture for the established pattern), so this file seeds its own SQLite
engine and schema, the same way test_stores_sqlite.py and test_reindex.py do, rather
than depending on a fixture that does not exist.

Run with:
    docker compose --profile full up -d chroma
    RAGFABRIC_TEST_CHROMA_URL=http://localhost:8000 uv run pytest -m integration \
        packages/core/tests/test_stores_chroma.py
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models import Base
from ragfabric_core.stores.chroma_store import ChromaVectorStore

pytestmark = pytest.mark.integration

URL = os.environ.get("RAGFABRIC_TEST_CHROMA_URL")


@pytest.fixture()
def sf(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'chroma.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def store(sf):
    if not URL:
        pytest.skip("RAGFABRIC_TEST_CHROMA_URL is not set")
    import chromadb

    host, _, port = URL.removeprefix("http://").partition(":")
    client = chromadb.HttpClient(host=host, port=int(port or 8000))
    name = f"test_{uuid.uuid4().hex[:8]}"
    yield ChromaVectorStore(client, collection_name=name, model="hashing-8", session_factory=sf)
    client.delete_collection(name)


@pytest.fixture
def two_collections(sf):
    """Two documents in two different collections, three chunks each."""
    from ragfabric_core.models.document import Chunk, Collection, Document

    with sf() as db:
        col_a = Collection(name="a", owner_id=None)
        col_b = Collection(name="b", owner_id=None)
        db.add_all([col_a, col_b])
        db.flush()
        made = {}
        for label, col in (("a", col_a), ("b", col_b)):
            doc = Document(
                filename=f"{label}.txt",
                format="txt",
                status="ready",
                owner_id=None,
                collection_id=col.id,
            )
            db.add(doc)
            db.flush()
            chunks = [
                Chunk(
                    document_id=doc.id,
                    collection_id=col.id,
                    text=f"{label} chunk {i}",
                    chunk_index=i,
                    embedding=[],
                )
                for i in range(3)
            ]
            db.add_all(chunks)
            db.commit()
            made[label] = (doc.id, col.id, [c.id for c in chunks])
        return made


def test_chroma_returns_chunks_resolved_from_the_database(store, two_collections):
    doc_id, col_id, chunk_ids = two_collections["a"]
    store.upsert(
        chunk_ids,
        [[1.0] + [0.0] * 7, [0.9, 0.1] + [0.0] * 6, [0.0] * 7 + [1.0]],
        [{"document_id": doc_id, "collection_id": col_id, "model": "hashing-8", "dim": 8}] * 3,
    )
    hits = store.query([1.0] + [0.0] * 7, top_k=2, access=AccessFilter.unrestricted())
    assert [h.chunk_id for h in hits] == chunk_ids[:2]
    assert hits[0].text == "a chunk 0", "text must come from the chunks table"
    assert hits[0].score is not None and hits[0].score > hits[1].score


def test_chroma_applies_a_restrictive_access_filter_inside_the_query(store, two_collections):
    a_doc, a_col, a_chunks = two_collections["a"]
    b_doc, b_col, b_chunks = two_collections["b"]
    for doc_id, col_id, ids in ((a_doc, a_col, a_chunks), (b_doc, b_col, b_chunks)):
        store.upsert(
            ids,
            [[1.0] + [0.0] * 7] * 3,
            [{"document_id": doc_id, "collection_id": col_id, "model": "hashing-8", "dim": 8}] * 3,
        )

    access = AccessFilter(document_ids=None, collection_ids=frozenset({a_col}))
    hits = store.query([1.0] + [0.0] * 7, top_k=10, access=access)

    assert hits, "the permitted collection should still return chunks"
    assert {h.collection_id for h in hits} == {a_col}
    assert all(h.document_id == a_doc for h in hits)


def test_chroma_denies_everything_for_an_empty_allow_set(store, two_collections):
    doc_id, col_id, chunk_ids = two_collections["a"]
    store.upsert(
        chunk_ids,
        [[1.0] + [0.0] * 7] * 3,
        [{"document_id": doc_id, "collection_id": col_id, "model": "hashing-8", "dim": 8}] * 3,
    )
    access = AccessFilter(document_ids=frozenset(), collection_ids=None)
    assert store.query([1.0] + [0.0] * 7, top_k=10, access=access) == []
