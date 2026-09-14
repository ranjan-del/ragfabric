"""Store behaviour on SQLite (the dialect fallbacks). The PostgreSQL paths are in test_stores_postgres.py."""

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore


@pytest.fixture()
def sf(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 's.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        c1, c2 = Collection(name="open"), Collection(name="secret")
        db.add_all([c1, c2])
        db.flush()
        d1 = Document(filename="a.txt", format="txt", collection_id=c1.id, status="ready")
        d2 = Document(filename="b.txt", format="txt", collection_id=c2.id, status="ready")
        db.add_all([d1, d2])
        db.flush()
        rows = [
            Chunk(
                document_id=d1.id,
                collection_id=c1.id,
                chunk_index=0,
                page=1,
                char_start=0,
                char_end=10,
                text="leave policy twelve days",
                embedding=[],
            ),
            Chunk(
                document_id=d1.id,
                collection_id=c1.id,
                chunk_index=1,
                page=1,
                char_start=0,
                char_end=10,
                text="kubernetes rollout guide",
                embedding=[],
            ),
            Chunk(
                document_id=d2.id,
                collection_id=c2.id,
                chunk_index=0,
                page=1,
                char_start=0,
                char_end=10,
                text="secret leave bonus rules",
                embedding=[],
            ),
        ]
        db.add_all(rows)
        db.commit()
        ids = [r.id for r in rows]
    return factory, ids, (d1.id, d2.id), (c1.id, c2.id)


def unit(v):
    a = np.asarray(v, dtype=float)
    return (a / np.linalg.norm(a)).tolist()


def test_pgvector_store_upsert_query_filter_and_delete(sf):
    factory, ids, (d1, d2), (c1, c2) = sf
    store = PgVectorStore(factory)
    vecs = [unit([1, 0, 0]), unit([0, 1, 0]), unit([0.9, 0.1, 0])]
    payloads = [
        {"model": "hashing-3", "dim": 3, "document_id": d, "collection_id": c}
        for d, c in ((d1, c1), (d1, c1), (d2, c2))
    ]
    store.upsert(ids, vecs, payloads)
    assert store.count() == 3
    hits = store.query(unit([1, 0, 0]), top_k=2, access=AccessFilter.unrestricted())
    assert [h.chunk_id for h in hits] == [ids[0], ids[2]] and hits[0].score > hits[1].score
    only_open = store.query(
        unit([1, 0, 0]), top_k=3, access=AccessFilter(collection_ids=frozenset({c1}))
    )
    assert {h.collection_id for h in only_open} == {c1}
    denied = store.query(
        unit([1, 0, 0]), top_k=3, access=AccessFilter(denied_document_ids=frozenset({d2}))
    )
    assert all(h.document_id != d2 for h in denied)
    store.upsert([ids[0]], [unit([0, 0, 1])], [payloads[0]])  # replace
    assert store.count() == 3
    store.delete_document(d1)
    assert store.count() == 1


def test_lexical_store_index_search_filter_and_delete(sf):
    factory, ids, (d1, d2), (c1, c2) = sf
    store = PostgresLexicalStore(factory)
    texts = ["leave policy twelve days", "kubernetes rollout guide", "secret leave bonus rules"]
    payloads = [{"document_id": d, "collection_id": c} for d, c in ((d1, c1), (d1, c1), (d2, c2))]
    store.index(ids, texts, payloads)
    hits = store.search("leave", top_k=5, access=AccessFilter.unrestricted())
    assert {h.chunk_id for h in hits} == {ids[0], ids[2]}
    assert hits[0].text in texts
    scoped = store.search("leave", top_k=5, access=AccessFilter(collection_ids=frozenset({c1})))
    assert [h.chunk_id for h in scoped] == [ids[0]]
    assert store.search("nothingmatches", top_k=5, access=AccessFilter.unrestricted()) == []
    store.delete_document(d2)
    assert {
        h.chunk_id for h in store.search("leave", top_k=5, access=AccessFilter.unrestricted())
    } == {ids[0]}


def test_stores_satisfy_the_protocols(sf):
    from ragfabric_core.stores.base import LexicalStore, VectorStore

    factory, *_ = sf
    assert isinstance(PgVectorStore(factory), VectorStore)
    assert isinstance(PostgresLexicalStore(factory), LexicalStore)
