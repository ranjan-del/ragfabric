"""Tests for reindex_all: batched re-embedding of an existing corpus.

There is no shared session_factory/seed_chunks fixture in this repo (see
test_stores_sqlite.py's local `sf` fixture for the established pattern), so
this file seeds its own SQLite engine and schema rather than depending on
fixtures that do not exist.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.ingest.reindex import reindex_all
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.providers.offline import HashingEmbeddingProvider
from ragfabric_core.stores.pgvector_store import PgVectorStore


@pytest.fixture()
def sf(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'reindex.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        doc = Document(filename="f.txt", format="txt", status="ready", owner_id=None)
        db.add(doc)
        db.flush()
        db.add_all(
            [
                Chunk(
                    document_id=doc.id,
                    collection_id=None,
                    chunk_index=i,
                    text=f"chunk {i}",
                    embedding=[],
                )
                for i in range(5)
            ]
        )
        db.commit()
        doc_id = doc.id
    return factory, doc_id


def test_reindex_writes_one_vector_per_chunk_under_the_new_model(sf):
    factory, _ = sf
    provider = HashingEmbeddingProvider(dim=8)
    store = PgVectorStore(factory, model=provider.model)
    with factory() as db:
        count = reindex_all(
            db, embedding_provider=provider, vector_store=store, lexical_store=None, batch_size=2
        )
    assert count == 5
    assert store.count() == 5
    hits = store.query([1.0] * 8, top_k=10, access=AccessFilter.unrestricted())
    assert len(hits) == 5


def test_reindex_replaces_vectors_from_the_previous_model(sf):
    factory, _ = sf
    old = HashingEmbeddingProvider(dim=4)
    old_store = PgVectorStore(factory, model=old.model)
    with factory() as db:
        reindex_all(
            db, embedding_provider=old, vector_store=old_store, lexical_store=None, batch_size=5
        )
    new = HashingEmbeddingProvider(dim=8)
    new_store = PgVectorStore(factory, model=new.model)
    with factory() as db:
        reindex_all(
            db, embedding_provider=new, vector_store=new_store, lexical_store=None, batch_size=5
        )

    assert new_store.count() == 5, "reindex must replace vectors, not accumulate them"
    assert old_store.query([1.0] * 4, top_k=10, access=AccessFilter.unrestricted()) == []


def test_reindex_reports_progress_in_batches(sf):
    factory, _ = sf
    provider = HashingEmbeddingProvider(dim=8)
    store = PgVectorStore(factory, model=provider.model)
    seen: list[tuple[int, int]] = []
    with factory() as db:
        reindex_all(
            db,
            embedding_provider=provider,
            vector_store=store,
            lexical_store=None,
            batch_size=2,
            on_progress=lambda done, total: seen.append((done, total)),
        )
    assert seen[-1] == (5, 5)
    assert all(total == 5 for _, total in seen)
