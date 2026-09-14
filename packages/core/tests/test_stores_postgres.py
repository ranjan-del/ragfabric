"""Real PostgreSQL paths: pgvector cosine distance and tsvector ranking. Needs RAGFABRIC_TEST_DATABASE_URL."""

import os

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL"),
]


@pytest.fixture()
def pg():
    downgrade(URL)
    upgrade(URL)
    engine = create_engine(URL)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        c = Collection(name="c")
        db.add(c)
        db.flush()
        d = Document(filename="a.txt", format="txt", collection_id=c.id, status="ready")
        db.add(d)
        db.flush()
        rows = [
            Chunk(
                document_id=d.id,
                collection_id=c.id,
                chunk_index=i,
                page=1,
                char_start=0,
                char_end=1,
                text=t,
                embedding=[],
            )
            for i, t in enumerate(
                [
                    "annual leave is twelve days",
                    "the cluster rollout failed",
                    "leave carry forward capped",
                ]
            )
        ]
        db.add_all(rows)
        db.commit()
        ids = [r.id for r in rows]
    yield factory, ids, d.id, c.id
    downgrade(URL)


def unit(v):
    a = np.asarray(v, dtype=float)
    return (a / np.linalg.norm(a)).tolist()


def test_pgvector_uses_the_vector_type_and_orders_by_cosine_distance(pg):
    factory, ids, d, c = pg
    store = PgVectorStore(factory)
    store.upsert(
        ids,
        [unit([1, 0]), unit([0, 1]), unit([0.8, 0.6])],
        [{"model": "m", "dim": 2, "document_id": d, "collection_id": c}] * 3,
    )
    with factory() as db:
        typ = db.execute(
            text(
                "select udt_name from information_schema.columns where table_name='chunk_embeddings' and column_name='embedding'"
            )
        ).scalar()
    assert typ == "vector"
    hits = store.query(unit([1, 0]), top_k=3, access=AccessFilter.unrestricted())
    assert [h.chunk_id for h in hits] == [ids[0], ids[2], ids[1]]


def test_fts_ranks_with_ts_rank_cd_and_uses_the_gin_index(pg):
    factory, ids, d, c = pg
    store = PostgresLexicalStore(factory)
    store.index(
        ids,
        ["annual leave is twelve days", "the cluster rollout failed", "leave carry forward capped"],
        [{"document_id": d, "collection_id": c}] * 3,
    )
    with factory() as db:
        idx = db.execute(
            text(
                "select indexdef from pg_indexes where tablename='chunk_search' and indexname='ix_chunk_search_tsv'"
            )
        ).scalar()
    assert idx and "gin" in idx.lower()
    hits = store.search("leave", top_k=5, access=AccessFilter.unrestricted())
    assert {h.chunk_id for h in hits} == {ids[0], ids[2]} and all(h.score is not None for h in hits)
