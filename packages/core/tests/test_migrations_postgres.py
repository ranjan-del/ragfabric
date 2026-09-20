"""PostgreSQL-gated tests for migration 0004. Needs RAGFABRIC_TEST_DATABASE_URL.

``test_migrations.py`` runs every migration, including 0004, against SQLite,
but 0004's actual body (the ``DELETE`` of mismatched-dimension rows, the
``ALTER COLUMN ... TYPE vector(768)``, and the HNSW index) is gated behind
``if bind.dialect.name != "postgresql": return`` and never runs there. That
means the one dialect this migration exists for is never exercised by the
migration test suite at all, so the exact mismatch this migration guards
against (the model saying one thing, the live column enforcing another)
could drift without a single test noticing. These tests run the real
migration against a real PostgreSQL database and check the three things it
does there: it deletes rows whose ``dim`` does not match ``EMBEDDING_DIM``,
it pins the column type to ``vector(EMBEDDING_DIM)``, and it builds the HNSW
index with the cosine opclass.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models.index import EMBEDDING_DIM

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL"),
]


@pytest.fixture()
def pg_at_0003():
    """A real PostgreSQL database migrated up to, but not past, 0003.

    Revision 0003 is where ``chunk_embeddings`` first exists, with its
    ``embedding`` column still the unconstrained ``vector`` type 0004 is
    about to pin. Tests insert rows at this schema version, then upgrade to
    head themselves so they can observe exactly what 0004's own body does.
    """
    downgrade(URL)
    upgrade(URL, "0003_ingestion_and_indexes")
    engine = create_engine(URL)
    yield engine
    engine.dispose()
    downgrade(URL)


def _insert_chunk_with_embedding(engine, dim: int) -> int:
    """Insert one document/chunk/chunk_embeddings row at schema 0003.

    Returns the chunk id, which doubles as the ``chunk_embeddings`` primary
    key. The embedding is an all-zero vector of ``dim`` dimensions, cast to
    ``vector`` the same way a real ``INSERT`` through the ORM would coerce a
    Python list, since at 0003 the column has no fixed width yet.
    """
    vector_literal = "[" + ",".join("0" for _ in range(dim)) + "]"
    with engine.begin() as conn:
        doc_id = conn.execute(
            text(
                "INSERT INTO documents (filename, content_type, format, document_type, "
                "storage_path, collection_id, owner_id, version, status, num_chunks, "
                "error, created_at) "
                "VALUES ('a.txt', '', 'txt', '', NULL, NULL, NULL, 1, 'ready', 0, '', now()) "
                "RETURNING id"
            )
        ).scalar_one()
        chunk_id = conn.execute(
            text(
                "INSERT INTO chunks (document_id, collection_id, chunk_index, page, "
                "char_start, char_end, text, embedding, section) "
                "VALUES (:doc_id, NULL, 0, 1, 0, 1, 'hello', '[]', NULL) "
                "RETURNING id"
            ),
            {"doc_id": doc_id},
        ).scalar_one()
        conn.execute(
            text(
                "INSERT INTO chunk_embeddings "
                "(chunk_id, document_id, collection_id, model, dim, embedding) "
                "VALUES (:chunk_id, :doc_id, NULL, 'm', :dim, CAST(:vector AS vector))"
            ),
            {"chunk_id": chunk_id, "doc_id": doc_id, "dim": dim, "vector": vector_literal},
        )
    return chunk_id


def test_0004_deletes_rows_whose_dim_does_not_match(pg_at_0003):
    """The migration's own docstring promises this deletion; prove it."""
    engine = pg_at_0003
    good_id = _insert_chunk_with_embedding(engine, EMBEDDING_DIM)
    bad_id = _insert_chunk_with_embedding(engine, 5)

    upgrade(URL, "head")

    with engine.connect() as conn:
        remaining = {
            row[0] for row in conn.execute(text("SELECT chunk_id FROM chunk_embeddings")).all()
        }
    assert good_id in remaining, "a row already at the pinned dimension must survive"
    assert bad_id not in remaining, "a row at the wrong dimension must be deleted, not left behind"


def test_0004_pins_the_column_to_the_shared_embedding_dim(pg_at_0003):
    """The live column type, read back from Postgres, must match the model's
    ``Vector(EMBEDDING_DIM)`` exactly, not merely "some fixed width"."""
    engine = pg_at_0003
    upgrade(URL, "head")

    with engine.connect() as conn:
        col_type = conn.execute(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod) "
                "FROM pg_attribute a "
                "WHERE a.attrelid = 'chunk_embeddings'::regclass "
                "AND a.attname = 'embedding'"
            )
        ).scalar_one()
    assert col_type == f"vector({EMBEDDING_DIM})"


def test_0004_builds_the_hnsw_index_with_the_cosine_opclass(pg_at_0003):
    """The index must be HNSW with ``vector_cosine_ops``, matching the
    cosine distance every store query (and ``ChromaVectorStore``'s
    ``hnsw:space: cosine`` metadata) actually uses."""
    engine = pg_at_0003
    upgrade(URL, "head")

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT am.amname, opc.opcname "
                "FROM pg_index i "
                "JOIN pg_class ic ON ic.oid = i.indexrelid "
                "JOIN pg_am am ON am.oid = ic.relam "
                "JOIN pg_opclass opc ON opc.oid = i.indclass[0] "
                "WHERE ic.relname = 'ix_chunk_embeddings_hnsw'"
            )
        ).one()
    assert row.amname == "hnsw"
    assert row.opcname == "vector_cosine_ops"
