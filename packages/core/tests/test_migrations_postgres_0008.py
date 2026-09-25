"""PostgreSQL-gated tests for migration 0008. Needs RAGFABRIC_TEST_DATABASE_URL.

``test_migrations.py`` runs migration 0008 against SQLite, including its
``CheckConstraint`` on ``confidence``, but SQLite and PostgreSQL can differ
in how strictly a schema-level constraint is enforced, and the whole point
of ADR 0004's range check is that it holds regardless of which process
writes the row. These tests run the real migration against a real
PostgreSQL database: the upgrade/downgrade round trip, and that the
confidence CHECK constraint actually rejects an out of range value there.
"""

from __future__ import annotations

import os

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import Entity, EntitySource

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL"),
]


@pytest.fixture()
def pg_at_head():
    downgrade(URL)
    upgrade(URL, "head")
    engine = create_engine(URL)
    yield engine
    engine.dispose()
    downgrade(URL)


def test_0008_upgrade_creates_provenance_and_merge_tables(pg_at_head):
    engine = pg_at_head
    tables = set(sa.inspect(engine).get_table_names())
    assert {"entity_sources", "relationship_sources", "entity_merges"} <= tables

    ecols = {c["name"] for c in sa.inspect(engine).get_columns("entities")}
    assert {"confidence", "extraction_model"} <= ecols
    assert "source_chunk_ids" not in ecols

    escols = {c["name"] for c in sa.inspect(engine).get_columns("entity_sources")}
    assert {"confidence", "extraction_model"} <= escols
    rscols = {c["name"] for c in sa.inspect(engine).get_columns("relationship_sources")}
    assert {"confidence", "extraction_model"} <= rscols


def test_0008_downgrade_restores_source_chunk_ids_and_drops_new_tables(pg_at_head):
    downgrade(URL, "0007_bm25_term_stats")
    engine = create_engine(URL)
    try:
        tables = set(sa.inspect(engine).get_table_names())
        assert "entity_sources" not in tables
        assert "relationship_sources" not in tables
        assert "entity_merges" not in tables

        ecols = {c["name"] for c in sa.inspect(engine).get_columns("entities")}
        assert "source_chunk_ids" in ecols
        assert "confidence" not in ecols
    finally:
        engine.dispose()
        upgrade(URL, "head")


def test_0008_confidence_check_constraint_rejects_out_of_range_values(pg_at_head):
    engine = pg_at_head
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add(
            Entity(
                name="Ada Lovelace",
                normalized_name="ada lovelace",
                entity_type="person",
                confidence=1.5,
            )
        )
        with pytest.raises(sa.exc.IntegrityError):
            db.commit()


def test_0008_confidence_check_constraint_allows_none_and_in_range_values(pg_at_head):
    engine = pg_at_head
    factory = sessionmaker(bind=engine)
    with factory() as db:
        unmeasured = Entity(name="Ada", normalized_name="ada", entity_type="person")
        measured = Entity(name="Bob", normalized_name="bob", entity_type="person", confidence=0.9)
        db.add_all([unmeasured, measured])
        db.commit()
        assert unmeasured.confidence is None
        assert measured.confidence == 0.9


def test_0008_entity_sources_confidence_check_constraint_rejects_out_of_range_values(pg_at_head):
    """R20: the per-source confidence column enforces the same [0, 1] range."""
    engine = pg_at_head
    factory = sessionmaker(bind=engine)
    with factory() as db:
        collection = Collection(name="c")
        db.add(collection)
        db.flush()
        document = Document(
            filename="a.txt", format="txt", collection_id=collection.id, status="ready"
        )
        db.add(document)
        db.flush()
        chunk = Chunk(
            document_id=document.id,
            collection_id=collection.id,
            chunk_index=0,
            text="Ada Lovelace",
            embedding=[0.0],
        )
        entity = Entity(name="Ada", normalized_name="ada", entity_type="person")
        db.add_all([chunk, entity])
        db.flush()

        db.add(
            EntitySource(entity_id=entity.id, chunk_id=chunk.id, confidence=1.5, surface_name="x")
        )
        with pytest.raises(sa.exc.IntegrityError):
            db.commit()
