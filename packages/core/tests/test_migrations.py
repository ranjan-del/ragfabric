"""Tests that the Alembic migrations actually match the models.

A migration suite that is never checked is worse than none, because it looks
like a safety net while quietly drifting away from the models. Adding a column
to `models/user.py` and forgetting the migration is the classic version of
this: every test passes, because the test database was built by `create_all`
from those same models, and the failure only appears on a real deploy where
the schema came from Alembic instead.

`compare_metadata` is the fix. It is the same diffing engine
`alembic revision --autogenerate` uses, pointed at a database built purely by
the migrations and asked whether the models still agree with it. An empty diff
is the assertion; anything else names the exact column that was forgotten.
"""

from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from ragfabric_core import models  # noqa: F401  (registers every table)
from ragfabric_core.db import migrate
from ragfabric_core.db.migrate import alembic_config as _alembic_config
from ragfabric_core.models.base import Base
from ragfabric_core.models.user import User

EXPECTED_TABLES = {
    "users",
    "collections",
    "documents",
    "chunks",
    "query_logs",
    "groups",
    "group_members",
    "collection_grants",
    "document_overrides",
    "api_keys",
    "audit_log",
    "conversations",
    "messages",
    "retrieval_runs",
    "sources",
    "evaluation_runs",
    "evaluation_results",
    "entities",
    "relationships",
    "chunk_embeddings",
    "chunk_search",
    "ingestion_runs",
    "entity_sources",
    "relationship_sources",
    "entity_merges",
}


def _migrated_engine(tmp_path):
    """Build a database using only the migrations, and return an engine on it."""
    db_url = f"sqlite:///{tmp_path / 'migrated.db'}"
    command.upgrade(_alembic_config(db_url), "head")
    return create_engine(db_url), db_url


def test_upgrade_head_creates_every_table(tmp_path):
    engine, _ = _migrated_engine(tmp_path)
    tables = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES <= tables
    # Alembic's own bookkeeping table, proof the revision was actually stamped.
    assert "alembic_version" in tables


def test_migrations_match_the_models(tmp_path):
    """The models and the migration history must describe the same schema."""
    engine, _ = _migrated_engine(tmp_path)
    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={"compare_type": True, "target_metadata": Base.metadata},
        )
        diff = compare_metadata(context, Base.metadata)

    # alembic_version is Alembic's own table and is not in the models, so the
    # comparison is asked to ignore it rather than report it as a stray table.
    diff = [d for d in diff if "alembic_version" not in repr(d)]

    assert diff == [], (
        "The models and the Alembic migrations have drifted apart. "
        f"Run `alembic revision --autogenerate -m '...'` and review: {diff}"
    )


def test_0003_adds_section_document_type_and_storage_path(tmp_path):
    engine, _ = _migrated_engine(tmp_path)
    cols = {c["name"] for c in inspect(engine).get_columns("chunks")}
    assert "section" in cols
    dcols = {c["name"] for c in inspect(engine).get_columns("documents")}
    assert {"document_type", "storage_path"} <= dcols


def test_downgrade_removes_every_table(tmp_path):
    """A migration that cannot be rolled back is not a migration."""
    db_url = f"sqlite:///{tmp_path / 'roundtrip.db'}"
    config = _alembic_config(db_url)
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_engine(db_url)
    remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
    assert remaining == set()


def test_0004_creates_the_model_index_on_every_dialect(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    migrate.upgrade(url, "head")
    engine = sa.create_engine(url)
    names = {ix["name"] for ix in sa.inspect(engine).get_indexes("chunk_embeddings")}
    assert "ix_chunk_embeddings_model" in names


def test_0004_downgrade_returns_to_0003(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    migrate.upgrade(url, "head")
    migrate.downgrade(url, "0003_ingestion_and_indexes")
    engine = sa.create_engine(url)
    names = {ix["name"] for ix in sa.inspect(engine).get_indexes("chunk_embeddings")}
    assert "ix_chunk_embeddings_model" not in names
    assert "chunk_embeddings" in sa.inspect(engine).get_table_names()


def test_0006_declares_timezone_aware_columns_but_sqlite_still_returns_naive_values(tmp_path):
    """Documents the exact dialect gap ``models/base.py`` describes in prose.

    Migration 0006 declares every timestamp column ``DateTime(timezone=True)``,
    but SQLite has no native timezone-aware datetime type: the column accepts
    the declaration, and still silently drops the UTC offset on write and
    returns a naive ``datetime`` on read. This is not a bug in the migration;
    it is a limitation of the dialect, asserted here so nobody re-reads the
    docstring's claim as an untested guess, and so a future SQLAlchemy or
    pysqlite change that actually starts preserving the offset would be
    caught (this assertion would start failing, which is the point).
    """
    url = f"sqlite:///{tmp_path / 'm.db'}"
    migrate.upgrade(url, "head")
    engine = sa.create_engine(url)
    db = sessionmaker(bind=engine)()
    written = datetime.now(UTC)
    user = User(email="a@x", hashed_password="h", created_at=written)
    db.add(user)
    db.commit()
    db.expire_all()  # force a fresh SELECT rather than reusing the in-memory attribute

    reread = db.get(User, user.id).created_at
    assert written.tzinfo is not None, "the value this test wrote was genuinely offset-aware"
    assert reread.tzinfo is None, (
        "SQLite dropped the offset on round-trip, exactly as models/base.py documents; "
        "if this starts failing, SQLite/pysqlite/SQLAlchemy now preserves it and the "
        "docstring should be updated"
    )
    assert reread.replace(tzinfo=UTC) == written


def test_0008_adds_confidence_extraction_hash_and_provenance_tables(tmp_path):
    engine, _ = _migrated_engine(tmp_path)
    ecols = {c["name"] for c in inspect(engine).get_columns("entities")}
    assert {"confidence", "extraction_model"} <= ecols
    assert "source_chunk_ids" not in ecols

    rcols = {c["name"] for c in inspect(engine).get_columns("relationships")}
    assert {"confidence", "extraction_model"} <= rcols
    assert "source_chunk_ids" not in rcols

    ccols = {c["name"] for c in inspect(engine).get_columns("chunks")}
    assert "extraction_hash" in ccols

    tables = set(inspect(engine).get_table_names())
    assert {"entity_sources", "relationship_sources", "entity_merges"} <= tables


def test_0008_downgrade_returns_to_0007(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    migrate.upgrade(url, "head")
    migrate.downgrade(url, "0007_bm25_term_stats")
    engine = sa.create_engine(url)

    tables = set(sa.inspect(engine).get_table_names())
    assert "entity_sources" not in tables
    assert "relationship_sources" not in tables
    assert "entity_merges" not in tables

    ecols = {c["name"] for c in sa.inspect(engine).get_columns("entities")}
    assert "source_chunk_ids" in ecols
    assert "confidence" not in ecols
    assert "extraction_model" not in ecols

    rcols = {c["name"] for c in sa.inspect(engine).get_columns("relationships")}
    assert "source_chunk_ids" in rcols

    ccols = {c["name"] for c in sa.inspect(engine).get_columns("chunks")}
    assert "extraction_hash" not in ccols
