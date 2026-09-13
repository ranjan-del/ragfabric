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

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from ragfabric_core import models  # noqa: F401  (registers every table)
from ragfabric_core.db.migrate import alembic_config as _alembic_config
from ragfabric_core.models.base import Base

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


def test_downgrade_removes_every_table(tmp_path):
    """A migration that cannot be rolled back is not a migration."""
    db_url = f"sqlite:///{tmp_path / 'roundtrip.db'}"
    config = _alembic_config(db_url)
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_engine(db_url)
    remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
    assert remaining == set()
