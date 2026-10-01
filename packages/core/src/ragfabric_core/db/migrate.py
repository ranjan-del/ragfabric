"""Programmatic Alembic entry points.

The migration scripts ship inside the package so `pip install ragfabric` can
migrate a database without a checkout. Both the CLI (`ragfabric db upgrade`)
and the tests go through these functions, so there is one way to run them.
"""

from __future__ import annotations

from importlib.resources import files

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = files("ragfabric_core") / "migrations"


def alembic_config(db_url: str) -> Config:
    """Alembic Config pointed at the packaged scripts and the given database."""
    config = Config(str(MIGRATIONS_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", db_url)
    return config


def upgrade(db_url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(db_url), revision)


def downgrade(db_url: str, revision: str = "base") -> None:
    command.downgrade(alembic_config(db_url), revision)


def current_revision(db_url: str) -> str | None:
    """The revision the database is at, or None when no migration has been applied."""
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine

    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()


def head_revision() -> str:
    """The newest revision the packaged scripts define."""
    from alembic.script import ScriptDirectory

    head = ScriptDirectory.from_config(alembic_config("sqlite://")).get_current_head()
    if head is None:
        raise RuntimeError("the packaged migrations define no revision")
    return head
