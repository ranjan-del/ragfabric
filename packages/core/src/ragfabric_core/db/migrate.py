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
