"""Alembic environment.

Two things are worth reading here rather than skipping as boilerplate:

1. The database URL comes from `ragfabric_core.config.settings`, not from
   alembic.ini. The application and the migrations therefore always agree on
   which database they are talking to, and no connection string is committed.

2. `target_metadata` is the application's own `Base.metadata`, populated by
   importing the models. That is what lets `alembic revision --autogenerate`
   diff the models against the live schema, and what lets the migration test in
   packages/core/tests/test_migrations.py assert that the two have not drifted apart.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from ragfabric_core.config import get_settings

# Importing the model modules has the side effect of registering every table on
# Base.metadata. Without it, autogenerate would see an empty schema and cheerfully
# emit a migration that drops all the tables.
from ragfabric_core.models import document, user  # noqa: F401
from ragfabric_core.models.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def database_url() -> str:
    """The database these migrations run against.

    Normally the application's own DATABASE_URL, so `alembic upgrade head` and
    the running app can never disagree about which database they mean. A caller
    holding the Config object may override it first, which is how the migration
    tests point a run at a throwaway SQLite file instead of the real database.
    """
    override = config.get_main_option("sqlalchemy.url", None)
    return override or get_settings().database_url


config.set_main_option("sqlalchemy.url", database_url())

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it, for review or manual apply."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # SQLite cannot ALTER most things in place. Batch mode rewrites the
            # table instead, so the same migration script works on both SQLite
            # (local) and PostgreSQL (Docker/production).
            render_as_batch=connection.dialect.name == "sqlite",
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
