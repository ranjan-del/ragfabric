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
from sqlalchemy import engine_from_config, pool, text

# Importing the models package has the side effect of registering every table on
# Base.metadata. Without it, autogenerate would see an empty schema and cheerfully
# emit a migration that drops all the tables.
from ragfabric_core import models  # noqa: F401
from ragfabric_core.config import get_settings
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
        sqlite = connection.dialect.name == "sqlite"
        if sqlite:
            _sqlite_foreign_keys(connection, enabled=False)
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
        if sqlite:
            _sqlite_foreign_keys(connection, enabled=True)


def _sqlite_foreign_keys(connection, *, enabled: bool) -> None:
    """Switch SQLite foreign-key enforcement for this connection, outside any transaction.

    The application turns enforcement on for every SQLite connection
    (``db/session.py`` registers a global ``connect`` listener, so Alembic's
    connection gets it too). A batch migration rebuilds a table by copying it,
    dropping the original and renaming the copy, and with enforcement on that
    DROP TABLE is an implicit DELETE that fires every ``ON DELETE CASCADE``
    pointing at the table: rebuilding ``entities`` emptied ``relationships``.
    This is the standard recipe for batch mode: enforcement off for the run,
    back on afterwards.

    ``PRAGMA foreign_keys`` is a no-op inside a transaction, so it cannot live
    in a migration script (every script runs inside ``begin_transaction``).
    Here it runs before that transaction opens and after it has committed;
    the ``commit`` closes the transaction SQLAlchemy autobegins around the
    pragma itself, so the migrations still get a transaction of their own.
    Before switching enforcement back on, ``PRAGMA foreign_key_check`` must
    come back empty, so a migration can never leave a dangling reference
    behind that enforcement would have refused.
    """
    if enabled:
        violations = connection.execute(text("PRAGMA foreign_key_check")).all()
        if violations:
            raise RuntimeError(f"migrations left foreign key violations: {violations}")
    connection.execute(text(f"PRAGMA foreign_keys={'ON' if enabled else 'OFF'}"))
    connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
