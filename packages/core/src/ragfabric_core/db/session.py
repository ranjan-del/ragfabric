"""Database engine and session factory (SQLAlchemy).

Works with both SQLite (local default) and PostgreSQL (Docker / hosting).
- ``get_db`` is a FastAPI dependency that yields a session and always closes it.
- ``init_db`` creates every table (demo convenience; use Alembic in production).

The declarative ``Base`` lives in ``app.models.base``; it is re-exported here so
callers can do ``from ragfabric_core.db.session import Base``.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ragfabric_core.config import get_settings
from ragfabric_core.models.base import Base

settings = get_settings()

# SQLite needs check_same_thread=False when used from FastAPI's threadpool.
# Postgres does not, so only add it for sqlite URLs.
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    settings.database_url,
    future=True,
    pool_pre_ping=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


if engine.dialect.name == "sqlite":

    @event.listens_for(engine, "connect")
    def _enforce_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:  # noqa: ANN001, ARG001
        """Turn on foreign-key enforcement for every new SQLite DB-API connection.

        SQLite ships with foreign-key checking OFF by default and SQLAlchemy
        does not turn it on for you, so every ``ondelete="CASCADE"`` (and
        ``SET NULL``) declared on the models is silently inert on SQLite, the
        dialect this project uses for local development and for the entire
        test suite. That let at least one real bug through: an
        ``embeddings.chunk_embeddings`` row survived the deletion of its
        parent ``chunks`` row, because nothing was actually enforcing the
        cascade. ``PRAGMA foreign_keys=ON`` must be set on every connection
        (SQLite does not persist it in the file), which is exactly what a
        ``connect`` event listener gives us.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def init_db() -> None:
    """Create any missing tables, unless Alembic is in charge.

    Alembic owns the schema (`alembic upgrade head`). create_all remains as a
    local convenience so the app and the tests can run against a throwaway
    SQLite file with no migration step, and settings forces it off in
    production so the two can never both be in charge: create_all never alters
    an existing table, so a deployment relying on it would keep booting happily
    while its schema drifted away from the models.
    """
    if not get_settings().auto_create_tables:
        return

    # Imported for its side effect of registering every table on Base.metadata.
    from ragfabric_core import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
