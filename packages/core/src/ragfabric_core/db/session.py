"""Database engine and session factory (SQLAlchemy).

Works with both SQLite (local default) and PostgreSQL (Docker / hosting).
- ``get_db`` is a FastAPI dependency that yields a session and always closes it.
- ``init_db`` creates every table (demo convenience; use Alembic in production).

The declarative ``Base`` lives in ``app.models.base``; it is re-exported here so
callers can do ``from ragfabric_core.db.session import Base``.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
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
