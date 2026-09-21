"""SQLAlchemy declarative base and shared helpers.

Every ORM model inherits from ``Base``. ``utcnow`` is the single source of
truth for timestamp defaults, and every timestamp column is declared
``DateTime(timezone=True)`` (migration 0006), so a value written by this
process is always offset-aware UTC.

What ``timezone=True`` actually buys differs by dialect, and the difference
matters for anyone comparing timestamps across a fleet where SQLite and
PostgreSQL deployments coexist:

- PostgreSQL stores the column as ``TIMESTAMPTZ``. The offset is honoured on
  write (converted to UTC for storage) and a value read back is
  offset-aware, so timestamps compare correctly regardless of the writing
  process's local timezone.
- SQLite has no native timezone-aware datetime type. ``DateTime(timezone=True)``
  is accepted, but SQLAlchemy's SQLite dialect formats and parses the value
  the same way it would a naive column: the offset is silently dropped on
  write, and a value read back is a naive ``datetime`` (verified empirically:
  writing an aware UTC value and reading it back yields the same wall-clock
  value with ``tzinfo=None``). In practice, every timestamp on a SQLite
  deployment (including every test in this suite) is naive UTC, even though
  the column is declared timezone-aware. Comparisons stay correct only
  because every value written by this codebase is UTC to begin with; the
  column type does not enforce or guarantee that the way it does on
  PostgreSQL.
"""

from datetime import UTC, datetime

from sqlalchemy.orm import DeclarativeBase


def utcnow() -> datetime:
    """Timezone-aware current UTC time (used as a column default)."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass
