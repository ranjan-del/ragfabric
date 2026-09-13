"""SQLAlchemy declarative base and shared helpers.

Every ORM model inherits from ``Base``. ``utcnow`` is the single source of truth
for timestamp defaults so all rows use timezone-aware UTC.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import DeclarativeBase


def utcnow() -> datetime:
    """Timezone-aware current UTC time (used as a column default)."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass
