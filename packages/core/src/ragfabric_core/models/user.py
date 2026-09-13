"""User ORM model with a simple two-role RBAC scheme.

Roles are stored as a short string on the user row (``admin`` / ``user``),
which is the simplest design that fully supports the admin controls described
in MEMORY.md without a separate roles table.
"""

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base, utcnow


class Role(str, enum.Enum):
    """Allowed roles. Inherits from str so it serialises cleanly to JSON."""

    ADMIN = "admin"
    USER = "user"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default=Role.USER.value, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )
