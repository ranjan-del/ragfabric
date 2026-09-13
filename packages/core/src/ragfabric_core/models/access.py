"""Access control tables (ADR 0003).

Groups hold users. Grants give a group read or write on a collection.
Overrides restrict or allow one document below its collection. API keys are
principals with scopes. The audit log records what every run returned and
what the access filter removed.
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base, utcnow


class Group(Base):
    __tablename__ = "groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class GroupMember(Base):
    __tablename__ = "group_members"

    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class CollectionGrant(Base):
    __tablename__ = "collection_grants"
    __table_args__ = (
        UniqueConstraint("group_id", "collection_id", name="uq_grant_group_collection"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), nullable=False, index=True)
    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id"), nullable=False, index=True
    )
    # "read" | "write"
    permission: Mapped[str] = mapped_column(String, default="read", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class DocumentOverride(Base):
    __tablename__ = "document_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False, index=True)
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id"), nullable=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    # "deny" | "read"
    permission: Mapped[str] = mapped_column(String, default="deny", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    principal_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    collection_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    strategies: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    principal_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True)
    action: Mapped[str] = mapped_column(String, nullable=False, index=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    retrieval_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("retrieval_runs.id"), nullable=True
    )
    sources_returned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sources_filtered: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False, index=True
    )
