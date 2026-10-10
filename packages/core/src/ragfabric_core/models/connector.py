"""ConnectorItem: what a connector last saw for one source file, and the document it became.

One row per (connector name, source id). It is what makes a re-scan cheap and
safe: a source whose size, modified time and version match the row is skipped
without being fetched; one whose bytes hash the same is not re-ingested; and a
document is only ever deleted by the connector whose row points at it, never an
uploaded document or another connector's.

``document_id`` is nullable: a source the validator refused has a row (so it is
not fetched again until it changes) but no document. Deleting a document by
hand removes its row with it (``ON DELETE CASCADE``), so the next scan treats
the source as new and ingests it again, which is what a mirror should do.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base, utcnow


class ConnectorItem(Base):
    __tablename__ = "connector_items"
    __table_args__ = (
        UniqueConstraint("connector", "source_id", name="uq_connector_items_connector_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # The connector's configured name (connectors[].name in ragfabric.yaml).
    connector: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # Stable id within the connector: a path relative to the folder, a Drive file id.
    source_id: Mapped[str] = mapped_column(String, nullable=False)
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True
    )
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # A source-side version where the source has one (Drive's md5Checksum), else "".
    source_version: Mapped[str] = mapped_column(String, default="", nullable=False)
    # Why the validator refused this source, while it stays refused.
    rejected_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
