"""Collection, Document, Chunk, and QueryLog ORM models.

- ``Collection`` groups documents (MEMORY.md: organise documents into collections).
- ``Document`` is one uploaded file plus its ingestion status and version.
- ``Chunk`` is a retrievable span of a document; its embedding is persisted as
  JSON so the in-memory vector index can be rebuilt after a restart.
- ``QueryLog`` records questions for the analytics dashboard.
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ragfabric_core.models.base import Base, utcnow


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, index=True, nullable=False)
    description: Mapped[str] = mapped_column(String, default="", nullable=False)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    documents: Mapped[list["Document"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, default="", nullable=False)
    # File extension used to route to the right parser (pdf/docx/pptx/txt/csv).
    format: Mapped[str] = mapped_column(String, default="", nullable=False)
    # document | presentation | table | text | other, derived from the extension
    document_type: Mapped[str] = mapped_column(
        String, server_default="", default="", nullable=False
    )
    # Relative path of the retained original under ingestion.uploads_dir, if kept
    storage_path: Mapped[str | None] = mapped_column(String, nullable=True)
    collection_id: Mapped[int | None] = mapped_column(
        ForeignKey("collections.id"), nullable=True, index=True
    )
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    # "processing" | "ready" | "failed"
    status: Mapped[str] = mapped_column(String, default="processing", nullable=False)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str] = mapped_column(String, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    collection: Mapped["Collection | None"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False, index=True)
    collection_id: Mapped[int | None] = mapped_column(
        ForeignKey("collections.id"), nullable=True, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # L2-normalised embedding stored as a JSON list of floats.
    embedding: Mapped[list] = mapped_column(JSON, nullable=False)
    # Nearest preceding heading detected before this chunk, if any.
    section: Mapped[str | None] = mapped_column(String, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    collection_id: Mapped[int | None] = mapped_column(ForeignKey("collections.id"), nullable=True)
    question: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # Document ids the answer actually cited, stored as a JSON list. Without
    # this, "most referenced documents" on the analytics page can only be
    # approximated by chunk count, which measures document SIZE, not usage.
    cited_document_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
