"""Index tables fed by the ingestion fan out.

chunk_embeddings holds one vector per chunk for the configured embedding model.
The column is JSON on SQLite (tests, development) and pgvector's VECTOR on
PostgreSQL, through a dialect variant, so one migration serves both. The
dimension is stored per row; an HNSW index needs a fixed dimension and is
created by Phase 3 once the deployment's embedding model is known.

chunk_search holds the tsvector for full text search on PostgreSQL (Text on
SQLite) with a GIN index. document_id and collection_id are copied here so the
access filter can be applied inside the index query without a join.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base

EmbeddingType = JSON().with_variant(Vector(), "postgresql")
TsvType = Text().with_variant(TSVECTOR(), "postgresql")


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    collection_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding = mapped_column(EmbeddingType, nullable=False)


class ChunkSearch(Base):
    __tablename__ = "chunk_search"
    __table_args__ = (Index("ix_chunk_search_tsv", "tsv", postgresql_using="gin"),)

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    collection_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    tsv = mapped_column(TsvType, nullable=False)
