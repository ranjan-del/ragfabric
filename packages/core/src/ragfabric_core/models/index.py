"""Index tables fed by the ingestion fan out.

chunk_embeddings holds one vector per chunk for the configured embedding model.
The column is JSON on SQLite (tests, development) and pgvector's VECTOR on
PostgreSQL, through a dialect variant, so one migration serves both. The
dimension is stored per row for provenance, but since migration 0004 the
PostgreSQL column is fixed at vector(768) with an HNSW cosine index, and every
query filters on the active model name. One deployment, one embedding model
(ADR 0006).

chunk_search holds the tsvector for full text search on PostgreSQL (Text on
SQLite) with a GIN index. document_id and collection_id are copied here so the
access filter can be applied inside the index query without a join.

term_stats and corpus_stats support BM25 scoring on top of chunk_search's
tsvector (Phase 4). term_stats.df is the number of chunks containing a term;
corpus_stats is a single row (id=1) holding the running totals behind avgdl,
n_chunks and sum_len, rather than a precomputed average, because updating a
mean incrementally is lossy while a running sum stays exact. chunk_search.doc_len
is the length of that chunk in terms, the |D| term in the BM25 formula.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, BigInteger, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base

# Single source of truth for the pgvector column width (ADR 0006: one
# deployment, one embedding model, nomic-embed-text's 768 dims). Migration
# 0004 imports this same constant for its ALTER TABLE rather than repeating
# the number, so the model and the migration cannot silently disagree about
# the dimension PostgreSQL actually enforces.
EMBEDDING_DIM = 768

EmbeddingType = JSON().with_variant(Vector(EMBEDDING_DIM), "postgresql")
TsvType = Text().with_variant(TSVECTOR(), "postgresql")


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    collection_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    model: Mapped[str] = mapped_column(String, nullable=False, index=True)
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
    doc_len: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class TermStat(Base):
    __tablename__ = "term_stats"

    term: Mapped[str] = mapped_column(String(255), primary_key=True)
    df: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class CorpusStat(Base):
    """One row, id=1. Running totals, not averages: a mean cannot be
    updated incrementally without drifting."""

    __tablename__ = "corpus_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    n_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sum_len: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
