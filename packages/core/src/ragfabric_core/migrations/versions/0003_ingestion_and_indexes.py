"""Ingestion metadata and index tables.

Adds documents.document_type and documents.storage_path, chunks.section, and
(in the same revision, added by the store task) chunk_embeddings and
chunk_search.

Revision ID: 0003_ingestion_and_indexes
Revises: 0002_platform_tables
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR

revision: str = "0003_ingestion_and_indexes"
down_revision: str | None = "0002_platform_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("section", sa.String(), nullable=True))

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("document_type", sa.String(), server_default="", nullable=False)
        )
        batch_op.add_column(sa.Column("storage_path", sa.String(), nullable=True))

    embedding_type = sa.JSON().with_variant(Vector(), "postgresql")
    tsv_type = sa.Text().with_variant(TSVECTOR(), "postgresql")
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("collection_id", sa.Integer(), nullable=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("embedding", embedding_type, nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    with op.batch_alter_table("chunk_embeddings", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_chunk_embeddings_collection_id"), ["collection_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_chunk_embeddings_document_id"), ["document_id"], unique=False
        )
    op.create_table(
        "chunk_search",
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("collection_id", sa.Integer(), nullable=True),
        sa.Column("tsv", tsv_type, nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    with op.batch_alter_table("chunk_search", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_chunk_search_collection_id"), ["collection_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_chunk_search_document_id"), ["document_id"], unique=False
        )
        batch_op.create_index("ix_chunk_search_tsv", ["tsv"], unique=False, postgresql_using="gin")


def downgrade() -> None:
    with op.batch_alter_table("chunk_search", schema=None) as batch_op:
        batch_op.drop_index("ix_chunk_search_tsv")
        batch_op.drop_index(batch_op.f("ix_chunk_search_document_id"))
        batch_op.drop_index(batch_op.f("ix_chunk_search_collection_id"))
    op.drop_table("chunk_search")

    with op.batch_alter_table("chunk_embeddings", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_chunk_embeddings_document_id"))
        batch_op.drop_index(batch_op.f("ix_chunk_embeddings_collection_id"))
    op.drop_table("chunk_embeddings")

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_column("storage_path")
        batch_op.drop_column("document_type")

    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.drop_column("section")
