"""Pin the vector dimension and build the HNSW index.

pgvector cannot index a vector column with no declared dimension, so Phase 3
fixes the column at 768 (nomic-embed-text, the default no key model) and builds
an HNSW index with cosine ops.

Rows whose dim is not 768 are deleted first, because the type change cannot
succeed while they exist. They are derived data: `ragfabric reindex` rebuilds
them from the chunks table, which is the source of truth. See ADR 0006.

Revision ID: 0004_pin_vector_dim_and_hnsw
Revises: 0003_ingestion_and_indexes
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_pin_vector_dim_and_hnsw"
down_revision: str | None = "0003_ingestion_and_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    bind = op.get_bind()
    op.create_index("ix_chunk_embeddings_model", "chunk_embeddings", ["model"])
    if bind.dialect.name != "postgresql":
        return
    op.execute(sa.text(f"DELETE FROM chunk_embeddings WHERE dim <> {EMBEDDING_DIM}"))
    op.execute(
        sa.text(
            f"ALTER TABLE chunk_embeddings "
            f"ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) "
            f"USING embedding::vector({EMBEDDING_DIM})"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX ix_chunk_embeddings_hnsw ON chunk_embeddings "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw"))
        op.execute(sa.text("ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector"))
    op.drop_index("ix_chunk_embeddings_model", table_name="chunk_embeddings")
