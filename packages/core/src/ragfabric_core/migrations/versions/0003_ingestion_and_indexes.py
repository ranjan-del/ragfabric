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

revision: str = "0003_ingestion_and_indexes"
down_revision: str | None = "0002_platform_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("section", sa.String(), nullable=True))

    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("document_type", sa.String(), server_default="", nullable=False)
        )
        batch_op.add_column(sa.Column("storage_path", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_column("storage_path")
        batch_op.drop_column("document_type")

    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.drop_column("section")
