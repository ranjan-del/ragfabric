"""Add connector_items: per-source sync state for the folder and Drive connectors.

See ``ragfabric_core.models.connector`` for what each column is for.

Revision ID: 0009_connector_items
Revises: 0008_graph_confidence_and_merges
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_connector_items"
down_revision: str | None = "0008_graph_confidence_and_merges"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "connector_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("connector", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_version", sa.String(), nullable=False),
        sa.Column("rejected_reason", sa.Text(), nullable=True),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("connector", "source_id", name="uq_connector_items_connector_source"),
    )
    with op.batch_alter_table("connector_items", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_connector_items_connector"), ["connector"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_connector_items_document_id"), ["document_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("connector_items", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_connector_items_document_id"))
        batch_op.drop_index(batch_op.f("ix_connector_items_connector"))
    op.drop_table("connector_items")
