"""Add ingestion_runs so ingestion spans are actually stored.

Before this, ``ragfabric_core.telemetry.tracing.trace()`` only appended a span
when a ``TraceContext`` was active, and the only code that ever opened one was
the query route. The ingestion pipeline and the worker called ``trace(...)``
too, but with nothing listening, so every ingestion timing was silently
dropped. This table gives both callers somewhere to write their spans: one row
per ingestion phase (see ``IngestionRun``'s docstring for what "phase" means),
exposed at ``GET /api/runs/ingestion/{id}``.

``created_at`` is declared ``DateTime(timezone=True)`` from the start, rather
than added naive and fixed by migration 0006, since this table did not exist
before that fix.

Revision ID: 0005_ingestion_runs
Revises: 0004_pin_vector_dim_and_hnsw
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_ingestion_runs"
down_revision: str | None = "0004_pin_vector_dim_and_hnsw"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column("phase", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("trace", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("ingestion_runs", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_ingestion_runs_document_id"), ["document_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_ingestion_runs_created_at"), ["created_at"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("ingestion_runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_ingestion_runs_created_at"))
        batch_op.drop_index(batch_op.f("ix_ingestion_runs_document_id"))
    op.drop_table("ingestion_runs")
