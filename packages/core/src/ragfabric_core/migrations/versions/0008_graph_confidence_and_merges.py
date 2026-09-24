"""Graph confidence, extraction hashes, provenance link tables and merge records.

``entities`` and ``relationships`` gain ``confidence`` (nullable, [0, 1] when
present, enforced by a CHECK constraint) and ``extraction_model``. ADR 0004:
confidence is nullable on purpose, because a row written before this
revision, or one an extractor never scored, has no confidence, and ``None``
says "not measured" where ``0.0`` would say "measured as worthless".
``chunks`` gains ``extraction_hash``, the hash of the chunk text as of its
last graph extraction, so a later extraction pass can skip unchanged chunks.

``entity_sources`` and ``relationship_sources`` replace the JSON
``source_chunk_ids`` columns on ``entities`` and ``relationships`` as the
single source of truth for provenance: composite primary key on
(entity_id, chunk_id) / (relationship_id, chunk_id), foreign keys to their
owning row and to ``chunks`` both ON DELETE CASCADE, indexed on chunk_id so
the access filter can join from a visible chunk set. The JSON columns are
dropped rather than kept alongside, because nothing has read or written them
yet and two provenance copies would only ever drift; downgrade restores them
as empty JSON lists.

``entity_merges`` records every entity resolution decision: the surviving
entity, the merged-away entity's name, type, aliases and source chunk ids
(kept so an unmerge can restore it), the evidence that justified the merge,
and the method and model that made the call.

Revision ID: 0008_graph_confidence_and_merges
Revises: 0007_bm25_term_stats
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_graph_confidence_and_merges"
down_revision: str | None = "0007_bm25_term_stats"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CONFIDENCE_RANGE = "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)"


def upgrade() -> None:
    with op.batch_alter_table("entities", schema=None) as batch_op:
        batch_op.add_column(sa.Column("confidence", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("extraction_model", sa.String(), nullable=True))
        batch_op.drop_column("source_chunk_ids")
        batch_op.create_check_constraint("ck_entity_confidence_range", _CONFIDENCE_RANGE)

    with op.batch_alter_table("relationships", schema=None) as batch_op:
        batch_op.add_column(sa.Column("confidence", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("extraction_model", sa.String(), nullable=True))
        batch_op.drop_column("source_chunk_ids")
        batch_op.create_check_constraint("ck_relationship_confidence_range", _CONFIDENCE_RANGE)

    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("extraction_hash", sa.String(), nullable=True))

    op.create_table(
        "entity_sources",
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["entity_id"], ["entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("entity_id", "chunk_id"),
    )
    op.create_index("ix_entity_sources_chunk_id", "entity_sources", ["chunk_id"])

    op.create_table(
        "relationship_sources",
        sa.Column("relationship_id", sa.Integer(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["relationship_id"], ["relationships.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("relationship_id", "chunk_id"),
    )
    op.create_index("ix_relationship_sources_chunk_id", "relationship_sources", ["chunk_id"])

    op.create_table(
        "entity_merges",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("surviving_entity_id", sa.Integer(), nullable=False),
        sa.Column("merged_name", sa.String(), nullable=False),
        sa.Column("merged_entity_type", sa.String(), nullable=False),
        sa.Column("merged_aliases", sa.JSON(), nullable=False),
        sa.Column("merged_source_chunk_ids", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("method", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["surviving_entity_id"], ["entities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_entity_merges_surviving_entity_id", "entity_merges", ["surviving_entity_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_entity_merges_surviving_entity_id", table_name="entity_merges")
    op.drop_table("entity_merges")

    op.drop_index("ix_relationship_sources_chunk_id", table_name="relationship_sources")
    op.drop_table("relationship_sources")

    op.drop_index("ix_entity_sources_chunk_id", table_name="entity_sources")
    op.drop_table("entity_sources")

    with op.batch_alter_table("chunks", schema=None) as batch_op:
        batch_op.drop_column("extraction_hash")

    with op.batch_alter_table("relationships", schema=None) as batch_op:
        batch_op.drop_constraint("ck_relationship_confidence_range", type_="check")
        batch_op.drop_column("extraction_model")
        batch_op.drop_column("confidence")
        batch_op.add_column(
            sa.Column("source_chunk_ids", sa.JSON(), server_default="[]", nullable=False)
        )

    with op.batch_alter_table("entities", schema=None) as batch_op:
        batch_op.drop_constraint("ck_entity_confidence_range", type_="check")
        batch_op.drop_column("extraction_model")
        batch_op.drop_column("confidence")
        batch_op.add_column(
            sa.Column("source_chunk_ids", sa.JSON(), server_default="[]", nullable=False)
        )
