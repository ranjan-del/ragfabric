"""Make every timestamp column timezone aware.

Every ``DateTime`` column up to this revision was declared naive even though
``models/base.py``'s ``utcnow()`` default has always produced an offset-aware
UTC value, and the module docstring claimed "all rows use timezone-aware
UTC". This revision closes that gap by declaring every timestamp column
``DateTime(timezone=True)``, matching the model changes in the same task.

What this actually buys differs by dialect (see ``models/base.py`` for the
full explanation, verified empirically against both dialects): PostgreSQL
stores these as real ``TIMESTAMPTZ`` columns, so a value read back is
offset-aware and comparisons are correct regardless of the writer's local
timezone. SQLite has no native timezone-aware datetime type; the type
change is accepted but SQLite continues to store and return naive values,
so on SQLite this revision changes the declared schema (which is what
``test_migrations.py``'s drift check compares against the models) without
changing SQLite's actual runtime behaviour.

``ingestion_runs`` (migration 0005) already declared its ``created_at``
timezone aware and is not touched here.

Revision ID: 0006_timezone_aware_timestamps
Revises: 0005_ingestion_runs
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_timezone_aware_timestamps"
down_revision: str | None = "0005_ingestion_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, column, nullable) triples whose DateTime column becomes
# DateTime(timezone=True). nullable is carried through explicitly on every
# alter_column call so batch mode's table-recreate on SQLite reproduces the
# exact NOT NULL constraint each column already has, rather than guessing.
_TIMESTAMP_COLUMNS: list[tuple[str, str, bool]] = [
    ("users", "created_at", False),
    ("collections", "created_at", False),
    ("documents", "created_at", False),
    ("query_logs", "created_at", False),
    ("entities", "created_at", False),
    ("relationships", "created_at", False),
    ("evaluation_runs", "started_at", False),
    ("evaluation_runs", "finished_at", True),
    ("groups", "created_at", False),
    ("group_members", "created_at", False),
    ("collection_grants", "created_at", False),
    ("document_overrides", "created_at", False),
    ("api_keys", "created_at", False),
    ("api_keys", "last_used_at", True),
    ("api_keys", "expires_at", True),
    ("audit_log", "created_at", False),
    ("conversations", "created_at", False),
    ("messages", "created_at", False),
    ("retrieval_runs", "created_at", False),
]


def upgrade() -> None:
    for table, column, nullable in _TIMESTAMP_COLUMNS:
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.alter_column(
                column,
                existing_type=sa.DateTime(),
                type_=sa.DateTime(timezone=True),
                existing_nullable=nullable,
            )


def downgrade() -> None:
    for table, column, nullable in reversed(_TIMESTAMP_COLUMNS):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.alter_column(
                column,
                existing_type=sa.DateTime(timezone=True),
                type_=sa.DateTime(),
                existing_nullable=nullable,
            )
