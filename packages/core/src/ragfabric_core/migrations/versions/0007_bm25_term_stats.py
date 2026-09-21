"""Term statistics schema for BM25 lexical retrieval.

BM25 needs three quantities the schema does not hold before this revision:
df(t), the number of chunks containing term t (term_stats.df); N and the
running total behind avgdl, the corpus-wide chunk count and term count
(corpus_stats.n_chunks and corpus_stats.sum_len); and |D|, the length of a
single chunk in terms (chunk_search.doc_len). avgdl is derived from
sum_len / n_chunks rather than stored as a precomputed average, because a
mean cannot be updated incrementally without drifting, while a running sum
stays exact.

corpus_stats holds exactly one row, id=1, updated in place as chunks are
indexed and removed.

Revision ID: 0007_bm25_term_stats
Revises: 0006_timezone_aware_timestamps
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_bm25_term_stats"
down_revision: str | None = "0006_timezone_aware_timestamps"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "term_stats",
        sa.Column("term", sa.String(length=255), nullable=False),
        sa.Column("df", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("term"),
    )
    op.create_table(
        "corpus_stats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("n_chunks", sa.Integer(), nullable=False),
        sa.Column("sum_len", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("chunk_search", schema=None) as batch_op:
        batch_op.add_column(sa.Column("doc_len", sa.Integer(), server_default="0", nullable=False))
    op.execute(sa.text("INSERT INTO corpus_stats (id, n_chunks, sum_len) VALUES (1, 0, 0)"))


def downgrade() -> None:
    with op.batch_alter_table("chunk_search", schema=None) as batch_op:
        batch_op.drop_column("doc_len")
    op.drop_table("corpus_stats")
    op.drop_table("term_stats")
