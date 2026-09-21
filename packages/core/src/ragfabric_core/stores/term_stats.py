"""Maintain the corpus-wide statistics BM25 needs: df(t), N and avgdl.

Term frequency, tf(t, D), is derivable from what the lexical index already
stores per chunk, so it needs no table. Document frequency does: df(t) is the
number of chunks that contain t anywhere in the corpus, which no single row
can answer. That is what term_stats holds, and what this module keeps honest.

Four rules decide the shape of the code below.

df counts chunks, not occurrences. A term appearing nine times in one chunk
contributes exactly 1 to df. Counting occurrences would deflate IDF for terms
that happen to repeat, which is the opposite of what IDF is for.

Re-indexing must decrement before it increments. Editing a document and
re-indexing it without removing the old terms first would raise df forever,
with no error and no way to notice short of recomputing the whole corpus.
Callers therefore pair record_removed with record_indexed; this module makes
the decrement exact so that pairing is enough.

Every update is arithmetic in SQL, never read-modify-write in Python. Two
ingests running at once would otherwise both read df = 5, both write 6, and
lose one increment. ``df = df + :delta`` cannot lose an update because the
database serialises it.

n_chunks and sum_len move in the same transaction as df. These functions do
not commit. They join whatever transaction the caller opened, so a crash
between the term update and the corpus update is impossible: either both
land or neither does. A corpus whose N disagrees with its df values computes
IDF against a corpus that never existed.

avgdl is derived from sum_len / n_chunks rather than stored. A running sum is
exact under repeated addition and subtraction; an incrementally updated mean
is not.
"""

from __future__ import annotations

from collections import Counter

from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session

from ragfabric_core.models.index import CorpusStat, TermStat

# term_stats.term is String(255): a B-tree key has to be bounded, and a lexeme
# longer than 255 characters is not a word. Truncating at write time keeps the
# decision here, where it is visible, instead of letting PostgreSQL reject the
# row mid-ingest.
MAX_TERM_LENGTH = 255

# corpus_stats holds exactly one row. Migration 0007 seeds it; a schema built
# straight from the metadata (unit tests) has no row until the first write,
# which _bump_corpus handles.
CORPUS_ROW_ID = 1


def _term_deltas(per_chunk_terms: list[dict[str, int]], sign: int) -> Counter[str]:
    """Collapse per-chunk term maps into one df delta per term.

    Each chunk contributes at most 1 per term however often the term occurs in
    it, which is the definition of document frequency.
    """
    deltas: Counter[str] = Counter()
    for terms in per_chunk_terms:
        for term in {t[:MAX_TERM_LENGTH] for t in terms if t}:
            deltas[term] += sign
    return deltas


def _apply_term_deltas(db: Session, deltas: Counter[str]) -> None:
    positive = {term: delta for term, delta in deltas.items() if delta > 0}
    negative = {term: delta for term, delta in deltas.items() if delta < 0}

    for term, delta in negative.items():
        db.execute(
            update(TermStat)
            .where(TermStat.term == term)
            .values(df=TermStat.df + delta)
            .execution_options(synchronize_session=False)
        )

    for term, delta in positive.items():
        _upsert_df(db, term, delta)

    if negative:
        # A term nothing contains any more is a dead row, and dead rows would
        # accumulate for the life of the deployment. df can only reach zero or
        # below through removals, so this runs only when something was removed.
        db.execute(
            delete(TermStat)
            .where(TermStat.term.in_(sorted(negative)))
            .where(TermStat.df <= 0)
            .execution_options(synchronize_session=False)
        )


def _upsert_df(db: Session, term: str, delta: int) -> None:
    """Insert df = delta, or add delta to the existing row, in one statement.

    PostgreSQL and SQLite both speak ON CONFLICT DO UPDATE, which is the only
    form that is atomic. Any other dialect falls back to UPDATE then INSERT,
    which is not, and that is stated rather than hidden.
    """
    dialect = db.bind.dialect.name if db.bind is not None else ""
    if dialect in ("postgresql", "sqlite"):
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as dialect_insert
        else:
            from sqlalchemy.dialects.sqlite import insert as dialect_insert

        stmt = dialect_insert(TermStat).values(term=term, df=delta)
        db.execute(
            stmt.on_conflict_do_update(
                index_elements=[TermStat.term],
                set_={"df": TermStat.__table__.c.df + delta},
            )
        )
        return

    updated = db.execute(
        update(TermStat)
        .where(TermStat.term == term)
        .values(df=TermStat.df + delta)
        .execution_options(synchronize_session=False)
    )
    if updated.rowcount == 0:
        db.execute(insert(TermStat).values(term=term, df=delta))


def _bump_corpus(db: Session, n_chunks: int, sum_len: int) -> None:
    if n_chunks == 0 and sum_len == 0:
        return
    updated = db.execute(
        update(CorpusStat)
        .where(CorpusStat.id == CORPUS_ROW_ID)
        .values(n_chunks=CorpusStat.n_chunks + n_chunks, sum_len=CorpusStat.sum_len + sum_len)
        .execution_options(synchronize_session=False)
    )
    if updated.rowcount == 0:
        db.execute(
            insert(CorpusStat).values(
                id=CORPUS_ROW_ID, n_chunks=max(0, n_chunks), sum_len=max(0, sum_len)
            )
        )


def _apply(db: Session, per_chunk_terms: list[dict[str, int]], sign: int) -> None:
    if not per_chunk_terms:
        return
    _apply_term_deltas(db, _term_deltas(per_chunk_terms, sign))
    occurrences = sum(sum(terms.values()) for terms in per_chunk_terms)
    _bump_corpus(db, sign * len(per_chunk_terms), sign * occurrences)


def record_indexed(db: Session, per_chunk_terms: list[dict[str, int]]) -> None:
    """Account for chunks entering the index.

    ``per_chunk_terms`` is one mapping of term to occurrence count per chunk.
    Does not commit: the caller's transaction decides when this becomes real,
    so the term counts and the corpus totals can never land separately.
    """
    _apply(db, per_chunk_terms, 1)


def record_removed(db: Session, per_chunk_terms: list[dict[str, int]]) -> None:
    """Account for chunks leaving the index, including re-indexed ones.

    Pass exactly what was passed to record_indexed for those chunks. Anything
    else leaves df permanently wrong, and nothing will raise to tell you.
    """
    _apply(db, per_chunk_terms, -1)


def corpus_stats(db: Session) -> tuple[int, float]:
    """Return (N, avgdl): the indexed chunk count and the mean chunk length.

    avgdl is 1.0 on an empty corpus. BM25's length-normalisation term divides
    by avgdl, so zero would be a division by zero, and 1.0 makes the term
    inert rather than wrong: with no chunks there is nothing to normalise.
    """
    row = db.execute(
        select(CorpusStat.n_chunks, CorpusStat.sum_len).where(CorpusStat.id == CORPUS_ROW_ID)
    ).first()
    if row is None:
        return 0, 1.0
    n_chunks, sum_len = int(row[0]), int(row[1])
    if n_chunks <= 0:
        return max(0, n_chunks), 1.0
    return n_chunks, sum_len / n_chunks
