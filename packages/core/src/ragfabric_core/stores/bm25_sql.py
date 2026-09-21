"""BM25 ranking computed next to the data.

    score(D, Q) = sum over t in Q of  IDF(t) * ( tf(t,D) * (k1 + 1) )
                                     / ( tf(t,D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

The ``+ 1`` inside the logarithm is the Lucene variant, and it is not
cosmetic. Robertson's original IDF goes negative for a term present in more
than half the corpus, which lets a chunk containing a common query term score
below a chunk containing none of them. The ``+ 1`` floors IDF at zero, so a
common term contributes nothing rather than actively subtracting.

Why in SQL rather than in Python. ``rank_bm25`` is real BM25 but holds the
whole corpus in process memory: every API worker keeps its own copy, the
copies drift apart as documents change, and the ceiling is RAM rather than
disk. Ranking therefore runs where the data is. See stores/bm25_memory.py for
the in-process alternative and the cap that keeps it honest.

Why not pg_search (ParadeDB), which would give native BM25 for free: pgvector
is available on essentially every managed PostgreSQL and pg_search is not.
Depending on it would force self-hosters onto a custom PostgreSQL image and
lock managed-PostgreSQL users out entirely.

Where the three quantities come from. tf(t, D) is read out of the tsvector
chunk_search already stores, so no row-per-term-per-chunk table is needed; at
a million chunks that table would run to hundreds of millions of rows.
df(t) comes from term_stats, N and avgdl from corpus_stats, all maintained by
stores/term_stats.py on every index and delete.

Rows with doc_len = 0 are excluded rather than scored. Those are rows written
before Phase 4, and a zero length would zero the ``b * |D| / avgdl`` term,
shrinking the denominator and ranking legacy rows above everything else.
Excluding them is visible in the results; scoring them would be silently
wrong. Bring them forward with `ragfabric reindex --lexical-only`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.models.index import ChunkSearch, TermStat
from ragfabric_core.stores.access_sql import access_clause
from ragfabric_core.stores.postgres_fts import (
    PostgresLexicalStore,
    _to_chunk,
    decode_tsv,
    token_list,
)
from ragfabric_core.stores.term_stats import corpus_stats
from ragfabric_core.strategies.base import RetrievedChunk

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

# The filter keys this store honours. Anything else raises: a typo'd key that
# is silently dropped returns an unfiltered result set that looks perfectly
# plausible, and nothing in the response says the filter did not happen. That
# is the same reasoning config_file.py gives for forbidding unknown keys.
#
# postgres_fts.search still ignores unknown keys, and is deliberately left
# that way here. metadata_filters is an open user-supplied dict on
# StrategyParams, and the vector path (pgvector_store) accepts unknown keys
# too, so tightening exactly one of the three stores would mean the same API
# request succeeded or failed depending on which strategy served it. This
# store is new, so it can be strict from the start without breaking anyone;
# making the other two strict is a cross-cutting behaviour change that should
# be made deliberately and all at once, not smuggled in here.
SUPPORTED_FILTERS = ("document_id", "collection_id", "format")


def bm25_score(
    tf: float,
    df: int,
    n: int,
    doc_len: float,
    avgdl: float,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> float:
    """One term's BM25 contribution to one chunk. No database, no state.

    Kept pure so it can be checked against the formula above by eye, and so
    the SQL store and the in-process store can be held to the same numbers.
    """
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
    denominator = tf + k1 * (1.0 - b + b * (doc_len / avgdl if avgdl else 1.0))
    if denominator == 0:
        return 0.0
    return idf * (tf * (k1 + 1.0)) / denominator


class Bm25Store:
    """LexicalStore that writes like postgres_fts and ranks with BM25.

    Indexing is delegated rather than reimplemented: one writer for
    chunk_search means one place that keeps term_stats and corpus_stats
    correct, and two writers would be two chances to drift.
    """

    name = "bm25"

    def __init__(
        self,
        session_factory: Callable[[], Session],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self._sf = session_factory
        self._k1 = k1
        self._b = b
        self._writer = PostgresLexicalStore(session_factory)

    def index(self, chunk_ids: list[int], texts: list[str], payloads: list[dict]) -> None:
        self._writer.index(chunk_ids, texts, payloads)

    def delete_document(self, document_id: int) -> None:
        self._writer.delete_document(document_id)

    def search(
        self, query: str, top_k: int, access: AccessFilter, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        # Validated before anything else, including the empty-corpus shortcut
        # below: a caller mistake must not be reported only when the corpus
        # happens to be non-empty and the query happens to have terms.
        self._check_filters(filters)
        with self._sf() as db:
            n, avgdl = corpus_stats(db)
            if n == 0:
                return []
            postgres = db.bind.dialect.name == "postgresql"
            terms = self._query_terms(db, query, postgres)
            if not terms:
                return []
            if postgres:
                return self._search_postgres(db, terms, top_k, access, filters, n, avgdl)
            return self._search_portable(db, terms, top_k, access, filters, n, avgdl)

    def _query_terms(self, db: Session, query: str, postgres: bool) -> list[str]:
        """The query's terms in the same vocabulary the index was written in.

        On PostgreSQL that means asking to_tsvector, because the stored
        lexemes are stemmed and stop-word-stripped by the database. Comparing
        raw words against stemmed lexemes would silently match nothing.
        """
        if postgres:
            rows = db.execute(
                text("SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', :q))"),
                {"q": query},
            )
            return [row[0] for row in rows]
        return sorted(set(token_list(query)))

    @staticmethod
    def _check_filters(filters: dict | None) -> None:
        unknown = sorted(set(filters or {}) - set(SUPPORTED_FILTERS))
        if unknown:
            raise ValueError(
                f"unknown filter {unknown} for the bm25 store; "
                f"supported keys are {list(SUPPORTED_FILTERS)}"
            )

    def _apply(self, stmt, access: AccessFilter, filters: dict | None):
        clause = access_clause(access, ChunkSearch.document_id, ChunkSearch.collection_id)
        if clause is not None:
            stmt = stmt.where(clause)
        for key, value in (filters or {}).items():
            if key == "document_id":
                stmt = stmt.where(ChunkSearch.document_id == value)
            elif key == "collection_id":
                stmt = stmt.where(ChunkSearch.collection_id == value)
            elif key == "format":
                stmt = stmt.where(Document.format == value)
        return stmt

    def _search_postgres(
        self,
        db: Session,
        terms: list[str],
        top_k: int,
        access: AccessFilter,
        filters: dict | None,
        n: int,
        avgdl: float,
    ) -> list[RetrievedChunk]:
        # Read the formula in the module docstring alongside this. The lateral
        # unnest turns each chunk's tsvector into one row per lexeme carrying
        # its positions, and array_length(positions, 1) is tf.
        lexemes = (
            func.unnest(ChunkSearch.tsv)
            .table_valued("lexeme", "positions", "weights")
            .render_derived(name="lex", with_types=False)
            .lateral()
        )
        tf = func.coalesce(func.array_length(lexemes.c.positions, 1), 1)
        idf = func.ln((n - TermStat.df + 0.5) / (TermStat.df + 0.5) + 1.0)
        denominator = tf + self._k1 * (1.0 - self._b + self._b * ChunkSearch.doc_len / avgdl)
        score = func.sum(idf * (tf * (self._k1 + 1.0)) / denominator).label("score")

        ranking = (
            select(ChunkSearch.chunk_id, score)
            .select_from(ChunkSearch)
            .join(Document, Document.id == ChunkSearch.document_id)
            .join(lexemes, text("true"))
            .join(TermStat, TermStat.term == lexemes.c.lexeme)
            .where(lexemes.c.lexeme.in_(terms))
            .where(ChunkSearch.doc_len > 0)
        )
        ranking = self._apply(ranking, access, filters)
        ranked = db.execute(
            ranking.group_by(ChunkSearch.chunk_id)
            .order_by(score.desc(), ChunkSearch.chunk_id)
            .limit(top_k)
        ).all()
        return self._materialise(db, [(int(r[0]), float(r[1])) for r in ranked])

    def _search_portable(
        self,
        db: Session,
        terms: list[str],
        top_k: int,
        access: AccessFilter,
        filters: dict | None,
        n: int,
        avgdl: float,
    ) -> list[RetrievedChunk]:
        """Scoring in Python for dialects with no tsvector, SQLite above all.

        The access filter and the metadata filters still run inside the SQL
        query (ADR 0003). Only the arithmetic moves, and it is the same
        bm25_score the PostgreSQL branch's SQL spells out.
        """
        df_by_term = {
            term: int(df)
            for term, df in db.execute(
                select(TermStat.term, TermStat.df).where(TermStat.term.in_(terms))
            )
        }
        if not df_by_term:
            return []
        rows = (
            select(
                Chunk,
                Document.filename,
                Document.format,
                ChunkSearch.tsv,
                ChunkSearch.doc_len,
            )
            .join(ChunkSearch, ChunkSearch.chunk_id == Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .where(ChunkSearch.doc_len > 0)
        )
        rows = self._apply(rows, access, filters)

        scored: list[tuple[float, Chunk, str, str]] = []
        for chunk, filename, fmt, tsv, doc_len in db.execute(rows).all():
            counts = decode_tsv(tsv)
            total = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                df = df_by_term.get(term)
                # A term with no term_stats row contributes nothing, matching
                # the inner join the PostgreSQL branch makes against it.
                if tf and df is not None:
                    total += bm25_score(
                        tf=tf,
                        df=df,
                        n=n,
                        doc_len=doc_len,
                        avgdl=avgdl,
                        k1=self._k1,
                        b=self._b,
                    )
            if total > 0:
                scored.append((total, chunk, filename, fmt))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [_to_chunk(c, f, m, s) for s, c, f, m in scored[:top_k]]

    def access_stats(self, filters: dict, access: AccessFilter) -> tuple[int, int]:
        """Candidate counts before and after the access filter, over chunk_search.

        Mirrors PgVectorStore.access_stats so the audit row's
        ``sources_filtered`` is a measured number on the vectorless path too,
        not a zero standing in for "we did not look" (ADR 0004). Counted over
        the same universe a real query sees, which excludes doc_len = 0 rows
        because BM25 excludes them.
        """
        self._check_filters({k: v for k, v in (filters or {}).items() if v is not None})
        with self._sf() as db:
            base = (
                select(ChunkSearch.chunk_id, ChunkSearch.document_id, ChunkSearch.collection_id)
                .join(Document, Document.id == ChunkSearch.document_id)
                .where(ChunkSearch.doc_len > 0)
            )
            for key, value in (filters or {}).items():
                if value is None:
                    continue
                if key == "document_id":
                    base = base.where(ChunkSearch.document_id == value)
                elif key == "collection_id":
                    base = base.where(ChunkSearch.collection_id == value)
                elif key == "format":
                    base = base.where(Document.format == value)
            subquery = base.subquery()
            clause = access_clause(access, subquery.c.document_id, subquery.c.collection_id)
            if clause is None:
                total = int(db.execute(select(func.count()).select_from(subquery)).scalar() or 0)
                return total, total
            after_flag = case((clause, 1), else_=0)
            before, after = db.execute(
                select(func.count(), func.coalesce(func.sum(after_flag), 0)).select_from(subquery)
            ).one()
            return int(before), int(after)

    def _materialise(self, db: Session, ranked: list[tuple[int, float]]) -> list[RetrievedChunk]:
        """Turn (chunk_id, score) pairs into the one result shape (ADR 0002)."""
        if not ranked:
            return []
        ids = [chunk_id for chunk_id, _ in ranked]
        loaded = {
            chunk.id: (chunk, filename, fmt)
            for chunk, filename, fmt in db.execute(
                select(Chunk, Document.filename, Document.format)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id.in_(ids))
            ).all()
        }
        out = []
        for chunk_id, score in ranked:
            found = loaded.get(chunk_id)
            if found is not None:
                out.append(_to_chunk(found[0], found[1], found[2], score))
        return out
