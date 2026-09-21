"""LexicalStore over the chunk_search table.

PostgreSQL: to_tsvector('english') at index time, plainto_tsquery at query
time, ts_rank_cd for ordering, a GIN index for speed, the access predicate in
the WHERE clause. Other dialects: a token overlap score computed in Python
over the permitted rows, so unit tests and SQLite development keep working.
Phase 4 adds BM25 and fusion on top; this store is the fan out target.

Phase 4 changes two things about what gets written here.

Term frequency now survives on every dialect. The non-PostgreSQL branch used
to store ``" ".join(sorted(set(tokens)))``, which destroys repetition: a term
occurring nine times became indistinguishable from one occurring once. BM25
without tf is not BM25, and since most of the suite runs on SQLite, leaving it
that way would have meant the phase's central algorithm was only ever
exercised behind a skipped guard. The tokens are still sorted, so the column
stays deterministic, but duplicates are kept.

chunk_search.doc_len is populated on both branches, and every write keeps
term_stats and corpus_stats in step through the Task 2 hooks. Rows written
before this change carry doc_len = 0 and a set based tsv; they are excluded
from BM25 rather than scored, and `ragfabric reindex --lexical-only` is the
supported way to bring them forward.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

from sqlalchemy import String, bindparam, delete, func, select, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.models.index import ChunkSearch
from ragfabric_core.stores.access_sql import access_clause
from ragfabric_core.stores.term_stats import record_indexed, record_removed
from ragfabric_core.strategies.base import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def token_list(text_value: str) -> list[str]:
    """Every token in order, repeats included. The off-PostgreSQL tf source."""
    return _TOKEN.findall(text_value.lower())


def _tokens(text_value: str) -> set[str]:
    return set(token_list(text_value))


def term_counts(text_value: str) -> dict[str, int]:
    """Term to occurrence count, the off-PostgreSQL equivalent of a tsvector."""
    return dict(Counter(token_list(text_value)))


def encode_tsv(text_value: str) -> str:
    """The off-PostgreSQL chunk_search.tsv payload: sorted tokens, repeats kept."""
    return " ".join(sorted(token_list(text_value)))


def decode_tsv(stored: str) -> dict[str, int]:
    """Read term counts back out of an off-PostgreSQL tsv column."""
    return dict(Counter(str(stored).split()))


# One round trip for a whole batch rather than one per chunk. WITH ORDINALITY
# is what ties each lexeme back to the text it came from, since unnest alone
# gives no way to know which input row produced a given output row.
_PG_TERMS_FOR_TEXTS = text(
    """
    SELECT t.ord - 1 AS idx, u.lexeme, coalesce(array_length(u.positions, 1), 1) AS tf
    FROM unnest(:texts) WITH ORDINALITY AS t(txt, ord),
         LATERAL unnest(to_tsvector('english', t.txt)) AS u(lexeme, positions, weights)
    """
).bindparams(bindparam("texts", type_=ARRAY(String)))


def _pg_terms_for_texts(db: Session, texts: list[str]) -> list[dict[str, int]]:
    """Ask PostgreSQL what to_tsvector('english') makes of each text.

    The lexemes have to come from PostgreSQL itself: the english stemmer and
    stop word list are the database's, and guessing at them in Python would
    put term_stats out of step with the tsvector actually stored.
    """
    per_text: list[dict[str, int]] = [{} for _ in texts]
    if not texts:
        return per_text
    for idx, lexeme, tf in db.execute(_PG_TERMS_FOR_TEXTS, {"texts": list(texts)}):
        per_text[int(idx)][lexeme] = int(tf)
    return per_text


def _pg_terms_for_rows(db: Session, chunk_ids: list[int]) -> dict[int, dict[str, int]]:
    """Recover the term counts already stored for these chunks, to undo them."""
    out: dict[int, dict[str, int]] = {}
    if not chunk_ids:
        return out
    rows = db.execute(
        text(
            "SELECT cs.chunk_id, u.lexeme, coalesce(array_length(u.positions, 1), 1) AS tf "
            "FROM chunk_search cs, LATERAL unnest(cs.tsv) "
            "AS u(lexeme, positions, weights) WHERE cs.chunk_id IN :ids"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": chunk_ids},
    )
    for chunk_id, lexeme, tf in rows:
        out.setdefault(int(chunk_id), {})[lexeme] = int(tf)
    return out


def _to_chunk(chunk: Chunk, filename: str, fmt: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        collection_id=chunk.collection_id,
        text=chunk.text,
        page=chunk.page,
        section=chunk.section,
        score=score,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        metadata={"filename": filename, "format": fmt},
    )


class PostgresLexicalStore:
    name = "postgres_fts"

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._sf = session_factory

    def index(self, chunk_ids: list[int], texts: list[str], payloads: list[dict]) -> None:
        with self._sf() as db:
            postgres = db.bind.dialect.name == "postgresql"
            new_terms = (
                _pg_terms_for_texts(db, list(texts))
                if postgres
                else [term_counts(t) for t in texts]
            )

            # Read and undo what these chunks previously contributed BEFORE any
            # row is rewritten. Re-indexing a document without this leaves df
            # permanently inflated, and nothing raises to say so.
            existing = {
                row.chunk_id: row
                for row in db.execute(
                    select(ChunkSearch).where(ChunkSearch.chunk_id.in_(chunk_ids))
                ).scalars()
            }
            old_terms = self._terms_for_rows(db, list(existing.values()), postgres)
            if old_terms:
                record_removed(db, old_terms)

            for chunk_id, text_value, payload, terms in zip(
                chunk_ids, texts, payloads, new_terms, strict=True
            ):
                tsv = (
                    func.to_tsvector("english", text_value) if postgres else encode_tsv(text_value)
                )
                doc_len = sum(terms.values())
                row = existing.get(chunk_id)
                if row is None:
                    db.add(
                        ChunkSearch(
                            chunk_id=chunk_id,
                            document_id=payload["document_id"],
                            collection_id=payload.get("collection_id"),
                            tsv=tsv,
                            doc_len=doc_len,
                        )
                    )
                else:
                    row.document_id, row.collection_id, row.tsv, row.doc_len = (
                        payload["document_id"],
                        payload.get("collection_id"),
                        tsv,
                        doc_len,
                    )
            record_indexed(db, list(new_terms))
            db.commit()

    def _terms_for_rows(
        self, db: Session, rows: list[ChunkSearch], postgres: bool
    ) -> list[dict[str, int]]:
        """What each stored row currently contributes to term_stats.

        Rows written before Phase 4 carry doc_len = 0 and a set based tsv, so
        they were never counted in and must not be counted out.
        """
        live = [row for row in rows if (row.doc_len or 0) > 0]
        if not live:
            return []
        if postgres:
            by_id = _pg_terms_for_rows(db, [row.chunk_id for row in live])
            return [by_id.get(row.chunk_id, {}) for row in live]
        return [decode_tsv(row.tsv) for row in live]

    def search(
        self, query: str, top_k: int, access: AccessFilter, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        clause = access_clause(access, ChunkSearch.document_id, ChunkSearch.collection_id)
        with self._sf() as db:
            base = (
                select(Chunk, Document.filename, Document.format, ChunkSearch.tsv)
                .join(ChunkSearch, ChunkSearch.chunk_id == Chunk.id)
                .join(Document, Document.id == Chunk.document_id)
            )
            if clause is not None:
                base = base.where(clause)
            for key, value in (filters or {}).items():
                if key == "document_id":
                    base = base.where(ChunkSearch.document_id == value)
                elif key == "collection_id":
                    base = base.where(ChunkSearch.collection_id == value)
                elif key == "format":
                    base = base.where(Document.format == value)
            if db.bind.dialect.name == "postgresql":
                tsq = func.plainto_tsquery("english", query)
                rank = func.ts_rank_cd(ChunkSearch.tsv, tsq)
                rows = db.execute(
                    base.add_columns(rank.label("rank"))
                    .where(ChunkSearch.tsv.op("@@")(tsq))
                    .order_by(rank.desc(), Chunk.id)
                    .limit(top_k)
                ).all()
                return [_to_chunk(r[0], r[1], r[2], float(r[4])) for r in rows]
            q = _tokens(query)
            if not q:
                return []
            scored = []
            for chunk, filename, fmt, tsv in db.execute(base).all():
                overlap = len(q & set(str(tsv).split()))
                if overlap:
                    scored.append((overlap / len(q), chunk, filename, fmt))
            scored.sort(key=lambda item: (-item[0], item[1].id))
            return [_to_chunk(c, f, m, s) for s, c, f, m in scored[:top_k]]

    def delete_document(self, document_id: int) -> None:
        with self._sf() as db:
            postgres = db.bind.dialect.name == "postgresql"
            rows = list(
                db.execute(
                    select(ChunkSearch).where(ChunkSearch.document_id == document_id)
                ).scalars()
            )
            gone = self._terms_for_rows(db, rows, postgres)
            if gone:
                record_removed(db, gone)
            db.execute(delete(ChunkSearch).where(ChunkSearch.document_id == document_id))
            db.commit()
