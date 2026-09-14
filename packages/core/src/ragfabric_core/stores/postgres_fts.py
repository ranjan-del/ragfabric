"""LexicalStore over the chunk_search table.

PostgreSQL: to_tsvector('english') at index time, plainto_tsquery at query
time, ts_rank_cd for ordering, a GIN index for speed, the access predicate in
the WHERE clause. Other dialects: a token overlap score computed in Python
over the permitted rows, so unit tests and SQLite development keep working.
Phase 4 adds BM25 and fusion on top; this store is the fan out target.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.models.index import ChunkSearch
from ragfabric_core.stores.access_sql import access_clause
from ragfabric_core.strategies.base import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


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
            for chunk_id, text, payload in zip(chunk_ids, texts, payloads, strict=True):
                tsv = (
                    func.to_tsvector("english", text)
                    if postgres
                    else " ".join(sorted(_tokens(text)))
                )
                row = db.get(ChunkSearch, chunk_id)
                if row is None:
                    db.add(
                        ChunkSearch(
                            chunk_id=chunk_id,
                            document_id=payload["document_id"],
                            collection_id=payload.get("collection_id"),
                            tsv=tsv,
                        )
                    )
                else:
                    row.document_id, row.collection_id, row.tsv = (
                        payload["document_id"],
                        payload.get("collection_id"),
                        tsv,
                    )
            db.commit()

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
            db.execute(delete(ChunkSearch).where(ChunkSearch.document_id == document_id))
            db.commit()
