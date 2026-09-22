"""Read one document's chunks in order, with the access predicate in the query.

Every other read path in the system is a search: a query goes in, a ranked list
comes out, and the access predicate is part of the SQL that ranks (ADR 0003).
The agent's ``fetch_document`` tool is the one path that does not search. It
names a document and asks for all of it, which is exactly the shape of request
that could quietly become a way to read what the principal is not allowed to
find.

So this reader is written the same way the stores are: the predicate is built
by ``access_clause`` and goes into the WHERE clause. A denied document returns
no rows from the database, rather than returning rows that are then dropped in
Python. The difference matters, because a post-filter is one refactor away from
being forgotten, and a WHERE clause is not.

Ordering is by ``chunk_index``, the document's own reading order, so the caller
gets the policy as it was written rather than as it happened to be stored.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.stores.access_sql import access_clause
from ragfabric_core.strategies.base import RetrievedChunk


class SqlDocumentChunkReader:
    """Whole-document reads over the ``chunks`` table."""

    name = "sql_document_chunks"

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._sf = session_factory

    def chunks_for_document(self, document_id: int, access: AccessFilter) -> list[RetrievedChunk]:
        clause = access_clause(access, Chunk.document_id, Chunk.collection_id)
        with self._sf() as db:
            statement = (
                select(Chunk, Document.filename, Document.format)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.document_id == document_id)
            )
            if clause is not None:
                statement = statement.where(clause)
            rows = db.execute(statement.order_by(Chunk.chunk_index, Chunk.id)).all()
            return [
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    collection_id=chunk.collection_id,
                    text=chunk.text,
                    page=chunk.page,
                    section=chunk.section,
                    # No score: nothing was ranked. A number here would be a
                    # fabricated relevance, which ADR 0004 forbids.
                    score=None,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    metadata={"filename": filename, "format": fmt},
                )
                for chunk, filename, fmt in rows
            ]
