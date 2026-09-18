"""VectorStore over the chunk_embeddings table.

On PostgreSQL the ranking is pgvector's cosine distance operator (`<=>`), with
the access predicate in the same WHERE clause, so a restricted caller never
has a forbidden row ranked. On SQLite (tests, development without Postgres)
the same table holds JSON vectors and the ranking is a NumPy dot product over
the permitted rows; behaviour is identical, only speed differs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import delete, func, select, type_coerce
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.embeddings.normalise import normalise
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.models.index import ChunkEmbedding
from ragfabric_core.stores.access_sql import access_clause
from ragfabric_core.strategies.base import RetrievedChunk


def _to_chunk(row, score: float) -> RetrievedChunk:
    chunk, filename, fmt = row
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


class PgVectorStore:
    name = "pgvector"

    def __init__(self, session_factory: Callable[[], Session], model: str | None = None) -> None:
        self._sf = session_factory
        self._model = model

    @property
    def model(self) -> str | None:
        return self._model

    def upsert(
        self, chunk_ids: list[int], vectors: list[list[float]], payloads: list[dict]
    ) -> None:
        with self._sf() as db:
            for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads, strict=True):
                row = db.get(ChunkEmbedding, chunk_id)
                values = dict(
                    document_id=payload["document_id"],
                    collection_id=payload.get("collection_id"),
                    model=payload["model"],
                    dim=payload.get("dim", len(vector)),
                    embedding=normalise(vector),
                )
                if row is None:
                    db.add(ChunkEmbedding(chunk_id=chunk_id, **values))
                else:
                    for k, v in values.items():
                        setattr(row, k, v)
            db.commit()

    def query(
        self, vector: list[float], top_k: int, access: AccessFilter, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        clause = access_clause(access, ChunkEmbedding.document_id, ChunkEmbedding.collection_id)
        vector = normalise(vector)
        with self._sf() as db:
            base = (
                select(Chunk, Document.filename, Document.format, ChunkEmbedding.embedding)
                .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
                .join(Document, Document.id == Chunk.document_id)
            )
            if clause is not None:
                base = base.where(clause)
            if self._model is not None:
                base = base.where(ChunkEmbedding.model == self._model)
            for key, value in (filters or {}).items():
                if key == "document_id":
                    base = base.where(ChunkEmbedding.document_id == value)
                elif key == "collection_id":
                    base = base.where(ChunkEmbedding.collection_id == value)
                elif key == "format":
                    base = base.where(Document.format == value)
            if db.bind.dialect.name == "postgresql":
                # The column type is a dialect variant (JSON base, Vector on
                # postgresql); the ORM attribute's comparator resolves off the
                # base type, so cosine_distance() is only reachable by
                # coercing the expression to Vector for this query.
                distance = type_coerce(ChunkEmbedding.embedding, Vector()).cosine_distance(
                    list(map(float, vector))
                )
                rows = db.execute(
                    base.add_columns(distance.label("distance"))
                    .order_by(distance, Chunk.id)
                    .limit(top_k)
                ).all()
                return [_to_chunk(r[:3], 1.0 - float(r[4])) for r in rows]
            rows = db.execute(base).all()
            if not rows:
                return []
            q = np.asarray(vector, dtype=np.float32)
            scored = []
            for r in rows:
                emb = np.asarray(r[3], dtype=np.float32)
                score = float(emb @ q) if emb.shape == q.shape else -1.0
                scored.append((score, r))
            scored.sort(key=lambda item: (-item[0], item[1][0].id))
            return [_to_chunk(r[:3], s) for s, r in scored[:top_k]]

    def delete_document(self, document_id: int) -> None:
        with self._sf() as db:
            db.execute(delete(ChunkEmbedding).where(ChunkEmbedding.document_id == document_id))
            db.commit()

    def count(self) -> int:
        with self._sf() as db:
            return int(db.execute(select(func.count()).select_from(ChunkEmbedding)).scalar() or 0)
