"""In-memory cosine-similarity vector store (offline-first, default backend).

Chunk vectors are kept in one ``(n, dim)`` NumPy matrix and ranked with a single
matrix-vector product; because the embedder L2-normalises every vector, that dot
product IS the cosine similarity. This needs no server, no network, and nothing
beyond NumPy, which keeps the whole pipeline runnable and testable offline.

Vectors are also persisted to the database (``chunks.embedding``), so this
in-memory index is rebuilt from the DB on startup via ``rebuild_from_db`` and
kept in sync on ingest/delete. A managed backend (e.g. Qdrant) could implement
the same ``upsert`` / ``search`` / ``delete_document`` surface later without
touching the retriever.

The module exposes a process-wide singleton via ``get_store`` because a single
FastAPI process shares one index across requests.
"""

from __future__ import annotations

import numpy as np

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.config import settings

# Metadata keys a caller may filter on. Keeping this explicit means an unknown
# filter name is a loud error rather than a filter that silently matches nothing.
FILTERABLE = ("collection_id", "document_id", "format")


class InMemoryVectorStore:
    """Cosine-similarity store over an in-memory NumPy matrix."""

    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.embedding_dim
        self._vectors: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        # Parallel metadata list; index i describes row i of ``_vectors``.
        self._meta: list[dict] = []
        # chunk_id -> row index, so re-indexing a chunk replaces it instead of
        # appending a duplicate (see ``upsert``).
        self._row_by_chunk: dict[int, int] = {}

    def upsert(self, records: list[dict]) -> None:
        """Insert or replace records, keyed by ``chunk_id``.

        Expected keys per record: ``vector`` (list/ndarray), ``chunk_id``,
        ``document_id``, ``collection_id``, ``filename``, ``format``, ``page``,
        ``chunk_index``, ``text``.

        This is a real upsert, not an append. Re-ingesting a document (the
        versioning flow) or calling :meth:`rebuild_from_db` twice would
        otherwise leave two rows per chunk, which double-counts a source in the
        citation list and inflates the index size on every restart.
        """
        if not records:
            return
        vectors = np.vstack(
            [np.asarray(r["vector"], dtype=np.float32).reshape(-1) for r in records]
        )
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} does not match store dim {self.dim}")

        # Rows already materialised in ``_vectors``. Anything at or beyond this
        # index is still staged in ``new_rows`` and has no matrix row yet.
        base = self._vectors.shape[0]
        new_rows: list[np.ndarray] = []
        for record, vector in zip(records, vectors, strict=True):
            meta = {k: v for k, v in record.items() if k != "vector"}
            chunk_id = meta.get("chunk_id")
            existing = self._row_by_chunk.get(chunk_id) if chunk_id is not None else None
            if existing is not None:
                self._meta[existing] = meta
                if existing < base:
                    self._vectors[existing] = vector
                else:
                    # Same chunk_id twice inside ONE batch: the row it maps to is
                    # still pending, so patch the staged vector. Writing into
                    # ``_vectors`` here would index past the end of the matrix.
                    new_rows[existing - base] = vector
                continue
            if chunk_id is not None:
                # ``_meta`` is appended to on the next line and the staged rows
                # are vstacked in the same order, so the row this record will
                # occupy is exactly the current length of ``_meta``. Adding
                # ``len(new_rows)`` here would double-count every record after
                # the first in a batch and leave the lookup pointing at the
                # wrong row (or past the end of the matrix).
                self._row_by_chunk[chunk_id] = len(self._meta)
            self._meta.append(meta)
            new_rows.append(vector)

        if new_rows:
            self._vectors = np.vstack([self._vectors, np.vstack(new_rows)])

    def candidate_rows(self, filters: dict, access: AccessFilter | None = None) -> list[int]:
        """Row indices whose metadata matches every non-None filter value.

        Only keys in FILTERABLE are honoured. Restricting here rather than
        trusting the caller means an unexpected key cannot silently narrow the
        candidate set to nothing and make a document look missing.

        ``access``, when given, is applied on top of the metadata filters,
        before ranking (ADR 0003), so a caller's AccessFilter narrows the
        candidate set rather than the result set.
        """
        active = {k: v for k, v in filters.items() if v is not None and k in FILTERABLE}
        if not active:
            rows = list(range(len(self._meta)))
        else:
            rows = [
                i
                for i, meta in enumerate(self._meta)
                if all(meta.get(key) == value for key, value in active.items())
            ]
        if access is not None and not access.is_unrestricted:
            rows = [
                i
                for i in rows
                if access.allows(
                    self._meta[i].get("document_id"), self._meta[i].get("collection_id")
                )
            ]
        return rows

    def access_stats(self, filters: dict, access: AccessFilter | None) -> tuple[int, int]:
        """Candidate counts before and after applying ``access`` (for audit rows)."""
        before = self.candidate_rows(filters)
        after = self.candidate_rows(filters, access)
        return len(before), len(after)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        collection_id: int | None = None,
        document_id: int | None = None,
        format: str | None = None,
        access: AccessFilter | None = None,
    ) -> list[dict]:
        """Return the ``top_k`` most similar chunks, with their metadata.

        Optional ``collection_id`` / ``document_id`` / ``format`` apply a
        metadata filter before ranking (semantic search scoped to a collection,
        a single document, or one file type). ``access`` applies the caller's
        AccessFilter before ranking too (ADR 0003).
        """
        if len(self._meta) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self._vectors @ query  # cosine similarity (vectors are unit-norm)

        candidates = self.candidate_rows(
            {
                "collection_id": collection_id,
                "document_id": document_id,
                "format": format,
            },
            access,
        )
        if not candidates:
            return []

        # Sort by score, breaking ties on chunk_id so results are stable across
        # restarts (NumPy's ordering of equal scores is not guaranteed).
        candidates.sort(key=lambda i: (-float(scores[i]), self._meta[i].get("chunk_id") or 0))
        results: list[dict] = []
        for i in candidates[:top_k]:
            record = dict(self._meta[i])
            record["score"] = float(scores[i])
            results.append(record)
        return results

    def all_meta(self) -> list[dict]:
        """Return the parallel metadata list (read-only view for hybrid search)."""
        return self._meta

    def score_all(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of ``query_vector`` against every stored chunk.

        Returns a ``(n,)`` array aligned with :meth:`all_meta`. Used by hybrid
        search, which needs the full score distribution (not just the top-k) to
        fuse with lexical scores.
        """
        if len(self._meta) == 0:
            return np.zeros((0,), dtype=np.float32)
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        return self._vectors @ query

    def delete_document(self, document_id: int) -> None:
        """Drop every chunk belonging to a document (on document deletion)."""
        keep = [i for i, meta in enumerate(self._meta) if meta.get("document_id") != document_id]
        if len(keep) == len(self._meta):
            return
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        self._meta = [self._meta[i] for i in keep]
        # Row indices shifted, so the chunk_id lookup has to be rebuilt or every
        # later upsert would overwrite the wrong row.
        self._reindex()

    def _reindex(self) -> None:
        self._row_by_chunk = {
            meta["chunk_id"]: i
            for i, meta in enumerate(self._meta)
            if meta.get("chunk_id") is not None
        }

    def clear(self) -> None:
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._meta = []
        self._row_by_chunk = {}

    def rebuild_from_db(self, db) -> int:
        """Reload the index from persisted chunk rows. Returns the row count.

        Called on startup so a restarted process answers queries against exactly
        the chunks that survived in the database. Two things make this safe:

        1. Rows are read in ``chunk.id`` order, so the in-memory layout after a
           restart matches the layout built incrementally during ingest and
           score ties break the same way.
        2. Embeddings whose stored width no longer matches ``EMBEDDING_DIM`` are
           re-embedded from the chunk text and written back, instead of raising
           and taking the whole application down at startup. Changing
           ``EMBEDDING_DIM`` between runs is the realistic way to get here.
        """
        # Local imports avoid a circular import at module load time.
        from ragfabric_core.ingest.embed import get_embedder
        from ragfabric_core.models.document import Chunk, Document

        self.clear()
        rows = (
            db.query(Chunk, Document.filename, Document.format)
            .join(Document, Chunk.document_id == Document.id)
            .order_by(Chunk.id)
            .all()
        )

        repaired = 0
        records: list[dict] = []
        for chunk, filename, fmt in rows:
            embedding = chunk.embedding or []
            if len(embedding) != self.dim:
                embedding = get_embedder().embed_one(chunk.text).tolist()
                chunk.embedding = embedding
                repaired += 1
            records.append(
                {
                    "vector": embedding,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "collection_id": chunk.collection_id,
                    "filename": filename,
                    "format": fmt,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                }
            )
        if repaired:
            db.commit()
        self.upsert(records)
        return len(self._meta)

    def stats(self) -> dict:
        """Index size and dimensionality (surfaced on the health endpoint)."""
        return {
            "vectors": len(self._meta),
            "dim": self.dim,
            "documents": len({m.get("document_id") for m in self._meta}),
        }

    def __len__(self) -> int:
        return len(self._meta)


_store: InMemoryVectorStore | None = None


def get_store() -> InMemoryVectorStore:
    """Return the process-wide vector store singleton."""
    global _store
    if _store is None:
        _store = InMemoryVectorStore()
    return _store
