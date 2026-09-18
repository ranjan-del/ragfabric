"""Re-embed an existing corpus under the active embedding model.

Why this is not "ingest again": re-ingesting re-parses and re-chunks, which
changes chunk ids and orphans every Source row recorded against an earlier
answer. Reindexing reads the chunks table, which is the source of truth for
text, and replaces only the derived vectors. Retrieval runs recorded before a
model change still resolve to real chunks afterwards.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ragfabric_core.models.document import Chunk
from ragfabric_core.providers.base import EmbeddingProvider
from ragfabric_core.stores.base import LexicalStore, VectorStore


def reindex_all(
    db: Session,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    lexical_store: LexicalStore | None = None,
    batch_size: int = 64,
    document_id: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Re-embed every chunk (or one document's chunks) and return how many."""
    base = select(Chunk).order_by(Chunk.id)
    counter = select(func.count()).select_from(Chunk)
    if document_id is not None:
        base = base.where(Chunk.document_id == document_id)
        counter = counter.where(Chunk.document_id == document_id)
        vector_store.delete_document(document_id)
    total = int(db.execute(counter).scalar() or 0)

    done = 0
    batch: list[Chunk] = []
    for chunk in db.execute(base).scalars():
        batch.append(chunk)
        if len(batch) >= batch_size:
            done += _flush(batch, embedding_provider, vector_store, lexical_store)
            batch = []
            if on_progress is not None:
                on_progress(done, total)
    if batch:
        done += _flush(batch, embedding_provider, vector_store, lexical_store)
    if on_progress is not None:
        on_progress(done, total)
    return done


def _flush(
    batch: list[Chunk],
    provider: EmbeddingProvider,
    vector_store: VectorStore,
    lexical_store: LexicalStore | None,
) -> int:
    texts = [c.text for c in batch]
    result = provider.embed(texts)
    ids = [c.id for c in batch]
    payloads = [
        {
            "document_id": c.document_id,
            "collection_id": c.collection_id,
            "model": provider.model,
            "dim": provider.dim,
        }
        for c in batch
    ]
    vector_store.upsert(ids, result.vectors, payloads)
    if lexical_store is not None:
        lexical_store.index(ids, texts, payloads)
    return len(batch)
