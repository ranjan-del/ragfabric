"""Re-embed an existing corpus under the active embedding model.

Why this is not "ingest again": re-ingesting re-parses and re-chunks, which
changes chunk ids and orphans every Source row recorded against an earlier
answer. Reindexing reads the chunks table, which is the source of truth for
text, and replaces only the derived vectors. Retrieval runs recorded before a
model change still resolve to real chunks afterwards.

Memory: chunks are fetched a page at a time by keyset (`id > last_seen_id
... LIMIT batch_size`), not by loading the whole corpus and slicing it in
Python, and the Session is expunged after every page. `batch_size` therefore
bounds both the size of each embed/upsert call and the number of ORM Chunk
objects the Session holds at once; a large corpus does not accumulate in
memory just because the query has not finished.

`yield_per`/streaming was tried first and rejected: it keeps the read cursor
open across the whole call, and on SQLite that holds a lock that the vector
store's own write connection (a separate session) then collides with
("database is locked"). Keyset paging closes each read fully before any write
happens, so it has no such conflict on either dialect.
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
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    lexical_store: LexicalStore | None = None,
    lexical_only: bool = False,
    batch_size: int = 64,
    document_id: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Re-embed every chunk (or one document's chunks) and return how many.

    ``lexical_only`` rewrites the lexical index and nothing else: no embedding
    call, no vector write. Phase 4 needs it because chunk_search rows written
    before BM25 carry doc_len = 0 and a term-frequency-free tsv, and those rows
    are excluded from BM25 rather than scored. Re-embedding a whole corpus to
    fix a lexical column would cost real provider money for no benefit, so the
    two are separable.
    """
    if lexical_only:
        if lexical_store is None:
            raise ValueError("lexical_only needs a lexical_store")
    elif embedding_provider is None or vector_store is None:
        raise ValueError("reindex needs an embedding_provider and a vector_store")

    counter = select(func.count()).select_from(Chunk)
    if document_id is not None:
        counter = counter.where(Chunk.document_id == document_id)
        if not lexical_only:
            vector_store.delete_document(document_id)
    total = int(db.execute(counter).scalar() or 0)

    if total == 0:
        if on_progress is not None:
            on_progress(0, 0)
        return 0

    done = 0
    last_id = 0
    while True:
        page = select(Chunk).where(Chunk.id > last_id)
        if document_id is not None:
            page = page.where(Chunk.document_id == document_id)
        page = page.order_by(Chunk.id).limit(batch_size)
        batch = list(db.execute(page).scalars().all())
        if not batch:
            break
        last_id = batch[-1].id
        done += _flush(batch, embedding_provider, vector_store, lexical_store, lexical_only)
        # Drop this page from the Session's identity map before fetching the
        # next one, so ORM memory is bounded by batch_size, not by corpus size.
        db.expunge_all()
        if on_progress is not None:
            on_progress(done, total)
    return done


def _flush(
    batch: list[Chunk],
    provider: EmbeddingProvider | None,
    vector_store: VectorStore | None,
    lexical_store: LexicalStore | None,
    lexical_only: bool = False,
) -> int:
    texts = [c.text for c in batch]
    ids = [c.id for c in batch]
    payloads = [
        {
            "document_id": c.document_id,
            "collection_id": c.collection_id,
            "model": provider.model if provider is not None else None,
            "dim": provider.dim if provider is not None else None,
        }
        for c in batch
    ]
    if not lexical_only:
        result = provider.embed(texts)
        vector_store.upsert(ids, result.vectors, payloads)
    if lexical_store is not None:
        lexical_store.index(ids, texts, payloads)
    return len(batch)
