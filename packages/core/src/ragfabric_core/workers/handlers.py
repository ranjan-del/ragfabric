"""Job handlers. Each takes a Session and does one document's worth of work.

index_document embeds the document's chunks with the configured provider and
writes the vector and lexical indexes. It marks the document ready only after
both writes succeed, so a document is never searchable in one index and
missing from the other. extract_graph is the Phase 6 hook; here it only
records that the job ran.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.providers.base import EmbeddingProvider
from ragfabric_core.stores.base import LexicalStore, VectorStore

log = logging.getLogger(__name__)


def index_document(
    db: Session,
    document_id: int,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    lexical_store: LexicalStore,
) -> int:
    document = db.get(Document, document_id)
    if document is None:
        log.warning("index_document: document %s no longer exists", document_id)
        return 0
    chunks = (
        db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()
    )
    if not chunks:
        document.status = "ready"
        db.commit()
        return 0
    result = embedding_provider.embed([c.text for c in chunks])
    ids = [c.id for c in chunks]
    payloads = [
        {
            "document_id": c.document_id,
            "collection_id": c.collection_id,
            "model": result.model,
            "dim": embedding_provider.dim,
        }
        for c in chunks
    ]
    vector_store.upsert(ids, result.vectors, payloads)
    lexical_store.index(ids, [c.text for c in chunks], payloads)
    document = db.get(Document, document_id)
    document.status = "ready"
    document.error = ""
    db.commit()
    return len(chunks)


def extract_graph(db: Session, document_id: int) -> None:
    log.info("extract_graph: document %s queued; graph extraction arrives in Phase 6", document_id)
