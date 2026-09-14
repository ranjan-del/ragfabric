"""Decide whether a freshly chunked document is indexed now or by a worker."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from ragfabric_core.models.document import Document
from ragfabric_core.providers.registry import build_embedding_provider
from ragfabric_core.queue.base import Job, JobQueue
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store
from ragfabric_core.workers.handlers import index_document


def _embedding_provider():
    return build_embedding_provider(get_config().embeddings)


def _stores():
    cfg = get_config()
    sf = get_session_factory()
    return build_vector_store(cfg.vector_store, sf), build_lexical_store(cfg.lexical_store, sf)


def index_inline(db: Session, document: Document) -> int:
    vector_store, lexical_store = _stores()
    return index_document(
        db,
        document.id,
        embedding_provider=_embedding_provider(),
        vector_store=vector_store,
        lexical_store=lexical_store,
    )


def schedule_indexing(db: Session, document: Document, queue: JobQueue | None) -> str:
    if queue is None:
        index_inline(db, document)
        return "ready"
    document.status = "indexing"
    db.commit()
    for kind in ("index_document", "extract_graph"):
        queue.enqueue(Job(id=uuid.uuid4().hex, kind=kind, payload={"document_id": document.id}))
    return "indexing"
