"""Decide whether a freshly chunked document is indexed now or by a worker."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from ragfabric_core.models.document import Document
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider
from ragfabric_core.queue.base import Job, JobQueue
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store
from ragfabric_core.workers.handlers import extract_graph, index_document


def _embedding_provider():
    return build_embedding_provider(get_config().embeddings)


def _stores():
    cfg = get_config()
    sf = get_session_factory()
    provider = _embedding_provider()
    return (
        build_vector_store(cfg.vector_store, sf, embedding_model=provider.model),
        build_lexical_store(cfg.lexical_store, sf),
    )


def index_inline(db: Session, document: Document) -> int:
    vector_store, lexical_store = _stores()
    return index_document(
        db,
        document.id,
        embedding_provider=_embedding_provider(),
        vector_store=vector_store,
        lexical_store=lexical_store,
    )


def extract_graph_inline(db: Session, document: Document) -> None:
    """Build the document's graph in the request, when the graph is enabled.

    Inline is the default indexing mode, so without this a deployment that
    set ``graph_store.enabled`` and never ran a worker would get no graph and
    no error. The model provider is built only when extraction is enabled.
    """
    settings = get_config().graph_store
    extract_graph(
        db,
        document.id,
        settings=settings,
        llm=build_llm_provider(get_config().llm) if settings.enabled else None,
        embedder=_embedding_provider(),
    )


def schedule_indexing(db: Session, document: Document, queue: JobQueue | None) -> str:
    if queue is None:
        index_inline(db, document)
        extract_graph_inline(db, document)
        return "ready"
    document.status = "indexing"
    db.commit()
    for kind in ("index_document", "extract_graph"):
        queue.enqueue(Job(id=uuid.uuid4().hex, kind=kind, payload={"document_id": document.id}))
    return "indexing"
