"""Delete a document from every place it lives.

Moved out of the documents route so the API and the connectors delete the same
way. The order matters:

1. Detach the document's chunks from the knowledge graph first, so entities and
   edges they supported have their confidence recomputed from the sources left
   (or are collected), rather than keeping a confidence no source reports (R42).
2. Delete the row. The ORM cascade removes ``chunks``; ``chunk_embeddings`` and
   ``chunk_search`` follow through their ``ON DELETE CASCADE`` foreign keys
   (enforced on SQLite too, see ``db/session.py``), and so does any
   ``connector_items`` row pointing at the document.
3. Delete from the configured vector and lexical stores explicitly. An external
   store such as Chroma has no foreign key, so nothing cascades there.
4. Delete the retained original from disk.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ragfabric_core.graph.extract import detach_documents
from ragfabric_core.ingest.storage import get_storage
from ragfabric_core.models.document import Document
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store


def delete_document_everywhere(db: Session, document_id: int) -> bool:
    """Delete ``document_id`` and everything derived from it. False if it does not exist."""
    document = db.get(Document, document_id)
    if document is None:
        return False
    detach_documents(db, [document.id])
    db.delete(document)
    db.commit()
    cfg = get_config()
    sf = get_session_factory()
    build_vector_store(cfg.vector_store, sf).delete_document(document_id)
    build_lexical_store(cfg.lexical_store, sf).delete_document(document_id)
    get_storage().delete(document_id)
    return True
