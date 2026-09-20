"""Job handlers. Each takes a Session and does one document's worth of work.

index_document embeds the document's chunks with the configured provider and
writes the vector and lexical indexes. It marks the document ready only after
both writes succeed, so a document is never searchable in one index and
missing from the other. extract_graph is the Phase 6 hook; here it only
records that the job ran.

Fan-out atomicity (deferred item, Phase 2 review): index_document writes the
vector store and the lexical store as two separate commits (each store opens
and commits its own session; see PgVectorStore.upsert / ChromaVectorStore.upsert
and PostgresLexicalStore.index), so a hard crash between the two leaves a
document embedded in one index and absent from the other. A single
transaction spanning both is not available in general: Chroma is an external
service with no transaction to join, so "one transaction" can never be an
honest answer for a Chroma deployment, only for a pgvector-plus-postgres_fts
one, and this module does not special-case the vector store kind.

The choice made here is the other option the item allows: keep the worker
idempotent (already true: both upsert() and index() key their writes by
chunk_id and overwrite, so calling index_document again for a document that
already has some or all of its rows reproduces the same end state, never a
duplicate) and reconcile on retry. Because a hard crash mid-job raises no
Python exception for run_once to catch, the crashed document's status never
advances past "indexing"; reconcile_stuck_indexing finds documents stuck
there past a grace period and simply re-runs the job. This is deliberately an
explicit, operator-triggered action (see `ragfabric reconcile`), not a
background timer: the grace period is judged against `Document.created_at`
(there is no `updated_at` column to compare against without a migration,
which this batch does not own), so retrying too eagerly could collide with a
document that is merely slow, not crashed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.providers.base import EmbeddingProvider
from ragfabric_core.stores.base import LexicalStore, VectorStore
from ragfabric_core.telemetry.tracing import trace

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


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
    with trace("embed", count=len(chunks)):
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
    with trace("vector_upsert"):
        vector_store.upsert(ids, result.vectors, payloads)
    with trace("lexical_index"):
        lexical_store.index(ids, [c.text for c in chunks], payloads)
    document = db.get(Document, document_id)
    document.status = "ready"
    document.error = ""
    db.commit()
    return len(chunks)


def extract_graph(db: Session, document_id: int) -> None:
    log.info("extract_graph: document %s queued; graph extraction arrives in Phase 6", document_id)


def reconcile_stuck_indexing(
    db: Session,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    lexical_store: LexicalStore,
    older_than_seconds: int = 300,
) -> int:
    """Re-run indexing for any document stuck in "indexing" past the grace period.

    A document only reaches "indexing" (as opposed to "processing", set while
    chunks are still being written) once its fan-out job has been enqueued or
    started; it only leaves "indexing" when index_document commits both the
    vector and lexical writes and flips it to "ready" (or, on a caught
    exception, to "failed" by the worker's own error handling in
    workers/runner.py). A hard crash between the two store writes raises
    nothing for either path to catch, so the document is left in "indexing"
    forever with no other signal that anything went wrong.

    Calling index_document again for such a document is safe: it is
    idempotent by construction (see the module docstring), so a document that
    already has, say, its vector rows from before the crash simply has them
    overwritten with the same values, and gets its missing lexical rows
    written for the first time, ending in exactly the state one clean run
    would have produced.
    """
    cutoff = _now() - timedelta(seconds=older_than_seconds)
    stuck = (
        db.query(Document)
        .filter(Document.status == "indexing", Document.created_at < cutoff)
        .order_by(Document.id)
        .all()
    )
    for document in stuck:
        index_document(
            db,
            document.id,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            lexical_store=lexical_store,
        )
    return len(stuck)
