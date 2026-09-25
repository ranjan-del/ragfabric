"""Job handlers. Each takes a Session and does one document's worth of work.

index_document embeds the document's chunks with the configured provider and
writes the vector and lexical indexes. It marks the document ready only after
both writes succeed, so a document is never searchable in one index and
missing from the other. extract_graph builds the document's slice of the
knowledge graph when ``graph_store.enabled`` is set, and returns without a
model call when it is not. The two jobs may run in either order, so
index_document leaves an earlier ``extract_graph failed:`` error and its
failed status in place, and only a later successful extract_graph clears it
(R44).

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
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ragfabric_core.config_file import GraphStoreConfig
from ragfabric_core.graph.extract import ExtractionReport, extract_chunks
from ragfabric_core.graph.resolve import ResolutionReport, resolve_entities
from ragfabric_core.models.document import Chunk, Document, IngestionRun
from ragfabric_core.models.graph import EntitySource
from ragfabric_core.providers.base import EmbeddingProvider, LLMProvider
from ragfabric_core.stores.base import LexicalStore, VectorStore
from ragfabric_core.telemetry.tracing import start_trace, trace

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# The prefix a failed extract_graph job records (the worker's ``"<kind> failed:"``
# and the inline pipeline's GraphExtractionFailed both write it).
GRAPH_FAILURE_PREFIX = "extract_graph failed:"


def _graph_failed(document: Document) -> bool:
    """Whether the document carries a graph extraction failure (R44).

    A document's index_document and extract_graph jobs can run in either
    order on a queue. index_document owns ``ready`` for the search indexes,
    but a graph failure recorded before it runs must survive it: resetting
    status and error unconditionally made a document with no graph look
    healthy. Only extract_graph clears its own failure, on its next success.
    """
    return document.status == "failed" and (document.error or "").startswith(GRAPH_FAILURE_PREFIX)


def _indexed_since_last_ingest(db: Session, document_id: int) -> bool:
    """Whether an index run finished after the document's latest ingest run.

    Run ids increase in commit order, so an ``index`` row newer than the
    newest ``ingest`` row means this revision's index_document job has
    completed. With no ingest row at all, any index row counts.
    """
    latest = {
        phase: run_id
        for phase, run_id in db.execute(
            select(IngestionRun.phase, func.max(IngestionRun.id))
            .where(IngestionRun.document_id == document_id)
            .group_by(IngestionRun.phase)
        )
    }
    indexed = latest.get("index")
    return indexed is not None and indexed > latest.get("ingest", 0)


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
    started = time.perf_counter()
    with start_trace() as tracing:
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
    if not _graph_failed(document):
        document.status = "ready"
        document.error = ""
    db.add(
        IngestionRun(
            document_id=document_id,
            phase="index",
            status="ready",
            chunk_count=len(chunks),
            embedding_model=result.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            trace=[s.model_dump() for s in tracing.spans],
        )
    )
    db.commit()
    return len(chunks)


@dataclass(frozen=True)
class GraphExtractionOutcome:
    """What one document's graph pass did: the extraction and resolution reports."""

    extraction: ExtractionReport
    resolution: ResolutionReport | None

    def summary(self) -> str:
        """One log-ready line with every count an operator needs to judge the pass."""
        extraction = self.extraction
        merges = 0 if self.resolution is None else len(self.resolution.merges)
        embedding_calls = 0 if self.resolution is None else self.resolution.embedding_calls
        return (
            f"{extraction.entities_stored} entities stored, "
            f"{extraction.relationships_stored} relationships stored, "
            f"{extraction.entities_discarded} entities discarded, "
            f"{extraction.relationships_discarded} relationships discarded, "
            f"{len(extraction.extracted_chunk_ids)} chunks extracted, "
            f"{extraction.chunks_skipped} chunks unchanged, "
            f"contract violation {extraction.contract_violation}, "
            f"{merges} merges, {embedding_calls} embedding calls"
        )


def extract_graph(
    db: Session,
    document_id: int,
    *,
    settings: GraphStoreConfig,
    llm: LLMProvider | None,
    embedder: EmbeddingProvider | None,
) -> GraphExtractionOutcome | None:
    """Extract the document's entities and edges, resolve them, and commit.

    Disabled (``settings.enabled`` false, the default): log and return
    ``None`` before touching the document, so a deployment without a graph
    makes no model call and needs no model provider (``llm`` may be ``None``).

    Enabled: ``extract_chunks`` over the document's chunks with the
    configured confidence floor, extraction model (``None`` means the
    provider's own default model) and enabled type lists; then
    ``resolve_entities`` with the embedder and similarity threshold; then one
    commit.

    Resolution is scoped to the entities sourced by the chunks that were
    actually extracted in this run (R36), and skipped entirely when none
    were: an unchanged chunk is skipped by the extractor's hash check, and
    its entities were already resolved when it was last extracted, so a
    re-run over unchanged text makes no model call and no embedding call.

    Known cost: stage 3 of resolution embeds the whole type group of every
    in-scope entity (only pairs with an in-scope member may merge, but the
    comparison needs the other side's vector too), so each changed document
    that names a person re-embeds every person in the corpus. That is
    O(type group) embedding per changed document. An approximate nearest
    neighbour or cached-vector prefilter is deferred to Phase 8, where it can
    be measured against the recall it would cost.
    """
    if not settings.enabled:
        log.info("extract_graph: graph extraction is disabled; document %s skipped", document_id)
        return None
    if llm is None:
        raise ValueError("graph_store.enabled is true but no LLM provider was supplied")
    document = db.get(Document, document_id)
    if document is None:
        log.warning("extract_graph: document %s no longer exists", document_id)
        return None
    chunks = (
        db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()
    )
    extraction = extract_chunks(
        db,
        chunks,
        llm,
        floor=settings.confidence_floor,
        model=settings.extraction_model or llm.default_model,
        entity_types=settings.entity_types,
        relation_types=settings.relation_types,
    )
    touched: list[int] = []
    if extraction.extracted_chunk_ids:
        touched = sorted(
            set(
                db.execute(
                    select(EntitySource.entity_id).where(
                        EntitySource.chunk_id.in_(extraction.extracted_chunk_ids)
                    )
                ).scalars()
            )
        )
    resolution = None
    if touched:
        resolution = resolve_entities(
            db,
            embedder,
            similarity_threshold=settings.similarity_threshold,
            entity_ids=touched,
        )
    if _graph_failed(document):
        # This job's own earlier failure, now fixed (R44). The document is
        # ready only if its index job has also finished for this revision;
        # otherwise it goes back to waiting for it, rather than claiming to
        # be searchable.
        document.error = ""
        document.status = "ready" if _indexed_since_last_ingest(db, document_id) else "indexing"
    db.commit()
    outcome = GraphExtractionOutcome(extraction=extraction, resolution=resolution)
    log.info("extract_graph: document %s: %s", document_id, outcome.summary())
    return outcome


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

    Known limitation: age against `Document.created_at` is the only signal
    used to tell "stuck" apart from "in flight". There is no lock, heartbeat,
    or worker-ownership record, so a document that is merely slow (a large
    file, or a rate-limited embedding provider) rather than crashed can be
    picked up here WHILE a live worker is still processing it. The two writes
    still converge to the same end state because they are idempotent, but the
    embedding provider itself gets called twice, which costs money and can
    itself trigger the same rate limiting that made the document slow in the
    first place, and two uncoordinated sessions end up committing the same
    `Document` row. This is not fixed here; a proper fix is a lease or
    heartbeat, tracked as hardening-phase work, not something this function
    does. The safe operating procedure until then is to stop the worker(s)
    before running a reconciliation pass.
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
