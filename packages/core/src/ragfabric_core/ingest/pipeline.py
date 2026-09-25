"""End-to-end ingestion: parse -> chunk -> embed -> persist -> index.

This is the single function the upload endpoint calls. It turns raw uploaded
bytes into a queryable document by running every stage of the pipeline:

    parse (parser.py) -> chunk (chunk.py) -> embed (embed.py)
      -> persist Document + Chunk rows (SQLite/Postgres)
      -> fan out to the configured vector and lexical stores (indexing.py)

Chunk embeddings are also persisted as JSON on the ``chunks`` table
(``Chunk.embedding``); nothing reads that column back today (Task 13 retired
the last reader, the v1 in-memory index's startup rebuild), but the column is
``NOT NULL`` and dropping it needs a migration, so it is still written here.
"""

from __future__ import annotations

import time

from sqlalchemy.orm import Session

from ragfabric_core.graph.extract import carry_graph_to_new_chunks
from ragfabric_core.ingest import parser
from ragfabric_core.ingest.chunk import chunk_text
from ragfabric_core.ingest.clean import clean_text, document_type_for
from ragfabric_core.ingest.embed import get_embedder
from ragfabric_core.ingest.indexing import GraphExtractionFailed, schedule_indexing
from ragfabric_core.ingest.storage import get_storage
from ragfabric_core.models.document import Chunk, Document, IngestionRun
from ragfabric_core.queue.registry import build_queue
from ragfabric_core.runtime import get_config
from ragfabric_core.telemetry.tracing import TraceContext, start_trace, trace

EMPTY_TEXT_NOTE = (
    "Parsed successfully but no extractable text was found "
    "(scanned or image-only file?). Nothing was indexed."
)


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _record_ingestion_run(
    db: Session, document: Document, tracing: TraceContext, started: float, *, chunk_count: int
) -> None:
    """Persist the pipeline's own spans as one ``IngestionRun`` row.

    Called exactly once per ``_index_content`` call, on every exit path
    (parse failure, empty text, or a clean run), so a failed ingest gets its
    own row too, carrying whatever spans ran before the failure.
    ``embedding_model`` is left null: this phase's embedder is the local
    hashing embedder used only to populate the legacy ``Chunk.embedding``
    column, not the provider the real vector/lexical indexes are built with,
    so naming a model here would not be a measured value (ADR 0004).
    """
    db.add(
        IngestionRun(
            document_id=document.id,
            phase="ingest",
            status=document.status,
            chunk_count=chunk_count,
            embedding_model=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=document.error or None,
            trace=[s.model_dump() for s in tracing.spans],
        )
    )
    db.commit()


def _index_content(
    db: Session,
    document: Document,
    data: bytes,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    replaced_chunk_ids: list[int] | None = None,
) -> Document:
    """Parse/chunk/embed ``data`` into chunks for an already-persisted document.

    Shared by first ingest and re-ingest (versioning). The document row is
    assumed to exist and to already carry ``filename`` / ``format``; this
    function owns everything from parsing to committing and indexing.

    ``replaced_chunk_ids`` (re-ingest only) are the previous revision's
    chunks. They are deleted in the same transaction that writes the new
    ones, on every exit path, after the graph state of each unchanged chunk
    has been carried over to its replacement (``_retire_chunks``, R41).

    ``chunk_size``/``chunk_overlap`` of ``None`` (the default for every
    existing caller) means "use the configured value from
    ``get_config().ingestion``", so a caller that never mentions either
    keeps the exact chunking behaviour it always had.
    """
    started = time.perf_counter()
    with start_trace() as tracing:
        try:
            with trace("parse", format=document.format):
                text = parser.parse(document.filename, data)
            with trace("clean"):
                text = clean_text(text)
            cfg = get_config().ingestion
            size = chunk_size if chunk_size is not None else cfg.chunk_size
            overlap = chunk_overlap if chunk_overlap is not None else cfg.chunk_overlap
            with trace("chunk", chunk_size=size, overlap=overlap):
                chunks = chunk_text(text, chunk_size=size, overlap=overlap, sections=True)
        except Exception as exc:  # unsupported format, corrupt file, etc.
            _retire_chunks(db, replaced_chunk_ids, [])
            document.status = "failed"
            document.error = str(exc)[:500]
            document.num_chunks = 0
            db.commit()
            db.refresh(document)
            _record_ingestion_run(db, document, tracing, started, chunk_count=0)
            return document

        if not chunks:
            # A parse that yields nothing is not a crash, but silently reporting
            # "ready, 0 chunks" hides why the document never shows up in search.
            _retire_chunks(db, replaced_chunk_ids, [])
            document.status = "ready"
            document.num_chunks = 0
            document.error = EMPTY_TEXT_NOTE
            db.commit()
            db.refresh(document)
            _record_ingestion_run(db, document, tracing, started, chunk_count=0)
            return document

        with trace("persist_chunks", count=len(chunks)):
            embedder = get_embedder()
            vectors = embedder.embed([c["text"] for c in chunks])

            new_rows: list[Chunk] = []
            for chunk_meta, vector in zip(chunks, vectors, strict=True):
                chunk_row = Chunk(
                    document_id=document.id,
                    collection_id=document.collection_id,
                    chunk_index=chunk_meta["chunk_index"],
                    page=chunk_meta["page"],
                    char_start=chunk_meta["char_start"],
                    char_end=chunk_meta["char_end"],
                    text=chunk_meta["text"],
                    embedding=vector.tolist(),
                    section=chunk_meta.get("section"),
                )
                db.add(chunk_row)
                new_rows.append(chunk_row)
            db.flush()
            _retire_chunks(db, replaced_chunk_ids, new_rows)

        document.status = "processing"
        document.num_chunks = len(chunks)
        document.error = ""
        db.commit()
        db.refresh(document)

        try:
            with trace("schedule_indexing", mode=get_config().ingestion.indexing):
                document.status = schedule_indexing(db, document, build_queue(get_config()))
        except Exception as exc:  # embedding, store or graph failure during inline indexing
            db.rollback()
            document = db.get(Document, document.id)
            document.status = "failed"
            stage = "extract_graph" if isinstance(exc, GraphExtractionFailed) else "indexing"
            document.error = f"{stage} failed: {exc}"[:500]
        db.commit()
        db.refresh(document)
        _record_ingestion_run(db, document, tracing, started, chunk_count=document.num_chunks)
        return document


def _retire_chunks(db: Session, old_ids: list[int] | None, new_chunks: list[Chunk]) -> None:
    """Delete a re-ingested document's previous chunks, carrying graph state over first.

    Unchanged text keeps its extraction hash and graph links on the chunk
    that replaces it, and whatever only the removed text sourced is garbage
    collected with every survivor's confidence recomputed
    (``carry_graph_to_new_chunks``, R41, R42), before the rows are deleted.
    """
    if not old_ids:
        return
    old_chunks = db.query(Chunk).filter(Chunk.id.in_(old_ids)).all()
    carry_graph_to_new_chunks(db, old_chunks, new_chunks)
    db.query(Chunk).filter(Chunk.id.in_(old_ids)).delete()
    db.flush()


def ingest_document(
    db: Session,
    *,
    filename: str,
    data: bytes,
    content_type: str = "",
    collection_id: int | None = None,
    owner_id: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Document:
    """Ingest one uploaded file and return the persisted ``Document`` row.

    The document is created immediately (status ``processing``) so a row always
    exists; on success it flips to ``ready`` with ``num_chunks`` set, and on any
    parse/index error it flips to ``failed`` with the error recorded.

    ``chunk_size``/``chunk_overlap`` default to ``None``, meaning "use the
    configured value from ``get_config().ingestion``" (see ``_index_content``),
    so every existing caller that never passes them keeps its current
    behaviour unchanged. The upload route validates any caller-supplied pair
    at the edge before this is ever called; this function trusts its inputs.
    """
    document = Document(
        filename=filename,
        content_type=content_type,
        format=_extension(filename),
        document_type=document_type_for(_extension(filename)),
        collection_id=collection_id,
        owner_id=owner_id,
        status="processing",
    )
    db.add(document)
    db.flush()  # assign document.id without committing yet
    if get_config().ingestion.retain_originals:
        document.storage_path = get_storage().save(document.id, filename, data)
    return _index_content(db, document, data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def reingest_document(
    db: Session,
    document: Document,
    *,
    filename: str,
    data: bytes,
    content_type: str = "",
) -> Document:
    """Replace a document's content with a new revision and bump its version.

    This is what "versioning" means here: the document keeps its id, owner and
    collection (so existing references stay valid) while its chunks are fully
    replaced. Old chunks are removed from the database in the same
    transaction that writes the new ones, otherwise stale text from the
    previous revision would keep surfacing in search results forever. The
    ``ON DELETE CASCADE`` from ``chunk_embeddings``/``chunk_search`` to
    ``chunks.id`` cleans the vector/lexical rows too, on PostgreSQL and on
    SQLite alike (foreign keys are enforced on every SQLite connection; see
    ``db/session.py``). A Chroma deployment has no foreign key at all, so
    nothing cascades there; that is a pre-existing gap this does not fix.

    The knowledge graph is carried across the replacement (R41): a new chunk
    whose text equals an old chunk's inherits that chunk's extraction hash and
    graph links, so an unchanged chunk is not extracted again, and only what
    the removed text alone sourced is garbage collected, with the confidence
    of everything it touched recomputed from the sources left (R42).
    """
    replaced = [
        chunk_id
        for (chunk_id,) in db.query(Chunk.id).filter(Chunk.document_id == document.id).all()
    ]

    document.filename = filename
    get_storage().delete(document.id)
    if get_config().ingestion.retain_originals:
        document.storage_path = get_storage().save(document.id, filename, data)
    document.format = _extension(filename)
    document.document_type = document_type_for(document.format)
    document.content_type = content_type or document.content_type
    document.version += 1
    document.status = "processing"
    document.num_chunks = 0
    db.flush()
    return _index_content(db, document, data, replaced_chunk_ids=replaced)
