"""End-to-end ingestion: parse -> chunk -> embed -> persist -> index.

This is the single function the upload endpoint calls. It turns raw uploaded
bytes into a queryable document by running every stage of the pipeline and
keeping the SQL database and the in-memory vector index in sync:

    parse (parser.py) -> chunk (chunk.py) -> embed (embed.py)
      -> persist Document + Chunk rows (SQLite/Postgres)
      -> upsert vectors into the in-memory store

Chunk embeddings are persisted as JSON on the ``chunks`` table, so the in-memory
index can be rebuilt from the database on startup (see
``InMemoryVectorStore.rebuild_from_db``).

Two indexes are written during Phase 2: the v1 in memory index (still the live
query path) and the pgvector plus full text tables that Phases 3 and 4 will
query. The duplication ends when Phase 3 retires the in memory index.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from ragfabric_core.ingest import parser
from ragfabric_core.ingest.chunk import chunk_text
from ragfabric_core.ingest.clean import clean_text, document_type_for
from ragfabric_core.ingest.embed import get_embedder
from ragfabric_core.ingest.indexing import schedule_indexing
from ragfabric_core.ingest.storage import get_storage
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.queue.registry import build_queue
from ragfabric_core.runtime import get_config
from ragfabric_core.store.vector_store import get_store
from ragfabric_core.telemetry.tracing import trace

EMPTY_TEXT_NOTE = (
    "Parsed successfully but no extractable text was found "
    "(scanned or image-only file?). Nothing was indexed."
)


def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _index_content(db: Session, document: Document, data: bytes) -> Document:
    """Parse/chunk/embed ``data`` into chunks for an already-persisted document.

    Shared by first ingest and re-ingest (versioning). The document row is
    assumed to exist and to already carry ``filename`` / ``format``; this
    function owns everything from parsing to committing and indexing.
    """
    try:
        with trace("parse", format=document.format):
            text = parser.parse(document.filename, data)
        with trace("clean"):
            text = clean_text(text)
        cfg = get_config().ingestion
        with trace("chunk", chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap):
            chunks = chunk_text(
                text, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap, sections=True
            )
    except Exception as exc:  # unsupported format, corrupt file, etc.
        document.status = "failed"
        document.error = str(exc)[:500]
        document.num_chunks = 0
        db.commit()
        db.refresh(document)
        return document

    if not chunks:
        # A parse that yields nothing is not a crash, but silently reporting
        # "ready, 0 chunks" hides why the document never shows up in search.
        document.status = "ready"
        document.num_chunks = 0
        document.error = EMPTY_TEXT_NOTE
        db.commit()
        db.refresh(document)
        return document

    with trace("persist_chunks", count=len(chunks)):
        embedder = get_embedder()
        vectors = embedder.embed([c["text"] for c in chunks])

        store_records: list[dict] = []
        for chunk_meta, vector in zip(chunks, vectors, strict=True):
            embedding = vector.tolist()
            chunk_row = Chunk(
                document_id=document.id,
                collection_id=document.collection_id,
                chunk_index=chunk_meta["chunk_index"],
                page=chunk_meta["page"],
                char_start=chunk_meta["char_start"],
                char_end=chunk_meta["char_end"],
                text=chunk_meta["text"],
                embedding=embedding,
                section=chunk_meta.get("section"),
            )
            db.add(chunk_row)
            db.flush()  # assign chunk_row.id for the vector-store record
            store_records.append(
                {
                    "vector": embedding,
                    "chunk_id": chunk_row.id,
                    "document_id": document.id,
                    "collection_id": document.collection_id,
                    "filename": document.filename,
                    "format": document.format,
                    "page": chunk_meta["page"],
                    "chunk_index": chunk_meta["chunk_index"],
                    "text": chunk_meta["text"],
                }
            )

    document.status = "processing"
    document.num_chunks = len(chunks)
    document.error = ""
    db.commit()
    db.refresh(document)

    # Index only after a successful commit so the vector store mirrors the DB.
    # If the commit had failed we would have raised before touching the index.
    get_store().upsert(store_records)

    try:
        with trace("schedule_indexing", mode=get_config().ingestion.indexing):
            document.status = schedule_indexing(db, document, build_queue(get_config()))
    except Exception as exc:  # embedding or store failure during inline indexing
        db.rollback()
        document = db.get(Document, document.id)
        document.status = "failed"
        document.error = f"indexing failed: {exc}"[:500]
    db.commit()
    db.refresh(document)
    return document


def ingest_document(
    db: Session,
    *,
    filename: str,
    data: bytes,
    content_type: str = "",
    collection_id: int | None = None,
    owner_id: int | None = None,
) -> Document:
    """Ingest one uploaded file and return the persisted ``Document`` row.

    The document is created immediately (status ``processing``) so a row always
    exists; on success it flips to ``ready`` with ``num_chunks`` set, and on any
    parse/index error it flips to ``failed`` with the error recorded.
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
    return _index_content(db, document, data)


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
    replaced. Old chunks are removed from BOTH the database and the vector index
    before the new ones are written, otherwise stale text from the previous
    revision would keep surfacing in search results forever.
    """
    db.query(Chunk).filter(Chunk.document_id == document.id).delete(synchronize_session=False)
    get_store().delete_document(document.id)

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
    return _index_content(db, document, data)
