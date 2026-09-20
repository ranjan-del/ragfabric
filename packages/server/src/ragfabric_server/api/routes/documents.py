"""Document upload + management routes.

Uploading a file runs the full ingestion pipeline (parse -> chunk -> embed ->
persist -> index) synchronously and returns the created document with its
ingestion status and chunk count. Deleting a document removes its rows and its
rows from the configured vector and lexical stores. Moving a document between
collections updates every denormalised copy of collection_id (chunks,
chunk_embeddings, chunk_search) in the same transaction as the move, so the
access filter and the index never disagree about which collection a document
belongs to. This covers the relational tables; a Chroma vector store keeps its
own copy of collection_id in its external metadata store, outside this
transaction, and is not touched by a move.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import update
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS
from ragfabric_core.ingest.pipeline import ingest_document
from ragfabric_core.ingest.storage import get_storage
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.index import ChunkEmbedding, ChunkSearch
from ragfabric_core.models.user import Role, User
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.access_sql import access_clause
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store
from ragfabric_server.deps import get_access_filter, get_current_user, get_principal
from ragfabric_server.schemas.document import DocumentList, DocumentMove, DocumentOut

router = APIRouter()

# Content type served on download, chosen from the file's own extension and
# never from the caller-supplied Content-Type recorded at upload (that value
# is untrusted input: an upload's multipart Content-Type can name anything
# regardless of the actual bytes or the extension the format allow-list
# checked). An extension outside this map, or no extension at all, serves as
# application/octet-stream so a caller's browser is never handed a type it
# might render, such as text/html, for a file this deployment never classified
# as one of its own supported formats.
_DOWNLOAD_CONTENT_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "txt": "text/plain",
    "csv": "text/csv",
    "md": "text/markdown",
}


def _download_content_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _DOWNLOAD_CONTENT_TYPES.get(ext, "application/octet-stream")


@router.get("", response_model=DocumentList)
def list_documents(
    collection_id: int | None = None,
    format: str | None = None,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
) -> DocumentList:
    """List documents, optionally filtered by collection and/or file format."""
    query = db.query(Document)
    if collection_id is not None:
        query = query.filter(Document.collection_id == collection_id)
    if format is not None:
        query = query.filter(Document.format == format.lower())
    clause = access_clause(access, Document.id, Document.collection_id)
    if clause is not None:
        query = query.filter(clause)
    # Order by id as the tie-breaker: several uploads in the same request batch
    # can share a created_at timestamp, which would make paging order unstable.
    items = query.order_by(Document.created_at.desc(), Document.id.desc()).all()
    return DocumentList(items=items, total=len(items))


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    collection_id: int | None = Form(default=None),
    chunk_size: int | None = Form(default=None),
    chunk_overlap: int | None = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access: AccessFilter = Depends(get_access_filter),
) -> Document:
    """Upload a file and ingest it into the knowledge base.

    ``chunk_size``/``chunk_overlap`` are optional per-upload overrides of the
    configured ingestion chunk size/overlap. Left absent (the default), the
    document is chunked exactly as it always was. Named explicitly, they are
    validated here, at the edge, rather than clamped: a caller who asks for
    an overlap that does not fit its chunk size gets a 422, never a silently
    adjusted document.
    """
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported format '{ext or filename}'. "
                f"Supported: {', '.join(SUPPORTED_FORMATS)}."
            ),
        )
    if collection_id is not None:
        if db.get(Collection, collection_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
            )
        if not access.allows(None, collection_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
            )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    if chunk_size is not None or chunk_overlap is not None:
        cfg_ingestion = get_config().ingestion
        effective_size = chunk_size if chunk_size is not None else cfg_ingestion.chunk_size
        effective_overlap = (
            chunk_overlap if chunk_overlap is not None else cfg_ingestion.chunk_overlap
        )
        if chunk_size is not None and chunk_size < 1:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="chunk_size must be a positive integer.",
            )
        if not (0 <= effective_overlap < effective_size):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size.",
            )

    return ingest_document(
        db,
        filename=filename,
        data=data,
        content_type=file.content_type or "",
        collection_id=collection_id,
        owner_id=current_user.id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
) -> Document:
    """Return a single document's detail and ingestion status."""
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if not access.allows(document.id, document.collection_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return document


@router.get("/{document_id}/download")
def download_document(
    document_id: int,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
) -> FileResponse:
    """Return the retained original file, if the deployment keeps originals.

    The response's content type is pinned from the file's own extension (see
    ``_download_content_type``), never from the value recorded at upload, and
    always carries ``Content-Disposition: attachment`` and
    ``X-Content-Type-Options: nosniff`` so an uploaded HTML (or otherwise
    renderable) file can never be served in a way a browser would render in
    this origin.
    """
    document = db.get(Document, document_id)
    if document is None or not document.storage_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original not available.")
    if not access.allows(document.id, document.collection_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original not available.")
    try:
        path = get_storage().path_for(document.storage_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Original not available."
        ) from exc
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original not available.")
    return FileResponse(
        path,
        filename=document.filename,
        media_type=_download_content_type(document.filename),
        content_disposition_type="attachment",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.post("/{document_id}/move", response_model=DocumentOut)
def move_document(
    document_id: int,
    payload: DocumentMove,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    access: AccessFilter = Depends(get_access_filter),
) -> Document:
    """Move a document into a different collection, or out of any collection.

    ``chunk_embeddings`` and ``chunk_search`` each carry their own
    denormalised ``collection_id`` (ADR 0003: it lets the access predicate
    apply inside the store query itself, with no join back to ``documents``),
    and ``chunks.collection_id`` denormalises the same value for the chunk
    metadata a retrieval ever hands back to a caller. Left stale after a
    move, a principal granted only the OLD collection could still retrieve
    the moved document out of the index: an access-control bug, not a
    cosmetic one. All three are updated here, inside the same transaction as
    the move itself, so no reader ever observes a document whose own row
    names one collection while its index rows still name another.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if current_user.role != Role.ADMIN.value and document.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You may only move your own documents.",
        )
    if payload.collection_id is not None:
        if db.get(Collection, payload.collection_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
            )
        if not access.allows(None, payload.collection_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
            )

    document.collection_id = payload.collection_id
    db.execute(
        update(Chunk)
        .where(Chunk.document_id == document_id)
        .values(collection_id=payload.collection_id)
    )
    db.execute(
        update(ChunkEmbedding)
        .where(ChunkEmbedding.document_id == document_id)
        .values(collection_id=payload.collection_id)
    )
    db.execute(
        update(ChunkSearch)
        .where(ChunkSearch.document_id == document_id)
        .values(collection_id=payload.collection_id)
    )
    db.commit()
    db.refresh(document)
    return document


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete a document (owner or admin) and drop its vectors from the index."""
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if current_user.role != Role.ADMIN.value and document.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You may only delete your own documents.",
        )
    db.delete(document)  # cascades to chunks
    db.commit()
    # The ORM cascade above deletes `chunks` rows. `chunk_embeddings` and
    # `chunk_search` have an ON DELETE CASCADE foreign key to `chunks.id`, but
    # that is a database-level constraint, and SQLite does not enforce foreign
    # keys unless PRAGMA foreign_keys=ON is set (it is not, here), so on
    # SQLite those rows survive as orphans. On PostgreSQL the FK happens to
    # cascade them away, which is why this went unnoticed until now. On a real
    # Chroma collection nothing will ever cascade it: Chroma is an external
    # service with no foreign key at all. Delete from the configured vector
    # and lexical stores explicitly so no orphaned rows accumulate and
    # silently crowd out live results.
    cfg = get_config()
    sf = get_session_factory()
    build_vector_store(cfg.vector_store, sf).delete_document(document_id)
    build_lexical_store(cfg.lexical_store, sf).delete_document(document_id)
    get_storage().delete(document_id)
    return {"detail": "Document deleted.", "id": document_id}
