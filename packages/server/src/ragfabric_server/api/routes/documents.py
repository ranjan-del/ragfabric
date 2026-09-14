"""Document upload + management routes.

Uploading a file runs the full ingestion pipeline (parse -> chunk -> embed ->
persist -> index) synchronously and returns the created document with its
ingestion status and chunk count. Deleting a document removes its rows and its
vectors from the in-memory index.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS
from ragfabric_core.ingest.pipeline import ingest_document
from ragfabric_core.ingest.storage import get_storage
from ragfabric_core.models.document import Collection, Document
from ragfabric_core.models.user import Role, User
from ragfabric_core.store.vector_store import get_store
from ragfabric_core.stores.access_sql import access_clause
from ragfabric_server.deps import get_access_filter, get_current_user, get_principal
from ragfabric_server.schemas.document import DocumentList, DocumentOut

router = APIRouter()


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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Document:
    """Upload a file and ingest it into the knowledge base."""
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
    if collection_id is not None and db.get(Collection, collection_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    return ingest_document(
        db,
        filename=filename,
        data=data,
        content_type=file.content_type or "",
        collection_id=collection_id,
        owner_id=current_user.id,
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
    """Return the retained original file, if the deployment keeps originals."""
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
        media_type=document.content_type or "application/octet-stream",
    )


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
    get_store().delete_document(document_id)
    get_storage().delete(document_id)
    return {"detail": "Document deleted.", "id": document_id}
