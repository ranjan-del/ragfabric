"""Admin routes (admin role required).

Covers the admin controls from MEMORY.md: list and manage users' roles/activation,
bump document versions, and hard-delete any document regardless of owner.

MEMORY.md checklist:
- [x] Admin: upload, delete, versioning, permissions
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ragfabric_core.db.session import get_db
from ragfabric_server.deps import require_role
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS
from ragfabric_core.ingest.pipeline import reingest_document
from ragfabric_core.models.document import Document
from ragfabric_core.models.user import Role, User
from ragfabric_server.schemas.document import DocumentOut
from ragfabric_server.schemas.user import PermissionUpdate, UserOut
from ragfabric_core.store.vector_store import get_store

router = APIRouter()

_VALID_ROLES = {Role.ADMIN.value, Role.USER.value}


@router.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
) -> list[User]:
    """List all users (admin only)."""
    return db.query(User).order_by(User.created_at.desc()).all()


@router.put("/users/{user_id}/permissions", response_model=UserOut)
def set_permissions(
    user_id: int,
    payload: PermissionUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
) -> User:
    """Update a user's role and/or active status (admin only)."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )
    if payload.role is not None:
        if payload.role not in _VALID_ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Allowed: {', '.join(sorted(_VALID_ROLES))}.",
            )
        user.role = payload.role
    if payload.is_active is not None:
        user.is_active = payload.is_active
    db.commit()
    db.refresh(user)
    return user


@router.post("/documents/{document_id}/versions", response_model=DocumentOut)
async def create_version(
    document_id: int,
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
) -> Document:
    """Publish a new version of a document (admin only).

    With a ``file``, the document's content is REPLACED: the old chunks are
    dropped from the database and the vector index, the new file is re-ingested
    under the same document id, and the version counter advances. Keeping the id
    means collection membership and any stored citation still resolve.

    Without a file, this only advances the counter, which is the "mark a
    reviewed revision" case where the content has not changed.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found."
        )

    if file is None:
        document.version += 1
        db.commit()
        db.refresh(document)
        return document

    filename = file.filename or document.filename
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported format '{ext or filename}'. "
                f"Supported: {', '.join(SUPPORTED_FORMATS)}."
            ),
        )
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )

    return reingest_document(
        db,
        document,
        filename=filename,
        data=data,
        content_type=file.content_type or "",
    )


@router.post("/index/rebuild", status_code=status.HTTP_200_OK)
def rebuild_index(
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
) -> dict:
    """Rebuild the in-memory vector index from the persisted chunk rows.

    The same call the application makes on startup, exposed so an operator can
    repair drift without a restart. ``/api/analytics/overview`` reports
    ``chunks`` (from SQL) next to ``indexed_vectors`` (from memory); when those
    two disagree, this is the fix.
    """
    restored = get_store().rebuild_from_db(db)
    return {"detail": "Vector index rebuilt.", "vectors": restored}


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
def admin_delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
) -> dict:
    """Hard-delete any document and its vectors (admin override)."""
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found."
        )
    db.delete(document)
    db.commit()
    get_store().delete_document(document_id)
    return {"detail": "Document deleted.", "id": document_id}
