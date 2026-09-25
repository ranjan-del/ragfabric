"""Admin routes (admin role required).

Covers the admin controls from MEMORY.md: list and manage users' roles/activation,
bump document versions, and hard-delete any document regardless of owner.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ragfabric_core.db.session import get_db
from ragfabric_core.graph.extract import detach_documents
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS
from ragfabric_core.ingest.pipeline import reingest_document
from ragfabric_core.ingest.storage import get_storage
from ragfabric_core.models.access import ApiKey, AuditLog
from ragfabric_core.models.document import Collection, Document, QueryLog
from ragfabric_core.models.runs import Conversation, RetrievalRun
from ragfabric_core.models.user import Role, User
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.security import hash_password
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store
from ragfabric_server.deps import require_role
from ragfabric_server.schemas.document import DocumentOut
from ragfabric_server.schemas.user import AdminUserCreate, PermissionUpdate, UserOut

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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
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


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: AdminUserCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
) -> User:
    """Create a user outright (admin only).

    ``/api/auth/register`` exists for self service and hardcodes the ``user``
    role, which is what stops anyone signing themselves up as an admin. This
    route is behind the admin guard, so it may name the role and the initial
    active state, and it is the reason an operator no longer has to insert a
    row by hand to onboard somebody.
    """
    if payload.role not in _VALID_ROLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Allowed: {', '.join(sorted(_VALID_ROLES))}.",
        )
    email = payload.email.lower()
    if db.query(User).filter(User.email == email).first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with that email already exists.",
        )
    user = User(
        email=email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        is_active=payload.is_active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_200_OK)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
) -> dict:
    """Delete a user, settling every row that references them (admin only).

    Nine tables carry a foreign key to ``users.id`` and each one is decided
    here rather than left to whatever the database does by default. Two of
    them already declare ``ON DELETE CASCADE`` on the column and are left to
    it; the other seven are nullable with no cascade, which on PostgreSQL
    (and on SQLite, where this project turns foreign keys on) means the
    delete is REFUSED until they are settled. They are settled as follows.

    Cascade, because the row means nothing without the user:

    - ``group_members.user_id``: a membership of a deleted user is not a fact
      about anything.
    - ``document_overrides.user_id``: a per user grant or denial likewise.

    Revoke and detach, because the row must not keep working but must not
    vanish either:

    - ``api_keys.principal_user_id``: the key is deactivated AND detached.
      Deactivating alone would leave a live foreign key and the delete would
      fail; deleting the key instead would break ``audit_log.api_key_id`` and
      erase the record of what that key did. A deactivated, detached key
      authenticates nobody and still anchors its own audit trail.

    Preserve and anonymise, because the history is the product:

    - ``audit_log.principal_user_id``, ``retrieval_runs.user_id``,
      ``query_logs.user_id``, ``conversations.user_id``: measurement and
      audit history outlives the account. The rows survive with the
      reference cleared.

    Preserve and disown, because the content belongs to the organisation:

    - ``collections.owner_id``, ``documents.owner_id``: deleting a person
      must never delete the corpus. Ownership is cleared and an admin can
      reassign it.

    Refusing to delete yourself is not a cascade decision, it is an
    availability one: an admin who deletes their own account can lock the
    last administrator out of the console.
    """
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own account.",
        )

    db.query(ApiKey).filter(ApiKey.principal_user_id == user_id).update(
        {"is_active": False, "principal_user_id": None}, synchronize_session=False
    )
    for model, column in (
        (AuditLog, AuditLog.principal_user_id),
        (RetrievalRun, RetrievalRun.user_id),
        (Conversation, Conversation.user_id),
        (QueryLog, QueryLog.user_id),
        (Collection, Collection.owner_id),
        (Document, Document.owner_id),
    ):
        db.query(model).filter(column == user_id).update(
            {column.key: None}, synchronize_session=False
        )

    db.delete(user)
    db.commit()
    return {"detail": "User deleted.", "id": user_id}


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

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


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
def admin_delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
) -> dict:
    """Hard-delete any document and its vectors (admin override)."""
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    detach_documents(db, [document.id])  # recompute the graph first (R42)
    db.delete(document)  # cascades to chunks
    db.commit()
    # As with the owner-facing delete route, the relational cascade does not
    # reach the configured vector/lexical stores (SQLite does not enforce the
    # chunk_embeddings/chunk_search foreign keys, and Chroma has none at all),
    # so those rows must be dropped explicitly or they accumulate as orphans.
    cfg = get_config()
    sf = get_session_factory()
    build_vector_store(cfg.vector_store, sf).delete_document(document_id)
    build_lexical_store(cfg.lexical_store, sf).delete_document(document_id)
    get_storage().delete(document_id)
    return {"detail": "Document deleted.", "id": document_id}
