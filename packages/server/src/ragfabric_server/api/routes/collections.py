"""Collection management routes.

Collections group related documents so search can be scoped to a subset of the
knowledge base. Deleting a collection cascades to its documents; the deleted
documents' vectors are also removed from the in-memory index.

MEMORY.md checklist:
- [x] Retriever + search: organised into collections
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ragfabric_core.db.session import get_db
from ragfabric_server.deps import get_current_user
from ragfabric_core.models.document import Collection, Document
from ragfabric_core.models.user import User
from ragfabric_server.schemas.document import CollectionCreate, CollectionDetail, CollectionOut
from ragfabric_core.store.vector_store import get_store

router = APIRouter()


def _to_out(collection: Collection, db: Session) -> CollectionOut:
    count = (
        db.query(Document).filter(Document.collection_id == collection.id).count()
    )
    return CollectionOut(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        owner_id=collection.owner_id,
        created_at=collection.created_at,
        document_count=count,
    )


@router.get("", response_model=list[CollectionOut])
def list_collections(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[CollectionOut]:
    """List all collections with their document counts."""
    collections = db.query(Collection).order_by(Collection.created_at.desc()).all()
    return [_to_out(c, db) for c in collections]


@router.post("", response_model=CollectionOut, status_code=status.HTTP_201_CREATED)
def create_collection(
    payload: CollectionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CollectionOut:
    """Create a new collection owned by the current user."""
    collection = Collection(
        name=payload.name,
        description=payload.description,
        owner_id=current_user.id,
    )
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return _to_out(collection, db)


@router.get("/{collection_id}", response_model=CollectionDetail)
def get_collection(
    collection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CollectionDetail:
    """Return a collection together with its documents."""
    collection = db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
        )
    documents = (
        db.query(Document)
        .filter(Document.collection_id == collection_id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return CollectionDetail(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        owner_id=collection.owner_id,
        created_at=collection.created_at,
        document_count=len(documents),
        documents=documents,
    )


@router.delete("/{collection_id}", status_code=status.HTTP_200_OK)
def delete_collection(
    collection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete a collection and its documents (owner or admin)."""
    collection = db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found."
        )
    if current_user.role != "admin" and collection.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You may only delete your own collections.",
        )
    document_ids = [
        d.id
        for d in db.query(Document).filter(Document.collection_id == collection_id).all()
    ]
    db.delete(collection)  # cascades to documents + chunks
    db.commit()
    store = get_store()
    for document_id in document_ids:
        store.delete_document(document_id)
    return {"detail": "Collection deleted.", "id": collection_id}
