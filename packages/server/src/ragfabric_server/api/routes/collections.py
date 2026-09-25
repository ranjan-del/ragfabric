"""Collection management routes.

Collections group related documents so search can be scoped to a subset of the
knowledge base. Deleting a collection cascades to its documents (and their
chunks) at the database level, and every one of those documents is also
dropped from the configured vector and lexical stores explicitly, since that
cascade does not reach them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ragfabric_core.db.session import get_db
from ragfabric_core.graph.extract import detach_documents
from ragfabric_core.models.document import Collection, Document
from ragfabric_core.models.user import User
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store
from ragfabric_server.deps import get_current_user
from ragfabric_server.schemas.document import (
    CollectionCreate,
    CollectionDetail,
    CollectionOut,
    CollectionUpdate,
)

router = APIRouter()


def _to_out(collection: Collection, db: Session) -> CollectionOut:
    count = db.query(Document).filter(Document.collection_id == collection.id).count()
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")
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


@router.put("/{collection_id}", response_model=CollectionOut)
def update_collection(
    collection_id: int,
    payload: CollectionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CollectionOut:
    """Rename a collection or change its description (owner or admin).

    A collection could be created and deleted but never edited, so fixing a
    typo in a name meant deleting the collection, which cascades to every
    document in it. The permission check is the same one delete uses: owner
    or admin, and nobody else.
    """
    collection = db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")
    if current_user.role != "admin" and collection.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You may only edit your own collections.",
        )
    if payload.name is not None:
        collection.name = payload.name
    if payload.description is not None:
        collection.description = payload.description
    db.commit()
    db.refresh(collection)
    return _to_out(collection, db)


@router.delete("/{collection_id}", status_code=status.HTTP_200_OK)
def delete_collection(
    collection_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete a collection and its documents (owner or admin)."""
    collection = db.get(Collection, collection_id)
    if collection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found.")
    if current_user.role != "admin" and collection.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You may only delete your own collections.",
        )
    document_ids = [
        row[0]
        for row in db.query(Document.id).filter(Document.collection_id == collection_id).all()
    ]
    detach_documents(db, document_ids)  # recompute the graph first (R42)
    db.delete(collection)  # cascades to documents + chunks
    db.commit()
    # The cascade above only reaches the relational `documents`/`chunks` rows.
    # `chunk_embeddings` and `chunk_search` are not joined through a foreign
    # key that every dialect enforces (SQLite does not), and a Chroma vector
    # store has no foreign key relationship to the database at all, so every
    # document the collection carried must be dropped from the configured
    # vector and lexical stores explicitly, the same as a single document
    # delete does.
    if document_ids:
        cfg = get_config()
        sf = get_session_factory()
        vector_store = build_vector_store(cfg.vector_store, sf)
        lexical_store = build_lexical_store(cfg.lexical_store, sf)
        for document_id in document_ids:
            vector_store.delete_document(document_id)
            lexical_store.delete_document(document_id)
    return {"detail": "Collection deleted.", "id": collection_id}
