"""Admin API for groups, grants, overrides and API keys."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ragfabric_core.auth import service
from ragfabric_core.auth.api_keys import create_api_key
from ragfabric_core.db.session import get_db
from ragfabric_core.models.access import ApiKey, CollectionGrant, DocumentOverride, Group
from ragfabric_core.models.document import Collection, Document
from ragfabric_core.models.user import User
from ragfabric_server.deps import require_role
from ragfabric_server.schemas.access import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyOut,
    GrantCreate,
    GrantOut,
    GroupCreate,
    GroupOut,
    MemberAdd,
    OverrideCreate,
    OverrideOut,
)

router = APIRouter(dependencies=[Depends(require_role("admin"))])


def _get_or_404(db: Session, model, ident, name: str):
    obj = db.get(model, ident)
    if obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{name} not found.")
    return obj


@router.post("/groups", response_model=GroupOut, status_code=status.HTTP_201_CREATED)
def create_group(payload: GroupCreate, db: Session = Depends(get_db)) -> Group:
    if db.query(Group).filter(Group.name == payload.name).first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Group name already exists."
        )
    group = service.create_group(db, payload.name, payload.description)
    db.commit()
    db.refresh(group)
    return group


@router.get("/groups", response_model=list[GroupOut])
def list_groups(db: Session = Depends(get_db)) -> list[Group]:
    return db.query(Group).order_by(Group.name).all()


@router.post("/groups/{group_id}/members", status_code=status.HTTP_200_OK)
def add_member(group_id: int, payload: MemberAdd, db: Session = Depends(get_db)) -> dict:
    _get_or_404(db, Group, group_id, "Group")
    _get_or_404(db, User, payload.user_id, "User")
    service.add_member(db, group_id, payload.user_id)
    db.commit()
    return {"detail": "Member added.", "group_id": group_id, "user_id": payload.user_id}


@router.delete("/groups/{group_id}/members/{user_id}", status_code=status.HTTP_200_OK)
def remove_member(group_id: int, user_id: int, db: Session = Depends(get_db)) -> dict:
    service.remove_member(db, group_id, user_id)
    db.commit()
    return {"detail": "Member removed.", "group_id": group_id, "user_id": user_id}


@router.post("/grants", response_model=GrantOut, status_code=status.HTTP_201_CREATED)
def create_grant(payload: GrantCreate, db: Session = Depends(get_db)) -> CollectionGrant:
    _get_or_404(db, Group, payload.group_id, "Group")
    _get_or_404(db, Collection, payload.collection_id, "Collection")
    grant = service.grant_collection(
        db, payload.group_id, payload.collection_id, payload.permission
    )
    db.commit()
    db.refresh(grant)
    return grant


@router.get("/grants", response_model=list[GrantOut])
def list_grants(db: Session = Depends(get_db)) -> list[CollectionGrant]:
    return db.query(CollectionGrant).order_by(CollectionGrant.id).all()


@router.delete("/grants/{grant_id}", status_code=status.HTTP_200_OK)
def delete_grant(grant_id: int, db: Session = Depends(get_db)) -> dict:
    grant = _get_or_404(db, CollectionGrant, grant_id, "Grant")
    db.delete(grant)
    db.commit()
    return {"detail": "Grant removed.", "id": grant_id}


@router.post("/overrides", response_model=OverrideOut, status_code=status.HTTP_201_CREATED)
def create_override(payload: OverrideCreate, db: Session = Depends(get_db)) -> DocumentOverride:
    _get_or_404(db, Document, payload.document_id, "Document")
    try:
        override = service.set_document_override(
            db,
            payload.document_id,
            group_id=payload.group_id,
            user_id=payload.user_id,
            permission=payload.permission,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    db.commit()
    db.refresh(override)
    return override


@router.get("/overrides", response_model=list[OverrideOut])
def list_overrides(db: Session = Depends(get_db)) -> list[DocumentOverride]:
    return db.query(DocumentOverride).order_by(DocumentOverride.id).all()


@router.post("/keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
def create_key(payload: ApiKeyCreate, db: Session = Depends(get_db)) -> ApiKeyCreated:
    _get_or_404(db, User, payload.user_id, "User")
    key, plaintext = create_api_key(
        db,
        name=payload.name,
        user_id=payload.user_id,
        collection_ids=payload.collection_ids,
        strategies=payload.strategies,
        rate_limit_per_minute=payload.rate_limit_per_minute,
        expires_at=payload.expires_at,
    )
    db.commit()
    db.refresh(key)
    return ApiKeyCreated(**ApiKeyOut.model_validate(key).model_dump(), key=plaintext)


@router.get("/keys", response_model=list[ApiKeyOut])
def list_keys(db: Session = Depends(get_db)) -> list[ApiKey]:
    return db.query(ApiKey).order_by(ApiKey.id).all()


@router.delete("/keys/{key_id}", status_code=status.HTTP_200_OK)
def revoke_key(key_id: int, db: Session = Depends(get_db)) -> dict:
    key = _get_or_404(db, ApiKey, key_id, "API key")
    key.is_active = False
    db.commit()
    return {"detail": "API key revoked.", "id": key_id}
