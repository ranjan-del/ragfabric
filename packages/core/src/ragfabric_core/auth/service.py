"""Groups, membership, collection grants and document overrides."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragfabric_core.models.access import CollectionGrant, DocumentOverride, Group, GroupMember

VALID_GRANTS = ("read", "write")
VALID_OVERRIDES = ("deny", "read")


def create_group(db: Session, name: str, description: str = "") -> Group:
    group = Group(name=name, description=description)
    db.add(group)
    db.flush()
    return group


def add_member(db: Session, group_id: int, user_id: int) -> None:
    if db.get(GroupMember, (group_id, user_id)) is None:
        db.add(GroupMember(group_id=group_id, user_id=user_id))
        db.flush()


def remove_member(db: Session, group_id: int, user_id: int) -> None:
    member = db.get(GroupMember, (group_id, user_id))
    if member is not None:
        db.delete(member)
        db.flush()


def grant_collection(
    db: Session, group_id: int, collection_id: int, permission: str = "read"
) -> CollectionGrant:
    if permission not in VALID_GRANTS:
        raise ValueError(f"permission must be one of {VALID_GRANTS}")
    grant = db.execute(
        select(CollectionGrant).where(
            CollectionGrant.group_id == group_id, CollectionGrant.collection_id == collection_id
        )
    ).scalar_one_or_none()
    if grant is None:
        grant = CollectionGrant(
            group_id=group_id, collection_id=collection_id, permission=permission
        )
        db.add(grant)
    else:
        grant.permission = permission
    db.flush()
    return grant


def revoke_collection(db: Session, group_id: int, collection_id: int) -> None:
    grant = db.execute(
        select(CollectionGrant).where(
            CollectionGrant.group_id == group_id, CollectionGrant.collection_id == collection_id
        )
    ).scalar_one_or_none()
    if grant is not None:
        db.delete(grant)
        db.flush()


def set_document_override(
    db: Session,
    document_id: int,
    *,
    group_id: int | None = None,
    user_id: int | None = None,
    permission: str = "deny",
) -> DocumentOverride:
    if permission not in VALID_OVERRIDES:
        raise ValueError(f"permission must be one of {VALID_OVERRIDES}")
    if (group_id is None) == (user_id is None):
        raise ValueError("exactly one of group_id or user_id is required")
    override = DocumentOverride(
        document_id=document_id, group_id=group_id, user_id=user_id, permission=permission
    )
    db.add(override)
    db.flush()
    return override


def group_ids_for_user(db: Session, user_id: int) -> list[int]:
    return list(
        db.execute(select(GroupMember.group_id).where(GroupMember.user_id == user_id)).scalars()
    )
