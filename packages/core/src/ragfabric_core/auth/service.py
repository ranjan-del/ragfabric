"""Groups, membership, collection grants and document overrides."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
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
    """Create or update the (group, collection) grant atomically.

    This used to be read-then-write: select the existing row, then either add
    a new one or update the one found. Two callers racing to grant the same
    pair could both see no row, both insert, and one of them would fail on
    ``uq_grant_group_collection`` (or, without the unique constraint, would
    have produced two rows for the same pair). A single dialect-appropriate
    upsert on that constraint closes the gap: the insert and the "does a row
    already exist" check happen as one statement the database itself
    serialises, not two round trips this process could be interrupted between.
    """
    if permission not in VALID_GRANTS:
        raise ValueError(f"permission must be one of {VALID_GRANTS}")
    insert = pg_insert if db.bind.dialect.name == "postgresql" else sqlite_insert
    stmt = insert(CollectionGrant).values(
        group_id=group_id, collection_id=collection_id, permission=permission
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[CollectionGrant.group_id, CollectionGrant.collection_id],
        set_={"permission": stmt.excluded.permission},
    )
    db.execute(stmt)
    db.flush()
    return db.execute(
        select(CollectionGrant).where(
            CollectionGrant.group_id == group_id, CollectionGrant.collection_id == collection_id
        )
    ).scalar_one()


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
