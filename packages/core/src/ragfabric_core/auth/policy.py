"""Compute the AccessFilter for a principal (ADR 0003).

The filter is computed once per request from grants and overrides and then
passed into every store query, so ranking only ever sees permitted rows.

Rules, applied in order:
1. Admins are unrestricted.
2. A collection is readable when one of the principal's groups holds a grant
   on it, when the principal owns it, or when it has no grants at all. The
   last rule keeps v1 behaviour (everyone sees everything) until an admin
   adds the first grant to a collection, at which point that collection is
   restricted to its grantees.
3. A document is readable when the principal owns it, when a read override
   names the principal or one of their groups, or when it belongs to no
   collection.
4. A deny override for the principal or one of their groups always wins.
5. An API key with collection scopes narrows everything above to those
   collections.
"""

from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.models.access import ApiKey, CollectionGrant, DocumentOverride
from ragfabric_core.models.document import Collection, Document


def compute_access_filter(db: Session, principal: Principal) -> AccessFilter:
    if principal.role == "admin":
        return AccessFilter.unrestricted()

    groups = list(principal.group_ids)
    granted_ids = set(db.execute(select(CollectionGrant.collection_id).distinct()).scalars())
    allowed_collections: set[int] = set()
    if groups:
        allowed_collections |= set(
            db.execute(
                select(CollectionGrant.collection_id).where(CollectionGrant.group_id.in_(groups))
            ).scalars()
        )
    if principal.user_id is not None:
        allowed_collections |= set(
            db.execute(
                select(Collection.id).where(Collection.owner_id == principal.user_id)
            ).scalars()
        )
    all_collections = set(db.execute(select(Collection.id)).scalars())
    allowed_collections |= all_collections - granted_ids

    allowed_documents: set[int] = set(
        db.execute(select(Document.id).where(Document.collection_id.is_(None))).scalars()
    )
    if principal.user_id is not None:
        allowed_documents |= set(
            db.execute(select(Document.id).where(Document.owner_id == principal.user_id)).scalars()
        )

    override_filter = []
    if principal.user_id is not None:
        override_filter.append(DocumentOverride.user_id == principal.user_id)
    if groups:
        override_filter.append(DocumentOverride.group_id.in_(groups))
    denied: set[int] = set()
    if override_filter:
        rows = db.execute(select(DocumentOverride).where(or_(*override_filter))).scalars().all()
        for row in rows:
            if row.permission == "deny":
                denied.add(row.document_id)
            elif row.permission == "read":
                allowed_documents.add(row.document_id)

    if principal.api_key_id is not None:
        key = db.get(ApiKey, principal.api_key_id)
        scopes = set(key.collection_ids or []) if key is not None else set()
        if scopes:
            allowed_collections &= scopes
            in_scope_docs = set(
                db.execute(
                    select(Document.id).where(Document.collection_id.in_(sorted(scopes)))
                ).scalars()
            )
            allowed_documents &= in_scope_docs

    return AccessFilter(
        document_ids=frozenset(allowed_documents - denied),
        collection_ids=frozenset(allowed_collections),
        denied_document_ids=frozenset(denied),
    )
