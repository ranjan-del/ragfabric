"""Compute the AccessFilter for a principal (ADR 0003).

The filter is computed once per request from grants and overrides and then
passed into every store query, so ranking only ever sees permitted rows.

Rules, applied in order:
0. A principal presenting a missing or inactive API key sees nothing.
1. Admins are unrestricted, unless the API key in use carries collection
   scopes, in which case the admin is narrowed to those collections like
   anyone else.
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

from sqlalchemy import exists, or_, select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.models.access import ApiKey, CollectionGrant, DocumentOverride
from ragfabric_core.models.document import Collection, Document


def compute_access_filter(db: Session, principal: Principal) -> AccessFilter:
    key = None
    if principal.api_key_id is not None:
        key = db.get(ApiKey, principal.api_key_id)
        if key is None or not key.is_active:
            return AccessFilter(
                document_ids=frozenset(),
                collection_ids=frozenset(),
                denied_document_ids=frozenset(),
            )

    scopes = set(key.collection_ids or []) if key is not None else set()

    if principal.role == "admin":
        if not scopes:
            return AccessFilter.unrestricted()
        in_scope_docs = set(
            db.execute(
                select(Document.id).where(Document.collection_id.in_(sorted(scopes)))
            ).scalars()
        )
        return AccessFilter(
            collection_ids=frozenset(scopes),
            document_ids=frozenset(in_scope_docs),
            denied_document_ids=frozenset(),
        )

    groups = list(principal.group_ids)

    # Collections axis, one query: a collection is allowed when a group grant
    # names one of the principal's groups, the principal owns it, or it has
    # no grants at all (open by default). The "no grants at all" check used
    # to be two unscoped reads (every grant's collection_id, then every
    # collection id, with the difference taken in Python); it is now a
    # correlated NOT EXISTS, so the database does the scoping and only the
    # already-allowed collection ids ever cross into Python.
    no_grants_at_all = ~exists(
        select(1).where(CollectionGrant.collection_id == Collection.id)
    )
    collection_conditions = [no_grants_at_all]
    if groups:
        collection_conditions.append(
            Collection.id.in_(
                select(CollectionGrant.collection_id).where(CollectionGrant.group_id.in_(groups))
            )
        )
    if principal.user_id is not None:
        collection_conditions.append(Collection.owner_id == principal.user_id)
    allowed_collections: set[int] = set(
        db.execute(select(Collection.id).where(or_(*collection_conditions))).scalars()
    )

    # Documents axis, one query: a document is allowed when it belongs to no
    # collection (open by default) or the principal owns it. This used to be
    # two reads, one of them (every uncollected document id) unscoped by the
    # principal; folding both conditions into a single OR keeps it to one
    # round trip regardless of how many documents exist.
    document_conditions = [Document.collection_id.is_(None)]
    if principal.user_id is not None:
        document_conditions.append(Document.owner_id == principal.user_id)
    allowed_documents: set[int] = set(
        db.execute(select(Document.id).where(or_(*document_conditions))).scalars()
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

    if key is not None:
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
