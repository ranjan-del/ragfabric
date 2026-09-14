"""Translate an AccessFilter into a SQL predicate applied inside the store query (ADR 0003)."""

from __future__ import annotations

from sqlalchemy import and_, false, not_, or_

from ragfabric_core.auth.principal import AccessFilter


def access_clause(access: AccessFilter, document_col, collection_col):
    if access.is_unrestricted:
        return None
    parts = []
    if access.document_ids is None and access.collection_ids is None:
        allow = None
    else:
        allows = []
        if access.document_ids is not None:
            allows.append(
                document_col.in_(sorted(access.document_ids)) if access.document_ids else false()
            )
        if access.collection_ids is not None:
            allows.append(
                collection_col.in_(sorted(access.collection_ids))
                if access.collection_ids
                else false()
            )
        allow = or_(*allows) if len(allows) > 1 else allows[0]
    if allow is not None:
        parts.append(allow)
    if access.denied_document_ids:
        parts.append(not_(document_col.in_(sorted(access.denied_document_ids))))
    return and_(*parts) if len(parts) > 1 else (parts[0] if parts else None)
