"""Translate an AccessFilter into a SQL predicate applied inside the store query (ADR 0003)."""

from __future__ import annotations

from collections.abc import Collection

from sqlalchemy import and_, false, not_, or_

from ragfabric_core.auth.principal import AccessFilter


def access_clause(
    access: AccessFilter,
    document_col,
    collection_col,
    collection_ids: Collection[int] | None = None,
):
    """The access predicate, narrowed to ``collection_ids`` when given (ruling R25).

    ``collection_ids`` is a request scope, not a permission: it only narrows,
    it is ANDed onto whatever ``access`` alone would admit, and it is never
    folded into ``access.collection_ids``, which is an allow axis ORed with
    ``access.document_ids`` and would widen instead of narrow. ``None`` or
    empty means the request asked for no scoping, so the access predicate
    alone decides.
    """
    if access.is_unrestricted:
        clause = None
    else:
        parts = []
        if access.document_ids is None and access.collection_ids is None:
            allow = None
        else:
            allows = []
            if access.document_ids is not None:
                allows.append(
                    document_col.in_(sorted(access.document_ids))
                    if access.document_ids
                    else false()
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
        clause = and_(*parts) if len(parts) > 1 else (parts[0] if parts else None)

    if collection_ids:
        scope = collection_col.in_(sorted(collection_ids))
        clause = scope if clause is None else and_(clause, scope)
    return clause
