"""Principal and AccessFilter.

A Principal is whoever made the request: a signed in user or an API key. An
AccessFilter is the set of documents that principal may read, computed once per
request and handed to every store query, so ranking only ever sees permitted
chunks (ADR 0003). Phase 1 defines the types; Phase 2 computes real filters
from groups, collection grants and document overrides. Until then every caller
uses ``AccessFilter.unrestricted()``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: int | None
    email: str | None
    role: str = "user"
    group_ids: list[int] = Field(default_factory=list)
    api_key_id: int | None = None


class AccessFilter(BaseModel):
    """``None`` for a field means "no restriction on that axis".

    A chunk is allowed when its document is explicitly permitted, or its
    collection is permitted, or neither axis is restricted, unless its
    document is on the deny list, which is checked first and wins over
    every allow axis. Phase 2 computes ``document_ids``, ``collection_ids``
    and ``denied_document_ids`` from a principal's groups, collection
    grants and document overrides; Phase 1 only defines the shape.
    """

    model_config = ConfigDict(frozen=True)

    document_ids: frozenset[int] | None = None
    collection_ids: frozenset[int] | None = None
    denied_document_ids: frozenset[int] | None = None

    @classmethod
    def unrestricted(cls) -> AccessFilter:
        return cls(document_ids=None, collection_ids=None, denied_document_ids=None)

    @property
    def is_unrestricted(self) -> bool:
        return (
            self.document_ids is None
            and self.collection_ids is None
            and self.denied_document_ids is None
        )

    def allows(self, document_id: int | None, collection_id: int | None) -> bool:
        if self.denied_document_ids is not None and document_id in self.denied_document_ids:
            return False
        if self.document_ids is None and self.collection_ids is None:
            return True
        if self.document_ids is not None and document_id in self.document_ids:
            return True
        if self.collection_ids is not None and collection_id in self.collection_ids:
            return True
        return False
