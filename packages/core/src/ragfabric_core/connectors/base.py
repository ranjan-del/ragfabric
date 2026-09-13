"""Connector interface. A connector lists documents and fetches their bytes; ingestion does the rest."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    source_id: str
    name: str
    content_type: str
    size_bytes: int | None = None
    modified_at: datetime | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


@runtime_checkable
class Connector(Protocol):
    name: str

    def list_documents(self) -> Iterator[SourceDocument]: ...
    def fetch(self, source_id: str) -> bytes: ...
