"""Connector interface. A connector lists documents and fetches their bytes; ingestion does the rest.

``connectors/sync.py`` drives any connector: it decides what is new, changed,
unchanged or gone, fetches only what it must, validates and ingests, and keeps
per-source state in ``connector_items``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    # Stable within the connector: a path relative to the folder, a Drive file id.
    source_id: str
    # The filename the document gets, extension included (it selects the parser).
    name: str
    content_type: str
    size_bytes: int | None = None
    modified_at: datetime | None = None
    # A source-side version, where the source has one (Drive's md5Checksum).
    version: str = ""
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


@runtime_checkable
class Connector(Protocol):
    name: str

    def list_documents(self) -> Iterator[SourceDocument]: ...
    def fetch(self, source_id: str) -> bytes: ...
