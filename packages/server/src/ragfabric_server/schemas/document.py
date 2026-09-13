"""Pydantic schemas for documents and collections."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    format: str
    content_type: str
    status: str
    collection_id: int | None
    owner_id: int | None
    version: int
    num_chunks: int
    error: str
    created_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""


class CollectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    owner_id: int | None
    created_at: datetime
    document_count: int = 0


class CollectionDetail(CollectionOut):
    documents: list[DocumentOut] = []
