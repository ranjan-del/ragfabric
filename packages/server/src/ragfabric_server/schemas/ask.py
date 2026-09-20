"""Request schema for POST /api/ask."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    collection_id: int | None = None
    document_id: int | None = None
    format: str | None = Field(default=None, pattern="^(pdf|docx|pptx|txt|csv|md)$")
    # Only "traditional" is servable in this phase. A pattern that accepted a
    # wider set of names would be a promise the server cannot keep; later
    # phases widen it as the strategies it names actually ship.
    strategy: str = Field(default="traditional", pattern="^(traditional)$")
    stream: bool = True
