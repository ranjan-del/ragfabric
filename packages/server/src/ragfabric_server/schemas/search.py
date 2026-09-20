"""Pydantic schemas for search and cited answers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    # Out of range is rejected (422 via this Field's bounds), never clamped:
    # a caller who asks for a 0.99 floor and silently gets 0.0 back would draw
    # false conclusions from the results.
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    # None means "use whatever the shared strategy is configured with" (no
    # override, no per-request strategy built). Naming a value, including
    # "none", asks for a strategy built fresh around that reranker instead of
    # the shared registry instance; see api/routes/search.py:_strategy_for.
    rerank: Literal["none", "llm", "cross_encoder"] | None = None
    # Metadata filters. All are ANDed together and applied before ranking.
    collection_id: int | None = None
    document_id: int | None = None
    format: str | None = Field(
        default=None, pattern="^(pdf|docx|pptx|txt|csv|md)$", description="File-type filter"
    )
    mode: str = Field(default="semantic", pattern="^(semantic|hybrid)$")


class SearchResultItem(BaseModel):
    chunk_id: int | None = None
    document_id: int | None = None
    filename: str | None = None
    format: str | None = None
    page: int | None = None
    chunk_index: int | None = None
    score: float
    # Only populated in hybrid mode; exposed so the ranking is inspectable
    # rather than a black box.
    lexical_score: float | None = None
    hybrid_score: float | None = None
    text: str


class SearchResults(BaseModel):
    query: str
    mode: str
    results: list[SearchResultItem]


class Highlight(BaseModel):
    term: str
    start: int
    end: int


class Span(BaseModel):
    """A character range plus the text it covers, for the UI to mark up."""

    text: str
    start: int
    end: int


class Citation(BaseModel):
    marker: str
    chunk_id: int | None = None
    document_id: int | None = None
    filename: str | None = None
    page: int | None = None
    score: float
    snippet: str
    # Whether the answer text actually carries this marker. Retrieval returns
    # top_k chunks but the answer may only lean on some of them.
    used: bool = False
    # Query-term spans within ``snippet`` (not the full chunk).
    highlights: list[Highlight] = []
    # The sentence the answer quoted from this chunk, also relative to
    # ``snippet``. None when the chunk was retrieved but not quoted.
    supporting_span: Span | None = None


class SourceDocument(BaseModel):
    document_id: int | None = None
    filename: str | None = None
    page: int | None = None
    collection_id: int | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    citations: list[Citation]
    # Query-term spans within ``answer``. Every offset in this response indexes
    # into a string the response also carries, so a client can render the
    # highlight without guessing at match positions itself.
    highlights: list[Highlight]
    source_document: SourceDocument | None = None
