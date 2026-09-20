"""Pydantic mirrors of the server's response schemas.

Each class here shadows one schema in
``packages/server/src/ragfabric_server/schemas/``. Field names and types are
copied from the real schema modules, not from the plan that first described
this client, because the schemas are the source of truth and drift between
the two is exactly what these models exist to catch early (a validation
error at parse time, not a confusing ``KeyError`` deep in caller code).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Highlight(BaseModel):
    """A query-term span, in ``schemas/search.py``."""

    term: str
    start: int
    end: int


class Span(BaseModel):
    """A character range plus the text it covers, in ``schemas/search.py``."""

    text: str
    start: int
    end: int


class Citation(BaseModel):
    """Mirrors ``schemas.search.Citation``."""

    marker: str
    chunk_id: int | None = None
    document_id: int | None = None
    filename: str | None = None
    page: int | None = None
    score: float
    snippet: str
    used: bool = False
    highlights: list[Highlight] = Field(default_factory=list)
    supporting_span: Span | None = None


class SourceDocument(BaseModel):
    """Mirrors ``schemas.search.SourceDocument``."""

    document_id: int | None = None
    filename: str | None = None
    page: int | None = None
    collection_id: int | None = None


class Answer(BaseModel):
    """Mirrors ``schemas.search.AnswerResponse``, returned by both
    ``POST /api/ask`` (non streaming) and ``POST /api/search/query``."""

    question: str
    answer: str
    confidence: float
    citations: list[Citation]
    highlights: list[Highlight]
    source_document: SourceDocument | None = None


class SearchResult(BaseModel):
    """Mirrors ``schemas.search.SearchResultItem``."""

    chunk_id: int | None = None
    document_id: int | None = None
    filename: str | None = None
    format: str | None = None
    page: int | None = None
    chunk_index: int | None = None
    score: float
    lexical_score: float | None = None
    hybrid_score: float | None = None
    text: str


class Document(BaseModel):
    """Mirrors ``schemas.document.DocumentOut``, returned by
    ``POST /api/documents/upload``, ``GET /api/documents`` and
    ``GET /api/documents/{id}``."""

    id: int
    filename: str
    format: str
    document_type: str = ""
    storage_path: str | None = None
    content_type: str
    status: str
    collection_id: int | None = None
    owner_id: int | None = None
    version: int
    num_chunks: int
    error: str
    created_at: datetime


class Source(BaseModel):
    """Mirrors ``schemas.runs.SourceOut``, one row of ``Run.sources``."""

    rank: int
    chunk_id: int | None = None
    document_id: int | None = None
    score: float | None = None
    cited: bool
    page: int | None = None


class Run(BaseModel):
    """Mirrors ``schemas.runs.RunOut``, returned by ``GET /api/runs/{id}``."""

    id: int
    question: str
    mode: str
    requested_strategy: str | None = None
    selected_strategy: str
    fallback_from: str | None = None
    answer: str | None = None
    latency_ms: int
    retrieval_latency_ms: int
    generation_latency_ms: int
    llm_calls: int
    retrieval_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None = None
    llm_model: str | None = None
    embedding_model: str | None = None
    trace: list[dict] = Field(default_factory=list)
    created_at: datetime
    sources: list[Source] = Field(default_factory=list)


class AskEvent(BaseModel):
    """One SSE event from ``POST /api/ask`` with ``stream: true``.

    ``event`` is one of ``retrieval``, ``token``, ``superseded``,
    ``citations`` or ``done``, in that order (``superseded`` only appears
    when the streamed answer failed the citation contract). ``data`` is the
    event's raw JSON payload, kept untyped because each event name carries a
    different shape and this class is a thin parsing result, not a schema.
    """

    event: str
    data: dict
