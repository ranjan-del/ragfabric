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
    # None for the graph strategy, whose chunks come from a traversal and
    # carry no similarity score.
    score: float | None = None
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


class Usage(BaseModel):
    """Mirrors ``schemas.search.Usage``: what the request actually cost.

    ``embedding_calls`` is zero for the vectorless strategy, which makes none,
    and one for the traditional strategy, which embeds the query exactly once.
    """

    embedding_calls: int = 0
    llm_calls: int = 0
    retrieval_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class SubQuestionReport(BaseModel):
    """Mirrors ``schemas.search.SubQuestionReportOut``.

    Only the agentic strategy fills this in. ``status`` is ``answered``,
    ``abandoned`` or ``open``, where ``open`` means the run stopped with this
    part of the question unanswered, and ``reason`` says what stopped it. It
    replaces what a single "this answer is partial" flag could not express:
    which part failed, and why.
    """

    text: str
    status: str
    reason: str | None = None
    chunk_ids: list[int] = Field(default_factory=list)


class DroppedClaim(BaseModel):
    """Mirrors ``schemas.search.DroppedClaimOut``: a claim the contract refused.

    The agent removes an unsupported claim rather than retrying retrieval for
    it, and reports the removal here so the caller can see what was cut.
    """

    text: str
    reason: str


class DatedSource(BaseModel):
    """Mirrors ``schemas.search.DatedSourceOut``."""

    marker: int
    chunk_id: int
    document_id: int
    effective_date: str


class DatedSubQuestion(BaseModel):
    """Mirrors ``schemas.search.DatedSubQuestionOut``.

    Sources for one part of the question carrying different effective dates.
    This is metadata: nothing here says they disagree in meaning.
    """

    sub_question: str
    sources: list[DatedSource] = Field(default_factory=list)


class GraphNode(BaseModel):
    """Mirrors ``schemas.search.GraphNodeOut``: an entity the graph walk reached.

    Names and types only; the server never sends a stored description.
    ``entity_type`` stays a plain string so a type added on the server does not
    make this client reject the response.
    """

    id: int
    name: str
    entity_type: str
    depth: int


class GraphEdge(BaseModel):
    """Mirrors ``schemas.search.GraphEdgeOut``: an edge the walk kept.

    ``walked_as`` is the relation as walked (the inverse name when
    ``reversed``). ``confidence`` is None when nothing measured it.
    """

    id: int
    source_id: int
    target_id: int
    relation_type: str
    walked_as: str
    reversed: bool
    confidence: float | None = None
    source_chunk_ids: list[int] = Field(default_factory=list)


class Subgraph(BaseModel):
    """Mirrors ``schemas.search.SubgraphOut``: what the graph strategy walked.

    ``empty_reason`` is ``no_graph_coverage``, ``no_entity_matched`` or
    ``no_walkable_edges`` when nothing was walked, and None otherwise.
    ``truncated`` is True when the server's node budget cut the walk.
    """

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    truncated: bool = False
    empty_reason: str | None = None


class Answer(BaseModel):
    """Mirrors ``schemas.search.AnswerResponse``, returned by both
    ``POST /api/ask`` (non streaming) and ``POST /api/search/query``."""

    question: str
    answer: str
    confidence: float
    citations: list[Citation]
    highlights: list[Highlight]
    source_document: SourceDocument | None = None
    # Optional so this client still parses a response from a server older
    # than Phase 4, which has no usage block to report.
    usage: Usage | None = None
    # Empty unless the agentic strategy served the request, and empty rather
    # than optional so a server older than Phase 5 parses without special
    # cases. ``trace`` stays a list of plain dicts for the same reason
    # ``Run.trace`` does: a span's attributes differ per node and typing them
    # would make the client reject a span shape it simply has not seen.
    sub_questions: list[SubQuestionReport] = Field(default_factory=list)
    dropped_claims: list[DroppedClaim] = Field(default_factory=list)
    dated_sources: list[DatedSubQuestion] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    # Only the graph strategy fills these in; optional and empty by default so
    # a server older than Phase 6 parses unchanged. ``dropped_relationship_claims``
    # are relationship claims the graph citation contract removed, with the
    # rule that removed each one as ``reason``.
    subgraph: Subgraph | None = None
    dropped_relationship_claims: list[DroppedClaim] = Field(default_factory=list)


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
    when the streamed answer failed the citation contract). With the graph
    strategy, ``retrieval`` carries ``subgraph`` (a ``Subgraph`` as a dict),
    and ``superseded`` carries ``dropped_claims`` and
    ``dropped_relationship_claims`` alongside the repaired ``text``. ``data`` is the
    event's raw JSON payload, kept untyped because each event name carries a
    different shape and this class is a thin parsing result, not a schema.
    """

    event: str
    data: dict
