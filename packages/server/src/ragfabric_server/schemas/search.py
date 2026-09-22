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
    # Which retrieval strategy serves this request. A Literal rather than a
    # free string so an unknown name is a 422 from validation rather than a
    # KeyError inside the registry, which would surface as a 500.
    # /api/search/hybrid rejects anything but "traditional": see
    # api/routes/search.py:hybrid_search for why.
    strategy: Literal["traditional", "vectorless", "agentic"] = "traditional"


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
    # Which strategy actually produced these results. Reported rather than
    # assumed, so a caller comparing strategies can tell from the response
    # alone which one it is looking at.
    strategy: str = "traditional"
    results: list[SearchResultItem]


class Usage(BaseModel):
    """What this request actually cost, in calls and tokens.

    Every field is a count of something that happened (ADR 0004).
    ``embedding_calls`` is the reason this block exists: the vectorless
    strategy makes none, and "zero" has to be a reported measurement rather
    than a field the response simply omits.
    """

    embedding_calls: int = 0
    llm_calls: int = 0
    retrieval_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


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


class SubQuestionReportOut(BaseModel):
    """Mirrors ``ragfabric_core.strategies.base.SubQuestionReport``.

    Only the agentic strategy fills this in, because only the agentic strategy
    decomposes a question. ``status`` is the agent ledger's own word:
    ``answered``, ``abandoned``, or ``open`` meaning the run stopped with this
    part unanswered. ``reason`` is set for everything except an answered part,
    so a caller is never shown a gap with nothing explaining it.
    """

    text: str
    status: str
    reason: str | None = None
    chunk_ids: list[int] = []


class DroppedClaimOut(BaseModel):
    """A claim removed from the answer because the citation contract refused it.

    Reported rather than silently deleted. A caller who can see what was cut,
    and which rule cut it, can tell a model that overreached from a retrieval
    that came up short; a caller shown only the surviving text cannot.
    """

    text: str
    reason: str


class DatedSourceOut(BaseModel):
    marker: int
    chunk_id: int
    document_id: int
    effective_date: str


class DatedSubQuestionOut(BaseModel):
    """Sources for one part of the question that carry different effective dates.

    Metadata, not a judgement: nothing here says the sources disagree in
    meaning or which one is right.
    """

    sub_question: str
    sources: list[DatedSourceOut]


class TraceSpanOut(BaseModel):
    """Mirrors ``ragfabric_core.strategies.base.TraceSpan``, one timed step."""

    name: str
    started_ms: int
    duration_ms: int
    attributes: dict[str, str | int | float | bool | None] = {}


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
    # None only where no retrieval ran to measure. Every route that retrieves
    # populates it.
    usage: Usage | None = None
    # Everything below is empty unless the agent served the request. Defaulted
    # rather than optional, so a client reads an empty list for "this strategy
    # has none of these" and never has to tell that apart from "absent".
    sub_questions: list[SubQuestionReportOut] = []
    dropped_claims: list[DroppedClaimOut] = []
    dated_sources: list[DatedSubQuestionOut] = []
    trace: list[TraceSpanOut] = []
