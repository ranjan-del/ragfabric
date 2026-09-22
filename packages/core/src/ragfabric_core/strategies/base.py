"""The RetrieverStrategy interface and the common RetrievalResult (ADR 0002).

Why one result shape: the product compares four retrieval architectures. The
comparison is only fair if generation, citation checking, metrics and
evaluation run identical code over identical data. So every strategy, however
different inside, returns a RetrievalResult, and nothing downstream is allowed
to ask which strategy produced it.

Why a Protocol rather than an abstract base class: adopters can implement a
strategy in their own package without inheriting from ours. ``runtime_checkable``
lets the registry and the tests verify conformance with ``isinstance``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ragfabric_core.auth.principal import AccessFilter, Principal


class StrategyName(StrEnum):
    TRADITIONAL = "traditional"
    VECTORLESS = "vectorless"
    AGENTIC = "agentic"
    GRAPH = "graph"


class TraceSpan(BaseModel):
    """One timed step inside a strategy, for the Trace page and OTel export."""

    name: str
    started_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: int
    document_id: int
    collection_id: int | None
    text: str
    page: int | None = None
    section: str | None = None
    score: float | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class StrategyParams(BaseModel):
    """Per request tuning. Strategies read what applies to them and ignore the rest."""

    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_filters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Budget(BaseModel):
    """Hard limits a strategy must respect. Agentic RAG stops when any is hit.

    This is the caller's ceiling for one request. A deployment's own limits
    live in ``AgenticConfig``, and the agent enforces whichever of the two is
    smaller: a request may ask for less than the deployment allows and may not
    ask for more.

    The defaults therefore match ``AgenticConfig``'s, and that is not a
    cosmetic tidy. ``max_llm_calls`` defaulted to eight here while the
    configuration and ``ragfabric.example.yaml`` both said twelve, and since
    nothing constructed a ``Budget`` on the request path, eight was the number
    actually enforced. The documented limit could never be reached, and an
    operator raising it saw no change at all.
    """

    max_llm_calls: int = Field(default=12, ge=0)
    max_latency_ms: int = Field(default=30_000, ge=0)
    max_cost_usd: float = Field(default=0.10, ge=0.0)


class RetrievalContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    principal: Principal
    access_filter: AccessFilter
    collection_ids: list[int] | None = None
    params: StrategyParams = Field(default_factory=StrategyParams)
    budget: Budget = Field(default_factory=Budget)


class SubQuestionReport(BaseModel):
    """What happened to one part of a decomposed question.

    This is what a single ``best_effort`` boolean could not say. A flag tells
    the caller that an answer is incomplete; it cannot tell them which part is
    missing or why, so the only honest thing they can do with it is distrust
    the whole answer. A row per sub-question lets them distrust exactly the
    part that failed.

    ``status`` uses the agent ledger's own three words rather than a second
    vocabulary meaning the same thing. ``open`` on a finished run means the run
    stopped with this part still unanswered, and ``reason`` says what stopped
    it: a budget, a stall, or an abandonment and what had been tried first.

    ``reason`` is ``None`` only for an answered sub-question. Every other
    status carries one, because an unexplained gap in an answer is the failure
    this whole report exists to prevent.

    ``chunk_ids`` names the evidence retrieved for this part, so a caller can
    tie a citation back to the sub-question it came from without re-running the
    grouping the agent already did.
    """

    text: str
    status: Literal["answered", "abandoned", "open"]
    reason: str | None = None
    chunk_ids: list[int] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    strategy: StrategyName
    chunks: list[RetrievedChunk]
    retrieval_calls: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    # Embedding provider calls actually issued. Reported separately from
    # llm_calls because the two are priced differently and, for the vectorless
    # strategy, the whole point is that this is zero. Defaults to 0 so a
    # strategy that embeds nothing does not have to remember to say so, and
    # every strategy that does embed sets it explicitly (ADR 0004: a count
    # reports what happened, it is never defaulted to a plausible number).
    embedding_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    trace: list[TraceSpan] = Field(default_factory=list)
    fallback_from: StrategyName | None = None
    # Per sub-question outcomes, for a strategy that decomposes the question.
    # Empty by default: no other strategy has parts to report, and defaulting
    # to empty is what keeps this addition invisible to all three of them.
    sub_questions: list[SubQuestionReport] = Field(default_factory=list)


@runtime_checkable
class RetrieverStrategy(Protocol):
    name: StrategyName

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult: ...


class StrategyRegistry:
    """Name to implementation. The router and the API resolve strategies here."""

    def __init__(self) -> None:
        self._by_name: dict[StrategyName, RetrieverStrategy] = {}

    def register(self, strategy: RetrieverStrategy) -> None:
        if not isinstance(strategy, RetrieverStrategy):
            raise TypeError(f"{type(strategy).__name__} does not implement RetrieverStrategy")
        self._by_name[StrategyName(strategy.name)] = strategy

    def get(self, name: StrategyName | str) -> RetrieverStrategy:
        key = StrategyName(name)
        if key not in self._by_name:
            raise KeyError(f"no strategy registered under {key!r}")
        return self._by_name[key]

    def names(self) -> list[StrategyName]:
        return list(self._by_name)
