"""What every evaluated system returns, and the protocol it implements (D4).

Metrics read a ``TargetAnswer`` and nothing else. A RagFabric strategy is one
target (``StrategyTarget``); Phase 11 adds adapters for other RAG tools that
return the same shape, and every metric, the store and the report then work
for them unchanged. A field a target cannot measure is ``None``, never a
plausible default (ADR 0004).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ContextItem(BaseModel):
    """One retrieved passage, as the generator saw it: ``rank`` is its [n] marker."""

    rank: int = Field(ge=1)
    document: str
    text: str
    score: float | None = None


class TargetAnswer(BaseModel):
    answer: str
    contexts: list[ContextItem] = Field(default_factory=list)
    strategy_used: str | None = None
    fallback_from: str | None = None
    llm_calls: int | None = None
    retrieval_calls: int | None = None
    embedding_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    retrieval_latency_ms: int | None = None
    generation_latency_ms: int | None = None
    llm_model: str | None = None
    estimated_cost_usd: float | None = None
    router_reasoning: str | None = None


class TargetSkipped(Exception):
    """This target cannot run in this deployment; the reason is recorded, no zeros are."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@runtime_checkable
class EvalTarget(Protocol):
    name: str

    def answer(self, question: str) -> TargetAnswer: ...
