from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    rank: int
    chunk_id: int | None
    document_id: int | None
    score: float | None
    cited: bool
    page: int | None


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    question: str
    mode: str
    requested_strategy: str | None
    selected_strategy: str
    fallback_from: str | None
    answer: str | None
    latency_ms: int
    retrieval_latency_ms: int
    generation_latency_ms: int
    llm_calls: int
    retrieval_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    llm_model: str | None
    embedding_model: str | None
    trace: list
    created_at: datetime
    sources: list[SourceOut] = []
