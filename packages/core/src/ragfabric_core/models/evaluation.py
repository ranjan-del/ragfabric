"""Benchmark runs and per question results (ADR 0004: numbers come from here, never from prose)."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base, utcnow


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    commit_sha: Mapped[str] = mapped_column(String, default="", nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False, index=True)
    llm_model: Mapped[str] = mapped_column(String, default="", nullable=False)
    embedding_model: Mapped[str] = mapped_column(String, default="", nullable=False)
    judge_model: Mapped[str | None] = mapped_column(String, nullable=True)
    question_set: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_run_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_id: Mapped[str] = mapped_column(String, nullable=False)
    question_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    difficulty: Mapped[str] = mapped_column(String, default="", nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, default="", nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("retrieval_runs.id"), nullable=True
    )
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reciprocal_rank: Mapped[float | None] = mapped_column(Float, nullable=True)
    correctness: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
