"""The numbers the console dashboards draw (D17: data in Phase 8, pages in Phase 9).

Latency, cost, calls and fallbacks come from ``retrieval_runs``, the
production traffic; evaluation answers are never written there (D10). The
quality trend comes from ``evaluation_runs``. Aggregation is done in Python so
the same code runs on SQLite and PostgreSQL. Cost is the estimate written with
each run (D11); runs with no price are counted, never summed as zero.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from ragfabric_core.evaluation.runner import percentile
from ragfabric_core.models.evaluation import EvaluationRun
from ragfabric_core.models.runs import RetrievalRun


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def dashboard(db: Session, days: int = 30, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    since = now - timedelta(days=days)
    runs = (
        db.query(
            RetrievalRun.selected_strategy,
            RetrievalRun.latency_ms,
            RetrievalRun.llm_calls,
            RetrievalRun.retrieval_calls,
            RetrievalRun.fallback_from,
            RetrievalRun.estimated_cost_usd,
            RetrievalRun.created_at,
        )
        .filter(RetrievalRun.created_at >= since)
        .all()
    )
    latencies: dict[str, list[int]] = defaultdict(list)
    calls: dict[str, dict] = defaultdict(lambda: {"runs": 0, "llm_calls": 0, "retrieval_calls": 0})
    per_day: dict[str, dict] = defaultdict(
        lambda: {"runs": 0, "estimated_cost_usd": None, "unpriced_runs": 0}
    )
    fallbacks = 0
    for strategy, latency, llm_calls, retrieval_calls, fallback, cost, created in runs:
        latencies[strategy].append(latency)
        c = calls[strategy]
        c["runs"] += 1
        c["llm_calls"] += llm_calls
        c["retrieval_calls"] += retrieval_calls
        day = per_day[_aware(created).date().isoformat()]
        day["runs"] += 1
        if cost is None:
            day["unpriced_runs"] += 1
        else:
            day["estimated_cost_usd"] = round((day["estimated_cost_usd"] or 0.0) + cost, 8)
        fallbacks += 1 if fallback else 0

    evals = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.finished_at.is_not(None), EvaluationRun.started_at >= since)
        .order_by(EvaluationRun.started_at, EvaluationRun.id)
        .all()
    )
    trend = []
    for run in evals:
        if run.summary.get("status") != "finished":
            continue
        metrics = run.summary.get("metrics", {})
        trend.append(
            {
                "run_id": run.id,
                "batch": run.name,
                "target": run.strategy,
                "started_at": _aware(run.started_at).isoformat(),
                "judge": run.judge_model,
                "hit": metrics.get("hit"),
                "reciprocal_rank": metrics.get("reciprocal_rank"),
                "correctness": metrics.get("correctness"),
                "citation_correct": metrics.get("citation_correct"),
            }
        )
    return {
        "window_days": days,
        "since": since.isoformat(),
        "runs": len(runs),
        "latency_ms": {
            s: {"p50": percentile(v, 50), "p95": percentile(v, 95), "runs": len(v)}
            for s, v in sorted(latencies.items())
        },
        "calls_per_strategy": dict(sorted(calls.items())),
        "cost_per_day": [{"date": d, **v} for d, v in sorted(per_day.items())],
        "fallback_rate": {
            "runs": len(runs),
            "fallbacks": fallbacks,
            "rate": round(fallbacks / len(runs), 4) if runs else None,
        },
        "quality_trend": trend,
    }
