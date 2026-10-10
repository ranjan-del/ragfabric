"""Dashboard aggregates over hand inserted runs."""

from datetime import UTC, datetime, timedelta

import pytest

from ragfabric_core.evaluation.dashboard import dashboard
from ragfabric_core.models.evaluation import EvaluationRun
from ragfabric_core.models.runs import RetrievalRun

NOW = datetime(2026, 10, 10, 12, tzinfo=UTC)


def _run(strategy, latency, *, cost=0.001, fallback=None, days_ago=0, llm_calls=1):
    return RetrievalRun(
        question="q",
        selected_strategy=strategy,
        latency_ms=latency,
        llm_calls=llm_calls,
        retrieval_calls=2,
        fallback_from=fallback,
        estimated_cost_usd=cost,
        created_at=NOW - timedelta(days=days_ago),
    )


def test_aggregates(eval_db):
    _, factory = eval_db
    with factory() as db:
        db.add_all(
            [
                _run("traditional", 100),
                _run("traditional", 300, days_ago=1),
                _run("traditional", 200, cost=None),
                _run("graph", 900, fallback="graph", llm_calls=3),
                _run("graph", 50, days_ago=40),  # outside the window
            ]
        )
        db.add(
            EvaluationRun(
                name="b1",
                strategy="traditional",
                question_set="s",
                judge_model="lexical",
                started_at=NOW - timedelta(days=2),
                finished_at=NOW - timedelta(days=2),
                summary={"status": "finished", "metrics": {"hit": 0.8, "correctness": 0.5}},
            )
        )
        db.add(
            EvaluationRun(
                name="b1",
                strategy="agentic",
                question_set="s",
                started_at=NOW - timedelta(days=2),
                finished_at=NOW - timedelta(days=2),
                summary={"status": "skipped", "skipped": "offline"},
            )
        )
        db.commit()
        out = dashboard(db, days=30, now=NOW)
    assert out["runs"] == 4
    assert out["latency_ms"]["traditional"] == {"p50": 200, "p95": 300, "runs": 3}
    assert out["latency_ms"]["graph"]["runs"] == 1
    assert out["calls_per_strategy"]["graph"] == {"runs": 1, "llm_calls": 3, "retrieval_calls": 2}
    days = {d["date"]: d for d in out["cost_per_day"]}
    today = days["2026-10-10"]
    assert today["runs"] == 3 and today["unpriced_runs"] == 1
    assert today["estimated_cost_usd"] == pytest.approx(0.002)
    assert days["2026-10-09"]["estimated_cost_usd"] == pytest.approx(0.001)
    assert out["fallback_rate"] == {"runs": 4, "fallbacks": 1, "rate": 0.25}
    assert [t["target"] for t in out["quality_trend"]] == ["traditional"]
    assert out["quality_trend"][0]["hit"] == 0.8


def test_an_empty_window_has_no_rate(eval_db):
    _, factory = eval_db
    with factory() as db:
        out = dashboard(db, days=7, now=NOW)
    assert out["runs"] == 0 and out["fallback_rate"]["rate"] is None
    assert out["cost_per_day"] == [] and out["quality_trend"] == []
