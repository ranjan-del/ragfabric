"""Read stored evaluation runs back, as rows and as plain dicts for the CLI and API."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ragfabric_core.evaluation.runner import router_agreement
from ragfabric_core.models.evaluation import EvaluationResult, EvaluationRun


def run_dict(run: EvaluationRun) -> dict:
    return {
        "id": run.id,
        "batch": run.name,
        "target": run.strategy,
        "commit": run.commit_sha,
        "llm_model": run.llm_model,
        "embedding_model": run.embedding_model,
        "judge": run.judge_model,
        "question_set": run.question_set,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "summary": run.summary,
    }


def result_dict(row: EvaluationResult) -> dict:
    return {
        "question_id": row.question_id,
        "question_type": row.question_type,
        "difficulty": row.difficulty,
        "question": row.question,
        "expected_answer": row.expected_answer,
        "answer": row.answer,
        "precision": row.precision,
        "recall": row.recall,
        "hit": row.hit,
        "reciprocal_rank": row.reciprocal_rank,
        "correctness": row.correctness,
        "faithfulness": row.faithfulness,
        "context_relevance": row.context_relevance,
        "citation_correct": row.citation_correct,
        "latency_ms": row.latency_ms,
        "estimated_cost_usd": row.estimated_cost_usd,
        "details": row.details,
    }


def list_runs(db: Session, limit: int = 20) -> list[EvaluationRun]:
    return (
        db.query(EvaluationRun)
        .order_by(EvaluationRun.started_at.desc(), EvaluationRun.id.desc())
        .limit(limit)
        .all()
    )


def get_run(db: Session, run_id: int) -> tuple[EvaluationRun, list[EvaluationResult]] | None:
    run = db.get(EvaluationRun, run_id)
    if run is None:
        return None
    results = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.evaluation_run_id == run_id)
        .order_by(EvaluationResult.id)
        .all()
    )
    return run, results


def latest_batch(db: Session) -> str | None:
    newest = list_runs(db, limit=1)
    return newest[0].name if newest else None


def batch_runs(db: Session, batch: str) -> list[tuple[EvaluationRun, list[EvaluationResult]]]:
    runs = db.query(EvaluationRun).filter(EvaluationRun.name == batch).order_by(EvaluationRun.id)
    out = []
    for run in runs:
        found = get_run(db, run.id)
        if found is not None:
            out.append(found)
    return out


def batch_router_agreement(
    runs: list[tuple[EvaluationRun, list[EvaluationResult]]],
) -> dict | None:
    return router_agreement(
        {run.strategy: {r.question_id: r.correctness for r in rows} for run, rows in runs if rows}
    )
