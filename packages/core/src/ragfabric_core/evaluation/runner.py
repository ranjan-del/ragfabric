"""Run a question set through targets, score every answer, persist every row.

Sequential and in file order (D13). One ``evaluation_runs`` row per target,
named after the batch (D8); one ``evaluation_results`` row per question,
committed as it is scored, so a run cut short keeps what it measured. A
target that raises on a question gets that error on the row and ``None``
scores, and the run goes on. A target that cannot run here gets a run row
saying why and no result rows, never a row of zeros (ADR 0004).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from statistics import fmean

from sqlalchemy.orm import Session

from ragfabric_core.evaluation.dataset import EvalQuestion, QuestionSet
from ragfabric_core.evaluation.judge import Judge, JudgeScores
from ragfabric_core.evaluation.metrics import citation_correct, retrieval_scores
from ragfabric_core.evaluation.target import EvalTarget, TargetAnswer
from ragfabric_core.generate.contract import NO_EVIDENCE
from ragfabric_core.models.base import utcnow
from ragfabric_core.models.evaluation import EvaluationResult, EvaluationRun

SCORE_FIELDS = (
    "precision",
    "recall",
    "hit",
    "reciprocal_rank",
    "correctness",
    "faithfulness",
    "context_relevance",
    "citation_correct",
)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least p% of values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(p / 100 * len(ordered)) - 1, 0)
    return ordered[index]


def is_refusal(answer: str | None) -> bool:
    """The generator's explicit no-evidence answer (the contract's own sentinel)."""
    return bool(answer) and NO_EVIDENCE in answer.lower()


def count_refusals(results: list[EvaluationResult]) -> int:
    """Answers that declined, counted apart so a refusal is not read as a retrieval miss."""
    return sum(1 for r in results if is_refusal(r.answer))


def _mean(values) -> float | None:
    measured = [float(v) for v in values if v is not None]
    return round(fmean(measured), 4) if measured else None


def summarise(results: list[EvaluationResult]) -> dict:
    """Means over measured values only; ``None`` where nothing was measured."""
    errors = sum(1 for r in results if r.details.get("error"))
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    costs = [r.estimated_cost_usd for r in results]
    by_category: dict[str, dict] = {}
    groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        groups[r.question_type].append(r)
    for category, rows in sorted(groups.items()):
        by_category[category] = {
            "questions": len(rows),
            **{f: _mean(getattr(r, f) for r in rows) for f in SCORE_FIELDS},
        }

    def detail_sum(key: str) -> int | None:
        values = [r.details.get(key) for r in results if r.details.get(key) is not None]
        return sum(values) if values else None

    return {
        "questions": len(results),
        "errors": errors,
        "metrics": {f: _mean(getattr(r, f) for r in results) for f in SCORE_FIELDS},
        "by_category": by_category,
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "llm_calls": detail_sum("llm_calls"),
        "retrieval_calls": detail_sum("retrieval_calls"),
        "input_tokens": detail_sum("input_tokens"),
        "output_tokens": detail_sum("output_tokens"),
        "judge_calls": detail_sum("judge_calls"),
        "estimated_cost_usd": round(sum(c for c in costs if c is not None), 6)
        if any(c is not None for c in costs)
        else None,
        "cost_unknown": sum(1 for c in costs if c is None),
        "fallbacks": sum(1 for r in results if r.details.get("fallback_from")),
        "refusals": count_refusals(results),
        "strategies_used": dict(
            Counter(r.details["strategy_used"] for r in results if r.details.get("strategy_used"))
        ),
    }


def _score(
    question: EvalQuestion, answer: TargetAnswer, judge: Judge
) -> tuple[dict, JudgeScores, dict]:
    retrieval = retrieval_scores(answer.contexts, question.expected_sources)
    citation = citation_correct(answer.answer, answer.contexts)
    try:
        judged = judge.score(
            question.question, question.expected_answer, answer.answer, answer.contexts
        )
        judge_error = None
    except Exception as exc:  # a judge failure must not lose the retrieval scores
        judged = JudgeScores()
        judge_error = f"{type(exc).__name__}: {exc}"
    scores = {
        **retrieval.model_dump(),
        "correctness": judged.correctness,
        "faithfulness": judged.faithfulness,
        "context_relevance": judged.context_relevance,
        "citation_correct": citation.correct,
    }
    extra = {"citation_reason": citation.reason, "judge_error": judge_error}
    return scores, judged, extra


def _result_row(
    run_id: int, question: EvalQuestion, answer: TargetAnswer | None, error: str | None, judge
) -> EvaluationResult:
    row = EvaluationResult(
        evaluation_run_id=run_id,
        question_id=question.id,
        question_type=str(question.question_type),
        difficulty=question.difficulty,
        question=question.question,
        expected_answer=question.expected_answer,
        details={},
    )
    if answer is None:
        row.details = {"error": error}
        return row
    scores, judged, extra = _score(question, answer, judge)
    for key, value in scores.items():
        setattr(row, key, value)
    row.answer = answer.answer
    row.latency_ms = answer.latency_ms
    row.estimated_cost_usd = answer.estimated_cost_usd
    row.details = {
        "error": None,
        "strategy_used": answer.strategy_used,
        "fallback_from": answer.fallback_from,
        "router_reasoning": answer.router_reasoning,
        "contexts": [
            {"rank": c.rank, "document": c.document, "score": c.score} for c in answer.contexts
        ],
        "llm_calls": answer.llm_calls,
        "retrieval_calls": answer.retrieval_calls,
        "embedding_calls": answer.embedding_calls,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "retrieval_latency_ms": answer.retrieval_latency_ms,
        "generation_latency_ms": answer.generation_latency_ms,
        "judge_reasons": judged.reasons,
        "judge_calls": judged.calls,
        **extra,
    }
    return row


def run_batch(
    targets: list[EvalTarget],
    questions: QuestionSet,
    judge: Judge,
    session_factory: Callable[[], Session],
    *,
    batch: str,
    meta: dict,
    skipped: list[tuple[str, str]] = (),
    llm_model: str | None = None,
    embedding_model: str | None = None,
    on_result: Callable[[str, EvaluationResult], None] | None = None,
) -> list[int]:
    """Run, score and persist; return the run ids in target order (skipped ones last)."""
    run_meta = {
        **meta,
        "judge_kind": judge.kind,
        "judge_model": judge.model,
        "judge_prompt_version": judge.prompt_version,
        "question_set": questions.name,
    }
    run_ids: list[int] = []
    for target in targets:
        with session_factory() as db:
            run = EvaluationRun(
                name=batch,
                commit_sha=str(meta.get("commit", "")),
                strategy=target.name,
                llm_model=llm_model or "",
                embedding_model=embedding_model or "",
                judge_model=f"{judge.kind}:{judge.model}" if judge.model else judge.kind,
                question_set=questions.name,
                summary={"meta": run_meta, "status": "running"},
            )
            db.add(run)
            db.commit()
            run_id = run.id
            rows: list[EvaluationResult] = []
            for question in questions.questions:
                try:
                    answer, error = target.answer(question.question), None
                except Exception as exc:  # recorded on the row; the run continues
                    answer, error = None, f"{type(exc).__name__}: {exc}"
                row = _result_row(run_id, question, answer, error, judge)
                db.add(row)
                db.commit()
                rows.append(row)
                if on_result is not None:
                    on_result(target.name, row)
            run.summary = {"meta": run_meta, "status": "finished", **summarise(rows)}
            run.finished_at = utcnow()
            db.commit()
            run_ids.append(run_id)
    for name, reason in skipped:
        with session_factory() as db:
            run = EvaluationRun(
                name=batch,
                commit_sha=str(meta.get("commit", "")),
                strategy=name,
                llm_model=llm_model or "",
                embedding_model=embedding_model or "",
                judge_model=f"{judge.kind}:{judge.model}" if judge.model else judge.kind,
                question_set=questions.name,
                summary={"meta": run_meta, "status": "skipped", "skipped": reason},
                finished_at=utcnow(),
            )
            db.add(run)
            db.commit()
            run_ids.append(run.id)
    return run_ids


def router_agreement(correctness: dict[str, dict[str, float | None]]) -> dict | None:
    """How often ``auto`` scored as well as the best fixed strategy on a question (D9).

    ``correctness`` maps target name to question id to correctness. Only
    questions where ``auto`` and at least one fixed strategy were measured
    count. Rerank variants are not fixed strategies a router could pick.
    """
    auto = correctness.get("auto")
    fixed = {n: v for n, v in correctness.items() if n != "auto" and "+" not in n}
    if not auto or not fixed:
        return None
    per_question = []
    for qid, score in auto.items():
        others = {n: v[qid] for n, v in fixed.items() if v.get(qid) is not None}
        if score is None or not others:
            continue
        best = max(others.values())
        per_question.append(
            {
                "question_id": qid,
                "auto": score,
                "best": best,
                "best_strategies": sorted(n for n, v in others.items() if v == best),
                "matched": score >= best - 1e-9,
            }
        )
    matched = sum(1 for p in per_question if p["matched"])
    return {"measured": len(per_question), "matched": matched, "questions": per_question}
