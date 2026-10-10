"""The runner, its summaries and the stored rows, with fake targets."""

import pytest

from ragfabric_core.evaluation.dataset import QuestionSet
from ragfabric_core.evaluation.judge import LexicalJudge
from ragfabric_core.evaluation.runner import percentile, router_agreement, run_batch
from ragfabric_core.evaluation.store import batch_router_agreement, batch_runs, get_run
from ragfabric_core.evaluation.target import ContextItem, TargetAnswer

QS = QuestionSet.model_validate(
    {
        "name": "tiny",
        "questions": [
            {
                "id": "q1",
                "question": "How many days of casual leave?",
                "expected_answer": "10 days of casual leave",
                "expected_sources": [{"document": "leave.md", "evidence": "10 days"}],
                "question_type": "simple_factual",
            },
            {
                "id": "q2",
                "question": "Hotel limit?",
                "expected_answer": "180 dollars per night",
                "expected_sources": [{"document": "travel.md"}],
                "question_type": "exact_match",
            },
            {
                "id": "q3",
                "question": "Unanswerable?",
                "expected_answer": "",
                "question_type": "ambiguous",
            },
        ],
    }
)

LEAVE = ContextItem(rank=1, document="leave.md", text="Employees receive 10 days of casual leave.")
TRAVEL = ContextItem(rank=1, document="travel.md", text="The hotel limit is 180 dollars per night.")


class Fake:
    def __init__(self, name, answers, latency=10):
        self.name = name
        self.answers = answers
        self.latency = latency

    def answer(self, question):
        value = self.answers[question]
        if isinstance(value, Exception):
            raise value
        text, contexts = value
        return TargetAnswer(
            answer=text,
            contexts=contexts,
            strategy_used=self.name,
            llm_calls=1,
            retrieval_calls=1,
            input_tokens=10,
            output_tokens=5,
            latency_ms=self.latency,
            estimated_cost_usd=None,
        )


GOOD = {
    "How many days of casual leave?": ("Employees receive 10 days of casual leave [1].", [LEAVE]),
    "Hotel limit?": ("The hotel limit is 180 dollars per night [1].", [TRAVEL]),
    "Unanswerable?": ("I could not find this in the documents.", []),
}


def test_one_run_per_target_and_one_row_per_question(eval_db):
    _, factory = eval_db
    ids = run_batch(
        [Fake("traditional", GOOD), Fake("vectorless", GOOD)],
        QS,
        LexicalJudge(),
        factory,
        batch="b1",
        meta={"commit": "abc123", "top_k": 5},
    )
    assert len(ids) == 2
    with factory() as db:
        runs = batch_runs(db, "b1")
        assert [r.strategy for r, _ in runs] == ["traditional", "vectorless"]
        run, rows = runs[0]
        assert len(rows) == 3 and run.commit_sha == "abc123" and run.finished_at is not None
        assert run.judge_model == "lexical"
        meta = run.summary["meta"]
        assert meta["judge_kind"] == "lexical" and meta["question_set"] == "tiny"
        assert meta["top_k"] == 5
        m = run.summary["metrics"]
        assert m["hit"] == 1.0 and m["reciprocal_rank"] == 1.0 and m["citation_correct"] == 1.0
        # q3 has no expected sources, so retrieval means are over q1 and q2 only.
        q3 = next(r for r in rows if r.question_id == "q3")
        assert (q3.precision, q3.recall, q3.hit, q3.reciprocal_rank) == (None, None, None, None)
        assert run.summary["by_category"]["ambiguous"]["hit"] is None
        assert run.summary["llm_calls"] == 3 and run.summary["input_tokens"] == 30
        assert run.summary["estimated_cost_usd"] is None and run.summary["cost_unknown"] == 3


def test_a_target_that_raises_records_the_error_and_continues(eval_db):
    _, factory = eval_db
    answers = dict(GOOD)
    answers["How many days of casual leave?"] = RuntimeError("model went away")
    (run_id,) = run_batch(
        [Fake("graph", answers)], QS, LexicalJudge(), factory, batch="b2", meta={}
    )
    with factory() as db:
        run, rows = get_run(db, run_id)
        q1 = rows[0]
        assert q1.details["error"] == "RuntimeError: model went away"
        assert q1.hit is None and q1.correctness is None
        assert rows[1].hit is True
        assert run.summary["errors"] == 1 and run.summary["questions"] == 3


def test_skipped_targets_get_a_run_row_with_the_reason_and_no_results(eval_db):
    _, factory = eval_db
    ids = run_batch(
        [], QS, LexicalJudge(), factory, batch="b3", meta={}, skipped=[("agentic", "offline")]
    )
    with factory() as db:
        run, rows = get_run(db, ids[0])
        assert rows == [] and run.summary["status"] == "skipped"
        assert run.summary["skipped"] == "offline"


def test_a_judge_that_raises_keeps_the_retrieval_scores(eval_db):
    class Broken(LexicalJudge):
        def score(self, *args):
            raise ValueError("bad judge")

    _, factory = eval_db
    (run_id,) = run_batch([Fake("t", GOOD)], QS, Broken(), factory, batch="b4", meta={})
    with factory() as db:
        _, rows = get_run(db, run_id)
        assert rows[0].hit is True and rows[0].correctness is None
        assert rows[0].details["judge_error"] == "ValueError: bad judge"


def test_latency_percentiles_are_nearest_rank():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 100
    assert percentile([7], 95) == 7
    assert percentile([], 50) is None


def test_router_agreement_against_the_best_fixed_strategy():
    out = router_agreement(
        {
            "auto": {"q1": 1.0, "q2": 0.5, "q3": None},
            "traditional": {"q1": 1.0, "q2": 0.5, "q3": 1.0},
            "graph": {"q1": 0.0, "q2": 1.0, "q3": 1.0},
            "traditional+rerank=llm": {"q1": 0.0, "q2": 0.0, "q3": 0.0},
        }
    )
    assert out["measured"] == 2 and out["matched"] == 1
    q2 = next(q for q in out["questions"] if q["question_id"] == "q2")
    assert q2["best_strategies"] == ["graph"] and q2["matched"] is False


def test_router_agreement_needs_auto_and_a_fixed_strategy():
    assert router_agreement({"traditional": {"q1": 1.0}}) is None
    assert router_agreement({"auto": {"q1": 1.0}}) is None


def test_batch_router_agreement_reads_stored_rows(eval_db):
    _, factory = eval_db
    run_batch(
        [Fake("auto", GOOD), Fake("traditional", GOOD)],
        QS,
        LexicalJudge(),
        factory,
        batch="b5",
        meta={},
    )
    with factory() as db:
        agreement = batch_router_agreement(batch_runs(db, "b5"))
    assert agreement["measured"] == 2 and agreement["matched"] == 2


@pytest.mark.parametrize("p", [50, 95])
def test_percentile_of_one_value_is_that_value(p):
    assert percentile([3.0], p) == 3.0
