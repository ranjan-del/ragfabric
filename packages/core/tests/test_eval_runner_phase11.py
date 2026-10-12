"""Runner additions for Phase 11: resume, retrieval only, judge metric selection, ``cites``."""

import json

import pytest
from test_eval_runner import GOOD, LEAVE, QS, Fake

from ragfabric_core.evaluation.judge import LexicalJudge, LLMJudge, NoJudge
from ragfabric_core.evaluation.runner import run_batch
from ragfabric_core.evaluation.store import batch_runs
from ragfabric_core.evaluation.target import TargetAnswer
from ragfabric_core.providers.offline import ScriptedLLMProvider


class Flaky(Fake):
    """Fails on the questions listed in ``fail_on`` until ``healed`` is set."""

    def __init__(self, name, answers, fail_on):
        super().__init__(name, answers)
        self.fail_on = set(fail_on)
        self.asked: list[str] = []
        self.healed = False

    def answer(self, question):
        self.asked.append(question)
        if question in self.fail_on and not self.healed:
            raise KeyboardInterrupt if question == "STOP" else RuntimeError("laptop slept")
        return super().answer(question)


class Interrupting(Fake):
    """Answers ``n`` questions, then raises like a killed process would."""

    def __init__(self, name, answers, n):
        super().__init__(name, answers)
        self.n = n
        self.asked: list[str] = []

    def answer(self, question):
        if len(self.asked) >= self.n:
            raise KeyboardInterrupt
        self.asked.append(question)
        return super().answer(question)


def _rows(factory, batch):
    with factory() as db:
        return [(run, rows) for run, rows in batch_runs(db, batch)]


def test_resume_continues_an_interrupted_run_and_skips_stored_questions(eval_db):
    _, factory = eval_db
    first = Interrupting("traditional", GOOD, n=2)
    with pytest.raises(KeyboardInterrupt):
        run_batch([first], QS, LexicalJudge(), factory, batch="r1", meta={})
    runs = _rows(factory, "r1")
    assert len(runs) == 1 and len(runs[0][1]) == 2
    assert runs[0][0].summary["status"] == "running"

    again = Fake("traditional", GOOD)
    asked = []
    original = again.answer
    again.answer = lambda q: (asked.append(q), original(q))[1]
    ids = run_batch([again], QS, LexicalJudge(), factory, batch="r1", meta={}, resume=True)
    assert asked == ["Unanswerable?"]
    runs = _rows(factory, "r1")
    assert len(runs) == 1 and ids == [runs[0][0].id]
    run, rows = runs[0]
    assert sorted(r.question_id for r in rows) == ["q1", "q2", "q3"]
    assert run.summary["status"] == "finished" and run.summary["questions"] == 3
    assert run.summary["metrics"]["hit"] == 1.0


def test_resume_retries_rows_that_failed_with_an_error(eval_db):
    _, factory = eval_db
    target = Flaky("traditional", GOOD, fail_on={"Hotel limit?"})
    run_batch([target], QS, LexicalJudge(), factory, batch="r2", meta={})
    assert _rows(factory, "r2")[0][0].summary["errors"] == 1
    target.healed = True
    target.asked.clear()
    run_batch([target], QS, LexicalJudge(), factory, batch="r2", meta={}, resume=True)
    assert target.asked == ["Hotel limit?"]
    run, rows = _rows(factory, "r2")[0]
    assert run.summary["errors"] == 0 and len(rows) == 3


def test_resume_starts_targets_that_had_not_started(eval_db):
    _, factory = eval_db
    run_batch([Fake("traditional", GOOD)], QS, LexicalJudge(), factory, batch="r3", meta={})
    run_batch(
        [Fake("traditional", GOOD), Fake("vectorless", GOOD)],
        QS,
        LexicalJudge(),
        factory,
        batch="r3",
        meta={},
        resume=True,
    )
    runs = _rows(factory, "r3")
    assert [r.strategy for r, _ in runs] == ["traditional", "vectorless"]
    assert all(len(rows) == 3 for _, rows in runs)


def test_resume_does_not_duplicate_a_skipped_target(eval_db):
    _, factory = eval_db
    for _ in range(2):
        run_batch(
            [],
            QS,
            LexicalJudge(),
            factory,
            batch="r4",
            meta={},
            skipped=[("graph", "no graph")],
            resume=True,
        )
    assert len(_rows(factory, "r4")) == 1


class Retrieving(Fake):
    def __init__(self, name, answers):
        super().__init__(name, answers)
        self.generated = 0

    def answer(self, question):
        self.generated += 1
        return super().answer(question)

    def retrieve(self, question):
        _, contexts = self.answers[question]
        return TargetAnswer(answer="", contexts=contexts, retrieval_calls=1, latency_ms=3)


def test_retrieval_only_never_generates_or_judges_and_leaves_answer_metrics_unmeasured(eval_db):
    _, factory = eval_db
    target = Retrieving("traditional", GOOD)
    llm = ScriptedLLMProvider(responses=[], model="judge")
    run_batch(
        [target], QS, LLMJudge(llm, "judge"), factory, batch="ro", meta={}, retrieval_only=True
    )
    assert target.generated == 0 and llm.calls == 0
    run, rows = _rows(factory, "ro")[0]
    q1 = next(r for r in rows if r.question_id == "q1")
    assert q1.hit == 1.0 and q1.reciprocal_rank == 1.0
    assert q1.correctness is None and q1.faithfulness is None and q1.citation_correct is None
    assert run.summary["meta"]["retrieval_only"] is True
    assert run.summary["metrics"]["correctness"] is None


def test_retrieval_only_on_a_target_without_retrieve_records_an_error_per_row(eval_db):
    _, factory = eval_db
    run_batch(
        [Fake("x", GOOD)], QS, LexicalJudge(), factory, batch="ro2", meta={}, retrieval_only=True
    )
    run, rows = _rows(factory, "ro2")[0]
    assert run.summary["errors"] == 3
    assert "retrieval only" in rows[0].details["error"]


def test_judge_metric_selection_asks_only_for_correctness():
    llm = ScriptedLLMProvider(responses=[json.dumps({"score": 0.9, "reason": "ok"})], model="j")
    judge = LLMJudge(llm, "j", metrics=("correctness",))
    s = judge.score("How many days?", "10 days.", "Ten days [1].", [LEAVE])
    assert s.correctness == 0.9
    assert s.faithfulness is None and s.context_relevance is None
    assert llm.calls == 1 and s.calls == 1


def test_lexical_judge_metric_selection():
    s = LexicalJudge(metrics=("correctness",)).score("q", "10 days", "10 days [1].", [LEAVE])
    assert s.correctness is not None and s.faithfulness is None and s.context_relevance is None


def test_unknown_judge_metric_is_refused():
    with pytest.raises(ValueError, match="faithfulness"):
        LexicalJudge(metrics=("speed",))


def test_the_judge_metrics_are_recorded_in_the_run_meta(eval_db):
    _, factory = eval_db
    run_batch(
        [Fake("t", GOOD)], QS, LexicalJudge(metrics=("correctness",)), factory, batch="jm", meta={}
    )
    run, rows = _rows(factory, "jm")[0]
    assert run.summary["meta"]["judge_metrics"] == ["correctness"]
    assert all(r.faithfulness is None for r in rows)


def test_no_judge_scores_nothing():
    s = NoJudge().score("q", "10 days", "10 days", [LEAVE])
    assert (s.correctness, s.faithfulness, s.context_relevance, s.calls) == (None, None, None, 0)


class NoMarkers(Fake):
    def answer(self, question):
        out = super().answer(question)
        return out.model_copy(update={"answer": "Ten days of casual leave.", "cites": False})


def test_a_target_that_does_not_cite_gets_no_citation_score_rather_than_zero(eval_db):
    _, factory = eval_db
    run_batch([NoMarkers("other-tool", GOOD)], QS, LexicalJudge(), factory, batch="c1", meta={})
    run, rows = _rows(factory, "c1")[0]
    assert all(r.citation_correct is None for r in rows)
    assert rows[0].details["citation_reason"] == "target does not emit [n] citation markers"
    assert run.summary["metrics"]["citation_correct"] is None


def test_target_answer_cites_defaults_to_true():
    assert TargetAnswer(answer="x").cites is True
