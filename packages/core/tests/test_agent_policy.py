"""The repair policy: six moves, and the guard that stops them oscillating.

The textbook agentic design has exactly one repair, "rewrite the query", which
makes it a retry loop with a vocabulary. These tests pin down the two
properties that make this a policy instead.

Each move does something structurally different, so a failure that a broader
query cannot fix can still be fixed by a different tool, a smaller question or
the whole document. And no move is ever repeated on a sub-question that has not
changed, because an agent free to alternate broaden and narrow will do exactly
that until the budget runs out, looking busy the whole time.
"""

from __future__ import annotations

from agent_doubles import chunk

from ragfabric_core.agent.nodes import RetrievalOverride
from ragfabric_core.agent.policy import (
    MOVES_WHEN_EVIDENCE_IS_OFF_POINT,
    apply_move,
    choose_move,
)
from ragfabric_core.agent.state import RepairMove, SubQuestion, SubQuestionStatus


def sq(text: str = "what is the retry limit", tool: str = "semantic_search") -> SubQuestion:
    return SubQuestion(text=text, tool=tool)


def working(question: SubQuestion, top_k: int = 5) -> RetrievalOverride:
    return RetrievalOverride(query=question.text, top_k=top_k)


def test_broaden_relaxes_the_query_and_raises_top_k():
    question = sq("what is the exact retry limit for the payments API in production")
    outcome = apply_move(
        question,
        RepairMove.BROADEN,
        working=working(question, top_k=5),
        missing="the number of retries",
    )

    assert outcome.move is RepairMove.BROADEN
    assert outcome.working.top_k > 5
    assert outcome.working.query != question.text
    assert len(outcome.working.query) < len(question.text)


def test_narrow_adds_the_missing_constraint_to_the_query():
    question = sq("what is the retry policy")
    outcome = apply_move(
        question,
        RepairMove.NARROW,
        working=working(question, top_k=20),
        missing="the number of retries allowed",
    )

    assert "the number of retries allowed" in outcome.working.query
    assert outcome.working.top_k <= 20


def test_switch_strategy_flips_the_tool():
    question = sq(tool="semantic_search")
    outcome = apply_move(question, RepairMove.SWITCH_STRATEGY, working=working(question))
    assert question.tool == "lexical_search"
    assert outcome.working.query == question.text

    back = apply_move(question, RepairMove.SWITCH_STRATEGY, working=outcome.working)
    assert question.tool == "semantic_search"
    assert back.move is RepairMove.SWITCH_STRATEGY


def test_decompose_splits_the_sub_question_further():
    question = sq("what is the retry limit and what is the backoff interval")
    outcome = apply_move(question, RepairMove.DECOMPOSE, working=working(question))

    assert len(outcome.new_sub_questions) == 2
    assert outcome.new_sub_questions[0].text == "what is the retry limit"
    assert outcome.new_sub_questions[1].text == "what is the backoff interval"
    assert question.status is SubQuestionStatus.ABANDONED
    assert question.reason is not None


def test_fetch_document_targets_a_document_already_matched():
    question = sq()
    outcome = apply_move(
        question,
        RepairMove.FETCH_DOCUMENT,
        working=working(question),
        evidence=[chunk(3, document_id=11, score=0.9), chunk(4, document_id=12, score=0.2)],
    )

    assert question.tool == "fetch_document"
    assert outcome.working.document_id == 11
    assert "11" in outcome.working.query


def test_abandon_records_a_reason():
    question = sq()
    outcome = apply_move(
        question,
        RepairMove.ABANDON,
        working=working(question),
        why="nothing in the corpus mentions a retry limit",
    )

    assert question.status is SubQuestionStatus.ABANDONED
    assert question.reason == "nothing in the corpus mentions a retry limit"
    assert outcome.abandoned is True


def test_abandon_without_a_given_reason_still_records_one():
    """A silent abandonment is the failure the structured report exists to prevent."""
    question = sq()
    question.record_attempt(move=RepairMove.BROADEN, query="q", failure="no chunks returned")
    apply_move(question, RepairMove.ABANDON, working=working(question))

    assert question.status is SubQuestionStatus.ABANDONED
    assert question.reason
    assert "broaden" in question.reason


def test_the_same_move_is_not_repeated_on_an_unchanged_sub_question():
    """Broaden then narrow then broaden is the oscillation this guard exists for."""
    question = sq()
    question.record_attempt(
        move=RepairMove.BROADEN, query="q broadened", failure="still no evidence"
    )

    chosen = choose_move(question, proposed=RepairMove.BROADEN, has_evidence=False)

    assert chosen is not RepairMove.BROADEN
    assert not question.has_tried(chosen)


def test_a_proposed_move_that_has_not_been_tried_is_honoured():
    """The model's judgement is used when it is usable. The guard is not a veto."""
    question = sq()
    assert choose_move(question, proposed=RepairMove.NARROW, has_evidence=True) is RepairMove.NARROW


def test_with_no_proposal_the_move_follows_the_situation():
    """No evidence at all and evidence that misses are different failures."""
    assert choose_move(sq(), proposed=None, has_evidence=False) is RepairMove.BROADEN
    assert (
        choose_move(sq(), proposed=None, has_evidence=True) is MOVES_WHEN_EVIDENCE_IS_OFF_POINT[0]
    )


def test_decompose_is_not_offered_when_the_sub_question_cannot_be_split():
    """A move that cannot do anything is not an option, it is a wasted iteration."""
    question = sq("what is the retry limit")
    for move in (RepairMove.BROADEN, RepairMove.NARROW, RepairMove.SWITCH_STRATEGY):
        question.record_attempt(move=move, query="q", failure="no")

    chosen = choose_move(question, proposed=RepairMove.DECOMPOSE, has_evidence=False)

    assert chosen is not RepairMove.DECOMPOSE


def test_fetch_document_is_not_offered_without_evidence_to_point_at():
    question = sq()
    chosen = choose_move(
        question, proposed=RepairMove.FETCH_DOCUMENT, has_evidence=False, has_document=False
    )
    assert chosen is not RepairMove.FETCH_DOCUMENT


def test_exhausting_the_moves_abandons_rather_than_loops():
    question = sq("what is the retry limit")
    for move in (
        RepairMove.BROADEN,
        RepairMove.NARROW,
        RepairMove.SWITCH_STRATEGY,
        RepairMove.FETCH_DOCUMENT,
    ):
        question.record_attempt(move=move, query="q", failure="nothing useful")

    chosen = choose_move(
        question, proposed=RepairMove.BROADEN, has_evidence=True, has_document=True
    )

    assert chosen is RepairMove.ABANDON


def test_applying_a_move_records_the_attempt_with_its_failure():
    """ "Try something different" is a uniqueness check. Recording why is learning."""
    question = sq()
    apply_move(
        question,
        RepairMove.BROADEN,
        working=working(question),
        missing="the number of retries",
    )

    assert question.attempts[-1].move is RepairMove.BROADEN
    assert question.attempts[-1].failure == "the number of retries"
