"""The agent's state and its budget accounting.

These cover the invariants the loop depends on and cannot check for itself:
a sub-question's lifecycle, that abandoning is never silent, that spending is
refused rather than merely reported once a cap is crossed, and that the
evidence pool cannot double count a chunk retrieved twice.
"""

from __future__ import annotations

import pytest

from ragfabric_core.agent.state import (
    AgentState,
    BudgetExceeded,
    NodeName,
    RepairMove,
    SubQuestion,
    SubQuestionStatus,
)
from ragfabric_core.strategies.base import RetrievedChunk


def _chunk(chunk_id: int, text: str = "some text") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, document_id=1, collection_id=None, text=text)


def test_a_sub_question_starts_open():
    sq = SubQuestion(text="what is the retry limit", tool="lexical_search")
    assert sq.status is SubQuestionStatus.OPEN
    assert sq.attempts == []
    assert sq.reason is None


def test_abandoning_requires_a_reason():
    """An abandoned sub-question with no reason is a silent failure, which is
    the thing the structured partial answer exists to prevent."""
    sq = SubQuestion(text="q", tool="semantic_search")
    with pytest.raises(ValueError):
        sq.abandon(reason="")
    sq.abandon(reason="no evidence found after three moves")
    assert sq.status is SubQuestionStatus.ABANDONED
    assert sq.reason == "no evidence found after three moves"


def test_recording_an_attempt_keeps_the_move_and_why_it_failed():
    sq = SubQuestion(text="q", tool="semantic_search")
    sq.record_attempt(move=RepairMove.BROADEN, query="q broadened", failure="no chunks returned")
    assert len(sq.attempts) == 1
    assert sq.attempts[0].move is RepairMove.BROADEN
    assert sq.attempts[0].failure == "no chunks returned"


def test_a_move_already_tried_is_reported_as_tried():
    sq = SubQuestion(text="q", tool="semantic_search")
    assert not sq.has_tried(RepairMove.NARROW)
    sq.record_attempt(move=RepairMove.NARROW, query="q narrowed", failure="still nothing")
    assert sq.has_tried(RepairMove.NARROW)


def test_spend_raises_when_the_global_llm_cap_is_crossed():
    state = AgentState(question="q", max_llm_calls=2)
    state.spend(NodeName.PLAN, llm_calls=1)
    state.spend(NodeName.ASSESS, llm_calls=1)
    with pytest.raises(BudgetExceeded) as exc:
        state.spend(NodeName.REPAIR, llm_calls=1)
    assert "max_llm_calls" in str(exc.value)


def test_spend_raises_when_a_per_node_cap_is_crossed():
    """A global cap alone lets one runaway node eat the whole budget."""
    state = AgentState(question="q", max_llm_calls=100, per_node_llm_calls={NodeName.ASSESS: 1})
    state.spend(NodeName.ASSESS, llm_calls=1)
    with pytest.raises(BudgetExceeded) as exc:
        state.spend(NodeName.ASSESS, llm_calls=1)
    assert "assess" in str(exc.value)


def test_spend_that_is_refused_does_not_increment_the_counter():
    state = AgentState(question="q", max_llm_calls=1)
    state.spend(NodeName.PLAN, llm_calls=1)
    with pytest.raises(BudgetExceeded):
        state.spend(NodeName.PLAN, llm_calls=1)
    assert state.llm_calls == 1


def test_evidence_pool_deduplicates_by_chunk_id():
    state = AgentState(question="q")
    first = state.add_evidence([_chunk(1), _chunk(2)])
    second = state.add_evidence([_chunk(2), _chunk(3)])
    assert first == {1, 2}
    assert second == {3}
    assert set(state.evidence) == {1, 2, 3}


def test_add_evidence_returns_empty_when_nothing_is_new():
    """This is what progress detection reads. An iteration that returns the
    same chunks has not progressed, however many rows came back."""
    state = AgentState(question="q")
    state.add_evidence([_chunk(1), _chunk(2)])
    assert state.add_evidence([_chunk(1), _chunk(2)]) == set()


def test_open_sub_questions_excludes_resolved_ones():
    state = AgentState(question="q")
    a = SubQuestion(text="a", tool="semantic_search")
    b = SubQuestion(text="b", tool="lexical_search")
    c = SubQuestion(text="c", tool="lexical_search")
    b.mark_answered()
    c.abandon(reason="nothing found")
    state.sub_questions = [a, b, c]
    assert [sq.text for sq in state.open_sub_questions()] == ["a"]
