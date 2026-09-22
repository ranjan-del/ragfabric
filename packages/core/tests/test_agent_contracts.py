"""The JSON contracts between the agent and whatever model is driving it.

Small local models do not return clean JSON. They wrap it in a fenced code
block, they preface it with "Sure, here is the plan:", and sometimes they return
something that is not JSON at all. None of that may crash a request, and none of
it may be quietly treated as success.
"""

from __future__ import annotations

from ragfabric_core.agent.contracts import (
    ContractViolation,
    parse_assess,
    parse_plan,
    parse_repair,
)
from ragfabric_core.agent.state import RepairMove

PLAN_JSON = """
{"sub_questions": [
  {"text": "what is the retry limit", "tool": "lexical_search", "why": "an identifier"},
  {"text": "what changed in 2026", "tool": "semantic_search", "why": "conceptual"}
]}
"""


def test_a_clean_plan_is_parsed():
    result = parse_plan(PLAN_JSON)
    assert not isinstance(result, ContractViolation)
    assert [sq.text for sq in result.sub_questions] == [
        "what is the retry limit",
        "what changed in 2026",
    ]
    assert result.sub_questions[0].tool == "lexical_search"


def test_json_inside_a_fenced_code_block_is_accepted():
    result = parse_plan(f"```json\n{PLAN_JSON}\n```")
    assert not isinstance(result, ContractViolation)
    assert len(result.sub_questions) == 2


def test_leading_prose_before_the_json_is_tolerated():
    result = parse_plan(f"Sure, here is the plan you asked for:\n{PLAN_JSON}")
    assert not isinstance(result, ContractViolation)
    assert len(result.sub_questions) == 2


def test_text_with_no_json_at_all_is_a_violation():
    result = parse_plan("I am afraid I cannot help with that.")
    assert isinstance(result, ContractViolation)
    assert result.contract == "plan"
    assert result.raw.startswith("I am afraid")


def test_a_plan_with_no_sub_questions_is_a_violation():
    """An empty plan would leave the loop with nothing to retrieve for, and it
    would then report no progress rather than the real problem."""
    result = parse_plan('{"sub_questions": []}')
    assert isinstance(result, ContractViolation)


def test_an_assessment_is_parsed():
    result = parse_assess(
        '{"verdicts": [{"sub_question": "a", "answered": true, "missing": null},'
        ' {"sub_question": "b", "answered": false, "missing": "no dates given"}]}'
    )
    assert not isinstance(result, ContractViolation)
    assert result.verdicts[0].answered is True
    assert result.verdicts[1].missing == "no dates given"


def test_an_unknown_repair_move_is_a_violation_not_a_crash():
    """The move comes from a model. An invented action must be refused by the
    schema rather than reaching the policy."""
    result = parse_repair('{"move": "delete_everything", "why": "seems fastest"}')
    assert isinstance(result, ContractViolation)
    assert "delete_everything" in result.error or "move" in result.error


def test_every_valid_repair_move_is_accepted():
    for move in RepairMove:
        result = parse_repair(f'{{"move": "{move.value}", "why": "because"}}')
        assert not isinstance(result, ContractViolation), move
        assert result.move is move


def test_a_violation_carries_the_raw_text_for_the_trace():
    """A violation that does not say what the model actually returned cannot be
    debugged after the fact."""
    result = parse_assess("not json")
    assert isinstance(result, ContractViolation)
    assert result.raw == "not json"
    assert result.error


def test_a_repair_may_carry_a_rewritten_query():
    result = parse_repair(
        '{"move": "broaden", "rewritten_query": "retry limit", "why": "too specific"}'
    )
    assert not isinstance(result, ContractViolation)
    assert result.rewritten_query == "retry limit"


def test_the_last_json_object_wins_when_a_model_rambles_after_it():
    """Some models emit the answer then explain themselves. Trailing prose must
    not defeat parsing."""
    result = parse_plan(f"{PLAN_JSON}\nI hope that helps!")
    assert not isinstance(result, ContractViolation)
