"""Scripted JSON doubles: queue a decision per node, drive the loop offline.

Issue #6 asks for every branch of the agent to be exercised without a network
and without a database. That needs a double that can answer three different
questions in one run, because the loop asks the model to plan, then to assess,
then to repair, and a flat list of responses only works while the call order
never changes. The moment a branch skips a node, a flat script silently hands
the assess answer to the repair prompt and the test passes for the wrong reason.

So the double is queued per node. This file proves the queueing itself: that a
node's responses come back for that node, in order, as JSON the contract layer
in ``agent/contracts.py`` can parse, and that a call the double cannot place
falls back to the ordered script rather than guessing.

The full branch matrix over the assembled loop (six repair moves, budget
exhaustion, the no-progress stall, a malformed response at each contract) lands
here after the two tracks merge, once ``agent/loop.py`` exists to drive.
"""

import pytest

from ragfabric_core.agent.contracts import (
    AssessResponse,
    ContractViolation,
    PlanResponse,
    RepairResponse,
    parse_assess,
    parse_plan,
    parse_repair,
)
from ragfabric_core.agent.state import NodeName, RepairMove
from ragfabric_core.providers.base import LLMProvider, Message, ProviderError
from ragfabric_core.providers.offline import ScriptedLLMProvider

PLAN_JSON = {
    "sub_questions": [
        {"text": "what is the retry limit", "tool": "lexical_search", "why": "an exact setting"}
    ]
}
ASSESS_JSON = {
    "verdicts": [
        {
            "sub_question": "what is the retry limit",
            "answered": False,
            "missing": "the number itself",
        }
    ]
}
REPAIR_JSON = {"move": "switch_strategy", "rewritten_query": "retry limit", "why": "an identifier"}


def prompt(node: NodeName | str, body: str = "") -> list[Message]:
    """A message pair shaped like a node's real prompt: the node names itself."""
    return [
        Message(role="system", content=f"You are the {node} step of a retrieval agent."),
        Message(role="user", content=body or "what is the retry limit and who approves it"),
    ]


def test_the_extended_double_is_still_an_llm_provider() -> None:
    assert isinstance(ScriptedLLMProvider([]), LLMProvider)


def test_a_flat_script_still_replays_in_order() -> None:
    """Phase 3 and 4 tests build this double with a plain list. They keep working."""
    llm = ScriptedLLMProvider(["first", "second"])
    assert llm.complete(prompt(NodeName.PLAN)).text == "first"
    assert llm.complete(prompt(NodeName.ASSESS)).text == "second"


def test_a_queued_node_response_comes_back_for_that_node() -> None:
    llm = ScriptedLLMProvider(node_responses={NodeName.PLAN: [PLAN_JSON]})
    parsed = parse_plan(llm.complete(prompt(NodeName.PLAN)).text)
    assert isinstance(parsed, PlanResponse)
    assert parsed.sub_questions[0].tool == "lexical_search"


def test_each_contract_is_answered_from_its_own_queue() -> None:
    """The point of the whole extension: three nodes, three answers, one double."""
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [PLAN_JSON],
            NodeName.ASSESS: [ASSESS_JSON],
            NodeName.REPAIR: [REPAIR_JSON],
        }
    )
    plan = parse_plan(llm.complete(prompt(NodeName.PLAN)).text)
    assess = parse_assess(llm.complete(prompt(NodeName.ASSESS)).text)
    repair = parse_repair(llm.complete(prompt(NodeName.REPAIR)).text)
    assert isinstance(plan, PlanResponse)
    assert isinstance(assess, AssessResponse)
    assert isinstance(repair, RepairResponse)
    assert assess.verdicts[0].answered is False
    assert repair.move is RepairMove.SWITCH_STRATEGY


def test_a_node_queue_is_consumed_in_order() -> None:
    first = {"move": "broaden", "rewritten_query": "retries", "why": "nothing came back"}
    second = {"move": "abandon", "rewritten_query": None, "why": "still nothing"}
    llm = ScriptedLLMProvider(node_responses={NodeName.REPAIR: [first, second]})
    moves = [parse_repair(llm.complete(prompt(NodeName.REPAIR)).text).move for _ in range(2)]
    assert moves == [RepairMove.BROADEN, RepairMove.ABANDON]


def test_responses_can_be_queued_after_construction() -> None:
    llm = ScriptedLLMProvider()
    llm.queue(NodeName.PLAN, PLAN_JSON)
    assert isinstance(parse_plan(llm.complete(prompt(NodeName.PLAN)).text), PlanResponse)


def test_a_string_is_returned_untouched_so_malformed_output_can_be_scripted() -> None:
    """Every contract needs a malformed branch, so the double must not sanitise."""
    llm = ScriptedLLMProvider(node_responses={NodeName.PLAN: ["Sure! I can help with that."]})
    assert isinstance(parse_plan(llm.complete(prompt(NodeName.PLAN)).text), ContractViolation)


def test_a_call_that_names_no_node_falls_back_to_the_ordered_script() -> None:
    """Not every prompt announces itself, and guessing would be worse than a queue."""
    llm = ScriptedLLMProvider(["fallback"], node_responses={NodeName.PLAN: [PLAN_JSON]})
    assert llm.complete([Message(role="user", content="say something")]).text == "fallback"


def test_a_prompt_naming_two_nodes_falls_back_rather_than_guessing() -> None:
    messages = [Message(role="user", content="the plan step failed, now repair it")]
    llm = ScriptedLLMProvider(["fallback"], node_responses={NodeName.PLAN: [PLAN_JSON]})
    assert llm.complete(messages).text == "fallback"


def test_a_node_with_no_queue_falls_back_to_the_ordered_script() -> None:
    llm = ScriptedLLMProvider(["fallback"], node_responses={NodeName.PLAN: [PLAN_JSON]})
    assert llm.complete(prompt(NodeName.ASSESS)).text == "fallback"


def test_an_exhausted_node_queue_says_which_node_ran_out() -> None:
    """An unhelpful "script exhausted" in a six-branch suite costs more than it saves."""
    llm = ScriptedLLMProvider(node_responses={NodeName.ASSESS: [ASSESS_JSON]})
    llm.complete(prompt(NodeName.ASSESS))
    with pytest.raises(ProviderError) as info:
        llm.complete(prompt(NodeName.ASSESS))
    assert "assess" in str(info.value)


def test_calls_are_counted_per_node() -> None:
    llm = ScriptedLLMProvider(
        node_responses={NodeName.PLAN: [PLAN_JSON], NodeName.ASSESS: [ASSESS_JSON, ASSESS_JSON]}
    )
    llm.complete(prompt(NodeName.PLAN))
    llm.complete(prompt(NodeName.ASSESS))
    llm.complete(prompt(NodeName.ASSESS))
    assert llm.calls == 3
    assert llm.calls_by_node == {"plan": 1, "assess": 2}


def test_pending_reports_what_was_queued_and_never_used() -> None:
    """A branch test that leaves a response unconsumed took a path it did not mean to."""
    llm = ScriptedLLMProvider(
        ["spare"], node_responses={NodeName.PLAN: [PLAN_JSON], NodeName.REPAIR: [REPAIR_JSON]}
    )
    llm.complete(prompt(NodeName.PLAN))
    assert llm.pending() == {"repair": 1, "": 1}


def test_a_plain_string_queue_entry_is_not_re_encoded_as_json() -> None:
    llm = ScriptedLLMProvider(node_responses={NodeName.GENERATE: ["the retry limit is five [1]"]})
    assert llm.complete(prompt(NodeName.GENERATE)).text == "the retry limit is five [1]"


def test_a_node_key_may_be_a_plain_string() -> None:
    llm = ScriptedLLMProvider(node_responses={"plan": [PLAN_JSON]})
    assert isinstance(parse_plan(llm.complete(prompt("plan")).text), PlanResponse)


def test_an_unknown_node_name_is_refused_at_construction() -> None:
    """A typo in a queue key would look like a node that never got called."""
    with pytest.raises(ValueError, match="assses"):
        ScriptedLLMProvider(node_responses={"assses": [ASSESS_JSON]})


def test_streaming_still_replays_the_ordered_script() -> None:
    llm = ScriptedLLMProvider(["two words"])
    assert "".join(llm.stream(prompt(NodeName.GENERATE))) == "two words"


def test_streaming_reads_a_node_queue_too() -> None:
    """Generation streams, so the generate node must be scriptable the same way."""
    llm = ScriptedLLMProvider(node_responses={NodeName.GENERATE: ["the retry limit is five [1]"]})
    assert "".join(llm.stream(prompt(NodeName.GENERATE))) == "the retry limit is five [1]"


def test_streaming_an_exhausted_script_still_yields_nothing() -> None:
    assert list(ScriptedLLMProvider().stream(prompt(NodeName.GENERATE))) == []
