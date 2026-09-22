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

The second half of this file is the full branch matrix over the assembled
loop: each of the six repair moves, both ways the budget stops a run, the
no-progress stall, a malformed response at each of the three contracts, a
sub-question abandoned with a reason, and a fully resolved multi-sub-question
run. Every one of them is driven by the queued double alone, so the whole
matrix runs with no network and no database. ``test_the_matrix_needs_no_socket
_and_no_database_session`` holds that shut rather than leaving it a claim in a
docstring.
"""

import socket

import pytest
from agent_doubles import FakeTool, SequenceTool, chunk
from agent_doubles import ctx as make_ctx

from ragfabric_core.agent.contracts import (
    AssessResponse,
    ContractViolation,
    PlanResponse,
    RepairResponse,
    parse_assess,
    parse_plan,
    parse_repair,
)
from ragfabric_core.agent.loop import (
    STOP_BUDGET,
    STOP_NO_PROGRESS,
    STOP_RESOLVED,
    AgentRun,
    run_agent,
)
from ragfabric_core.agent.state import NodeName, RepairMove, SubQuestionStatus
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


# ---------------------------------------------------------------------------
# The branch matrix over the assembled loop.
#
# Each test below drives ``run_agent`` with nothing but queued JSON and fake
# tools, so the branch it names is the branch that ran. The queues are the
# reason that claim holds: a response queued for ``repair`` can only be
# returned to a repair call, so a run that took a different path raises
# "script exhausted for node repair" instead of quietly passing.
# ---------------------------------------------------------------------------


def plan_json(*pairs: tuple[str, str]) -> dict:
    return {"sub_questions": [{"text": text, "tool": tool, "why": ""} for text, tool in pairs]}


def assess_json(*verdicts: tuple[str, bool, str | None]) -> dict:
    return {
        "verdicts": [
            {"sub_question": text, "answered": answered, "missing": missing}
            for text, answered, missing in verdicts
        ]
    }


def repair_json(move: str, rewritten: str | None = None, why: str = "") -> dict:
    return {"move": move, "rewritten_query": rewritten, "why": why}


def spans(run: AgentRun, name: str) -> list:
    return [span for span in run.trace if span.name == name]


def repair_moves(run: AgentRun) -> list:
    return [span.attributes["move"] for span in spans(run, "repair")]


def registry(*tools) -> dict:
    return {tool.name: tool for tool in tools}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every socket for the whole module, not just the test that says so.

    The branch matrix exists to prove the agent can be driven offline. A
    fixture that only guarded one test would leave the other twenty free to
    reach for a provider, and the suite would still be green on a machine with
    a model server running.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("the agent branch matrix must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def test_the_matrix_needs_no_socket_and_no_database_session() -> None:
    """No network and no database, asserted rather than asserted about.

    The socket ban comes from the autouse fixture above. The database session
    factory is broken deliberately for the duration of this run, so a tool or
    a node that reached for one would fail here rather than in whichever
    deployment first ran the agent without a database in front of it.
    """
    import ragfabric_core.db.session as session_module

    def refuse_session(*args, **kwargs):
        raise AssertionError("the agent branch matrix must not open a database session")

    original = session_module.SessionLocal
    session_module.SessionLocal = refuse_session
    try:
        tool = FakeTool("semantic_search", chunks=[chunk(1, text="the limit is five attempts")])
        llm = ScriptedLLMProvider(
            node_responses={
                NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
                NodeName.ASSESS: [assess_json(("what is the retry limit", True, None))],
            }
        )
        run = run_agent("what is the retry limit", llm=llm, tools=registry(tool), ctx=make_ctx())
    finally:
        session_module.SessionLocal = original

    assert run.stop_reason == STOP_RESOLVED
    assert llm.pending() == {}


def test_a_fully_resolved_multi_sub_question_run() -> None:
    semantic = FakeTool("semantic_search", chunks=[chunk(1, text="the limit is five attempts")])
    lexical = FakeTool("lexical_search", chunks=[chunk(2, text="the owner signs it off")])
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [
                plan_json(
                    ("what is the retry limit", "semantic_search"),
                    ("who signs off the change", "lexical_search"),
                )
            ],
            NodeName.ASSESS: [
                assess_json(
                    ("what is the retry limit", True, None),
                    ("who signs off the change", True, None),
                )
            ],
        }
    )

    run = run_agent(
        "what is the retry limit and who signs off the change",
        llm=llm,
        tools=registry(semantic, lexical),
        ctx=make_ctx(),
    )

    assert run.stop_reason == STOP_RESOLVED
    assert [sq.status for sq in run.state.sub_questions] == [
        SubQuestionStatus.ANSWERED,
        SubQuestionStatus.ANSWERED,
    ]
    assert run.retrieval_calls == 2
    assert {c.chunk_id for c in run.chunks} == {1, 2}
    assert run.state.llm_calls == 2
    assert llm.pending() == {}


def test_the_broaden_branch_relaxes_the_query_and_widens_the_candidate_set() -> None:
    tool = SequenceTool("semantic_search", [[], [chunk(1, text="the limit is five attempts")]])
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [
                plan_json(("what is the exact retry limit for the payments API", "semantic_search"))
            ],
            NodeName.ASSESS: [
                assess_json(
                    ("what is the exact retry limit for the payments API", False, "the number")
                ),
                assess_json(("what is the exact retry limit for the payments API", True, None)),
            ],
            NodeName.REPAIR: [repair_json("broaden")],
        }
    )

    run = run_agent(
        "what is the exact retry limit for the payments API",
        llm=llm,
        tools=registry(tool),
        ctx=make_ctx(top_k=5),
    )

    assert repair_moves(run) == ["broaden"]
    assert tool.queries == [
        "what is the exact retry limit for the payments API",
        "what is the retry limit",
    ]
    assert tool.contexts[1].params.top_k == 10
    assert run.stop_reason == STOP_RESOLVED


def test_the_narrow_branch_adds_the_missing_constraint_to_the_query() -> None:
    tool = SequenceTool(
        "semantic_search",
        [[chunk(1, text="retries are discussed at length")], [chunk(2, text="five attempts")]],
    )
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
            NodeName.ASSESS: [
                assess_json(("what is the retry limit", False, "the number of retries allowed")),
                assess_json(("what is the retry limit", True, None)),
            ],
            NodeName.REPAIR: [repair_json("narrow")],
        }
    )

    run = run_agent(
        "what is the retry limit", llm=llm, tools=registry(tool), ctx=make_ctx(top_k=20)
    )

    assert repair_moves(run) == ["narrow"]
    assert tool.queries[1] == "what is the retry limit the number of retries allowed"
    assert tool.contexts[1].params.top_k == 5
    assert run.stop_reason == STOP_RESOLVED


def test_the_switch_strategy_branch_flips_the_tool_the_next_retrieval_uses() -> None:
    lexical = FakeTool("lexical_search", chunks=[chunk(1, text="a near miss")])
    semantic = FakeTool("semantic_search", chunks=[chunk(2, text="five attempts")])
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "lexical_search"))],
            NodeName.ASSESS: [
                assess_json(("what is the retry limit", False, "the number")),
                assess_json(("what is the retry limit", True, None)),
            ],
            NodeName.REPAIR: [repair_json("switch_strategy")],
        }
    )

    run = run_agent(
        "what is the retry limit", llm=llm, tools=registry(lexical, semantic), ctx=make_ctx()
    )

    assert repair_moves(run) == ["switch_strategy"]
    assert len(lexical.queries) == 1
    assert len(semantic.queries) == 1
    assert run.state.sub_questions[0].tool == "semantic_search"
    assert run.stop_reason == STOP_RESOLVED


def test_the_decompose_branch_splits_the_sub_question_and_retires_the_parent() -> None:
    tool = SequenceTool(
        "semantic_search",
        [
            [chunk(1, text="a long policy document")],
            [chunk(2, text="five attempts")],
            [chunk(3, text="the owner signs it off")],
        ],
    )
    compound = "what is the retry limit and who signs off the change"
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json((compound, "semantic_search"))],
            NodeName.ASSESS: [
                assess_json((compound, False, "both halves")),
                assess_json(
                    ("what is the retry limit", True, None),
                    ("who signs off the change", True, None),
                ),
            ],
            NodeName.REPAIR: [repair_json("decompose")],
        }
    )

    run = run_agent(compound, llm=llm, tools=registry(tool), ctx=make_ctx())

    assert repair_moves(run) == ["decompose"]
    assert [sq.text for sq in run.state.sub_questions] == [
        compound,
        "what is the retry limit",
        "who signs off the change",
    ]
    assert run.state.sub_questions[0].status is SubQuestionStatus.ABANDONED
    assert "decomposed into 2" in (run.state.sub_questions[0].reason or "")
    assert run.stop_reason == STOP_RESOLVED


def test_the_fetch_document_branch_pulls_the_matched_document_whole() -> None:
    semantic = FakeTool(
        "semantic_search", chunks=[chunk(1, document_id=7, text="a fragment", score=0.9)]
    )
    fetch = FakeTool("fetch_document", chunks=[chunk(2, document_id=7, text="five attempts")])
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
            NodeName.ASSESS: [
                assess_json(("what is the retry limit", False, "the number")),
                assess_json(("what is the retry limit", True, None)),
            ],
            NodeName.REPAIR: [repair_json("fetch_document")],
        }
    )

    run = run_agent(
        "what is the retry limit", llm=llm, tools=registry(semantic, fetch), ctx=make_ctx()
    )

    assert repair_moves(run) == ["fetch_document"]
    assert fetch.queries == ["7"]
    assert run.state.sub_questions[0].tool == "fetch_document"
    assert run.stop_reason == STOP_RESOLVED


def test_the_abandon_branch_records_a_reason_and_still_stops_as_resolved() -> None:
    """Resolved is not a synonym for answered, and the report has to show that."""
    tool = SequenceTool(
        "semantic_search",
        [
            [chunk(1, text="a near miss")],
            [chunk(2, text="another near miss")],
        ],
    )
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
            NodeName.ASSESS: [
                assess_json(("what is the retry limit", False, "the number")),
                assess_json(("what is the retry limit", False, "the number")),
            ],
            NodeName.REPAIR: [
                repair_json("broaden"),
                repair_json("abandon", why="nothing in the corpus states the limit"),
            ],
        }
    )

    run = run_agent("what is the retry limit", llm=llm, tools=registry(tool), ctx=make_ctx())

    assert repair_moves(run) == ["broaden", "abandon"]
    assert run.state.sub_questions[0].status is SubQuestionStatus.ABANDONED
    assert run.state.sub_questions[0].reason == "nothing in the corpus states the limit"
    assert run.stop_reason == STOP_RESOLVED


def test_an_abandon_proposed_before_anything_was_tried_is_refused() -> None:
    """The guard that stops a model turning the agent into a plain retriever."""
    tool = SequenceTool(
        "semantic_search",
        [[chunk(1, text="a near miss")], [chunk(2, text="five attempts")]],
    )
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
            NodeName.ASSESS: [
                assess_json(("what is the retry limit", False, "the number")),
                assess_json(("what is the retry limit", True, None)),
            ],
            NodeName.REPAIR: [repair_json("abandon", why="I give up")],
        }
    )

    run = run_agent("what is the retry limit", llm=llm, tools=registry(tool), ctx=make_ctx())

    assert repair_moves(run) == ["narrow"]
    assert run.state.sub_questions[0].status is SubQuestionStatus.ANSWERED


def test_the_loop_stops_on_the_iteration_cap() -> None:
    tool = SequenceTool(
        "semantic_search",
        [[chunk(1, text="a near miss")], [chunk(2, text="another near miss")]],
    )
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
            NodeName.ASSESS: [assess_json(("what is the retry limit", False, "the number"))],
            NodeName.REPAIR: [repair_json("broaden")],
        }
    )

    run = run_agent(
        "what is the retry limit",
        llm=llm,
        tools=registry(tool),
        ctx=make_ctx(),
        max_iterations=1,
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_iterations of 1" in run.stop_detail
    assert run.state.iterations == 1


def test_the_loop_stops_when_a_spend_is_refused() -> None:
    """The other budget: the call cap, refused at the node that would cross it."""
    tool = FakeTool("semantic_search", chunks=[chunk(1, text="a near miss")])
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
            NodeName.ASSESS: [assess_json(("what is the retry limit", False, "the number"))],
            NodeName.REPAIR: [repair_json("broaden")],
        }
    )

    run = run_agent(
        "what is the retry limit",
        llm=llm,
        tools=registry(tool),
        ctx=make_ctx(max_llm_calls=2),
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_llm_calls of 2" in run.stop_detail
    assert run.state.llm_calls == 2
    assert llm.pending() == {"repair": 1}


def test_the_loop_stops_on_a_stall_rather_than_re_reading_the_same_chunk() -> None:
    tool = FakeTool("semantic_search", chunks=[chunk(1, text="a near miss")])
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
            NodeName.ASSESS: [assess_json(("what is the retry limit", False, "the number"))],
            NodeName.REPAIR: [repair_json("broaden")],
        }
    )

    run = run_agent("what is the retry limit", llm=llm, tools=registry(tool), ctx=make_ctx())

    assert run.stop_reason == STOP_NO_PROGRESS
    assert run.state.iterations == 2
    assert len(spans(run, "assess")) == 1


def test_a_malformed_plan_does_not_fail_the_request() -> None:
    tool = FakeTool("semantic_search", chunks=[chunk(1, text="five attempts")])
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: ["Sure! Let me think about that one."],
            NodeName.ASSESS: [assess_json(("what is the retry limit", True, None))],
        }
    )

    run = run_agent("what is the retry limit", llm=llm, tools=registry(tool), ctx=make_ctx())

    plan_span = spans(run, "plan")[0]
    assert plan_span.attributes["fallback"] is True
    assert plan_span.attributes["violation"]
    assert [sq.text for sq in run.state.sub_questions] == ["what is the retry limit"]
    assert run.stop_reason == STOP_RESOLVED


def test_a_malformed_assessment_leaves_the_sub_question_open() -> None:
    tool = SequenceTool(
        "semantic_search",
        [[chunk(1, text="a near miss")], [chunk(2, text="five attempts")]],
    )
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
            NodeName.ASSESS: [
                "I am not sure how to judge that.",
                assess_json(("what is the retry limit", True, None)),
            ],
            NodeName.REPAIR: [repair_json("narrow", rewritten="retry limit attempts")],
        }
    )

    run = run_agent("what is the retry limit", llm=llm, tools=registry(tool), ctx=make_ctx())

    assert spans(run, "assess")[0].attributes["violation"]
    assert spans(run, "assess")[0].attributes["unjudged"] == 1
    assert repair_moves(run) == ["narrow"]
    assert run.stop_reason == STOP_RESOLVED


def test_a_malformed_repair_still_makes_a_move() -> None:
    tool = SequenceTool(
        "semantic_search",
        [[chunk(1, text="a near miss")], [chunk(2, text="five attempts")]],
    )
    llm = ScriptedLLMProvider(
        node_responses={
            NodeName.PLAN: [plan_json(("what is the retry limit", "semantic_search"))],
            NodeName.ASSESS: [
                assess_json(("what is the retry limit", False, "the number")),
                assess_json(("what is the retry limit", True, None)),
            ],
            NodeName.REPAIR: ["I would try searching again, maybe?"],
        }
    )

    run = run_agent("what is the retry limit", llm=llm, tools=registry(tool), ctx=make_ctx())

    repair_span = spans(run, "repair")[0]
    assert repair_span.attributes["violation"]
    assert repair_span.attributes["proposed"] is None
    assert repair_span.attributes["move"] == "narrow"
    assert run.stop_reason == STOP_RESOLVED
