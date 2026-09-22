"""The loop: three ways to stop, and a trace that says which one fired.

Issue #6 calls stopping the hard part, and these tests are why. An agent that
cannot stop honestly has three ways to lie about it: it can keep iterating on
evidence it already has, it can run out of budget and present the result as if
it were finished, or it can stop for a reason nobody recorded. Each stop
condition is therefore asserted by the reason it reports, not just by the fact
that the call returned.

The stop reason is set at the branch that decides it. The LangGraph spike lost
it precisely by separating those two, returning ``stop_reason=None`` from a
conditional edge that only knew which node came next, and that is the failure
these tests are shaped to catch.
"""

from __future__ import annotations

import json

from agent_doubles import FakeTool, RecordingLLM, SequenceTool, chunk, ctx

from ragfabric_core.agent.loop import (
    STOP_BUDGET,
    STOP_NO_PROGRESS,
    STOP_RESOLVED,
    run_agent,
)
from ragfabric_core.agent.state import RepairMove, SubQuestionStatus


def plan_json(*pairs: tuple[str, str]) -> str:
    return json.dumps(
        {"sub_questions": [{"text": text, "tool": tool, "why": ""} for text, tool in pairs]}
    )


def assessed(text: str, answered: bool, missing: str | None = None) -> str:
    return json.dumps(
        {"verdicts": [{"sub_question": text, "answered": answered, "missing": missing}]}
    )


def repair_json(move: str, query: str | None = None) -> str:
    return json.dumps({"move": move, "rewritten_query": query, "why": "because"})


def test_the_loop_terminates_when_all_sub_questions_resolve():
    tools = {"semantic_search": FakeTool("semantic_search", chunks=[chunk(1), chunk(2)])}
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assessed("what is the retry limit", True),
    )

    run = run_agent("what is the retry limit", llm=llm, tools=tools, ctx=ctx())

    assert run.stop_reason == STOP_RESOLVED
    assert run.state.stop_reason == STOP_RESOLVED
    assert run.state.all_resolved()
    assert run.state.iterations == 1
    assert sorted(run.state.evidence) == [1, 2]


def test_the_loop_terminates_on_no_progress():
    """A tool that keeps returning the same chunks must not buy another iteration."""
    tools = {"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])}
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assessed("what is the retry limit", False, "the number of retries"),
        repair_json("broaden"),
        assessed("what is the retry limit", False, "the number of retries"),
        repair_json("switch_strategy"),
        assessed("what is the retry limit", False, "the number of retries"),
        repair_json("narrow"),
    )

    run = run_agent("what is the retry limit", llm=llm, tools=tools, ctx=ctx(), max_iterations=4)

    assert run.stop_reason == STOP_NO_PROGRESS
    assert run.state.iterations == 2
    assert run.state.sub_questions[0].status is SubQuestionStatus.OPEN


def test_the_loop_terminates_on_budget():
    """The cap is refused before it is crossed, and the refusal is the stop reason."""
    tools = {
        "semantic_search": SequenceTool(
            "semantic_search",
            batches=[[chunk(1)], [chunk(2)], [chunk(3)], [chunk(4)], [chunk(5)]],
        )
    }
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assessed("what is the retry limit", False, "the number"),
        repair_json("broaden"),
        assessed("what is the retry limit", False, "the number"),
        repair_json("narrow"),
        assessed("what is the retry limit", False, "the number"),
    )

    run = run_agent(
        "what is the retry limit",
        llm=llm,
        tools=tools,
        ctx=ctx(max_llm_calls=4),
        max_iterations=10,
    )

    assert run.stop_reason == STOP_BUDGET
    assert run.state.llm_calls <= 4
    assert "max_llm_calls" in run.stop_detail


def test_the_loop_terminates_on_the_iteration_cap():
    tools = {
        "semantic_search": SequenceTool(
            "semantic_search",
            batches=[[chunk(i)] for i in range(1, 9)],
        )
    }
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assessed("what is the retry limit", False, "the number"),
        repair_json("broaden"),
        assessed("what is the retry limit", False, "the number"),
        repair_json("narrow"),
    )

    run = run_agent("what is the retry limit", llm=llm, tools=tools, ctx=ctx(), max_iterations=2)

    assert run.stop_reason == STOP_BUDGET
    assert run.state.iterations == 2
    assert "max_iterations" in run.stop_detail


def test_the_stop_reason_is_recorded():
    """Recorded at the branch, on the state and on the run, never inferred later."""
    tools = {"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])}
    llm = RecordingLLM(
        plan_json(("q", "semantic_search")),
        assessed("q", True),
    )

    run = run_agent("q", llm=llm, tools=tools, ctx=ctx())

    assert run.stop_reason in {STOP_RESOLVED, STOP_BUDGET, STOP_NO_PROGRESS}
    assert run.state.stop_reason == run.stop_reason
    assert run.trace[-1].name == "finalize"
    assert run.trace[-1].attributes["stop_reason"] == run.stop_reason


def test_every_node_appends_a_trace_span():
    tools = {"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])}
    llm = RecordingLLM(
        plan_json(("q", "semantic_search")),
        assessed("q", False, "the number"),
        repair_json("broaden"),
        assessed("q", False, "the number"),
        repair_json("narrow"),
    )

    run = run_agent("q", llm=llm, tools=tools, ctx=ctx(), max_iterations=2)

    names = [span.name for span in run.trace]
    assert names[0] == "plan"
    assert names[-1] == "finalize"
    for expected in ("retrieve", "assess", "repair"):
        assert expected in names
    assert all(span.duration_ms >= 0 for span in run.trace)


def test_a_repair_changes_the_query_that_is_retrieved_next():
    tool = SequenceTool("semantic_search", batches=[[chunk(1)], [chunk(2)]])
    llm = RecordingLLM(
        plan_json(("what is the exact retry limit for the payments API", "semantic_search")),
        assessed("what is the exact retry limit for the payments API", False, "the number"),
        repair_json("broaden"),
        assessed("what is the exact retry limit for the payments API", True),
    )

    run = run_agent("q", llm=llm, tools={"semantic_search": tool}, ctx=ctx(), max_iterations=3)

    assert len(tool.queries) == 2
    assert tool.queries[1] != tool.queries[0]
    assert run.stop_reason == STOP_RESOLVED
    assert run.state.sub_questions[0].attempts[0].move is RepairMove.BROADEN


def test_a_decomposing_repair_adds_the_new_sub_questions_to_the_ledger():
    tool = SequenceTool("semantic_search", batches=[[chunk(1)], [chunk(2)], [chunk(3)]])
    question = "what is the retry limit and what is the backoff interval"
    llm = RecordingLLM(
        plan_json((question, "semantic_search")),
        assessed(question, False, "both numbers"),
        repair_json("decompose"),
        json.dumps(
            {
                "verdicts": [
                    {"sub_question": "what is the retry limit", "answered": True},
                    {"sub_question": "what is the backoff interval", "answered": True},
                ]
            }
        ),
    )

    run = run_agent(question, llm=llm, tools={"semantic_search": tool}, ctx=ctx())

    assert len(run.state.sub_questions) == 3
    assert run.state.sub_questions[0].status is SubQuestionStatus.ABANDONED
    assert run.stop_reason == STOP_RESOLVED


def test_an_abandoned_sub_question_resolves_the_loop_with_its_reason():
    tool = SequenceTool("semantic_search", batches=[[chunk(1)], [chunk(2)]])
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assessed("what is the retry limit", False, "the number"),
        repair_json("broaden"),
        assessed("what is the retry limit", False, "the number"),
        repair_json("abandon"),
    )

    run = run_agent("what is the retry limit", llm=llm, tools={"semantic_search": tool}, ctx=ctx())

    assert run.stop_reason == STOP_RESOLVED
    assert run.state.sub_questions[0].status is SubQuestionStatus.ABANDONED
    assert run.state.sub_questions[0].reason


def test_a_malformed_repair_still_makes_a_move():
    """An unreadable repair response must not cost an iteration for nothing."""
    tool = SequenceTool("semantic_search", batches=[[chunk(1)], [chunk(2)]])
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assessed("what is the retry limit", False, "the number"),
        "I would suggest looking somewhere else entirely.",
        assessed("what is the retry limit", True),
    )

    run = run_agent("what is the retry limit", llm=llm, tools={"semantic_search": tool}, ctx=ctx())

    assert run.state.sub_questions[0].attempts
    assert run.stop_reason == STOP_RESOLVED


def test_the_access_filter_is_the_callers_on_every_tool_call():
    tool = SequenceTool("semantic_search", batches=[[chunk(1)], [chunk(2)]])
    llm = RecordingLLM(
        plan_json(("q", "semantic_search")),
        assessed("q", False, "the number"),
        repair_json("broaden"),
        assessed("q", True),
    )
    context = ctx()

    run_agent("q", llm=llm, tools={"semantic_search": tool}, ctx=context)

    assert len(tool.contexts) == 2
    assert all(seen.access_filter == context.access_filter for seen in tool.contexts)
    assert all(seen.principal == context.principal for seen in tool.contexts)


def test_the_tokens_the_model_reported_are_accumulated():
    tools = {"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])}
    llm = RecordingLLM(plan_json(("q", "semantic_search")), assessed("q", True))

    run = run_agent("q", llm=llm, tools=tools, ctx=ctx())

    assert run.input_tokens > 0
    assert run.output_tokens > 0
    assert run.retrieval_calls == 1
