"""Retrieval, the evidence pool and the progress check.

This is the file that guards the failure the textbook agentic design has no
answer to. An agent that re-runs a query, gets the same chunks back and counts
that as an iteration will spend its whole budget learning nothing, and it will
do it while every counter looks healthy: tools were called, rows were returned,
latency was spent. The only signal that separates work from motion is whether
the evidence pool gained a chunk id it did not already hold, so that is what
these tests assert on, and they assert it in a way that a count based check
cannot satisfy.
"""

from __future__ import annotations

from agent_doubles import FakeTool, SequenceTool, chunk, ctx, restricted_ctx

from ragfabric_core.agent.nodes import RetrievalOverride, retrieve
from ragfabric_core.agent.state import AgentState, SubQuestion


def state_with(*sub_questions: SubQuestion, question: str = "the question") -> AgentState:
    return AgentState(question=question, sub_questions=list(sub_questions))


def test_an_iteration_that_adds_no_new_chunks_is_marked_no_progress():
    """The same chunks returned a second time are not a second iteration's worth.

    The tool here returns a non-empty list on every call, so anything that
    measures progress by how much came back would report progress forever.
    """
    tool = FakeTool("semantic_search", chunks=[chunk(1), chunk(2), chunk(3)])
    tools = {"semantic_search": tool}
    state = state_with(SubQuestion(text="q", tool="semantic_search"))

    first = retrieve(state, tools=tools, ctx=ctx())
    second = retrieve(state, tools=tools, ctx=ctx())

    assert first.made_progress is True
    assert first.new_chunk_ids == {1, 2, 3}
    assert second.made_progress is False
    assert second.new_chunk_ids == set()
    # The evidence was really returned again, so "nothing came back" is not
    # what is being detected here.
    assert second.tool_calls[0].returned == 3
    assert second.span.attributes["progress"] is False


def test_evidence_is_deduplicated_across_iterations():
    tool = SequenceTool(
        "semantic_search",
        batches=[[chunk(1), chunk(2)], [chunk(2), chunk(3)]],
    )
    tools = {"semantic_search": tool}
    state = state_with(SubQuestion(text="q", tool="semantic_search"))

    retrieve(state, tools=tools, ctx=ctx())
    second = retrieve(state, tools=tools, ctx=ctx())

    assert sorted(state.evidence) == [1, 2, 3]
    assert second.new_chunk_ids == {3}
    assert second.made_progress is True


def test_only_open_sub_questions_are_retrieved_for():
    """Re-retrieving a resolved part is what judging evidence globally looks like."""
    semantic = FakeTool("semantic_search", chunks=[chunk(1)])
    lexical = FakeTool("lexical_search", chunks=[chunk(2)])
    tools = {"semantic_search": semantic, "lexical_search": lexical}
    answered = SubQuestion(text="already known", tool="semantic_search")
    answered.mark_answered()
    abandoned = SubQuestion(text="given up on", tool="semantic_search")
    abandoned.abandon(reason="no evidence after three moves")
    state = state_with(
        answered,
        abandoned,
        SubQuestion(text="still open", tool="lexical_search"),
    )

    outcome = retrieve(state, tools=tools, ctx=ctx())

    assert semantic.queries == []
    assert lexical.queries == ["still open"]
    assert [call.sub_question_index for call in outcome.tool_calls] == [2]


def test_the_overridden_query_is_what_reaches_the_tool():
    """A repaired sub-question retrieves on its new query, not the planned wording."""
    tool = FakeTool("semantic_search", chunks=[chunk(1)])
    tools = {"semantic_search": tool}
    state = state_with(SubQuestion(text="the original wording", tool="semantic_search"))

    retrieve(
        state,
        tools=tools,
        ctx=ctx(top_k=5),
        overrides={0: RetrievalOverride(query="the broadened wording", top_k=10)},
    )

    assert tool.queries == ["the broadened wording"]
    assert tool.contexts[0].params.top_k == 10
    # The plan's wording is the record of what was asked and is not rewritten.
    assert state.sub_questions[0].text == "the original wording"


def test_the_access_filter_survives_a_widened_top_k():
    """Broadening must widen the search, never what the principal may see."""
    tool = FakeTool("semantic_search", chunks=[chunk(1)])
    tools = {"semantic_search": tool}
    state = state_with(SubQuestion(text="q", tool="semantic_search"))
    context = restricted_ctx(document_ids={7})

    retrieve(
        state,
        tools=tools,
        ctx=context,
        overrides={0: RetrievalOverride(query="q broadened", top_k=50)},
    )

    assert tool.contexts[0].access_filter == context.access_filter
    assert tool.contexts[0].access_filter.document_ids == frozenset({7})
    assert tool.contexts[0].principal == context.principal


def test_every_open_sub_question_gets_its_own_tool_call():
    semantic = FakeTool("semantic_search", chunks=[chunk(1)])
    lexical = FakeTool("lexical_search", chunks=[chunk(2)])
    tools = {"semantic_search": semantic, "lexical_search": lexical}
    state = state_with(
        SubQuestion(text="the concept", tool="semantic_search"),
        SubQuestion(text="ERR_4019", tool="lexical_search"),
    )

    outcome = retrieve(state, tools=tools, ctx=ctx())

    assert [call.tool for call in outcome.tool_calls] == ["semantic_search", "lexical_search"]
    assert outcome.returned_by_sub_question == {0: 1, 1: 1}
    assert outcome.new_chunk_ids == {1, 2}


def test_a_sub_question_whose_tool_is_not_registered_returns_nothing_rather_than_raising():
    tools = {"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])}
    state = state_with(SubQuestion(text="q", tool="fetch_document"))

    outcome = retrieve(state, tools=tools, ctx=ctx())

    assert outcome.tool_calls == []
    assert outcome.made_progress is False
