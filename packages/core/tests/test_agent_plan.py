"""The plan node: one call that decomposes the question and routes each part.

Two properties are worth more than the rest. Decomposition is not mandatory, so
a simple question costs exactly one sub-question and one retrieval rather than
three of each. And a plan the model returns badly cannot stop the request: a
violation falls back to a single sub-question over the whole question, which is
the behaviour of a plain retriever and is always better than an error.
"""

from __future__ import annotations

import json

from agent_doubles import FakeTool, RecordingLLM

from ragfabric_core.agent.nodes import plan
from ragfabric_core.agent.state import AgentState, NodeName, SubQuestionStatus


def tools() -> dict:
    return {
        "semantic_search": FakeTool(
            "semantic_search", description="meaning based search over embeddings"
        ),
        "lexical_search": FakeTool("lexical_search", description="exact word and BM25 search"),
        "fetch_document": FakeTool("fetch_document", description="the whole document in order"),
    }


def plan_json(*pairs: tuple[str, str]) -> str:
    return json.dumps(
        {"sub_questions": [{"text": text, "tool": tool, "why": ""} for text, tool in pairs]}
    )


def test_a_simple_question_yields_one_sub_question():
    """Forcing decomposition on a single fact question spends calls to learn nothing."""
    state = AgentState(question="what is the retry limit")
    llm = RecordingLLM(plan_json(("what is the retry limit", "lexical_search")))

    outcome = plan(state, llm=llm, tools=tools())

    assert len(state.sub_questions) == 1
    assert state.sub_questions[0].text == "what is the retry limit"
    assert state.sub_questions[0].status is SubQuestionStatus.OPEN
    assert outcome.violation is None
    assert state.llm_calls == 1
    assert state.node_llm_calls[NodeName.PLAN] == 1


def test_a_comparison_question_is_decomposed():
    state = AgentState(question="how do the refund and cancellation policies differ")
    llm = RecordingLLM(
        plan_json(
            ("what is the refund policy", "semantic_search"),
            ("what is the cancellation policy", "semantic_search"),
        )
    )

    plan(state, llm=llm, tools=tools())

    assert [sq.text for sq in state.sub_questions] == [
        "what is the refund policy",
        "what is the cancellation policy",
    ]
    assert len(state.open_sub_questions()) == 2


def test_an_identifier_question_is_routed_to_lexical_search():
    """The routing rule has to reach the model, not just survive its answer."""
    state = AgentState(question="what does error ERR_4019 mean")
    llm = RecordingLLM(plan_json(("what does error ERR_4019 mean", "lexical_search")))

    plan(state, llm=llm, tools=tools())

    assert state.sub_questions[0].tool == "lexical_search"
    prompt = llm.last_prompt()
    assert "lexical_search" in prompt
    assert "semantic_search" in prompt
    assert "identifier" in prompt.lower()


def test_a_malformed_plan_falls_back_to_a_single_sub_question():
    """A model that cannot produce a plan must not be able to fail the request."""
    state = AgentState(question="what is the retry limit")
    llm = RecordingLLM("I think we should look at the policy document first.")

    outcome = plan(state, llm=llm, tools=tools())

    assert outcome.violation is not None
    assert outcome.violation.contract == "plan"
    assert len(state.sub_questions) == 1
    assert state.sub_questions[0].text == "what is the retry limit"
    assert state.sub_questions[0].tool in tools()
    assert outcome.span.attributes["fallback"] is True


def test_an_unknown_tool_name_falls_back_to_a_registered_tool():
    """A model may name a tool that does not exist. Retrieval still has to run."""
    state = AgentState(question="what is the retry limit")
    llm = RecordingLLM(plan_json(("what is the retry limit", "web_search")))

    plan(state, llm=llm, tools=tools())

    assert state.sub_questions[0].tool in tools()


def test_repeated_sub_questions_are_collapsed():
    """Two identical sub-questions would retrieve the same evidence twice."""
    state = AgentState(question="what is the retry limit")
    llm = RecordingLLM(
        plan_json(
            ("what is the retry limit", "lexical_search"),
            ("What Is The Retry Limit", "semantic_search"),
        )
    )

    plan(state, llm=llm, tools=tools())

    assert len(state.sub_questions) == 1


def test_the_plan_node_appends_a_trace_span():
    state = AgentState(question="what is the retry limit")
    llm = RecordingLLM(plan_json(("what is the retry limit", "semantic_search")))

    outcome = plan(state, llm=llm, tools=tools())

    assert outcome.span.name == "plan"
    assert outcome.span.attributes["sub_questions"] == 1
