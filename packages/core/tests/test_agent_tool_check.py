import time

from agent_doubles import FakeTool

from ragfabric_core.agent.nodes import check_tools
from ragfabric_core.agent.state import AgentState, SubQuestion

TOOLS = {name: FakeTool(name) for name in ("semantic_search", "lexical_search", "graph_search")}
RELATIONS = ["REPORTS_TO", "OWNS"]


def state_with(*pairs):
    state = AgentState(question="q", max_llm_calls=5)
    state.sub_questions = [SubQuestion(text=text, tool=tool) for text, tool in pairs]
    return state


def test_a_decisive_signal_overrides_the_planner_and_is_traced():
    state = state_with(("What does ERR_QUOTA_4419 mean?", "semantic_search"))
    outcome = check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "lexical_search"
    assert outcome.span.name == "tool_check"
    assert outcome.span.attributes["overrides"] == 1
    assert "semantic_search->lexical_search" in outcome.span.attributes["detail"]


def test_a_paraphrase_planned_to_lexical_is_corrected_to_semantic():
    state = state_with(("How many days of unused annual leave carry forward?", "lexical_search"))
    outcome = check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "semantic_search"
    assert outcome.span.attributes["detail"] == "0:lexical_search->semantic_search(no_exact_terms)"


def test_an_identifier_planned_to_lexical_is_left_alone():
    state = state_with(("What does ERR_QUOTA_4419 mean?", "lexical_search"))
    outcome = check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "lexical_search"
    assert outcome.span.attributes["overrides"] == 0


def test_a_plain_question_planned_to_graph_is_left_alone():
    state = state_with(("What is the retry limit?", "graph_search"))
    check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "graph_search"


def test_a_plain_question_planned_to_lexical_is_not_moved_by_the_short_default_alone():
    # Moved by R13 (no exact terms), not by a fired signal.
    state = state_with(("What is the retry limit?", "lexical_search"))
    outcome = check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert "no_exact_terms" in outcome.span.attributes["detail"]


def test_a_relationship_sub_question_goes_to_the_graph():
    state = state_with(("Who does Ravi Sharma report to?", "semantic_search"))
    check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "graph_search"


def test_an_undecided_sub_question_keeps_the_planners_tool():
    long_text = " ".join(["onboarding"] * 20) + "?"
    state = state_with((long_text, "graph_search"))
    outcome = check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "graph_search"
    assert outcome.span.attributes["overrides"] == 0
    assert outcome.span.attributes["detail"] is None


def test_no_override_to_a_tool_this_run_does_not_have():
    tools = {k: v for k, v in TOOLS.items() if k != "graph_search"}
    state = state_with(("Who does Ravi Sharma report to?", "lexical_search"))
    check_tools(state, tools=tools, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool != "graph_search"


def test_fetch_document_is_never_overridden():
    state = state_with(("document 12", "fetch_document"))
    check_tools(
        state,
        tools=TOOLS | {"fetch_document": FakeTool("fetch_document")},
        relation_types=RELATIONS,
        origin=time.perf_counter(),
    )
    assert state.sub_questions[0].tool == "fetch_document"


def test_the_check_makes_no_model_call():
    state = state_with(("What does ERR_QUOTA_4419 mean?", "semantic_search"))
    check_tools(state, tools=TOOLS, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.llm_calls == 0


def test_a_run_with_none_of_the_routable_tools_is_left_alone():
    tools = {"fetch_document": FakeTool("fetch_document")}
    state = state_with(("What does ERR_QUOTA_4419 mean?", "fetch_document"))
    outcome = check_tools(state, tools=tools, relation_types=RELATIONS, origin=time.perf_counter())
    assert state.sub_questions[0].tool == "fetch_document"
    assert outcome.span.attributes["overrides"] == 0


def test_the_loop_runs_the_check_right_after_plan_and_the_override_is_used():
    import json

    from agent_doubles import RecordingLLM, chunk, ctx

    from ragfabric_core.agent.loop import run_agent

    text = "What does ERR_QUOTA_4419 mean?"
    plan = json.dumps({"sub_questions": [{"text": text, "tool": "semantic_search", "why": ""}]})
    assess = json.dumps({"verdicts": [{"sub_question": text, "answered": True, "missing": None}]})
    semantic = FakeTool("semantic_search", chunks=[chunk(1)])
    lexical = FakeTool("lexical_search", chunks=[chunk(2)])
    run = run_agent(
        "q",
        llm=RecordingLLM(plan, assess),
        tools={"semantic_search": semantic, "lexical_search": lexical},
        ctx=ctx(),
    )
    names = [span.name for span in run.trace]
    assert names[:3] == ["plan", "tool_check", "retrieve"]
    assert [t.tool for t in run.tool_calls] == ["lexical_search"]
    assert semantic.queries == []
