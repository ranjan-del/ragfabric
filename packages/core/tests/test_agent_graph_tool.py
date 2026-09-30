"""The graph strategy as an agent tool: its evidence, its budget, its per request availability."""

from __future__ import annotations

import json

import pytest
import test_graph_traverse as traversal_tests
from agent_doubles import FakeTool, RecordingLLM, chunk, ctx
from sqlalchemy.orm import sessionmaker

from ragfabric_core.agent.loop import run_agent
from ragfabric_core.agent.nodes import retrieve
from ragfabric_core.agent.state import AgentState, BudgetExceeded, NodeName, SubQuestion
from ragfabric_core.agent.tools import (
    GraphSearchTool,
    GraphToolRun,
    tools_for_request,
)
from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.graph.contracts import EntityType, RelationType
from ragfabric_core.graph.merge import merge_subgraphs
from ragfabric_core.strategies import registry_defaults
from ragfabric_core.strategies.base import StrategyParams
from ragfabric_core.strategies.graph import GraphRAGStrategy

graph = traversal_tests.graph  # the seeded two-dialect fixture; PostgreSQL skips without a URL


class FakeGraphTool:
    name = "graph_search"
    description = "relationships"

    def __init__(self, run, provider="recording", model="recording"):
        self._run, self.seen, self.provider, self.model = run, [], provider, model

    def run(self, query, ctx):
        return self.run_graph(query, ctx).chunks

    def run_graph(self, query, ctx):
        self.seen.append(ctx)
        return self._run


def _graph_state(max_llm_calls=5, **kwargs):
    state = AgentState(question="q", max_llm_calls=max_llm_calls, **kwargs)
    state.sub_questions = [SubQuestion(text="Who owns Billing?", tool="graph_search")]
    return state


def test_a_graph_call_is_charged_to_the_agents_budget(small_subgraph):
    tool = FakeGraphTool(
        GraphToolRun(
            chunks=[chunk(1)],
            subgraph=small_subgraph,
            llm_calls=1,
            input_tokens=7,
            output_tokens=3,
        )
    )
    state = _graph_state()
    outcome = retrieve(state, tools={"graph_search": tool}, ctx=ctx())
    assert state.llm_calls == 1 and outcome.input_tokens == 7
    assert outcome.output_tokens == 3
    assert (outcome.provider, outcome.model) == ("recording", "recording")
    assert outcome.subgraphs == [small_subgraph]


def test_a_graph_call_over_budget_is_refused_and_reported_not_raised():
    tool = FakeGraphTool(
        GraphToolRun(chunks=[], subgraph=None, llm_calls=1, input_tokens=0, output_tokens=0)
    )
    state = _graph_state(max_llm_calls=0)
    outcome = retrieve(state, tools={"graph_search": tool}, ctx=ctx())
    assert tool.seen == []  # refused before the model call was made
    assert outcome.budget_stop and "max_llm_calls" in outcome.budget_stop
    assert state.llm_calls == 0


def test_ensure_moves_no_counter_and_spending_nothing_never_raises():
    state = AgentState(question="q", max_llm_calls=1)
    state.ensure(NodeName.RETRIEVE, llm_calls=1)
    assert state.llm_calls == 0
    state.spend(NodeName.RETRIEVE, llm_calls=1)
    with pytest.raises(BudgetExceeded):
        state.ensure(NodeName.RETRIEVE, llm_calls=1)
    state.spend(NodeName.RETRIEVE, llm_calls=0)
    state.ensure(NodeName.RETRIEVE, llm_calls=0)
    assert state.llm_calls == 1


def test_a_no_coverage_graph_run_is_not_charged_and_adds_no_ledger_model():
    tool = FakeGraphTool(
        GraphToolRun(chunks=[], subgraph=None, llm_calls=0, input_tokens=0, output_tokens=0)
    )
    state = _graph_state()
    outcome = retrieve(state, tools={"graph_search": tool}, ctx=ctx())
    assert state.llm_calls == 0
    assert (outcome.provider, outcome.model) == ("", "")
    assert outcome.budget_stop is None


def test_a_budget_refusal_part_way_keeps_what_was_already_harvested(small_subgraph):
    first = FakeGraphTool(
        GraphToolRun(
            chunks=[chunk(1)],
            subgraph=small_subgraph,
            llm_calls=1,
            input_tokens=7,
            output_tokens=3,
        )
    )
    second = FakeGraphTool(
        GraphToolRun(chunks=[chunk(9)], subgraph=None, llm_calls=1, input_tokens=1, output_tokens=1)
    )
    sem = FakeTool("semantic_search", chunks=[chunk(2)])
    llm = RecordingLLM(
        _plan(
            ("a", "semantic_search"),
            ("Who does Ravi Sharma report to?", "graph_search"),
            ("Who does Asha Rao report to?", "graph_search"),
        )
    )
    # plan spends 1 of 2; the first graph call spends the last; the second is refused.
    run = run_agent(
        "q",
        llm=llm,
        tools={"semantic_search": sem, "graph_search": _Sequenced(first, second)},
        ctx=ctx(max_llm_calls=2),
        relation_types=["REPORTS_TO"],
    )
    assert run.stop_reason == "budget"
    assert {c.chunk_id for c in run.chunks} == {1, 2}
    assert run.input_tokens >= 7 and run.subgraph is not None
    assert [t.tool for t in run.tool_calls] == ["semantic_search", "graph_search"]
    assert sum(1 for span in run.trace if span.name == "retrieve") == 1
    assert run.state.llm_calls == 2


class _Sequenced(FakeGraphTool):
    """Answers each call with the next scripted run."""

    def __init__(self, *tools):
        super().__init__(None)
        self._tools = list(tools)

    def run_graph(self, query, ctx):
        self.seen.append(ctx)
        return self._tools[len(self.seen) - 1]._run


def test_retrieve_has_no_per_node_cap_unless_one_is_configured():
    tool = FakeGraphTool(
        GraphToolRun(chunks=[], subgraph=None, llm_calls=1, input_tokens=0, output_tokens=0)
    )
    state = _graph_state(per_node_llm_calls={NodeName.PLAN: 1})
    retrieve(state, tools={"graph_search": tool}, ctx=ctx())
    assert state.node_llm_calls[NodeName.RETRIEVE] == 1


def test_an_edgeless_subgraph_is_not_carried(small_subgraph):
    from ragfabric_core.graph.contracts import EmptyReason, Subgraph

    empty = Subgraph(
        nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_ENTITY_MATCHED
    )
    tool = FakeGraphTool(
        GraphToolRun(chunks=[], subgraph=empty, llm_calls=1, input_tokens=0, output_tokens=0)
    )
    outcome = retrieve(_graph_state(), tools={"graph_search": tool}, ctx=ctx())
    assert outcome.subgraphs == []


def test_the_graph_tool_is_dropped_under_a_filter_the_walk_cannot_apply():
    tools = {"semantic_search": FakeTool("semantic_search"), "graph_search": FakeGraphTool(None)}
    for key, value in (("document_id", 4), ("format", "pdf")):
        filtered = ctx().model_copy(
            update={"params": StrategyParams(metadata_filters={key: value})}
        )
        assert set(tools_for_request(tools, filtered)) == {"semantic_search"}
    assert set(tools_for_request(tools, ctx())) == {"semantic_search", "graph_search"}


def test_merging_subgraphs_unions_by_id_and_keeps_truncation(small_subgraph):
    merged = merge_subgraphs([small_subgraph, small_subgraph])
    assert [n.id for n in merged.nodes] == [n.id for n in small_subgraph.nodes]
    assert [e.id for e in merged.edges] == [e.id for e in small_subgraph.edges]
    assert not merged.truncated
    assert merge_subgraphs([]) is None
    truncated = small_subgraph.model_copy(update={"truncated": True})
    assert merge_subgraphs([small_subgraph, truncated]).truncated is True


def _plan(*pairs):
    return json.dumps(
        {"sub_questions": [{"text": t, "tool": tool, "why": ""} for t, tool in pairs]}
    )


def test_the_loop_reports_the_walked_subgraph(small_subgraph):
    tool = FakeGraphTool(
        GraphToolRun(
            chunks=[chunk(1)],
            subgraph=small_subgraph,
            llm_calls=1,
            input_tokens=2,
            output_tokens=1,
        )
    )
    llm = RecordingLLM(
        _plan(("Who is on Platform Team?", "graph_search")),
        json.dumps(
            {
                "verdicts": [
                    {"sub_question": "Who is on Platform Team?", "answered": True, "missing": None}
                ]
            }
        ),
    )
    run = run_agent("q", llm=llm, tools={"graph_search": tool}, ctx=ctx())
    assert run.subgraph == merge_subgraphs([small_subgraph])
    assert run.state.llm_calls == 3  # plan, the graph's entity call, assess
    assert run.input_tokens >= 2


def test_a_graph_call_that_busts_the_budget_stops_the_run_not_the_request():
    tool = FakeGraphTool(
        GraphToolRun(chunks=[], subgraph=None, llm_calls=1, input_tokens=0, output_tokens=0)
    )
    llm = RecordingLLM(_plan(("Who owns Billing?", "graph_search")))
    run = run_agent("q", llm=llm, tools={"graph_search": tool}, ctx=ctx(max_llm_calls=1))
    assert run.stop_reason == "budget"
    assert run.subgraph is None


def test_the_agent_strategy_hides_graph_search_from_a_filtered_request(small_subgraph):
    from ragfabric_core.strategies.agentic import AgenticRAGStrategy

    tool = FakeGraphTool(
        GraphToolRun(
            chunks=[chunk(1)], subgraph=small_subgraph, llm_calls=1, input_tokens=0, output_tokens=0
        )
    )
    sem = FakeTool("semantic_search", chunks=[chunk(2)])
    plan_text = _plan(("Who does Ravi Sharma report to?", "graph_search"))
    assess = json.dumps(
        {
            "verdicts": [
                {
                    "sub_question": "Who does Ravi Sharma report to?",
                    "answered": True,
                    "missing": None,
                }
            ]
        }
    )
    strategy = AgenticRAGStrategy(
        llm=RecordingLLM(plan_text, assess),
        tools={"semantic_search": sem, "graph_search": tool},
        relation_types=["REPORTS_TO"],
    )
    result = strategy.retrieve("q", ctx())
    assert result.subgraph is not None and tool.seen

    tool.seen.clear()
    filtered = ctx().model_copy(
        update={"params": StrategyParams(metadata_filters={"document_id": 4})}
    )
    llm = RecordingLLM(plan_text, assess)
    strategy = AgenticRAGStrategy(
        llm=llm,
        tools={"semantic_search": sem, "graph_search": tool},
        relation_types=["REPORTS_TO"],
    )
    result = strategy.retrieve("q", filtered)
    assert tool.seen == [] and result.subgraph is None
    assert "graph_search:" not in llm.prompts[0][-1].content


def test_graph_search_tool_reports_the_strategys_counters(small_subgraph):
    class Strategy:
        def retrieve(self, query, ctx):
            from ragfabric_core.strategies.base import RetrievalResult

            return RetrievalResult(
                strategy="graph",
                chunks=[chunk(1)],
                retrieval_calls=1,
                embedding_calls=0,
                llm_calls=1,
                input_tokens=11,
                output_tokens=5,
                latency_ms=0,
                subgraph=small_subgraph,
            )

    llm = RecordingLLM(provider="p", model="m")
    tool = GraphSearchTool(Strategy(), llm)
    assert (tool.name, tool.provider, tool.model) == ("graph_search", "p", "m")
    run = tool.run_graph("q", ctx())
    assert (run.llm_calls, run.input_tokens, run.output_tokens) == (1, 11, 5)
    assert run.subgraph is small_subgraph
    assert [c.chunk_id for c in tool.run("q", ctx())] == [1]


# --- configuration -----------------------------------------------------------------------


@pytest.fixture()
def sqlite_factory(tmp_path):
    from sqlalchemy import create_engine

    from ragfabric_core.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'wiring.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _cfg(*, enabled: bool, tools: list[str]) -> RagFabricConfig:
    return RagFabricConfig(
        graph_store={"enabled": enabled},
        strategies={"agentic": {"tools": tools}},
    )


def test_the_default_tools_do_not_include_graph_search(sqlite_factory):
    agentic = registry_defaults.default_registry(
        RagFabricConfig(graph_store={"enabled": True}), sqlite_factory
    ).get("agentic")
    assert "graph_search" not in agentic.tools


def test_graph_search_is_wired_when_graph_is_enabled_and_listed(sqlite_factory):
    cfg = _cfg(enabled=True, tools=["semantic_search", "graph_search"])
    registry = registry_defaults.default_registry(cfg, sqlite_factory)
    agentic = registry.get("agentic")
    assert sorted(agentic.tools) == ["graph_search", "semantic_search"]
    assert isinstance(agentic.tools["graph_search"], GraphSearchTool)


def test_listing_graph_search_with_graph_disabled_is_the_unknown_tool_error(sqlite_factory):
    cfg = _cfg(enabled=False, tools=["semantic_search", "graph_search"])
    with pytest.raises(KeyError, match="unknown tool.*graph_search"):
        registry_defaults.default_registry(cfg, sqlite_factory)


def test_the_registry_builds_the_graph_strategy_once(sqlite_factory, monkeypatch):
    built = []
    real = registry_defaults._build_graph

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(registry_defaults, "_build_graph", counting)
    cfg = _cfg(enabled=True, tools=["semantic_search", "graph_search"])
    registry = registry_defaults.default_registry(cfg, sqlite_factory)
    assert len(built) == 1
    assert registry.get("agentic").tools["graph_search"]._strategy is registry.get("graph")


# --- access (ADR 0003), on a real walk --------------------------------------------------


def test_a_restricted_caller_gets_no_denied_evidence_through_graph_search(graph):
    """Runs on SQLite always and on PostgreSQL when RAGFABRIC_TEST_DATABASE_URL is set."""
    graph.entity("Ravi Sharma", EntityType.PERSON, chunks=(graph.open_chunk,))
    graph.entity("Platform Team", EntityType.TEAM, chunks=(graph.open_chunk,))
    graph.entity("Meera Iyer", EntityType.PERSON, chunks=(graph.secret_chunk,))
    graph.edge("Ravi Sharma", RelationType.MEMBER_OF, "Platform Team", chunks=(graph.open_chunk,))
    graph.edge("Ravi Sharma", RelationType.REPORTS_TO, "Meera Iyer", chunks=(graph.secret_chunk,))
    graph.db.commit()

    question = "Who does Ravi Sharma report to, and what team is he on?"
    extraction = json.dumps(
        {
            "entities": [{"name": "Ravi Sharma", "entity_type": "person"}],
            "implied_relation_types": [],
        }
    )
    verdict = json.dumps(
        {"verdicts": [{"sub_question": question, "answered": True, "missing": None}]}
    )

    def run_as(access):
        factory = sessionmaker(bind=graph.db.get_bind(), expire_on_commit=False)
        llm = RecordingLLM(_plan((question, "graph_search")), verdict, provider="p", model="m")
        strategy = GraphRAGStrategy(llm=RecordingLLM(extraction), session_factory=factory)
        tool = GraphSearchTool(strategy, llm)
        return run_agent(question, llm=llm, tools={"graph_search": tool}, ctx=ctx(access=access))

    control = run_as(AccessFilter.unrestricted())
    assert graph.secret_chunk in {c.chunk_id for c in control.chunks}
    assert {n.name for n in control.subgraph.nodes} >= {"Meera Iyer"}

    restricted = run_as(graph.restricted)
    assert graph.secret_chunk not in {c.chunk_id for c in restricted.chunks}
    assert restricted.chunks and restricted.subgraph is not None
    names = {n.name for n in restricted.subgraph.nodes}
    assert "Meera Iyer" not in names and {"Ravi Sharma", "Platform Team"} <= names
    assert all(graph.secret_chunk not in e.source_chunk_ids for e in restricted.subgraph.edges)
    assert len(restricted.subgraph.edges) == 1
