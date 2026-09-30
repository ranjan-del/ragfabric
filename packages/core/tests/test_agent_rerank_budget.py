"""The agent's semantic_search with an LLM reranker: its rerank call is the agent's to pay (R29).

The shared Traditional strategy the tool wraps makes one model call per search
when ``reranker.kind`` is ``llm``. That call has to be checked against the
agent's budget before it is made and counted after, exactly as a graph_search
call is, or the agent reports fewer calls than it made and can exceed the
caller's ``max_llm_calls``.
"""

from __future__ import annotations

import json

import test_agent_tools as tool_tests
from agent_doubles import FakeTool, RecordingLLM, chunk, ctx

from ragfabric_core.agent.nodes import retrieve
from ragfabric_core.agent.state import AgentState, SubQuestion
from ragfabric_core.agent.tools import SemanticSearchTool
from ragfabric_core.rerank.llm_reranker import LlmReranker
from ragfabric_core.strategies.agentic import AgenticRAGStrategy
from ragfabric_core.strategies.auto import AutoStrategy
from ragfabric_core.strategies.base import StrategyName as S
from ragfabric_core.strategies.base import StrategyRegistry
from ragfabric_core.strategies.contract import assert_strategy_contract
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy

SCORES = json.dumps({"scores": [0.9]})
AGGREGATION = "How many offices does the company have?"


def _reranked(*scores: str):
    """A semantic tool over a Traditional strategy that reranks with a model."""
    store = tool_tests.StubVectorStore([tool_tests.chunk(1)])
    rerank_llm = RecordingLLM(*scores)
    strategy = TraditionalRAGStrategy(
        embedding_provider=tool_tests.StubEmbedder(),
        vector_store=store,
        reranker=LlmReranker(rerank_llm),
    )
    return SemanticSearchTool(strategy), store, rerank_llm, strategy


def _plan(*pairs):
    return json.dumps(
        {"sub_questions": [{"text": t, "tool": tool, "why": ""} for t, tool in pairs]}
    )


def _answered(text):
    return json.dumps({"verdicts": [{"sub_question": text, "answered": True, "missing": None}]})


def test_the_semantic_tool_counts_the_rerank_calls_it_made():
    tool, _, rerank_llm, _ = _reranked(SCORES)
    assert tool.spends_llm_calls is True
    tool.run("retry limit", ctx())
    assert tool.llm_calls == 1 == rerank_llm.calls


def test_a_semantic_tool_with_no_model_reranker_spends_nothing():
    tool, _ = tool_tests.semantic_tool([tool_tests.chunk(1)])
    assert tool.spends_llm_calls is False
    tool.run("retry limit", ctx())
    assert tool.llm_calls == 0


def test_an_llm_rerank_search_over_budget_is_refused_and_keeps_what_was_harvested():
    tool, store, rerank_llm, _ = _reranked(SCORES)
    lexical = FakeTool("lexical_search", chunks=[chunk(7)])
    state = AgentState(question="q", max_llm_calls=0)
    state.sub_questions = [
        SubQuestion(text="ERR_7", tool="lexical_search"),
        SubQuestion(text="retry limit", tool="semantic_search"),
    ]
    outcome = retrieve(state, tools={"lexical_search": lexical, "semantic_search": tool}, ctx=ctx())
    assert store.last_access is None  # refused before the search and its rerank ran
    assert rerank_llm.calls == 0 and state.llm_calls == 0
    assert outcome.budget_stop and "max_llm_calls" in outcome.budget_stop
    assert list(state.evidence) == [7]


def test_an_llm_rerank_search_is_charged_to_the_agents_budget():
    tool, _, _, _ = _reranked(SCORES)
    state = AgentState(question="q", max_llm_calls=5)
    state.sub_questions = [SubQuestion(text="retry limit", tool="semantic_search")]
    retrieve(state, tools={"semantic_search": tool}, ctx=ctx())
    assert state.llm_calls == 1


def test_the_agentic_strategy_reports_the_rerank_call_exactly_once():
    tool, _, rerank_llm, _ = _reranked(SCORES)
    agent_llm = RecordingLLM(_plan(("retry limit", "semantic_search")), _answered("retry limit"))
    strategy = AgenticRAGStrategy(llm=agent_llm, tools={"semantic_search": tool})
    result = assert_strategy_contract(strategy, "What is the retry limit?", ctx())
    assert (agent_llm.calls, rerank_llm.calls) == (2, 1)
    assert result.llm_calls == 3  # plan, the rerank, assess


def test_the_auto_contract_holds_over_an_agent_that_reranks_with_a_model():
    tool, _, rerank_llm, traditional = _reranked(SCORES, SCORES)
    agent_llm = RecordingLLM(_plan((AGGREGATION, "semantic_search")), _answered(AGGREGATION))
    registry = StrategyRegistry()
    registry.register(traditional)
    registry.register(AgenticRAGStrategy(llm=agent_llm, tools={"semantic_search": tool}))
    auto = AutoStrategy(
        registry=registry,
        llm=None,
        min_confidence=0.6,
        classifier_model=None,
        graph_enabled=False,
        relation_types=[],
    )
    registry.register(auto)
    budget = ctx(max_llm_calls=3)
    result = assert_strategy_contract(auto, AGGREGATION, budget)
    assert result.strategy is S.AGENTIC and result.router.source == "signals"
    assert result.llm_calls == 3 == agent_llm.calls + rerank_llm.calls
    assert result.llm_calls <= budget.budget.max_llm_calls
