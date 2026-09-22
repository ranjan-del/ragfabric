"""AgenticRAGStrategy: the loop behind the one result shape every strategy returns.

ADR 0002 says nothing downstream may ask which strategy produced a result, so
the agent has to report the same counters as the others, and ADR 0004 says
those counters report what happened. The two meet here, and the temptation this
file exists to block is estimating: an agent makes a variable number of calls,
so it is the one strategy where a plausible number would never be noticed.

The access filter matters more here than in any other strategy. Traditional and
vectorless make one store query under one filter. The agent makes several,
across iterations, and pools the results into evidence that is later summarised
into an answer, so a single call made without the caller's filter contaminates
everything after it.
"""

from __future__ import annotations

import json

from agent_doubles import FakeTool, RecordingLLM, SequenceTool, SlowTool, chunk, restricted_ctx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.models.base import Base
from ragfabric_core.strategies import registry_defaults
from ragfabric_core.strategies.agentic import AgenticRAGStrategy
from ragfabric_core.strategies.base import (
    RetrievalResult,
    RetrieverStrategy,
    StrategyName,
    StrategyRegistry,
)


def plan_json(*pairs: tuple[str, str]) -> str:
    return json.dumps(
        {"sub_questions": [{"text": text, "tool": tool, "why": ""} for text, tool in pairs]}
    )


def assessed(text: str, answered: bool, missing: str | None = None) -> str:
    return json.dumps(
        {"verdicts": [{"sub_question": text, "answered": answered, "missing": missing}]}
    )


def test_the_strategy_conforms_to_the_retriever_protocol():
    strategy = AgenticRAGStrategy(llm=RecordingLLM(), tools={"semantic_search": FakeTool("s")})
    assert isinstance(strategy, RetrieverStrategy)
    assert strategy.name is StrategyName.AGENTIC


def test_counters_report_actual_calls():
    """Two LLM calls and one tool call really happened. Nothing is rounded up."""
    tool = FakeTool("semantic_search", chunks=[chunk(1), chunk(2)], embedding_calls_per_run=1)
    llm = RecordingLLM(plan_json(("q", "semantic_search")), assessed("q", True))
    strategy = AgenticRAGStrategy(llm=llm, tools={"semantic_search": tool})

    result = strategy.retrieve("q", restricted_ctx(document_ids={1}))

    assert isinstance(result, RetrievalResult)
    assert result.strategy is StrategyName.AGENTIC
    assert result.llm_calls == 2
    assert result.llm_calls == llm.calls
    assert result.retrieval_calls == 1
    assert result.retrieval_calls == tool.retrieval_calls
    assert result.embedding_calls == 1
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert [c.chunk_id for c in result.chunks] == [1, 2]


def test_counters_grow_with_the_iterations_that_actually_ran():
    tool = SequenceTool(
        "semantic_search",
        batches=[[chunk(1)], [chunk(2)]],
        embedding_calls_per_run=1,
    )
    llm = RecordingLLM(
        plan_json(("q", "semantic_search")),
        assessed("q", False, "the number"),
        json.dumps({"move": "broaden", "rewritten_query": None, "why": "w"}),
        assessed("q", True),
    )
    strategy = AgenticRAGStrategy(llm=llm, tools={"semantic_search": tool})

    result = strategy.retrieve("q", restricted_ctx(document_ids={1}))

    assert result.llm_calls == 4
    assert result.retrieval_calls == 2
    assert result.embedding_calls == 2


def test_embedding_calls_is_zero_when_only_lexical_tools_ran():
    """Zero because nothing embedded, reported rather than omitted (ADR 0004)."""
    lexical = FakeTool("lexical_search", chunks=[chunk(1)], embedding_calls_per_run=0)
    semantic = FakeTool("semantic_search", chunks=[chunk(9)], embedding_calls_per_run=1)
    llm = RecordingLLM(plan_json(("ERR_4019", "lexical_search")), assessed("ERR_4019", True))
    strategy = AgenticRAGStrategy(
        llm=llm, tools={"lexical_search": lexical, "semantic_search": semantic}
    )

    result = strategy.retrieve("what does ERR_4019 mean", restricted_ctx(document_ids={1}))

    assert result.embedding_calls == 0
    assert result.retrieval_calls == 1
    assert semantic.retrieval_calls == 0


def test_the_access_filter_reaches_every_tool_call():
    semantic = SequenceTool("semantic_search", batches=[[chunk(1)], [chunk(2)]])
    lexical = SequenceTool("lexical_search", batches=[[chunk(3)], [chunk(4)]])
    llm = RecordingLLM(
        plan_json(("the concept", "semantic_search"), ("ERR_4019", "lexical_search")),
        json.dumps(
            {
                "verdicts": [
                    {"sub_question": "the concept", "answered": False, "missing": "the number"},
                    {"sub_question": "ERR_4019", "answered": False, "missing": "the code"},
                ]
            }
        ),
        json.dumps({"move": "broaden", "rewritten_query": None, "why": "w"}),
        json.dumps({"move": "broaden", "rewritten_query": None, "why": "w"}),
        json.dumps(
            {
                "verdicts": [
                    {"sub_question": "the concept", "answered": True},
                    {"sub_question": "ERR_4019", "answered": True},
                ]
            }
        ),
    )
    strategy = AgenticRAGStrategy(
        llm=llm, tools={"semantic_search": semantic, "lexical_search": lexical}
    )
    ctx = restricted_ctx(document_ids={7, 8})

    strategy.retrieve("two part question", ctx)

    seen = semantic.contexts + lexical.contexts
    assert len(seen) == 4
    assert all(call.access_filter == ctx.access_filter for call in seen)
    assert all(call.access_filter.document_ids == frozenset({7, 8}) for call in seen)
    assert all(call.principal == ctx.principal for call in seen)


def test_the_trace_and_stop_reason_are_carried_onto_the_result():
    tool = FakeTool("semantic_search", chunks=[chunk(1)])
    llm = RecordingLLM(plan_json(("q", "semantic_search")), assessed("q", True))
    strategy = AgenticRAGStrategy(llm=llm, tools={"semantic_search": tool})

    result = strategy.retrieve("q", restricted_ctx(document_ids={1}))

    names = [span.name for span in result.trace]
    assert names[0] == "plan"
    assert names[-1] == "finalize"
    assert result.trace[-1].attributes["stop_reason"] == "resolved"


def test_latency_is_measured_not_assumed():
    tool = SlowTool("semantic_search", delay_s=0.02, chunks=[chunk(1)])
    llm = RecordingLLM(plan_json(("q", "semantic_search")), assessed("q", True))
    strategy = AgenticRAGStrategy(llm=llm, tools={"semantic_search": tool})

    result = strategy.retrieve("q", restricted_ctx(document_ids={1}))

    assert result.latency_ms >= 15


def test_the_strategy_is_resolvable_from_the_registry():
    registry = StrategyRegistry()
    registry.register(
        AgenticRAGStrategy(llm=RecordingLLM(), tools={"semantic_search": FakeTool("s")})
    )

    assert StrategyName.AGENTIC in registry.names()
    assert isinstance(registry.get("agentic"), AgenticRAGStrategy)


def test_agentic_is_registered_by_the_default_registry(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'registry.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    registry = registry_defaults.default_registry(RagFabricConfig(), session_factory)

    assert StrategyName.AGENTIC in registry.names()
    strategy = registry.get("agentic")
    assert isinstance(strategy, AgenticRAGStrategy)
    # fetch_document joined the default registry when the concrete tools landed.
    assert sorted(strategy.tools) == ["fetch_document", "lexical_search", "semantic_search"]
