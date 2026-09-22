"""Where the two halves of Phase 5 meet.

The agent pipeline and the concrete tools were built separately, against an
agreed protocol. These tests hold the seam between them, and every one of them
covers something neither half could have checked on its own.

The seam is genuinely sharp in two places. The registry reads a typed
``AgenticConfig`` that replaced a loose dict, so any surviving ``.get`` call is
an ``AttributeError`` at startup. And the strategy counts embeddings by summing
a cumulative counter on each tool, so a tool that embeds without carrying one
makes the strategy report zero for a run that really did embed, which is exactly
the fabricated number ADR 0004 forbids.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.agent.state import NodeName
from ragfabric_core.agent.tools import FetchDocumentTool, LexicalSearchTool, SemanticSearchTool
from ragfabric_core.config_file import RagFabricConfig, load_config
from ragfabric_core.models import Base
from ragfabric_core.strategies import registry_defaults
from ragfabric_core.strategies.agentic import AgenticRAGStrategy
from ragfabric_core.strategies.base import Budget

EXAMPLE = Path(__file__).resolve().parents[3] / "ragfabric.example.yaml"


@pytest.fixture()
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'wiring.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _agentic(cfg: RagFabricConfig, session_factory) -> AgenticRAGStrategy:
    return registry_defaults.default_registry(cfg, session_factory).get("agentic")


def test_the_registry_reads_the_typed_agentic_config(session_factory):
    """The config stopped being a dict. A surviving .get would raise here."""
    cfg = RagFabricConfig()
    cfg.strategies.agentic.max_iterations = 7
    strategy = _agentic(cfg, session_factory)
    assert strategy.max_iterations == 7


def test_all_three_tools_are_registered_by_default(session_factory):
    strategy = _agentic(RagFabricConfig(), session_factory)
    assert sorted(strategy.tools) == ["fetch_document", "lexical_search", "semantic_search"]


def test_the_registry_uses_the_real_tools_not_a_local_adapter(session_factory):
    """The tools carry descriptions written for the planning model. An adapter
    built inside the registry would show the planner different text than the
    one the tool tests cover."""
    strategy = _agentic(RagFabricConfig(), session_factory)
    assert isinstance(strategy.tools["semantic_search"], SemanticSearchTool)
    assert isinstance(strategy.tools["lexical_search"], LexicalSearchTool)
    assert isinstance(strategy.tools["fetch_document"], FetchDocumentTool)


def test_the_configured_tool_list_is_honoured(session_factory):
    """An operator who disables a tool must actually see it gone, or the
    planner keeps choosing something the deployment meant to switch off."""
    cfg = RagFabricConfig()
    cfg.strategies.agentic.tools = ["lexical_search"]
    strategy = _agentic(cfg, session_factory)
    assert sorted(strategy.tools) == ["lexical_search"]


def test_the_semantic_tool_carries_an_embedding_counter(session_factory):
    """ADR 0004: the strategy sums this counter to report embedding_calls.

    Without it the strategy reports zero for a run that embedded, which is a
    plausible number rather than a measured one.
    """
    strategy = _agentic(RagFabricConfig(), session_factory)
    assert hasattr(strategy.tools["semantic_search"], "embedding_calls")
    assert strategy.tools["semantic_search"].embedding_calls == 0


def test_the_lexical_and_fetch_tools_carry_no_embedding_counter(session_factory):
    """A counter that is always zero would state a fact nobody measured. The
    vectorless path embeds nothing and should report nothing."""
    strategy = _agentic(RagFabricConfig(), session_factory)
    assert not hasattr(strategy.tools["lexical_search"], "embedding_calls")
    assert not hasattr(strategy.tools["fetch_document"], "embedding_calls")


def test_the_per_node_caps_reach_the_strategy(session_factory):
    cfg = RagFabricConfig()
    cfg.strategies.agentic.per_node_llm_calls.assess = 3
    strategy = _agentic(cfg, session_factory)
    assert strategy.per_node_llm_calls[NodeName.ASSESS] == 3


def test_every_configured_limit_reaches_the_strategy(session_factory):
    """Four limits were typed, validated and printed back while nothing read them.

    The registry passed only the tools, the iteration cap and the per-node
    caps, so an operator could edit three spend limits and a rubric setting,
    see them echoed by ``config validate``, and get no change in behaviour.
    """
    cfg = RagFabricConfig()
    cfg.strategies.agentic.max_llm_calls = 9
    cfg.strategies.agentic.max_cost_usd = 0.25
    cfg.strategies.agentic.max_latency_ms = 12_000
    cfg.strategies.agentic.assess_strictness = "lenient"
    strategy = _agentic(cfg, session_factory)

    assert strategy.max_llm_calls == 9
    assert strategy.max_cost_usd == 0.25
    assert strategy.max_latency_ms == 12_000
    assert strategy.assess_strictness == "lenient"


def test_the_example_files_advertised_limits_are_the_ones_enforced(session_factory):
    """The numbers in ``ragfabric.example.yaml`` are what a default run gets.

    ``max_llm_calls`` advertised twelve while the enforced cap was the
    ``Budget`` default of eight, so the documented number was unreachable.
    """
    cfg = load_config(EXAMPLE)
    strategy = _agentic(cfg, session_factory)

    assert strategy.max_llm_calls == 12
    assert strategy.max_cost_usd == 0.10
    assert strategy.max_latency_ms == 30_000
    assert strategy.assess_strictness == "strict"
    assert min(strategy.max_llm_calls, Budget().max_llm_calls) == 12
    assert min(strategy.max_latency_ms, Budget().max_latency_ms) == 30_000
    assert min(strategy.max_cost_usd, Budget().max_cost_usd) == 0.10
