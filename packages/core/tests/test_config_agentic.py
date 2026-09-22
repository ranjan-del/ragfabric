"""The agent's configuration: typed, bounded, and strict about typos.

The agent is the one strategy that can spend an unbounded amount of somebody's
money if its limits are wrong, so every limit is a declared field with a floor
rather than a loose dict. A key this file does not know is an error, because a
misspelled cap does not raise, it silently leaves the default in place, and the
operator only finds out from the bill.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ragfabric_core.agent.state import NodeName
from ragfabric_core.config_file import AgenticConfig, RagFabricConfig, load_config

EXAMPLE = Path(__file__).resolve().parents[3] / "ragfabric.example.yaml"


def config(body: str, tmp_path: Path) -> RagFabricConfig:
    path = tmp_path / "ragfabric.yaml"
    path.write_text(body)
    return load_config(path)


def test_the_agent_settings_are_typed_not_a_loose_dict() -> None:
    agentic = RagFabricConfig().strategies.agentic
    assert isinstance(agentic, AgenticConfig)


def test_the_documented_defaults_are_the_real_defaults() -> None:
    agentic = AgenticConfig()
    assert agentic.max_iterations == 4
    assert agentic.max_cost_usd == 0.10
    assert agentic.max_latency_ms == 30000
    assert agentic.assess_strictness == "strict"
    assert agentic.tools == ["semantic_search", "lexical_search", "fetch_document"]


def test_the_per_node_caps_have_defaults_for_every_node_that_calls_a_model() -> None:
    caps = AgenticConfig().per_node_llm_calls
    assert (caps.plan, caps.assess, caps.repair, caps.generate) == (2, 6, 6, 2)


def test_the_per_node_caps_map_onto_the_agent_state_node_names() -> None:
    """The caps are handed to ``AgentState.spend``, so they must key by NodeName."""
    caps = AgenticConfig().node_caps()
    assert set(caps) == {NodeName.PLAN, NodeName.ASSESS, NodeName.REPAIR, NodeName.GENERATE}
    assert caps[NodeName.PLAN] == 2


def test_an_unknown_agent_key_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError) as info:
        config("strategies:\n  agentic:\n    max_iteration: 4\n", tmp_path)
    assert "max_iteration" in str(info.value)


def test_an_unknown_per_node_key_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError) as info:
        config("strategies:\n  agentic:\n    per_node_llm_calls:\n      planning: 2\n", tmp_path)
    assert "planning" in str(info.value)


def test_max_iterations_below_one_is_rejected(tmp_path) -> None:
    """Zero iterations is not a cheap agent, it is an agent that cannot retrieve."""
    with pytest.raises(ValidationError):
        config("strategies:\n  agentic:\n    max_iterations: 0\n", tmp_path)


def test_a_negative_cost_cap_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        config("strategies:\n  agentic:\n    max_cost_usd: -1\n", tmp_path)


def test_a_zero_latency_cap_is_rejected(tmp_path) -> None:
    """A cap of zero milliseconds stops the agent before its first call."""
    with pytest.raises(ValidationError):
        config("strategies:\n  agentic:\n    max_latency_ms: 0\n", tmp_path)


def test_a_per_node_cap_below_one_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        config("strategies:\n  agentic:\n    per_node_llm_calls:\n      plan: 0\n", tmp_path)


def test_a_tool_that_does_not_exist_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError) as info:
        config("strategies:\n  agentic:\n    tools: [semantic_serch]\n", tmp_path)
    assert "semantic_serch" in str(info.value)


def test_an_empty_tool_list_is_rejected(tmp_path) -> None:
    """An agent with no tools cannot retrieve, and should say so at load time."""
    with pytest.raises(ValidationError):
        config("strategies:\n  agentic:\n    tools: []\n", tmp_path)


def test_a_duplicated_tool_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        config("strategies:\n  agentic:\n    tools: [lexical_search, lexical_search]\n", tmp_path)


def test_an_unknown_assess_strictness_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        config("strategies:\n  agentic:\n    assess_strictness: relaxed\n", tmp_path)


def test_a_partial_agent_section_keeps_the_other_defaults(tmp_path) -> None:
    cfg = config("strategies:\n  agentic:\n    max_iterations: 2\n", tmp_path)
    assert cfg.strategies.agentic.max_iterations == 2
    assert cfg.strategies.agentic.max_cost_usd == 0.10
    assert cfg.strategies.vectorless.k1 == 1.2


def test_a_global_cap_below_a_per_node_cap_is_rejected(tmp_path) -> None:
    """A per-node cap above the global cap is a number that can never bind.

    Not a style preference: the loop would stop on the global cap while the
    per-node number in the file was never the reason, and the stop reason the
    caller sees would name a limit the operator did not think they had set.
    """
    with pytest.raises(ValidationError) as info:
        config(
            "strategies:\n  agentic:\n    max_llm_calls: 2\n"
            "    per_node_llm_calls:\n      plan: 2\n      assess: 6\n",
            tmp_path,
        )
    assert "max_llm_calls" in str(info.value)


def test_the_example_file_documents_every_agent_key() -> None:
    """Every key is documented with a comment, so the example must carry them all."""
    text = EXAMPLE.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    documented = loaded["strategies"]["agentic"]
    assert set(documented) == set(AgenticConfig.model_fields)
    agentic_block = text[text.index("  agentic:") :]
    agentic_block = agentic_block[: agentic_block.index("\n  graph:")]
    for key in AgenticConfig.model_fields:
        line = next(
            line for line in agentic_block.splitlines() if line.strip().startswith(f"{key}:")
        )
        assert "#" in line, f"{key} is not documented in ragfabric.example.yaml"


def test_the_example_file_still_loads() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.strategies.agentic.max_iterations == 4
    assert cfg.strategies.agentic.tools == [
        "semantic_search",
        "lexical_search",
        "fetch_document",
    ]
