import pytest
from typer.testing import CliRunner

from ragfabric_cli.commands.strategies import ROWS
from ragfabric_cli.main import app
from ragfabric_cli.ui import console
from ragfabric_core.config_file import GraphStoreConfig
from ragfabric_core.router.signals import extract_signals, propose
from ragfabric_core.strategies.base import StrategyName

runner = CliRunner()
NAMES = ["auto", "traditional", "vectorless", "agentic", "graph"]


def test_rows_are_the_five_strategies_in_order():
    assert [row.name for row in ROWS] == NAMES


@pytest.mark.parametrize("row", [r for r in ROWS if r.name != "auto"], ids=lambda r: r.name)
def test_each_example_routes_to_its_own_strategy(row):
    signals = extract_signals(row.example, relation_types=GraphStoreConfig().relation_types)
    proposal = propose(signals, available=set(StrategyName))
    assert proposal.strategy.value == row.name
    assert proposal.decisive


def test_plain_output_lists_every_name_and_example():
    result = runner.invoke(app, ["strategies"])
    assert result.exit_code == 0
    for row in ROWS:
        assert f"{row.name}:" in result.output
        assert row.example in result.output
    assert "\x1b[" not in result.output


def test_rich_output_lists_every_name_and_example(monkeypatch):
    monkeypatch.setattr(console, "is_rich", lambda stream=None: True)
    monkeypatch.setenv("COLUMNS", "200")
    console.get_console.cache_clear()
    try:
        result = runner.invoke(app, ["strategies"])
    finally:
        console.get_console.cache_clear()
    assert result.exit_code == 0
    for row in ROWS:
        assert row.name in result.output
        assert row.example in result.output


@pytest.mark.parametrize("row", [r for r in ROWS if r.name != "auto"], ids=lambda r: r.name)
def test_relative_cost_is_the_routers_engineering_level(row):
    from ragfabric_core.router.decision import ENGINEERING_LEVELS

    assert row.relative_cost == ENGINEERING_LEVELS[row.name][1]


def test_auto_cost_depends_on_the_choice_and_plain_output_has_no_model_call_counts():
    auto = next(r for r in ROWS if r.name == "auto")
    assert auto.relative_cost == "depends on the strategy chosen"
    assert "model calls" not in runner.invoke(app, ["strategies"]).output
