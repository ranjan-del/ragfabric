"""``ragfabric strategies``: the five ways to answer, and a question each one suits."""

from __future__ import annotations

from ragfabric_cli.ui.panels import StrategyRow, render_strategies
from ragfabric_core.router.decision import ENGINEERING_LEVELS


def _cost(strategy: str) -> str:
    """The engineering assessment of cost (index 1), the router's own table, not a measurement."""
    return ENGINEERING_LEVELS[strategy][1]


# The examples are the router's own signal cases (packages/core/tests/test_router_signals.py);
# a test asserts each one still routes to its row. "auto" has no example of its own.
ROWS: list[StrategyRow] = [
    StrategyRow(
        "auto",
        "letting the router pick one of the others for each question",
        "depends on the strategy chosen",
        "What is our refund policy?",
    ),
    StrategyRow(
        "traditional",
        "questions about meaning, found by similarity",
        _cost("traditional"),
        "What is our refund policy?",
    ),
    StrategyRow(
        "vectorless",
        "exact identifiers and quoted phrases, found by words",
        _cost("vectorless"),
        "What does ERR_QUOTA_4419 mean?",
    ),
    StrategyRow(
        "agentic",
        "comparisons, counts and multi part questions",
        _cost("agentic"),
        "Compare the leave policy for Pune and Delhi",
    ),
    StrategyRow(
        "graph",
        "how named people and things are related",
        _cost("graph"),
        "Who does Ravi Sharma report to?",
    ),
]


def strategies() -> None:
    """List the retrieval strategies, what each is best at and an example question.

    \b
    Examples:
      ragfabric strategies
      ragfabric ask "What is our refund policy?" --strategy traditional
    """
    render_strategies(ROWS)
