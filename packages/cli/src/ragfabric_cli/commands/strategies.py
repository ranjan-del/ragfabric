"""``ragfabric strategies``: the five ways to answer, and a question each one suits."""

from __future__ import annotations

from ragfabric_cli.ui.panels import StrategyRow, render_strategies

# The examples are the router's own signal cases (packages/core/tests/test_router_signals.py);
# a test asserts each one still routes to its row. "auto" has no example of its own.
ROWS: list[StrategyRow] = [
    StrategyRow(
        "auto",
        "letting the router pick one of the others for each question",
        "0 to 1",
        "What is our refund policy?",
    ),
    StrategyRow(
        "traditional",
        "questions about meaning, found by similarity",
        "1",
        "What is our refund policy?",
    ),
    StrategyRow(
        "vectorless",
        "exact identifiers and quoted phrases, found by words",
        "1",
        "What does ERR_QUOTA_4419 mean?",
    ),
    StrategyRow(
        "agentic",
        "comparisons, counts and multi part questions",
        "several",
        "Compare the leave policy for Pune and Delhi",
    ),
    StrategyRow(
        "graph",
        "how named people and things are related",
        "1",
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
