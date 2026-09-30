"""What the router decided, and who decided it.

Imports nothing from the rest of core, because ``strategies/base.py`` imports
this module to type ``RetrievalResult.router``. Strategy names are therefore
spelled as literals here rather than taken from ``StrategyName``.

``confidence`` is ``None`` for a signals decision. A rule that fired is not a
measurement, and a number attached to it would be one nobody measured
(ADR 0004). A classifier confidence is what the model reported and is
uncalibrated until Phase 8 measures it against labelled questions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Servable = Literal["traditional", "vectorless", "agentic", "graph"]
QueryType = Literal[
    "simple_factual",
    "exact_match",
    "relationship",
    "multi_hop",
    "comparison",
    "aggregation",
    "compound",
    "ambiguous",
]
Level = Literal["low", "medium", "high"]
Source = Literal["signals", "classifier", "signals_fallback"]

# Engineering assessments of each strategy's shape, not measurements: one
# embedding call and one query is low, a model call per step is high.
# (complexity, cost, latency)
ENGINEERING_LEVELS: dict[str, tuple[Level, Level, Level]] = {
    "traditional": ("low", "low", "low"),
    "vectorless": ("low", "low", "low"),
    "graph": ("medium", "medium", "medium"),
    "agentic": ("high", "high", "high"),
}

MAX_REASONING_CHARS = 200


class RouterDecision(BaseModel):
    selected_strategy: Servable
    source: Source
    decisive: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1, max_length=MAX_REASONING_CHARS)
    query_type: QueryType
    estimated_complexity: Level
    expected_cost_level: Level
    expected_latency_level: Level
    fused: bool = False

    @model_validator(mode="after")
    def _signals_carry_no_confidence(self) -> RouterDecision:
        # ADR 0004 by construction: a rule that fired is not a measurement, so a
        # decision made by the signals cannot carry a number, even by mistake.
        if self.source in ("signals", "signals_fallback") and self.confidence is not None:
            raise ValueError(f"a {self.source} decision has no confidence; got {self.confidence}")
        return self


def levels_for(strategy: Servable) -> dict[str, Level]:
    complexity, cost, latency = ENGINEERING_LEVELS[strategy]
    return {
        "estimated_complexity": complexity,
        "expected_cost_level": cost,
        "expected_latency_level": latency,
    }
