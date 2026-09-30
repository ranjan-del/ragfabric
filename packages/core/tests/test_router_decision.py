import pytest
from pydantic import ValidationError

from ragfabric_core.router.decision import ENGINEERING_LEVELS, RouterDecision
from ragfabric_core.strategies.base import RetrievalResult, StrategyName
from ragfabric_core.strategies.contract import assert_strategy_contract


def decision(**overrides) -> RouterDecision:
    fields = {
        "selected_strategy": "graph",
        "source": "signals",
        "decisive": True,
        "confidence": None,
        "reasoning": "The question asks who reports to whom.",
        "query_type": "relationship",
        "estimated_complexity": "medium",
        "expected_cost_level": "medium",
        "expected_latency_level": "medium",
    }
    return RouterDecision(**(fields | overrides))


def test_a_signals_decision_carries_no_confidence_number():
    assert decision().confidence is None


def test_auto_is_never_a_selected_strategy():
    with pytest.raises(ValidationError):
        decision(selected_strategy="auto")


def test_reasoning_longer_than_one_short_sentence_is_refused():
    with pytest.raises(ValidationError):
        decision(reasoning="x" * 201)


def test_every_servable_strategy_has_engineering_levels_and_auto_does_not():
    assert set(ENGINEERING_LEVELS) == {"traditional", "vectorless", "agentic", "graph"}


def test_auto_is_in_the_vocabulary():
    assert StrategyName("auto") is StrategyName.AUTO


def test_router_defaults_to_none_so_the_four_strategies_are_unaffected():
    result = RetrievalResult(
        strategy=StrategyName.TRADITIONAL,
        chunks=[],
        retrieval_calls=0,
        llm_calls=0,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
    )
    assert result.router is None


class _Auto:
    name = StrategyName.AUTO

    def __init__(self, router):
        self._router = router

    def retrieve(self, query, ctx):
        return RetrievalResult(
            strategy=StrategyName.TRADITIONAL,
            chunks=[],
            retrieval_calls=0,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            router=self._router,
        )


def test_the_contract_accepts_auto_reporting_the_strategy_that_ran(make_ctx):
    assert_strategy_contract(_Auto(decision(selected_strategy="traditional")), "q", make_ctx())


def test_the_contract_refuses_auto_without_a_decision(make_ctx):
    with pytest.raises(AssertionError):
        assert_strategy_contract(_Auto(None), "q", make_ctx())


@pytest.mark.parametrize("source", ["signals", "signals_fallback"])
def test_a_signals_decision_refuses_a_confidence_number(source):
    # ADR 0004 by construction: a rule that fired measured nothing.
    with pytest.raises(ValidationError, match="confidence"):
        decision(source=source, confidence=0.5)


@pytest.mark.parametrize("source", ["signals", "signals_fallback"])
def test_a_signals_decision_accepts_no_confidence(source):
    assert decision(source=source).confidence is None


def test_a_classifier_decision_keeps_the_reported_confidence():
    assert decision(source="classifier", decisive=False, confidence=0.5).confidence == 0.5
