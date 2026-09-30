import json

import pytest
from agent_doubles import RecordingLLM, chunk, ctx

from ragfabric_core.strategies.auto import AutoStrategy
from ragfabric_core.strategies.base import (
    RetrievalResult,
    StrategyParams,
    StrategyRegistry,
)
from ragfabric_core.strategies.base import StrategyName as S
from ragfabric_core.strategies.contract import assert_strategy_contract

RELATIONS = ["REPORTS_TO", "OWNS"]
UNDECIDED = " ".join(["onboarding"] * 20) + "?"


class Fake:
    def __init__(self, name, chunks=()):
        self.name, self._chunks, self.seen = name, list(chunks), []

    def retrieve(self, query, ctx):
        self.seen.append(ctx)
        return RetrievalResult(
            strategy=self.name,
            chunks=self._chunks,
            retrieval_calls=1,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=1,
        )


def build(*, found=None, llm=None, graph_enabled=True, min_confidence=0.6):
    found = found or {}
    fakes = {
        name: Fake(name, found.get(name, [chunk(9)]))
        for name in (S.TRADITIONAL, S.VECTORLESS, S.AGENTIC, S.GRAPH)
    }
    registry = StrategyRegistry()
    for fake in fakes.values():
        registry.register(fake)
    auto = AutoStrategy(
        registry=registry,
        llm=llm,
        min_confidence=min_confidence,
        classifier_model=None,
        graph_enabled=graph_enabled,
        relation_types=RELATIONS,
    )
    registry.register(auto)
    return auto, fakes


def classifier(strategy="agentic", confidence=0.9, reasoning="Two documents."):
    return RecordingLLM(
        json.dumps(
            {
                "query_type": "multi_hop",
                "strategy": strategy,
                "confidence": confidence,
                "reasoning": reasoning,
            }
        )
    )


def test_a_decisive_question_is_routed_with_no_model_call():
    llm = classifier()
    auto, fakes = build(llm=llm)
    result = assert_strategy_contract(auto, "What does ERR_QUOTA_4419 mean?", ctx())
    assert result.strategy is S.VECTORLESS and result.router.source == "signals"
    assert llm.calls == 0 and result.llm_calls == 0


def test_an_undecided_question_is_classified_and_the_call_is_counted():
    llm = classifier()
    auto, fakes = build(llm=llm)
    result = auto.retrieve(UNDECIDED, ctx())
    assert result.strategy is S.AGENTIC and result.router.source == "classifier"
    assert result.llm_calls == 1 and result.router.confidence == 0.9


def test_the_chosen_strategy_gets_a_budget_with_the_classifier_call_deducted():
    auto, fakes = build(llm=classifier())
    auto.retrieve(UNDECIDED, ctx(max_llm_calls=5))
    assert fakes[S.AGENTIC].seen[0].budget.max_llm_calls == 4


def test_a_zero_call_budget_skips_the_classifier():
    llm = classifier()
    auto, _ = build(llm=llm)
    result = auto.retrieve(UNDECIDED, ctx(max_llm_calls=0))
    assert llm.calls == 0 and result.router.source == "signals_fallback"


def test_low_confidence_fuses_traditional_and_vectorless():
    auto, fakes = build(
        llm=classifier(confidence=0.2),
        found={S.TRADITIONAL: [chunk(1)], S.VECTORLESS: [chunk(2)]},
    )
    result = auto.retrieve(UNDECIDED, ctx())
    assert result.router.fused and {c.chunk_id for c in result.chunks} == {1, 2}


def test_a_blank_reasoning_from_the_classifier_is_treated_as_a_violation_and_fused():
    auto, _ = build(llm=classifier(reasoning="   "))
    result = auto.retrieve(UNDECIDED, ctx())
    assert result.router.fused and result.router.reasoning


def test_an_empty_choice_falls_back_once_and_records_it():
    auto, fakes = build(found={S.VECTORLESS: []})
    result = auto.retrieve("What does ERR_QUOTA_4419 mean?", ctx())
    assert result.strategy is S.TRADITIONAL and result.fallback_from is S.VECTORLESS
    assert len(fakes[S.TRADITIONAL].seen) == 1


def test_a_fallback_that_also_finds_nothing_stops():
    auto, fakes = build(found={S.VECTORLESS: [], S.TRADITIONAL: []})
    result = auto.retrieve("What does ERR_QUOTA_4419 mean?", ctx())
    assert result.chunks == [] and len(fakes[S.TRADITIONAL].seen) == 1


@pytest.mark.parametrize("filters", [{"document_id": 3}, {"format": "pdf"}])
def test_graph_is_never_used_under_a_filter_the_walk_cannot_apply(filters):
    auto, fakes = build()
    filtered = ctx().model_copy(update={"params": StrategyParams(metadata_filters=filters)})
    result = auto.retrieve("Who does Ravi Sharma report to?", filtered)
    assert fakes[S.GRAPH].seen == [] and result.strategy is not S.GRAPH


def test_graph_disabled_is_never_selected():
    auto, fakes = build(graph_enabled=False)
    auto.retrieve("Who does Ravi Sharma report to?", ctx())
    assert fakes[S.GRAPH].seen == []


def test_every_strategy_receives_the_callers_own_access_filter():
    auto, fakes = build(found={S.VECTORLESS: []})
    caller = ctx()
    auto.retrieve("What does ERR_QUOTA_4419 mean?", caller)
    for fake in fakes.values():
        for seen in fake.seen:
            assert seen.access_filter == caller.access_filter
            assert seen.principal == caller.principal


def test_auto_is_never_a_routing_target():
    auto, _ = build()
    assert S.AUTO not in auto.available(ctx())
