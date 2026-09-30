import json

import pytest
from agent_doubles import RecordingLLM, chunk, ctx

from ragfabric_core.graph.contracts import EmptyReason, Subgraph
from ragfabric_core.providers.base import ProviderError
from ragfabric_core.strategies.auto import AutoStrategy
from ragfabric_core.strategies.base import (
    RetrievalResult,
    StrategyParams,
    StrategyRegistry,
    SubQuestionReport,
)
from ragfabric_core.strategies.base import StrategyName as S
from ragfabric_core.strategies.contract import assert_strategy_contract

RELATIONS = ["REPORTS_TO", "OWNS"]
UNDECIDED = " ".join(["onboarding"] * 20) + "?"


class Fake:
    def __init__(self, name, chunks=(), *, spend=None, extra=None):
        self.name, self._chunks, self.seen = name, list(chunks), []
        self._spend, self._extra = spend, extra or {}

    def retrieve(self, query, ctx):
        self.seen.append(ctx)
        return RetrievalResult(
            **self._extra,
            strategy=self.name,
            chunks=self._chunks,
            retrieval_calls=1,
            llm_calls=self._spend(ctx) if self._spend else 0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=1,
        )


class Boom:
    """A strategy whose model is unreachable."""

    def __init__(self, name):
        self.name = name

    def retrieve(self, query, ctx):
        raise ProviderError("fake", "model unreachable " + "x" * 400)


def build(*, found=None, llm=None, graph_enabled=True, min_confidence=0.6, spend=None, extra=None):
    found, spend, extra = found or {}, spend or {}, extra or {}
    fakes = {
        name: Fake(name, found.get(name, [chunk(9)]), spend=spend.get(name), extra=extra.get(name))
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
    span = next(s for s in result.trace if s.name == "router")
    assert span.attributes["fallback_from"] == "vectorless"
    assert span.attributes["fallback_reason"] == "no evidence"


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


def _spend_all(ctx):
    return ctx.budget.max_llm_calls


def _spend_one(ctx):
    return min(1, ctx.budget.max_llm_calls)


def test_the_classifier_path_keeps_to_the_callers_budget():
    auto, _ = build(llm=classifier(), spend={S.AGENTIC: _spend_all})
    result = assert_strategy_contract(auto, UNDECIDED, ctx(max_llm_calls=5))
    assert result.llm_calls == 5 and result.router.source == "classifier"


def test_the_fused_path_keeps_to_the_callers_budget():
    auto, _ = build(
        llm=classifier(confidence=0.2),
        spend={S.TRADITIONAL: _spend_one, S.VECTORLESS: _spend_one},
    )
    result = assert_strategy_contract(auto, UNDECIDED, ctx(max_llm_calls=2))
    assert result.router.fused and result.llm_calls <= 2


def test_classifier_then_fallback_keeps_to_the_callers_budget():
    auto, fakes = build(
        llm=classifier(),
        found={S.AGENTIC: []},
        spend={S.AGENTIC: _spend_all, S.TRADITIONAL: _spend_one},
    )
    result = assert_strategy_contract(auto, UNDECIDED, ctx(max_llm_calls=5))
    assert result.fallback_from is S.AGENTIC and result.llm_calls <= 5
    assert fakes[S.TRADITIONAL].seen[0].budget.max_llm_calls == 0


def test_a_fallback_after_a_decisive_choice_keeps_to_the_callers_budget():
    auto, _ = build(
        found={S.VECTORLESS: []},
        spend={S.VECTORLESS: _spend_all, S.TRADITIONAL: _spend_one},
    )
    result = assert_strategy_contract(auto, "What does ERR_QUOTA_4419 mean?", ctx(max_llm_calls=3))
    assert result.llm_calls <= 3


def test_the_access_filter_survives_the_deducted_context_on_the_classifier_path():
    auto, fakes = build(llm=classifier())
    caller = ctx(max_llm_calls=5)
    auto.retrieve(UNDECIDED, caller)
    seen = fakes[S.AGENTIC].seen[0]
    assert seen.budget.max_llm_calls == 4
    assert seen.access_filter == caller.access_filter and seen.principal == caller.principal


def test_the_access_filter_survives_the_deducted_context_on_the_fused_path():
    auto, fakes = build(llm=classifier(confidence=0.2))
    caller = ctx(max_llm_calls=5)
    auto.retrieve(UNDECIDED, caller)
    for name in (S.TRADITIONAL, S.VECTORLESS):
        seen = fakes[name].seen[0]
        assert seen.access_filter == caller.access_filter and seen.principal == caller.principal


def test_a_fallback_result_carries_nothing_from_the_failed_attempt():
    failed = {
        "sub_questions": [SubQuestionReport(text="part", status="open", reason="stalled")],
        "subgraph": Subgraph(
            nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_ENTITY_MATCHED
        ),
    }
    auto, _ = build(found={S.VECTORLESS: []}, extra={S.VECTORLESS: failed})
    result = auto.retrieve("What does ERR_QUOTA_4419 mean?", ctx())
    assert result.fallback_from is S.VECTORLESS
    assert result.sub_questions == [] and result.subgraph is None


def _break(auto, name):
    auto._registry.register(Boom(name))


def test_a_routed_strategy_that_raises_falls_back_to_traditional():
    auto, fakes = build(llm=classifier())
    _break(auto, S.AGENTIC)
    result = assert_strategy_contract(auto, UNDECIDED, ctx())
    assert result.strategy is S.TRADITIONAL and result.fallback_from is S.AGENTIC
    assert result.chunks and len(fakes[S.TRADITIONAL].seen) == 1
    span = next(s for s in result.trace if s.name == "router")
    assert span.attributes["fallback_from"] == "agentic"
    reason = span.attributes["fallback_reason"]
    assert reason.startswith("error: ProviderError: ") and len(reason) <= 200


def test_the_fallback_after_a_raise_keeps_to_the_callers_budget():
    auto, fakes = build(llm=classifier(), spend={S.TRADITIONAL: _spend_one})
    _break(auto, S.AGENTIC)
    result = assert_strategy_contract(auto, UNDECIDED, ctx(max_llm_calls=5))
    assert result.fallback_from is S.AGENTIC and result.llm_calls <= 5
    assert fakes[S.TRADITIONAL].seen[0].budget.max_llm_calls == 4


def test_traditional_raising_as_the_chosen_strategy_propagates():
    auto, _ = build()
    _break(auto, S.TRADITIONAL)
    with pytest.raises(ProviderError):
        auto.retrieve("What is the onboarding process?", ctx())


def test_traditional_raising_as_the_fallback_propagates():
    auto, _ = build(llm=classifier())
    _break(auto, S.AGENTIC)
    _break(auto, S.TRADITIONAL)
    with pytest.raises(ProviderError):
        auto.retrieve(UNDECIDED, ctx())


def test_a_vectorless_leg_that_raises_keeps_the_traditional_leg():
    auto, fakes = build(llm=classifier(confidence=0.2))
    _break(auto, S.VECTORLESS)
    result = assert_strategy_contract(auto, UNDECIDED, ctx())
    assert result.router.fused and result.strategy is S.TRADITIONAL
    assert result.chunks == fakes[S.TRADITIONAL]._chunks
    span = next(s for s in result.trace if s.name == "router")
    assert span.attributes["fallback_reason"].startswith("error: ProviderError")


def test_a_traditional_leg_that_raises_on_the_fused_path_propagates():
    auto, _ = build(llm=classifier(confidence=0.2))
    _break(auto, S.TRADITIONAL)
    with pytest.raises(ProviderError):
        auto.retrieve(UNDECIDED, ctx())
