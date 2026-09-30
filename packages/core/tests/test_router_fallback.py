import pytest
from agent_doubles import chunk

from ragfabric_core.graph.contracts import EmptyReason, Subgraph
from ragfabric_core.router.fallback import combine, empty_reason, fallback_for, fuse
from ragfabric_core.strategies.base import RetrievalResult, SubQuestionReport, TraceSpan
from ragfabric_core.strategies.base import StrategyName as S


def result(strategy, chunks=(), **extra):
    fields = {
        "retrieval_calls": 1,
        "llm_calls": 1,
        "embedding_calls": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "latency_ms": 100,
    }
    return RetrievalResult(strategy=strategy, chunks=list(chunks), **(fields | extra))


@pytest.mark.parametrize("chosen", [S.GRAPH, S.VECTORLESS, S.AGENTIC])
def test_an_empty_result_falls_back_to_traditional(chosen):
    assert fallback_for(chosen, result(chosen)) is S.TRADITIONAL


def test_traditional_never_falls_back():
    assert fallback_for(S.TRADITIONAL, result(S.TRADITIONAL)) is None


def test_a_result_with_evidence_never_falls_back():
    assert fallback_for(S.AGENTIC, result(S.AGENTIC, [chunk(1)])) is None


def test_the_graph_reason_is_reported_when_there_is_one():
    empty = Subgraph(
        nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_ENTITY_MATCHED
    )
    assert empty_reason(result(S.GRAPH, subgraph=empty)) == "no_entity_matched"


def test_a_generic_reason_is_reported_without_a_subgraph():
    assert empty_reason(result(S.VECTORLESS)) == "no evidence"


def test_combine_sums_every_counter_and_keeps_both_traces():
    merged = combine(result(S.AGENTIC), result(S.TRADITIONAL, [chunk(1)]), fallback_from=S.AGENTIC)
    assert (merged.retrieval_calls, merged.llm_calls, merged.embedding_calls) == (2, 2, 2)
    assert (merged.input_tokens, merged.output_tokens, merged.latency_ms) == (20, 10, 200)
    assert merged.strategy is S.TRADITIONAL and merged.fallback_from is S.AGENTIC


def test_combine_drops_the_failed_attempts_sub_questions_and_subgraph():
    empty = Subgraph(
        nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_GRAPH_COVERAGE
    )
    failed = result(
        S.AGENTIC,
        sub_questions=[SubQuestionReport(text="a", status="open", reason="budget")],
        subgraph=empty,
    )
    merged = combine(failed, result(S.TRADITIONAL, [chunk(1)]), fallback_from=S.AGENTIC)
    assert merged.sub_questions == [] and merged.subgraph is None


def test_fuse_only_reorders_chunks_the_strategies_returned():
    fused = fuse(
        result(S.TRADITIONAL, [chunk(1), chunk(2)]),
        result(S.VECTORLESS, [chunk(2), chunk(3)]),
        top_k=5,
    )
    assert {c.chunk_id for c in fused.chunks} == {1, 2, 3}
    assert fused.chunks[0].chunk_id == 2
    assert fused.strategy is S.TRADITIONAL


def test_fuse_cuts_to_top_k_after_fusion():
    fused = fuse(
        result(S.TRADITIONAL, [chunk(i) for i in range(10)]), result(S.VECTORLESS, []), top_k=3
    )
    assert len(fused.chunks) == 3


def span(name):
    return TraceSpan(name=name, started_ms=0, duration_ms=1)


def test_combine_keeps_both_traces_in_order():
    merged = combine(
        result(S.AGENTIC, trace=[span("first")]),
        result(S.TRADITIONAL, [chunk(1)], trace=[span("second")]),
        fallback_from=S.AGENTIC,
    )
    assert merged.trace == [span("first"), span("second")]


def test_fuse_keeps_both_traces_in_order():
    fused = fuse(
        result(S.TRADITIONAL, [chunk(1)], trace=[span("trad")]),
        result(S.VECTORLESS, [chunk(2)], trace=[span("vec")]),
        top_k=5,
    )
    assert fused.trace == [span("trad"), span("vec")]


def test_fuse_sums_every_counter():
    fused = fuse(
        result(
            S.TRADITIONAL,
            [chunk(1)],
            retrieval_calls=1,
            llm_calls=2,
            embedding_calls=3,
            input_tokens=4,
            output_tokens=5,
            latency_ms=6,
        ),
        result(
            S.VECTORLESS,
            [chunk(2)],
            retrieval_calls=10,
            llm_calls=20,
            embedding_calls=30,
            input_tokens=40,
            output_tokens=50,
            latency_ms=60,
        ),
        top_k=5,
    )
    assert (
        fused.retrieval_calls,
        fused.llm_calls,
        fused.embedding_calls,
        fused.input_tokens,
        fused.output_tokens,
        fused.latency_ms,
    ) == (11, 22, 33, 44, 55, 66)
