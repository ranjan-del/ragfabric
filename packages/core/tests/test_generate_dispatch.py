"""Generation dispatch lives in core so the API and the evaluation run one code path (D5)."""

from ragfabric_core.generate import dispatch
from ragfabric_core.generate.cited import CitedAnswer
from ragfabric_core.providers.offline import OfflineLLMProvider
from ragfabric_core.strategies.base import (
    RetrievalResult,
    RetrievedChunk,
    StrategyName,
    SubQuestionReport,
)


def _chunk(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=i, document_id=1, collection_id=None, text=text, page=None)


def _result(strategy=StrategyName.TRADITIONAL, **extra) -> RetrievalResult:
    values = dict(
        strategy=strategy,
        chunks=[_chunk(1, "Employees get 10 days of casual leave in 2026.")],
        retrieval_calls=1,
        llm_calls=0,
        input_tokens=0,
        output_tokens=0,
        latency_ms=1,
    )
    values.update(extra)
    return RetrievalResult(**values)


def test_a_traditional_result_gets_a_cited_answer():
    generated = dispatch.generate_for_result(
        "How many days of casual leave?", _result(), OfflineLLMProvider()
    )
    assert "[1]" in generated.text


def test_a_graph_result_takes_the_graph_path():
    assert dispatch.uses_graph_path(_result(StrategyName.GRAPH)) is True
    assert dispatch.uses_graph_path(_result()) is False


def test_a_result_with_sub_questions_takes_the_agentic_path(monkeypatch):
    seen = {}

    def fake(query, chunks, llm, sub_question_evidence):
        seen["called"] = True
        from ragfabric_core.generate.cited import AgenticAnswer

        return AgenticAnswer(
            text="ten days [1]", model="m", generator="llm", input_tokens=3, output_tokens=2
        )

    monkeypatch.setattr(dispatch, "generate_agentic_answer", fake)
    result = _result(
        StrategyName.AGENTIC,
        sub_questions=[SubQuestionReport(text="leave", status="answered", chunk_ids=[1])],
    )
    generated = dispatch.generate_for_result("q", result, OfflineLLMProvider())
    assert seen == {"called": True}
    assert generated.llm_calls == 1


def test_cited_llm_calls_counts_real_calls():
    def cited(generator, retried):
        return CitedAnswer(text="x", model="m", generator=generator, retried=retried)

    assert dispatch.cited_llm_calls(cited("llm", True)) == 2
    assert dispatch.cited_llm_calls(cited("extractive", True)) == 2
    assert dispatch.cited_llm_calls(cited("llm", False)) == 1
    assert dispatch.cited_llm_calls(cited("extractive", False)) == 0


def test_the_server_reexports_the_same_objects():
    from ragfabric_server.api.routes import search

    assert search._generate is dispatch.generate_for_result
    assert search.Generated is dispatch.Generated
    assert search.uses_graph_path is dispatch.uses_graph_path
    assert search._cited_llm_calls is dispatch.cited_llm_calls
