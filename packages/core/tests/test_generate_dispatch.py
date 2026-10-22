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


def _agentic_answer(monkeypatch, text="ten days [1]"):
    from ragfabric_core.generate.cited import AgenticAnswer

    monkeypatch.setattr(
        dispatch,
        "generate_agentic_answer",
        lambda query, chunks, llm, sub_question_evidence: AgenticAnswer(
            text=text, model="m", generator="llm"
        ),
    )


def _agentic_result(**extra):
    return _result(
        StrategyName.AGENTIC,
        sub_questions=[SubQuestionReport(text="leave", status="open", chunk_ids=[1])],
        **extra,
    )


def test_a_budget_stop_says_so_in_the_answer(monkeypatch):
    _agentic_answer(monkeypatch)
    result = _agentic_result(
        stop_reason="budget", stop_detail="max_latency_ms of 30000 reached after 31000ms"
    )

    text = dispatch.generate_for_result("q", result, OfflineLLMProvider()).text

    assert text.startswith("ten days [1]")
    assert "stopped" in text and "max_latency_ms of 30000" in text


def test_a_budget_stop_with_no_evidence_gives_the_reason_not_a_bare_refusal(monkeypatch):
    result = _agentic_result(stop_reason="budget", stop_detail="max_latency_ms of 30000 reached")
    result = result.model_copy(update={"chunks": []})

    text = dispatch.generate_for_result("q", result, OfflineLLMProvider()).text

    assert "max_latency_ms of 30000" in text
    assert "any evidence" in text
    assert "could not find" in text  # still recognised as a refusal


def test_a_resolved_agentic_answer_gets_no_note(monkeypatch):
    _agentic_answer(monkeypatch)
    result = _agentic_result(stop_reason="resolved", stop_detail="every sub-question is answered")

    text = dispatch.generate_for_result("q", result, OfflineLLMProvider()).text

    assert text == "ten days [1]"


def test_the_system_prompt_lets_a_model_choose_between_conflicting_versions():
    """Two policy versions in one context made llama3.1:8b refuse a question that the
    matching passage answers, so the prompt says to use the matching passage and to
    refuse only when no passage answers."""
    from ragfabric_core.generate.cited import NO_EVIDENCE_ANSWER, SYSTEM_PROMPT

    assert "different documents, years or versions" in SYSTEM_PROMPT
    assert "Only if no passage answers" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.endswith(NO_EVIDENCE_ANSWER)


def test_the_graph_system_prompt_carries_the_same_refusal_rule():
    """The graph prompt kept the old unconditional refusal when the traditional one was
    fixed, so graph still refused questions whose retrieved passages answer them. It now
    says the same thing the other prompt says."""
    from ragfabric_core.generate.cited import GRAPH_SYSTEM_PROMPT, NO_EVIDENCE_ANSWER

    assert "different documents, years or versions" in GRAPH_SYSTEM_PROMPT
    assert "Answer whenever any passage contains the information asked for" in GRAPH_SYSTEM_PROMPT
    assert "Only if no passage answers" in GRAPH_SYSTEM_PROMPT
    assert GRAPH_SYSTEM_PROMPT.endswith(NO_EVIDENCE_ANSWER)
