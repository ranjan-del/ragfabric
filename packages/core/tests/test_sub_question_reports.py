"""Per sub-question outcomes on the result, in place of one best-effort flag.

A boolean cannot say which part of a decomposed question failed, or why, and a
caller who is told only "this answer is partial" has to guess at the gap. The
report says it: one row per sub-question, its final status, and for anything
still open the reason the run left it open.

The vocabulary is the ledger's own (``answered``, ``abandoned``, ``open``)
rather than a second set of words meaning the same thing. ``open`` on a
finished run means exactly what it says: the run stopped with this part still
unanswered, and ``reason`` says what stopped it.

The field defaults to empty, so a strategy that does not decompose reports
nothing rather than being made to describe itself in terms of an agent.
"""

from __future__ import annotations

import json

from agent_doubles import FakeTool, RecordingLLM, SequenceTool, chunk
from agent_doubles import ctx as make_ctx
from test_strategy_vectorless import StubLexicalStore
from test_traditional_strategy import StubStore

from ragfabric_core.providers.offline import HashingEmbeddingProvider
from ragfabric_core.strategies.agentic import AgenticRAGStrategy
from ragfabric_core.strategies.base import RetrievedChunk, SubQuestionReport
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy
from ragfabric_core.strategies.vectorless import VectorlessRAGStrategy


def plan_json(*pairs: tuple[str, str]) -> str:
    return json.dumps(
        {"sub_questions": [{"text": text, "tool": tool, "why": ""} for text, tool in pairs]}
    )


def assess_json(*verdicts: tuple[str, bool, str | None]) -> str:
    return json.dumps(
        {
            "verdicts": [
                {"sub_question": text, "answered": answered, "missing": missing}
                for text, answered, missing in verdicts
            ]
        }
    )


def repair_json(move: str, why: str = "") -> str:
    return json.dumps({"move": move, "rewritten_query": None, "why": why})


def report_for(reports: list[SubQuestionReport], text: str) -> SubQuestionReport:
    return next(report for report in reports if report.text == text)


def test_an_answered_sub_question_reports_no_reason() -> None:
    tool = FakeTool("semantic_search", chunks=[chunk(1)])
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assess_json(("what is the retry limit", True, None)),
    )
    strategy = AgenticRAGStrategy(llm=llm, tools={"semantic_search": tool})

    result = strategy.retrieve("what is the retry limit", make_ctx())

    assert [(r.text, r.status, r.reason) for r in result.sub_questions] == [
        ("what is the retry limit", "answered", None)
    ]
    assert result.sub_questions[0].chunk_ids == [1]


def test_an_unanswered_sub_question_reports_why() -> None:
    """The budget stopped the run, and the report says so on the part it left open."""
    tool = FakeTool("semantic_search", chunks=[chunk(1)])
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assess_json(("what is the retry limit", False, "the number of retries")),
        repair_json("broaden"),
    )
    strategy = AgenticRAGStrategy(llm=llm, tools={"semantic_search": tool})

    result = strategy.retrieve("what is the retry limit", make_ctx(max_llm_calls=2))

    report = result.sub_questions[0]
    assert report.status == "open"
    assert report.reason is not None
    assert "budget" in report.reason
    assert "max_llm_calls of 2" in report.reason


def test_a_stalled_run_says_it_stopped_making_progress() -> None:
    tool = FakeTool("semantic_search", chunks=[chunk(1)])
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assess_json(("what is the retry limit", False, "the number of retries")),
        repair_json("broaden"),
    )
    strategy = AgenticRAGStrategy(llm=llm, tools={"semantic_search": tool})

    result = strategy.retrieve("what is the retry limit", make_ctx())

    report = result.sub_questions[0]
    assert report.status == "open"
    assert "no_progress" in (report.reason or "")
    assert "1 chunk" in (report.reason or "")


def test_a_sub_question_nothing_was_retrieved_for_says_exactly_that() -> None:
    tool = FakeTool("semantic_search", chunks=[])
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assess_json(("what is the retry limit", False, "the number of retries")),
        repair_json("broaden"),
    )
    strategy = AgenticRAGStrategy(llm=llm, tools={"semantic_search": tool})

    result = strategy.retrieve(
        "what is the retry limit",
        make_ctx(),
    )

    report = result.sub_questions[0]
    assert report.status == "open"
    assert "no evidence was retrieved" in (report.reason or "")
    assert report.chunk_ids == []


def test_an_abandoned_sub_question_carries_the_reason_it_was_abandoned() -> None:
    tool = SequenceTool("semantic_search", batches=[[chunk(1)], [chunk(2)]])
    llm = RecordingLLM(
        plan_json(("what is the retry limit", "semantic_search")),
        assess_json(("what is the retry limit", False, "the number of retries")),
        repair_json("broaden"),
        assess_json(("what is the retry limit", False, "the number of retries")),
        repair_json("abandon", why="nothing in the corpus states the limit"),
    )
    strategy = AgenticRAGStrategy(llm=llm, tools={"semantic_search": tool})

    result = strategy.retrieve("what is the retry limit", make_ctx())

    report = result.sub_questions[0]
    assert report.status == "abandoned"
    assert report.reason == "nothing in the corpus states the limit"


def test_every_sub_question_of_a_decomposed_question_is_reported() -> None:
    semantic = FakeTool("semantic_search", chunks=[chunk(1)])
    lexical = FakeTool("lexical_search", chunks=[chunk(2, document_id=2)])
    llm = RecordingLLM(
        plan_json(
            ("what is the retry limit", "semantic_search"),
            ("who signs off the change", "lexical_search"),
        ),
        assess_json(
            ("what is the retry limit", True, None),
            ("who signs off the change", False, "the approver"),
        ),
        repair_json("broaden"),
    )
    strategy = AgenticRAGStrategy(
        llm=llm, tools={"semantic_search": semantic, "lexical_search": lexical}
    )

    result = strategy.retrieve("what is the retry limit and who signs off the change", make_ctx())

    assert [r.text for r in result.sub_questions] == [
        "what is the retry limit",
        "who signs off the change",
    ]
    assert report_for(result.sub_questions, "what is the retry limit").status == "answered"
    unanswered = report_for(result.sub_questions, "who signs off the change")
    assert unanswered.status == "open"
    assert unanswered.reason
    assert unanswered.chunk_ids == [2]


def test_other_strategies_return_an_empty_report() -> None:
    """Nothing but the agent decomposes, so nothing but the agent reports parts."""
    hits = [
        RetrievedChunk(
            chunk_id=1, document_id=1, collection_id=None, text="annual leave", score=0.9
        )
    ]
    traditional = TraditionalRAGStrategy(
        embedding_provider=HashingEmbeddingProvider(dim=64),
        vector_store=StubStore(hits),
    )
    vectorless = VectorlessRAGStrategy(
        bm25_store=StubLexicalStore("bm25", hits),
        ts_rank_store=StubLexicalStore("ts_rank", hits),
    )

    assert traditional.retrieve("annual leave", make_ctx()).sub_questions == []
    assert vectorless.retrieve("annual leave", make_ctx()).sub_questions == []
