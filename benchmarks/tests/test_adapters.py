"""Comparison adapters with duck typed fakes: the frameworks themselves are never imported."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters import langchain_target, llamaindex_target  # noqa: E402
from adapters.common import FrameworkTarget, Passage  # noqa: E402

from ragfabric_core.evaluation.target import EvalTarget, RetrievingTarget  # noqa: E402

P = [
    Passage(document="a.txt", text="Alpha.", score=0.9),
    Passage(document="b.txt", text="Beta.", score=0.5),
]


def make_target():
    return FrameworkTarget(
        "fake", lambda q: list(P), lambda q: ("Alpha is first.", list(P)), llm_model="m"
    )


def test_framework_target_is_a_retrieving_eval_target():
    t = make_target()
    assert isinstance(t, EvalTarget) and isinstance(t, RetrievingTarget)


def test_answer_maps_passages_to_ranked_contexts_and_declares_no_citations():
    out = make_target().answer("q")
    assert out.answer == "Alpha is first." and out.cites is False
    assert [(c.rank, c.document) for c in out.contexts] == [(1, "a.txt"), (2, "b.txt")]
    assert out.input_tokens is None and out.output_tokens is None
    assert out.llm_model == "m" and out.latency_ms is not None


def test_retrieve_generates_nothing():
    out = make_target().retrieve("q")
    assert out.answer == "" and len(out.contexts) == 2 and out.llm_calls is None


def test_llamaindex_nodes_become_passages():
    node = SimpleNamespace(
        metadata={"document": "d.txt", "collection_id": 3, "chunk_id": 7},
        get_content=lambda: "text",
    )
    (p,) = llamaindex_target._passages([SimpleNamespace(node=node, score=0.4)])
    assert (p.document, p.text, p.score, p.collection_id, p.chunk_id) == (
        "d.txt",
        "text",
        0.4,
        3,
        7,
    )


def test_langchain_documents_become_passages():
    doc = SimpleNamespace(page_content="t", metadata={"document": "e.txt", "collection_id": 2})
    (p,) = langchain_target._passages([(doc, 0.25)])
    assert (p.document, p.text, p.score, p.collection_id) == ("e.txt", "t", 0.25, 2)


def test_langchain_prompt_has_both_slots():
    assert (
        "{question}" in langchain_target.RAG_PROMPT and "{context}" in langchain_target.RAG_PROMPT
    )
