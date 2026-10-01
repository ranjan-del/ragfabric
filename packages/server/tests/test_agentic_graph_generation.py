"""An agent answer built on walked graph edges is held to the graph citation contract.

``RecordingLLM`` and ``chunk`` are local copies of the doubles in
``packages/core/tests/agent_doubles.py``: the server test package does not put
the core tests on its import path, and a cross-package import would couple them.
"""

from __future__ import annotations

import json

from ragfabric_core.db.session import SessionLocal
from ragfabric_core.graph.contracts import EmptyReason
from ragfabric_core.models.document import Chunk
from ragfabric_core.providers.base import Completion, Message, ProviderError
from ragfabric_core.strategies.base import (
    RetrievalResult,
    RetrievedChunk,
    StrategyName,
    SubQuestionReport,
)
from ragfabric_core.testing.fixtures import make_txt
from ragfabric_server.api.routes.search import _generate, uses_graph_path
from ragfabric_server.deps import get_llm_provider
from ragfabric_server.main import app


class RecordingLLM:
    """A scripted provider that remembers what it was asked."""

    name = "recording"
    default_model = "recording"

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[list[Message]] = []

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.prompts.append(list(messages))
        if not self.responses:
            raise ProviderError(self.name, "script exhausted: no more responses")
        text = self.responses.pop(0)
        return Completion(
            text=text,
            model=model or self.default_model,
            provider=self.name,
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(text.split()),
            latency_ms=0,
        )

    def stream(self, messages, *, model=None, max_tokens=1024, temperature=0.0):
        yield from ()


def chunk(chunk_id: int, *, text: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        collection_id=None,
        text=text if text is not None else f"chunk {chunk_id}",
    )


def agentic_result(*, chunks, subgraph):
    return RetrievalResult(
        strategy=StrategyName.AGENTIC,
        chunks=chunks,
        retrieval_calls=1,
        llm_calls=2,
        input_tokens=0,
        output_tokens=0,
        latency_ms=1,
        subgraph=subgraph,
        sub_questions=[
            SubQuestionReport(
                text="Who is Ravi Sharma's team?",
                status="answered",
                chunk_ids=[c.chunk_id for c in chunks],
            )
        ],
    )


def test_an_agentic_result_with_edges_is_answered_on_the_graph_path(small_subgraph):
    assert uses_graph_path(agentic_result(chunks=[chunk(1)], subgraph=small_subgraph))


def test_an_agentic_result_without_edges_stays_on_the_agentic_path():
    assert not uses_graph_path(agentic_result(chunks=[chunk(1)], subgraph=None))


def test_an_unbacked_relationship_claim_in_an_agentic_answer_is_dropped(small_subgraph):
    evidence = [chunk(1, text="Ravi Sharma is a member of the Platform Team.")]
    # The backed claim cites its edge and its passage; the unbacked one names two
    # sub-graph entities that no edge joins, so the contract must drop it.
    llm = RecordingLLM(
        "Ravi Sharma is a member of the Platform Team [E 1] [1]. Ravi Sharma owns Billing [1]."
    )
    generated = _generate(
        "Tell me about Ravi Sharma",
        agentic_result(chunks=evidence, subgraph=small_subgraph),
        llm,
    )
    assert "member of the Platform Team" in generated.text
    assert "owns Billing" not in generated.text
    assert generated.dropped_relationship_claims


def test_a_subgraph_with_no_edges_is_not_the_graph_path(small_subgraph):
    empty = small_subgraph.model_copy(
        update={"edges": [], "empty_reason": EmptyReason.NO_WALKABLE_EDGES}
    )
    assert not uses_graph_path(agentic_result(chunks=[chunk(1)], subgraph=empty))


def _dated(chunk_id: int, document_id: int, date: str | None, text: str) -> RetrievedChunk:
    metadata = {"effective_date": date} if date else {}
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        collection_id=None,
        text=text,
        metadata=metadata,
    )


def test_an_agentic_graph_answer_reports_differing_effective_dates(small_subgraph):
    evidence = [
        _dated(1, 10, "2024-01-01", "Ravi Sharma is a member of the Platform Team."),
        _dated(2, 11, "2025-06-01", "Ravi Sharma leads the Platform Team."),
    ]
    llm = RecordingLLM("Ravi Sharma is a member of the Platform Team [E 1] [1].")
    generated = _generate(
        "Tell me about Ravi Sharma",
        agentic_result(chunks=evidence, subgraph=small_subgraph),
        llm,
    )
    assert len(generated.dated_sources) == 1
    assert [s.effective_date for s in generated.dated_sources[0].sources] == [
        "2024-01-01",
        "2025-06-01",
    ]
    assert "carry different effective dates" in generated.text
    assert "document 10 dated 2024-01-01 [1]" in generated.text


def test_an_agentic_graph_answer_with_one_date_adds_no_note(small_subgraph):
    evidence = [_dated(1, 10, "2024-01-01", "Ravi Sharma is a member of the Platform Team.")]
    llm = RecordingLLM("Ravi Sharma is a member of the Platform Team [E 1] [1].")
    generated = _generate(
        "Tell me about Ravi Sharma",
        agentic_result(chunks=evidence, subgraph=small_subgraph),
        llm,
    )
    assert generated.dated_sources == []
    assert "effective dates" not in generated.text


class _CannedAgent:
    """Stands in for the agent: returns a fixed result carrying a walked sub-graph."""

    name = StrategyName.AGENTIC

    def __init__(self, result: RetrievalResult) -> None:
        self.result = result

    def retrieve(self, query, ctx):
        return self.result

    def access_stats(self, filters, access):
        return (1, 1)


class _StreamingLLM:
    name = "streaming"
    default_model = "streaming"

    def complete(self, messages, **kwargs):
        raise AssertionError("the streaming route must not call complete")

    def stream(self, messages, **kwargs):
        text = (
            "Ravi Sharma is a member of the Platform Team [E 1] [1]. Ravi Sharma owns Billing [1]."
        )
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


def _events(raw: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for block in raw.strip().split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        if name:
            out[name] = data
    return out


def test_the_stream_holds_an_agentic_graph_answer_to_the_graph_contract(
    client, admin_token, small_subgraph
):
    text = "Ravi Sharma is a member of the Platform Team."
    uploaded = client.post(
        "/api/documents/upload",
        files={"file": ("org.txt", make_txt(text), "text/plain")},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert uploaded.status_code == 201, uploaded.text
    document_id = uploaded.json()["id"]
    with SessionLocal() as db:
        chunk_id = db.query(Chunk).filter(Chunk.document_id == document_id).one().id
    evidence = [
        RetrievedChunk(chunk_id=chunk_id, document_id=document_id, collection_id=None, text=text)
    ]
    # The sub-graph fixture sources its edge from chunk 1; point it at the real one.
    subgraph = small_subgraph.model_copy(
        update={
            "edges": [
                e.model_copy(update={"source_chunk_ids": [chunk_id]}) for e in small_subgraph.edges
            ]
        }
    )
    client.app.state.strategy_registry.register(
        _CannedAgent(agentic_result(chunks=evidence, subgraph=subgraph))
    )
    app.dependency_overrides[get_llm_provider] = lambda: _StreamingLLM()
    try:
        with client.stream(
            "POST",
            "/api/ask",
            json={"query": "Tell me about Ravi Sharma", "strategy": "agentic", "stream": True},
            headers={"Authorization": f"Bearer {admin_token}"},
        ) as res:
            assert res.status_code == 200
            events = _events("".join(res.iter_text()))
    finally:
        app.dependency_overrides.pop(get_llm_provider, None)

    superseded = events["superseded"]
    assert "owns Billing" not in superseded["text"]
    assert [c["text"] for c in superseded["dropped_relationship_claims"]] == [
        "Ravi Sharma owns Billing [1]."
    ]
