"""An agent answer built on walked graph edges is held to the graph citation contract.

``RecordingLLM`` and ``chunk`` are local copies of the doubles in
``packages/core/tests/agent_doubles.py``: the server test package does not put
the core tests on its import path, and a cross-package import would couple them.
"""

from __future__ import annotations

from ragfabric_core.providers.base import Completion, Message, ProviderError
from ragfabric_core.strategies.base import (
    RetrievalResult,
    RetrievedChunk,
    StrategyName,
    SubQuestionReport,
)
from ragfabric_server.api.routes.search import _generate, uses_graph_path


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
