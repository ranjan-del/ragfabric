"""Tests for the ragfabric_sdk client, against httpx.MockTransport only.

No test here talks to a network or a running server: every handler below
builds the response by hand, shaped exactly like the real server's routes in
packages/server/src/ragfabric_server/api/routes/ and schemas in
packages/server/src/ragfabric_server/schemas/.
"""

import email
import json

import httpx
import pytest

from ragfabric_sdk import Client
from ragfabric_sdk.errors import AuthError, NotFoundError, RateLimitError
from ragfabric_sdk.models import Document


def transport(handler):
    return httpx.MockTransport(handler)


def client_with(handler, **kwargs):
    return Client("http://server", token="t", transport=transport(handler), **kwargs)


def parse_multipart(request: httpx.Request) -> dict[str, str | bytes]:
    """Decode a MockTransport-captured multipart/form-data request into a
    {field name: value} dict, keyed exactly by the field's ``name`` so
    "collection" and "collection_id" are never confused with one another.

    Uses the standard library's ``email`` parser rather than substring
    matching on the raw bytes, since a substring check for ``name="x"``
    would also match inside ``name="x_id"``.
    """
    content_type = request.headers["content-type"]
    raw = b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + request.content
    message = email.message_from_bytes(raw)
    fields: dict[str, str | bytes] = {}
    for part in message.get_payload():
        name = part.get_param("name", header="Content-Disposition")
        payload = part.get_payload(decode=True)
        fields[name] = payload if part.get_filename() else payload.decode()
    return fields


_DOCUMENT_RESPONSE = {
    "id": 1,
    "filename": "doc.txt",
    "format": "txt",
    "document_type": "",
    "storage_path": None,
    "content_type": "text/plain",
    "status": "ready",
    "collection_id": 42,
    "owner_id": 1,
    "version": 1,
    "num_chunks": 2,
    "error": "",
    "created_at": "2026-09-18T00:00:00",
}


def test_ask_returns_a_typed_answer():
    def handler(request):
        assert request.url.path == "/api/ask"
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "a [1]",
                "confidence": 0.8,
                "citations": [
                    {"marker": "[1]", "chunk_id": 3, "document_id": 1, "score": 0.9, "snippet": "s"}
                ],
                "highlights": [],
            },
        )

    answer = client_with(handler).ask("q")
    assert answer.answer == "a [1]"
    assert answer.citations[0].chunk_id == 3
    assert answer.confidence == pytest.approx(0.8)
    # Fields absent from the response body fall back to the schema's
    # defaults rather than raising.
    assert answer.citations[0].used is False
    assert answer.citations[0].filename is None
    assert answer.source_document is None


def test_the_bearer_token_is_sent():
    def handler(request):
        assert request.headers["authorization"] == "Bearer t"
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "",
                "confidence": 0.0,
                "citations": [],
                "highlights": [],
            },
        )

    client_with(handler).ask("q")


def test_an_api_key_is_sent_in_the_x_api_key_header():
    def handler(request):
        assert request.headers["x-api-key"] == "rf_abc"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "",
                "confidence": 0.0,
                "citations": [],
                "highlights": [],
            },
        )

    Client("http://server", api_key="rf_abc", transport=transport(handler)).ask("q")


def test_an_api_key_wins_when_both_are_given():
    """The class docstring documents this precedence; pin it with a test."""

    def handler(request):
        assert request.headers["x-api-key"] == "rf_abc"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "",
                "confidence": 0.0,
                "citations": [],
                "highlights": [],
            },
        )

    Client("http://server", token="t", api_key="rf_abc", transport=transport(handler)).ask("q")


def test_ask_stream_yields_parsed_events():
    body = (
        'event: retrieval\ndata: {"chunks": 2}\n\n'
        'event: token\ndata: {"text": "hello "}\n\n'
        'event: token\ndata: {"text": "world"}\n\n'
        'event: citations\ndata: {"citations": []}\n\n'
        'event: done\ndata: {"run_id": 7, "latency_ms": 12}\n\n'
    )

    def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = list(client_with(handler).ask_stream("q"))
    assert [e.event for e in events] == ["retrieval", "token", "token", "citations", "done"]
    assert "".join(e.data["text"] for e in events if e.event == "token") == "hello world"
    assert events[-1].data["run_id"] == 7


def test_ask_stream_surfaces_a_superseded_event():
    """A citation contract violation on the streamed text is reported as a
    superseded event carrying the corrected text, not silently swallowed and
    not raised as an error: the SDK must pass it through so a caller can
    replace what it already rendered."""
    body = (
        'event: retrieval\ndata: {"chunks": 1}\n\n'
        'event: token\ndata: {"text": "unverified claim"}\n\n'
        'event: superseded\ndata: {"text": "corrected [1] answer", "reason": "citation contract"}\n\n'
        'event: citations\ndata: {"citations": []}\n\n'
        'event: done\ndata: {"run_id": 9, "latency_ms": 40}\n\n'
    )

    def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = list(client_with(handler).ask_stream("q"))
    assert [e.event for e in events] == ["retrieval", "token", "superseded", "citations", "done"]
    superseded = events[2]
    assert superseded.data["text"] == "corrected [1] answer"
    assert superseded.data["reason"] == "citation contract"


def test_ask_stream_defaults_a_bare_data_line_to_the_message_event():
    """Per the SSE wire format, a data: line with no preceding event: line
    defaults to the event name "message". The real server always sends an
    explicit event: line today, but the parser should follow the spec, not
    just the one server it was written against."""
    body = 'data: {"chunks": 2}\n\n'

    def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = list(client_with(handler).ask_stream("q"))
    assert len(events) == 1
    assert events[0].event == "message"
    assert events[0].data == {"chunks": 2}


def test_a_401_becomes_an_auth_error():
    def handler(request):
        return httpx.Response(401, json={"detail": "not authenticated"})

    with pytest.raises(AuthError):
        client_with(handler).ask("q")


def test_a_403_also_becomes_an_auth_error():
    def handler(request):
        return httpx.Response(403, json={"detail": "insufficient permissions"})

    with pytest.raises(AuthError) as exc_info:
        client_with(handler).ask("q")
    assert "insufficient permissions" in str(exc_info.value)


def test_a_429_becomes_a_rate_limit_error():
    def handler(request):
        return httpx.Response(429, json={"detail": "too many"})

    with pytest.raises(RateLimitError):
        client_with(handler).ask("q")


def test_a_404_becomes_a_not_found_error():
    def handler(request):
        return httpx.Response(404, json={"detail": "no such run"})

    with pytest.raises(NotFoundError):
        client_with(handler).run(99)


def test_search_returns_typed_results():
    def handler(request):
        assert request.url.path == "/api/search/semantic"
        return httpx.Response(
            200,
            json={
                "query": "q",
                "mode": "semantic",
                "results": [{"chunk_id": 1, "document_id": 2, "score": 0.5, "text": "t"}],
            },
        )

    results = client_with(handler).search("q")
    assert results[0].score == pytest.approx(0.5)
    assert results[0].text == "t"


def test_search_hybrid_uses_the_hybrid_path():
    def handler(request):
        assert request.url.path == "/api/search/hybrid"
        return httpx.Response(
            200,
            json={"query": "q", "mode": "hybrid", "results": []},
        )

    client_with(handler).search("q", mode="hybrid")


def test_ingest_sends_the_real_multipart_shape(tmp_path):
    """The plan's reference sent the collection as a form field named
    "collection"; the real upload route (api/routes/documents.py) reads
    "collection_id", "chunk_size" and "chunk_overlap" as Form fields. This
    test inspects the outgoing request rather than only the parsed response,
    so a regression back to the wrong field name would be caught here."""
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello world")
    captured: dict[str, httpx.Request] = {}

    def handler(request):
        assert request.url.path == "/api/documents/upload"
        captured["request"] = request
        return httpx.Response(201, json=_DOCUMENT_RESPONSE)

    doc = client_with(handler).ingest(file_path, collection=42, chunk_size=500)

    fields = parse_multipart(captured["request"])
    assert fields["file"] == b"hello world"
    assert fields["collection_id"] == "42"
    assert "collection" not in fields
    assert fields["chunk_size"] == "500"
    # chunk_overlap was never supplied: the server treats an absent field as
    # "use the configured value", which is a different thing from a form
    # field carrying an empty string or the literal text "None".
    assert "chunk_overlap" not in fields
    assert isinstance(doc, Document)
    assert doc.filename == "doc.txt"


def test_ingest_omits_every_optional_field_when_none_are_given(tmp_path):
    """With no collection/chunk_size/chunk_overlap supplied, the client must
    send only the file: it must not stringify None into any of these form
    fields."""
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello world")
    captured: dict[str, httpx.Request] = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(201, json=_DOCUMENT_RESPONSE)

    client_with(handler).ingest(file_path)

    fields = parse_multipart(captured["request"])
    assert set(fields) == {"file"}


def test_documents_parses_the_items_envelope():
    """GET /api/documents always returns {"items": [...], "total": N}
    (schemas.document.DocumentList); there is no bare-list shape to
    tolerate."""

    def handler(request):
        assert request.url.path == "/api/documents"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": 1,
                        "filename": "a.pdf",
                        "format": "pdf",
                        "document_type": "",
                        "storage_path": None,
                        "content_type": "application/pdf",
                        "status": "ready",
                        "collection_id": None,
                        "owner_id": 1,
                        "version": 1,
                        "num_chunks": 3,
                        "error": "",
                        "created_at": "2026-09-18T00:00:00",
                    }
                ],
                "total": 1,
            },
        )

    docs = client_with(handler).documents()
    assert len(docs) == 1
    assert docs[0].filename == "a.pdf"
    assert docs[0].num_chunks == 3


def test_run_returns_a_typed_run_with_sources():
    def handler(request):
        assert request.url.path == "/api/runs/7"
        return httpx.Response(
            200,
            json={
                "id": 7,
                "question": "q",
                "mode": "manual",
                "requested_strategy": "traditional",
                "selected_strategy": "traditional",
                "fallback_from": None,
                "answer": "a",
                "latency_ms": 100,
                "retrieval_latency_ms": 40,
                "generation_latency_ms": 60,
                "llm_calls": 1,
                "retrieval_calls": 1,
                "input_tokens": 10,
                "output_tokens": 20,
                "estimated_cost_usd": 0.0,
                "llm_model": None,
                "embedding_model": "hashing",
                "trace": [],
                "created_at": "2026-09-18T00:00:00",
                "sources": [
                    {
                        "rank": 1,
                        "chunk_id": 3,
                        "document_id": 1,
                        "score": 0.9,
                        "cited": True,
                        "page": None,
                    }
                ],
            },
        )

    run = client_with(handler).run(7)
    assert run.id == 7
    assert run.sources[0].chunk_id == 3
    assert run.sources[0].cited is True


def test_ask_sends_the_agentic_strategy_and_parses_the_sub_question_report():
    """The agent's partial answers are structured, so the client types them."""

    def handler(request):
        assert json.loads(request.content)["strategy"] == "agentic"
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "a [1]",
                "confidence": 0.8,
                "citations": [],
                "highlights": [],
                "sub_questions": [
                    {
                        "text": "what is the retry limit",
                        "status": "answered",
                        "reason": None,
                        "chunk_ids": [3],
                    },
                    {
                        "text": "who signs it off",
                        "status": "open",
                        "reason": "no evidence was retrieved for this sub-question",
                        "chunk_ids": [],
                    },
                ],
                "dropped_claims": [{"text": "the board approved it", "reason": "uncited answer"}],
                "dated_sources": [
                    {
                        "sub_question": "what is the retry limit",
                        "sources": [
                            {
                                "marker": 1,
                                "chunk_id": 3,
                                "document_id": 7,
                                "effective_date": "2025-06-01",
                            }
                        ],
                    }
                ],
                "trace": [{"name": "plan", "started_ms": 0, "duration_ms": 2, "attributes": {}}],
            },
        )

    answer = client_with(handler).ask("q", strategy="agentic")

    assert [report.status for report in answer.sub_questions] == ["answered", "open"]
    assert answer.sub_questions[1].reason
    assert answer.dropped_claims[0].reason == "uncited answer"
    assert answer.dated_sources[0].sources[0].effective_date == "2025-06-01"
    assert answer.trace[0]["name"] == "plan"


def test_an_answer_from_a_server_without_the_agent_fields_still_parses():
    """Every field this task adds is optional, so an older server is readable."""

    def handler(request):
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "a [1]",
                "confidence": 0.8,
                "citations": [],
                "highlights": [],
            },
        )

    answer = client_with(handler).ask("q")

    assert answer.sub_questions == []
    assert answer.dropped_claims == []
    assert answer.dated_sources == []
    assert answer.trace == []
    assert answer.subgraph is None
    assert answer.dropped_relationship_claims == []


def test_ask_sends_the_graph_strategy_and_parses_the_subgraph():
    def handler(request):
        assert json.loads(request.content)["strategy"] == "graph"
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "The Platform Team is a member of Engineering [E 1] [1].",
                "confidence": 0.8,
                "citations": [],
                "highlights": [],
                "subgraph": {
                    "nodes": [
                        {"id": 1, "name": "Platform Team", "entity_type": "team", "depth": 0},
                        {"id": 2, "name": "Engineering", "entity_type": "organisation", "depth": 1},
                    ],
                    "edges": [
                        {
                            "id": 5,
                            "source_id": 1,
                            "target_id": 2,
                            "relation_type": "MEMBER_OF",
                            "walked_as": "MEMBER_OF",
                            "reversed": False,
                            "confidence": None,
                            "source_chunk_ids": [9],
                        }
                    ],
                    "truncated": True,
                    "empty_reason": None,
                },
                "dropped_relationship_claims": [
                    {
                        "text": "Engineering reports to the Board [E 99] [1].",
                        "reason": "edge_not_in_subgraph",
                    }
                ],
            },
        )

    answer = client_with(handler).ask("q", strategy="graph")

    assert [node.name for node in answer.subgraph.nodes] == ["Platform Team", "Engineering"]
    assert answer.subgraph.nodes[1].depth == 1
    [edge] = answer.subgraph.edges
    assert (edge.walked_as, edge.reversed, edge.confidence) == ("MEMBER_OF", False, None)
    assert edge.source_chunk_ids == [9]
    assert answer.subgraph.truncated is True
    assert answer.subgraph.empty_reason is None
    assert answer.dropped_relationship_claims[0].reason == "edge_not_in_subgraph"


def test_an_empty_subgraph_reports_why():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "I could not find an answer to that in the documents provided.",
                "confidence": 0.0,
                "citations": [],
                "highlights": [],
                "subgraph": {
                    "nodes": [],
                    "edges": [],
                    "truncated": False,
                    "empty_reason": "no_graph_coverage",
                },
            },
        )

    answer = client_with(handler).ask("q", strategy="graph")

    assert answer.subgraph.empty_reason == "no_graph_coverage"
