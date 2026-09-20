"""Tests for the ragfabric_sdk client, against httpx.MockTransport only.

No test here talks to a network or a running server: every handler below
builds the response by hand, shaped exactly like the real server's routes in
packages/server/src/ragfabric_server/api/routes/ and schemas in
packages/server/src/ragfabric_server/schemas/.
"""

import json

import httpx
import pytest

from ragfabric_sdk import Client
from ragfabric_sdk.errors import AuthError, NotFoundError, RateLimitError


def transport(handler):
    return httpx.MockTransport(handler)


def client_with(handler, **kwargs):
    return Client("http://server", token="t", transport=transport(handler), **kwargs)


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
