"""`ragfabric ask --strategy agentic` reaches the server as a strategy choice.

The CLI talks to a running server through the SDK, so what is asserted here is
the request body the SDK puts on the wire, not a local retrieval.
"""

from __future__ import annotations

import json

import httpx
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()

_ANSWER = {
    "question": "q",
    "answer": "an answer [1]",
    "confidence": 0.9,
    "citations": [],
    "highlights": [],
    "source_document": None,
    "usage": {
        "embedding_calls": 0,
        "llm_calls": 3,
        "retrieval_calls": 2,
        "input_tokens": 0,
        "output_tokens": 0,
    },
    "sub_questions": [
        {"text": "how much leave", "status": "answered", "reason": None, "chunk_ids": [1]}
    ],
    "dropped_claims": [],
    "dated_sources": [],
    "trace": [],
}


def _transport(captured: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_ANSWER)

    return httpx.MockTransport(handler)


def _patched(monkeypatch, captured: list[dict]) -> None:
    import ragfabric_sdk.client as sdk_client

    real_client = sdk_client.httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = _transport(captured)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client.httpx, "Client", fake_client)


def test_ask_sends_the_agentic_strategy(monkeypatch):
    captured: list[dict] = []
    _patched(monkeypatch, captured)

    result = runner.invoke(
        app, ["ask", "how much leave", "--token", "t", "--strategy", "agentic", "--no-stream"]
    )

    assert result.exit_code == 0, result.output
    assert captured and captured[0]["strategy"] == "agentic"


def test_an_unknown_strategy_is_refused_before_a_request_is_made(monkeypatch):
    """Typer rejects it while parsing, so no call leaves the machine."""
    captured: list[dict] = []
    _patched(monkeypatch, captured)

    result = runner.invoke(
        app, ["ask", "how much leave", "--token", "t", "--strategy", "agentik", "--no-stream"]
    )

    assert result.exit_code != 0
    assert captured == []


def test_the_json_output_carries_the_sub_question_report(monkeypatch):
    captured: list[dict] = []
    _patched(monkeypatch, captured)

    result = runner.invoke(
        app,
        ["ask", "how much leave", "--token", "t", "--strategy", "agentic", "--json"],
    )

    assert result.exit_code == 0, result.output
    printed = json.loads(result.stdout)
    assert printed["sub_questions"][0]["status"] == "answered"
