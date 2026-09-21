"""`ragfabric ask --strategy vectorless` reaches the server as a strategy choice.

The CLI talks to a running server through the SDK, so what is asserted here is
the request body the SDK puts on the wire, not a local retrieval.
"""

from __future__ import annotations

import json

import httpx
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()


def _transport(captured: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "an answer [1]",
                "confidence": 0.9,
                "citations": [],
                "highlights": [],
                "source_document": None,
                "usage": {
                    "embedding_calls": 0,
                    "llm_calls": 0,
                    "retrieval_calls": 2,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        )

    return httpx.MockTransport(handler)


def test_ask_sends_the_requested_strategy(monkeypatch):
    captured: list[dict] = []
    import ragfabric_sdk.client as sdk_client

    real_client = sdk_client.httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = _transport(captured)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client.httpx, "Client", fake_client)
    result = runner.invoke(
        app,
        ["ask", "how much leave", "--token", "t", "--strategy", "vectorless", "--no-stream"],
    )
    assert result.exit_code == 0, result.output
    assert captured and captured[0]["strategy"] == "vectorless"


def test_ask_defaults_to_traditional(monkeypatch):
    captured: list[dict] = []
    import ragfabric_sdk.client as sdk_client

    real_client = sdk_client.httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = _transport(captured)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client.httpx, "Client", fake_client)
    result = runner.invoke(app, ["ask", "how much leave", "--token", "t", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert captured and captured[0]["strategy"] == "traditional"


def test_an_unknown_strategy_is_refused_by_the_cli_before_any_request(monkeypatch):
    captured: list[dict] = []
    import ragfabric_sdk.client as sdk_client

    real_client = sdk_client.httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = _transport(captured)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client.httpx, "Client", fake_client)
    result = runner.invoke(app, ["ask", "q", "--token", "t", "--strategy", "nope", "--no-stream"])
    assert result.exit_code != 0
    assert captured == []
