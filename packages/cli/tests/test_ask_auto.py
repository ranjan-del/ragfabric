"""``ragfabric ask`` with the router: auto is a choice, the default sends nothing."""

from __future__ import annotations

import json

import httpx
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()

_REASON = "The question asks how named things are related."
_ROUTER = {"selected_strategy": "graph", "source": "signals", "reasoning": _REASON}
_ANSWER = {
    "question": "q",
    "answer": "an answer [1]",
    "confidence": 0.9,
    "citations": [],
    "highlights": [],
    "strategy": "graph",
    "router": _ROUTER,
}


def _patched(monkeypatch, captured: list[dict], payload: dict, stream: str | None = None) -> None:
    import ragfabric_sdk.client as sdk_client

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if body.get("stream"):
            return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=payload)

    real_client = sdk_client.httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client.httpx, "Client", fake_client)


def test_auto_is_a_choice(monkeypatch):
    captured: list[dict] = []
    _patched(monkeypatch, captured, _ANSWER)
    result = runner.invoke(app, ["ask", "q", "--token", "t", "--strategy", "auto", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert captured[0]["strategy"] == "auto"


def test_the_default_sends_no_strategy(monkeypatch):
    captured: list[dict] = []
    _patched(monkeypatch, captured, _ANSWER)
    result = runner.invoke(app, ["ask", "q", "--token", "t", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert "strategy" not in captured[0]


def test_one_line_names_the_strategy_and_why(monkeypatch):
    _patched(monkeypatch, [], _ANSWER)
    result = runner.invoke(app, ["ask", "q", "--token", "t", "--no-stream"])
    assert f"Strategy: graph (signals). {_REASON}" in result.stdout
    assert "Fell back" not in result.stdout


def test_the_line_says_when_it_fell_back(monkeypatch):
    _patched(monkeypatch, [], {**_ANSWER, "fallback_from": "vectorless"})
    result = runner.invoke(app, ["ask", "q", "--token", "t", "--no-stream"])
    assert "Fell back from vectorless, which found nothing." in result.stdout


def test_no_line_when_the_server_sends_no_router(monkeypatch):
    plain = {k: v for k, v in _ANSWER.items() if k not in ("router", "strategy")}
    _patched(monkeypatch, [], plain)
    result = runner.invoke(app, ["ask", "q", "--token", "t", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert "Strategy:" not in result.stdout


def test_the_streaming_path_prints_the_line_from_the_retrieval_event(monkeypatch):
    retrieval = json.dumps({"router": _ROUTER, "fallback_from": "vectorless"})
    stream = (
        f"event: retrieval\ndata: {retrieval}\n\n"
        'event: token\ndata: {"text": "hi"}\n\n'
        'event: done\ndata: {"run_id": 3, "latency_ms": 5}\n\n'
    )
    _patched(monkeypatch, [], _ANSWER, stream=stream)
    result = runner.invoke(app, ["ask", "q", "--token", "t"])
    assert result.exit_code == 0, result.output
    assert f"Strategy: graph (signals). {_REASON} Fell back from vectorless" in result.stdout
