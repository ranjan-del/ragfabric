"""``ragfabric ask`` on a terminal (rich) and piped (plain, byte identical)."""

from __future__ import annotations

import json

import httpx
from typer.testing import CliRunner

from ragfabric_cli.main import app
from ragfabric_cli.ui import console

runner = CliRunner()

_SNIPPET = "Employees may carry forward ten days of unused leave into the next year."
_ANSWER = {
    "question": "q",
    "answer": "Ten days carry forward [1].",
    "confidence": 0.9,
    "citations": [
        {
            "marker": "[1]",
            "document_id": 7,
            "filename": "leave.pdf",
            "page": 2,
            "snippet": _SNIPPET,
            "used": True,
        },
        {"marker": "[2]", "filename": "unused.pdf", "snippet": "x", "used": False},
    ],
    "highlights": [],
    "strategy": "graph",
    "router": {"selected_strategy": "graph", "source": "signals", "reasoning": "Why."},
    "subgraph": {
        "nodes": [
            {"id": 1, "name": "Ravi Sharma", "entity_type": "person", "depth": 0},
            {"id": 2, "name": "Asha Rao", "entity_type": "person", "depth": 1},
        ],
        "edges": [
            {
                "id": 9,
                "source_id": 1,
                "target_id": 2,
                "relation_type": "REPORTS_TO",
                "walked_as": "REPORTS_TO",
                "reversed": False,
            }
        ],
    },
    "dropped_claims": [{"text": "Ravi founded the firm.", "reason": "unsupported"}],
}

_PLAIN = (
    "Ten days carry forward [1].\n\n"
    "sources:\n  [1] leave.pdf p2\n"
    "graph:\n  [E 1] Ravi Sharma REPORTS_TO Asha Rao\n"
    "Strategy: graph (signals). Why.\n"
)


def _patched(monkeypatch, payload: dict) -> None:
    import ragfabric_sdk.client as sdk_client

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    real_client = sdk_client.httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client.httpx, "Client", fake_client)


def _rich(monkeypatch) -> None:
    monkeypatch.setattr(console, "is_rich", lambda stream=None: True)
    monkeypatch.setenv("COLUMNS", "100")
    console.get_console.cache_clear()


def test_rich_answer_has_panel_sources_strategy_and_graph(monkeypatch):
    _patched(monkeypatch, _ANSWER)
    _rich(monkeypatch)
    try:
        result = runner.invoke(app, ["ask", "q", "--token", "t", "--no-stream"])
    finally:
        console.get_console.cache_clear()
    out = result.output
    assert result.exit_code == 0, out
    for needle in ("Answer", "Sources", "leave.pdf", "Strategy", "─REPORTS_TO→"):
        assert needle in out, needle
    assert "Ravi Sharma ─REPORTS_TO→ Asha Rao" in out
    assert "Removed (unsupported)" in out and "Ravi founded the firm." in out
    assert "unused.pdf" not in out
    assert _SNIPPET[:60].rstrip() in out and _SNIPPET not in out


def test_rich_sources_work_without_a_snippet(monkeypatch):
    payload = {**_ANSWER, "citations": [{**_ANSWER["citations"][0], "snippet": ""}]}
    _patched(monkeypatch, payload)
    _rich(monkeypatch)
    try:
        result = runner.invoke(app, ["ask", "q", "--token", "t", "--no-stream"])
    finally:
        console.get_console.cache_clear()
    assert result.exit_code == 0, result.output
    assert "leave.pdf" in result.output


def test_plain_output_is_byte_identical(monkeypatch):
    _patched(monkeypatch, _ANSWER)
    result = runner.invoke(app, ["ask", "q", "--token", "t", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert result.stdout == _PLAIN


def test_json_output_equals_the_sdk_model_dump(monkeypatch):
    from ragfabric_sdk.models import Answer

    _patched(monkeypatch, _ANSWER)
    _rich(monkeypatch)
    try:
        result = runner.invoke(app, ["ask", "q", "--token", "t", "--json"])
    finally:
        console.get_console.cache_clear()
    assert result.exit_code == 0, result.output
    expected = json.loads(Answer.model_validate(_ANSWER).model_dump_json())
    assert json.loads(result.stdout) == expected


def test_ask_with_no_server_is_friendly_and_has_no_traceback():
    result = runner.invoke(
        app, ["ask", "q", "--token", "t", "--url", "http://127.0.0.1:1", "--no-stream"]
    )
    assert result.exit_code == 1
    assert "No RagFabric server at" in result.output
    assert "ragfabric serve" in result.output
    assert "Traceback" not in result.output
