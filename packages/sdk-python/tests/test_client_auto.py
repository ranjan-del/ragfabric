"""The SDK leaves the strategy to the server and parses the router's decision."""

import json

import httpx

from ragfabric_sdk import Client

_BASE = {
    "question": "q",
    "answer": "a [1]",
    "confidence": 0.9,
    "citations": [],
    "highlights": [],
}
_ROUTER = {
    "selected_strategy": "graph",
    "source": "signals",
    "decisive": True,
    "confidence": None,
    "reasoning": "The question asks how named things are related.",
    "query_type": "relational",
    "estimated_complexity": "medium",
    "expected_cost_level": "low",
    "expected_latency_level": "low",
    "fused": False,
    "a_field_from_a_newer_server": 1,
}


def _client(captured: list[dict], payload: dict) -> Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    return Client("http://server", token="t", transport=httpx.MockTransport(handler))


def test_ask_omits_strategy_when_unset():
    captured: list[dict] = []
    _client(captured, _BASE).ask("q")
    assert "strategy" not in captured[0]


def test_ask_sends_strategy_when_given():
    captured: list[dict] = []
    _client(captured, _BASE).ask("q", strategy="auto")
    assert captured[0]["strategy"] == "auto"


def test_search_omits_strategy_when_unset():
    captured: list[dict] = []
    _client(captured, {"results": []}).search("q")
    assert "strategy" not in captured[0]


def test_ask_stream_omits_strategy_when_unset():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, text='event: done\ndata: {"run_id": 1}\n\n')

    client = Client("http://server", token="t", transport=httpx.MockTransport(handler))
    assert [e.event for e in client.ask_stream("q")] == ["done"]
    assert "strategy" not in captured[0]


def test_answer_parses_router_and_tolerates_an_old_server():
    new = _client(
        [], {**_BASE, "strategy": "vectorless", "router": _ROUTER, "fallback_from": "graph"}
    ).ask("q")
    assert new.strategy == "vectorless"
    assert new.fallback_from == "graph"
    assert new.router is not None
    assert new.router.source == "signals"
    assert new.router.confidence is None
    assert new.router.reasoning.startswith("The question")

    old = _client([], _BASE).ask("q")
    assert old.router is None
    assert old.strategy is None
    assert old.fallback_from is None
