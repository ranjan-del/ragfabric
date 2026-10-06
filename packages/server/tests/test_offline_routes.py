"""The offline LLM provider over HTTP: extractive answers, no phantom call, agentic and graph
refused."""

from __future__ import annotations

import json

import pytest

from ragfabric_core.providers.offline import OfflineLLMProvider
from ragfabric_server.deps import get_llm_provider

QUESTION = "how many days of annual leave"
AGENTIC_REFUSAL = (
    "the agentic strategy needs a model (offline mode); run ragfabric doctor, then follow its "
    "upgrade steps: install Ollama, or set OPENAI_API_KEY or ANTHROPIC_API_KEY, then "
    "mv ragfabric.yaml ragfabric.yaml.bak && ragfabric quickstart && ragfabric reindex --yes"
)

GRAPH_REFUSAL = (
    "the graph strategy needs a model (offline mode); run ragfabric doctor, then follow its "
    "upgrade steps: install Ollama, or set OPENAI_API_KEY or ANTHROPIC_API_KEY, then "
    "mv ragfabric.yaml ragfabric.yaml.bak && ragfabric quickstart && ragfabric reindex --yes"
)


@pytest.fixture()
def offline(client):
    llm = OfflineLLMProvider()
    client.app.dependency_overrides[get_llm_provider] = lambda: llm
    yield client
    client.app.dependency_overrides.pop(get_llm_provider, None)


def _events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.strip().split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        if name:
            out.append((name, data))
    return out


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _stream(client, token, **extra):
    body = {"query": QUESTION, "top_k": 3, "stream": True, **extra}
    with client.stream("POST", "/api/ask", json=body, headers=auth(token)) as res:
        status = res.status_code
        raw = "".join(res.iter_text())
    return status, raw


def test_streamed_ask_answers_extractively_with_one_token_and_no_superseded(
    offline, admin_token, ingested_doc
):
    status, raw = _stream(offline, admin_token)
    assert status == 200
    events = _events(raw)
    names = [n for n, _ in events]
    assert "superseded" not in names
    assert names.count("token") == 1
    text = next(d["text"] for n, d in events if n == "token")
    assert "twenty five days of annual leave" in text and "[1]" in text
    done = dict(events)["done"]
    assert done["usage"]["llm_calls"] == 0
    run = offline.get(f"/api/runs/{done['run_id']}", headers=auth(admin_token)).json()
    assert run["llm_calls"] == 0


def test_non_streamed_ask_answers_extractively_without_a_phantom_call(
    offline, admin_token, ingested_doc
):
    r = offline.post(
        "/api/ask",
        json={"query": QUESTION, "top_k": 3, "stream": False},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "twenty five days of annual leave" in body["answer"]
    assert body["usage"]["llm_calls"] == 0


@pytest.mark.parametrize("stream", [True, False])
def test_explicit_agentic_ask_is_a_422_before_any_streaming(offline, admin_token, stream):
    r = offline.post(
        "/api/ask",
        json={"query": QUESTION, "strategy": "agentic", "stream": stream},
        headers=auth(admin_token),
    )
    assert r.status_code == 422
    assert r.json()["detail"] == AGENTIC_REFUSAL
    assert r.headers["content-type"].startswith("application/json")


def test_explicit_agentic_query_is_a_422(offline, admin_token):
    r = offline.post(
        "/api/search/query",
        json={"query": QUESTION, "strategy": "agentic"},
        headers=auth(admin_token),
    )
    assert r.status_code == 422
    assert r.json()["detail"] == AGENTIC_REFUSAL


# issue #55: offline, nothing extracts a graph, so an explicit graph request is refused


@pytest.mark.parametrize("stream", [True, False])
def test_explicit_graph_ask_is_a_422_before_any_streaming(offline, admin_token, stream):
    r = offline.post(
        "/api/ask",
        json={"query": QUESTION, "strategy": "graph", "stream": stream},
        headers=auth(admin_token),
    )
    assert r.status_code == 422
    assert r.json()["detail"] == GRAPH_REFUSAL
    assert r.headers["content-type"].startswith("application/json")


def test_explicit_graph_query_is_a_422(offline, admin_token):
    r = offline.post(
        "/api/search/query",
        json={"query": QUESTION, "strategy": "graph"},
        headers=auth(admin_token),
    )
    assert r.status_code == 422
    assert r.json()["detail"] == GRAPH_REFUSAL
