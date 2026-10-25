"""A streamed answer of citation markers alone ("[1]") is superseded, never recorded as is.

Found on the quickstart's own sample question with llama3.2:3b: the model replied
"[1]", the contract accepted it, and the user saw "[1]" as the answer.
"""

from __future__ import annotations

import json

from ragfabric_server.deps import get_llm_provider
from ragfabric_server.main import app


def _events(client, token, stream=True):
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "how much annual leave", "top_k": 3, "stream": stream},
        headers={"Authorization": f"Bearer {token}"},
    ) as res:
        raw = "".join(res.iter_text())
    if not stream:
        return json.loads(raw)
    out = []
    for block in raw.strip().split("\n\n"):
        lines = block.splitlines()
        if lines and lines[0].startswith("event:"):
            out.append((lines[0].split(":", 1)[1].strip(), json.loads(lines[1].split(":", 1)[1])))
    return out


def _marker_first(fake_cls):
    """The suite's citing fake, except that its first reply, streamed or not, is "[1]"."""

    class MarkerFirst(fake_cls):
        calls = 0

        def complete(self, messages, **kwargs):
            done = super().complete(messages, **kwargs)
            MarkerFirst.calls += 1
            if MarkerFirst.calls == 1:
                return done.model_copy(update={"text": "[1]"})
            return done

        def stream(self, messages, *, model=None, max_tokens=1024, temperature=0.0, usage=None):
            yield from ("[", "1", "]")
            MarkerFirst.calls += 1

    return MarkerFirst


def test_a_streamed_answer_of_markers_alone_is_superseded(client, admin_token, ingested_doc):
    fake_cls = type(app.dependency_overrides[get_llm_provider]())
    app.dependency_overrides[get_llm_provider] = lambda: _marker_first(fake_cls)()
    events = _events(client, admin_token)
    names = [name for name, _ in events]
    assert "superseded" in names
    superseded = dict(events)["superseded"]
    assert superseded["text"].strip() != "[1]"
    assert superseded["reason"] == "citation contract"


def test_a_json_answer_of_markers_alone_is_not_returned(client, admin_token, ingested_doc):
    fake_cls = type(app.dependency_overrides[get_llm_provider]())
    app.dependency_overrides[get_llm_provider] = lambda: _marker_first(fake_cls)()
    body = _events(client, admin_token, stream=False)
    assert body["answer"].strip() != "[1]"
