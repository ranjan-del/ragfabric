"""Fix wave: a streamed answer reports the generation tokens the provider measured, never a 0.

Phase 9's first real run showed ``Tokens in / out 18 / 0`` for streamed
answers: the stream path added 0 for generation because the stream yielded
text only. Providers now fill a ``StreamUsage`` when they report usage, and
the ``done`` event carries those counts; when a provider reports nothing the
event says ``null`` rather than a number nobody measured (ADR 0004).
"""

from __future__ import annotations

import json

from ragfabric_core.providers.base import StreamUsage
from ragfabric_server.deps import get_llm_provider
from ragfabric_server.main import app


def _done(client, token) -> dict:
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "how much annual leave", "top_k": 3, "stream": True},
        headers={"Authorization": f"Bearer {token}"},
    ) as res:
        raw = "".join(res.iter_text())
    for block in raw.strip().split("\n\n"):
        lines = block.splitlines()
        if lines and lines[0] == "event: done":
            return json.loads(lines[1].split(":", 1)[1])
    raise AssertionError("no done event")


def _reporting(fake_cls):
    """The suite's citing fake, plus the usage a real provider reports at stream end."""

    class Reporting(fake_cls):
        def stream(self, messages, *, model=None, max_tokens=1024, temperature=0.0, usage=None):
            yield from super().stream(messages, model=model, max_tokens=max_tokens)
            if usage is not None:
                usage.input_tokens, usage.output_tokens = 321, 17

    return Reporting


def test_streamed_generation_tokens_come_from_the_provider(client, admin_token, ingested_doc):
    fake_cls = type(app.dependency_overrides[get_llm_provider]())
    app.dependency_overrides[get_llm_provider] = lambda: _reporting(fake_cls)()
    done = _done(client, admin_token)
    assert done["usage"]["output_tokens"] == 17
    assert done["usage"]["input_tokens"] >= 321

    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.runs import RetrievalRun

    with SessionLocal() as db:
        run = db.get(RetrievalRun, done["run_id"])
        assert run.output_tokens == 17


def test_a_provider_that_reports_no_usage_gives_null_not_zero(client, admin_token, ingested_doc):
    done = _done(client, admin_token)  # the default fake reports nothing
    assert done["usage"]["output_tokens"] is None
    assert done["usage"]["input_tokens"] is None


def test_stream_usage_starts_unknown():
    u = StreamUsage()
    assert (u.input_tokens, u.output_tokens) == (None, None)
