"""The end-to-end scenario both E2E tiers run (design D26, D27).

One user story through every layer: sign up, upload, list, ask in ``auto``
mode, stream the same question, fetch the run, download a cited document, and
hit a rate limit. Everything goes through the published Python SDK where the
SDK has a method for it, over the app in process (no socket), so the test proves
the client a user installs as well as the server. Registration, sign-in and
download have no SDK method yet, so they use a plain ``httpx`` client over the
same transport.

``RecordingTransport`` sits between both clients and the app and keeps every
response's ``X-Request-ID``, so "every response carries a request id" is
checked on all of them, not on a sample.

Not a test module itself (no ``test_`` prefix): ``test_e2e_flow.py`` runs it on
SQLite and ``test_e2e_postgres.py`` on PostgreSQL with pgvector and full text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from ragfabric_core import runtime
from ragfabric_core.testing.fixtures import make_docx, make_txt
from ragfabric_sdk import Client
from ragfabric_sdk.errors import RagFabricError, RateLimitError

BASE_URL = "http://testserver"
EMAIL = "e2e@example.com"
PASSWORD = "e2e-password-123"
QUESTION = "how much annual leave"
LEAVE_TEXT = (
    "Annual leave policy. Every employee is entitled to twenty five days of annual leave "
    "per year, in addition to public holidays. Unused annual leave carries over for three months."
)
EXPENSES_PARAGRAPHS = [
    "Travel expenses policy.",
    "Economy class is the default for flights under six hours.",
    "Receipts are required for every claim over fifty euros.",
]


class RecordingTransport(httpx.BaseTransport):
    """An ``httpx`` transport that hands each request to the app and records its request id.

    Starlette's ``TestClient`` is built on its own copy of the HTTP client
    (``httpx2``), so its internal transport cannot be handed to the SDK's
    ``httpx.Client`` directly. This bridges the two: the SDK builds a real
    ``httpx`` request, the ``TestClient`` runs it through the ASGI app
    (middleware, lifespan state and all), and the response is rebuilt as an
    ``httpx`` response. A streamed response arrives whole, which is how the
    ``TestClient`` delivers it anyway; the SDK still parses it as a stream.
    """

    SKIP_HEADERS = {"host", "content-length", "transfer-encoding"}

    def __init__(self, test_client) -> None:
        self.test_client = test_client
        self.responses: list[tuple[str, str, int, str | None]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.read()
        headers = [(k, v) for k, v in request.headers.multi_items() if k not in self.SKIP_HEADERS]
        result = self.test_client.request(
            request.method, str(request.url), headers=headers, content=body
        )
        self.responses.append(
            (
                request.method,
                request.url.path,
                result.status_code,
                result.headers.get("x-request-id"),
            )
        )
        return httpx.Response(
            result.status_code,
            headers=[
                (k, v)
                for k, v in result.headers.multi_items()
                if k not in {"content-length", "content-encoding", "transfer-encoding"}
            ],
            content=result.content,
            request=request,
        )


@dataclass
class Outcome:
    """What the scenario saw, for a tier to make extra assertions on."""

    document_ids: list[int] = field(default_factory=list)
    strategy: str | None = None
    run_id: int | None = None


def _wait_ready(sdk: Client, document_id: int) -> dict:
    import time

    deadline = time.monotonic() + 30
    while True:
        [doc] = [d for d in sdk.documents() if d.id == document_id]
        if doc.status != "processing" or time.monotonic() > deadline:
            return doc.model_dump()
        time.sleep(0.05)


def run_scenario(test_client, tmp_path: Path) -> Outcome:
    transport = RecordingTransport(test_client)
    outcome = Outcome()
    raw = httpx.Client(base_url=BASE_URL, transport=transport)

    # 1. Register and sign in: the token works.
    res = raw.post("/api/auth/register", json={"email": EMAIL, "password": PASSWORD})
    assert res.status_code in (200, 201), res.text
    res = raw.post("/api/auth/login", data={"username": EMAIL, "password": PASSWORD})
    assert res.status_code == 200, res.text
    token = res.json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}
    assert raw.get("/api/auth/me", headers=auth).json()["email"] == EMAIL

    sdk = Client(BASE_URL, token=token, transport=transport)

    # 2. Upload a TXT and a DOCX built here: 201, ready, chunked.
    originals: dict[str, bytes] = {
        "leave-policy.txt": make_txt(LEAVE_TEXT),
        "expenses.docx": make_docx(EXPENSES_PARAGRAPHS),
    }
    for name, data in originals.items():
        path = tmp_path / name
        path.write_bytes(data)
        uploaded = sdk.ingest(path)
        doc = _wait_ready(sdk, uploaded.id)
        assert doc["status"] == "ready", doc
        assert doc["num_chunks"] > 0, doc
        outcome.document_ids.append(uploaded.id)
    assert {d.filename for d in sdk.documents()} >= set(originals)

    # 3. A binary renamed to .pdf is refused for what it is: 415.
    fake = tmp_path / "invoice.pdf"
    fake.write_bytes(b"\x89PNG\r\n\x1a\n not a pdf at all")
    with pytest.raises(RagFabricError) as refused:
        sdk.ingest(fake)
    assert refused.value.status_code == 415

    # 4. Ask in auto mode: the router decides by signals, a strategy runs, the
    #    answer is non-empty and every marker in it maps to a returned citation.
    answer = sdk.ask(QUESTION, strategy="auto")
    assert answer.answer.strip()
    assert answer.strategy in {"traditional", "vectorless", "agentic", "graph"}
    assert answer.router is not None and answer.router.source == "signals"
    markers = set(re.findall(r"\[\d+\]", answer.answer))
    assert markers, answer.answer
    cited = {c.marker for c in answer.citations}
    assert markers <= cited, (markers, cited)
    leave_id = outcome.document_ids[0]
    assert any(c.document_id == leave_id for c in answer.citations)
    outcome.strategy = answer.strategy

    # 5. The same question streamed: tokens, then done with a retrievable run.
    events = list(sdk.ask_stream(QUESTION, strategy="auto"))
    names = [e.event for e in events]
    assert "token" in names and names[-1] == "done", names
    run_id = events[-1].data["run_id"]
    run = sdk.run(run_id)
    assert run.id == run_id and run.question == QUESTION
    assert run.requested_strategy == "auto"
    outcome.run_id = run_id

    # 6. A cited document downloads byte for byte.
    cited_ids = {c.document_id for c in answer.citations if c.document_id is not None}
    for document_id in cited_ids:
        res = raw.get(f"/api/documents/{document_id}/download", headers=auth)
        assert res.status_code == 200
        [name] = [
            n for n, i in zip(originals, outcome.document_ids, strict=True) if i == document_id
        ]
        assert res.content == originals[name]

    # 7. Every response carried a request id.
    missing = [r for r in transport.responses if not r[3]]
    assert not missing, missing
    assert len({r[3] for r in transport.responses}) == len(transport.responses)

    # 8. A tiny per-user limit: 429 with Retry-After.
    limits = runtime.get_config().limits
    saved = limits.user_rate_limit_per_minute
    limits.user_rate_limit_per_minute = 1
    try:
        with pytest.raises(RateLimitError):
            for _ in range(3):
                sdk.documents()
        res = raw.get("/api/documents", headers=auth)
        assert res.status_code == 429
        assert 1 <= int(res.headers["retry-after"]) <= 60
        assert res.headers["x-request-id"]
    finally:
        limits.user_rate_limit_per_minute = saved

    sdk.close()
    raw.close()
    return outcome
