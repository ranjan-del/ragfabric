"""POST /api/ask streams retrieval, tokens, citations and a run id, in that order."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ragfabric_core.db.session import SessionLocal
from ragfabric_core.models.index import ChunkEmbedding
from ragfabric_core.testing.fixtures import make_txt


def _events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.strip().split("\n\n"):
        name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        if name:
            out.append((name, data))
    return out


@pytest.fixture()
def restricted_token(client: TestClient) -> str:
    """Bare bearer token for an ordinary user with no group memberships."""
    client.post(
        "/api/auth/register", json={"email": "restricted@example.com", "password": "password123"}
    )
    resp = client.post(
        "/api/auth/login",
        data={"username": "restricted@example.com", "password": "password123"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture()
def restricted_setup(client: TestClient, admin_token: str, restricted_token: str) -> dict:
    """A collection the restricted user cannot read, plus proof it is not vacuous.

    Follows test_access_enforcement.py's pattern: the forbidden document is
    proven to be genuinely indexed and genuinely retrievable, by an
    unrestricted principal, for the exact query the restricted test below
    uses, before the collection is locked down. A second, allowed document is
    also seeded so the restricted principal's request can produce a real,
    non-empty answer, which means a leaked citation cannot hide behind an
    empty response and enforcement cannot hide behind a corpus that never
    matched anything.
    """
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    restricted_headers = {"Authorization": f"Bearer {restricted_token}"}

    payroll = client.post(
        "/api/collections", json={"name": "payroll-ask"}, headers=admin_headers
    ).json()
    forbidden_text = (
        "Confidential salary band memo: executive base pay ranges from two hundred "
        "thousand to four hundred thousand dollars per year, reviewed each quarter."
    )
    forbidden = client.post(
        "/api/documents/upload",
        files={"file": ("payroll-bands.txt", make_txt(forbidden_text), "text/plain")},
        data={"collection_id": str(payroll["id"])},
        headers=admin_headers,
    ).json()
    assert forbidden["status"] == "ready"
    forbidden_id = forbidden["id"]

    query_text = "confidential salary band pay ranges"

    # Proof #1: real chunk_embeddings rows exist for the forbidden document.
    with SessionLocal() as db:
        embedding_rows = (
            db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == forbidden_id).all()
        )
    assert embedding_rows, (
        "forbidden document has no chunk_embeddings rows; the access-filter "
        "property below would hold vacuously"
    )

    # Proof #2: an unrestricted principal genuinely retrieves it for this
    # exact query, through the same endpoint the restricted test below uses.
    admin_answer = client.post(
        "/api/ask",
        json={"query": query_text, "top_k": 10, "stream": False},
        headers=admin_headers,
    ).json()
    assert any(c["document_id"] == forbidden_id for c in admin_answer["citations"]), (
        "admin (unrestricted) did not retrieve the forbidden document for this "
        "query; the property below would hold vacuously"
    )

    # Restrict the collection to a group the restricted user is not in.
    group = client.post(
        "/api/admin/groups", json={"name": "payroll-ask-only"}, headers=admin_headers
    ).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": payroll["id"], "permission": "read"},
        headers=admin_headers,
    )

    # Something else real for the restricted user to find with the same
    # query, uploaded outside any collection (open by v1's "no collection"
    # rule), so a genuine non-empty answer is possible without ever touching
    # the forbidden document.
    allowed_text = (
        "Public salary band overview: entry level roles start near fifty thousand "
        "dollars, and pay bands are published on the intranet every year."
    )
    allowed = client.post(
        "/api/documents/upload",
        files={"file": ("public-bands.txt", make_txt(allowed_text), "text/plain")},
        headers=restricted_headers,
    ).json()
    assert allowed["status"] == "ready"

    return {"forbidden_document_id": forbidden_id, "query": query_text}


def test_ask_streams_events_in_the_documented_order(client, admin_token, ingested_doc):
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "how much annual leave", "top_k": 3, "stream": True},
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        events = _events("".join(res.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "retrieval"
    assert "token" in names
    assert names[-2] == "citations"
    assert names[-1] == "done"
    assert names.index("citations") > max(i for i, n in enumerate(names) if n == "token"), (
        "citations must arrive after the last token, because used is only known then"
    )


def test_the_done_event_carries_a_run_id_that_resolves(client, admin_token, ingested_doc):
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "leave", "stream": True},
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as res:
        events = _events("".join(res.iter_text()))
    run_id = dict(events)["done"]["run_id"]

    res = client.get(f"/api/runs/{run_id}", headers={"Authorization": f"Bearer {admin_token}"})
    assert res.status_code == 200
    assert res.json()["sources"]


def test_ask_records_no_cost_estimate_on_the_run(client, admin_token, ingested_doc):
    """``estimated_cost_usd`` must stay ``None``, not a fabricated ``0.0``.

    Mirrors the same fix in ``/api/search/query``: no price table is wired
    in yet, so the honest value is "not computed", not "free".
    """
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "leave", "stream": True},
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as res:
        events = _events("".join(res.iter_text()))
    run_id = dict(events)["done"]["run_id"]

    run_res = client.get(f"/api/runs/{run_id}", headers={"Authorization": f"Bearer {admin_token}"})
    assert run_res.status_code == 200
    assert run_res.json()["estimated_cost_usd"] is None


def test_ask_without_streaming_returns_the_same_shape_as_query(client, admin_token, ingested_doc):
    res = client.post(
        "/api/ask",
        json={"query": "leave", "stream": False},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 200
    body = res.json()
    assert set(body) >= {"question", "answer", "confidence", "citations", "highlights"}


def test_ask_applies_the_access_filter(client, restricted_token, restricted_setup):
    res = client.post(
        "/api/ask",
        json={"query": restricted_setup["query"], "stream": False},
        headers={"Authorization": f"Bearer {restricted_token}"},
    )
    assert res.status_code == 200
    body = res.json()
    forbidden = restricted_setup["forbidden_document_id"]
    assert all(c["document_id"] != forbidden for c in body["citations"])
    # Not vacuous: the restricted user still got a real, non-empty answer
    # (from the allowed document seeded alongside the forbidden one).
    assert body["citations"], "restricted user's ask returned no citations at all"
    assert "could not find" not in body["answer"]
