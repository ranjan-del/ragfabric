"""POST /api/documents/{id}/move: relational tables update, Chroma cannot be
made safe by this endpoint, so a chroma-backed deployment refuses the move.
"""

from __future__ import annotations

import ragfabric_server.api.routes.documents as documents_routes


class _StubVectorStoreConfig:
    def __init__(self, kind: str) -> None:
        self.kind = kind


class _StubConfig:
    def __init__(self, kind: str) -> None:
        self.vector_store = _StubVectorStoreConfig(kind)


def test_move_updates_the_relational_tables_on_pgvector(client, admin_token, ingested_doc):
    headers = {"Authorization": f"Bearer {admin_token}"}
    target = client.post("/api/collections", json={"name": "move-target"}, headers=headers).json()

    res = client.post(
        f"/api/documents/{ingested_doc['id']}/move",
        json={"collection_id": target["id"]},
        headers=headers,
    )
    assert res.status_code == 200, res.text
    assert res.json()["collection_id"] == target["id"]


def test_move_is_refused_with_409_when_vector_store_is_chroma(
    client, admin_token, ingested_doc, monkeypatch
):
    """A chroma-backed deployment cannot keep Chroma's own metadata copy of
    collection_id in sync inside this transaction, so the move must fail
    closed rather than silently leave the access filter and the index
    disagreeing. No live Chroma is needed to prove this: only the configured
    vector store kind matters.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}
    target = client.post(
        "/api/collections", json={"name": "move-target-chroma"}, headers=headers
    ).json()

    monkeypatch.setattr(documents_routes, "get_config", lambda: _StubConfig("chroma"))

    res = client.post(
        f"/api/documents/{ingested_doc['id']}/move",
        json={"collection_id": target["id"]},
        headers=headers,
    )
    assert res.status_code == 409, res.text
    assert "chroma" in res.json()["detail"].lower()
