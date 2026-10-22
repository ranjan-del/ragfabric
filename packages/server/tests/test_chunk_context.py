"""GET /api/chunks/{id}/context: the cited chunk with its neighbours, under the read predicate."""

from ragfabric_core.db.session import SessionLocal
from ragfabric_core.models.document import Chunk
from ragfabric_core.testing.fixtures import make_txt

SENTENCES = [f"Sentence number {i} says something distinct about topic {i}. " for i in range(40)]


def _upload(client, headers, name, text, collection_id=None):
    data = {"collection_id": str(collection_id)} if collection_id is not None else {}
    r = client.post(
        "/api/documents/upload",
        files={"file": (name, make_txt(text), "text/plain")},
        data={**data, "chunk_size": "120", "chunk_overlap": "20"},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


def _chunks(document_id):
    with SessionLocal() as db:
        rows = db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        return [(c.id, c.chunk_index, c.text) for c in rows]


def test_a_middle_chunk_comes_back_with_a_neighbour_on_each_side(client, admin_headers):
    doc = _upload(client, admin_headers, "long.txt", "".join(SENTENCES))
    rows = _chunks(doc["id"])
    assert len(rows) >= 5
    cid, index, text = rows[2]

    res = client.get(f"/api/chunks/{cid}/context", headers=admin_headers)

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["chunk"]["text"] == text and body["chunk"]["chunk_index"] == index
    assert body["document_id"] == doc["id"] and body["filename"] == "long.txt"
    assert [p["chunk_index"] for p in body["before"]] == [index - 1]
    assert [p["chunk_index"] for p in body["after"]] == [index + 1]
    # Overlap is removed, so the pieces never repeat the cited chunk's own text.
    assert not body["before"][0]["text"].endswith(text[:20])
    assert not body["after"][0]["text"].startswith(text[-20:])


def test_neighbours_can_be_widened_or_turned_off_and_stop_at_the_ends(client, admin_headers):
    doc = _upload(client, admin_headers, "long.txt", "".join(SENTENCES))
    rows = _chunks(doc["id"])

    first = client.get(f"/api/chunks/{rows[0][0]}/context?neighbours=2", headers=admin_headers)
    assert [p["chunk_index"] for p in first.json()["before"]] == []
    assert [p["chunk_index"] for p in first.json()["after"]] == [1, 2]

    alone = client.get(f"/api/chunks/{rows[2][0]}/context?neighbours=0", headers=admin_headers)
    assert alone.json()["before"] == [] and alone.json()["after"] == []

    too_many = client.get(f"/api/chunks/{rows[2][0]}/context?neighbours=6", headers=admin_headers)
    assert too_many.status_code == 422


def test_a_missing_chunk_is_a_404(client, admin_headers):
    assert client.get("/api/chunks/987654/context", headers=admin_headers).status_code == 404


def test_it_needs_a_login(client, admin_headers):
    doc = _upload(client, admin_headers, "long.txt", "".join(SENTENCES))
    cid = _chunks(doc["id"])[0][0]
    assert client.get(f"/api/chunks/{cid}/context").status_code in (401, 403)


def test_a_user_without_access_gets_404_and_never_the_text(client, admin_headers, auth_headers):
    """The same predicate as GET /api/documents/{id}: no grant, no chunk, no hint it exists."""
    hr = client.post("/api/collections", json={"name": "hr"}, headers=admin_headers).json()
    doc = _upload(client, admin_headers, "secret.txt", "".join(SENTENCES), hr["id"])
    cid = _chunks(doc["id"])[1][0]
    group = client.post("/api/admin/groups", json={"name": "hr-only"}, headers=admin_headers).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": hr["id"], "permission": "read"},
        headers=admin_headers,
    )

    denied = client.get(f"/api/chunks/{cid}/context", headers=auth_headers)
    missing = client.get("/api/chunks/987654/context", headers=auth_headers)

    assert denied.status_code == 404
    assert denied.json() == missing.json()
    assert "Sentence number" not in denied.text
    # The document read is refused the same way, and the admin still reads the chunk.
    assert client.get(f"/api/documents/{doc['id']}", headers=auth_headers).status_code == 404
    assert client.get(f"/api/chunks/{cid}/context", headers=admin_headers).status_code == 200

    # Granting the viewer's group restores access, so the 404 was the predicate and nothing else.
    me = client.get("/api/auth/me", headers=auth_headers).json()
    client.post(
        f"/api/admin/groups/{group['id']}/members",
        json={"user_id": me["id"]},
        headers=admin_headers,
    )
    assert client.get(f"/api/chunks/{cid}/context", headers=auth_headers).status_code == 200
