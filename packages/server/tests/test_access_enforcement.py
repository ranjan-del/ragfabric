"""Phase 2 exit criterion: a viewer without a grant gets zero chunks from a restricted collection."""

from ragfabric_core.testing.fixtures import make_txt


def _upload(client, headers, name, text, collection_id):
    r = client.post(
        "/api/documents/upload",
        files={"file": (name, make_txt(text), "text/plain")},
        data={"collection_id": str(collection_id)},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_viewer_without_grant_gets_nothing_from_a_restricted_collection(
    client, admin_headers, auth_headers
):
    hr = client.post("/api/collections", json={"name": "hr"}, headers=admin_headers).json()
    doc = _upload(
        client, admin_headers, "policy.txt", "annual leave is twelve days for everyone", hr["id"]
    )
    # Before any grant exists the collection is open (v1 behaviour).
    assert client.post(
        "/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers
    ).json()["results"]
    group = client.post("/api/admin/groups", json={"name": "hr-only"}, headers=admin_headers).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": hr["id"], "permission": "read"},
        headers=admin_headers,
    )
    # Now the collection is restricted to the group, which the viewer is not in.
    r = client.post("/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers)
    assert r.status_code == 200 and r.json()["results"] == []
    answer = client.post(
        "/api/search/query", json={"query": "how many days of annual leave"}, headers=auth_headers
    ).json()
    assert answer["citations"] == [] and "enough information" in answer["answer"]
    assert client.get(f"/api/documents/{doc['id']}", headers=auth_headers).status_code == 404
    assert all(
        d["id"] != doc["id"]
        for d in client.get("/api/documents", headers=auth_headers).json()["items"]
    )
    assert (
        client.get(f"/api/documents/{doc['id']}/download", headers=auth_headers).status_code == 404
    )
    # Admin still sees it, and the audit row records what the filter removed for the viewer.
    assert client.get(f"/api/documents/{doc['id']}", headers=admin_headers).status_code == 200
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.access import AuditLog

    with SessionLocal() as db:
        rows = (
            db.query(AuditLog).filter(AuditLog.action == "query").order_by(AuditLog.id.desc()).all()
        )
    assert rows and rows[0].sources_returned == 0 and rows[0].sources_filtered >= 1
    # Adding the viewer to the group restores access.
    me = client.get("/api/auth/me", headers=auth_headers).json()
    client.post(
        f"/api/admin/groups/{group['id']}/members",
        json={"user_id": me["id"]},
        headers=admin_headers,
    )
    assert client.post(
        "/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers
    ).json()["results"]


def test_viewer_cannot_upload_into_a_collection_they_cannot_read(client, admin_headers, auth_headers):
    hr = client.post("/api/collections", json={"name": "hr"}, headers=admin_headers).json()
    group = client.post("/api/admin/groups", json={"name": "hr-only"}, headers=admin_headers).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": hr["id"], "permission": "read"},
        headers=admin_headers,
    )
    r = client.post(
        "/api/documents/upload",
        files={"file": ("policy.txt", make_txt("annual leave is twelve days"), "text/plain")},
        data={"collection_id": str(hr["id"])},
        headers=auth_headers,
    )
    assert r.status_code == 404
    # Admins are unrestricted, so their upload into the same collection still succeeds.
    admin_doc = _upload(client, admin_headers, "policy.txt", "annual leave is twelve days", hr["id"])
    assert admin_doc["id"]
    # The viewer can still upload when no collection is named.
    loose = client.post(
        "/api/documents/upload",
        files={"file": ("loose.txt", make_txt("no collection here"), "text/plain")},
        headers=auth_headers,
    )
    assert loose.status_code == 201


def test_query_records_a_retrieval_run_with_sources_and_is_readable_by_its_owner(
    client, auth_headers, admin_headers
):
    col = client.post("/api/collections", json={"name": "c"}, headers=auth_headers).json()
    _upload(client, auth_headers, "n.txt", "kubernetes rollout guide with three steps", col["id"])
    client.post("/api/search/query", json={"query": "rollout steps"}, headers=auth_headers)
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.runs import RetrievalRun

    with SessionLocal() as db:
        run = db.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert (
        run.selected_strategy == "traditional" and run.mode == "manual" and run.retrieval_calls == 1
    )
    assert run.embedding_model.startswith("hashing-") and run.latency_ms >= 0
    r = client.get(f"/api/runs/{run.id}", headers=auth_headers)
    assert r.status_code == 200 and r.json()["sources"] and r.json()["question"] == "rollout steps"
    names = [s["name"] for s in r.json()["trace"]]
    assert "hybrid_search" in names or "semantic_search" in names
    assert "answer" in names
    assert client.get(f"/api/runs/{run.id}", headers=admin_headers).status_code == 200
    client.post(
        "/api/auth/register", json={"email": "other@example.com", "password": "password123"}
    )
    tok = client.post(
        "/api/auth/login", data={"username": "other@example.com", "password": "password123"}
    ).json()["access_token"]
    assert (
        client.get(f"/api/runs/{run.id}", headers={"Authorization": f"Bearer {tok}"}).status_code
        == 404
    )
