from ragfabric_core.testing.fixtures import make_txt


def _upload(client, headers, name, text, collection_id=None):
    data = {"collection_id": str(collection_id)} if collection_id else {}
    r = client.post(
        "/api/documents/upload",
        files={"file": (name, make_txt(text), "text/plain")},
        data=data,
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_admin_manages_groups_grants_and_keys(client, admin_headers, auth_headers):
    me = client.get("/api/auth/me", headers=auth_headers).json()
    g = client.post("/api/admin/groups", json={"name": "hr-team"}, headers=admin_headers)
    assert g.status_code == 201 and g.json()["name"] == "hr-team"
    gid = g.json()["id"]
    assert (
        client.post(
            f"/api/admin/groups/{gid}/members", json={"user_id": me["id"]}, headers=admin_headers
        ).status_code
        == 200
    )
    col = client.post("/api/collections", json={"name": "hr"}, headers=admin_headers).json()
    grant = client.post(
        "/api/admin/grants",
        json={"group_id": gid, "collection_id": col["id"], "permission": "read"},
        headers=admin_headers,
    )
    assert grant.status_code == 201
    assert any(
        x["collection_id"] == col["id"]
        for x in client.get("/api/admin/grants", headers=admin_headers).json()
    )
    assert (
        client.post("/api/admin/groups", json={"name": "x"}, headers=auth_headers).status_code
        == 403
    )


def test_api_key_is_shown_once_and_authenticates(client, admin_headers):
    admin = client.get("/api/auth/me", headers=admin_headers).json()
    created = client.post(
        "/api/admin/keys",
        json={"name": "ci", "user_id": admin["id"], "rate_limit_per_minute": 2},
        headers=admin_headers,
    )
    assert created.status_code == 201
    plain = created.json()["key"]
    assert plain.startswith("rf_")
    listed = client.get("/api/admin/keys", headers=admin_headers).json()
    assert listed and "key" not in listed[0] and listed[0]["key_prefix"] == plain[:12]
    _upload(client, admin_headers, "a.txt", "leave policy is twelve days")
    r = client.post("/api/search/semantic", json={"query": "leave"}, headers={"X-API-Key": plain})
    assert r.status_code == 200 and r.json()["results"]
    r = client.post(
        "/api/search/semantic",
        json={"query": "leave"},
        headers={"Authorization": f"Bearer {plain}"},
    )
    assert r.status_code == 200
    r = client.post("/api/search/semantic", json={"query": "leave"}, headers={"X-API-Key": plain})
    assert r.status_code == 429
    assert (
        client.post(
            "/api/search/semantic", json={"query": "leave"}, headers={"X-API-Key": "rf_wrong"}
        ).status_code
        == 401
    )
    key_id = listed[0]["id"]
    assert client.delete(f"/api/admin/keys/{key_id}", headers=admin_headers).status_code == 200
    assert (
        client.post(
            "/api/search/semantic", json={"query": "leave"}, headers={"X-API-Key": plain}
        ).status_code
        == 401
    )
