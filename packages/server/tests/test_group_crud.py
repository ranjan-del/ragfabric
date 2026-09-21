"""Group update, delete and member listing (Task 11).

Groups could be created and have members added and removed, but never
renamed or deleted, and their membership could not be read back. A console
that cannot show who is in a group cannot answer the only question an
operator ever asks of one.
"""

from __future__ import annotations


def test_an_admin_renames_a_group(client, admin_headers):
    group = client.post(
        "/api/admin/groups", json={"name": "hr-team", "description": "old"}, headers=admin_headers
    ).json()
    resp = client.put(
        f"/api/admin/groups/{group['id']}",
        json={"name": "people-team", "description": "new"},
        headers=admin_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "people-team"
    assert resp.json()["description"] == "new"


def test_renaming_a_group_onto_an_existing_name_is_a_conflict(client, admin_headers):
    first = client.post("/api/admin/groups", json={"name": "alpha"}, headers=admin_headers).json()
    client.post("/api/admin/groups", json={"name": "beta"}, headers=admin_headers)
    resp = client.put(
        f"/api/admin/groups/{first['id']}", json={"name": "beta"}, headers=admin_headers
    )
    assert resp.status_code == 409, resp.text


def test_updating_a_group_that_does_not_exist_is_404(client, admin_headers):
    assert (
        client.put(
            "/api/admin/groups/999999", json={"name": "x"}, headers=admin_headers
        ).status_code
        == 404
    )


def test_group_members_can_be_listed(client, admin_headers, auth_headers):
    me = client.get("/api/auth/me", headers=auth_headers).json()
    group = client.post("/api/admin/groups", json={"name": "readers"}, headers=admin_headers).json()
    client.post(
        f"/api/admin/groups/{group['id']}/members",
        json={"user_id": me["id"]},
        headers=admin_headers,
    )
    listed = client.get(f"/api/admin/groups/{group['id']}/members", headers=admin_headers)
    assert listed.status_code == 200, listed.text
    assert [u["id"] for u in listed.json()] == [me["id"]]

    client.delete(f"/api/admin/groups/{group['id']}/members/{me['id']}", headers=admin_headers)
    assert (
        client.get(f"/api/admin/groups/{group['id']}/members", headers=admin_headers).json() == []
    )


def test_deleting_a_group_removes_its_grants_not_its_users(client, admin_headers, auth_headers):
    me = client.get("/api/auth/me", headers=auth_headers).json()
    group = client.post("/api/admin/groups", json={"name": "doomed"}, headers=admin_headers).json()
    client.post(
        f"/api/admin/groups/{group['id']}/members",
        json={"user_id": me["id"]},
        headers=admin_headers,
    )
    collection = client.post(
        "/api/collections", json={"name": "granted"}, headers=admin_headers
    ).json()
    grant = client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": collection["id"], "permission": "read"},
        headers=admin_headers,
    )
    assert grant.status_code == 201, grant.text

    resp = client.delete(f"/api/admin/groups/{group['id']}", headers=admin_headers)
    assert resp.status_code == 200, resp.text

    assert client.get("/api/admin/grants", headers=admin_headers).json() == []
    assert [g["id"] for g in client.get("/api/admin/groups", headers=admin_headers).json()] == []
    assert me["id"] in [
        u["id"] for u in client.get("/api/admin/users", headers=admin_headers).json()
    ]


def test_deleting_a_group_that_does_not_exist_is_404(client, admin_headers):
    assert client.delete("/api/admin/groups/999999", headers=admin_headers).status_code == 404


def test_a_non_admin_cannot_update_or_delete_a_group(client, admin_headers, auth_headers):
    group = client.post("/api/admin/groups", json={"name": "guarded"}, headers=admin_headers).json()
    assert (
        client.put(
            f"/api/admin/groups/{group['id']}", json={"name": "hijacked"}, headers=auth_headers
        ).status_code
        == 403
    )
    assert (
        client.delete(f"/api/admin/groups/{group['id']}", headers=auth_headers).status_code == 403
    )


def test_an_admin_renames_a_collection(client, admin_headers):
    collection = client.post(
        "/api/collections", json={"name": "before", "description": ""}, headers=admin_headers
    ).json()
    resp = client.put(
        f"/api/collections/{collection['id']}",
        json={"name": "after", "description": "renamed"},
        headers=admin_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "after"
    assert resp.json()["description"] == "renamed"
    assert (
        client.get(f"/api/collections/{collection['id']}", headers=admin_headers).json()["name"]
        == "after"
    )


def test_only_the_owner_or_an_admin_may_rename_a_collection(client, admin_headers, auth_headers):
    collection = client.post(
        "/api/collections", json={"name": "admins-own"}, headers=admin_headers
    ).json()
    resp = client.put(
        f"/api/collections/{collection['id']}", json={"name": "stolen"}, headers=auth_headers
    )
    assert resp.status_code == 403, resp.text
