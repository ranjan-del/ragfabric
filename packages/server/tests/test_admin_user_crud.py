"""Admin user create and delete (Task 11).

The console's whole reason to exist is that access can be managed without
anyone opening a database client. Creating and deleting a user was the
largest remaining hole: the API could list users and change their role, but
an operator had to insert or delete the row by hand.

Deleting a user is the interesting half, because a user row is referenced
from nine other tables. Every one of those references is settled deliberately
here rather than left to whatever the database happens to do.
"""

from __future__ import annotations


def _login(client, email: str, password: str) -> str:
    resp = client.post("/api/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_an_admin_creates_a_user(client, admin_headers):
    resp = client.post(
        "/api/admin/users",
        json={"email": "New.Person@Example.com", "password": "password123", "role": "admin"},
        headers=admin_headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["email"] == "new.person@example.com"
    assert body["role"] == "admin"
    assert body["is_active"] is True
    assert "hashed_password" not in body
    token = _login(client, "new.person@example.com", "password123")
    assert token


def test_creating_a_user_with_an_existing_email_is_a_conflict(client, admin_headers):
    payload = {"email": "dup@example.com", "password": "password123"}
    assert client.post("/api/admin/users", json=payload, headers=admin_headers).status_code == 201
    assert client.post("/api/admin/users", json=payload, headers=admin_headers).status_code == 409


def test_creating_a_user_with_an_unknown_role_is_rejected(client, admin_headers):
    resp = client.post(
        "/api/admin/users",
        json={"email": "bad@example.com", "password": "password123", "role": "wizard"},
        headers=admin_headers,
    )
    assert resp.status_code == 400, resp.text


def test_a_non_admin_cannot_create_or_delete_a_user(client, admin_headers, auth_headers):
    victim = client.post(
        "/api/admin/users",
        json={"email": "victim@example.com", "password": "password123"},
        headers=admin_headers,
    ).json()
    assert (
        client.post(
            "/api/admin/users",
            json={"email": "sneaky@example.com", "password": "password123"},
            headers=auth_headers,
        ).status_code
        == 403
    )
    assert (
        client.delete(f"/api/admin/users/{victim['id']}", headers=auth_headers).status_code == 403
    )


def test_deleting_a_user_revokes_their_api_keys(client, admin_headers):
    victim = client.post(
        "/api/admin/users",
        json={"email": "keyholder@example.com", "password": "password123"},
        headers=admin_headers,
    ).json()
    created = client.post(
        "/api/admin/keys",
        json={"name": "victim-key", "user_id": victim["id"]},
        headers=admin_headers,
    )
    assert created.status_code == 201, created.text
    victim_key = created.json()["key"]

    before = client.post(
        "/api/search/semantic", json={"query": "anything"}, headers={"X-API-Key": victim_key}
    )
    assert before.status_code == 200, before.text

    assert (
        client.delete(f"/api/admin/users/{victim['id']}", headers=admin_headers).status_code == 200
    )

    after = client.post(
        "/api/search/semantic", json={"query": "anything"}, headers={"X-API-Key": victim_key}
    )
    assert after.status_code == 401


def test_deleting_a_user_keeps_their_collections_and_clears_the_owner(client, admin_headers):
    victim = client.post(
        "/api/admin/users",
        json={"email": "owner@example.com", "password": "password123"},
        headers=admin_headers,
    ).json()
    victim_headers = {
        "Authorization": f"Bearer {_login(client, 'owner@example.com', 'password123')}"
    }
    collection = client.post(
        "/api/collections", json={"name": "orphan-candidate"}, headers=victim_headers
    ).json()

    assert (
        client.delete(f"/api/admin/users/{victim['id']}", headers=admin_headers).status_code == 200
    )

    listed = client.get("/api/collections", headers=admin_headers).json()
    survivor = next(c for c in listed if c["id"] == collection["id"])
    assert survivor["owner_id"] is None


def test_deleting_a_user_removes_their_group_memberships(client, admin_headers, db_session):
    from ragfabric_core.models.access import GroupMember

    victim = client.post(
        "/api/admin/users",
        json={"email": "member@example.com", "password": "password123"},
        headers=admin_headers,
    ).json()
    group = client.post(
        "/api/admin/groups", json={"name": "delete-me-members"}, headers=admin_headers
    ).json()
    assert (
        client.post(
            f"/api/admin/groups/{group['id']}/members",
            json={"user_id": victim["id"]},
            headers=admin_headers,
        ).status_code
        == 200
    )

    assert (
        client.delete(f"/api/admin/users/{victim['id']}", headers=admin_headers).status_code == 200
    )

    db_session.expire_all()
    remaining = db_session.query(GroupMember).filter(GroupMember.user_id == victim["id"]).all()
    assert remaining == []


def test_an_admin_cannot_delete_their_own_account(client, admin_headers):
    me = client.get("/api/auth/me", headers=admin_headers).json()
    resp = client.delete(f"/api/admin/users/{me['id']}", headers=admin_headers)
    assert resp.status_code == 400, resp.text
    assert client.get("/api/auth/me", headers=admin_headers).status_code == 200


def test_deleting_a_user_who_does_not_exist_is_404(client, admin_headers):
    assert client.delete("/api/admin/users/999999", headers=admin_headers).status_code == 404
