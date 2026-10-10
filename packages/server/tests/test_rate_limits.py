"""Rate limits for signed-in users, API keys and sign-in attempts (Phase 10, Task 2)."""

from __future__ import annotations

import logging

import pytest

from ragfabric_core import runtime
from ragfabric_core.testing.fixtures import make_txt


@pytest.fixture()
def limits():
    cfg = runtime.get_config().limits
    saved = cfg.model_copy()
    yield cfg
    cfg.user_rate_limit_per_minute = saved.user_rate_limit_per_minute
    cfg.auth_attempts_per_minute = saved.auth_attempts_per_minute


def _login(client, email="admin@example.com", password="adminpass123", headers=None):
    return client.post(
        "/api/auth/login", data={"username": email, "password": password}, headers=headers or {}
    )


def test_a_signed_in_user_over_the_limit_gets_429_with_retry_after(client, admin_headers, limits):
    limits.user_rate_limit_per_minute = 2
    assert client.get("/api/auth/me", headers=admin_headers).status_code == 200
    assert client.get("/api/auth/me", headers=admin_headers).status_code == 200
    res = client.get("/api/auth/me", headers=admin_headers)
    assert res.status_code == 429
    assert res.json() == {"detail": "Rate limit exceeded"}
    assert 1 <= int(res.headers["retry-after"]) <= 60
    assert res.headers["x-ratelimit-limit"] == "2"
    assert res.headers["x-ratelimit-remaining"] == "0"
    assert res.headers["x-request-id"]


def test_one_user_over_the_limit_does_not_block_another(
    client, admin_headers, auth_headers, limits
):
    limits.user_rate_limit_per_minute = 1
    assert client.get("/api/auth/me", headers=admin_headers).status_code == 200
    assert client.get("/api/auth/me", headers=admin_headers).status_code == 429
    assert client.get("/api/auth/me", headers=auth_headers).status_code == 200


def test_a_route_that_resolves_the_caller_twice_charges_once(client, admin_headers, limits):
    """Upload depends on get_current_user and on get_access_filter, which
    resolves the caller again through get_principal. Charging both would halve
    the user's real limit."""
    limits.user_rate_limit_per_minute = 2
    for name in ("a.txt", "b.txt"):
        res = client.post(
            "/api/documents/upload",
            files={"file": (name, make_txt("annual leave is twenty days"), "text/plain")},
            headers=admin_headers,
        )
        assert res.status_code == 201, res.text


def test_login_attempts_are_limited_per_client_address(client, limits):
    limits.auth_attempts_per_minute = 2
    assert _login(client, password="wrong").status_code == 401
    assert _login(client, password="wrong").status_code == 401
    res = _login(client)  # even the right password: the address is over its limit
    assert res.status_code == 429
    assert int(res.headers["retry-after"]) >= 1


def test_register_counts_toward_the_same_auth_limit(client, limits):
    limits.auth_attempts_per_minute = 1
    first = client.post(
        "/api/auth/register", json={"email": "a@example.com", "password": "password123"}
    )
    assert first.status_code == 201
    second = client.post(
        "/api/auth/register", json={"email": "b@example.com", "password": "password123"}
    )
    assert second.status_code == 429


def test_a_spoofed_forwarded_for_header_does_not_change_the_address(client, limits):
    """The client address comes from the ASGI scope, which uvicorn fills from
    X-Forwarded-For only for a trusted proxy. A header from anyone else must not
    give each attempt a fresh address."""
    limits.auth_attempts_per_minute = 1
    assert (
        _login(client, password="wrong", headers={"X-Forwarded-For": "10.0.0.1"}).status_code == 401
    )
    res = _login(client, password="wrong", headers={"X-Forwarded-For": "10.0.0.2"})
    assert res.status_code == 429


def test_an_api_key_keeps_its_own_limit_and_is_not_charged_as_a_user(client, admin_headers, limits):
    limits.user_rate_limit_per_minute = 1
    admin = client.get("/api/auth/me", headers=admin_headers).json()  # the user's one request
    created = client.post(
        "/api/admin/keys",
        json={"name": "ci", "user_id": admin["id"], "rate_limit_per_minute": 3},
        headers=admin_headers,
    )
    # The admin's own second request is over the user limit.
    assert created.status_code == 429
    limits.user_rate_limit_per_minute = 100
    created = client.post(
        "/api/admin/keys",
        json={"name": "ci", "user_id": admin["id"], "rate_limit_per_minute": 3},
        headers=admin_headers,
    )
    assert created.status_code == 201
    limits.user_rate_limit_per_minute = 1
    key = {"X-API-Key": created.json()["key"]}
    codes = [client.get("/api/documents", headers=key).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_an_unreachable_limiter_allows_the_request_and_logs_an_error(
    client, admin_headers, limits, caplog
):
    class _Down:
        name = "down"

        def incr(self, key, ttl_seconds=None):
            raise ConnectionError("redis is down")

        def get(self, key):
            return None

        def set(self, key, value, ttl_seconds=None):
            pass

    limits.user_rate_limit_per_minute = 1
    client.app.state.cache = _Down()
    caplog.set_level(logging.ERROR)
    for _ in range(3):
        res = client.get("/api/auth/me", headers={**admin_headers, "X-Request-ID": "rl-down"})
        assert res.status_code == 200
    errors = [r for r in caplog.records if "rate limit" in r.getMessage().lower()]
    assert errors and errors[0].request_id == "rl-down"
