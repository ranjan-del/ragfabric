"""Request ids, the access log line and the JSON 500 (Phase 10, Task 1)."""

from __future__ import annotations

import logging

import pytest
from fastapi import APIRouter

from ragfabric_core.telemetry.logs import current_request_id

ACCESS_LOGGER = "ragfabric.access"


def test_every_response_carries_a_generated_request_id(client):
    res = client.get("/health")
    rid = res.headers["x-request-id"]
    assert len(rid) == 32 and int(rid, 16) >= 0


def test_a_safe_incoming_request_id_is_echoed(client):
    res = client.get("/health", headers={"X-Request-ID": "edge-proxy.42_a"})
    assert res.headers["x-request-id"] == "edge-proxy.42_a"


@pytest.mark.parametrize("bad", ["a b", "x" * 129, 'q"uote', "brace{"])
def test_an_unsafe_incoming_request_id_is_replaced(client, bad):
    res = client.get("/health", headers={"X-Request-ID": bad})
    rid = res.headers["x-request-id"]
    assert rid != bad and len(rid) == 32


def test_errors_carry_the_request_id_too(client):
    res = client.get("/api/documents")  # no credentials: 401
    assert res.status_code == 401
    assert res.headers["x-request-id"]


def test_one_access_line_per_request_without_the_query_string(client, caplog):
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER)
    res = client.get("/health?token=secret-value", headers={"X-Request-ID": "acc-1"})
    lines = [r for r in caplog.records if r.name == ACCESS_LOGGER]
    assert len(lines) == 1
    record = lines[0]
    assert record.method == "GET"
    assert record.path == "/health"
    assert record.status == res.status_code == 200
    assert record.duration_ms >= 0
    assert record.request_id == "acc-1"
    assert "secret-value" not in record.getMessage()


def test_a_log_line_inside_a_route_carries_the_request_id(client, auth_headers):
    seen: list[str | None] = []

    router = APIRouter()

    @router.get("/__rid_probe")
    def probe() -> dict:
        seen.append(current_request_id())
        return {}

    client.app.include_router(router)
    try:
        res = client.get("/__rid_probe", headers={"X-Request-ID": "probe-1"})
    finally:
        client.app.router.routes = [
            r for r in client.app.router.routes if getattr(r, "path", "") != "/__rid_probe"
        ]
    assert res.status_code == 200
    assert seen == ["probe-1"]


def test_an_unhandled_error_is_a_json_500_with_the_request_id(client, caplog):
    router = APIRouter()

    @router.get("/__boom")
    def boom() -> dict:
        raise RuntimeError("database password is hunter2")

    client.app.include_router(router)
    caplog.set_level(logging.ERROR)
    try:
        res = client.get("/__boom", headers={"X-Request-ID": "boom-1"})
    finally:
        client.app.router.routes = [
            r for r in client.app.router.routes if getattr(r, "path", "") != "/__boom"
        ]
    assert res.status_code == 500
    assert res.json() == {"detail": "Internal server error", "request_id": "boom-1"}
    assert "hunter2" not in res.text
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and r.exc_info]
    assert errors and errors[0].request_id == "boom-1"


def test_a_streamed_ask_still_streams_and_carries_the_request_id(client, admin_token, ingested_doc):
    headers = {"Authorization": f"Bearer {admin_token}", "X-Request-ID": "stream-1"}
    body = {"query": "how many days of annual leave", "stream": True, "top_k": 3}
    with client.stream("POST", "/api/ask", json=body, headers=headers) as res:
        assert res.headers["x-request-id"] == "stream-1"
        raw = "".join(res.iter_text())
    assert res.status_code == 200
    assert "event: done" in raw


def test_startup_warns_when_the_shipped_admin_password_is_in_use(caplog):
    """The sign-in page no longer prints credentials; the server log says the
    development defaults are in use instead, without printing the password."""
    from fastapi.testclient import TestClient

    from ragfabric_server.main import app

    caplog.set_level(logging.WARNING)
    with TestClient(app):
        pass
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("shipped default password" in m for m in messages)
    assert not any("adminpass123" in m for m in messages)
