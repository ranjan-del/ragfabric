"""Pytest fixtures: isolated offline test environment.

Environment variables are set BEFORE any application module is imported so the
cached ``Settings`` and the SQLAlchemy engine bind to a throwaway SQLite file and
a deterministic JWT secret. Every test runs fully offline — no API key, no
Postgres, no vector-DB server.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import pytest

# --- configure the environment before importing the app -----------------------
_TMP_DB = os.path.join(tempfile.gettempdir(), "rag_test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["FIRST_ADMIN_EMAIL"] = "admin@example.com"
os.environ["FIRST_ADMIN_PASSWORD"] = "adminpass123"
os.environ.pop("ANTHROPIC_API_KEY", None)  # force the offline answer path

from fastapi.testclient import TestClient  # noqa: E402

from ragfabric_core.db.session import Base, engine  # noqa: E402
from ragfabric_server.main import app  # noqa: E402
from ragfabric_core.store.vector_store import get_store  # noqa: E402


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """A TestClient backed by a fresh database and empty vector index."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    get_store().clear()
    with TestClient(app) as test_client:  # triggers lifespan (seeds admin)
        yield test_client
    Base.metadata.drop_all(bind=engine)
    get_store().clear()


def _register_and_login(client: TestClient, email: str, password: str = "password123") -> str:
    """Register a user (ignore duplicates) and return a bearer token."""
    client.post("/api/auth/register", json={"email": email, "password": password})
    resp = client.post(
        "/api/auth/login", data={"username": email, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture()
def auth_headers(client: TestClient) -> dict:
    """Authorization header for a regular (non-admin) user."""
    token = _register_and_login(client, "user@example.com")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def admin_headers(client: TestClient) -> dict:
    """Authorization header for the seeded bootstrap admin."""
    resp = client.post(
        "/api/auth/login",
        data={"username": "admin@example.com", "password": "adminpass123"},
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}
