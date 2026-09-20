"""Pytest fixtures: isolated offline test environment.

Environment variables are set BEFORE any application module is imported so the
cached ``Settings`` and the SQLAlchemy engine bind to a throwaway SQLite file and
a deterministic JWT secret. Every test runs fully offline — no API key, no
Postgres, no vector-DB server.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session

# The names below are bound inside ``pytest_configure`` (see below), not here,
# so nothing at module scope forces the app's modules to import (and therefore
# bind DATABASE_URL/RAGFABRIC_CONFIG) before we know this process's own tmp dir.
TestClient = runtime = Base = SessionLocal = engine = None
Completion = Message = get_llm_provider = app = None


def pytest_configure(config: pytest.Config) -> None:
    """Point the app at a database and uploads dir private to this process.

    The test database and uploads dir used to be fixed paths under
    ``tempfile.gettempdir()``, shared by every pytest process on the machine.
    Two overlapping runs (e.g. two CI jobs, or a developer running the suite
    twice) would then read and write the same SQLite file and the same
    uploads directory at once, producing spurious SQLAlchemy errors that look
    like product bugs but are really just two processes fighting over one
    file. ``pytest.TempPathFactory`` is the same machinery the ``tmp_path``
    and ``tmp_path_factory`` fixtures are built on; calling it here, from
    ``pytest_configure``, gives this process its own numbered directory
    (``pytest-of-<user>/pytest-<N>/...``) using the same collision-proof
    locking pytest itself uses, before those fixtures would normally be
    available. ``pytest_configure`` is a "historic" hook, so pytest replays it
    for this conftest the moment the module is registered, which happens
    before any test module in this directory is imported, even though this
    conftest is discovered during collection rather than at startup. That
    ordering is what lets us set environment variables here and still have
    them in place before ``ragfabric_core.db.session`` (and everything else
    imported below) binds to them at import time.
    """
    global TestClient, runtime, Base, SessionLocal, engine
    global Completion, Message, get_llm_provider, app

    tmp_path_factory = pytest.TempPathFactory.from_config(config, _ispytest=True)
    tmp_dir = tmp_path_factory.mktemp("ragfabric-server-tests")

    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_dir / 'rag_test.db'}"
    os.environ["JWT_SECRET"] = "test-secret"
    os.environ["FIRST_ADMIN_EMAIL"] = "admin@example.com"
    os.environ["FIRST_ADMIN_PASSWORD"] = "adminpass123"
    os.environ.pop("ANTHROPIC_API_KEY", None)  # force the offline answer path

    uploads_dir = tmp_dir / "uploads"
    cfg_path = tmp_dir / "config.yaml"
    cfg_path.write_text(
        f"ingestion:\n  uploads_dir: {uploads_dir}\ncache:\n  kind: memory\n"
        # dim: 32 is too small for the bag-of-words hashing embedder to
        # reliably discriminate unrelated text from related text (hash
        # collisions dominate at that width); 128 is still fast and offline
        # but stops "unrelated query" tests from scoring a false positive.
        "embeddings:\n  provider: offline\n  dim: 128\n"
    )
    os.environ["RAGFABRIC_CONFIG"] = str(cfg_path)

    from fastapi.testclient import TestClient as _TestClient

    from ragfabric_core import runtime as _runtime
    from ragfabric_core.db.session import Base as _Base
    from ragfabric_core.db.session import SessionLocal as _SessionLocal
    from ragfabric_core.db.session import engine as _engine
    from ragfabric_core.providers.base import Completion as _Completion
    from ragfabric_core.providers.base import Message as _Message
    from ragfabric_server.deps import get_llm_provider as _get_llm_provider
    from ragfabric_server.main import app as _app

    TestClient, runtime, Base, SessionLocal, engine = (
        _TestClient,
        _runtime,
        _Base,
        _SessionLocal,
        _engine,
    )
    Completion, Message, get_llm_provider, app = (
        _Completion,
        _Message,
        _get_llm_provider,
        _app,
    )

    runtime.reset_config()


class _FakeCitingLLM:
    """Offline, deterministic ``LLMProvider`` test double.

    ``get_llm_provider`` builds a real provider from ``ragfabric.yaml``, whose
    default (``ollama``) would try to reach a local model server, and whose
    ``offline`` config value builds a ``ScriptedLLMProvider`` with no canned
    responses, which raises the instant it is called. Neither is usable for a
    route test that actually wants an answer, and unit tests must never touch
    the network, so ``/api/search/query`` gets this double instead: it reads
    every numbered passage back out of the prompt
    ``generate.cited.build_prompt`` builds and quotes each one verbatim, in
    full, tagged with its own marker. Quoting a passage in full rather than a
    clipped prefix means any word a test looks for in the source chunk (and
    any sentence ``select_support`` picks out of it) is guaranteed to be a
    substring of the answer, the same way a real model's paraphrase would
    normally still surface the source's key terms. This satisfies the
    citation contract (marker validity, quote fidelity, grounding) on the
    first attempt, with no model call of any kind.
    """

    name = "fake"
    default_model = "fake-test-model"

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        user_content = next(m.content for m in messages if m.role == "user")
        passages_block = user_content.split("Passages:\n", 1)[-1]
        quotes = []
        for segment in passages_block.split("\n\n"):
            match = re.match(r"\[(\d+)\] (.*)", segment, re.S)
            if not match:
                continue
            marker, passage = match.group(1), match.group(2).strip()
            if passage:
                quotes.append(f'"{passage}" [{marker}]')
        text = (
            "Based on the sources: " + " ".join(quotes)
            if quotes
            else "I could not find an answer to that in the documents provided."
        )
        return Completion(
            text=text,
            model=model or self.default_model,
            provider=self.name,
            input_tokens=len(user_content.split()),
            output_tokens=len(text.split()),
            latency_ms=0,
            finish_reason="stop",
        )

    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        """Yield the same deterministic text ``complete`` would return, in
        pieces, so ``/api/ask`` can be exercised as a real stream without a
        network call. Built on ``complete`` rather than duplicating its
        passage-quoting logic, so a streamed answer and a non-streamed one for
        the same prompt are always textually identical.
        """
        text = self.complete(
            messages, model=model, max_tokens=max_tokens, temperature=temperature
        ).text
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """A TestClient backed by a fresh database and empty vector index.

    ``get_llm_provider`` is overridden with the offline fake above so that
    ``/api/search/query`` never attempts a real model call in a unit test.
    """
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_llm_provider] = lambda: _FakeCitingLLM()
    with TestClient(app) as test_client:  # triggers lifespan (seeds admin)
        yield test_client
    app.dependency_overrides.pop(get_llm_provider, None)
    Base.metadata.drop_all(bind=engine)


def _register_and_login(client: TestClient, email: str, password: str = "password123") -> str:
    """Register a user (ignore duplicates) and return a bearer token."""
    client.post("/api/auth/register", json={"email": email, "password": password})
    resp = client.post("/api/auth/login", data={"username": email, "password": password})
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


@pytest.fixture()
def admin_token(client: TestClient) -> str:
    """Bare bearer token for the seeded bootstrap admin (no header wrapper)."""
    resp = client.post(
        "/api/auth/login",
        data={"username": "admin@example.com", "password": "adminpass123"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture()
def db_session() -> Iterator[Session]:
    """A raw SQLAlchemy session over the same test database the app uses."""
    with SessionLocal() as session:
        yield session


@pytest.fixture()
def ingested_doc(client: TestClient, admin_token: str) -> dict:
    """Upload a small text document and wait for it to reach status ``ready``.

    Real ``chunk_embeddings``/``chunk_search`` rows only exist once a document
    has been through the fan-out ingestion writes, so any test exercising the
    real (non in-memory) retrieval path needs one of these first.
    """
    from ragfabric_core.testing.fixtures import make_txt

    headers = {"Authorization": f"Bearer {admin_token}"}
    resp = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "leave-policy.txt",
                make_txt(
                    "Employees are entitled to twenty five days of annual leave per year, "
                    "in addition to public holidays."
                ),
                "text/plain",
            )
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    document = resp.json()

    deadline = time.monotonic() + 5.0
    while document["status"] == "processing" and time.monotonic() < deadline:
        time.sleep(0.05)
        document = client.get(f"/api/documents/{document['id']}", headers=headers).json()

    assert document["status"] == "ready", document
    return document
