"""End to end on PostgreSQL: the same scenario on pgvector and full text (Phase 10, Task 7).

The SQLite tier (``test_e2e_flow.py``) proves the wiring; SQLite has neither
pgvector nor PostgreSQL full text, so the stores there run on their fallbacks.
This tier runs the identical scenario against a real PostgreSQL with the
``pgvector`` vector store and the ``postgres_fts`` lexical store, migrated by
Alembic exactly as a deployment is. It runs in the required "Migrations apply
cleanly" CI job, which already has PostgreSQL 18 with pgvector, and is skipped
anywhere ``RAGFABRIC_TEST_DATABASE_URL`` is not set.

The app module binds its engine at import time to the SQLite file conftest.py
chose, so this fixture rebinds every place that holds a session factory to the
PostgreSQL database for the length of the test and puts them back afterwards.
"""

from __future__ import annotations

import os

import pytest
from e2e_scenario import run_scenario
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ragfabric_core import runtime
from ragfabric_core.config_file import (
    EmbeddingsConfig,
    LexicalStoreConfig,
    VectorStoreConfig,
)
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models.index import EMBEDDING_DIM

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL"),
]


@pytest.fixture()
def pg_client(tmp_path, monkeypatch):
    import conftest

    import ragfabric_core.db.session as session_module
    import ragfabric_server.main as main_module
    from ragfabric_server.api.routes import ask as ask_route
    from ragfabric_server.deps import get_llm_provider

    downgrade(URL)
    upgrade(URL)
    engine = create_engine(URL, pool_pre_ping=True)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    for module in (session_module, main_module, ask_route):
        monkeypatch.setattr(module, "SessionLocal", factory)
    monkeypatch.setattr(session_module, "engine", engine)

    base = runtime.get_config()
    runtime._config = base.model_copy(
        update={
            # The column is vector(EMBEDDING_DIM), pinned by migration 0004.
            "embeddings": EmbeddingsConfig(provider="offline", dim=EMBEDDING_DIM),
            "vector_store": VectorStoreConfig(kind="pgvector"),
            "lexical_store": LexicalStoreConfig(kind="postgres_fts"),
            "ingestion": base.ingestion.model_copy(update={"uploads_dir": str(tmp_path / "up")}),
        }
    )
    main_module.app.dependency_overrides[get_llm_provider] = lambda: conftest._FakeCitingLLM()
    try:
        with TestClient(main_module.app) as client:
            yield client, engine
    finally:
        main_module.app.dependency_overrides.pop(get_llm_provider, None)
        runtime.reset_config()
        engine.dispose()
        downgrade(URL)


def test_upload_to_cited_answer_on_postgres(pg_client, tmp_path):
    client, engine = pg_client
    outcome = run_scenario(client, tmp_path)

    # The rows the answer came from really are in pgvector and PostgreSQL full text.
    with engine.connect() as conn:
        assert conn.execute(text("select current_setting('server_version_num')::int")).scalar()
        vectors = conn.execute(
            text("select count(*) from chunk_embeddings where embedding is not null")
        ).scalar()
        indexed = conn.execute(
            text("select count(*) from chunk_search where tsv is not null")
        ).scalar()
    assert vectors > 0
    assert indexed > 0
    assert len(outcome.document_ids) == 2
