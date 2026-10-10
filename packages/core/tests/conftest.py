"""Shared fixtures for the core tests."""

import pytest

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.strategies.base import Budget, RetrievalContext, StrategyParams


def _make_ctx(**overrides) -> RetrievalContext:
    values = dict(
        principal=Principal(
            user_id=1, email="u@example.com", role="user", group_ids=[], api_key_id=None
        ),
        access_filter=AccessFilter.unrestricted(),
        collection_ids=None,
        params=StrategyParams(),
        budget=Budget(),
    )
    values.update(overrides)
    return RetrievalContext(**values)


@pytest.fixture
def make_ctx():
    """The context builder as a fixture, so tests call ``make_ctx(**overrides)``."""
    return _make_ctx


@pytest.fixture
def small_subgraph():
    """Three nodes and one edge: Ravi Sharma MEMBER_OF Platform Team, sourced from chunk 1."""
    from ragfabric_core.graph.contracts import (
        EntityType,
        GraphEdge,
        GraphNode,
        RelationType,
        Subgraph,
    )

    return Subgraph(
        nodes=[
            GraphNode(id=1, name="Ravi Sharma", entity_type=EntityType.PERSON, depth=0),
            GraphNode(id=2, name="Platform Team", entity_type=EntityType.TEAM, depth=1),
            GraphNode(id=3, name="Billing", entity_type=EntityType.PRODUCT, depth=0),
        ],
        edges=[
            GraphEdge(
                id=1,
                source_id=1,
                target_id=2,
                relation_type=RelationType.MEMBER_OF,
                walked_as="MEMBER_OF",
                reversed=False,
                confidence=0.9,
                source_chunk_ids=[1],
            )
        ],
        truncated=False,
        empty_reason=None,
    )


@pytest.fixture()
def eval_db(tmp_path, monkeypatch):
    """A migrated SQLite database and an offline configuration, wired the way the CLI's is.

    Yields ``(config_path, session_factory)``. The session module caches its
    engine at import, so a fresh one is patched in, as the CLI tests do.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import ragfabric_core.db.session as session_module
    from ragfabric_core import runtime
    from ragfabric_core.db.migrate import upgrade

    url = f"sqlite:///{tmp_path / 'eval.db'}"
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text(
        "llm:\n  provider: offline\nembeddings:\n  provider: offline\n  dim: 16\n"
        f"cache:\n  kind: memory\ningestion:\n  uploads_dir: {tmp_path / 'blobs'}\n"
    )
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(cfg))
    runtime.reset_config()
    engine = create_engine(url, connect_args={"check_same_thread": False})
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", factory)
    upgrade(url)
    yield cfg, factory
    runtime.reset_config()
