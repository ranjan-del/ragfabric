"""The graph strategy and the merge records from the command line (Task 12, ruling R37).

``ragfabric ask`` goes through the SDK, so its tests assert what reaches the
wire and what is printed. ``ragfabric graph merges`` is an operator command
over the database, like ``users`` and ``groups``, so its tests seed a real
merge with ``resolve_entities`` and undo it through the command.
"""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()

_SUBGRAPH = {
    "nodes": [
        {"id": 1, "name": "Platform Team", "entity_type": "team", "depth": 0},
        {"id": 2, "name": "Engineering", "entity_type": "organisation", "depth": 1},
    ],
    "edges": [
        {
            "id": 5,
            "source_id": 1,
            "target_id": 2,
            "relation_type": "MEMBER_OF",
            "walked_as": "MEMBER_OF",
            "reversed": False,
            "confidence": 0.8,
            "source_chunk_ids": [9],
        }
    ],
    "truncated": False,
    "empty_reason": None,
}
_DROPPED = [
    {"text": "Engineering reports to the Board [E 99] [1].", "reason": "edge_not_in_subgraph"}
]
_ANSWER = {
    "question": "q",
    "answer": "The Platform Team is a member of Engineering [E 1] [1].",
    "confidence": 0.9,
    "citations": [],
    "highlights": [],
    "source_document": None,
    "usage": None,
    "subgraph": _SUBGRAPH,
    "dropped_relationship_claims": _DROPPED,
}


def _patched(monkeypatch, captured: list[dict], stream: str | None = None) -> None:
    import ragfabric_sdk.client as sdk_client

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if body.get("stream"):
            return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_ANSWER)

    real_client = sdk_client.httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(sdk_client.httpx, "Client", fake_client)


def test_ask_sends_the_graph_strategy(monkeypatch):
    captured: list[dict] = []
    _patched(monkeypatch, captured)

    result = runner.invoke(
        app,
        [
            "ask",
            "what is the platform team part of",
            "--token",
            "t",
            "--strategy",
            "graph",
            "--no-stream",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured and captured[0]["strategy"] == "graph"
    assert "Platform Team MEMBER_OF Engineering" in result.stdout
    assert "edge_not_in_subgraph" in result.stdout
    assert "Engineering reports to the Board" in result.stdout


def test_the_json_output_carries_the_subgraph_and_the_dropped_claims(monkeypatch):
    _patched(monkeypatch, [])

    result = runner.invoke(app, ["ask", "q", "--token", "t", "--strategy", "graph", "--json"])

    assert result.exit_code == 0, result.output
    printed = json.loads(result.stdout)
    assert printed["subgraph"]["edges"][0]["walked_as"] == "MEMBER_OF"
    assert printed["dropped_relationship_claims"] == _DROPPED


def test_the_stream_prints_the_subgraph_and_the_dropped_claims(monkeypatch):
    def event(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"

    stream = (
        event(
            "retrieval",
            {
                "chunks": 1,
                "strategy": "graph",
                "trace": [],
                "sub_questions": [],
                "subgraph": _SUBGRAPH,
            },
        )
        + event("token", {"text": "The Platform Team is a member of Engineering [E 1] [1]. "})
        + event("token", {"text": "Engineering reports to the Board [E 99] [1]."})
        + event(
            "superseded",
            {
                "text": "The Platform Team is a member of Engineering [E 1] [1].",
                "reason": "unsupported claims removed",
                "dropped_claims": [],
                "dropped_relationship_claims": _DROPPED,
            },
        )
        + event("citations", {"citations": []})
        + event("done", {"run_id": 3, "latency_ms": 5, "usage": {}})
    )
    _patched(monkeypatch, [], stream=stream)

    result = runner.invoke(app, ["ask", "q", "--token", "t", "--strategy", "graph"])

    assert result.exit_code == 0, result.output
    assert "Platform Team MEMBER_OF Engineering" in result.stdout
    assert "edge_not_in_subgraph" in result.stdout


# ---------------------------------------------------------------------------
# ragfabric graph merges list | show | undo
# ---------------------------------------------------------------------------


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import ragfabric_core.db.session as session_module
    from ragfabric_core import runtime

    url = f"sqlite:///{tmp_path / 'cli-graph.db'}"
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text("embeddings:\n  provider: offline\n  dim: 16\ncache:\n  kind: memory\n")
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(cfg))
    runtime.reset_config()
    engine = create_engine(url, connect_args={"check_same_thread": False})
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", factory)
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    yield factory
    runtime.reset_config()
    engine.dispose()


def _seed_merges(factory, pairs: list[tuple[str, str]]) -> list[int]:
    """One alias merge per (canonical name, alias) pair; returns the merge ids."""
    from ragfabric_core.graph.contracts import normalise
    from ragfabric_core.graph.resolve import resolve_entities
    from ragfabric_core.models.document import Chunk, Collection, Document
    from ragfabric_core.models.graph import Entity, EntitySource

    with factory() as db:
        collection = Collection(name="c")
        db.add(collection)
        db.flush()
        document = Document(
            filename="a.txt", format="txt", collection_id=collection.id, status="ready"
        )
        db.add(document)
        db.flush()
        index = 0
        for canonical, alias in pairs:
            for name, aliases in ((canonical, [alias]), (alias, [])):
                chunk = Chunk(
                    document_id=document.id,
                    collection_id=collection.id,
                    chunk_index=index,
                    text=f"{name} is here",
                    embedding=[0.0],
                )
                index += 1
                db.add(chunk)
                db.flush()
                entity = Entity(
                    name=name,
                    normalized_name=normalise(name),
                    entity_type="person",
                    aliases=aliases,
                )
                db.add(entity)
                db.flush()
                db.add(EntitySource(entity_id=entity.id, chunk_id=chunk.id, confidence=0.9))
        db.flush()
        report = resolve_entities(db, None, similarity_threshold=0.9)
        db.commit()
        return [merge.merge_id for merge in report.merges]


def _entity_names(factory) -> set[str]:
    from ragfabric_core.models.graph import Entity

    with factory() as db:
        return {entity.name for entity in db.query(Entity).all()}


def test_the_cli_lists_and_shows_merges(env):
    [merge_id] = _seed_merges(env, [("Robert Sharma", "Bob Sharma")])

    listed = runner.invoke(app, ["graph", "merges", "list"])
    assert listed.exit_code == 0, listed.output
    assert str(merge_id) in listed.stdout
    assert "alias" in listed.stdout
    assert "Bob Sharma" in listed.stdout
    assert "Robert Sharma" in listed.stdout

    shown = runner.invoke(app, ["graph", "merges", "show", str(merge_id)])
    assert shown.exit_code == 0, shown.output
    assert "Bob Sharma" in shown.stdout
    assert '"normalized_alias": "bob sharma"' in shown.stdout


def test_the_cli_can_undo_a_merge(env):
    [merge_id] = _seed_merges(env, [("Robert Sharma", "Bob Sharma")])
    assert _entity_names(env) == {"Robert Sharma"}

    result = runner.invoke(app, ["graph", "merges", "undo", str(merge_id)])

    assert result.exit_code == 0, result.output
    assert _entity_names(env) == {"Robert Sharma", "Bob Sharma"}
    for field in (
        "restored_entity_id",
        "survivor_entity_id",
        "moved_relationship_ids",
        "shared_relationship_ids",
        "restored_relationship_ids",
        "unrestored_relationship_ids",
        "changed_chunk_ids",
        "restored_without_sources",
    ):
        assert field in result.stdout
    assert "restored_without_sources: False" in result.stdout
    listed = runner.invoke(app, ["graph", "merges", "list"])
    assert str(merge_id) not in listed.stdout.split()


def test_undo_reports_the_blocking_merges_and_exits_non_zero(env):
    first, second = _seed_merges(
        env, [("Robert Sharma", "Bob Sharma"), ("Alice Wong", "Ally Wong")]
    )

    result = runner.invoke(app, ["graph", "merges", "undo", str(first)])

    assert result.exit_code == 1, result.output
    assert f"blocked by later merges: {second}" in result.output
    # Nothing was written: both merges are still in place.
    assert _entity_names(env) == {"Robert Sharma", "Alice Wong"}


def test_undo_of_an_unknown_merge_exits_non_zero(env):
    result = runner.invoke(app, ["graph", "merges", "undo", "999"])
    assert result.exit_code == 1
    assert "999" in result.output
