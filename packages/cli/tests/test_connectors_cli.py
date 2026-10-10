"""``ragfabric connectors`` (Phase 10, Task 5)."""

from __future__ import annotations

import os
import time

import pytest
from typer.testing import CliRunner

from ragfabric_cli.commands import connectors as connectors_module
from ragfabric_cli.main import app

runner = CliRunner()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import ragfabric_core.db.session as session_module
    from ragfabric_core import runtime

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    url = f"sqlite:///{tmp_path / 'connectors.db'}"
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text(
        "embeddings:\n  provider: offline\n  dim: 16\ncache:\n  kind: memory\n"
        f"ingestion:\n  uploads_dir: {tmp_path / 'blobs'}\n"
        "connectors:\n"
        f"  - name: inbox\n    kind: folder\n    path: {inbox}\n    collection: handbook\n"
        "    settle_seconds: 0\n    interval_seconds: 1\n"
        f"  - name: archive\n    kind: folder\n    path: {tmp_path / 'archive'}\n"
        "    on_delete: keep\n"
    )
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(cfg))
    monkeypatch.setenv("JWT_SECRET", "cli-test-secret")
    runtime.reset_config()
    engine = create_engine(url, connect_args={"check_same_thread": False})
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(
        session_module, "SessionLocal", sessionmaker(bind=engine, autoflush=False, autocommit=False)
    )
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    yield inbox
    runtime.reset_config()


def _drop(inbox, name, content=b"annual leave is twenty days"):
    path = inbox / name
    path.write_bytes(content)
    old = time.time() - 60
    os.utime(path, (old, old))


def test_list_shows_each_connector(env):
    result = runner.invoke(app, ["connectors", "list"])
    assert result.exit_code == 0, result.stdout
    assert "inbox\tfolder\t" in result.stdout and "collection handbook" in result.stdout
    assert "archive\tfolder\t" in result.stdout and "on_delete keep" in result.stdout


def test_sync_once_ingests_a_dropped_file_then_reports_it_unchanged(env):
    _drop(env, "leave.txt")
    result = runner.invoke(app, ["connectors", "sync", "inbox"])
    assert result.exit_code == 0, result.stdout
    assert "inbox: 1 added, 0 updated, 0 unchanged" in result.stdout
    result = runner.invoke(app, ["connectors", "sync", "inbox"])
    assert "inbox: 0 added, 0 updated, 1 unchanged" in result.stdout

    from ragfabric_core.models.document import Collection, Document
    from ragfabric_core.runtime import get_session_factory

    with get_session_factory()() as db:
        [doc] = db.query(Document).all()
        assert db.get(Collection, doc.collection_id).name == "handbook"


def test_an_unknown_name_is_an_error(env):
    result = runner.invoke(app, ["connectors", "sync", "nope"])
    assert result.exit_code == 1
    assert "no connector named nope" in result.stdout


def test_a_refused_file_is_reported_and_exits_1(env):
    _drop(env, "fake.pdf", b"not a pdf")
    result = runner.invoke(app, ["connectors", "sync", "inbox"])
    assert result.exit_code == 1
    assert "fake.pdf: File is named .pdf but is not a PDF" in result.stdout


def test_a_missing_folder_fails_that_connector_only(env):
    _drop(env, "leave.txt")
    result = runner.invoke(app, ["connectors", "sync"])  # both: archive's folder does not exist
    assert result.exit_code == 1
    assert "inbox: 1 added" in result.stdout
    assert "archive: failed: connector folder" in result.stdout


def test_an_owner_that_does_not_exist_is_an_error(env, monkeypatch):
    from ragfabric_core import runtime

    cfg = runtime.get_config()
    cfg.connectors[0].owner = "ghost@example.com"
    result = runner.invoke(app, ["connectors", "sync", "inbox"])
    assert result.exit_code == 1
    assert "user not found: ghost@example.com" in result.stdout


def test_watch_runs_passes_until_stopped(env, monkeypatch):
    _drop(env, "leave.txt")
    passes = []
    clock = [0.0]
    monkeypatch.setattr(connectors_module, "_now", lambda: clock[0])

    def fake_wait(stop, seconds):
        clock[0] += seconds
        passes.append(seconds)
        if len(passes) == 1:
            _drop(env, "more.txt", b"sick leave is ten days")
        else:
            stop.set()

    monkeypatch.setattr(connectors_module, "_wait", fake_wait)
    result = runner.invoke(app, ["connectors", "sync", "inbox", "--watch"])
    assert result.exit_code == 0, result.stdout
    assert "inbox: 1 added" in result.stdout
    assert result.stdout.count("1 added") == 2  # the second file on the second pass
    assert "stopped" in result.stdout
    assert passes[0] == 1  # interval_seconds of the inbox connector
