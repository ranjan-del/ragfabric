from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from ragfabric_core import runtime

    url = f"sqlite:///{tmp_path / 'cli2.db'}"
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text(
        f"embeddings:\n  provider: offline\n  dim: 16\ncache:\n  kind: memory\ningestion:\n  uploads_dir: {tmp_path / 'blobs'}\n"
    )
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(cfg))
    monkeypatch.setenv("JWT_SECRET", "cli-test-secret")
    runtime.reset_config()
    # Fresh engine bound to this database: the session module caches settings at import.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import ragfabric_core.db.session as session_module

    engine = create_engine(url, connect_args={"check_same_thread": False})
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(
        session_module, "SessionLocal", sessionmaker(bind=engine, autoflush=False, autocommit=False)
    )
    from ragfabric_core.store.vector_store import get_store

    get_store().clear()
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    yield tmp_path
    runtime.reset_config()


def test_init_copies_examples_and_validates(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    for name in (".env.example", "ragfabric.example.yaml"):
        (tmp_path / name).write_text((root / name).read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / ".env").exists() and (tmp_path / "ragfabric.yaml").exists()
    assert "configuration ok" in result.stdout
    again = runner.invoke(app, ["init"])
    assert "already exists" in again.stdout


def test_users_groups_grants_keys_flow(env):
    assert (
        runner.invoke(
            app, ["users", "create", "--email", "a@x.io", "--password", "password123"]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            [
                "users",
                "create",
                "--email",
                "root@x.io",
                "--password",
                "password123",
                "--role",
                "admin",
            ],
        ).exit_code
        == 0
    )
    listed = runner.invoke(app, ["users", "list"])
    assert "a@x.io" in listed.stdout and "root@x.io" in listed.stdout
    assert runner.invoke(app, ["groups", "create", "hr"]).exit_code == 0
    assert runner.invoke(app, ["groups", "add-member", "hr", "a@x.io"]).exit_code == 0
    assert "hr" in runner.invoke(app, ["groups", "list"]).stdout
    from ragfabric_cli.commands.common import collection_by_name, session

    with session() as db:
        collection_by_name(db, "policies", create=True)
        db.commit()
    assert (
        runner.invoke(
            app,
            ["grants", "add", "--group", "hr", "--collection", "policies", "--permission", "read"],
        ).exit_code
        == 0
    )
    assert "policies" in runner.invoke(app, ["grants", "list"]).stdout
    created = runner.invoke(
        app,
        [
            "keys",
            "create",
            "--name",
            "ci",
            "--user",
            "a@x.io",
            "--collection",
            "policies",
            "--rate-limit",
            "10",
        ],
    )
    assert created.exit_code == 0 and "rf_" in created.stdout
    listed = runner.invoke(app, ["keys", "list"])
    assert (
        "ci" in listed.stdout
        and "rf_" in listed.stdout
        and created.stdout.split("rf_")[1].strip().split()[0] not in listed.stdout
    )
    assert runner.invoke(app, ["keys", "revoke", "1"]).exit_code == 0
    assert runner.invoke(app, ["users", "set-role", "a@x.io", "admin"]).exit_code == 0
    assert runner.invoke(app, ["users", "deactivate", "a@x.io"]).exit_code == 0
    assert runner.invoke(app, ["users", "set-role", "nobody@x.io", "admin"]).exit_code == 1


def test_ingest_directory_creates_collection_and_indexes(env):
    docs = env / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("annual leave is twelve days")
    (docs / "b.md").write_text("# Rollout\nkubernetes rollout guide")
    (docs / "skip.xyz").write_text("ignored")
    result = runner.invoke(app, ["ingest", str(docs), "--collection", "handbook"])
    assert result.exit_code == 0, result.stdout
    assert (
        "a.txt: ready" in result.stdout
        and "2 ingested" in result.stdout
        and "skip.xyz" not in result.stdout
    )
    from ragfabric_cli.commands.common import session
    from ragfabric_core.models.document import Document
    from ragfabric_core.models.index import ChunkEmbedding

    with session() as db:
        assert db.query(Document).count() == 2 and db.query(ChunkEmbedding).count() >= 2


def test_worker_refuses_inline_mode(env):
    result = runner.invoke(app, ["worker", "--once"])
    assert result.exit_code == 1 and "indexing: queue" in result.stdout


def test_worker_stores_carry_the_active_embedding_model(env):
    # Running the worker command end to end needs Redis (build_queue only offers a
    # RedisJobQueue in queue mode), which unit tests must not depend on. _stores() is
    # the seam the command itself uses to build the vector store, so it is asserted
    # on directly rather than through a live worker loop.
    from ragfabric_cli.commands.worker import _stores

    vector_store, _ = _stores()
    assert vector_store.model == "hashing-16"


def test_reindex_reports_the_number_of_chunks_re_embedded(env):
    docs = env / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("annual leave is twelve days")
    assert runner.invoke(app, ["ingest", str(docs), "--collection", "handbook"]).exit_code == 0
    result = runner.invoke(app, ["reindex", "--yes", "--batch-size", "2"])
    assert result.exit_code == 0, result.stdout
    assert "re-embedded" in result.stdout


def test_reindex_without_yes_asks_for_confirmation(env):
    result = runner.invoke(app, ["reindex"], input="n\n")
    assert result.exit_code == 1
    assert "aborted" in result.stdout.lower()


def test_reindex_announces_the_model_and_dims_even_with_yes(env):
    # --yes is the unattended/CI path: nobody is there to see a prompt, so the
    # announcement of what is about to be overwritten must print regardless,
    # not only on the interactive branch that --yes skips.
    result = runner.invoke(app, ["reindex", "--yes"])
    assert result.exit_code == 0, result.stdout
    assert "hashing-16" in result.stdout and "16 dims" in result.stdout
