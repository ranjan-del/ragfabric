"""``ragfabric ingest`` progress on a terminal, and byte identical output when piped."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ragfabric_cli.commands import ingest as ingest_module
from ragfabric_cli.main import app
from ragfabric_cli.ui import console

runner = CliRunner()

# Stored from the pre-progress implementation: one line per file, then the summary.
_EXPECTED_PLAIN = "a.txt: ready (1 chunks)\nb.md: ready (1 chunks)\n2 ingested, 0 failed\n"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import ragfabric_core.db.session as session_module
    from ragfabric_core import runtime

    url = f"sqlite:///{tmp_path / 'progress.db'}"
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text(
        "embeddings:\n  provider: offline\n  dim: 16\ncache:\n  kind: memory\n"
        f"ingestion:\n  uploads_dir: {tmp_path / 'blobs'}\n"
    )
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(cfg))
    monkeypatch.setenv("JWT_SECRET", "cli-test-secret")
    monkeypatch.setenv("COLUMNS", "100")
    runtime.reset_config()
    engine = create_engine(url, connect_args={"check_same_thread": False})
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(
        session_module, "SessionLocal", sessionmaker(bind=engine, autoflush=False, autocommit=False)
    )
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("annual leave is twelve days")
    (docs / "b.md").write_text("# Rollout\nkubernetes rollout guide")
    yield docs
    runtime.reset_config()


def test_plain_output_is_byte_identical(env, monkeypatch):
    monkeypatch.setattr(console, "is_rich", lambda stream=None: False)
    result = runner.invoke(app, ["ingest", str(env), "--collection", "handbook"])
    assert result.exit_code == 0, result.stdout
    assert result.stdout == _EXPECTED_PLAIN


def test_rich_run_completes_and_prints_summary(env, monkeypatch):
    monkeypatch.setattr(console, "is_rich", lambda stream=None: True)
    result = runner.invoke(app, ["ingest", str(env), "--collection", "handbook"])
    assert result.exit_code == 0, result.stdout
    assert "2 ingested, 0 failed" in result.stdout
    # The per-file lines belong to piped output; the bar replaces them on a terminal.
    assert "a.txt: ready" not in result.stdout


def test_ingest_files_returns_counts_and_calls_on_file(env, monkeypatch):
    monkeypatch.setattr(console, "is_rich", lambda stream=None: False)
    seen: list[tuple[str, str]] = []
    files = sorted(env.iterdir())
    ingested, failed = ingest_module.ingest_files(
        files,
        collection="handbook",
        owner=None,
        on_file=lambda path, doc: seen.append((path.name, doc.status)),
    )
    assert (ingested, failed) == (2, 0)
    assert seen == [("a.txt", "ready"), ("b.md", "ready")]


@pytest.mark.parametrize("rich", [False, True])
def test_failing_file_exits_one_and_shows_error(env, monkeypatch, rich):
    monkeypatch.setattr(console, "is_rich", lambda stream=None: rich)

    def fake_ingest(db, *, filename, data, collection_id, owner_id):
        bad = filename == "b.md"
        return SimpleNamespace(
            status="failed" if bad else "ready",
            num_chunks=0 if bad else 1,
            error="parse exploded" if bad else None,
        )

    monkeypatch.setattr(ingest_module, "ingest_document", fake_ingest)
    result = runner.invoke(app, ["ingest", str(env)])
    assert result.exit_code == 1
    assert "b.md: failed (0 chunks) parse exploded" in result.stdout
    assert "1 ingested, 1 failed" in result.stdout


@pytest.mark.parametrize("rich", [False, True])
def test_quiet_prints_nothing_but_still_calls_on_file(env, monkeypatch, capsys, rich):
    monkeypatch.setattr(console, "is_rich", lambda stream=None: rich)
    seen: list[str] = []
    result = ingest_module.ingest_files(
        sorted(env.iterdir()),
        collection=None,
        owner=None,
        on_file=lambda path, doc: seen.append(path.name),
        quiet=True,
    )
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert result == (2, 0)
    assert seen == ["a.txt", "b.md"]


def test_a_file_whose_content_does_not_match_its_extension_is_skipped_with_the_reason(
    env, monkeypatch
):
    """The CLI applies the same validation as the API, so a fake PDF never
    reaches the parser and the run says why."""
    monkeypatch.setattr(console, "is_rich", lambda stream=None: False)
    (env / "fake.pdf").write_bytes(b"this is not a pdf")
    seen: list[str] = []
    result = runner.invoke(app, ["ingest", str(env)])
    assert result.exit_code == 1
    assert "fake.pdf: rejected (File is named .pdf but is not a PDF" in result.stdout
    assert "2 ingested, 1 failed" in result.stdout
    ingested, failed = ingest_module.ingest_files(
        [env / "fake.pdf"],
        collection=None,
        owner=None,
        on_file=lambda path, doc: seen.append(path.name),
        quiet=True,
    )
    assert (ingested, failed) == (0, 1)
    assert seen == []  # no Document was created, so on_file has nothing to report
