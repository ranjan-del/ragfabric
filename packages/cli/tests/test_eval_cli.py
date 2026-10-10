"""``ragfabric eval``, in process on SQLite with the offline provider."""

import json

import pytest
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import ragfabric_core.db.session as session_module
    from ragfabric_core import runtime

    url = f"sqlite:///{tmp_path / 'eval.db'}"
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text(
        "llm:\n  provider: offline\nembeddings:\n  provider: offline\n  dim: 16\n"
        f"cache:\n  kind: memory\ningestion:\n  uploads_dir: {tmp_path / 'blobs'}\n"
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
    yield tmp_path
    runtime.reset_config()


def test_corpus_loads_eight_documents_once(env):
    first = runner.invoke(app, ["eval", "corpus"])
    assert first.exit_code == 0, first.output
    assert "ragfabric-eval: 8 ingested, 0 already loaded, 0 failed" in first.output
    again = runner.invoke(app, ["eval", "corpus"])
    assert "0 ingested, 8 already loaded" in again.output


def test_run_without_the_corpus_says_how_to_load_it(env):
    result = runner.invoke(app, ["eval", "run", "--strategy", "traditional"])
    assert result.exit_code == 1 and "ragfabric eval corpus" in result.output


def test_a_run_stores_both_targets_and_writes_the_report(env):
    assert runner.invoke(app, ["eval", "corpus"]).exit_code == 0
    out = env / "latest.md"
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--strategy",
            "traditional",
            "--strategy",
            "vectorless",
            "--strategy",
            "graph",
            "--category",
            "exact_match",
            "--report",
            str(out),
            "--batch",
            "b1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3 questions x 2 targets, judge lexical" in result.output
    assert "skipped graph:" in result.output
    text = out.read_text()
    assert "| traditional | 3 |" in text and "| vectorless | 3 |" in text
    assert "## Skipped targets" in text

    listed = runner.invoke(app, ["eval", "list", "--json"])
    runs = json.loads(listed.output)
    assert sorted(r["target"] for r in runs) == ["graph", "traditional", "vectorless"]
    run_id = next(r["id"] for r in runs if r["target"] == "vectorless")
    shown = json.loads(runner.invoke(app, ["eval", "show", str(run_id), "--json"]).output)
    assert len(shown["results"]) == 3
    assert {r["question_type"] for r in shown["results"]} == {"exact_match"}
    # BW-7731 is an identifier: the lexical strategy finds the runbook first.
    q16 = next(r for r in shown["results"] if r["question_id"] == "q-016")
    assert q16["hit"] is True and q16["details"]["contexts"][0]["document"] == "it-runbook.md"

    rerendered = runner.invoke(app, ["eval", "report"])
    assert rerendered.exit_code == 0 and "batch" in rerendered.output.lower()
    assert "| vectorless | 3 |" in rerendered.output


def test_json_output_parses(env):
    runner.invoke(app, ["eval", "corpus"])
    result = runner.invoke(
        app,
        ["eval", "run", "--strategy", "vectorless", "--category", "relationship", "--json"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["runs"][0]["summary"]["questions"] == 3


def test_a_bad_questions_file_names_the_field(env):
    runner.invoke(app, ["eval", "corpus"])
    bad = env / "q.json"
    bad.write_text(
        json.dumps(
            {"name": "x", "questions": [{"id": "a", "question": "q", "question_type": "trivia"}]}
        )
    )
    result = runner.invoke(app, ["eval", "run", "--questions", str(bad)])
    assert result.exit_code == 1 and "question_type" in result.output


def test_an_unknown_category_or_target_is_refused(env):
    runner.invoke(app, ["eval", "corpus"])
    assert "unknown category" in runner.invoke(app, ["eval", "run", "--category", "x"]).output
    result = runner.invoke(app, ["eval", "run", "--strategy", "magic"])
    assert result.exit_code == 1 and "unknown target" in result.output


def test_list_and_report_on_an_empty_database(env):
    assert "no evaluation runs yet" in runner.invoke(app, ["eval", "list"]).output
    assert runner.invoke(app, ["eval", "report"]).exit_code == 1
    assert runner.invoke(app, ["eval", "show", "99"]).exit_code == 1
