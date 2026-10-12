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


def test_datasets_lists_every_importable_set_with_its_licence():
    result = runner.invoke(app, ["eval", "datasets"])
    assert result.exit_code == 0, result.output
    for name in ("scifact", "nfcorpus", "fiqa", "multihop-rag", "financebench"):
        assert name in result.output
    assert "CC BY-NC 4.0" in result.output
    parsed = json.loads(runner.invoke(app, ["eval", "datasets", "--json"]).output)
    assert {d["name"] for d in parsed} >= {"scifact", "financebench"}
    assert all(d["licence"] for d in parsed)


def test_import_passes_sampling_options_and_prints_the_licence(tmp_path, monkeypatch):
    from ragfabric_cli.commands import eval as eval_cmd

    seen = {}

    def fake_import(name, out, **kwargs):
        seen.update(name=name, out=out, **kwargs)
        return {
            "dataset": name,
            "licence": "CC BY 4.0",
            "counts": {"documents": 100, "questions": 12},
            "dropped": [{"what": "question", "count": 3, "reason": "not sampled"}],
        }

    monkeypatch.setattr(eval_cmd, "import_dataset", fake_import)
    result = runner.invoke(
        app,
        [
            "eval",
            "import",
            "scifact",
            "--limit",
            "100",
            "--questions",
            "12",
            "--seed",
            "3",
            "--out",
            str(tmp_path / "o"),
            "--cache-dir",
            str(tmp_path / "c"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["name"] == "scifact" and seen["limit"] == 100 and seen["questions"] == 12
    assert seen["seed"] == 3 and str(seen["cache_dir"]) == str(tmp_path / "c")
    assert "100 documents, 12 questions" in result.output
    assert "CC BY 4.0" in result.output
    assert "ragfabric eval corpus --path" in result.output


def test_import_default_out_names_the_dataset_and_limit(tmp_path, monkeypatch):
    from ragfabric_cli.commands import eval as eval_cmd

    seen = {}

    def fake_import(name, out, **kwargs):
        seen["out"] = out
        return {
            "dataset": name,
            "licence": "x",
            "counts": {"documents": 1, "questions": 1},
            "dropped": [],
        }

    monkeypatch.setattr(eval_cmd, "import_dataset", fake_import)
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["eval", "import", "fiqa", "--limit", "50"]).exit_code == 0
    assert str(seen["out"]) == "bench/fiqa-50"
    assert runner.invoke(app, ["eval", "import", "fiqa"]).exit_code == 0
    assert str(seen["out"]) == "bench/fiqa-full"


def test_import_of_an_unknown_dataset_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["eval", "import", "nope", "--out", str(tmp_path / "o")])
    assert result.exit_code == 1 and "scifact" in result.output


def test_corpus_path_ingests_an_imported_corpus_once(env):
    corpus = env / "imported" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "a.txt").write_text("Alpha document about apples.")
    (corpus / "b.txt").write_text("Beta document about bananas.")
    (corpus / ".hidden").write_text("not a document")
    first = runner.invoke(app, ["eval", "corpus", "--path", str(corpus), "--collection", "imp"])
    assert first.exit_code == 0, first.output
    assert "imp: 2 ingested, 0 already loaded, 0 failed" in first.output
    again = runner.invoke(app, ["eval", "corpus", "--path", str(corpus), "--collection", "imp"])
    assert "0 ingested, 2 already loaded" in again.output


def test_corpus_path_that_is_not_a_directory_fails(env):
    result = runner.invoke(app, ["eval", "corpus", "--path", str(env / "missing")])
    assert result.exit_code == 1 and "not a directory" in result.output


def _run(*extra):
    return runner.invoke(
        app, ["eval", "run", "--strategy", "traditional", "--category", "exact_match", *extra]
    )


def test_an_existing_batch_needs_resume(env):
    assert runner.invoke(app, ["eval", "corpus"]).exit_code == 0
    assert _run("--batch", "same").exit_code == 0
    again = _run("--batch", "same")
    assert again.exit_code == 1 and "--resume" in again.output
    resumed = _run("--batch", "same", "--resume", "--json")
    assert resumed.exit_code == 0, resumed.output
    runs = json.loads(resumed.output)["runs"]
    assert len(runs) == 1 and runs[0]["summary"]["questions"] == 3


def test_retrieval_only_and_judge_metric_are_recorded(env):
    assert runner.invoke(app, ["eval", "corpus"]).exit_code == 0
    out = _run("--batch", "ro", "--retrieval-only", "--json")
    assert out.exit_code == 0, out.output
    summary = json.loads(out.output)["runs"][0]["summary"]
    assert summary["meta"]["retrieval_only"] is True
    assert summary["metrics"]["correctness"] is None and summary["metrics"]["hit"] is not None
    jm = _run("--batch", "jm", "--judge-metric", "correctness", "--json")
    assert jm.exit_code == 0, jm.output
    meta = json.loads(jm.output)["runs"][0]["summary"]["meta"]
    assert meta["judge_metrics"] == ["correctness"]
    bad = _run("--batch", "jm2", "--judge-metric", "speed")
    assert bad.exit_code == 1 and "faithfulness" in bad.output
