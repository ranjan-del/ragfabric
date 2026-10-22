"""Export and import on real PostgreSQL with the shipped corpus and its 24 questions.

Needs RAGFABRIC_TEST_DATABASE_URL. The answers are scripted, the judge is the lexical one, the
embeddings are the offline provider's: no model is called.
"""

import json
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typer.testing import CliRunner

from ragfabric_cli.main import app
from ragfabric_core.evaluation.dataset import shipped_questions
from ragfabric_core.evaluation.metrics import retrieval_scores
from ragfabric_core.evaluation.target import ContextItem

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL"),
]

runner = CliRunner()


@pytest.fixture()
def pg(tmp_path, monkeypatch):
    import ragfabric_core.db.session as session_module
    from ragfabric_core import runtime
    from ragfabric_core.db.migrate import downgrade, upgrade
    from ragfabric_core.models.index import EMBEDDING_DIM

    downgrade(URL)
    upgrade(URL)
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text(
        f"llm:\n  provider: offline\nembeddings:\n  provider: offline\n  dim: {EMBEDDING_DIM}\n"
        f"cache:\n  kind: memory\ningestion:\n  uploads_dir: {tmp_path / 'blobs'}\n"
    )
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(cfg))
    monkeypatch.setenv("JWT_SECRET", "pg-test-secret")
    runtime.reset_config()
    engine = create_engine(URL)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(
        session_module, "SessionLocal", sessionmaker(bind=engine, autoflush=False, autocommit=False)
    )
    yield tmp_path
    runtime.reset_config()
    engine.dispose()


def test_export_then_import_stores_scored_rows_with_provenance(pg):
    assert runner.invoke(app, ["eval", "corpus"]).exit_code == 0
    prompts = pg / "prompts.jsonl"
    exported = runner.invoke(
        app, ["eval", "export-prompts", "--strategy", "traditional", "--out", str(prompts)]
    )
    assert exported.exit_code == 0, exported.output
    header, *records = [json.loads(x) for x in prompts.read_text().splitlines()]
    assert len(records) == 24 and header["collection"] == "ragfabric-eval"
    assert header["question_set"] == shipped_questions().name

    scripted = {}
    for r in records[:-1]:
        scripted[r["id"]] = {"id": r["id"], "answer": f"{r['passages'][0]['text'][:70]} [1]"}
    last = records[-1]
    scripted[last["id"]] = {
        "id": last["id"],
        "answer": "I could not find an answer to that in the documents provided.",
        "input_tokens": 99,
    }
    answers = pg / "answers.jsonl"
    answers.write_text("\n".join(json.dumps(v) for v in scripted.values()))

    result = runner.invoke(
        app,
        ["eval", "import-answers", "--prompts", str(prompts), "--answers", str(answers),
         "--label", "hosted model, alias A", "--batch", "ext-pg", "--judge", "lexical", "--json"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output

    from ragfabric_core.evaluation.store import batch_runs
    from ragfabric_core.runtime import get_session_factory

    with get_session_factory()() as db:
        ((run, rows),) = batch_runs(db, "ext-pg")
        assert run.strategy == "external:hosted-model-alias-a"
        assert run.judge_model == "lexical" and len(rows) == 24
        meta = run.summary["meta"]
        assert meta["answered_by"] == "hosted model, alias A"
        assert meta["answer_source"] == "external answers file"
        assert meta["export_date"] == header["exported_at"]
        assert len(meta["prompt_file_sha256"]) == 64
        assert run.summary["errors"] == 0 and run.summary["refusals"] == 1
        by_id = {r.question_id: r for r in rows}
        questions = {q.id: q for q in shipped_questions().questions}
        for record in records:
            contexts = [
                ContextItem(rank=p["number"], document=p["document"], text=p["text"])
                for p in record["passages"]
            ]
            expected = retrieval_scores(contexts, questions[record["id"]].expected_sources)
            row = by_id[record["id"]]
            assert (row.hit, row.recall, row.reciprocal_rank, row.precision) == (
                expected.hit,
                expected.recall,
                expected.reciprocal_rank,
                expected.precision,
            )
            assert row.correctness is not None
            assert row.question == questions[record["id"]].question
        assert by_id[records[0]["id"]].faithfulness is not None
        assert by_id[last["id"]].details["input_tokens"] == 99
        assert by_id[records[0]["id"]].details["input_tokens"] is None
