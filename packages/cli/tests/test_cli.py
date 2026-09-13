from sqlalchemy import create_engine, inspect
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()


def test_version_prints_the_core_version():
    from ragfabric_core import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0 and __version__ in result.stdout


def test_db_upgrade_and_downgrade_on_sqlite(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    assert "retrieval_runs" in inspect(create_engine(url)).get_table_names()
    assert runner.invoke(app, ["db", "downgrade"]).exit_code == 0
    assert inspect(create_engine(url)).get_table_names() == ["alembic_version"]


def test_config_validate_reports_active_implementations(tmp_path):
    p = tmp_path / "ragfabric.yaml"
    p.write_text(
        "llm:\n  provider: offline\nembeddings:\n  provider: offline\n  dim: 16\ncache:\n  kind: memory\n"
    )
    result = runner.invoke(app, ["config", "validate", "--path", str(p)])
    assert result.exit_code == 0, result.stdout
    assert "llm: offline" in result.stdout and "embeddings: offline (hashing-16)" in result.stdout
    assert "vector_store: pgvector" in result.stdout


def test_config_validate_fails_on_missing_provider_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = tmp_path / "ragfabric.yaml"
    p.write_text("llm:\n  provider: openai\n")
    result = runner.invoke(app, ["config", "validate", "--path", str(p)])
    assert result.exit_code == 1 and "OPENAI_API_KEY" in result.stdout


def test_config_validate_fails_on_typo(tmp_path):
    p = tmp_path / "ragfabric.yaml"
    p.write_text("llm:\n  provdier: openai\n")
    result = runner.invoke(app, ["config", "validate", "--path", str(p)])
    assert result.exit_code == 1 and "provdier" in result.stdout
