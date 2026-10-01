import json

import pytest
from typer.testing import CliRunner

from ragfabric_cli.main import app
from ragfabric_core import diagnostics
from ragfabric_core.db import migrate

runner = CliRunner()


@pytest.fixture
def env(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'd.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    (tmp_path / "ragfabric.yaml").write_text(
        "llm:\n  provider: offline\nembeddings:\n  provider: offline\n  dim: 16\n"
    )
    return url


def test_json_parses_and_exit_is_one_when_a_check_fails(env):
    result = runner.invoke(app, ["doctor", "--json", "--no-network"])
    rows = json.loads(result.stdout)
    assert {"name", "status", "detail", "fix"} <= set(rows[0])
    assert any(r["status"] == "fail" for r in rows)
    assert result.exit_code == 1


def test_exit_is_zero_when_nothing_fails(env):
    migrate.upgrade(env)
    result = runner.invoke(app, ["doctor", "--json", "--no-network"])
    assert result.exit_code == 0, result.stdout
    assert not any(r["status"] == "fail" for r in json.loads(result.stdout))


def test_plain_output_has_a_line_per_check_with_status_and_fix(env, tmp_path):
    (tmp_path / "ragfabric.yaml").write_text("llm:\n  provider: ollama\n")
    result = runner.invoke(app, ["doctor", "--no-network"])
    lines = result.stdout.splitlines()
    assert any(line.startswith("fail") and "migrations" in line for line in lines)
    assert any("fix: ragfabric db upgrade" in line for line in lines)
    assert any(line.startswith("skip") and "llm" in line for line in lines)


def test_no_network_makes_no_provider_call(env, monkeypatch, tmp_path):
    (tmp_path / "ragfabric.yaml").write_text(
        "llm:\n  provider: ollama\nembeddings:\n  provider: ollama\n"
    )

    def boom(*args, **kwargs):
        raise AssertionError("provider was built")

    monkeypatch.setattr(diagnostics, "build_llm_provider", boom)
    monkeypatch.setattr(diagnostics, "build_embedding_provider", boom)
    result = runner.invoke(app, ["doctor", "--no-network", "--json"])
    assert result.exit_code in (0, 1)
    rows = {r["name"]: r for r in json.loads(result.stdout)}
    assert rows["llm"]["status"] == "skip"
    assert rows["embeddings"]["status"] == "skip"


def test_help_has_examples():
    assert "Examples:" in runner.invoke(app, ["doctor", "--help"]).stdout


def test_offline_is_labelled_even_with_no_network(env):
    rows = {
        r["name"]: r
        for r in json.loads(runner.invoke(app, ["doctor", "--no-network", "--json"]).stdout)
    }
    assert rows["llm"]["status"] == "warn"
    assert rows["llm"]["detail"] == "offline mode: answers are extractive"
