from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragfabric_cli import templates
from ragfabric_cli.main import app
from ragfabric_core.ingest.parser import parse

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", ["env.example", "ragfabric.example.yaml", "docker-compose.yml"])
def test_every_template_exists(name):
    assert templates.template_path(name).is_file()


def test_unknown_template_names_itself():
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        templates.template_path("nope.txt")


def test_samples_parse_with_the_core_parser():
    paths = templates.sample_paths()
    assert [p.name for p in paths] == ["handbook.md", "leave-policy.md", "team.md"]
    for p in paths:
        text = parse(p.name, p.read_bytes())
        assert text.strip()
        assert len(text.splitlines()) < 40
        assert "\u2014" not in text
    leave = parse("leave-policy.md", paths[1].read_bytes())
    assert "up to 10 days of unused annual leave carry forward" in leave
    team = parse("team.md", paths[2].read_bytes())
    for token in ("Ravi Sharma", "Asha Rao", "Platform Team"):
        assert token in team
    assert "annual leave" in templates.SAMPLE_QUESTION


def test_packaged_yaml_matches_the_repo_root_copy():
    root = (REPO_ROOT / "ragfabric.example.yaml").read_text()
    assert templates.template_path("ragfabric.example.yaml").read_text() == root


def test_packaged_env_matches_the_repo_root_copy():
    root = (REPO_ROOT / ".env.example").read_text()
    assert templates.template_path("env.example").read_text() == root


def test_packaged_compose_has_only_postgres_and_redis():
    import yaml

    doc = yaml.safe_load(templates.template_path("docker-compose.yml").read_text())
    assert set(doc["services"]) == {"postgres", "redis"}


def test_init_runs_in_an_empty_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.stdout
    assert "wrote .env" in result.stdout and "wrote ragfabric.yaml" in result.stdout
    assert (tmp_path / ".env").is_file() and (tmp_path / "ragfabric.yaml").is_file()


def test_init_keeps_existing_files_unless_forced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init"]).exit_code == 0
    (tmp_path / ".env").write_text("MINE=1\n")
    (tmp_path / "ragfabric.yaml").write_text("llm:\n  provider: offline\n")
    again = runner.invoke(app, ["init"])
    assert "already exists (use --force to overwrite)" in again.stdout
    assert (tmp_path / ".env").read_text() == "MINE=1\n"
    assert (tmp_path / "ragfabric.yaml").read_text() == "llm:\n  provider: offline\n"
    forced = runner.invoke(app, ["init", "--force"])
    assert forced.exit_code == 0, forced.stdout
    assert (tmp_path / ".env").read_text() != "MINE=1\n"
    assert "MINE" not in (tmp_path / ".env").read_text()


def test_the_sample_question_routes_to_traditional():
    """I5: the packaged sample question must not route to the agent (no model offline)."""
    from ragfabric_cli.templates import SAMPLE_QUESTION
    from ragfabric_core.router.signals import extract_signals, propose
    from ragfabric_core.strategies.base import StrategyName

    signals = extract_signals(SAMPLE_QUESTION, relation_types=["REPORTS_TO"])
    proposal = propose(signals, available=list(StrategyName))
    assert proposal.strategy is StrategyName.TRADITIONAL and proposal.decisive


def test_init_writes_dot_env_readable_only_by_its_owner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    runner.invoke(app, ["init"])
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600


def test_the_env_template_has_no_dead_embedding_dim():
    assert "EMBEDDING_DIM" not in templates.template_path("env.example").read_text()


def test_write_private_lives_in_envfile(tmp_path):
    from ragfabric_cli.envfile import write_private

    path = tmp_path / "x"
    path.write_text("old")
    path.chmod(0o644)
    write_private(path, b"new")
    assert path.read_bytes() == b"new" and path.stat().st_mode & 0o777 == 0o600
