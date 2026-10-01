"""Unit tests for ragfabric quickstart: each step on its own, no server, no network."""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragfabric_cli import templates
from ragfabric_cli.commands import quickstart as qs
from ragfabric_cli.main import app
from ragfabric_core.config_file import load_config

runner = CliRunner()

SECRET = "sk-quickstart-secret-do-not-print"


@pytest.fixture
def no_ollama(monkeypatch):
    monkeypatch.setattr(qs, "_ollama_models", lambda timeout=2.0: None)


@pytest.fixture
def no_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def steps_after_config_are_noops(monkeypatch):
    """Stop the command after the config, model and database steps."""
    calls = []
    monkeypatch.setattr(qs, "_step_migrate", lambda ctx: calls.append("migrate"))
    monkeypatch.setattr(qs, "_step_ingest", lambda ctx: calls.append("ingest"))
    monkeypatch.setattr(qs, "_step_ask", lambda ctx: calls.append("ask"))
    return calls


# choose_model ---------------------------------------------------------------


def test_ollama_with_a_chat_and_an_embedding_model_is_chosen(monkeypatch):
    monkeypatch.setattr(
        qs, "_ollama_models", lambda timeout=2.0: ["llama3.2:3b", "nomic-embed-text:latest"]
    )
    choice = qs.choose_model(None, {"OPENAI_API_KEY": "sk-x"})
    assert choice.kind == "ollama"
    assert choice.llm["provider"] == "ollama" and choice.llm["model"] == "llama3.2:3b"
    assert choice.embeddings == {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dim": 768,
        "base_url": "http://localhost:11434/v1",
    }
    assert "Ollama" in choice.reason


def test_ollama_without_the_embedding_model_falls_through_and_says_what_to_pull(monkeypatch):
    monkeypatch.setattr(qs, "_ollama_models", lambda timeout=2.0: ["llama3.2:3b"])
    choice = qs.choose_model(None, {})
    assert choice.kind == "offline"
    assert "ollama pull nomic-embed-text" in choice.reason


def test_openai_key_gives_openai(no_ollama):
    choice = qs.choose_model(None, {"OPENAI_API_KEY": SECRET})
    assert choice.kind == "openai"
    assert choice.llm["provider"] == "openai" and choice.embeddings["provider"] == "openai"
    assert SECRET not in choice.reason


def test_anthropic_key_gives_anthropic_with_offline_embeddings_at_the_configured_dim(
    no_ollama, tmp_path
):
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text("embeddings:\n  provider: offline\n  dim: 384\n")
    choice = qs.choose_model(cfg, {"ANTHROPIC_API_KEY": SECRET})
    assert choice.kind == "anthropic"
    assert choice.llm["provider"] == "anthropic"
    assert choice.embeddings["provider"] == "offline" and choice.embeddings["dim"] == 384


def test_nothing_available_gives_offline(no_ollama):
    choice = qs.choose_model(None, {})
    assert choice.kind == "offline"
    assert choice.llm["provider"] == "offline" and choice.embeddings["provider"] == "offline"
    assert choice.embeddings["dim"] == 768
    assert "extractive" in choice.reason


def test_no_model_check_skips_the_ollama_probe(monkeypatch):
    def boom(timeout=2.0):
        raise AssertionError("probed")

    monkeypatch.setattr(qs, "_ollama_models", boom)
    assert qs.choose_model(None, {}, probe=False).kind == "offline"


# config patching (ruling R2) ------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "env"),
    [
        ("offline", {}),
        ("openai", {"OPENAI_API_KEY": "sk-x"}),
        ("anthropic", {"ANTHROPIC_API_KEY": "k"}),
    ],
)
def test_patched_template_loads_with_the_chosen_values_and_keeps_its_comments(
    tmp_path, no_ollama, kind, env
):
    choice = qs.choose_model(None, env)
    assert choice.kind == kind
    text = qs.patch_config(
        templates.template_path("ragfabric.example.yaml").read_text(), choice, cache_kind="memory"
    )
    path = tmp_path / "ragfabric.yaml"
    path.write_text(text)
    cfg = load_config(path)
    assert cfg.llm.provider == choice.llm["provider"]
    assert cfg.llm.model == choice.llm["model"]
    assert cfg.embeddings.provider == choice.embeddings["provider"]
    assert cfg.embeddings.dim == choice.embeddings["dim"]
    assert cfg.cache.kind == "memory"
    assert "# the chat model provider" in text
    assert "# openai | anthropic | ollama | offline" in text
    # Nothing outside the three sections moved.
    original = templates.template_path("ragfabric.example.yaml").read_text().splitlines()
    patched = text.splitlines()
    assert len(original) == len(patched)
    changed = [i for i, (a, b) in enumerate(zip(original, patched, strict=True)) if a != b]
    assert all(i < 40 for i in changed)


# the command, stopped after the database step -------------------------------


def test_an_existing_ragfabric_yaml_is_never_overwritten_without_force(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    original = b"llm:\n  provider: offline\nembeddings:\n  provider: offline\n  dim: 16\n"
    (tmp_path / "ragfabric.yaml").write_bytes(original)
    kept_env = b"JWT_SECRET=x\nDATABASE_URL=sqlite:///./mine.db\n"
    (tmp_path / ".env").write_bytes(kept_env)
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ragfabric.yaml").read_bytes() == original
    assert (tmp_path / ".env").read_bytes() == kept_env
    assert "--force" in result.output

    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes", "--force"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ragfabric.yaml").read_bytes() != original
    assert load_config(tmp_path / "ragfabric.yaml").llm.provider == "offline"


def test_offline_mode_is_labelled_with_the_upgrade_command(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "offline" in result.output and "extractive" in result.output
    assert "ollama pull llama3.2:3b" in result.output
    assert "ollama pull nomic-embed-text" in result.output
    env = (tmp_path / ".env").read_text()
    assert f"DATABASE_URL=sqlite:///{tmp_path / 'ragfabric.db'}" in env
    assert steps_after_config_are_noops == ["migrate", "ingest", "ask"]


def test_no_secret_reaches_the_output(
    tmp_path, monkeypatch, no_ollama, steps_after_config_are_noops
):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql+psycopg://u:hunter2pass@localhost:5432/db\n"
        f"OPENAI_API_KEY={SECRET}\n"
    )
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    # A .env with no RagFabric key is someone else's: the run stops (C2), masked.
    assert result.exit_code == 1, result.output
    assert "u:***@localhost" in result.output
    assert SECRET not in result.output
    assert "hunter2pass" not in result.output


def test_docker_unavailable_with_yes_falls_back_to_sqlite_and_says_so(
    tmp_path, monkeypatch, no_ollama, no_keys, steps_after_config_are_noops
):
    monkeypatch.setattr(qs, "_docker_available", lambda: False)

    def never(*args, **kwargs):
        raise AssertionError("docker compose must not run")

    monkeypatch.setattr(qs, "_compose_up", never)
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--docker", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Docker is not reachable" in result.output
    assert "falling back to SQLite" in result.output
    assert "sqlite:///" in (tmp_path / ".env").read_text()
    assert not (tmp_path / "docker-compose.yml").exists()


def test_docker_unavailable_without_yes_asks_and_stops_on_no(
    tmp_path, monkeypatch, no_ollama, no_keys, steps_after_config_are_noops
):
    monkeypatch.setattr(qs, "_docker_available", lambda: False)
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--docker"], input="n\n")
    assert result.exit_code == 1
    assert steps_after_config_are_noops == []


# free_port and temporary_server ---------------------------------------------


def test_free_port_is_bindable():
    port = qs.free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


FAKE_SERVER = textwrap.dedent(
    """
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
    """
)


def _fake_server(monkeypatch, tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    started: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        started.append(proc)
        return proc

    monkeypatch.setattr(
        qs, "_server_command", lambda port: [sys.executable, str(script), str(port)]
    )
    monkeypatch.setattr(qs.subprocess, "Popen", popen)
    return started


def _refused(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def test_temporary_server_terminates_the_process_when_the_body_raises(monkeypatch, tmp_path):
    started = _fake_server(monkeypatch, tmp_path)
    port = qs.free_port()
    with pytest.raises(RuntimeError, match="body failed"):
        with qs.temporary_server(tmp_path, port) as url:
            assert url == f"http://127.0.0.1:{port}"
            assert not _refused(port)
            raise RuntimeError("body failed")
    assert len(started) == 1 and started[0].poll() is not None
    assert _refused(port)


def test_temporary_server_terminates_on_keyboard_interrupt(monkeypatch, tmp_path):
    started = _fake_server(monkeypatch, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with qs.temporary_server(tmp_path, qs.free_port()):
            raise KeyboardInterrupt
    assert started[0].poll() is not None


def test_temporary_server_that_never_becomes_healthy_is_stopped(monkeypatch, tmp_path):
    started: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        started.append(proc)
        return proc

    monkeypatch.setattr(
        qs, "_server_command", lambda port: [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    monkeypatch.setattr(qs.subprocess, "Popen", popen)
    monkeypatch.setattr(qs, "HEALTH_TIMEOUT_S", 1.0)
    with pytest.raises(qs.QuickstartError, match="did not become healthy"):
        with qs.temporary_server(tmp_path, qs.free_port()):
            pass
    assert started[0].poll() is not None


def test_docker_reachable_writes_compose_and_a_masked_postgres_url(
    tmp_path, monkeypatch, no_ollama, no_keys, steps_after_config_are_noops
):
    started = []
    monkeypatch.setattr(qs, "_docker_available", lambda: True)
    monkeypatch.setattr(qs, "_compose_up", lambda dir: started.append(dir))
    monkeypatch.setattr(qs, "_wait_for_database", lambda url, timeout=60.0: None)
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--docker", "--yes"])
    assert result.exit_code == 0, result.output
    assert started == [tmp_path.resolve()]
    assert (tmp_path / "docker-compose.yml").is_file()
    env = (tmp_path / ".env").read_text()
    assert "DATABASE_URL=postgresql+psycopg://ragfabric:ragfabric@localhost:5432/ragfabric" in env
    assert "ragfabric:ragfabric@" not in result.output
    # PostgreSQL keeps the template's redis cache; SQLite is the only case forced to memory.
    assert load_config(tmp_path / "ragfabric.yaml").cache.kind == "redis"


# fix round 1: upgrade hint, resumability, credentials, next steps, port retry


def test_the_upgrade_hint_moves_the_config_aside_and_never_forces(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    hint = qs.upgrade_hint(tmp_path)
    assert hint == (
        "ollama pull llama3.2:3b && ollama pull nomic-embed-text && "
        f"cd {tmp_path} && mv ragfabric.yaml ragfabric.yaml.bak && ragfabric quickstart && "
        "ragfabric reindex --yes"
    )
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "mv ragfabric.yaml ragfabric.yaml.bak" in result.output
    assert "--force &&" not in result.output and "quickstart --force" not in result.output


def test_an_unpatched_template_left_by_a_partial_run_is_patched(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    template = templates.template_path("ragfabric.example.yaml").read_bytes()
    (tmp_path / "ragfabric.yaml").write_bytes(template)
    (tmp_path / ".env").write_text(
        f"JWT_SECRET=x\nDATABASE_URL=sqlite:///{tmp_path / 'ragfabric.db'}\n"
    )
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "unpatched template" in result.output
    cfg = load_config(tmp_path / "ragfabric.yaml")
    assert cfg.llm.provider == "offline" and cfg.cache.kind == "memory"


def test_an_edited_ragfabric_yaml_is_still_kept(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    edited = templates.template_path("ragfabric.example.yaml").read_bytes() + b"\n# mine\n"
    (tmp_path / "ragfabric.yaml").write_bytes(edited)
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ragfabric.yaml").read_bytes() == edited


def test_a_failed_sample_is_ingested_again(tmp_path, no_ollama, no_keys):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from ragfabric_core.db import migrate
    from ragfabric_core.models.document import Document

    (tmp_path / "ragfabric.yaml").write_text(
        "llm:\n  provider: offline\nembeddings:\n  provider: offline\n  dim: 768\n"
        "cache:\n  kind: memory\n"
    )
    db_url = f"sqlite:///{tmp_path / 'ragfabric.db'}"
    migrate.upgrade(db_url)
    engine = create_engine(db_url)
    with Session(engine) as db:
        db.add(Document(filename="team.md", status="failed", error="boom"))
        db.commit()
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    ctx.db_url = db_url
    qs._step_ingest(ctx)
    with Session(engine) as db:
        statuses = {
            (d.filename, d.status) for d in db.query(Document).filter(Document.status == "ready")
        }
    engine.dispose()
    assert statuses == {
        ("handbook.md", "ready"),
        ("leave-policy.md", "ready"),
        ("team.md", "ready"),
    }


class _Response:
    status_code = 200

    def json(self):
        return {"access_token": "jwt-not-a-secret-here"}


def test_a_kept_env_gets_the_key_command_and_never_a_key(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=sqlite:///x.db\n")
    before = env.read_bytes()
    monkeypatch.setattr(qs.httpx, "post", lambda *a, **k: _Response())

    def never(*args, **kwargs):
        raise AssertionError("no key may be created for a kept .env")

    monkeypatch.setattr(qs, "_create_api_key", never)
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    creds = qs._credentials(ctx, "http://127.0.0.1:1", "admin@example.com", "pw")
    assert creds == {"token": "jwt-not-a-secret-here"}
    assert env.read_bytes() == before
    out = capsys.readouterr().out
    assert (
        f"cd {tmp_path} && export RAGFABRIC_API_KEY="
        '"$(ragfabric keys create --name cli --user admin@example.com | tail -n 1)"'
    ) in out
    assert "jwt-not-a-secret-here" not in out


def test_a_written_env_gets_the_key_and_url_appended_and_the_key_is_not_printed(
    tmp_path, monkeypatch, capsys
):
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=sqlite:///x.db\n")
    monkeypatch.setattr(qs, "port_in_use", lambda host, port: False)
    monkeypatch.setattr(qs, "_create_api_key", lambda ctx, email: "rf_secretkeyvalue")
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    ctx.env_written = True
    assert qs._credentials(ctx, "http://127.0.0.1:1", "admin@example.com", "pw") == {
        "api_key": "rf_secretkeyvalue"
    }
    text = env.read_text()
    assert "RAGFABRIC_API_KEY=rf_secretkeyvalue\n" in text
    assert "RAGFABRIC_URL=http://127.0.0.1:8000\n" in text
    assert "rf_secretkeyvalue" not in capsys.readouterr().out


def test_next_steps_cd_first_and_drop_the_token(tmp_path):
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    ctx.key_in_env = True
    steps = [command for command, _ in qs._next_steps(ctx)]
    assert steps[0] == f"cd {tmp_path}"
    assert f'ragfabric ask "{templates.SAMPLE_QUESTION}"' in steps
    assert any(step.startswith("ragfabric serve") for step in steps)
    assert "ragfabric doctor" in steps
    assert "ragfabric ingest ./my-docs --recursive" in steps
    assert "ragfabric strategies" in steps
    assert not any("--token" in step for step in steps)


def test_a_port_taken_before_the_server_binds_is_retried_once(monkeypatch, tmp_path):
    from contextlib import ExitStack

    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    taken = tmp_path / "taken.py"
    taken.write_text(
        "import sys\nprint('ERROR: [Errno 48] error while attempting to bind: "
        "address already in use')\nsys.exit(1)\n"
    )
    commands = iter([[sys.executable, str(taken)], None])

    def command(port):
        first = next(commands)
        return first if first is not None else [sys.executable, str(script), str(port)]

    monkeypatch.setattr(qs, "_server_command", command)
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    with ExitStack() as stack:
        url = qs._start_server(ctx, stack)
        port = int(url.rsplit(":", 1)[1])
        assert not _refused(port)
    assert _refused(port)


def test_alembic_info_lines_are_kept_out_of_the_migrate_step(tmp_path, capfd):
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    ctx.db_url = f"sqlite:///{tmp_path / 'ragfabric.db'}"
    qs._step_migrate(ctx)
    captured = capfd.readouterr()
    assert "Running upgrade" not in captured.err + captured.out
    import logging

    assert logging.root.manager.disable == logging.NOTSET


def test_read_env_file_ignores_comments_and_blanks_and_strips_quotes(tmp_path):
    from ragfabric_cli.envfile import read_env_file

    path = tmp_path / ".env"
    path.write_text("# a comment\n\nA=1\nB = 'two'\nexport C=\"three\"\nnot a pair\nD=\n# E=no\n")
    assert read_env_file(path) == {"A": "1", "B": "two", "C": "three", "D": ""}
    assert read_env_file(tmp_path / "missing") == {}


class _RecordingClient:
    """Stands in for the SDK client: records where ask would connect, then stops."""

    seen: list[dict] = []

    def __init__(self, base_url, token=None, api_key=None, **kwargs):
        _RecordingClient.seen.append({"url": base_url, "token": token, "api_key": api_key})

    def ask(self, *args, **kwargs):
        raise RuntimeError("stop here")

    def close(self):
        pass


@pytest.fixture
def ask_env(tmp_path, monkeypatch):
    from ragfabric_cli.commands import ask as ask_module

    for name in ("RAGFABRIC_API_KEY", "RAGFABRIC_TOKEN", "RAGFABRIC_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ask_module, "Client", _RecordingClient)
    _RecordingClient.seen = []
    (tmp_path / ".env").write_text(
        '# quickstart\nRAGFABRIC_API_KEY="rf_fromdotenv"\nRAGFABRIC_URL=http://evil.example:9\n'
    )
    return tmp_path


def _asked(*args):
    result = runner.invoke(app, ["ask", "q", "--no-stream", *args])
    return result, (_RecordingClient.seen[-1] if _RecordingClient.seen else None)


def test_ask_with_nothing_set_takes_url_and_key_from_dot_env_together(ask_env):
    result, seen = _asked()
    assert result.exit_code == 1
    assert seen == {"url": "http://evil.example:9", "token": None, "api_key": "rf_fromdotenv"}
    assert "rf_fromdotenv" not in result.output


def test_ask_with_a_dot_env_key_and_no_url_uses_the_default_url(ask_env):
    (ask_env / ".env").write_text("RAGFABRIC_API_KEY=rf_fromdotenv\n")
    _, seen = _asked()
    assert seen == {"url": "http://localhost:8000", "token": None, "api_key": "rf_fromdotenv"}


def test_ask_with_an_env_key_never_uses_the_dot_env_url(ask_env, monkeypatch):
    monkeypatch.setenv("RAGFABRIC_API_KEY", "rf_real")
    _, seen = _asked()
    assert seen == {"url": "http://localhost:8000", "token": None, "api_key": "rf_real"}
    _, seen = _asked("--url", "http://127.0.0.1:7")
    assert seen["url"] == "http://127.0.0.1:7"


def test_ask_with_an_env_token_never_uses_the_dot_env_url_or_key(ask_env, monkeypatch):
    monkeypatch.setenv("RAGFABRIC_TOKEN", "jwt-real")
    _, seen = _asked()
    assert seen == {"url": "http://localhost:8000", "token": "jwt-real", "api_key": None}


def test_ask_with_an_api_key_flag_never_uses_the_dot_env_url(ask_env):
    _, seen = _asked("--api-key", "rf_flag")
    assert seen == {"url": "http://localhost:8000", "token": None, "api_key": "rf_flag"}


def test_ask_with_an_env_url_ignores_the_dot_env_key(ask_env, monkeypatch):
    monkeypatch.setenv("RAGFABRIC_URL", "http://127.0.0.1:9")
    result, seen = _asked()
    assert result.exit_code == 2 and seen is None
    assert "./.env was not read because RAGFABRIC_URL" in result.output


def test_ask_with_no_credentials_anywhere_exits_2(tmp_path, monkeypatch):
    for name in ("RAGFABRIC_API_KEY", "RAGFABRIC_TOKEN", "RAGFABRIC_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["ask", "q"]).exit_code == 2


# fix round 2: .env mode, quoting, key rotation


def test_quickstart_writes_dot_env_readable_only_by_its_owner(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600


def test_printed_commands_quote_a_directory_with_a_space(tmp_path):
    import shlex

    spaced = tmp_path / "my rag"
    spaced.mkdir()
    quoted = shlex.quote(str(spaced))
    assert f"cd {quoted} && mv ragfabric.yaml" in qs.upgrade_hint(spaced)
    ctx = qs.Context(dir=spaced, force=False, yes=True, docker=False, model_check=False)
    steps = [command for command, _ in qs._next_steps(ctx)]
    assert steps[0] == f"cd {quoted}"
    assert all(f"cd {spaced} " not in step for step in steps)
    command = qs._key_command(ctx, "o'brien@example.com")
    assert f"cd {quoted} && " in command
    assert shlex.quote("o'brien@example.com") in command
    # The whole command still parses as shell words.
    inner = command.split('"$(', 1)[1].rsplit(')"', 1)[0]
    assert shlex.split(inner)[:6] == ["ragfabric", "keys", "create", "--name", "cli", "--user"]
    assert shlex.split(inner)[6] == "o'brien@example.com"


# final fix wave: C1, an optional client package that is not installed ---------


@pytest.fixture
def hidden_packages(monkeypatch):
    """Make importlib.util.find_spec report the named top level packages as missing."""
    import importlib.util

    real = importlib.util.find_spec
    hidden: set[str] = set()

    def fake(name, *args, **kwargs):
        if name.split(".")[0] in hidden:
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    return hidden


def _ollama_up(monkeypatch):
    monkeypatch.setattr(
        qs, "_ollama_models", lambda timeout=2.0: ["llama3.2:3b", "nomic-embed-text:latest"]
    )


def test_ollama_running_without_the_openai_package_falls_through_to_offline(
    monkeypatch, hidden_packages
):
    _ollama_up(monkeypatch)
    hidden_packages.add("openai")
    choice = qs.choose_model(None, {})
    assert choice.kind == "offline"
    assert (
        "Ollama is running with llama3.2:3b and nomic-embed-text, but its client package "
        "(openai) is not installed"
    ) in choice.reason
    assert "pip install 'ragfabric[openai]'" in choice.reason
    assert "uv pip" not in choice.reason


def test_an_openai_key_without_the_openai_package_falls_through(no_ollama, hidden_packages):
    hidden_packages.add("openai")
    choice = qs.choose_model(None, {"OPENAI_API_KEY": SECRET, "ANTHROPIC_API_KEY": "k"})
    assert choice.kind == "anthropic"
    assert "OPENAI_API_KEY is set, but its client package (openai) is not installed" in (
        choice.reason
    )
    assert SECRET not in choice.reason


def test_an_anthropic_key_without_the_anthropic_package_ends_offline(no_ollama, hidden_packages):
    hidden_packages.add("anthropic")
    choice = qs.choose_model(None, {"ANTHROPIC_API_KEY": "k"})
    assert choice.kind == "offline"
    assert "client package (anthropic) is not installed" in choice.reason
    assert "pip install 'ragfabric[anthropic]'" in choice.reason


def test_the_upgrade_hint_starts_with_the_pip_install_while_openai_is_missing(
    tmp_path, hidden_packages
):
    hidden_packages.add("openai")
    assert qs.upgrade_hint(tmp_path).startswith("pip install 'ragfabric[openai]' && ollama pull")
    hidden_packages.discard("openai")
    assert qs.upgrade_hint(tmp_path).startswith("ollama pull")


# final fix wave: C2, a kept .env that is not RagFabric's ----------------------


def test_a_foreign_dot_env_stops_before_migrating_and_leaves_its_database_alone(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    elsewhere = tmp_path / "myapp"
    elsewhere.mkdir()
    foreign_db = elsewhere / "myapp.db"
    foreign_db.write_bytes(b"not a ragfabric database")
    work = tmp_path / "work"
    work.mkdir()
    (work / ".env").write_text(f"DATABASE_URL=sqlite:///{foreign_db}\nSECRET_KEY=django\n")
    result = runner.invoke(app, ["quickstart", "--dir", str(work), "--yes"])
    assert result.exit_code == 1
    assert f"the .env in {work} is not RagFabric's (DATABASE_URL sqlite:///{foreign_db})" in (
        result.output
    )
    assert "ragfabric quickstart --dir ./ragfabric" in result.output
    assert steps_after_config_are_noops == []
    assert foreign_db.read_bytes() == b"not a ragfabric database"


def test_yes_never_accepts_a_non_sqlite_url_from_a_kept_dot_env(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    (tmp_path / ".env").write_text(
        "JWT_SECRET=x\nDATABASE_URL=postgresql+psycopg://u:hunter2@db.example.com:5432/app\n"
    )
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 1
    assert "u:***@db.example.com" in result.output and "hunter2" not in result.output
    assert steps_after_config_are_noops == []


def test_a_ragfabric_dot_env_with_sqlite_inside_the_dir_continues(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    (tmp_path / ".env").write_text(
        "JWT_SECRET=x\nFIRST_ADMIN_EMAIL=admin@example.com\nDATABASE_URL=sqlite:///./ragfabric.db\n"
    )
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert steps_after_config_are_noops == ["migrate", "ingest", "ask"]


# final fix wave: I3, copyable next steps --------------------------------------

_BOX = set("─│┌┐└┘├┤┬┴┼╭╮╯╰═║╔╗╚╝")


def test_rich_next_steps_at_60_columns_have_no_ellipsis_and_no_box(tmp_path, monkeypatch, capsys):
    from ragfabric_cli.ui import console
    from ragfabric_cli.ui.panels import render_next_steps

    deep = tmp_path / ("a-rather-long-directory-name-" * 3) / "my-rag"
    deep.mkdir(parents=True)
    ctx = qs.Context(dir=deep, force=False, yes=True, docker=False, model_check=False)
    ctx.key_in_env = True
    ctx.choice = qs.choose_model(None, {}, probe=False)  # offline: the long upgrade hint
    monkeypatch.setattr(console, "is_rich", lambda stream=None: True)
    monkeypatch.setenv("COLUMNS", "60")
    console.get_console.cache_clear()
    try:
        render_next_steps(qs._next_steps(ctx))
    finally:
        console.get_console.cache_clear()
    out = capsys.readouterr().out
    assert "…" not in out
    assert not (_BOX & set(out))
    flat = "".join(out.split())
    assert "".join(f"cd {deep}".split()) in flat
    assert "".join(qs.upgrade_hint(None).split()) in flat


def test_next_steps_cd_once_and_drop_the_prefix(tmp_path):
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    steps = qs._next_steps(ctx)
    assert steps[0][0] == f"cd {tmp_path}"
    assert not any(" && ragfabric" in command for command, _ in steps[1:] if "mv " not in command)
    serve = next((c, why) for c, why in steps if c.startswith("ragfabric serve"))
    assert "leave running; use a second terminal for the rest" in serve[1]
    assert serve[1].endswith(f", in {tmp_path}")


def test_next_steps_serve_names_no_directory_when_it_is_the_current_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = qs.Context(dir=tmp_path.resolve(), force=False, yes=True, docker=False, model_check=False)
    serve = next(why for c, why in qs._next_steps(ctx) if c.startswith("ragfabric serve"))
    assert ", in " not in serve


def test_next_steps_have_no_cd_when_the_dir_is_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = qs.Context(dir=tmp_path.resolve(), force=False, yes=True, docker=False, model_check=False)
    assert not any(command.startswith("cd ") for command, _ in qs._next_steps(ctx))


# final fix wave: I6, --docker after an earlier SQLite quickstart --------------


@pytest.fixture
def fake_docker(monkeypatch):
    monkeypatch.setattr(qs, "_docker_available", lambda: True)
    monkeypatch.setattr(qs, "_compose_up", lambda dir: None)
    monkeypatch.setattr(qs, "_wait_for_database", lambda url, timeout=60.0: None)


def _sqlite_env(tmp_path) -> str:
    text = (
        "JWT_SECRET=x\nPOSTGRES_PASSWORD=ragfabric\n"
        f"DATABASE_URL=sqlite:///{tmp_path / 'ragfabric.db'}\nRAGFABRIC_API_KEY=rf_old\n"
    )
    (tmp_path / ".env").write_text(text)
    (tmp_path / ".env").chmod(0o600)
    return text


def test_docker_after_sqlite_updates_only_database_url_with_yes(
    tmp_path, no_ollama, no_keys, fake_docker, steps_after_config_are_noops
):
    before = _sqlite_env(tmp_path)
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--docker", "--yes"])
    assert result.exit_code == 0, result.output
    docker_url = "postgresql+psycopg://ragfabric:ragfabric@localhost:5432/ragfabric"
    after = (tmp_path / ".env").read_text()
    assert after == before.replace(f"sqlite:///{tmp_path / 'ragfabric.db'}", docker_url)
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600
    assert "rerun with --force" not in result.output
    assert steps_after_config_are_noops == ["migrate", "ingest", "ask"]


def test_docker_after_sqlite_declined_stops_before_migrating(
    tmp_path, no_ollama, no_keys, fake_docker, steps_after_config_are_noops
):
    before = _sqlite_env(tmp_path)
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--docker"], input="n\n")
    assert result.exit_code == 1
    assert (tmp_path / ".env").read_text() == before
    assert steps_after_config_are_noops == []
    assert "rerun with --force" not in result.output


def test_a_dot_env_key_not_active_in_this_database_is_replaced(tmp_path, monkeypatch, capsys):
    env = tmp_path / ".env"
    env.write_text("JWT_SECRET=x\nRAGFABRIC_API_KEY=rf_old\nRAGFABRIC_URL=http://127.0.0.1:8000\n")
    monkeypatch.setattr(qs, "_key_is_active", lambda ctx, key: False)
    monkeypatch.setattr(qs, "_create_api_key", lambda ctx, email: "rf_newsecretvalue")
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=True, model_check=False)
    assert qs._credentials(ctx, "http://127.0.0.1:1", "admin@example.com", "pw") == {
        "api_key": "rf_newsecretvalue"
    }
    assert env.read_text() == (
        "JWT_SECRET=x\nRAGFABRIC_API_KEY=rf_newsecretvalue\nRAGFABRIC_URL=http://127.0.0.1:8000\n"
    )
    assert "rf_newsecretvalue" not in capsys.readouterr().out


def test_key_is_active_reads_the_database_in_use(tmp_path):
    from ragfabric_core.db import migrate

    db_url = f"sqlite:///{tmp_path / 'ragfabric.db'}"
    migrate.upgrade(db_url)
    (tmp_path / "ragfabric.yaml").write_text("llm:\n  provider: offline\n")
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    ctx.db_url = db_url
    assert qs._key_is_active(ctx, "rf_not_a_key_in_this_database") is False


# final fix wave: I8, no known secrets in a written .env -----------------------


def test_a_written_dot_env_has_a_random_jwt_secret_and_admin_password_never_printed(
    tmp_path, no_ollama, no_keys, steps_after_config_are_noops
):
    from ragfabric_cli.envfile import read_env_file

    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    values = read_env_file(tmp_path / ".env")
    assert values["JWT_SECRET"] != "change-me" and len(values["JWT_SECRET"]) >= 64
    assert values["FIRST_ADMIN_PASSWORD"] not in ("", "adminpass123")
    assert values["JWT_SECRET"] not in result.output
    assert values["FIRST_ADMIN_PASSWORD"] not in result.output
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600
    other = tmp_path / "other"
    runner.invoke(app, ["quickstart", "--dir", str(other), "--yes"])
    assert read_env_file(other / ".env")["JWT_SECRET"] != values["JWT_SECRET"]


# final fix wave: the packaged docker-compose.yml ------------------------------


def test_the_written_compose_file_has_its_own_project_name_and_the_bind_warning(
    tmp_path, no_ollama, no_keys, fake_docker, steps_after_config_are_noops
):
    import yaml

    work = tmp_path / "My RAG"
    result = runner.invoke(app, ["quickstart", "--dir", str(work), "--docker", "--yes"])
    assert result.exit_code == 0, result.output
    text = (work / "docker-compose.yml").read_text()
    name = yaml.safe_load(text)["name"]
    assert name.startswith("ragfabric-quickstart-my-rag-") and name != "ragfabric"
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name)
    assert "written by `ragfabric quickstart --docker`" in text
    assert "ragfabric init" not in text
    assert "change the bind address" in text and "ENVIRONMENT=production" in text
    other = tmp_path / "other" / "My RAG"
    runner.invoke(app, ["quickstart", "--dir", str(other), "--docker", "--yes"])
    assert yaml.safe_load((other / "docker-compose.yml").read_text())["name"] != name
    assert "docker compose down" in result.output


# final fix wave: a 500 from the temporary server shows its log ----------------

FAILING_SERVER = textwrap.dedent(
    """
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "service": "ragfabric"}')

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            print("Traceback: boom while reading postgresql://u:secretpw@h/db", flush=True)
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail": "Internal Server Error"}')

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
    """
)


def test_a_500_during_the_sample_question_prints_the_server_log_tail_and_path(
    monkeypatch, tmp_path
):
    script = tmp_path / "failing_server.py"
    script.write_text(FAILING_SERVER)
    monkeypatch.setattr(
        qs, "_server_command", lambda port: [sys.executable, str(script), str(port)]
    )
    monkeypatch.setattr(qs, "_admin_credentials", lambda ctx: ("admin@example.com", "pw"))
    monkeypatch.setattr(qs, "_credentials", lambda ctx, url, email, pw: {"api_key": "rf_x"})
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    with pytest.raises(qs.QuickstartError) as caught:
        qs._step_ask(ctx)
    message = str(caught.value)
    assert "HTTP 500" in message
    assert "boom while reading postgresql://u:***@h/db" in message
    assert "secretpw" not in message
    found = re.search(r"full log: (\S+)", message)
    assert found, message
    log = Path(found.group(1))
    try:
        assert log.is_file() and "boom" in log.read_text()
    finally:
        log.unlink(missing_ok=True)
