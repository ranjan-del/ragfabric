"""Unit tests for ragfabric quickstart: each step on its own, no server, no network."""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap

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
    (tmp_path / ".env").write_bytes(b"DATABASE_URL=sqlite:///./mine.db\n")
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ragfabric.yaml").read_bytes() == original
    assert (tmp_path / ".env").read_bytes() == b"DATABASE_URL=sqlite:///./mine.db\n"
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
    assert result.exit_code == 0, result.output
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
