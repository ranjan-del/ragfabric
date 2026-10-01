"""I2: a port held by another program, and a server that is not RagFabric."""

from __future__ import annotations

import json
import re
import socket

import pytest
from typer.testing import CliRunner

from ragfabric_cli.commands import quickstart as qs
from ragfabric_cli.main import app
from ragfabric_sdk.errors import NotFoundError

runner = CliRunner()


@pytest.fixture
def held_port():
    """A port on 127.0.0.1 that another socket is listening on for the test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        yield sock.getsockname()[1]


def test_serve_on_a_port_in_use_exits_1_and_names_a_free_port(held_port, monkeypatch):
    import uvicorn

    def never(*args, **kwargs):
        raise AssertionError("uvicorn must not start on a port in use")

    monkeypatch.setattr(uvicorn, "run", never)
    result = runner.invoke(app, ["serve", "--port", str(held_port)])
    assert result.exit_code == 1
    assert f"127.0.0.1:{held_port} is already in use by another program" in result.output
    found = re.search(
        r"ragfabric serve --port (\d+), then set RAGFABRIC_URL=http://127.0.0.1:(\d+) in \.env",
        result.output,
    )
    assert found and found.group(1) == found.group(2)
    assert int(found.group(1)) != held_port
    assert "export" not in result.output
    assert "Traceback" not in result.output


def test_quickstart_writes_a_free_port_when_8000_is_taken(tmp_path, monkeypatch):
    monkeypatch.setattr(qs, "port_in_use", lambda host, port: port == 8000)
    monkeypatch.setattr(qs, "free_port", lambda: 8765)
    monkeypatch.setattr(qs, "_create_api_key", lambda ctx, email: "rf_secretkeyvalue")
    env = tmp_path / ".env"
    env.write_text("JWT_SECRET=x\n")
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    ctx.env_written = True
    qs._credentials(ctx, "http://127.0.0.1:1", "admin@example.com", "pw")
    assert "RAGFABRIC_URL=http://127.0.0.1:8765\n" in env.read_text()
    steps = [command for command, _ in qs._next_steps(ctx)]
    assert any(step.endswith("ragfabric serve --port 8765") for step in steps)


def test_quickstart_keeps_8000_when_it_is_free(tmp_path, monkeypatch):
    monkeypatch.setattr(qs, "port_in_use", lambda host, port: False)
    monkeypatch.setattr(qs, "_create_api_key", lambda ctx, email: "rf_secretkeyvalue")
    (tmp_path / ".env").write_text("JWT_SECRET=x\n")
    ctx = qs.Context(dir=tmp_path, force=False, yes=True, docker=False, model_check=False)
    ctx.env_written = True
    qs._credentials(ctx, "http://127.0.0.1:1", "admin@example.com", "pw")
    assert "RAGFABRIC_URL=http://127.0.0.1:8000\n" in (tmp_path / ".env").read_text()
    assert any(step.endswith("ragfabric serve") for step, _ in qs._next_steps(ctx))


class _NotFoundClient:
    detail = "Not Found"

    def __init__(self, *args, **kwargs):
        pass

    def ask(self, *args, **kwargs):
        raise NotFoundError(self.detail, 404)

    def close(self):
        pass


@pytest.fixture
def no_ask_env(tmp_path, monkeypatch):
    for name in ("RAGFABRIC_API_KEY", "RAGFABRIC_TOKEN", "RAGFABRIC_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_ask_against_a_server_that_is_not_ragfabric_says_so(no_ask_env, monkeypatch):
    from ragfabric_cli.commands import ask as ask_module

    monkeypatch.setattr(ask_module, "Client", _NotFoundClient)
    result = runner.invoke(
        app, ["ask", "q", "--no-stream", "--url", "http://127.0.0.1:8000", "--api-key", "rf_x"]
    )
    assert result.exit_code == 1
    assert "The server at http://127.0.0.1:8000 is not a RagFabric server" in result.output
    assert "ragfabric serve --port " in result.output and "in .env" in result.output


def test_ask_keeps_the_message_of_any_other_404(no_ask_env, monkeypatch):
    from ragfabric_cli.commands import ask as ask_module

    class Missing(_NotFoundClient):
        detail = "collection 5 not found"

    monkeypatch.setattr(ask_module, "Client", Missing)
    result = runner.invoke(app, ["ask", "q", "--no-stream", "--api-key", "rf_x"])
    assert result.exit_code == 1
    assert "collection 5 not found" in result.output
    assert "not a RagFabric server" not in result.output


def test_ask_says_why_dot_env_was_not_read(no_ask_env, monkeypatch):
    (no_ask_env / ".env").write_text("RAGFABRIC_API_KEY=rf_fromdotenv\n")
    monkeypatch.setenv("RAGFABRIC_URL", "http://127.0.0.1:9")
    result = runner.invoke(app, ["ask", "q"])
    assert result.exit_code == 2
    assert (
        "./.env was not read because RAGFABRIC_URL (or --url, --token, --api-key) was given; "
        "pass --api-key too, or put the URL in .env"
    ) in result.output
    assert "rf_fromdotenv" not in result.output


def _doctor_server_url(monkeypatch, *args):
    from ragfabric_core import diagnostics

    seen = {}

    def fake_run_all(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(diagnostics, "run_all", fake_run_all)
    result = runner.invoke(app, ["doctor", "--json", *args])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []
    return seen["server_url"]


def test_doctor_reads_the_server_url_from_dot_env(no_ask_env, monkeypatch):
    (no_ask_env / ".env").write_text("RAGFABRIC_URL=http://127.0.0.1:8765\n")
    assert _doctor_server_url(monkeypatch) == "http://127.0.0.1:8765"
    assert _doctor_server_url(monkeypatch, "--url", "http://h:1") == "http://h:1"


def test_doctor_ignores_dot_env_when_the_environment_names_a_credential(no_ask_env, monkeypatch):
    (no_ask_env / ".env").write_text("RAGFABRIC_URL=http://127.0.0.1:8765\n")
    monkeypatch.setenv("RAGFABRIC_API_KEY", "rf_env")
    assert _doctor_server_url(monkeypatch) == "http://localhost:8000"
    monkeypatch.setenv("RAGFABRIC_URL", "http://127.0.0.1:7")
    assert _doctor_server_url(monkeypatch) == "http://127.0.0.1:7"
