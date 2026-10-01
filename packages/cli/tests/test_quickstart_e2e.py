"""ragfabric quickstart end to end: a real server, a real SQLite file, a cited answer.

No Ollama, no API key and no Docker: the probe is forced to fail and the key
variables are unset, so this exercises the offline path a first-time user with
nothing installed gets.
"""

from __future__ import annotations

import socket

from typer.testing import CliRunner

from ragfabric_cli.commands import quickstart as qs
from ragfabric_cli.main import app

runner = CliRunner()


def test_quickstart_from_nothing_to_a_cited_answer_and_again(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(qs, "_ollama_models", lambda timeout=2.0: None)
    ports: list[int] = []
    real_free_port = qs.free_port

    def recording_free_port() -> int:
        port = real_free_port()
        ports.append(port)
        return port

    monkeypatch.setattr(qs, "free_port", recording_free_port)

    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "up to 10 days of unused annual leave carry forward" in result.output
    assert "[1]" in result.output
    assert "leave-policy.md" in result.output
    assert (tmp_path / "ragfabric.db").is_file()
    assert ports, "the temporary server was never started"
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            assert sock.connect_ex(("127.0.0.1", port)) != 0, f"port {port} still listening"

    again = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert again.exit_code == 0, again.output
    assert again.output.count("already done") >= 3
    assert "up to 10 days of unused annual leave carry forward" in again.output
