"""ragfabric quickstart end to end: a real server, a real SQLite file, a cited answer.

No Ollama, no API key and no Docker: the probe is forced to fail and the key
variables are unset, so this exercises the offline path a first-time user with
nothing installed gets.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import httpx
from typer.testing import CliRunner

from ragfabric_cli.commands import quickstart as qs
from ragfabric_cli.envfile import read_env_file
from ragfabric_cli.main import app
from ragfabric_cli.templates import SAMPLE_QUESTION

runner = CliRunner()

WHOLE_SENTENCE = (
    "At the end of the year, up to 10 days of unused annual leave carry forward into the next year."
)


def _answer_and_sources(output: str) -> tuple[str, list[str]]:
    """The plain answer text (the lines before ``sources:``) and the source lines."""
    lines = output.splitlines()
    end = lines.index("sources:")
    start = max(i for i, line in enumerate(lines[:end]) if line.startswith("ask: ")) + 1
    sources = []
    for line in lines[end + 1 :]:
        if not line.startswith("  ["):
            break
        sources.append(line)
    return "\n".join(lines[start:end]).strip(), sources


def _assert_no_heading_quoted(answer: str) -> None:
    import re

    for segment in re.split(r"\[\d+\]", answer):
        assert not segment.strip().startswith("#"), answer
    assert "\n#" not in answer and not answer.startswith("#"), answer


def test_quickstart_from_nothing_to_a_cited_answer_and_again(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in ("RAGFABRIC_API_KEY", "RAGFABRIC_TOKEN", "RAGFABRIC_URL"):
        monkeypatch.delenv(name, raising=False)
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
    answer, sources = _answer_and_sources(result.output)
    assert WHOLE_SENTENCE in answer
    _assert_no_heading_quoted(answer)
    assert sources and all("leave-policy.md" in line for line in sources), sources
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

    key = read_env_file(tmp_path / ".env")["RAGFABRIC_API_KEY"]
    assert key.startswith("rf_")
    written_url = read_env_file(tmp_path / ".env")["RAGFABRIC_URL"]
    from ragfabric_core.diagnostics import port_in_use

    if port_in_use("127.0.0.1", 8000):
        # Another program holds 8000 (I2): .env names a free port and serve is told it.
        assert written_url != "http://127.0.0.1:8000"
        assert f"ragfabric serve --port {written_url.rsplit(':', 1)[1]}" in result.output
    else:
        assert written_url == "http://127.0.0.1:8000"
    assert key not in result.output and key not in again.output
    assert "--token" not in result.output

    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600
    # The printed ask step works against `ragfabric serve`, with only RAGFABRIC_URL moved.
    printed = next(line for line in result.output.splitlines() if "ragfabric ask " in line)
    assert printed.strip().startswith(f'ragfabric ask "{SAMPLE_QUESTION}"')
    assert f"  cd {tmp_path}  #" in result.output
    port = real_free_port()
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DATABASE_URL", "RAGFABRIC_CONFIG", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "ragfabric_cli.main", "serve", "--port", str(port)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert server.poll() is None, "ragfabric serve exited"
            assert time.monotonic() < deadline, "ragfabric serve never became healthy"
            time.sleep(0.2)
        # Point .env at this server's port. RAGFABRIC_URL in the
        # environment would make ask ignore .env entirely (ruling R18), key included.
        dotenv = tmp_path / ".env"
        dotenv.write_text(
            dotenv.read_text().replace(
                f"RAGFABRIC_URL={written_url}", f"RAGFABRIC_URL=http://127.0.0.1:{port}"
            )
        )
        monkeypatch.chdir(tmp_path)
        asked = runner.invoke(app, ["ask", SAMPLE_QUESTION])
        # The help example: whatever it quotes, it quotes no heading.
        refund = runner.invoke(app, ["ask", "What is our refund policy?", "--no-stream"])
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    assert asked.exit_code == 0, asked.output
    assert "up to 10 days of unused annual leave carry forward" in asked.output
    assert "leave-policy.md" in asked.output
    assert refund.exit_code == 0, refund.output
    _assert_no_heading_quoted(refund.output.split("sources:")[0])


def test_a_force_rerun_leaves_exactly_one_active_quickstart_key(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from ragfabric_core.models.access import ApiKey

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in ("RAGFABRIC_API_KEY", "RAGFABRIC_TOKEN", "RAGFABRIC_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(qs, "_ollama_models", lambda timeout=2.0: None)
    first = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert first.exit_code == 0, first.output
    old_key = read_env_file(tmp_path / ".env")["RAGFABRIC_API_KEY"]
    again = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes", "--force"])
    assert again.exit_code == 0, again.output
    new_key = read_env_file(tmp_path / ".env")["RAGFABRIC_API_KEY"]
    assert new_key != old_key
    engine = create_engine(f"sqlite:///{tmp_path / 'ragfabric.db'}")
    with Session(engine) as db:
        keys = db.query(ApiKey).filter(ApiKey.name == "quickstart").all()
        active = [k for k in keys if k.is_active]
    engine.dispose()
    assert len(keys) == 2 and len(active) == 1
    assert new_key.startswith(active[0].key_prefix)


def test_ollama_running_without_the_openai_package_still_reaches_a_cited_answer(
    tmp_path, monkeypatch
):
    """C1: a base install (no openai package) with Ollama running ends offline, not broken."""
    import importlib.util

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in ("RAGFABRIC_API_KEY", "RAGFABRIC_TOKEN", "RAGFABRIC_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        qs, "_ollama_models", lambda timeout=2.0: ["llama3.2:3b", "nomic-embed-text:latest"]
    )
    real = importlib.util.find_spec

    def without_openai(name, *args, **kwargs):
        return None if name.split(".")[0] == "openai" else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_openai)
    result = runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "client package (openai) is not installed" in result.output
    assert "model: offline" in result.output
    assert "up to 10 days of unused annual leave carry forward" in result.output
    assert "leave-policy.md" in result.output
