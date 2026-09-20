import pytest
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def cli_env(monkeypatch):
    """Ask reads credentials from the environment when no flag is given, so
    every test in this file starts from a clean slate rather than whatever
    happens to be exported in the shell running the suite."""
    monkeypatch.delenv("RAGFABRIC_TOKEN", raising=False)
    monkeypatch.delenv("RAGFABRIC_API_KEY", raising=False)
    monkeypatch.delenv("RAGFABRIC_URL", raising=False)


def test_ask_streams_tokens_to_stdout(monkeypatch):
    from ragfabric_sdk.models import AskEvent

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def ask_stream(self, query, **params):
            yield AskEvent(event="retrieval", data={"chunks": 1})
            yield AskEvent(event="token", data={"text": "twenty four "})
            yield AskEvent(event="token", data={"text": "days"})
            yield AskEvent(event="citations", data={"citations": [{"marker": "[1]"}]})
            yield AskEvent(event="done", data={"run_id": 5, "latency_ms": 9})

    monkeypatch.setattr("ragfabric_cli.commands.ask.Client", FakeClient)
    result = runner.invoke(app, ["ask", "how much leave", "--token", "t"])
    assert result.exit_code == 0, result.output
    assert "twenty four days" in result.output
    assert "run 5" in result.output


def test_ask_without_any_credential_exits_2():
    result = runner.invoke(app, ["ask", "q", "--url", "http://server"])
    assert result.exit_code == 2
    assert "token" in result.output.lower()


def test_ask_prints_sources_and_reports_a_superseded_correction(monkeypatch):
    from ragfabric_sdk.models import AskEvent

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def ask_stream(self, query, **params):
            yield AskEvent(event="retrieval", data={"chunks": 1})
            yield AskEvent(event="token", data={"text": "twenty "})
            yield AskEvent(event="token", data={"text": "days"})
            yield AskEvent(
                event="superseded", data={"text": "twenty four days [1]", "reason": "citation contract"}
            )
            yield AskEvent(
                event="citations",
                data={
                    "citations": [
                        {"marker": "[1]", "filename": "handbook.pdf", "page": 3, "used": True}
                    ]
                },
            )
            yield AskEvent(event="done", data={"run_id": 7, "latency_ms": 12})

    monkeypatch.setattr("ragfabric_cli.commands.ask.Client", FakeClient)
    result = runner.invoke(app, ["ask", "how much leave", "--token", "t"])
    assert result.exit_code == 0, result.output
    # The draft tokens stay visible (a terminal cannot be un-printed), but the
    # corrected, complete answer must also be present in full.
    assert "twenty days" in result.output
    assert "twenty four days [1]" in result.output
    assert "corrected" in result.output.lower()
    assert "handbook.pdf" in result.output and "p3" in result.output
    assert "run 7" in result.output


def test_ask_no_stream_prints_the_finished_answer(monkeypatch):
    from ragfabric_sdk.models import Answer, Citation

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def ask(self, query, **params):
            return Answer(
                question=query,
                answer="twenty four days",
                confidence=0.9,
                citations=[
                    Citation(
                        marker="[1]",
                        filename="handbook.pdf",
                        page=3,
                        score=0.8,
                        snippet="...",
                        used=True,
                    )
                ],
                highlights=[],
            )

    monkeypatch.setattr("ragfabric_cli.commands.ask.Client", FakeClient)
    result = runner.invoke(app, ["ask", "how much leave", "--token", "t", "--no-stream"])
    assert result.exit_code == 0, result.output
    assert "twenty four days" in result.output
    assert "handbook.pdf" in result.output and "p3" in result.output


def test_ask_json_implies_no_stream(monkeypatch):
    from ragfabric_sdk.models import Answer

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def ask(self, query, **params):
            return Answer(
                question=query,
                answer="twenty four days",
                confidence=0.9,
                citations=[],
                highlights=[],
            )

        def ask_stream(self, query, **params):
            raise AssertionError("--json must not stream")
            yield  # pragma: no cover - never reached

    monkeypatch.setattr("ragfabric_cli.commands.ask.Client", FakeClient)
    result = runner.invoke(app, ["ask", "how much leave", "--token", "t", "--json"])
    assert result.exit_code == 0, result.output
    assert '"answer": "twenty four days"' in result.output


def test_ask_maps_a_ragfabric_error_to_exit_1(monkeypatch):
    from ragfabric_sdk.errors import AuthError

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def ask_stream(self, query, **params):
            raise AuthError("bad token", 401)
            yield  # pragma: no cover - never reached

    monkeypatch.setattr("ragfabric_cli.commands.ask.Client", FakeClient)
    result = runner.invoke(app, ["ask", "q", "--token", "t"])
    assert result.exit_code == 1
    assert "bad token" in result.output


def test_ask_maps_a_connection_failure_to_exit_1(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def ask_stream(self, query, **params):
            raise ConnectionError("connection refused")
            yield  # pragma: no cover - never reached

    monkeypatch.setattr("ragfabric_cli.commands.ask.Client", FakeClient)
    result = runner.invoke(app, ["ask", "q", "--token", "t", "--url", "http://down"])
    assert result.exit_code == 1
    assert "could not reach http://down" in result.output
