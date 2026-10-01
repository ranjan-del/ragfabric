import httpx
import pytest
import typer
from typer.testing import CliRunner

from ragfabric_cli.main import app
from ragfabric_cli.ui import errors
from ragfabric_cli.ui.errors import friendly_error
from ragfabric_core.providers.base import ProviderError
from ragfabric_sdk.errors import AuthError, RagFabricError


def test_no_server_names_the_url_and_the_fix():
    exc = httpx.ConnectError(
        "refused", request=httpx.Request("POST", "http://localhost:8000/api/ask")
    )
    fe = friendly_error(exc)
    assert "http://localhost:8000" in fe.problem and fe.fix == "ragfabric serve"


def test_a_database_password_never_appears():
    from sqlalchemy.exc import OperationalError

    exc = OperationalError(
        "connect", {}, Exception("could not connect to postgresql+psycopg://rf:s3cret@db/rf")
    )
    fe = friendly_error(exc)
    assert "s3cret" not in fe.problem + (fe.fix or "")
    assert "rf:***@db" in fe.problem


@pytest.mark.parametrize(
    ("message", "needle"),
    [("HTTP 401 invalid_api_key", "rejected"), ("429 insufficient_quota", "quota")],
)
def test_provider_errors_are_explained(message, needle):
    assert needle in friendly_error(ProviderError("openai", message)).problem


def test_provider_connection_fix_depends_on_provider():
    assert friendly_error(ProviderError("ollama", "Connection refused")).fix == "ollama serve"
    assert friendly_error(ProviderError("openai", "Connection reset")).fix == (
        "check the provider URL"
    )


def test_other_provider_errors_keep_their_message():
    fe = friendly_error(ProviderError("openai", "boom"))
    assert "openai" in fe.problem and "boom" in fe.problem


def test_auth_is_matched_before_the_generic_server_error():
    fe = friendly_error(AuthError("no", 401))
    assert "refused the credentials" in fe.problem
    assert fe.fix == (
        "set RAGFABRIC_API_KEY, or pass --token or --api-key "
        "(ragfabric quickstart writes a key to .env)"
    )
    assert "returned an error" in friendly_error(RagFabricError("bad", 500)).problem


def test_a_config_error_points_at_validate():
    from pydantic import BaseModel, ValidationError

    class M(BaseModel):
        n: int

    with pytest.raises(ValidationError) as info:
        M(n="x")
    fe = friendly_error(info.value)
    assert fe.problem.startswith("ragfabric.yaml is invalid: n:")
    assert fe.fix == "ragfabric config validate"


def test_an_unknown_error_is_not_claimed():
    assert friendly_error(KeyError("x")) is None


@pytest.fixture
def boom_app():
    @app.command("boom-test")
    def boom() -> None:
        raise httpx.ConnectError("refused", request=httpx.Request("GET", "http://localhost:8000/x"))

    @app.command("exit-test")
    def exit_cmd() -> None:
        raise typer.Exit(3)

    yield app
    app.registered_commands[:] = [
        c for c in app.registered_commands if c.name not in ("boom-test", "exit-test")
    ]
    errors.DEBUG = False


def test_a_command_failure_is_friendly(boom_app):
    result = CliRunner().invoke(boom_app, ["boom-test"])
    assert result.exit_code == 1
    assert "No RagFabric server at http://localhost:8000" in result.output
    assert "ragfabric serve" in result.output
    assert "Traceback" not in result.output


def test_debug_shows_the_traceback(boom_app):
    result = CliRunner().invoke(boom_app, ["--debug", "boom-test"])
    assert result.exit_code == 1
    assert isinstance(result.exception, httpx.ConnectError)


def test_typer_exit_passes_through(boom_app):
    assert CliRunner().invoke(boom_app, ["exit-test"]).exit_code == 3


def test_an_unexpected_error_asks_for_debug(boom_app):
    @boom_app.command("key-test")
    def key() -> None:
        raise KeyError("x")

    try:
        result = CliRunner().invoke(boom_app, ["key-test"])
    finally:
        boom_app.registered_commands[:] = [
            c for c in boom_app.registered_commands if c.name != "key-test"
        ]
    assert result.exit_code == 1
    assert "Unexpected error: KeyError" in result.output
    assert "rerun with --debug" in result.output


SECRET_URL = "postgresql+psycopg://rf:s3cret@db:5432/rf"


def test_the_unexpected_fallback_masks_passwords(boom_app):
    @boom_app.command("secret-test")
    def secret() -> None:
        raise RuntimeError(f"failed against {SECRET_URL}")

    try:
        result = CliRunner().invoke(boom_app, ["secret-test"])
    finally:
        boom_app.registered_commands[:] = [
            c for c in boom_app.registered_commands if c.name != "secret-test"
        ]
    assert "Unexpected error: RuntimeError" in result.output
    assert "s3cret" not in result.output


def test_server_and_provider_messages_mask_passwords():
    assert "s3cret" not in friendly_error(RagFabricError(f"bad {SECRET_URL}", 500)).problem
    assert "s3cret" not in friendly_error(ProviderError("openai", f"x {SECRET_URL}")).problem


def test_two_urls_in_a_database_error_are_masked():
    from sqlalchemy.exc import OperationalError

    exc = OperationalError("c", {}, Exception(f"tried {SECRET_URL} then postgresql://u:other@h/d"))
    fe = friendly_error(exc)
    assert "s3cret" not in fe.problem and "other" not in fe.problem


def test_missing_argument_and_unknown_command_are_usage_errors():
    runner = CliRunner()
    for args in (["ask"], ["nope"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 2, args
        assert "Unexpected error" not in result.output
        assert "Usage" in result.output


def test_a_broken_pipe_is_quiet(boom_app):
    @boom_app.command("pipe-test")
    def pipe() -> None:
        raise BrokenPipeError()

    try:
        result = CliRunner().invoke(boom_app, ["pipe-test"])
    finally:
        boom_app.registered_commands[:] = [
            c for c in boom_app.registered_commands if c.name != "pipe-test"
        ]
    assert result.exit_code == 1
    assert "Unexpected error" not in result.output


def test_typer_usage_errors_derive_from_the_public_typer_exception():
    # errors.py relies on this relationship; a Typer upgrade that breaks it fails here.
    result = CliRunner().invoke(app, ["nope"], standalone_mode=False)
    assert isinstance(result.exception, typer.TyperException)


def _ask_raising(monkeypatch, exc):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def ask(self, query, **params):
            raise exc

    monkeypatch.setattr("ragfabric_cli.commands.ask.Client", FakeClient)
    return CliRunner().invoke(app, ["ask", "q", "--token", "t", "--no-stream"])


@pytest.mark.parametrize(
    ("exc", "problem", "fix"),
    [
        (
            httpx.ReadTimeout("", request=httpx.Request("POST", "http://h:8000/api/ask")),
            "The server at http://h:8000 did not answer in time",
            "retry, or check that ragfabric serve is still running",
        ),
        (
            httpx.RemoteProtocolError("", request=httpx.Request("POST", "http://h:8000/api/ask")),
            "Lost the connection to http://h:8000",
            "check that ragfabric serve is still running",
        ),
    ],
)
def test_ask_timeouts_and_dropped_connections_are_friendly(monkeypatch, exc, problem, fix):
    result = _ask_raising(monkeypatch, exc)
    assert result.exit_code == 1
    assert problem in result.output and fix in result.output
    assert "Traceback" not in result.output and "Unexpected" not in result.output


def test_a_provider_that_is_not_installed_gets_the_pip_fix():
    fe = friendly_error(
        ProviderError(
            "ollama", "openai is not installed. Install it with: pip install 'ragfabric[openai]'"
        )
    )
    assert fe.fix == "pip install 'ragfabric[openai]'"


@pytest.mark.parametrize("value", ["127.0.0.1:8000", "localhost:8000"])
def test_a_url_without_a_scheme_says_it_needs_http(value):
    client = httpx.Client(base_url=value)
    try:
        with pytest.raises(httpx.UnsupportedProtocol) as caught:
            client.post("/api/ask")
    finally:
        client.close()
    fe = friendly_error(caught.value)
    assert fe.problem == f"The URL {value} needs http:// or https://"
    assert fe.fix == "use http://127.0.0.1:8000 (or your server's address)"
    assert "://" not in fe.problem.replace("http://", "").replace("https://", "")
