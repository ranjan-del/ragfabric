from typer.testing import CliRunner

from ragfabric_cli.main import app
from ragfabric_core import __version__

runner = CliRunner()


def test_bare_ragfabric_shows_the_welcome_screen():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert __version__ in result.output
    for command in ("quickstart", "doctor", "strategies"):
        assert f"ragfabric {command}" in result.output
    assert "github.com/ranjan-del/ragfabric" in result.output


def test_dash_dash_help_still_shows_the_normal_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Commands" in result.output


def test_debug_alone_still_shows_the_welcome_screen():
    result = runner.invoke(app, ["--debug"])
    assert result.exit_code == 0
    assert "ragfabric strategies" in result.output


def test_a_subcommand_does_not_print_the_welcome_screen():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "quickstart" not in result.output
