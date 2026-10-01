import pytest
import typer
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()

# Owned by the other track; its docstring gets its Examples block there.
# The merge step removes this set.
OTHER_TRACK = {"init", "doctor", "quickstart"}


def _walk(typer_app: typer.Typer, prefix: tuple[str, ...] = ()):
    for command in typer_app.registered_commands:
        name = command.name or command.callback.__name__.replace("_", "-")
        yield (*prefix, name)
    for group in typer_app.registered_groups:
        yield from _walk(group.typer_instance, (*prefix, group.name))


PATHS = sorted(_walk(app))


def test_the_walk_finds_the_commands():
    assert ("ask",) in PATHS
    assert ("keys", "create") in PATHS or ("keys", "create-key") in PATHS


@pytest.mark.parametrize("path", PATHS, ids=lambda p: " ".join(p))
def test_every_command_help_has_examples(path):
    if path[-1] in OTHER_TRACK:
        pytest.skip("covered when the tracks merge")
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0
    assert "Examples:" in result.output
    assert f"ragfabric {' '.join(path[:1])}" in result.output
