import pytest
import typer
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()


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
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0
    assert "Examples:" in result.output
    assert f"ragfabric {' '.join(path[:1])}" in result.output


def _find_command(path):
    node = app
    for part in path[:-1]:
        node = next(g.typer_instance for g in node.registered_groups if g.name == part)
    for command in node.registered_commands:
        if (command.name or command.callback.__name__.replace("_", "-")) == path[-1]:
            return command
    raise AssertionError(path)


@pytest.mark.parametrize("path", PATHS, ids=lambda p: " ".join(p))
def test_example_flags_exist_on_the_command(path):
    import re

    click_command = typer.main.get_command(app)
    for part in path:
        click_command = click_command.commands[part]
    known = {opt for param in click_command.params for opt in (*param.opts, *param.secondary_opts)}
    known.add("--help")
    out = runner.invoke(app, [*path, "--help"]).output
    lines = [ln for ln in out.splitlines() if ln.strip().startswith(f"ragfabric {' '.join(path)}")]
    assert lines
    for line in lines:
        for flag in re.findall(r"(?<!\S)(--[a-z][a-z-]*)", line):
            assert flag in known, f"{line!r} uses unknown {flag}"
