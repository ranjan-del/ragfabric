import re

import pytest
import typer
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()
# Typer forces terminal styling when GITHUB_ACTIONS is set, so CI's help text carries ANSI codes.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _help(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, [*args, "--help"])
    return result.exit_code, _ANSI.sub("", result.output)


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
    code, out = _help(*path)
    assert code == 0
    assert "Examples:" in out
    assert f"ragfabric {' '.join(path[:1])}" in out


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

    click_command = typer.main.get_command(app)
    for part in path:
        click_command = click_command.commands[part]
    known = {opt for param in click_command.params for opt in (*param.opts, *param.secondary_opts)}
    known.add("--help")
    _, out = _help(*path)
    lines = [ln for ln in out.splitlines() if ln.strip().startswith(f"ragfabric {' '.join(path)}")]
    assert lines
    for line in lines:
        for flag in re.findall(r"(?<!\S)(--[a-z][a-z-]*)", line):
            assert flag in known, f"{line!r} uses unknown {flag}"


@pytest.mark.parametrize("path", PATHS, ids=lambda p: " ".join(p))
def test_examples_paste_into_a_shell(path):
    """No <placeholder> (a shell redirection when pasted) and no example that overwrites .env."""
    _, out = _help(*path)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("ragfabric ")]
    for line in lines:
        assert "<" not in line and ">" not in line, line
        assert line != "ragfabric init --force", line


def test_the_key_example_uses_the_quickstart_admin():
    _, out = _help("keys", "create")
    assert "--user admin@example.com" in out
