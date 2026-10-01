"""Every `ragfabric ...` command quoted in the five minute start sections must exist."""

import re
from pathlib import Path

import pytest
import typer.main
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[3]
DOCS = {
    "README.md": "### Five minute start",
    "docs/getting-started.md": "## Five minute start",
}

# TEMPORARY: quickstart and doctor are registered by another track, so they are skipped
# when not registered. The merge step removes UNREGISTERED_OK and the skip below.
UNREGISTERED_OK = {"quickstart", "doctor"}


def _section(path: str, heading: str) -> str:
    text = (ROOT / path).read_text()
    start = text.index(heading)
    level = len(heading) - len(heading.lstrip("#"))
    rest = text[start + len(heading) :]
    end = re.search(rf"^#{{1,{level}}} ", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


def _quoted_commands(text: str) -> list[list[str]]:
    commands = []
    for block in re.findall(r"```(?:bash)?\n(.*?)```", text, re.DOTALL):
        for line in block.splitlines():
            match = re.match(r"\s*ragfabric\s+(.*)", line.split("#")[0])
            if match:
                words = []
                for word in match.group(1).split():
                    if not re.fullmatch(r"[a-z][a-z-]*", word):
                        break
                    words.append(word)
                if words:
                    commands.append(words)
    return commands


def _resolve(words: list[str]) -> list[str] | None:
    """The registered command path at the start of words, or None."""
    node = typer.main.get_command(app)
    path: list[str] = []
    for word in words:
        commands = getattr(node, "commands", None)
        if commands is None or word not in commands:
            break
        node = commands[word]
        path.append(word)
    return path or None


CASES = [
    (doc, cmd) for doc, heading in DOCS.items() for cmd in _quoted_commands(_section(doc, heading))
]


def test_the_sections_quote_commands():
    assert len(CASES) >= 8


@pytest.mark.parametrize(("doc", "words"), CASES, ids=[f"{d}:{' '.join(w)}" for d, w in CASES])
def test_quoted_command_exists_and_has_help(doc, words):
    path = _resolve(words)
    if path is None and words[0] in UNREGISTERED_OK:
        pytest.skip(f"{words[0]} is registered by another track; removed at merge")
    assert path is not None, f"{doc} quotes `ragfabric {' '.join(words)}`, which does not exist"
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
