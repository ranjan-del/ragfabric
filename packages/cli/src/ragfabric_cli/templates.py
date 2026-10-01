"""Packaged starter files: config templates and a small sample corpus.

They live under ``ragfabric_cli/data`` so they ship inside the wheel and
``ragfabric init`` works from any directory, not only a source checkout.
"""

from importlib.resources import as_file, files
from pathlib import Path

SAMPLE_QUESTION = "How much unused annual leave carries forward into the next year?"


def _data_dir() -> Path:
    with as_file(files("ragfabric_cli") / "data") as path:
        return Path(path)


def template_path(name: str) -> Path:
    """Return the path of a packaged template, or raise FileNotFoundError naming it."""
    path = _data_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"packaged template not found: {name}")
    return path


def sample_paths() -> list[Path]:
    """Return the sample corpus files, sorted by name."""
    return sorted((_data_dir() / "samples").glob("*.md"))
