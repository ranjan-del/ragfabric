"""A tiny KEY=VALUE reader and a 0600 writer for .env files. Neither prints a value.

Blank lines and ``#`` comments are ignored, an optional ``export`` prefix is
accepted, and one pair of matching quotes around a value is stripped.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_env_file(path: Path) -> dict[str, str]:
    """The KEY=VALUE pairs in ``path``, or an empty dict when it does not exist."""
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return {}
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, sep, value = line.partition("=")
        if not sep or not name.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[name.strip()] = value
    return values


def write_private(path: Path, data: bytes) -> None:
    """Write ``path`` readable and writable by its owner only (0600): .env holds secrets."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.chmod(path, 0o600)  # O_CREAT's mode does not apply to a file that already existed
