"""One console per process, and the rules for when it may use colour and boxes."""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from typing import IO

from rich.console import Console

_PASSWORD = re.compile(r"(://[^:/@\s]*:)[^@/\s]+(@)")


def is_rich(stream: IO[str] | None = None) -> bool:
    """True only on a terminal with NO_COLOR and RAGFABRIC_PLAIN both unset."""
    if os.environ.get("NO_COLOR") or os.environ.get("RAGFABRIC_PLAIN"):
        return False
    target = stream if stream is not None else sys.stdout
    try:
        return bool(target.isatty())
    except (AttributeError, ValueError):
        return False


@lru_cache(maxsize=1)
def get_console() -> Console:
    """The process wide console. Never highlights and never wraps mid word silently."""
    return Console(
        highlight=False,
        soft_wrap=False,
        no_color=bool(os.environ.get("NO_COLOR")),
    )


def mask_url(url: str) -> str:
    """Replace the password in a URL with ``***``; anything else comes back unchanged."""
    return _PASSWORD.sub(r"\1***\2", url)
