"""One console per process, and the rules for when it may use colour and boxes."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import IO

from rich.console import Console

from ragfabric_core.diagnostics import mask_url, mask_urls_in

__all__ = ["get_console", "is_rich", "mask_url", "mask_urls_in"]


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
