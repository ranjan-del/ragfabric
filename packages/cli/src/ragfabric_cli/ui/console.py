"""One console per process, and the rules for when it may use colour and boxes."""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from typing import IO

from rich.console import Console

_AUTHORITY = re.compile(r"(\b[a-z][a-z0-9+.-]*://)([^/\s]*)", re.IGNORECASE)


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


def _mask_authority(match: re.Match[str]) -> str:
    scheme, authority = match.group(1), match.group(2)
    if "@" not in authority:
        return match.group(0)
    # The password may itself contain '@', so split at the LAST one.
    userinfo, _, host = authority.rpartition("@")
    user, colon, _password = userinfo.partition(":")
    if not colon:
        return match.group(0)
    return f"{scheme}{user}:***@{host}"


def mask_urls_in(text: str) -> str:
    """Mask the password of every URL found in ``text``."""
    return _AUTHORITY.sub(_mask_authority, text)


def mask_url(url: str) -> str:
    """Replace the password in a URL with ``***``; anything else comes back unchanged."""
    return mask_urls_in(url)
