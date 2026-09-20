"""Exception types, mapped from HTTP status codes in one place.

Every client method funnels through ``raise_for_status`` so a caller can
catch one hierarchy (``RagFabricError``) or one specific cause
(``AuthError``, ``NotFoundError``, ``RateLimitError``) regardless of which
endpoint raised it.
"""

from __future__ import annotations

import httpx


class RagFabricError(Exception):
    """Base error for anything the server rejected. Carries the HTTP status
    code and the response body's ``detail`` as the message."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(RagFabricError):
    """401 (not authenticated) or 403 (authenticated but not permitted)."""


class NotFoundError(RagFabricError):
    """404: no such resource."""


class RateLimitError(RagFabricError):
    """429: an API key went over its rate limit."""


def raise_for_status(response: httpx.Response) -> None:
    """Raise the mapped error for a 4xx/5xx response; do nothing otherwise."""
    if response.status_code < 400:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    code = response.status_code
    if code in (401, 403):
        raise AuthError(str(detail), code)
    if code == 404:
        raise NotFoundError(str(detail), code)
    if code == 429:
        raise RateLimitError(str(detail), code)
    raise RagFabricError(str(detail), code)
