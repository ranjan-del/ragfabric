"""Charging requests against the rate limits, once per request.

Three subjects, each with its own limit (see ``LimitsConfig``):

- ``key:<id>`` for an API key, at the key's own ``rate_limit_per_minute``;
- ``user:<id>`` for a signed-in user, at ``limits.user_rate_limit_per_minute``;
- ``auth:<address>`` for login and register, at ``limits.auth_attempts_per_minute``.

A request is charged at most once, however many dependencies resolve the
caller: an upload resolves the user through ``get_current_user`` and again
through ``get_access_filter``, and charging both would halve the real limit.

If the cache cannot be reached the request is allowed and an error is logged
with the request id. The limiter protects capacity, not data: access control
does not live in the cache, and an outage of Redis should not become an outage
of the API. docs/operations.md records the trade-off.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status

from ragfabric_core.auth.ratelimit import hit
from ragfabric_core.runtime import get_config

log = logging.getLogger(__name__)

_CHARGED = "rate_limit_charged"


def charge(request: Request, subject: str, limit_per_minute: int) -> None:
    """Count this request against ``subject``; raise 429 when over the limit."""
    if getattr(request.state, _CHARGED, False):
        return
    setattr(request.state, _CHARGED, True)
    try:
        result = hit(request.app.state.cache, subject, limit_per_minute)
    except Exception:
        log.error(
            "rate limit store unavailable; request allowed without counting (%s)",
            subject.split(":", 1)[0],
            exc_info=True,
        )
        return
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": str(result.retry_after),
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
            },
        )


def client_address(request: Request) -> str:
    """The peer address from the ASGI scope, never from a header.

    uvicorn rewrites the scope from ``X-Forwarded-For`` only when the peer is
    listed in ``FORWARDED_ALLOW_IPS``, so a client cannot choose its own
    address by sending the header itself.
    """
    return request.client.host if request.client else "unknown"


def limit_auth_attempts(request: Request) -> None:
    """Dependency for login and register: limit attempts per client address."""
    charge(request, f"auth:{client_address(request)}", get_config().limits.auth_attempts_per_minute)
