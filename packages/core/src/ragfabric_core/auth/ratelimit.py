"""Fixed window rate limit on the Cache interface: one counter per subject per minute.

A fixed window is cheap (one atomic increment per request) and predictable.
Its known weakness is the window edge: a caller can spend a full minute's
allowance in the last second of one window and again in the first second of
the next, so the worst case is twice the limit across two seconds. That is
acceptable for what this limiter is for, protecting capacity and slowing
password guessing, and it is documented in docs/operations.md.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ragfabric_core.stores.base import Cache

WINDOW_SECONDS = 60


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    count: int
    # Whole seconds until the current window ends, at least 1, for Retry-After.
    retry_after: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.count)


def hit(
    cache: Cache, subject: str, limit_per_minute: int, now: float | None = None
) -> RateLimitResult:
    """Count one request for ``subject`` and say whether it is within the limit."""
    current = time.time() if now is None else now
    minute = int(current // WINDOW_SECONDS)
    count = cache.incr(f"ratelimit:{subject}:{minute}", ttl_seconds=2 * WINDOW_SECONDS)
    retry_after = max(1, math.ceil((minute + 1) * WINDOW_SECONDS - current))
    return RateLimitResult(
        allowed=count <= limit_per_minute,
        limit=limit_per_minute,
        count=count,
        retry_after=retry_after,
    )


def check_rate_limit(
    cache: Cache, subject: str, limit_per_minute: int, now: float | None = None
) -> bool:
    return hit(cache, subject, limit_per_minute, now=now).allowed
