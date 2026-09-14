"""Fixed window rate limit on the Cache interface: one counter per subject per minute."""

from __future__ import annotations

import time

from ragfabric_core.stores.base import Cache


def check_rate_limit(
    cache: Cache, subject: str, limit_per_minute: int, now: float | None = None
) -> bool:
    current = time.time() if now is None else now
    minute = int(current // 60)
    count = cache.incr(f"ratelimit:{subject}:{minute}", ttl_seconds=120)
    return count <= limit_per_minute
