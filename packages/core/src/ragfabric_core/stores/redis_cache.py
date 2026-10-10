"""Cache over Redis: embeddings cache, answer cache and rate limit counters."""

from __future__ import annotations

from typing import Any


class RedisCache:
    name = "redis"

    def __init__(self, url: str, client: Any | None = None) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(url)
        self._r = client

    def get(self, key: str) -> bytes | None:
        value = self._r.get(key)
        return bytes(value) if value is not None else None

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        self._r.set(key, value, ex=ttl_seconds)

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        """Increment and, with a TTL, set the expiry in the same MULTI/EXEC.

        Two separate round trips (INCR, then EXPIRE when the count is 1) left a
        key that never expires if the process died between them. EXPIRE NX
        only sets an expiry the key does not already have, so a window is
        never pushed further out by later requests.
        """
        if not ttl_seconds:
            return int(self._r.incr(key))
        pipe = self._r.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, ttl_seconds, nx=True)
        new, _ = pipe.execute()
        return int(new)
