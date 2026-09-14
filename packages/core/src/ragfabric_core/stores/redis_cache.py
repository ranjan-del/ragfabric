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
        new = int(self._r.incr(key))
        if new == 1 and ttl_seconds:
            self._r.expire(key, ttl_seconds)
        return new
