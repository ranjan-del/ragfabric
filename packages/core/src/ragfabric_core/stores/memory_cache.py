"""In process Cache for tests and single process development."""

from __future__ import annotations

import time


class MemoryCache:
    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, float | None]] = {}

    def _live(self, key: str) -> bytes | None:
        item = self._data.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and time.monotonic() >= expires:
            del self._data[key]
            return None
        return value

    def get(self, key: str) -> bytes | None:
        return self._live(key)

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        self._data[key] = (value, time.monotonic() + ttl_seconds if ttl_seconds else None)

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        current = self._live(key)
        new = (int(current) if current else 0) + 1
        expires = (
            self._data[key][1]
            if current is not None
            else (time.monotonic() + ttl_seconds if ttl_seconds else None)
        )
        self._data[key] = (str(new).encode(), expires)
        return new
