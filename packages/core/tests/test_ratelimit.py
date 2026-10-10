"""The fixed-window limiter's result and the atomic Redis counter (Phase 10, Task 2)."""

from __future__ import annotations

from ragfabric_core.auth.ratelimit import RateLimitResult, check_rate_limit, hit
from ragfabric_core.stores.memory_cache import MemoryCache
from ragfabric_core.stores.redis_cache import RedisCache


def test_hit_reports_limit_remaining_and_seconds_to_the_next_window():
    cache = MemoryCache()
    first = hit(cache, "user:1", 2, now=120.0)
    assert first == RateLimitResult(allowed=True, limit=2, count=1, retry_after=60)
    assert first.remaining == 1
    second = hit(cache, "user:1", 2, now=150.5)
    assert second.allowed is True and second.remaining == 0
    third = hit(cache, "user:1", 2, now=179.2)
    assert third.allowed is False and third.remaining == 0
    assert third.retry_after == 1  # 0.8 s left rounds up, never 0


def test_check_rate_limit_still_answers_a_bool():
    cache = MemoryCache()
    assert check_rate_limit(cache, "k", 1, now=0.0) is True
    assert check_rate_limit(cache, "k", 1, now=1.0) is False


class _Pipeline:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def incr(self, key):
        self.calls.append(("incr", key))
        return self

    def expire(self, key, ttl, nx=False):
        self.calls.append(("expire", key, ttl, nx))
        return self

    def execute(self):
        self.calls.append(("execute",))
        return [3, True]


class _Redis:
    def __init__(self) -> None:
        self.calls: list = []
        self.transactions: list[bool] = []

    def pipeline(self, transaction=True):
        self.transactions.append(transaction)
        return _Pipeline(self.calls)


def test_redis_incr_sets_the_expiry_in_the_same_transaction():
    """INCR then a separate EXPIRE left a key with no expiry if the process died
    between the two round trips. Both now go in one MULTI/EXEC, and EXPIRE NX
    never pushes an existing window's expiry further out."""
    fake = _Redis()
    cache = RedisCache("redis://unused", client=fake)
    assert cache.incr("ratelimit:user:1:2", ttl_seconds=120) == 3
    assert fake.transactions == [True]
    assert fake.calls == [
        ("incr", "ratelimit:user:1:2"),
        ("expire", "ratelimit:user:1:2", 120, True),
        ("execute",),
    ]
