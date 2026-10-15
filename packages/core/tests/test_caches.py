from ragfabric_core.config_file import CacheConfig
from ragfabric_core.stores.base import Cache
from ragfabric_core.stores.memory_cache import MemoryCache
from ragfabric_core.stores.registry import build_cache


def test_memory_cache_get_set_incr_and_ttl(monkeypatch):
    c = MemoryCache()
    assert isinstance(c, Cache)
    assert c.get("k") is None
    c.set("k", b"v", ttl_seconds=60)
    assert c.get("k") == b"v"
    assert c.incr("n", ttl_seconds=60) == 1 and c.incr("n") == 2
    now = [1000.0]
    monkeypatch.setattr("ragfabric_core.stores.memory_cache.time.monotonic", lambda: now[0])
    c.set("t", b"x", ttl_seconds=1)
    now[0] += 2
    assert c.get("t") is None


def test_registry_builds_memory_and_redis_kinds():
    assert isinstance(build_cache(CacheConfig(kind="memory")), MemoryCache)
    redis_cache = build_cache(CacheConfig(kind="redis"), url="redis://localhost:6379/9")
    assert redis_cache.name == "redis"


def test_redis_cache_connects_with_short_timeouts(monkeypatch):
    """A Redis that hangs, not just one that refuses, must not hang every request:
    the limiter allows the request when the cache fails, but only once it fails."""
    import redis

    from ragfabric_core.stores.redis_cache import RedisCache

    seen = {}

    def fake_from_url(url, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(redis.Redis, "from_url", staticmethod(fake_from_url))
    RedisCache("redis://example:6379/0")
    assert 0 < seen["socket_connect_timeout"] <= 2
    assert 0 < seen["socket_timeout"] <= 2
