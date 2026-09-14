from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.queue.base import Job, JobQueue
from ragfabric_core.queue.memory_queue import MemoryJobQueue
from ragfabric_core.queue.redis_queue import RedisJobQueue
from ragfabric_core.queue.registry import build_queue


def test_memory_queue_is_fifo_and_drains():
    q = MemoryJobQueue()
    assert isinstance(q, JobQueue)
    q.enqueue(Job(id="1", kind="a", payload={}))
    q.enqueue(Job(id="2", kind="b", payload={"x": 1}))
    assert q.size() == 2
    seen = []
    assert q.drain(lambda job: seen.append(job.kind)) == 2
    assert seen == ["a", "b"] and q.size() == 0 and q.dequeue(timeout_seconds=0) is None


def test_redis_queue_serialises_jobs():
    calls = []

    class FakeRedis:
        def __init__(self):
            self.items = []

        def rpush(self, key, value):
            calls.append(("rpush", key))
            self.items.append(value)

        def blpop(self, keys, timeout):
            return (keys[0], self.items.pop(0)) if self.items else None

        def lpop(self, key):
            return self.items.pop(0) if self.items else None

        def llen(self, key):
            return len(self.items)

    q = RedisJobQueue("redis://unused", client=FakeRedis())
    q.enqueue(Job(id="j1", kind="index_document", payload={"document_id": 5}))
    assert q.size() == 1 and calls[0] == ("rpush", "ragfabric:jobs")
    job = q.dequeue(timeout_seconds=1)
    assert job.id == "j1" and job.payload == {"document_id": 5} and job.attempts == 0
    assert q.dequeue(timeout_seconds=0) is None


def test_build_queue_returns_none_for_inline_and_redis_queue_for_queue_mode(monkeypatch):
    assert build_queue(RagFabricConfig()) is None
    cfg = RagFabricConfig.model_validate({"ingestion": {"indexing": "queue"}})
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/9")
    q = build_queue(cfg)
    assert isinstance(q, RedisJobQueue)
