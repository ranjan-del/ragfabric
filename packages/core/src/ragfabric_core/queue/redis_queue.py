"""FIFO job queue on a Redis list: RPUSH to enqueue, BLPOP to dequeue.

Why not Celery: one list and two commands cover ingestion fan out, and the
worker is a plain loop that any adopter can read in a minute. Retries and dead
letters arrive when a later phase needs them; the Job carries `attempts` so
that change is additive.
"""

from __future__ import annotations

from typing import Any

from ragfabric_core.queue.base import Job


class RedisJobQueue:
    def __init__(self, url: str, key: str = "ragfabric:jobs", client: Any | None = None) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(url)
        self._r = client
        self.key = key

    def enqueue(self, job: Job) -> None:
        self._r.rpush(self.key, job.model_dump_json())

    def dequeue(self, timeout_seconds: float = 1.0) -> Job | None:
        timeout = max(int(round(timeout_seconds)), 0)
        if timeout == 0:
            raw = self._r.lpop(self.key)
            if raw is None:
                return None
        else:
            item = self._r.blpop([self.key], timeout=timeout)
            if item is None:
                return None
            _, raw = item
        return Job.model_validate_json(raw if isinstance(raw, str) else raw.decode("utf-8"))

    def size(self) -> int:
        return int(self._r.llen(self.key))
