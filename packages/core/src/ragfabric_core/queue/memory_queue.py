from __future__ import annotations

from collections import deque
from collections.abc import Callable

from ragfabric_core.queue.base import Job


class MemoryJobQueue:
    def __init__(self) -> None:
        self._items: deque[Job] = deque()

    def enqueue(self, job: Job) -> None:
        self._items.append(job)

    def dequeue(self, timeout_seconds: float = 1.0) -> Job | None:
        return self._items.popleft() if self._items else None

    def size(self) -> int:
        return len(self._items)

    def drain(self, handler: Callable[[Job], None]) -> int:
        n = 0
        while self._items:
            handler(self._items.popleft())
            n += 1
        return n
