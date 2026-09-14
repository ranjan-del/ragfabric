from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Job(BaseModel):
    id: str
    kind: str
    payload: dict = Field(default_factory=dict)
    attempts: int = 0


@runtime_checkable
class JobQueue(Protocol):
    def enqueue(self, job: Job) -> None: ...
    def dequeue(self, timeout_seconds: float = 1.0) -> Job | None: ...
    def size(self) -> int: ...
