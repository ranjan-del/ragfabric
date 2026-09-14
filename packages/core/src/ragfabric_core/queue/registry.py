from __future__ import annotations

import os

from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.queue.base import JobQueue
from ragfabric_core.queue.redis_queue import RedisJobQueue


def build_queue(cfg: RagFabricConfig) -> JobQueue | None:
    if cfg.ingestion.indexing != "queue":
        return None
    return RedisJobQueue(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
