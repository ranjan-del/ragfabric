"""Build stores and caches from configuration. The only place store kinds are switched on."""

from __future__ import annotations

import os
from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.config_file import CacheConfig, LexicalStoreConfig, VectorStoreConfig
from ragfabric_core.stores.base import Cache, LexicalStore, VectorStore
from ragfabric_core.stores.memory_cache import MemoryCache
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.stores.redis_cache import RedisCache


def build_vector_store(
    cfg: VectorStoreConfig, session_factory: Callable[[], Session]
) -> VectorStore:
    if cfg.kind in ("pgvector", "memory"):
        # "memory" keeps the v1 in process index for queries in this phase; the
        # fan out still writes to the relational table so nothing is lost.
        return PgVectorStore(session_factory)
    raise NotImplementedError(f"vector store {cfg.kind!r} arrives in Phase 3")


def build_lexical_store(
    cfg: LexicalStoreConfig, session_factory: Callable[[], Session]
) -> LexicalStore:
    if cfg.kind == "postgres_fts":
        return PostgresLexicalStore(session_factory)
    raise NotImplementedError(f"lexical store {cfg.kind!r} arrives in Phase 4")


def build_cache(cfg: CacheConfig, url: str | None = None) -> Cache:
    if cfg.kind == "redis":
        return RedisCache(url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    return MemoryCache()
