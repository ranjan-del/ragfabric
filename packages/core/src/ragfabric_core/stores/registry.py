"""Build stores and caches from configuration. The only place store kinds are switched on."""

from __future__ import annotations

import os
from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.config_file import CacheConfig, LexicalStoreConfig, VectorStoreConfig
from ragfabric_core.providers.base import ProviderError
from ragfabric_core.stores.base import Cache, LexicalStore, VectorStore
from ragfabric_core.stores.bm25_sql import Bm25Store
from ragfabric_core.stores.chroma_store import ChromaVectorStore
from ragfabric_core.stores.memory_cache import MemoryCache
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.stores.redis_cache import RedisCache


def _import_chromadb():
    """Indirection so a test can simulate the ``chroma`` extra being missing
    (monkeypatching this) without actually uninstalling chromadb."""
    import chromadb

    return chromadb


def build_vector_store(
    cfg: VectorStoreConfig,
    session_factory: Callable[[], Session],
    embedding_model: str | None = None,
) -> VectorStore:
    if cfg.kind in ("pgvector", "memory"):
        # "memory" keeps the v1 in process index for queries in this phase; the
        # fan out still writes to the relational table so nothing is lost.
        return PgVectorStore(session_factory, model=embedding_model)
    if cfg.kind == "chroma":
        # chromadb is a heavy import and lives behind an optional extra, so a
        # pgvector deployment never pays for it and a missing extra fails with
        # a clear, actionable error instead of a raw ImportError traceback.
        try:
            chromadb = _import_chromadb()
        except ImportError as exc:
            raise ProviderError(
                "chroma",
                "chromadb is not installed. Install it with: uv pip install 'ragfabric[chroma]'",
            ) from exc
        url = os.environ.get("CHROMA_URL", "http://localhost:8000")
        host, _, port = url.removeprefix("http://").removeprefix("https://").partition(":")
        client = chromadb.HttpClient(host=host, port=int(port or 8000))
        return ChromaVectorStore(client, model=embedding_model, session_factory=session_factory)
    raise NotImplementedError(f"unknown vector store kind {cfg.kind!r}")


def build_lexical_store(
    cfg: LexicalStoreConfig, session_factory: Callable[[], Session]
) -> LexicalStore:
    if cfg.kind == "postgres_fts":
        return PostgresLexicalStore(session_factory)
    if cfg.kind == "bm25_memory":
        try:
            from ragfabric_core.stores.bm25_memory import InMemoryBm25Store
        except ImportError as exc:
            raise ProviderError(
                "bm25_memory",
                "rank_bm25 is not installed. Install it with: uv pip install 'ragfabric[bm25]'",
            ) from exc
        return InMemoryBm25Store(max_chunks=cfg.max_chunks)
    if cfg.kind == "bm25":
        # k1 and b are BM25 ranking parameters and belong to the strategy that
        # ranks, not to the fan out that indexes. A store built here is built
        # for indexing, where they have no effect, so it takes the defaults;
        # default_registry constructs the ranking instance with the configured
        # values.
        return Bm25Store(session_factory)
    raise NotImplementedError(f"unknown lexical store kind {cfg.kind!r}")


def build_cache(cfg: CacheConfig, url: str | None = None) -> Cache:
    if cfg.kind == "redis":
        return RedisCache(url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    return MemoryCache()
