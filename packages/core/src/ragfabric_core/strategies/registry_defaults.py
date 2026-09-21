"""Build the strategy registry from configuration.

One place assembles a strategy from config so the API, the CLI and the tests all
get the same object graph. Phases 4 to 6 register their strategies here too.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider
from ragfabric_core.rerank.registry import build_reranker
from ragfabric_core.stores.bm25_sql import Bm25Store
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.stores.registry import build_vector_store
from ragfabric_core.strategies.base import StrategyRegistry
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy
from ragfabric_core.strategies.vectorless import VectorlessRAGStrategy


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def default_registry(
    cfg: RagFabricConfig, session_factory: Callable[[], Session]
) -> StrategyRegistry:
    embedder = build_embedding_provider(cfg.embeddings)
    store = build_vector_store(cfg.vector_store, session_factory, embedding_model=embedder.model)
    llm = build_llm_provider(cfg.llm) if cfg.reranker.kind == "llm" else None
    reranker = build_reranker(cfg.reranker, llm=llm)
    registry = StrategyRegistry()
    registry.register(
        TraditionalRAGStrategy(
            embedding_provider=embedder,
            vector_store=store,
            reranker=reranker,
            max_context_tokens=_int(cfg.strategies.traditional.get("max_context_tokens"), 6000),
            generation_model=cfg.llm.model,
        )
    )
    registry.register(_build_vectorless(cfg, session_factory))
    return registry


def _build_vectorless(
    cfg: RagFabricConfig, session_factory: Callable[[], Session]
) -> VectorlessRAGStrategy:
    """Two lexical stores over the same chunk_search table, ranked differently.

    Both are constructed here rather than through build_lexical_store, because
    this strategy needs one specific pair (BM25 and ts_rank_cd) regardless of
    which single lexical store the ingestion fan out is configured to write
    through. Asking the registry for "the" lexical store would give one of the
    two and leave the fusion with nothing to fuse.
    """
    settings = cfg.strategies.vectorless
    return VectorlessRAGStrategy(
        bm25_store=Bm25Store(session_factory, k1=settings.k1, b=settings.b),
        ts_rank_store=PostgresLexicalStore(session_factory),
        phrase_boost=settings.phrase_boost,
        identifier_boost=settings.identifier_boost,
        fusion_k=settings.fusion_k,
        fusion_weights=settings.fusion_weights,
        max_context_tokens=settings.max_context_tokens,
        generation_model=cfg.llm.model,
    )
