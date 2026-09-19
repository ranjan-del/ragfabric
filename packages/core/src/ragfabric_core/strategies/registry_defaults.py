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
from ragfabric_core.stores.registry import build_vector_store
from ragfabric_core.strategies.base import StrategyRegistry
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy


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
    return registry
