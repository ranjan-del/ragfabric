"""Build the strategy registry from configuration.

One place assembles a strategy from config so the API, the CLI and the tests all
get the same object graph. Phases 4 to 6 register their strategies here too.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.agent.tools import (
    FetchDocumentTool,
    LexicalSearchTool,
    SemanticSearchTool,
    build_tool_registry,
)
from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.providers.base import LLMProvider
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider
from ragfabric_core.rerank.registry import build_reranker
from ragfabric_core.stores.bm25_sql import Bm25Store
from ragfabric_core.stores.document_chunks import SqlDocumentChunkReader
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.stores.registry import build_vector_store
from ragfabric_core.strategies.agentic import AgenticRAGStrategy
from ragfabric_core.strategies.base import StrategyRegistry
from ragfabric_core.strategies.graph import GraphRAGStrategy
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
    traditional = TraditionalRAGStrategy(
        embedding_provider=embedder,
        vector_store=store,
        reranker=reranker,
        max_context_tokens=_int(cfg.strategies.traditional.get("max_context_tokens"), 6000),
        generation_model=cfg.llm.model,
    )
    vectorless = _build_vectorless(cfg, session_factory)
    registry.register(traditional)
    registry.register(vectorless)
    registry.register(_build_agentic(cfg, traditional, vectorless, session_factory, llm=llm))
    registry.register(_build_graph(cfg, session_factory, llm=llm))
    return registry


def _build_agentic(
    cfg: RagFabricConfig,
    traditional: TraditionalRAGStrategy,
    vectorless: VectorlessRAGStrategy,
    session_factory: Callable[[], Session],
    *,
    llm: LLMProvider | None,
) -> AgenticRAGStrategy:
    """The agent over the two strategies this deployment already builds.

    The tools wrap the strategy instances the registry just constructed rather
    than new ones, so the agent searches through exactly the stores, reranker
    and embedder the other strategies use. A second set built here would drift
    from them the moment configuration changed.

    The LLM is built unconditionally, unlike the reranker's, because an agent
    with no model cannot plan, assess or repair. Building a provider makes no
    call, so a deployment that never asks for the agentic strategy pays nothing
    for it being registered.

    Every field of ``AgenticConfig`` is passed on. Four of them once were not,
    and a limit that is loaded, validated and printed back by ``config
    validate`` while nothing reads it is worse than no limit: the operator
    edits it, sees it echoed, and gets the old behaviour.
    """
    settings = cfg.strategies.agentic
    return AgenticRAGStrategy(
        llm=llm if llm is not None else build_llm_provider(cfg.llm),
        tools=build_tool_registry(
            [
                SemanticSearchTool(traditional),
                LexicalSearchTool(vectorless),
                FetchDocumentTool(SqlDocumentChunkReader(session_factory)),
            ],
            enabled=settings.tools,
        ),
        max_iterations=settings.max_iterations,
        per_node_llm_calls=settings.node_caps(),
        max_llm_calls=settings.max_llm_calls,
        max_latency_ms=settings.max_latency_ms,
        max_cost_usd=settings.max_cost_usd,
        assess_strictness=settings.assess_strictness,
    )


def _build_graph(
    cfg: RagFabricConfig,
    session_factory: Callable[[], Session],
    *,
    llm: LLMProvider | None,
) -> GraphRAGStrategy:
    """The graph strategy, bounded by ``cfg.strategies.graph``.

    Both fields of ``GraphStrategyConfig`` are passed on; a test iterates over
    the model's fields and fails for any that does not reach the strategy.

    The LLM is built unconditionally, the same reasoning as the agentic
    strategy's: entity extraction cannot run without a model, and building a
    provider makes no call, so a deployment that never asks for the graph
    strategy pays nothing for it being registered.
    """
    return GraphRAGStrategy(
        llm=llm if llm is not None else build_llm_provider(cfg.llm),
        session_factory=session_factory,
        max_hops=cfg.strategies.graph.max_hops,
        node_budget=cfg.strategies.graph.node_budget,
    )


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
