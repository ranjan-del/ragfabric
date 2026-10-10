"""RagFabric's own strategies as evaluation targets (D4, D5, D12, D14, D15).

A ``StrategyTarget`` retrieves through a registered strategy and generates
through ``generate_for_result``, the function the API uses, so the evaluation
measures the code path users get. It retrieves as an unrestricted system
principal scoped to the evaluation collection (D12) at the batch's ``top_k``
(D14). Nothing is written to ``retrieval_runs`` (D10).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.config_file import RagFabricConfig, RerankerConfig
from ragfabric_core.evaluation.target import ContextItem, TargetAnswer, TargetSkipped
from ragfabric_core.generate.dispatch import generate_for_result
from ragfabric_core.models.document import Document
from ragfabric_core.pricing import PricingTable, estimate_cost
from ragfabric_core.providers.base import LLMProvider, ProviderError, is_offline
from ragfabric_core.providers.registry import answering_model
from ragfabric_core.rerank.registry import build_reranker
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrieverStrategy,
    StrategyName,
    StrategyParams,
    StrategyRegistry,
)
from ragfabric_core.strategies.registry_defaults import _int
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy

DEFAULT_TARGETS = ("traditional", "vectorless", "agentic", "graph", "auto")
RERANK_KINDS = ("none", "llm", "cross_encoder")
EVAL_PRINCIPAL = Principal(user_id=None, email=None, role="admin")


@dataclass(frozen=True)
class TargetSpec:
    strategy: StrategyName
    rerank: str | None = None

    @property
    def name(self) -> str:
        return f"{self.strategy}+rerank={self.rerank}" if self.rerank else str(self.strategy)


def parse_target_spec(text: str) -> TargetSpec:
    """``traditional``, ``graph``, ``auto`` or ``traditional+rerank=<none|llm|cross_encoder>``."""
    base, _, variant = text.strip().partition("+")
    try:
        strategy = StrategyName(base)
    except ValueError:
        known = ", ".join(DEFAULT_TARGETS)
        raise ValueError(f"unknown target {text!r}: expected one of {known}") from None
    if not variant:
        return TargetSpec(strategy)
    key, _, value = variant.partition("=")
    if strategy is not StrategyName.TRADITIONAL or key != "rerank" or value not in RERANK_KINDS:
        raise ValueError(
            f"unknown target {text!r}: only traditional takes a variant, "
            f"traditional+rerank=<{'|'.join(RERANK_KINDS)}>"
        )
    return TargetSpec(strategy, value)


class StrategyTarget:
    def __init__(
        self,
        name: str,
        strategy: RetrieverStrategy,
        llm: LLMProvider,
        *,
        collection_id: int | None,
        top_k: int,
        provider: str,
        llm_model: str | None,
        session_factory: Callable[[], Session],
        pricing: PricingTable | None = None,
    ) -> None:
        self.name = name
        self.strategy = strategy
        self.llm = llm
        self.collection_id = collection_id
        self.top_k = top_k
        self.provider = provider
        self.llm_model = llm_model
        self._session_factory = session_factory
        self._pricing = pricing
        self._names: dict[int, str] = {}

    def _document_name(self, document_id: int, metadata: dict) -> str:
        filename = metadata.get("filename")
        if isinstance(filename, str) and filename:
            return filename
        if document_id not in self._names:
            with self._session_factory() as db:
                doc = db.get(Document, document_id)
                self._names[document_id] = doc.filename if doc else f"document-{document_id}"
        return self._names[document_id]

    def answer(self, question: str) -> TargetAnswer:
        ctx = RetrievalContext(
            principal=EVAL_PRINCIPAL,
            access_filter=AccessFilter.unrestricted(),
            collection_ids=[self.collection_id] if self.collection_id is not None else None,
            params=StrategyParams(top_k=self.top_k),
        )
        started = time.perf_counter()
        result = self.strategy.retrieve(question, ctx)
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        generated = generate_for_result(question, result, self.llm)
        total_ms = int((time.perf_counter() - started) * 1000)
        input_tokens = result.input_tokens + generated.input_tokens
        output_tokens = result.output_tokens + generated.output_tokens
        cost = None
        if self._pricing is not None and self.llm_model:
            cost = estimate_cost(
                self._pricing, self.provider, self.llm_model, input_tokens, output_tokens
            ).usd
        return TargetAnswer(
            answer=generated.text,
            contexts=[
                ContextItem(
                    rank=i,
                    document=self._document_name(c.document_id, c.metadata),
                    text=c.text,
                    score=c.score,
                )
                for i, c in enumerate(result.chunks, start=1)
            ],
            strategy_used=str(result.strategy),
            fallback_from=str(result.fallback_from) if result.fallback_from else None,
            llm_calls=result.llm_calls + generated.llm_calls,
            retrieval_calls=result.retrieval_calls,
            embedding_calls=result.embedding_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=total_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=total_ms - retrieval_ms,
            llm_model=self.llm_model,
            estimated_cost_usd=cost,
            router_reasoning=result.router.reasoning if result.router else None,
        )


def _skip_reason(spec: TargetSpec, cfg: RagFabricConfig, llm: LLMProvider) -> str | None:
    if spec.strategy in (StrategyName.AGENTIC, StrategyName.GRAPH) and is_offline(llm):
        return f"the {spec.strategy} strategy needs a model and llm.provider is offline"
    if spec.strategy is StrategyName.GRAPH and not cfg.graph_store.enabled:
        return "graph_store.enabled is false, so no graph was extracted to walk"
    if spec.rerank == "llm" and is_offline(llm):
        return "the llm reranker needs a model and llm.provider is offline"
    return None


def _strategy_for(
    spec: TargetSpec, registry: StrategyRegistry, llm: LLMProvider, cfg: RagFabricConfig
) -> RetrieverStrategy:
    base = registry.get(spec.strategy)
    if spec.rerank is None:
        return base
    if not isinstance(base, TraditionalRAGStrategy):  # pragma: no cover - parse refuses it
        raise ValueError(f"{spec.name}: only the traditional strategy reranks")
    try:
        reranker = build_reranker(RerankerConfig(kind=spec.rerank), llm=llm)
    except (ImportError, ProviderError) as exc:
        raise TargetSkipped(f"the {spec.rerank} reranker is not available: {exc}") from exc
    return TraditionalRAGStrategy(
        embedding_provider=base.embedder,
        vector_store=base.store,
        reranker=reranker,
        # The same coercion default_registry applies, so the variant differs
        # from the shared strategy in its reranker only.
        max_context_tokens=_int(cfg.strategies.traditional.get("max_context_tokens"), 6000),
        generation_model=cfg.llm.model,
    )


def build_targets(
    specs: list[str],
    cfg: RagFabricConfig,
    registry: StrategyRegistry,
    llm: LLMProvider,
    *,
    collection_id: int | None,
    session_factory: Callable[[], Session],
    pricing: PricingTable | None = None,
) -> tuple[list[StrategyTarget], list[tuple[str, str]]]:
    """Targets that can run here, and ``(name, reason)`` for each that cannot."""
    targets: list[StrategyTarget] = []
    skipped: list[tuple[str, str]] = []
    model = answering_model(cfg.llm, llm)
    for text in specs:
        spec = parse_target_spec(text)
        reason = _skip_reason(spec, cfg, llm)
        if reason is None:
            try:
                strategy = _strategy_for(spec, registry, llm, cfg)
            except TargetSkipped as exc:
                reason = exc.reason
        if reason is not None:
            skipped.append((spec.name, reason))
            continue
        targets.append(
            StrategyTarget(
                spec.name,
                strategy,
                llm,
                collection_id=collection_id,
                top_k=cfg.evaluation.top_k,
                provider=cfg.llm.provider,
                llm_model=model,
                session_factory=session_factory,
                pricing=pricing,
            )
        )
    return targets, skipped
