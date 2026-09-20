"""Traditional RAG: embed the question, search one vector index, rerank, fit a budget.

The order of operations is deliberate and is asserted by the tests:

  embed -> store query with the access filter inside it -> similarity threshold
  -> rerank -> cut to top_k -> context budget

The threshold runs before the rerank because it is expressed against the
retrieval score; comparing a reranker's score to a cosine threshold would make
the configured number meaningless. The store is asked for more candidates than
top_k so the reranker has something to improve, and the cut to top_k happens
after reranking, so with the noop reranker the behaviour is identical to asking
for top_k directly.

This strategy does not generate the answer. It returns a RetrievalResult, the
one shape every strategy returns (ADR 0002), and generation happens above it so
that all four strategies are generated and cited by identical code.
"""

from __future__ import annotations

import time

from ragfabric_core.providers.base import EmbeddingProvider
from ragfabric_core.rerank.base import Reranker
from ragfabric_core.stores.base import VectorStore
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    StrategyName,
    TraceSpan,
)
from ragfabric_core.tokens import fit_to_budget, token_count_method

DEFAULT_CANDIDATE_MULTIPLIER = 3


class TraditionalRAGStrategy:
    name = StrategyName.TRADITIONAL

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        max_context_tokens: int = 6000,
        generation_model: str | None = None,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        self._embedder = embedding_provider
        self._store = vector_store
        self._reranker = reranker
        self._max_context_tokens = max_context_tokens
        self._generation_model = generation_model
        self._multiplier = max(1, candidate_multiplier)

    @property
    def store(self) -> VectorStore:
        """The vector store this strategy was built with (read only).

        A later task needs to reach the store a running strategy holds, for
        example to report its backend or its pinned embedding model, and must
        never reach for a private attribute to get it.
        """
        return self._store

    @property
    def embedder(self) -> EmbeddingProvider:
        """The embedding provider this strategy was built with (read only).

        Per request tuning (Task 12) may build a fresh strategy instance
        around a different reranker without mutating the shared registry
        instance; that fresh instance must embed with the exact same
        provider the shared one uses, not an equivalent-by-config copy, so
        this is read off here rather than rebuilt from configuration.
        """
        return self._embedder

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        started = time.perf_counter()
        spans: list[TraceSpan] = []

        def span(name: str, begin: float, **attributes) -> None:
            spans.append(
                TraceSpan(
                    name=name,
                    started_ms=int((begin - started) * 1000),
                    duration_ms=int((time.perf_counter() - begin) * 1000),
                    attributes=attributes,
                )
            )

        mark = time.perf_counter()
        embedded = self._embedder.embed([query])
        span("embed_query", mark, model=self._embedder.model, dim=self._embedder.dim)

        filters: dict = {}
        if ctx.collection_ids:
            filters["collection_id"] = ctx.collection_ids[0]
        filters.update(dict(ctx.params.metadata_filters))

        mark = time.perf_counter()
        candidates = self._store.query(
            embedded.vectors[0],
            top_k=ctx.params.top_k * self._multiplier,
            access=ctx.access_filter,
            filters=filters or None,
        )
        span(
            "vector_search",
            mark,
            store=self._store.name,
            requested=ctx.params.top_k * self._multiplier,
            returned=len(candidates),
        )

        threshold = ctx.params.similarity_threshold
        if threshold > 0.0:
            candidates = [c for c in candidates if (c.score or 0.0) >= threshold]

        # Captured before the rerank call, not after: LlmReranker.rerank() returns
        # [] immediately on an empty candidate list without ever calling the LLM,
        # so llm_calls must reflect whether there was anything to score, not just
        # which reranker is configured.
        had_candidates = bool(candidates)

        if self._reranker is not None and self._reranker.name != "none":
            mark = time.perf_counter()
            candidates = self._reranker.rerank(query, candidates, ctx.params.top_k)
            span("rerank", mark, reranker=self._reranker.name, kept=len(candidates))
        else:
            candidates = candidates[: ctx.params.top_k]

        mark = time.perf_counter()
        chunks, tokens_used = fit_to_budget(
            candidates, self._max_context_tokens, self._generation_model
        )
        span(
            "context_budget",
            mark,
            max_context_tokens=self._max_context_tokens,
            tokens_used=tokens_used,
            token_count_method=token_count_method(self._generation_model),
            kept=len(chunks),
        )

        return RetrievalResult(
            strategy=self.name,
            chunks=chunks,
            retrieval_calls=1,
            # This assumes LlmReranker makes exactly one model call per non-empty
            # invocation, which is true today (rerank/llm_reranker.py: a single
            # self._llm.complete() call per rerank()). A future reranker that
            # batches across chunks or caches by content would need the true
            # count to come from the reranker itself, not from this name check,
            # so this flag should not be extended to cover that case blindly.
            llm_calls=1
            if (had_candidates and self._reranker is not None and self._reranker.name == "llm")
            else 0,
            input_tokens=embedded.input_tokens,
            output_tokens=0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            trace=spans,
        )
