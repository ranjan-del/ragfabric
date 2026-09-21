"""Vectorless RAG: two lexical rankings, fused, boosted, budgeted. No embeddings.

The order of operations is deliberate and is asserted by the tests:

  BM25 store query      (access filter INSIDE, metadata filters INSIDE)
  ts_rank_cd store query (access filter INSIDE, metadata filters INSIDE)
  -> RRF fuse the two ranked lists
  -> apply phrase and identifier boosts
  -> cut to top_k
  -> context budget

Read this next to strategies/traditional.py. The shape is deliberately the
same, minus the embedding call, so the two are comparable. Three differences
matter.

There is no embedding call, so there is no embedding provider on this class at
all. That is stronger than remembering not to call one.

There is no similarity threshold. BM25 scores are unbounded and corpus
dependent: a fixed floor means one thing on one corpus and something else on
the next, so honouring ``similarity_threshold`` here would be honouring a
number that cannot mean anything. It is ignored, and said so here rather than
silently dropped.

There are two store queries, not one. BM25 and ts_rank_cd fail differently:
BM25's IDF tells a rare identifier from a common word, while ts_rank_cd
rewards proximity and phrase cohesion and has no corpus-wide term at all.
Fusing them by rank (stores/fusion.py) needs no shared scale.

**The cut to top_k happens after fusion, never per store.** Cutting each store
to top_k first would throw away exactly the evidence fusion exists to use: a
chunk ranked sixth by both stores can legitimately beat a chunk ranked first
by one and absent from the other, and that comparison is impossible once both
lists have been truncated to three.

**On counting honestly (ADR 0004).** No embedding call is made, so the
embedding cost is zero, reported as zero rather than omitted or defaulted.
``llm_calls`` is zero because this strategy makes none: generation happens
above every strategy (ADR 0002), so nothing here calls a model.
"""

from __future__ import annotations

import time

from ragfabric_core.stores.base import LexicalStore
from ragfabric_core.stores.boosting import apply_boosts
from ragfabric_core.stores.fusion import DEFAULT_K, rrf
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    StrategyName,
    TraceSpan,
)
from ragfabric_core.tokens import fit_to_budget, token_count_method

# Each store is asked for more than top_k so fusion has something to fuse. With
# one candidate each there is nothing for RRF to disagree about.
DEFAULT_CANDIDATE_MULTIPLIER = 3

DEFAULT_PHRASE_BOOST = 2.0
DEFAULT_IDENTIFIER_BOOST = 3.0


class VectorlessRAGStrategy:
    name = StrategyName.VECTORLESS

    def __init__(
        self,
        bm25_store: LexicalStore,
        ts_rank_store: LexicalStore,
        phrase_boost: float = DEFAULT_PHRASE_BOOST,
        identifier_boost: float = DEFAULT_IDENTIFIER_BOOST,
        fusion_k: int = DEFAULT_K,
        fusion_weights: tuple[float, float] = (1.0, 1.0),
        max_context_tokens: int = 6000,
        generation_model: str | None = None,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        self._bm25 = bm25_store
        self._ts_rank = ts_rank_store
        self._phrase_boost = phrase_boost
        self._identifier_boost = identifier_boost
        self._fusion_k = fusion_k
        self._fusion_weights = list(fusion_weights)
        self._max_context_tokens = max_context_tokens
        self._generation_model = generation_model
        self._multiplier = max(1, candidate_multiplier)

    @property
    def bm25_store(self) -> LexicalStore:
        """The BM25 store this strategy was built with (read only)."""
        return self._bm25

    @property
    def ts_rank_store(self) -> LexicalStore:
        """The ts_rank_cd store this strategy was built with (read only)."""
        return self._ts_rank

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

        filters: dict = {}
        if ctx.collection_ids:
            filters["collection_id"] = ctx.collection_ids[0]
        filters.update(dict(ctx.params.metadata_filters))

        candidates = ctx.params.top_k * self._multiplier

        mark = time.perf_counter()
        bm25_hits = self._bm25.search(
            query, top_k=candidates, access=ctx.access_filter, filters=filters or None
        )
        span(
            "bm25_search",
            mark,
            store=self._bm25.name,
            requested=candidates,
            returned=len(bm25_hits),
        )

        mark = time.perf_counter()
        ts_hits = self._ts_rank.search(
            query, top_k=candidates, access=ctx.access_filter, filters=filters or None
        )
        span(
            "ts_rank_search",
            mark,
            store=self._ts_rank.name,
            requested=candidates,
            returned=len(ts_hits),
        )

        mark = time.perf_counter()
        fused = rrf([bm25_hits, ts_hits], k=self._fusion_k, weights=self._fusion_weights)
        span("fuse", mark, k=self._fusion_k, fused=len(fused))

        mark = time.perf_counter()
        boosted = apply_boosts(fused, query, self._phrase_boost, self._identifier_boost)
        span(
            "boost",
            mark,
            phrase_boost=self._phrase_boost,
            identifier_boost=self._identifier_boost,
        )

        # After fusion and after boosting, never before either. See the module
        # docstring for why cutting per store loses the comparison RRF makes.
        top = boosted[: ctx.params.top_k]

        mark = time.perf_counter()
        chunks, tokens_used = fit_to_budget(top, self._max_context_tokens, self._generation_model)
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
            retrieval_calls=2,
            # Zero because nothing was called, not because nothing was counted.
            # This strategy embeds nothing and generates nothing (ADR 0004).
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            trace=spans,
        )
