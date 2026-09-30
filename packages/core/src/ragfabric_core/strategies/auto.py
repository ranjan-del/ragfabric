"""The auto strategy: route, run, fall back at most once, and say what happened.

Registered like the four strategies it routes between, so every entry point
accepts ``strategy: auto`` without its own wiring (ADR 0013). It returns the
result of the strategy that actually ran, with the decision attached, and
never reports itself as the strategy (ADR 0002).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ragfabric_core.providers.base import LLMProvider
from ragfabric_core.router.classifier import ClassifierOutcome, classify, one_sentence
from ragfabric_core.router.decision import RouterDecision, levels_for
from ragfabric_core.router.fallback import combine, empty_reason, fallback_for, fuse
from ragfabric_core.router.signals import Proposal, extract_signals, propose
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    StrategyName,
    StrategyRegistry,
    TraceSpan,
)

# Request filters the graph walk cannot apply. /api/ask refuses them with a 422
# when a caller names graph; under auto the graph is left out instead.
_GRAPH_BLIND_FILTERS = ("document_id", "format")


class AutoStrategy:
    name = StrategyName.AUTO

    def __init__(
        self,
        *,
        registry: StrategyRegistry,
        llm: LLMProvider | None,
        min_confidence: float,
        classifier_model: str | None,
        graph_enabled: bool,
        relation_types: Sequence[str],
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._min_confidence = min_confidence
        self._classifier_model = classifier_model
        self._graph_enabled = graph_enabled
        self._relation_types = list(relation_types)

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    @property
    def classifier_model(self) -> str | None:
        return self._classifier_model

    @property
    def graph_enabled(self) -> bool:
        return self._graph_enabled

    def available(self, ctx: RetrievalContext) -> list[StrategyName]:
        names = [n for n in self._registry.names() if n is not StrategyName.AUTO]
        filters = ctx.params.metadata_filters
        if not self._graph_enabled or any(key in filters for key in _GRAPH_BLIND_FILTERS):
            names = [n for n in names if n is not StrategyName.GRAPH]
        return names

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        started = time.perf_counter()
        available = self.available(ctx)
        signals = extract_signals(query, relation_types=self._relation_types)
        proposal = propose(signals, available=available)
        outcome: ClassifierOutcome | None = None

        if proposal.decisive:
            decision = _from_proposal(proposal, source="signals")
        elif self._llm is None or ctx.budget.max_llm_calls < 1:
            decision = _from_proposal(proposal, source="signals_fallback")
        else:
            outcome = classify(
                query,
                signals,
                llm=self._llm,
                available=available,
                model=self._classifier_model,
            )
            decision = self._from_classifier(proposal, outcome)

        inner = _deduct_calls(ctx, outcome.llm_calls if outcome else 0)
        chosen = StrategyName(decision.selected_strategy)
        failure: str | None = None  # why the routed strategy's attempt was replaced, if it was
        next_strategy = None
        if decision.fused:
            traditional = self._run(StrategyName.TRADITIONAL, query, inner)
            try:
                vectorless = self._run(
                    StrategyName.VECTORLESS, query, _deduct_calls(inner, traditional.llm_calls)
                )
            except Exception as exc:
                # Keep the Traditional leg. The failed leg's spend is unknown,
                # so it is counted as 0 (see _attempt).
                failure = _describe(exc)
                result = traditional
            else:
                result = fuse(traditional, vectorless, top_k=ctx.params.top_k)
            reason = failure or ""
        else:
            attempt = self._attempt(chosen, query, inner)
            if isinstance(attempt, Exception):
                failure = _describe(attempt)
                # Nothing to add to the fallback's counters: an exception carries
                # no result, so what the failed strategy spent cannot be known.
                # Only what can be counted is counted, which here is 0 for it;
                # the fallback still gets the full remaining budget, so the
                # caller's limit is held by the classifier deduction alone.
                next_strategy = StrategyName.TRADITIONAL
                result = self._run(next_strategy, query, inner).model_copy(
                    update={"fallback_from": chosen}
                )
                reason = failure
            else:
                result = attempt
                next_strategy = fallback_for(chosen, result)
                reason = empty_reason(result) if next_strategy else ""
                if next_strategy is not None:
                    # What the first attempt spent is gone, so the fallback gets only the rest.
                    remaining = _deduct_calls(inner, result.llm_calls)
                    result = combine(
                        result, self._run(next_strategy, query, remaining), fallback_from=chosen
                    )

        span = TraceSpan(
            name="router",
            started_ms=0,
            # Routing's own time only. Clamped because the two clocks are read
            # at different moments and TraceSpan refuses a negative duration.
            duration_ms=max(0, int((time.perf_counter() - started) * 1000) - result.latency_ms),
            attributes={
                "selected": decision.selected_strategy,
                "source": decision.source,
                "decisive": decision.decisive,
                "fused": decision.fused,
                "fallback_from": str(chosen) if next_strategy else None,
                "fallback_reason": reason or None,
                "classifier_violation": outcome.violation.error
                if outcome and outcome.violation
                else None,
                "classifier_error": outcome.error if outcome else None,
            },
        )
        extra = outcome or ClassifierOutcome()
        return result.model_copy(
            update={
                "router": decision,
                "trace": [span, *result.trace],
                "llm_calls": result.llm_calls + extra.llm_calls,
                "input_tokens": result.input_tokens + extra.input_tokens,
                "output_tokens": result.output_tokens + extra.output_tokens,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )

    def _attempt(
        self, name: StrategyName, query: str, ctx: RetrievalContext
    ) -> RetrievalResult | Exception:
        """Run a routed strategy, returning its failure instead of raising it.

        Traditional is the floor everything falls back to, so its failure is
        the request's failure and propagates, as it would if the caller had
        named it.
        """
        if name is StrategyName.TRADITIONAL:
            return self._run(name, query, ctx)
        try:
            return self._run(name, query, ctx)
        except Exception as exc:
            return exc

    def _run(self, name: StrategyName, query: str, ctx: RetrievalContext) -> RetrievalResult:
        if name is StrategyName.AUTO:
            raise RuntimeError("auto cannot route to itself")
        return self._registry.get(name).retrieve(query, ctx)

    def _from_classifier(self, proposal: Proposal, outcome: ClassifierOutcome) -> RouterDecision:
        if outcome.error is not None:
            return _from_proposal(proposal, source="signals_fallback")
        reply = outcome.reply
        if reply is None or reply.confidence < self._min_confidence:
            return RouterDecision(
                selected_strategy="traditional",
                source="classifier",
                decisive=False,
                confidence=reply.confidence if reply else None,
                reasoning=(
                    "The question was unclear, so meaning and exact word search were combined."
                ),
                query_type=reply.query_type if reply else "ambiguous",
                fused=True,
                **levels_for("traditional"),
            )
        return RouterDecision(
            selected_strategy=reply.strategy,
            source="classifier",
            decisive=False,
            confidence=reply.confidence,
            reasoning=one_sentence(reply.reasoning),
            query_type=reply.query_type,
            **levels_for(reply.strategy),
        )


def _from_proposal(proposal: Proposal, *, source: str) -> RouterDecision:
    strategy = str(proposal.strategy)
    return RouterDecision(
        selected_strategy=strategy,
        source=source,
        decisive=proposal.decisive,
        confidence=None,
        reasoning=one_sentence(proposal.reasons[0]),
        query_type=proposal.query_type,
        **levels_for(strategy),
    )


def _describe(exc: Exception) -> str:
    """``error: <Type>: <message>``, at most 200 characters, for the router span."""
    return f"error: {type(exc).__name__}: {exc}"[:200]


def _deduct_calls(ctx: RetrievalContext, calls: int) -> RetrievalContext:
    """The caller's budget less what routing spent, so llm_calls stays within it.

    ``model_copy`` keeps the principal and the access filter untouched (ADR 0003).
    """
    if calls == 0:
        return ctx
    budget = ctx.budget.model_copy(
        update={"max_llm_calls": max(0, ctx.budget.max_llm_calls - calls)}
    )
    return ctx.model_copy(update={"budget": budget})
