"""Agentic RAG: the bounded loop behind the one result shape (ADR 0002).

Read this next to ``strategies/traditional.py`` and ``strategies/vectorless.py``.
Those two do a fixed amount of work: embed or not, query the stores, rank, cut,
return. This one decides how much work to do while it is doing it, and that
single difference is what the counters here have to survive.

**Counting an agent honestly (ADR 0004).** The number of calls is not knowable
before the run, so every counter is read back off what happened rather than
derived from the shape of the strategy:

``llm_calls``       the state's own spend counter, which the budget already
                    refuses to let drift.
``retrieval_calls`` one per tool invocation the loop actually made, not
                    iterations times sub-questions.
``embedding_calls`` measured across the run from the tools themselves. A tool
                    that embeds reports what it spent through an
                    ``embedding_calls`` counter; a tool that embeds nothing has
                    nothing to report and contributes zero. So an agent that
                    only ever reached for lexical search reports zero because
                    zero is what happened, exactly as the vectorless strategy
                    does.

**The pool is the result.** ``chunks`` is the evidence accumulated across every
iteration, best scored first, and it is deliberately not cut to ``top_k``: the
agent retrieved several times precisely because one retrieval was not enough,
and throwing away the later halves of that work would make the extra calls
pointless. The context budget is applied at generation, where it is applied for
every strategy.

**The access filter is the caller's, on every call.** Nothing here builds a
``RetrievalContext``. The one the request arrived with is handed to the loop and
from there to each tool, because ADR 0003 puts the filter inside the store query
and an agent that pools evidence across iterations has no way to un-pool a chunk
it should never have seen.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from ragfabric_core.agent.loop import DEFAULT_MAX_ITERATIONS, AgentRun, run_agent
from ragfabric_core.agent.state import NodeName
from ragfabric_core.agent.tools import ToolRegistry
from ragfabric_core.providers.base import LLMProvider
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    StrategyName,
)


@runtime_checkable
class CountsEmbeddings(Protocol):
    """A tool that reports what it spent on embeddings.

    Optional on purpose. A lexical tool has nothing to report and should not be
    made to carry a counter that is always zero, and a tool that does embed
    reports its own spend rather than having this strategy guess from the tool's
    name. Guessing from the name is how a counter starts describing the
    configuration instead of the run.
    """

    embedding_calls: int


class AgenticRAGStrategy:
    name = StrategyName.AGENTIC

    def __init__(
        self,
        *,
        llm: LLMProvider,
        tools: ToolRegistry,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        per_node_llm_calls: dict[NodeName, int] | None = None,
    ) -> None:
        self._llm = llm
        self._tools = dict(tools)
        self._max_iterations = max(1, max_iterations)
        self._per_node_llm_calls = dict(per_node_llm_calls or {})

    @property
    def tools(self) -> ToolRegistry:
        """The tools this agent was built with (read only).

        Exposed so a test can prove which tools a deployment actually wired up,
        rather than inferring it from what the agent happened to call.
        """
        return dict(self._tools)

    @property
    def max_iterations(self) -> int:
        """The iteration cap this agent was built with (read only).

        Exposed for the same reason as ``tools``: a deployment's limits should
        be checkable directly rather than inferred from how long a run happened
        to take, which would only prove the cap on a run that reached it.
        """
        return self._max_iterations

    @property
    def per_node_llm_calls(self) -> dict[NodeName, int]:
        """The per-node call caps this agent was built with (read only)."""
        return dict(self._per_node_llm_calls)

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        started = time.perf_counter()
        before = _embedding_spend(self._tools)

        run: AgentRun = run_agent(
            query,
            llm=self._llm,
            tools=self._tools,
            ctx=ctx,
            max_iterations=self._max_iterations,
            per_node_llm_calls=self._per_node_llm_calls,
        )

        return RetrievalResult(
            strategy=self.name,
            chunks=run.chunks,
            retrieval_calls=run.retrieval_calls,
            # Measured across this run, so a tool instance shared between
            # requests reports only what this request spent on it.
            embedding_calls=_embedding_spend(self._tools) - before,
            llm_calls=run.state.llm_calls,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            trace=run.trace,
            # What was and was not answered, per part of the question. The
            # other strategies leave this empty because they have no parts.
            sub_questions=run.sub_question_reports(),
        )


def _embedding_spend(tools: ToolRegistry) -> int:
    return sum(
        tool.embedding_calls for tool in tools.values() if isinstance(tool, CountsEmbeddings)
    )
