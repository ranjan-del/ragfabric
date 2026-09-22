"""Test doubles shared by the agent test modules.

The agent is a state machine over three things it does not own: a model, a set
of tools and a retrieval context. Each one is faked here rather than in every
test file, so a test says what it is exercising instead of restating the
scaffolding.

``RecordingLLM`` keeps the messages it was sent. That matters more than it
looks: several of the agent's guarantees live in the prompt (the tool routing
rules, the strict assessment rubric), and a test that only checks the parsed
response would pass even if the prompt stopped saying any of it.
"""

from __future__ import annotations

import time

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.providers.base import Completion, Message, ProviderError
from ragfabric_core.strategies.base import (
    Budget,
    RetrievalContext,
    RetrievedChunk,
    StrategyParams,
)


class RecordingLLM:
    """A scripted provider that remembers what it was asked.

    Separate from ``ScriptedLLMProvider`` because that double is shared with
    the rest of the suite and does not keep the prompts.
    """

    name = "recording"

    def __init__(
        self, *responses: str, model: str = "recording", provider: str = "recording"
    ) -> None:
        self.responses = list(responses)
        self.default_model = model
        # Named per instance so a test can speak as a model the pricing table
        # knows. The cost cap can only bind on a priced model, and a double
        # that could only ever be "recording" could not exercise that branch.
        self.name = provider
        self.prompts: list[list[Message]] = []
        self.calls = 0

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        self.prompts.append(list(messages))
        self.calls += 1
        if not self.responses:
            raise ProviderError(self.name, "script exhausted: no more responses")
        text = self.responses.pop(0)
        return Completion(
            text=text,
            model=model or self.default_model,
            provider=self.name,
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(text.split()),
            latency_ms=0,
        )

    def stream(self, messages, *, model=None, max_tokens=1024, temperature=0.0):
        yield from ()

    def last_prompt(self) -> str:
        return "\n".join(m.content for m in self.prompts[-1])


class FakeTool:
    """A tool that returns what the test told it to, and records how it was called.

    ``contexts`` is what the access filter assertions read: a tool that drops
    the caller's filter is invisible to a test that only inspects the chunks it
    returned.
    """

    def __init__(
        self,
        name: str,
        *,
        description: str = "a fake tool",
        chunks: list[RetrievedChunk] | None = None,
        embedding_calls_per_run: int = 0,
    ) -> None:
        self.name = name
        self.description = description
        self._chunks = list(chunks or [])
        self._per_run = embedding_calls_per_run
        self.queries: list[str] = []
        self.contexts: list[RetrievalContext] = []
        self.embedding_calls = 0
        self.retrieval_calls = 0

    def set_chunks(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = list(chunks)

    def run(self, query: str, ctx: RetrievalContext) -> list[RetrievedChunk]:
        self.queries.append(query)
        self.contexts.append(ctx)
        self.retrieval_calls += 1
        self.embedding_calls += self._per_run
        return list(self._chunks)


class SequenceTool(FakeTool):
    """Returns a different batch on each call, then repeats the last one forever.

    Written for the progress tests: an agent that keeps retrieving the same
    chunks must be detected, and that needs a tool whose later calls are
    deliberately unproductive.
    """

    def __init__(self, name: str, batches: list[list[RetrievedChunk]], **kwargs) -> None:
        super().__init__(name, **kwargs)
        self._batches = [list(batch) for batch in batches]

    def run(self, query: str, ctx: RetrievalContext) -> list[RetrievedChunk]:
        self.queries.append(query)
        self.contexts.append(ctx)
        self.retrieval_calls += 1
        self.embedding_calls += self._per_run
        index = min(len(self.queries) - 1, len(self._batches) - 1)
        return list(self._batches[index]) if self._batches else []


class SlowTool(FakeTool):
    """A tool that costs measurable wall clock, for the latency assertions."""

    def __init__(self, name: str, *, delay_s: float = 0.005, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self._delay = delay_s

    def run(self, query: str, ctx: RetrievalContext) -> list[RetrievedChunk]:
        time.sleep(self._delay)
        return super().run(query, ctx)


def chunk(
    chunk_id: int,
    *,
    text: str | None = None,
    document_id: int = 1,
    collection_id: int | None = None,
    score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        collection_id=collection_id,
        text=text if text is not None else f"chunk {chunk_id}",
        score=score,
    )


def ctx(
    *,
    top_k: int = 5,
    access: AccessFilter | None = None,
    max_llm_calls: int = 12,
    max_latency_ms: int = 30_000,
    max_cost_usd: float = 0.10,
) -> RetrievalContext:
    return RetrievalContext(
        principal=Principal(user_id=1, email="engineer@example.com"),
        access_filter=access or AccessFilter.unrestricted(),
        params=StrategyParams(top_k=top_k),
        budget=Budget(
            max_llm_calls=max_llm_calls,
            max_latency_ms=max_latency_ms,
            max_cost_usd=max_cost_usd,
        ),
    )


def restricted_ctx(*, document_ids: set[int], top_k: int = 5) -> RetrievalContext:
    return ctx(top_k=top_k, access=AccessFilter(document_ids=frozenset(document_ids)))
