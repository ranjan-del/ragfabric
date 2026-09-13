"""Offline providers: the test doubles that keep CI free of keys and networks.

ScriptedLLMProvider replays canned responses so agent loops and routers can be
unit tested branch by branch. HashingEmbeddingProvider wraps the v1 hashing
embedder: deterministic, no download, useful for tests and for the "no key"
first run, not for production quality retrieval.
"""

from __future__ import annotations

import time

from ragfabric_core.ingest.embed import HashingEmbedder, tokenize
from ragfabric_core.providers.base import Completion, EmbeddingResult, Message, ProviderError


class ScriptedLLMProvider:
    name = "scripted"

    def __init__(self, responses: list[str], model: str = "scripted") -> None:
        self._responses = list(responses)
        self.default_model = model
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
        if not self._responses:
            raise ProviderError(self.name, "script exhausted: no more responses")
        started = time.perf_counter()
        text = self._responses.pop(0)
        self.calls += 1
        return Completion(
            text=text,
            model=model or self.default_model,
            provider=self.name,
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(text.split()),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason="stop",
        )


class HashingEmbeddingProvider:
    name = "offline"

    def __init__(self, dim: int | None = None) -> None:
        self._embedder = HashingEmbedder(dim=dim)
        self.dim = self._embedder.dim
        self.model = f"hashing-{self.dim}"

    def embed(self, texts: list[str]) -> EmbeddingResult:
        started = time.perf_counter()
        vectors = [row.tolist() for row in self._embedder.embed(texts)] if texts else []
        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            provider=self.name,
            input_tokens=sum(len(tokenize(t)) for t in texts),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
