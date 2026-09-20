"""Provider interfaces.

Why so small: strategies need exactly two things from the outside world, a
completion and an embedding. Everything vendor specific (retries, auth, model
names, token accounting quirks) stays inside the provider module, so a strategy
written against these protocols runs unchanged on OpenAI, Anthropic, Ollama or
the offline doubles used in tests.

Token counts are copied from the provider's response when it reports them and
are never estimated silently; a provider that cannot report tokens documents
how it counts.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ProviderError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Completion(BaseModel):
    text: str
    model: str
    provider: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    finish_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    default_model: str

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion: ...

    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]: ...


class EmbeddingResult(BaseModel):
    vectors: list[list[float]]
    model: str
    provider: str
    input_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> EmbeddingResult: ...
