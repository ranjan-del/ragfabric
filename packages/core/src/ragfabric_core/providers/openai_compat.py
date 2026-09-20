"""OpenAI and OpenAI compatible providers (Ollama, Azure, vLLM, LM Studio).

One class, several vendors: Ollama serves the OpenAI chat and embeddings API at
``/v1``, so the same code talks to a hosted model or a local one. Vendor
differences live in small subclasses that only set defaults.

Token accounting comes from ``usage`` on the response; both OpenAI and Ollama
populate it. Reasoning model families reject a ``temperature`` argument, so it
is only sent for models outside REASONING_MODEL_PREFIXES.

``max_completion_tokens`` is used rather than the deprecated ``max_tokens``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from ragfabric_core.providers.base import Completion, EmbeddingResult, Message, ProviderError

REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _build_client(api_key: str, base_url: str | None) -> Any:
    from openai import OpenAI  # imported lazily so the module loads without the SDK

    return OpenAI(api_key=api_key, base_url=base_url)


class OpenAICompatibleLLM:
    def __init__(
        self,
        name: str,
        api_key: str,
        default_model: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(
                name, "api_key is required (set the provider's key in the environment)"
            )
        self.name = name
        self.default_model = default_model
        self._client = client or _build_client(api_key, base_url)

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        chosen = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": chosen,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": max_tokens,
        }
        if not chosen.startswith(REASONING_MODEL_PREFIXES):
            kwargs["temperature"] = temperature
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # the SDK raises many types; callers get one
            raise ProviderError(self.name, str(exc)) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not response.choices:
            raise ProviderError(self.name, "no choices in response")
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            text=(choice.message.content or ""),
            model=getattr(response, "model", None) or chosen,
            provider=self.name,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency_ms,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yield text deltas from a server sent event stream.

        Shared by ``OpenAIProvider`` and ``OllamaProvider``, since Ollama serves
        the same streamed chunk shape at its OpenAI compatible endpoint. Usage
        is not reported on streamed chunks by either vendor, so no token count
        is returned here; the caller accounts for that honestly rather than
        estimating (ADR 0004).
        """
        chosen = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": chosen,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        if not chosen.startswith(REASONING_MODEL_PREFIXES):
            kwargs["temperature"] = temperature
        try:
            chunks = self._client.chat.completions.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # the SDK raises many types; callers get one
            raise ProviderError(self.name, str(exc)) from exc
        for chunk in chunks:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta


class OpenAIProvider(OpenAICompatibleLLM):
    def __init__(
        self, api_key: str, default_model: str = "gpt-5.4-mini", client: Any | None = None
    ) -> None:
        super().__init__("openai", api_key, default_model, base_url=None, client=client)


class OllamaProvider(OpenAICompatibleLLM):
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        default_model: str = "llama3.2",
        client: Any | None = None,
    ) -> None:
        # Ollama ignores the key but the SDK requires a non empty string.
        super().__init__("ollama", "ollama", default_model, base_url=base_url, client=client)


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        dim: int,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(
                name, "api_key is required (set the provider's key in the environment)"
            )
        self.name = name
        self.model = model
        self.dim = dim
        self._client = client or _build_client(api_key, base_url)

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=[], model=self.model, provider=self.name, input_tokens=0, latency_ms=0
            )
        started = time.perf_counter()
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        usage = getattr(response, "usage", None)
        return EmbeddingResult(
            vectors=[list(map(float, item.embedding)) for item in ordered],
            model=getattr(response, "model", None) or self.model,
            provider=self.name,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class OpenAIEmbeddingProvider(OpenAICompatibleEmbeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        client: Any | None = None,
    ) -> None:
        super().__init__("openai", api_key, model, dim, base_url=None, client=client)


class OllamaEmbeddingProvider(OpenAICompatibleEmbeddings):
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "nomic-embed-text",
        dim: int = 768,
        client: Any | None = None,
    ) -> None:
        super().__init__("ollama", "ollama", model, dim, base_url=base_url, client=client)
