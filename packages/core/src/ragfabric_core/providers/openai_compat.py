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

import json
import time
from collections.abc import Iterator
from typing import Any

from ragfabric_core.providers.base import (
    Completion,
    EmbeddingResult,
    Message,
    ProviderError,
    StreamUsage,
)

REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _import_openai():
    """Indirection so a test can simulate the ``openai`` extra being missing
    (monkeypatching this) without actually uninstalling the package.

    Both ``OpenAIProvider`` and ``OllamaProvider`` go through this: Ollama
    serves the OpenAI chat and embeddings API at ``/v1``, so the same SDK
    client talks to either, and both need the ``openai`` extra installed.
    """
    from openai import OpenAI

    return OpenAI


def _build_client(name: str, api_key: str, base_url: str | None) -> Any:
    try:
        openai_cls = _import_openai()
    except ImportError as exc:
        raise ProviderError(
            name, "openai is not installed. Install it with: pip install 'ragfabric[openai]'"
        ) from exc
    return openai_cls(api_key=api_key, base_url=base_url)


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
        self._client = client or _build_client(name, api_key, base_url)

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
        usage: StreamUsage | None = None,
    ) -> Iterator[str]:
        """Yield text deltas from a server sent event stream.

        ``stream_options.include_usage`` asks for a final chunk that carries
        ``usage`` and no choices; its counts fill ``usage`` when one is given.
        A server that ignores the option leaves ``usage`` unknown (``None``).
        """
        chosen = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": chosen,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
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
            reported = getattr(chunk, "usage", None)
            if reported is not None and usage is not None:
                usage.input_tokens = int(getattr(reported, "prompt_tokens", 0) or 0)
                usage.output_tokens = int(getattr(reported, "completion_tokens", 0) or 0)
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


DEFAULT_OLLAMA_NUM_CTX = 8192


class _UrllibTransport:
    """Standard library HTTP for Ollama's native API: no extra dependency."""

    def __init__(self, timeout: float = 600.0) -> None:
        self.timeout = timeout

    def _open(self, url: str, body: dict) -> Any:
        import urllib.request

        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        return urllib.request.urlopen(request, timeout=self.timeout)

    def post(self, url: str, body: dict) -> dict:
        with self._open(url, body) as response:
            return json.loads(response.read().decode("utf-8"))

    def stream(self, url: str, body: dict) -> Iterator[str]:
        with self._open(url, body) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line:
                    yield line


def _native_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


class OllamaProvider:
    """Ollama chat through the native ``/api/chat`` endpoint.

    Why not ``/v1`` like the embeddings: Ollama's OpenAI compatible endpoint
    ignores ``num_ctx`` in the request and runs every prompt in the model's
    default window, cutting a longer prompt silently and keeping its end. The
    system prompt and the first ranked passages are what is lost. Measured on
    Ollama 0.34: a 6,400 token prompt arrived as 2,050 tokens through ``/v1``
    and whole (6,443) through ``/api/chat`` with ``options.num_ctx`` 8192.

    ``num_ctx`` is sent on every call; ``None`` leaves the model's own default.
    Keeping one value for every call also avoids Ollama reloading the model
    each time the window changes.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        default_model: str = "llama3.2",
        num_ctx: int | None = DEFAULT_OLLAMA_NUM_CTX,
        transport: Any | None = None,
    ) -> None:
        self.name = "ollama"
        self.default_model = default_model
        self.num_ctx = num_ctx
        self._chat_url = _native_root(base_url) + "/api/chat"
        self._transport = transport or _UrllibTransport()

    def _body(
        self,
        messages: list[Message],
        model: str | None,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return {
            "model": model or self.default_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": options,
        }

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        body = self._body(messages, model, max_tokens, temperature, stream=False)
        if json_schema is not None:
            body["format"] = json_schema
        started = time.perf_counter()
        try:
            data = self._transport.post(self._chat_url, body)
        except ProviderError:
            raise
        except Exception as exc:  # network and HTTP errors; callers get one type
            raise ProviderError(self.name, str(exc)) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        if "error" in data:
            raise ProviderError(self.name, str(data["error"]))
        message = data.get("message") or {}
        return Completion(
            text=message.get("content") or "",
            model=data.get("model") or body["model"],
            provider=self.name,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason"),
        )

    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        usage: StreamUsage | None = None,
    ) -> Iterator[str]:
        """Yield content deltas; the final chunk's counts fill ``usage`` when given."""
        body = self._body(messages, model, max_tokens, temperature, stream=True)
        try:
            for line in self._transport.stream(self._chat_url, body):
                data = json.loads(line)
                if "error" in data:
                    raise ProviderError(self.name, str(data["error"]))
                delta = (data.get("message") or {}).get("content")
                if delta:
                    yield delta
                if data.get("done") and usage is not None:
                    usage.input_tokens = int(data.get("prompt_eval_count") or 0)
                    usage.output_tokens = int(data.get("eval_count") or 0)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc


# Inputs per embeddings request. Well under every hosted limit (OpenAI allows 2048) and
# under the point where a local Ollama runner was seen to fail.
EMBED_BATCH_SIZE = 64


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
        self._client = client or _build_client(name, api_key, base_url)
        # Sent only where the model honours it (OpenAI's text-embedding-3
        # family shortens its vectors on request); other servers may reject it.
        self.send_dimensions = False

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=[], model=self.model, provider=self.name, input_tokens=0, latency_ms=0
            )
        started = time.perf_counter()
        vectors: list[list[float]] = []
        model = None
        input_tokens = 0
        # A long list goes out in bounded requests: Ollama's embedding runner dies on a
        # request with a few hundred inputs ("tokenize: EOF"), which failed 9 of the 10
        # FinanceBench filings, each of which has over a thousand chunks.
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            try:
                extra = {"dimensions": self.dim} if self.send_dimensions else {}
                response = self._client.embeddings.create(model=self.model, input=batch, **extra)
            except Exception as exc:
                raise ProviderError(self.name, str(exc)) from exc
            ordered = sorted(response.data, key=lambda item: item.index)
            usage = getattr(response, "usage", None)
            vectors.extend(list(map(float, item.embedding)) for item in ordered)
            model = model or getattr(response, "model", None)
            input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        return EmbeddingResult(
            vectors=vectors,
            model=model or self.model,
            provider=self.name,
            input_tokens=input_tokens,
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
        # Lets a deployment keep the 768 dimension column migration 0004 pins.
        self.send_dimensions = model.startswith("text-embedding-3")


class OllamaEmbeddingProvider(OpenAICompatibleEmbeddings):
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "nomic-embed-text",
        dim: int = 768,
        client: Any | None = None,
    ) -> None:
        super().__init__("ollama", "ollama", model, dim, base_url=base_url, client=client)
