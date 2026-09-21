"""Anthropic provider.

Differences from the OpenAI shape that this module absorbs so strategies do not
have to: the system prompt is a separate parameter, ``max_tokens`` is required,
the response is a list of content blocks, and usage is reported as
``input_tokens`` and ``output_tokens``. Structured output in Phase 1 is a
system instruction to answer with JSON matching the schema; Phase 5 upgrades
this to tool use when the agent needs guaranteed shapes. The temperature argument is
accepted for interface parity and not forwarded: current Claude models reject sampling
parameters while adaptive thinking is active, which is the default.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from ragfabric_core.providers.base import Completion, Message, ProviderError


def _import_anthropic():
    """Indirection so a test can simulate the ``anthropic`` extra being
    missing (monkeypatching this) without actually uninstalling the package."""
    from anthropic import Anthropic

    return Anthropic


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self, api_key: str, default_model: str = "claude-sonnet-5", client: Any | None = None
    ) -> None:
        if not api_key:
            raise ProviderError(self.name, "api_key is required (set ANTHROPIC_API_KEY)")
        self.default_model = default_model
        if client is None:
            try:
                anthropic_cls = _import_anthropic()
            except ImportError as exc:
                raise ProviderError(
                    self.name,
                    "anthropic is not installed. Install it with: "
                    "uv pip install 'ragfabric[anthropic]'",
                ) from exc
            client = anthropic_cls(api_key=api_key)
        self._client = client

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        system_parts = [m.content for m in messages if m.role == "system"]
        if json_schema is not None:
            system_parts.append(
                "Respond with JSON only, no prose, matching this JSON schema exactly:\n"
                + json.dumps(json_schema, indent=2)
            )
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages if m.role != "system"
            ],
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=getattr(response, "model", None) or kwargs["model"],
            provider=self.name,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=getattr(response, "stop_reason", None),
        )

    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yield text deltas using the SDK's streaming context manager.

        Usage is not surfaced by ``text_stream``, so no token count is
        returned here; zero would be a lie and an estimate would be a
        fabrication (ADR 0004), so the caller of a streamed answer accounts
        for tokens as unknown rather than guessing.
        """
        system_parts = [m.content for m in messages if m.role == "system"]
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages if m.role != "system"
            ],
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        try:
            with self._client.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc
