"""Anthropic provider.

Differences from the OpenAI shape that this module absorbs so strategies do not
have to: the system prompt is a separate parameter, ``max_tokens`` is required,
the response is a list of content blocks, and usage is reported as
``input_tokens`` and ``output_tokens``. Structured output in Phase 1 is a
system instruction to answer with JSON matching the schema; Phase 5 upgrades
this to tool use when the agent needs guaranteed shapes.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ragfabric_core.providers.base import Completion, Message, ProviderError


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self, api_key: str, default_model: str = "claude-sonnet-5", client: Any | None = None
    ) -> None:
        if not api_key:
            raise ProviderError(self.name, "api_key is required (set ANTHROPIC_API_KEY)")
        self.default_model = default_model
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
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
            "temperature": temperature,
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
