"""Build providers from configuration. The only place vendor names are switched on."""

from __future__ import annotations

import os
from collections.abc import Mapping

from ragfabric_core.config_file import EmbeddingsConfig, LLMConfig
from ragfabric_core.providers.anthropic_provider import AnthropicProvider
from ragfabric_core.providers.base import EmbeddingProvider, LLMProvider, ProviderError
from ragfabric_core.providers.offline import HashingEmbeddingProvider, ScriptedLLMProvider
from ragfabric_core.providers.openai_compat import (
    OllamaEmbeddingProvider,
    OllamaProvider,
    OpenAIEmbeddingProvider,
    OpenAIProvider,
)

OPENAI_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


def _key(env: Mapping[str, str], name: str, provider: str) -> str:
    value = env.get(name, "")
    if not value:
        raise ProviderError(provider, f"{name} is not set")
    return value


def _set(cfg: LLMConfig | EmbeddingsConfig, field: str) -> bool:
    """True when the user wrote this key, rather than inheriting the default."""
    return field in cfg.model_fields_set


def build_llm_provider(cfg: LLMConfig, env: Mapping[str, str] | None = None) -> LLMProvider:
    env = os.environ if env is None else env
    if cfg.provider == "openai":
        model = cfg.model if _set(cfg, "model") else None
        return OpenAIProvider(
            api_key=_key(env, "OPENAI_API_KEY", "openai"), default_model=model or "gpt-5.4-mini"
        )
    if cfg.provider == "anthropic":
        model = cfg.model if _set(cfg, "model") else None
        return AnthropicProvider(
            api_key=_key(env, "ANTHROPIC_API_KEY", "anthropic"),
            default_model=model or "claude-sonnet-5",
        )
    if cfg.provider == "ollama":
        model = cfg.model if _set(cfg, "model") else None
        return OllamaProvider(
            base_url=cfg.base_url or DEFAULT_OLLAMA_URL, default_model=model or "llama3.2"
        )
    model = cfg.model if _set(cfg, "model") else None
    return ScriptedLLMProvider(responses=[], model=model or "scripted")


def build_embedding_provider(
    cfg: EmbeddingsConfig, env: Mapping[str, str] | None = None
) -> EmbeddingProvider:
    env = os.environ if env is None else env
    if cfg.provider == "openai":
        model = cfg.model if _set(cfg, "model") else None
        dim = cfg.dim if _set(cfg, "dim") else None
        model = model or "text-embedding-3-small"
        dim = dim or OPENAI_EMBEDDING_DIMS.get(model)
        if dim is None:
            raise ProviderError("openai", f"embeddings.dim must be set for unknown model {model!r}")
        return OpenAIEmbeddingProvider(
            api_key=_key(env, "OPENAI_API_KEY", "openai"), model=model, dim=dim
        )
    if cfg.provider == "ollama":
        model = cfg.model if _set(cfg, "model") else None
        dim = cfg.dim if _set(cfg, "dim") else None
        return OllamaEmbeddingProvider(
            base_url=cfg.base_url or DEFAULT_OLLAMA_URL,
            model=model or "nomic-embed-text",
            dim=dim or 768,
        )
    dim = cfg.dim if _set(cfg, "dim") else None
    return HashingEmbeddingProvider(dim=dim)
