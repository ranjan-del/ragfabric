import pytest

from ragfabric_core.config import settings
from ragfabric_core.config_file import EmbeddingsConfig, LLMConfig
from ragfabric_core.providers.base import ProviderError
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider


def test_offline_providers_need_no_keys():
    llm = build_llm_provider(LLMConfig(provider="offline"), env={})
    emb = build_embedding_provider(EmbeddingsConfig(provider="offline", dim=32), env={})
    assert llm.name == "scripted" and emb.name == "offline" and emb.dim == 32


def test_openai_requires_key_from_env():
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        build_llm_provider(LLMConfig(provider="openai"), env={})
    llm = build_llm_provider(
        LLMConfig(provider="openai", model="gpt-4.1-mini"), env={"OPENAI_API_KEY": "sk-x"}
    )
    assert llm.name == "openai" and llm.default_model == "gpt-4.1-mini"


def test_anthropic_requires_key_and_uses_default_model():
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        build_llm_provider(LLMConfig(provider="anthropic"), env={})
    llm = build_llm_provider(LLMConfig(provider="anthropic"), env={"ANTHROPIC_API_KEY": "k"})
    assert llm.default_model == "claude-sonnet-5"


def test_ollama_uses_base_url_and_no_key():
    llm = build_llm_provider(
        LLMConfig(provider="ollama", base_url="http://ollama:11434/v1"), env={}
    )
    assert llm.name == "ollama"
    emb = build_embedding_provider(EmbeddingsConfig(provider="ollama"), env={})
    assert (emb.model, emb.dim) == ("nomic-embed-text", 768)


def test_openai_embeddings_default_dim_matches_model():
    emb = build_embedding_provider(
        EmbeddingsConfig(provider="openai"), env={"OPENAI_API_KEY": "sk"}
    )
    assert (emb.model, emb.dim) == ("text-embedding-3-small", 1536)
    large = build_embedding_provider(
        EmbeddingsConfig(provider="openai", model="text-embedding-3-large"),
        env={"OPENAI_API_KEY": "sk"},
    )
    assert large.dim == 3072


def test_offline_llm_with_no_model_set_uses_scripted_default():
    llm = build_llm_provider(LLMConfig(provider="offline"), env={})
    assert llm.default_model == "scripted"


def test_offline_embeddings_with_no_dim_set_uses_settings_default():
    emb = build_embedding_provider(EmbeddingsConfig(provider="offline"), env={})
    assert emb.dim == settings.embedding_dim


def test_offline_llm_with_explicit_model_keeps_that_model():
    llm = build_llm_provider(LLMConfig(provider="offline", model="my-custom-model"), env={})
    assert llm.default_model == "my-custom-model"


def test_openai_embeddings_with_no_model_still_defaults_correctly():
    emb = build_embedding_provider(
        EmbeddingsConfig(provider="openai"), env={"OPENAI_API_KEY": "sk"}
    )
    assert (emb.model, emb.dim) == ("text-embedding-3-small", 1536)


def test_ollama_embeddings_with_explicit_dim_keeps_that_dim():
    emb = build_embedding_provider(EmbeddingsConfig(provider="ollama", dim=512), env={})
    assert emb.dim == 512
