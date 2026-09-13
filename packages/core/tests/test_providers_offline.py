import pytest

from ragfabric_core.providers.base import (
    Completion,
    EmbeddingProvider,
    LLMProvider,
    Message,
    ProviderError,
)
from ragfabric_core.providers.offline import HashingEmbeddingProvider, ScriptedLLMProvider


def test_scripted_provider_returns_responses_in_order_and_counts_tokens():
    llm = ScriptedLLMProvider(["first answer", "second"])
    assert isinstance(llm, LLMProvider)
    one = llm.complete([Message(role="user", content="hello there world")])
    two = llm.complete([Message(role="user", content="again")])
    assert isinstance(one, Completion)
    assert (one.text, two.text) == ("first answer", "second")
    assert one.input_tokens == 3 and one.output_tokens == 2
    assert one.provider == "scripted" and one.model == "scripted"
    assert llm.calls == 2


def test_scripted_provider_raises_when_exhausted():
    llm = ScriptedLLMProvider(["only"])
    llm.complete([Message(role="user", content="x")])
    with pytest.raises(ProviderError) as info:
        llm.complete([Message(role="user", content="y")])
    assert info.value.provider == "scripted"


def test_hashing_embedding_provider_is_deterministic_and_normalised():
    emb = HashingEmbeddingProvider(dim=64)
    assert isinstance(emb, EmbeddingProvider)
    result = emb.embed(["leave policy", "leave policy", "kubernetes"])
    assert result.provider == "offline" and result.model == "hashing-64"
    assert len(result.vectors) == 3 and len(result.vectors[0]) == 64
    assert result.vectors[0] == result.vectors[1]
    norm = sum(v * v for v in result.vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-5
    assert result.input_tokens == 5


def test_hashing_embedding_provider_handles_empty_input():
    result = HashingEmbeddingProvider(dim=8).embed([])
    assert result.vectors == [] and result.input_tokens == 0
