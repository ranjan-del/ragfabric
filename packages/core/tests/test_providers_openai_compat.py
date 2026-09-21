import os
from types import SimpleNamespace

import pytest

from ragfabric_core.providers.base import LLMProvider, Message, ProviderError
from ragfabric_core.providers.openai_compat import (
    OllamaEmbeddingProvider,
    OllamaProvider,
    OpenAIEmbeddingProvider,
    OpenAIProvider,
)


class FakeChat:
    def __init__(self, text="hi", prompt_tokens=11, completion_tokens=2, finish="stop"):
        self.calls = []
        self._resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            model="gpt-5.4-mini-2026-03-17",
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


class FakeEmbeddings:
    def __init__(self, dim=4):
        self.calls = []
        self._dim = dim

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [
            SimpleNamespace(embedding=[float(i)] * self._dim, index=i)
            for i, _ in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(
            data=data, usage=SimpleNamespace(prompt_tokens=7), model=kwargs["model"]
        )


def fake_client(chat=None, embeddings=None):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=chat or FakeChat()),
        embeddings=embeddings or FakeEmbeddings(),
    )


def test_openai_provider_maps_response_and_usage():
    chat = FakeChat(text="12 days", prompt_tokens=20, completion_tokens=3)
    llm = OpenAIProvider(api_key="sk-test", client=fake_client(chat=chat))
    assert isinstance(llm, LLMProvider)
    out = llm.complete(
        [Message(role="system", content="answer briefly"), Message(role="user", content="leave?")]
    )
    assert out.text == "12 days"
    assert (out.input_tokens, out.output_tokens) == (20, 3)
    assert out.provider == "openai" and out.model == "gpt-5.4-mini-2026-03-17"
    assert out.finish_reason == "stop"
    sent = chat.calls[0]
    assert sent["model"] == "gpt-5.4-mini"
    assert sent["messages"][0] == {"role": "system", "content": "answer briefly"}
    assert sent["max_completion_tokens"] == 1024


def test_reasoning_models_do_not_receive_temperature_but_others_do():
    chat = FakeChat()
    OpenAIProvider(api_key="k", client=fake_client(chat=chat)).complete(
        [Message(role="user", content="x")], model="gpt-5.4-mini"
    )
    assert "temperature" not in chat.calls[0]
    OpenAIProvider(api_key="k", client=fake_client(chat=chat)).complete(
        [Message(role="user", content="x")], model="gpt-4.1-mini", temperature=0.2
    )
    assert chat.calls[1]["temperature"] == 0.2


def test_json_schema_is_passed_as_response_format():
    chat = FakeChat(text='{"a": 1}')
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    OpenAIProvider(api_key="k", client=fake_client(chat=chat)).complete(
        [Message(role="user", content="x")], json_schema=schema
    )
    rf = chat.calls[0]["response_format"]
    assert (
        rf["type"] == "json_schema"
        and rf["json_schema"]["schema"] == schema
        and rf["json_schema"]["strict"] is True
    )


def test_ollama_provider_uses_local_defaults_and_sends_temperature():
    chat = FakeChat()
    llm = OllamaProvider(client=fake_client(chat=chat))
    assert llm.name == "ollama" and llm.default_model == "llama3.2"
    llm.complete([Message(role="user", content="x")])
    assert chat.calls[0]["model"] == "llama3.2" and chat.calls[0]["temperature"] == 0.0


def test_provider_errors_are_wrapped():
    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("connection refused")

    with pytest.raises(ProviderError, match="connection refused"):
        OpenAIProvider(api_key="k", client=fake_client(chat=Boom())).complete(
            [Message(role="user", content="x")]
        )


def test_empty_choices_is_a_provider_error():
    chat = FakeChat()
    chat._resp.choices = []
    with pytest.raises(ProviderError, match="no choices"):
        OpenAIProvider(api_key="k", client=fake_client(chat=chat)).complete(
            [Message(role="user", content="x")]
        )


def test_openai_embeddings_map_vectors_in_input_order():
    emb = OpenAIEmbeddingProvider(api_key="k", client=fake_client(embeddings=FakeEmbeddings(dim=4)))
    result = emb.embed(["a", "b"])
    assert result.provider == "openai" and result.model == "text-embedding-3-small"
    assert result.vectors == [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    assert result.input_tokens == 7 and emb.dim == 1536


def test_ollama_embeddings_defaults():
    emb = OllamaEmbeddingProvider(client=fake_client(embeddings=FakeEmbeddings(dim=3)))
    assert (emb.name, emb.model, emb.dim) == ("ollama", "nomic-embed-text", 768)
    assert emb.embed([]).vectors == []


def test_missing_api_key_is_rejected_at_construction():
    with pytest.raises(ProviderError, match="api_key"):
        OpenAIProvider(api_key="")


def test_openai_provider_without_the_extra_names_the_install_command(monkeypatch):
    """Task 16 C7: openai is now an optional extra, imported lazily and only
    when no fake client is supplied, so a deployment missing it gets a
    ProviderError naming the extra instead of a raw ImportError.
    """
    from ragfabric_core.providers import openai_compat

    def boom():
        raise ImportError("no module named openai")

    monkeypatch.setattr(openai_compat, "_import_openai", boom)
    with pytest.raises(ProviderError) as exc:
        OpenAIProvider(api_key="k")
    assert "ragfabric[openai]" in str(exc.value)


def test_ollama_provider_without_the_extra_also_names_the_openai_install_command(monkeypatch):
    """Ollama is served through the OpenAI-compatible client too, so it hits
    the same extra, not a made-up "ollama" one."""
    from ragfabric_core.providers import openai_compat

    def boom():
        raise ImportError("no module named openai")

    monkeypatch.setattr(openai_compat, "_import_openai", boom)
    with pytest.raises(ProviderError) as exc:
        OllamaProvider()
    assert "ragfabric[openai]" in str(exc.value)


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
def test_openai_live_roundtrip():
    llm = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
    out = llm.complete(
        [Message(role="user", content="Reply with the single word: pong")], max_tokens=16
    )
    assert "pong" in out.text.lower()
    assert out.input_tokens > 0 and out.output_tokens > 0
