"""Ollama chat goes through the native ``/api/chat`` endpoint so ``num_ctx`` is honoured.

Ollama 0.34 reached through its OpenAI compatible ``/v1`` endpoint ignores
``num_ctx`` in the request body and silently cuts every prompt to the model's
default window (about 2,048 tokens), keeping the end. The system prompt and the
first ranked passages are what is lost. The native endpoint takes
``options.num_ctx``, so the provider sends it there on every call.
"""

import json

import pytest

from ragfabric_core.config_file import LLMConfig
from ragfabric_core.providers.base import LLMProvider, Message, ProviderError
from ragfabric_core.providers.openai_compat import DEFAULT_OLLAMA_NUM_CTX, OllamaProvider
from ragfabric_core.providers.registry import build_llm_provider


class FakeTransport:
    """Records each request and replays canned native API responses."""

    def __init__(self, response=None, stream_lines=None, error=None):
        self.calls = []
        self._response = response or {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "12 days"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 6443,
            "eval_count": 4,
        }
        self._stream_lines = stream_lines or []
        self._error = error

    def post(self, url, body):
        self.calls.append((url, body))
        if self._error:
            raise self._error
        return self._response

    def stream(self, url, body):
        self.calls.append((url, body))
        if self._error:
            raise self._error
        yield from self._stream_lines


def _user(text="x"):
    return [Message(role="user", content=text)]


def test_complete_sends_num_ctx_to_the_native_chat_endpoint():
    transport = FakeTransport()
    llm = OllamaProvider(transport=transport)
    assert isinstance(llm, LLMProvider)
    llm.complete(_user())
    url, body = transport.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert body["options"]["num_ctx"] == DEFAULT_OLLAMA_NUM_CTX == 8192
    assert body["stream"] is False
    assert body["model"] == "llama3.2"
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["num_predict"] == 1024


def test_complete_maps_the_native_response_and_its_token_counts():
    llm = OllamaProvider(transport=FakeTransport())
    out = llm.complete(_user())
    assert out.text == "12 days"
    assert (out.input_tokens, out.output_tokens) == (6443, 4)
    assert out.provider == "ollama" and out.model == "llama3.2"
    assert out.finish_reason == "stop"


def test_num_ctx_is_configurable_and_none_leaves_the_model_default():
    transport = FakeTransport()
    OllamaProvider(transport=transport, num_ctx=16384).complete(_user())
    assert transport.calls[0][1]["options"]["num_ctx"] == 16384
    OllamaProvider(transport=transport, num_ctx=None).complete(_user())
    assert "num_ctx" not in transport.calls[1][1]["options"]


def test_a_v1_base_url_is_mapped_to_the_native_root():
    transport = FakeTransport()
    OllamaProvider(base_url="http://gpu-box:11434/v1/", transport=transport).complete(_user())
    assert transport.calls[0][0] == "http://gpu-box:11434/api/chat"
    OllamaProvider(base_url="http://gpu-box:11434", transport=transport).complete(_user())
    assert transport.calls[1][0] == "http://gpu-box:11434/api/chat"


def test_json_schema_goes_into_the_native_format_field():
    transport = FakeTransport()
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    OllamaProvider(transport=transport).complete(_user(), json_schema=schema)
    assert transport.calls[0][1]["format"] == schema


def test_stream_sends_num_ctx_and_yields_content_deltas():
    lines = [
        json.dumps({"message": {"content": "one "}, "done": False}),
        json.dumps({"message": {"content": "two"}, "done": False}),
        json.dumps(
            {"message": {"content": ""}, "done": True, "eval_count": 2, "prompt_eval_count": 9}
        ),
    ]
    transport = FakeTransport(stream_lines=lines)
    llm = OllamaProvider(transport=transport)
    assert "".join(llm.stream(_user())) == "one two"
    body = transport.calls[0][1]
    assert body["stream"] is True and body["options"]["num_ctx"] == 8192


def test_transport_errors_become_provider_errors():
    llm = OllamaProvider(transport=FakeTransport(error=OSError("connection refused")))
    with pytest.raises(ProviderError, match="connection refused"):
        llm.complete(_user())


def test_an_error_body_from_ollama_is_a_provider_error():
    llm = OllamaProvider(transport=FakeTransport(response={"error": "model not found"}))
    with pytest.raises(ProviderError, match="model not found"):
        llm.complete(_user())


def test_the_config_default_requests_an_8k_window_and_can_be_changed():
    assert LLMConfig().num_ctx == 8192
    llm = build_llm_provider(LLMConfig(provider="ollama", num_ctx=4096))
    assert llm.num_ctx == 4096
    with pytest.raises(ValueError):
        LLMConfig(num_ctx=10)


def test_ollama_chat_no_longer_needs_the_openai_extra(monkeypatch):
    from ragfabric_core.providers import openai_compat

    def boom():
        raise ImportError("no module named openai")

    monkeypatch.setattr(openai_compat, "_import_openai", boom)
    assert OllamaProvider().name == "ollama"
