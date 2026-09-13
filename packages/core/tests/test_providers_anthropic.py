from types import SimpleNamespace

import pytest

from ragfabric_core.providers.anthropic_provider import AnthropicProvider
from ragfabric_core.providers.base import LLMProvider, Message, ProviderError


class FakeMessages:
    def __init__(self, text="ok", in_tok=9, out_tok=1, stop="end_turn"):
        self.calls = []
        self._resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
            model="claude-sonnet-5",
            stop_reason=stop,
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


def client_with(messages):
    return SimpleNamespace(messages=messages)


def test_system_messages_become_the_system_parameter():
    fm = FakeMessages(text="12 days", in_tok=30, out_tok=3)
    llm = AnthropicProvider(api_key="k", client=client_with(fm))
    assert isinstance(llm, LLMProvider) and llm.name == "anthropic"
    out = llm.complete(
        [
            Message(role="system", content="be brief"),
            Message(role="user", content="leave?"),
            Message(role="assistant", content="asking"),
        ],
        max_tokens=64,
        temperature=0.3,
    )
    sent = fm.calls[0]
    assert sent["system"] == "be brief"
    assert sent["messages"] == [
        {"role": "user", "content": "leave?"},
        {"role": "assistant", "content": "asking"},
    ]
    assert (
        sent["model"] == "claude-sonnet-5"
        and sent["max_tokens"] == 64
        and sent["temperature"] == 0.3
    )
    assert out.text == "12 days" and (out.input_tokens, out.output_tokens) == (30, 3)
    assert out.finish_reason == "end_turn" and out.provider == "anthropic"


def test_no_system_message_means_no_system_parameter():
    fm = FakeMessages()
    AnthropicProvider(api_key="k", client=client_with(fm)).complete(
        [Message(role="user", content="x")]
    )
    assert "system" not in fm.calls[0]


def test_json_schema_adds_a_json_only_instruction_to_system():
    fm = FakeMessages(text='{"a": 1}')
    AnthropicProvider(api_key="k", client=client_with(fm)).complete(
        [Message(role="user", content="x")], json_schema={"type": "object"}
    )
    assert "JSON" in fm.calls[0]["system"] and '"type": "object"' in fm.calls[0]["system"]


def test_text_blocks_are_joined_and_errors_wrapped():
    fm = FakeMessages()
    fm._resp.content = [
        SimpleNamespace(type="text", text="a"),
        SimpleNamespace(type="tool_use"),
        SimpleNamespace(type="text", text="b"),
    ]
    assert (
        AnthropicProvider(api_key="k", client=client_with(fm))
        .complete([Message(role="user", content="x")])
        .text
        == "ab"
    )

    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("overloaded")

    with pytest.raises(ProviderError, match="overloaded"):
        AnthropicProvider(api_key="k", client=client_with(Boom())).complete(
            [Message(role="user", content="x")]
        )


def test_missing_key_rejected():
    with pytest.raises(ProviderError, match="api_key"):
        AnthropicProvider(api_key="")
