from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.providers.offline import ScriptedLLMProvider


def test_the_scripted_provider_streams_its_text_in_pieces():
    provider = ScriptedLLMProvider(responses=["one two three"], model="scripted")
    pieces = list(provider.stream([Message(role="user", content="q")]))
    assert "".join(pieces) == "one two three"
    assert len(pieces) > 1, "a stream of one piece is not a stream"


def test_the_scripted_provider_still_satisfies_the_protocol_after_the_addition():
    assert isinstance(ScriptedLLMProvider(responses=[], model="scripted"), LLMProvider)


def test_streaming_an_exhausted_script_yields_nothing_rather_than_raising():
    provider = ScriptedLLMProvider(responses=[], model="scripted")
    assert list(provider.stream([Message(role="user", content="q")])) == []
