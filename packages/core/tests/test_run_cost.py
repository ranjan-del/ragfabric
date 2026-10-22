"""Cost and model recorded with a run (design decision D11)."""

from ragfabric_core.config_file import LLMConfig
from ragfabric_core.pricing import run_cost
from ragfabric_core.providers.offline import OfflineLLMProvider
from ragfabric_core.providers.registry import answering_model


def test_a_priced_model_gets_a_number_and_an_unpriced_one_none():
    # Local inference is in the table at zero: no token price, so nothing to estimate.
    assert run_cost("ollama", "llama3.1:8b", 1000, 200) is None
    assert run_cost("offline", "scripted", 1000, 200) is None
    assert run_cost("openai", "no-such-model-xyz", 1000, 200) is None
    assert run_cost("openai", None, 1000, 200) is None
    priced = run_cost("openai", "gpt-4.1-mini", 1_000_000, 0)
    assert priced is not None and priced > 0


def test_the_answering_model_is_the_configured_one_only_when_set():
    llm = OfflineLLMProvider(model="scripted")
    assert answering_model(LLMConfig(provider="offline"), llm) == "scripted"
    assert answering_model(LLMConfig(provider="offline", model="m1"), llm) == "m1"


def test_a_zero_price_is_still_known_to_the_budget_but_never_recorded_as_a_cost():
    from ragfabric_core.pricing import default_table, estimate_cost

    estimate = estimate_cost(default_table(), "ollama", "llama3.1:8b", 1000, 200)
    assert estimate.known is True and estimate.usd == 0.0
    assert run_cost("ollama", "llama3.1:8b", 1000, 200) is None
