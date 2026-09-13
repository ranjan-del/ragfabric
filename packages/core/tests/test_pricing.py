from datetime import date
from pathlib import Path

from ragfabric_core.pricing import CostEstimate, PricingTable, estimate_cost

SAMPLE = """
currency: USD
unit: per_million_tokens
models:
  openai/test-chat:
    input: 1.00
    output: 4.00
    as_of: 2026-09-13
    source: https://example.test/pricing
  openai/test-embed:
    embedding: 0.02
    as_of: 2026-09-13
    source: https://example.test/pricing
  openai/test-chat-2026*:
    input: 2.00
    output: 8.00
    as_of: 2026-09-13
    source: https://example.test/pricing
  ollama/*:
    input: 0
    output: 0
    embedding: 0
    as_of: 2026-09-13
    source: local inference has no per token price
    note: local
"""


def table(tmp_path: Path) -> PricingTable:
    p = tmp_path / "pricing.yaml"
    p.write_text(SAMPLE)
    return PricingTable.load(p)


def test_exact_lookup_and_cost_arithmetic(tmp_path):
    t = table(tmp_path)
    est = estimate_cost(t, "openai", "test-chat", input_tokens=1_000_000, output_tokens=500_000)
    assert isinstance(est, CostEstimate)
    assert est.known is True
    assert est.usd == 1.00 + 2.00
    assert est.note == "estimate from configured pricing"
    assert est.as_of == date(2026, 9, 13) and est.source == "https://example.test/pricing"


def test_embedding_cost(tmp_path):
    est = estimate_cost(
        table(tmp_path), "openai", "test-embed", input_tokens=0, embedding_tokens=2_000_000
    )
    assert est.usd == 0.04


def test_dated_snapshot_ids_match_by_prefix_wildcard(tmp_path):
    est = estimate_cost(table(tmp_path), "openai", "test-chat-2026-03-17", input_tokens=1_000_000)
    assert est.usd == 2.00


def test_provider_wildcard_matches_any_local_model(tmp_path):
    est = estimate_cost(
        table(tmp_path), "ollama", "llama3.2", input_tokens=123456, output_tokens=999
    )
    assert est.usd == 0.0 and est.known is True


def test_unknown_model_is_reported_not_guessed(tmp_path):
    est = estimate_cost(table(tmp_path), "anthropic", "mystery", input_tokens=10)
    assert est.usd is None and est.known is False


def test_packaged_pricing_file_loads_and_every_entry_is_dated():
    t = PricingTable.load()
    assert t.currency == "USD"
    assert "openai/text-embedding-3-small" in t.models
    for key, price in t.models.items():
        assert price.as_of is not None and price.source, f"{key} lacks as_of or source"
