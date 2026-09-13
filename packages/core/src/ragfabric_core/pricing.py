"""Cost estimation from configured pricing (ADR 0004).

Prices are data, not code. They live in pricing.yaml with a date and a source
so anyone can see how old a number is and where it came from. A model that is
not in the table yields ``known=False`` and ``usd=None``; the system never
guesses a price. Every estimate carries the same note so UIs cannot present
it as a measured bill.

Lookup order for ``provider/model``: exact key, then the longest matching
``provider/prefix*`` key, then ``provider/*``.
"""

from __future__ import annotations

from datetime import date
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ESTIMATE_NOTE = "estimate from configured pricing"
PER_TOKENS = 1_000_000


class ModelPrice(BaseModel):
    input: float | None = Field(default=None, ge=0)
    output: float | None = Field(default=None, ge=0)
    embedding: float | None = Field(default=None, ge=0)
    as_of: date
    source: str
    note: str | None = None


class PricingTable(BaseModel):
    currency: str = "USD"
    unit: str = "per_million_tokens"
    models: dict[str, ModelPrice]

    @classmethod
    def load(cls, path: Path | None = None) -> PricingTable:
        source = Path(path) if path else files("ragfabric_core") / "pricing.yaml"
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def lookup(self, provider: str, model: str) -> ModelPrice | None:
        exact = f"{provider}/{model}"
        if exact in self.models:
            return self.models[exact]
        best: tuple[int, ModelPrice] | None = None
        for key, price in self.models.items():
            if not key.startswith(f"{provider}/") or not key.endswith("*"):
                continue
            prefix = key[len(provider) + 1 : -1]
            if prefix and model.startswith(prefix) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), price)
        if best is not None:
            return best[1]
        return self.models.get(f"{provider}/*")


class CostEstimate(BaseModel):
    provider: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    embedding_tokens: int = Field(ge=0)
    usd: float | None
    known: bool
    note: str = ESTIMATE_NOTE
    as_of: date | None = None
    source: str | None = None


def estimate_cost(
    table: PricingTable,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
) -> CostEstimate:
    price = table.lookup(provider, model)
    base = dict(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_tokens=embedding_tokens,
    )
    if price is None:
        return CostEstimate(usd=None, known=False, **base)
    usd = 0.0
    usd += (price.input or 0.0) * input_tokens / PER_TOKENS
    usd += (price.output or 0.0) * output_tokens / PER_TOKENS
    usd += (price.embedding or 0.0) * embedding_tokens / PER_TOKENS
    return CostEstimate(
        usd=round(usd, 8), known=True, as_of=price.as_of, source=price.source, **base
    )
