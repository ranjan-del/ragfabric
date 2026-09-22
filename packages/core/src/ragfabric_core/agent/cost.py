"""What a run has spent, and the honest answer when nobody knows (ADR 0004).

The agent is the one strategy whose spend is not knowable before it runs, so a
cost cap has to be checked against tokens that have actually been returned.
This ledger holds those tokens, per model, and prices them through the same
``pricing.yaml`` table the rest of the product reads.

**The unpriced model is the whole design.** ``estimate_cost`` reports
``usd=None, known=False`` for a model the table has no price for, and there are
only two ways to carry that into a cap, one of which is a lie:

- treat unknown as zero, and the cap silently never binds while the trace still
  implies it was watching;
- treat unknown as infinite, and every run on a self hosted model stops on its
  first call.

So unknown means the cap cannot bind, and the ledger says which models it could
not price. The caller records that alongside the run rather than reporting a
number nobody measured. A price of zero is a different thing entirely: local
inference is priced at zero in the table, with a source saying why, and that is
a known cost of nothing rather than an unknown cost.
"""

from __future__ import annotations

from functools import lru_cache

from ragfabric_core.pricing import PricingTable, estimate_cost


@lru_cache(maxsize=1)
def _shared_table() -> PricingTable:
    """The packaged pricing table, read once per process.

    Cached because the loop asks for a price several times per run and the file
    cannot change under a running process.
    """
    return PricingTable.load()


class CostLedger:
    """Real token counts per model, priced only where a price exists."""

    def __init__(self, table: PricingTable | None = None) -> None:
        self._table = table if table is not None else _shared_table()
        self._tokens: dict[tuple[str, str], list[int]] = {}

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
        """Add one completion's reported tokens.

        A node that made no model call passes an empty provider and is ignored,
        because an entry for a call that never happened would put a model in
        the unpriced list on the strength of nothing.
        """
        if not provider:
            return
        spent = self._tokens.setdefault((provider, model), [0, 0])
        spent[0] += input_tokens
        spent[1] += output_tokens

    def unpriced_models(self) -> list[str]:
        """The ``provider/model`` keys this run used that the table cannot price."""
        return sorted(
            f"{provider}/{model}"
            for (provider, model), (input_tokens, output_tokens) in self._tokens.items()
            if not estimate_cost(self._table, provider, model, input_tokens, output_tokens).known
        )

    @property
    def known(self) -> bool:
        """True when every model used so far has a price.

        A run that has made no call at all is known: nothing has been spent,
        and that is a measurement rather than a guess.
        """
        return not self.unpriced_models()

    def usd(self) -> float | None:
        """The estimated spend, or ``None`` when any model used has no price.

        Deliberately all or nothing. Summing the priced part of a run and
        presenting it as the total would understate the bill by exactly the
        amount nobody can compute.
        """
        if not self.known:
            return None
        total = 0.0
        for (provider, model), (input_tokens, output_tokens) in self._tokens.items():
            estimate = estimate_cost(self._table, provider, model, input_tokens, output_tokens)
            total += estimate.usd or 0.0
        return round(total, 8)
