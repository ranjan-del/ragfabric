"""Which strategy serves a request that did not name one."""

from __future__ import annotations

from ragfabric_core.config_file import RouterConfig
from ragfabric_core.strategies.base import StrategyName


def resolve_requested(requested: str | None, router: RouterConfig) -> StrategyName:
    if requested:
        return StrategyName(requested)
    return StrategyName.AUTO if router.mode == "auto" else StrategyName.TRADITIONAL
