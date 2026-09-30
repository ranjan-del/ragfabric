"""Shared fixtures for the core tests."""

import pytest

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.strategies.base import Budget, RetrievalContext, StrategyParams


def _make_ctx(**overrides) -> RetrievalContext:
    values = dict(
        principal=Principal(
            user_id=1, email="u@example.com", role="user", group_ids=[], api_key_id=None
        ),
        access_filter=AccessFilter.unrestricted(),
        collection_ids=None,
        params=StrategyParams(),
        budget=Budget(),
    )
    values.update(overrides)
    return RetrievalContext(**values)


@pytest.fixture
def make_ctx():
    """The context builder as a fixture, so tests call ``make_ctx(**overrides)``."""
    return _make_ctx
