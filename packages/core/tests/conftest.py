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


@pytest.fixture
def small_subgraph():
    """Three nodes and one edge: Ravi Sharma MEMBER_OF Platform Team, sourced from chunk 1."""
    from ragfabric_core.graph.contracts import (
        EntityType,
        GraphEdge,
        GraphNode,
        RelationType,
        Subgraph,
    )

    return Subgraph(
        nodes=[
            GraphNode(id=1, name="Ravi Sharma", entity_type=EntityType.PERSON, depth=0),
            GraphNode(id=2, name="Platform Team", entity_type=EntityType.TEAM, depth=1),
            GraphNode(id=3, name="Billing", entity_type=EntityType.PRODUCT, depth=0),
        ],
        edges=[
            GraphEdge(
                id=1,
                source_id=1,
                target_id=2,
                relation_type=RelationType.MEMBER_OF,
                walked_as="MEMBER_OF",
                reversed=False,
                confidence=0.9,
                source_chunk_ids=[1],
            )
        ],
        truncated=False,
        empty_reason=None,
    )
