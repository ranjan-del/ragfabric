"""Several walks' sub-graphs as one, for an agent that walked more than once."""

from __future__ import annotations

from collections.abc import Sequence

from ragfabric_core.graph.contracts import Subgraph


def merge_subgraphs(parts: Sequence[Subgraph]) -> Subgraph | None:
    """The union by id of every walk that kept an edge, or ``None`` when none did.

    Walks with no edges are left out: they carry an ``empty_reason`` and nothing
    a citation could point at, and a merged graph with edges has no reason to
    give. ``truncated`` survives if any contributing walk was cut short.
    """
    with_edges = [part for part in parts if part.edges]
    if not with_edges:
        return None
    nodes = {node.id: node for part in with_edges for node in part.nodes}
    edges = {edge.id: edge for part in with_edges for edge in part.edges}
    return Subgraph(
        nodes=list(nodes.values()),
        edges=list(edges.values()),
        truncated=any(part.truncated for part in with_edges),
        empty_reason=None,
    )
