"""Question-guided traversal: a node budget and a relation-type filter (ruling R16).

The node budget is the safety property. Without it a dense graph makes a
two-hop walk return most of the corpus, so the reached nodes are ordered by
(depth, id), seeds first, and only the first ``node_budget`` are kept. A walk
the budget cut short says so in ``truncated``, because a silently truncated
traversal looks identical to a complete one.

The relation-type filter narrows the walk to the types the question implies.
It is applied inside the recursive query and the edge fetch, never to the
result afterwards, so a node reachable only through an unimplied type is never
reached and never spends the budget.

Every test runs on SQLite and on PostgreSQL through the ``graph`` fixture of
the traversal tests. The PostgreSQL parameter skips without
``RAGFABRIC_TEST_DATABASE_URL``, and a skip is not a pass.
"""

import pytest
import test_graph_traverse as traversal_tests

from ragfabric_core.graph.contracts import EmptyReason, EntityType, RelationType
from ragfabric_core.graph.traverse import DEFAULT_NODE_BUDGET, traverse

ALL = traversal_tests.ALL
# The seeded two-dialect fixture of the traversal tests, reused so both files
# test against the same graph and the same skip rule.
graph = traversal_tests.graph


def _star(graph, arms: int) -> None:
    """A team ``hub`` with ``arms`` members, each one hop from it as HAS_MEMBER."""
    graph.entity("hub", EntityType.TEAM)
    for index in range(arms):
        name = f"arm{index}"
        graph.entity(name)
        graph.edge(name, RelationType.MEMBER_OF, "hub")


# ---------------------------------------------------------------------------
# Node budget
# ---------------------------------------------------------------------------


def test_the_node_budget_caps_the_frontier(graph):
    # ``far`` has the lowest id but sits two hops out, and ``seed`` is not the
    # lowest id either: depth decides before id, and the seed comes first.
    for name in ("far", "seed", "near1", "near2", "near3"):
        graph.entity(name)
    graph.edge("seed", RelationType.REPORTS_TO, "near1")
    graph.edge("seed", RelationType.REPORTS_TO, "near2")
    graph.edge("seed", RelationType.REPORTS_TO, "near3")
    graph.edge("near1", RelationType.REPORTS_TO, "far")
    seed = [graph.ids["seed"]]

    whole = traverse(graph.db, seed, ALL, max_hops=2)
    assert graph.names(whole) == {"seed", "near1", "near2", "near3", "far"}

    four = traverse(graph.db, seed, ALL, max_hops=2, node_budget=4)
    assert graph.names(four) == {"seed", "near1", "near2", "near3"}

    three = traverse(graph.db, seed, ALL, max_hops=2, node_budget=3)
    assert graph.names(three) == {"seed", "near1", "near2"}
    assert [node.id for node in three.nodes] == sorted(node.id for node in three.nodes)

    # Same inputs, same answer.
    again = traverse(graph.db, seed, ALL, max_hops=2, node_budget=3)
    assert again == three


def test_truncation_is_reported(graph):
    _star(graph, arms=5)
    hub = [graph.ids["hub"]]

    assert traverse(graph.db, hub, ALL, max_hops=1).truncated is False
    exact = traverse(graph.db, hub, ALL, max_hops=1, node_budget=6)
    assert len(exact.nodes) == 6
    assert exact.truncated is False

    cut = traverse(graph.db, hub, ALL, max_hops=1, node_budget=5)
    assert len(cut.nodes) == 5
    assert cut.truncated is True


def test_every_edge_touching_a_dropped_node_is_dropped(graph):
    for name in ("ada", "grace", "linus", "alan"):
        graph.entity(name)
    kept = graph.edge("ada", RelationType.REPORTS_TO, "grace")
    graph.edge("ada", RelationType.REPORTS_TO, "linus")
    graph.edge("grace", RelationType.REPORTS_TO, "alan")

    cut = traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=2, node_budget=2)
    assert graph.names(cut) == {"ada", "grace"}
    assert [edge.id for edge in cut.edges] == [kept]
    assert cut.truncated is True
    assert cut.empty_reason is None


def test_a_truncated_walk_with_no_edges_is_still_a_subgraph(graph):
    _star(graph, arms=2)

    cut = traverse(graph.db, [graph.ids["hub"]], ALL, max_hops=1, node_budget=1)
    assert graph.names(cut) == {"hub"}
    assert cut.edges == []
    assert cut.truncated is True
    assert cut.empty_reason == EmptyReason.NO_WALKABLE_EDGES


def test_the_budget_is_a_hard_cap_even_on_seeds(graph):
    for name in ("ada", "grace", "linus"):
        graph.entity(name)
    graph.edge("ada", RelationType.REPORTS_TO, "linus")
    seeds = [graph.ids["linus"], graph.ids["grace"], graph.ids["ada"]]

    cut = traverse(graph.db, seeds, ALL, max_hops=1, node_budget=2)
    assert graph.names(cut) == {"ada", "grace"}
    assert cut.truncated is True


def test_a_denied_node_does_not_spend_the_budget(graph):
    graph.entity("hub", EntityType.TEAM)
    graph.entity("hidden", chunks=(graph.secret_chunk,))
    graph.entity("shown")
    graph.edge("hidden", RelationType.MEMBER_OF, "hub")
    graph.edge("shown", RelationType.MEMBER_OF, "hub")

    walk = traverse(graph.db, [graph.ids["hub"]], graph.restricted, max_hops=1, node_budget=2)
    assert graph.names(walk) == {"hub", "shown"}
    assert walk.truncated is False


@pytest.mark.parametrize("budget", [0, -1])
def test_a_node_budget_below_one_is_refused(graph, budget):
    graph.entity("ada")
    with pytest.raises(ValueError, match="node_budget"):
        traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=1, node_budget=budget)


def test_the_default_budget_is_modest():
    assert 1 <= DEFAULT_NODE_BUDGET <= 100


# ---------------------------------------------------------------------------
# Relation-type filter
# ---------------------------------------------------------------------------


def _mixed(graph) -> None:
    """ada reports to grace, who reports to linus; ada is also in a team that
    belongs to an org, and is related to grace. ``team`` and ``org`` are
    reachable only through MEMBER_OF and BELONGS_TO."""
    for name in ("ada", "grace", "linus"):
        graph.entity(name)
    graph.entity("team", EntityType.TEAM)
    graph.entity("org", EntityType.ORGANISATION)
    graph.edge("ada", RelationType.REPORTS_TO, "grace")
    graph.edge("grace", RelationType.REPORTS_TO, "linus")
    graph.edge("ada", RelationType.MEMBER_OF, "team")
    graph.edge("team", RelationType.BELONGS_TO, "org")
    graph.edge("ada", RelationType.RELATED_TO, "grace")


def test_only_the_implied_relation_types_are_walked(graph):
    _mixed(graph)
    ada = [graph.ids["ada"]]

    walk = traverse(graph.db, ada, ALL, max_hops=2, relation_types=[RelationType.REPORTS_TO])
    assert graph.names(walk) == {"ada", "grace", "linus"}
    assert {edge.relation_type for edge in walk.edges} == {RelationType.REPORTS_TO}
    assert len(walk.edges) == 2


def test_a_node_reachable_only_through_an_unimplied_type_does_not_spend_the_budget(graph):
    _mixed(graph)

    walk = traverse(
        graph.db,
        [graph.ids["ada"]],
        ALL,
        max_hops=1,
        node_budget=2,
        relation_types=[RelationType.REPORTS_TO],
    )
    assert graph.names(walk) == {"ada", "grace"}
    assert walk.truncated is False


def test_a_relation_walked_backwards_is_filtered_by_its_stored_type(graph):
    _mixed(graph)
    org = [graph.ids["org"]]

    implied = traverse(graph.db, org, ALL, max_hops=1, relation_types=[RelationType.BELONGS_TO])
    assert graph.names(implied) == {"org", "team"}
    (edge,) = implied.edges
    assert edge.reversed is True
    assert edge.walked_as == "CONTAINS"

    other = traverse(graph.db, org, ALL, max_hops=1, relation_types=[RelationType.MEMBER_OF])
    assert graph.names(other) == {"org"}
    assert other.edges == []


def test_a_directed_type_is_still_not_walked_backwards_when_implied(graph):
    _mixed(graph)

    walk = traverse(
        graph.db, [graph.ids["linus"]], ALL, max_hops=2, relation_types=[RelationType.REPORTS_TO]
    )
    assert graph.names(walk) == {"linus"}


def test_the_filter_does_not_bypass_access(graph):
    graph.entity("ada")
    graph.entity("grace")
    graph.edge("ada", RelationType.REPORTS_TO, "grace", chunks=(graph.secret_chunk,))

    walk = traverse(
        graph.db,
        [graph.ids["ada"]],
        graph.restricted,
        max_hops=1,
        relation_types=[RelationType.REPORTS_TO],
    )
    assert graph.names(walk) == {"ada"}


@pytest.mark.parametrize("implied", [None, [], ()])
def test_no_implied_types_falls_back_to_all_types(graph, implied):
    _mixed(graph)
    ada = [graph.ids["ada"]]

    walk = traverse(graph.db, ada, ALL, max_hops=2, relation_types=implied)
    assert graph.names(walk) == {"ada", "grace", "linus", "team", "org"}
    assert walk == traverse(graph.db, ada, ALL, max_hops=2)
    assert {edge.relation_type for edge in walk.edges} == {
        RelationType.REPORTS_TO,
        RelationType.MEMBER_OF,
        RelationType.BELONGS_TO,
        RelationType.RELATED_TO,
    }
