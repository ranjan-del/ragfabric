"""Directed, access-aware traversal of the knowledge graph (ADR 0003, rulings R2 and R10).

Two properties matter and both were wrong in the spike.

Direction: ``REPORTS_TO`` read backwards asserts the opposite fact, so a
relation walks backwards only when ``INVERSES`` names it, and then under the
inverse name. Access: the predicate is part of the recursive query, on the
seed match and on every edge walked, so a denied chunk can neither reveal an
entity nor carry the walk across an edge.

Every test runs on SQLite and on PostgreSQL. The PostgreSQL parameter skips
without ``RAGFABRIC_TEST_DATABASE_URL``, and a skip is not a pass.
"""

import os
from dataclasses import dataclass, field

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.graph.contracts import (
    EmptyReason,
    EntityMention,
    EntityType,
    RelationType,
)
from ragfabric_core.graph.traverse import (
    MAX_HOPS_CEILING,
    match_entities,
    traverse,
    visible_entity_chunks,
)
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import Entity, EntitySource, Relationship, RelationshipSource

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")

ALL = AccessFilter.unrestricted()


@dataclass
class Graph:
    """A seeded database with one open and one secret document.

    ``restricted`` denies the secret document, so anything whose only source
    is the secret chunk must be invisible under it.
    """

    db: Session
    open_chunk: int
    open_chunk_2: int
    secret_chunk: int
    secret_document: int
    secret_collection: int
    open_collection: int
    other_open_chunk: int
    other_open_document: int
    ids: dict[str, int] = field(default_factory=dict)

    @property
    def restricted(self) -> AccessFilter:
        return AccessFilter(denied_document_ids=frozenset({self.secret_document}))

    def entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.PERSON,
        *,
        chunks: tuple[int, ...] | None = None,
        aliases: object = None,
    ) -> int:
        row = Entity(
            name=name,
            normalized_name=name.casefold(),
            entity_type=entity_type.value,
            aliases=[] if aliases is None else aliases,
        )
        self.db.add(row)
        self.db.flush()
        for chunk_id in chunks if chunks is not None else (self.open_chunk,):
            self.db.add(EntitySource(entity_id=row.id, chunk_id=chunk_id))
        self.db.flush()
        self.ids[name] = row.id
        return row.id

    def edge(
        self,
        source: str,
        relation: RelationType | str,
        target: str,
        *,
        chunks: tuple[int, ...] | None = None,
        confidence: float | None = 0.9,
    ) -> int:
        value = relation.value if isinstance(relation, RelationType) else relation
        row = Relationship(
            source_entity_id=self.ids[source],
            target_entity_id=self.ids[target],
            relation_type=value,
            confidence=confidence,
        )
        self.db.add(row)
        self.db.flush()
        for chunk_id in chunks if chunks is not None else (self.open_chunk,):
            self.db.add(RelationshipSource(relationship_id=row.id, chunk_id=chunk_id))
        self.db.flush()
        return row.id

    def names(self, subgraph) -> set[str]:
        by_id = {v: k for k, v in self.ids.items()}
        return {by_id[node.id] for node in subgraph.nodes}


@pytest.fixture(params=["sqlite", "postgresql"])
def graph(request, tmp_path):
    if request.param == "postgresql":
        if not URL:
            pytest.skip("needs RAGFABRIC_TEST_DATABASE_URL")
        downgrade(URL)
        upgrade(URL)
        engine = create_engine(URL)
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'graph.db'}")
        Base.metadata.create_all(engine)

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        open_collection, secret_collection = Collection(name="open"), Collection(name="secret")
        db.add_all([open_collection, secret_collection])
        db.flush()
        open_document = Document(
            filename="open.txt", format="txt", collection_id=open_collection.id, status="ready"
        )
        secret_document = Document(
            filename="secret.txt",
            format="txt",
            collection_id=secret_collection.id,
            status="ready",
        )
        other_open_document = Document(
            filename="other.txt", format="txt", collection_id=open_collection.id, status="ready"
        )
        db.add_all([open_document, secret_document, other_open_document])
        db.flush()

        def chunk(document, index):
            row = Chunk(
                document_id=document.id,
                collection_id=document.collection_id,
                chunk_index=index,
                text=f"chunk {index}",
                embedding=[],
            )
            db.add(row)
            db.flush()
            return row.id

        seeded = Graph(
            db=db,
            open_chunk=chunk(open_document, 0),
            open_chunk_2=chunk(open_document, 1),
            secret_chunk=chunk(secret_document, 0),
            secret_document=secret_document.id,
            secret_collection=secret_collection.id,
            open_collection=open_collection.id,
            other_open_chunk=chunk(other_open_document, 0),
            other_open_document=other_open_document.id,
        )
        yield seeded
    engine.dispose()


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------


def test_a_denied_edge_is_never_walked(graph):
    graph.entity("ada")
    graph.entity("grace")
    graph.edge("ada", RelationType.REPORTS_TO, "grace", chunks=(graph.secret_chunk,))

    # Control: with no restriction the edge is there to be walked, so the
    # restricted assertion below is about access and not about seeding.
    seen = traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=2)
    assert len(seen.edges) == 1

    hidden = traverse(graph.db, [graph.ids["ada"]], graph.restricted, max_hops=2)
    assert hidden.edges == []
    assert graph.names(hidden) == {"ada"}
    assert hidden.empty_reason == EmptyReason.NO_WALKABLE_EDGES


def test_a_denied_edge_does_not_carry_the_walk_further(graph):
    for name in ("ada", "grace", "linus"):
        graph.entity(name)
    graph.edge("ada", RelationType.REPORTS_TO, "grace", chunks=(graph.secret_chunk,))
    graph.edge("grace", RelationType.REPORTS_TO, "linus")

    hidden = traverse(graph.db, [graph.ids["ada"]], graph.restricted, max_hops=3)
    assert graph.names(hidden) == {"ada"}


def test_a_denied_edge_between_reached_nodes_is_not_reported(graph):
    for name in ("ada", "grace", "linus"):
        graph.entity(name)
    graph.edge("ada", RelationType.REPORTS_TO, "grace")
    graph.edge("ada", RelationType.REPORTS_TO, "linus")
    denied = graph.edge("grace", RelationType.REPORTS_TO, "linus", chunks=(graph.secret_chunk,))

    seen = traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=2)
    assert denied in {edge.id for edge in seen.edges}
    hidden = traverse(graph.db, [graph.ids["ada"]], graph.restricted, max_hops=2)
    assert graph.names(hidden) == {"ada", "grace", "linus"}
    assert denied not in {edge.id for edge in hidden.edges}


def test_a_denied_entity_does_not_match(graph):
    graph.entity("ada", chunks=(graph.secret_chunk,))
    mention = [EntityMention(name="Ada")]

    assert match_entities(graph.db, mention, ALL) == [graph.ids["ada"]]
    assert match_entities(graph.db, mention, graph.restricted) == []


def test_an_entity_with_one_admitted_source_matches(graph):
    graph.entity("ada", chunks=(graph.secret_chunk, graph.open_chunk))
    assert match_entities(graph.db, [EntityMention(name="ada")], graph.restricted) == [
        graph.ids["ada"]
    ]


def test_a_denied_seed_is_not_a_node(graph):
    graph.entity("ada", chunks=(graph.secret_chunk,))
    graph.entity("grace")
    graph.edge("ada", RelationType.REPORTS_TO, "grace")

    hidden = traverse(graph.db, [graph.ids["ada"]], graph.restricted, max_hops=2)
    assert hidden.nodes == []
    assert hidden.edges == []
    assert hidden.empty_reason == EmptyReason.NO_ENTITY_MATCHED


def test_an_edge_into_an_invisible_node_is_not_walked(graph):
    graph.entity("ada")
    graph.entity("grace", chunks=(graph.secret_chunk,))
    graph.edge("ada", RelationType.REPORTS_TO, "grace")

    assert len(traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=1).edges) == 1
    hidden = traverse(graph.db, [graph.ids["ada"]], graph.restricted, max_hops=1)
    assert hidden.edges == []
    assert graph.names(hidden) == {"ada"}


def test_an_edge_carries_only_admitted_source_chunks(graph):
    graph.entity("ada")
    graph.entity("grace")
    graph.edge(
        "ada",
        RelationType.REPORTS_TO,
        "grace",
        chunks=(graph.open_chunk, graph.secret_chunk, graph.open_chunk_2),
    )

    everything = traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=1)
    assert everything.edges[0].source_chunk_ids == sorted(
        [graph.open_chunk, graph.secret_chunk, graph.open_chunk_2]
    )
    restricted = traverse(graph.db, [graph.ids["ada"]], graph.restricted, max_hops=1)
    assert restricted.edges[0].source_chunk_ids == sorted([graph.open_chunk, graph.open_chunk_2])


def test_an_allow_list_filter_hides_a_seed_outside_it(graph):
    graph.entity("ada")
    graph.entity("grace")
    graph.entity("linus", chunks=(graph.secret_chunk,))
    graph.edge("ada", RelationType.REPORTS_TO, "grace")
    graph.edge("ada", RelationType.OWNS, "linus", chunks=(graph.secret_chunk,))

    only_secret = AccessFilter(collection_ids=frozenset({graph.secret_collection}))
    assert traverse(graph.db, [graph.ids["ada"]], only_secret, max_hops=2).nodes == []
    assert match_entities(graph.db, [EntityMention(name="linus")], only_secret) == [
        graph.ids["linus"]
    ]


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------


def test_a_directed_relation_is_not_walked_backwards(graph):
    graph.entity("ada")
    graph.entity("grace")
    graph.edge("ada", RelationType.REPORTS_TO, "grace")

    from_target = traverse(graph.db, [graph.ids["grace"]], ALL, max_hops=3)
    assert from_target.edges == []
    assert graph.names(from_target) == {"grace"}

    from_source = traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=3)
    [edge] = from_source.edges
    assert edge.walked_as == "REPORTS_TO"
    assert edge.reversed is False
    assert (edge.source_id, edge.target_id) == (graph.ids["ada"], graph.ids["grace"])


def test_an_invertible_relation_is_walked_both_ways_under_its_inverse_name(graph):
    graph.entity("platform", EntityType.TEAM)
    graph.entity("acme", EntityType.ORGANISATION)
    graph.edge("platform", RelationType.BELONGS_TO, "acme")

    [forwards] = traverse(graph.db, [graph.ids["platform"]], ALL, max_hops=1).edges
    assert forwards.walked_as == "BELONGS_TO"
    assert forwards.reversed is False

    backwards_graph = traverse(graph.db, [graph.ids["acme"]], ALL, max_hops=1)
    [backwards] = backwards_graph.edges
    assert backwards.walked_as == "CONTAINS"
    assert backwards.reversed is True
    assert backwards.relation_type == RelationType.BELONGS_TO
    # The stored direction is reported as stored; only walked_as reads it backwards.
    assert (backwards.source_id, backwards.target_id) == (
        graph.ids["platform"],
        graph.ids["acme"],
    )
    assert graph.names(backwards_graph) == {"platform", "acme"}


def test_a_relation_type_with_no_direction_rule_is_not_walked(graph):
    graph.entity("ada")
    graph.entity("grace")
    graph.edge("ada", "INVENTED_TYPE", "grace")

    assert traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=2).edges == []
    assert traverse(graph.db, [graph.ids["grace"]], ALL, max_hops=2).edges == []


def test_an_edge_reached_both_ways_is_reported_once(graph):
    graph.entity("platform", EntityType.TEAM)
    graph.entity("acme", EntityType.ORGANISATION)
    graph.edge("platform", RelationType.BELONGS_TO, "acme")

    both = traverse(graph.db, [graph.ids["platform"], graph.ids["acme"]], ALL, max_hops=2)
    [edge] = both.edges
    assert edge.reversed is False


# ---------------------------------------------------------------------------
# Termination and bounds
# ---------------------------------------------------------------------------


def test_a_cycle_terminates(graph):
    for name in ("ada", "grace", "linus"):
        graph.entity(name)
    graph.edge("ada", RelationType.REPORTS_TO, "grace")
    graph.edge("grace", RelationType.REPORTS_TO, "linus")
    graph.edge("linus", RelationType.REPORTS_TO, "ada")
    graph.edge("ada", RelationType.RELATED_TO, "linus")

    looped = traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=MAX_HOPS_CEILING)
    assert graph.names(looped) == {"ada", "grace", "linus"}
    assert len(looped.edges) == 4


def test_the_hop_limit_is_respected(graph):
    for name in ("a", "b", "c", "d"):
        graph.entity(name)
    graph.edge("a", RelationType.REPORTS_TO, "b")
    graph.edge("b", RelationType.REPORTS_TO, "c")
    graph.edge("c", RelationType.REPORTS_TO, "d")

    two = traverse(graph.db, [graph.ids["a"]], ALL, max_hops=2)
    assert graph.names(two) == {"a", "b", "c"}
    assert len(two.edges) == 2

    one = traverse(graph.db, [graph.ids["a"]], ALL, max_hops=1)
    assert graph.names(one) == {"a", "b"}
    assert len(one.edges) == 1


@pytest.mark.parametrize("hops", [0, -1, MAX_HOPS_CEILING + 1, 50])
def test_max_hops_outside_one_to_the_ceiling_is_refused(graph, hops):
    graph.entity("a")
    with pytest.raises(ValueError):
        traverse(graph.db, [graph.ids["a"]], ALL, max_hops=hops)


def test_the_ceiling_is_four():
    assert MAX_HOPS_CEILING == 4


def test_an_edge_between_two_nodes_at_the_hop_limit_is_not_reported(graph):
    for name in ("s", "x", "y"):
        graph.entity(name)
    graph.edge("s", RelationType.REPORTS_TO, "x")
    graph.edge("s", RelationType.REPORTS_TO, "y")
    graph.edge("x", RelationType.REPORTS_TO, "y")

    # Both x and y are reached at depth 1; walking x -> y would be a second hop.
    one = traverse(graph.db, [graph.ids["s"]], ALL, max_hops=1)
    assert graph.names(one) == {"s", "x", "y"}
    assert len(one.edges) == 2
    assert len(traverse(graph.db, [graph.ids["s"]], ALL, max_hops=2).edges) == 3


def _reach_both_ends_of(graph, relation):
    """b is the seed, a is reached at depth 2 via a shared team, then a --relation--> b."""
    graph.entity("a")
    graph.entity("b")
    graph.entity("t", EntityType.TEAM)
    graph.edge("b", RelationType.MEMBER_OF, "t")
    graph.edge("a", RelationType.MEMBER_OF, "t")
    return graph.edge("a", relation, "b")


def test_a_directed_edge_between_reached_nodes_is_never_reported_reversed(graph):
    edge_id = _reach_both_ends_of(graph, RelationType.REPORTS_TO)

    # a sits at the hop limit, so a -> b cannot be walked forwards, and
    # REPORTS_TO cannot be walked backwards from b.
    result = traverse(graph.db, [graph.ids["b"]], ALL, max_hops=2)
    assert graph.names(result) == {"a", "b", "t"}
    assert edge_id not in {edge.id for edge in result.edges}
    assert not any(edge.reversed and edge.relation_type == "REPORTS_TO" for edge in result.edges)

    # With a hop to spare the walk can use it forwards from a.
    wider = traverse(graph.db, [graph.ids["b"]], ALL, max_hops=3)
    [edge] = [edge for edge in wider.edges if edge.id == edge_id]
    assert edge.reversed is False and edge.walked_as == "REPORTS_TO"


def test_an_invertible_edge_is_reported_in_the_direction_the_walk_could_use(graph):
    for name in ("s", "b", "a"):
        graph.entity(name)
    graph.edge("s", RelationType.REPORTS_TO, "b")
    edge_id = graph.edge("a", RelationType.BELONGS_TO, "b")

    # s (0) -> b (1) -> a (2) backwards: a is at the limit, so only the
    # backwards reading from b was walkable, and it carries the inverse name.
    at_limit = traverse(graph.db, [graph.ids["s"]], ALL, max_hops=2)
    [edge] = [edge for edge in at_limit.edges if edge.id == edge_id]
    assert edge.reversed is True and edge.walked_as == "CONTAINS"

    # With a hop to spare it is walkable forwards from a too, and forwards wins.
    wider = traverse(graph.db, [graph.ids["s"]], ALL, max_hops=3)
    [edge] = [edge for edge in wider.edges if edge.id == edge_id]
    assert edge.reversed is False and edge.walked_as == "BELONGS_TO"


def test_a_hub_walk_stays_bounded(graph):
    # 30 members of one hub, chained by a symmetric relation: many paths reach
    # each node within four hops, but each node and each edge is reported once.
    graph.entity("hub")
    people = [f"p{i}" for i in range(30)]
    for name in people:
        graph.entity(name)
        graph.edge(name, RelationType.MEMBER_OF, "hub")
    for left, right in zip(people, people[1:], strict=False):
        graph.edge(left, RelationType.RELATED_TO, right)

    result = traverse(graph.db, [graph.ids["hub"]], ALL, max_hops=MAX_HOPS_CEILING)
    assert len(result.nodes) == 31
    assert len(result.edges) == 30 + 29


def test_node_depth_is_the_walks_minimum_hop_count(graph):
    for name in ("a", "b", "c"):
        graph.entity(name)
    graph.edge("a", RelationType.REPORTS_TO, "b")
    graph.edge("b", RelationType.REPORTS_TO, "c")

    result = traverse(graph.db, [graph.ids["a"]], ALL, max_hops=2)
    depths = {node.id: node.depth for node in result.nodes}
    assert depths == {graph.ids["a"]: 0, graph.ids["b"]: 1, graph.ids["c"]: 2}


def test_a_node_reached_only_backwards_still_gets_its_true_depth(graph):
    """A node's ``depth`` must come from the walk's own minimum, not from the
    direction its reported edge happens to read. ``team`` is reached only by
    walking BELONGS_TO backwards from ``dept`` (there is no other path to
    it), at depth 1 with a hop still to spare, so ``_edge_statement``'s
    ``goes_forwards`` preference reports the edge in the forwards reading
    even though the walk reached ``team`` backwards. A depth reconstruction
    keyed off ``edge.reversed`` would never visit ``team`` at all.
    """
    graph.entity("dept", EntityType.ORGANISATION)
    graph.entity("team", EntityType.TEAM)
    graph.edge("team", RelationType.BELONGS_TO, "dept")

    result = traverse(graph.db, [graph.ids["dept"]], ALL, max_hops=2)
    depths = {node.id: node.depth for node in result.nodes}
    assert depths == {graph.ids["dept"]: 0, graph.ids["team"]: 1}

    [edge] = result.edges
    assert edge.reversed is False
    assert edge.walked_as == "BELONGS_TO"


def test_edges_report_names_types_and_confidence(graph):
    graph.entity("ada", EntityType.PERSON)
    graph.entity("engine", EntityType.PROJECT)
    graph.edge("ada", RelationType.WORKS_ON, "engine", confidence=None)

    result = traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=1)
    nodes = {node.id: node for node in result.nodes}
    assert nodes[graph.ids["engine"]].entity_type == EntityType.PROJECT
    assert nodes[graph.ids["engine"]].name == "engine"
    assert result.edges[0].confidence is None
    assert result.truncated is False
    assert result.empty_reason is None


def test_no_seeds_is_no_entity_matched(graph):
    result = traverse(graph.db, [], ALL, max_hops=2)
    assert result.nodes == [] and result.edges == []
    assert result.empty_reason == EmptyReason.NO_ENTITY_MATCHED


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_a_mention_matches_by_normalised_name_alias_and_type(graph):
    graph.entity("platform team", EntityType.TEAM, aliases=["Plat Team", "PT"])
    graph.entity("platform team", EntityType.PROJECT)
    team = graph.db.query(Entity).filter_by(entity_type="team").one().id
    project = graph.db.query(Entity).filter_by(entity_type="project").one().id

    assert match_entities(graph.db, [EntityMention(name='"Platform  Team."')], ALL) == sorted(
        [team, project]
    )
    assert match_entities(
        graph.db, [EntityMention(name="platform team", entity_type=EntityType.TEAM)], ALL
    ) == [team]
    assert match_entities(graph.db, [EntityMention(name="plat team")], ALL) == [team]
    assert (
        match_entities(graph.db, [EntityMention(name="pt", entity_type=EntityType.PERSON)], ALL)
        == []
    )
    assert match_entities(graph.db, [], ALL) == []


def test_a_denied_entity_does_not_match_by_alias(graph):
    graph.entity("platform team", EntityType.TEAM, chunks=(graph.secret_chunk,), aliases=["PT"])
    assert match_entities(graph.db, [EntityMention(name="PT")], graph.restricted) == []
    assert len(match_entities(graph.db, [EntityMention(name="PT")], ALL)) == 1


def test_a_malformed_aliases_value_on_another_entity_does_not_break_matching(graph):
    graph.entity("platform team", EntityType.TEAM, aliases=["PT"])
    graph.entity("broken object", EntityType.TEAM, aliases={"not": "a list"})
    graph.entity("broken scalar", EntityType.TEAM, aliases="PT")
    graph.entity("mixed", EntityType.TEAM, aliases=[7, None, "Mixed Up"])
    graph.db.commit()

    assert match_entities(graph.db, [EntityMention(name="pt")], ALL) == [graph.ids["platform team"]]
    assert match_entities(graph.db, [EntityMention(name="mixed up")], ALL) == [graph.ids["mixed"]]


# ---------------------------------------------------------------------------
# Visible entity source chunks (for the strategy)
# ---------------------------------------------------------------------------


def test_visible_entity_chunks_excludes_denied_chunks(graph):
    graph.entity("ada", chunks=(graph.open_chunk, graph.secret_chunk, graph.open_chunk_2))
    graph.entity("grace", chunks=(graph.secret_chunk,))
    graph.entity("linus")
    ids = [graph.ids["ada"], graph.ids["grace"], graph.ids["linus"]]

    assert visible_entity_chunks(graph.db, ids, ALL) == {
        graph.ids["ada"]: sorted([graph.open_chunk, graph.secret_chunk, graph.open_chunk_2]),
        graph.ids["grace"]: [graph.secret_chunk],
        graph.ids["linus"]: [graph.open_chunk],
    }
    restricted = visible_entity_chunks(graph.db, ids, graph.restricted)
    assert restricted == {
        graph.ids["ada"]: sorted([graph.open_chunk, graph.open_chunk_2]),
        graph.ids["linus"]: [graph.open_chunk],
    }
    assert graph.ids["grace"] not in restricted
    assert visible_entity_chunks(graph.db, [], ALL) == {}


# ---------------------------------------------------------------------------
# Allow lists applied to the walk itself, not only to the seed
# ---------------------------------------------------------------------------


def test_a_collection_allow_list_blocks_an_edge_whose_only_chunk_is_outside_it(graph):
    graph.entity("ada")
    graph.entity("grace")
    graph.entity("linus")
    graph.edge("ada", RelationType.REPORTS_TO, "grace")
    graph.edge("ada", RelationType.OWNS, "linus", chunks=(graph.secret_chunk,))

    only_open = AccessFilter(collection_ids=frozenset({graph.open_collection}))
    assert len(traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=1).edges) == 2
    result = traverse(graph.db, [graph.ids["ada"]], only_open, max_hops=2)
    assert graph.names(result) == {"ada", "grace"}
    [edge] = result.edges
    assert edge.walked_as == "REPORTS_TO"


def test_a_document_deny_inside_an_allowed_collection_is_applied_in_the_walk(graph):
    graph.entity("ada")
    graph.entity("grace")
    graph.entity("linus")
    graph.entity("hopper", chunks=(graph.other_open_chunk,))
    graph.edge("ada", RelationType.REPORTS_TO, "grace")
    # Walkable only through a chunk of the denied document.
    graph.edge("ada", RelationType.OWNS, "linus", chunks=(graph.other_open_chunk,))
    # Admitted edge, but it leads to a node only the denied document mentions.
    graph.edge("ada", RelationType.WORKS_ON, "hopper")

    access = AccessFilter(
        collection_ids=frozenset({graph.open_collection}),
        denied_document_ids=frozenset({graph.other_open_document}),
    )
    assert len(traverse(graph.db, [graph.ids["ada"]], ALL, max_hops=1).edges) == 3
    result = traverse(graph.db, [graph.ids["ada"]], access, max_hops=2)
    assert graph.names(result) == {"ada", "grace"}
    assert [edge.walked_as for edge in result.edges] == ["REPORTS_TO"]
    assert match_entities(graph.db, [EntityMention(name="hopper")], access) == []
