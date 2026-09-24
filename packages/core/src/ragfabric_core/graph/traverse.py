"""Walk the knowledge graph outward from matched entities, with the access filter inside the query.

Two functions, used in order by the graph strategy:

- ``match_entities`` turns the entities a question mentions into the ids of
  entities the caller may see.
- ``traverse`` walks from those ids up to ``max_hops`` and returns a
  ``Subgraph``.

**Access (ADR 0003, ruling R10).** A node is visible iff at least one
``entity_sources`` row points at a chunk the ``AccessFilter`` admits. An edge
is walkable iff at least one ``relationship_sources`` chunk is admitted and the
node it leads to is visible. Both predicates are correlated ``EXISTS``
subqueries built with ``access_clause`` and placed in the SQL itself: in the
match, in the anchor of the recursive CTE, and in its recursive term. Nothing
is filtered after the walk, because a walk that crosses a denied edge and drops
it afterwards has already used it to reach whatever lies beyond, and matching
an entity at all reveals that it exists.

**Direction (ruling R2).** Every known ``RelationType`` walks forwards. It
walks backwards only when it is in ``INVERSES``, and is then reported under
``INVERSES[rel]`` with ``reversed=True``. A relation not in ``INVERSES``
(``REPORTS_TO`` today) is never walked backwards, since reading it that way
asserts the opposite fact. A stored relation type that is not a
``RelationType`` has no direction rule and is not walked at all.

**Termination.** Each CTE row carries the path of node ids it took, as a
delimited string. A step into a node already on its path is kept, so the edge
that closes a cycle is reported, but it is flagged ``closes_cycle`` and never
expanded, so cycles terminate; ``depth < max_hops`` bounds the rest. Because visited nodes
are tracked per path rather than globally, a node can be reached along several
paths; the result is deduplicated, and Task 7's node budget is what caps the
work on dense graphs.

The CTE uses only portable SQL (``WITH RECURSIVE``, ``CASE``, ``||``, ``LIKE``,
``CAST(... AS TEXT)``) and runs unchanged on SQLite and PostgreSQL. It has a
single recursive reference joined with one ``OR`` condition, rather than two
``UNION ALL`` branches, because PostgreSQL allows the recursive table to be
referenced only once in the recursive term.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from sqlalchemy import (
    Integer,
    Text,
    and_,
    case,
    cast,
    func,
    literal_column,
    null,
    or_,
    select,
)
from sqlalchemy.orm import Session, aliased

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.graph.contracts import (
    INVERSES,
    EmptyReason,
    EntityMention,
    EntityType,
    GraphEdge,
    GraphNode,
    RelationType,
    Subgraph,
    normalise,
)
from ragfabric_core.models.document import Chunk
from ragfabric_core.models.graph import Entity, EntitySource, Relationship, RelationshipSource
from ragfabric_core.stores.access_sql import access_clause

_KNOWN = sorted(rel.value for rel in RelationType)
_INVERTIBLE = sorted(rel.value for rel in INVERSES)


def _entity_visible(entity_id, access: AccessFilter):
    """``EXISTS`` an admitted chunk among the entity's sources, correlated to ``entity_id``."""
    source = aliased(EntitySource)
    chunk = aliased(Chunk)
    query = (
        select(literal_column("1"))
        .select_from(source)
        .join(chunk, chunk.id == source.chunk_id)
        .where(source.entity_id == entity_id)
    )
    clause = access_clause(access, chunk.document_id, chunk.collection_id)
    if clause is not None:
        query = query.where(clause)
    return query.exists()


def _relationship_visible(relationship_id, access: AccessFilter):
    """``EXISTS`` an admitted chunk among the relationship's sources."""
    source = aliased(RelationshipSource)
    chunk = aliased(Chunk)
    query = (
        select(literal_column("1"))
        .select_from(source)
        .join(chunk, chunk.id == source.chunk_id)
        .where(source.relationship_id == relationship_id)
    )
    clause = access_clause(access, chunk.document_id, chunk.collection_id)
    if clause is not None:
        query = query.where(clause)
    return query.exists()


def match_entities(
    db: Session, mentions: Sequence[EntityMention], access: AccessFilter
) -> list[int]:
    """Ids of visible entities a question's mentions name, sorted ascending.

    A mention matches an entity when ``normalise(mention.name)`` equals the
    entity's ``normalized_name`` or the ``normalise`` of one of its aliases,
    and, when the mention carries an ``entity_type``, the types agree. The
    visibility predicate is in both queries, so an entity whose every source
    chunk is denied never matches.

    Aliases are compared in Python after the SQL has already restricted the
    candidates to visible entities that have aliases, because ``normalise``
    (casefold, Unicode punctuation) has no portable SQL equivalent.
    """
    wanted = [(normalise(m.name), m.entity_type) for m in mentions]
    wanted = [(name, entity_type) for name, entity_type in wanted if name]
    if not wanted:
        return []

    visible = _entity_visible(Entity.id, access)
    by_name = or_(
        *(
            Entity.normalized_name == name
            if entity_type is None
            else and_(Entity.normalized_name == name, Entity.entity_type == entity_type.value)
            for name, entity_type in wanted
        )
    )
    matched = set(db.execute(select(Entity.id).where(visible, by_name)).scalars())

    aliased_rows = db.execute(
        select(Entity.id, Entity.entity_type, Entity.aliases).where(
            visible, func.json_array_length(Entity.aliases) > 0
        )
    ).all()
    for entity_id, entity_type, aliases in aliased_rows:
        names = {normalise(alias) for alias in aliases or [] if isinstance(alias, str)}
        for name, want_type in wanted:
            if name in names and (want_type is None or want_type.value == entity_type):
                matched.add(entity_id)
                break
    return sorted(matched)


def _walk_statement(seed_ids: list[int], access: AccessFilter, max_hops: int):
    """The recursive CTE and the select over it: one row per (node reached, edge used)."""
    path_of = cast(literal_column("','"), Text)
    anchor = select(
        Entity.id.label("node_id"),
        cast(literal_column("0"), Integer).label("depth"),
        (path_of + cast(Entity.id, Text) + path_of).label("path"),
        cast(null(), Integer).label("rel_id"),
        cast(literal_column("0"), Integer).label("is_reversed"),
        cast(literal_column("0"), Integer).label("closes_cycle"),
    ).where(Entity.id.in_(seed_ids), _entity_visible(Entity.id, access))
    walk = anchor.cte("walk", recursive=True)

    rel = aliased(Relationship)
    forwards = and_(rel.source_entity_id == walk.c.node_id, rel.relation_type.in_(_KNOWN))
    backwards = and_(rel.target_entity_id == walk.c.node_id, rel.relation_type.in_(_INVERTIBLE))
    # A row joined through ``forwards`` leads to the target; otherwise it came
    # through ``backwards`` and leads to the source. A self-loop reads as forwards.
    goes_forwards = rel.source_entity_id == walk.c.node_id
    next_node = case((goes_forwards, rel.target_entity_id), else_=rel.source_entity_id)
    on_path = walk.c.path.contains(path_of + cast(next_node, Text) + path_of)
    step = (
        select(
            next_node.label("node_id"),
            (walk.c.depth + 1).label("depth"),
            (walk.c.path + cast(next_node, Text) + path_of).label("path"),
            rel.id.label("rel_id"),
            case((goes_forwards, 0), else_=1).label("is_reversed"),
            case((on_path, 1), else_=0).label("closes_cycle"),
        )
        .select_from(walk.join(rel, or_(forwards, backwards)))
        .where(
            walk.c.depth < max_hops,
            walk.c.closes_cycle == 0,
            _relationship_visible(rel.id, access),
            _entity_visible(next_node, access),
        )
    )
    walk = walk.union_all(step)
    return select(walk.c.node_id, walk.c.rel_id, walk.c.is_reversed).distinct()


def traverse(
    db: Session,
    seed_ids: Iterable[int],
    access: AccessFilter,
    *,
    max_hops: int,
) -> Subgraph:
    """Walk from ``seed_ids`` up to ``max_hops`` edges, under ``access``.

    Seeds are re-checked for visibility inside the anchor, so passing an id the
    caller may not see yields no node for it. Each edge appears once; when it
    was reached in both directions the forward reading is reported. An edge's
    ``source_chunk_ids`` are its admitted source chunks only.

    ``empty_reason`` is set only as far as the ``Subgraph`` contract requires:
    ``NO_ENTITY_MATCHED`` when no seed is visible, ``NO_WALKABLE_EDGES`` when
    seeds are visible but no edge was walked. Telling a graph with no coverage
    apart from a non-match is the strategy's job. ``truncated`` is always False
    here; the node budget that can truncate a walk is added separately.
    """
    if max_hops < 0:
        raise ValueError(f"max_hops must be at least 0, got {max_hops}")
    seeds = sorted(set(seed_ids))
    if not seeds:
        return Subgraph(
            nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_ENTITY_MATCHED
        )

    node_ids: set[int] = set()
    edge_reversed: dict[int, bool] = {}
    for node_id, rel_id, is_reversed in db.execute(_walk_statement(seeds, access, max_hops)):
        node_ids.add(node_id)
        if rel_id is not None:
            # Forward wins when an edge was reached both ways: it needs no inverse.
            edge_reversed[rel_id] = edge_reversed.get(rel_id, True) and bool(is_reversed)

    if not node_ids:
        return Subgraph(
            nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_ENTITY_MATCHED
        )

    nodes = [
        GraphNode(id=entity_id, name=name, entity_type=EntityType(entity_type))
        for entity_id, name, entity_type in db.execute(
            select(Entity.id, Entity.name, Entity.entity_type)
            .where(Entity.id.in_(sorted(node_ids)))
            .order_by(Entity.id)
        )
    ]
    edges = _edges(db, edge_reversed, access)
    return Subgraph(
        nodes=nodes,
        edges=edges,
        truncated=False,
        empty_reason=None if edges else EmptyReason.NO_WALKABLE_EDGES,
    )


def _edges(db: Session, edge_reversed: dict[int, bool], access: AccessFilter) -> list[GraphEdge]:
    if not edge_reversed:
        return []
    ids = sorted(edge_reversed)
    chunk_query = (
        select(RelationshipSource.relationship_id, RelationshipSource.chunk_id)
        .join(Chunk, Chunk.id == RelationshipSource.chunk_id)
        .where(RelationshipSource.relationship_id.in_(ids))
    )
    clause = access_clause(access, Chunk.document_id, Chunk.collection_id)
    if clause is not None:
        chunk_query = chunk_query.where(clause)
    chunks: dict[int, list[int]] = {}
    for relationship_id, chunk_id in db.execute(chunk_query):
        chunks.setdefault(relationship_id, []).append(chunk_id)

    edges = []
    for row in db.execute(
        select(
            Relationship.id,
            Relationship.source_entity_id,
            Relationship.target_entity_id,
            Relationship.relation_type,
            Relationship.confidence,
        )
        .where(Relationship.id.in_(ids))
        .order_by(Relationship.id)
    ):
        relation = RelationType(row.relation_type)
        is_reversed = edge_reversed[row.id]
        edges.append(
            GraphEdge(
                id=row.id,
                source_id=row.source_entity_id,
                target_id=row.target_entity_id,
                relation_type=relation,
                walked_as=INVERSES[relation] if is_reversed else relation.value,
                reversed=is_reversed,
                confidence=row.confidence,
                source_chunk_ids=sorted(chunks.get(row.id, [])),
            )
        )
    return edges
