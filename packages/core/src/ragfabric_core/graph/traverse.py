"""Walk the knowledge graph outward from matched entities, with the access filter inside the query.

Three functions, used in order by the graph strategy:

- ``match_entities`` turns the entities a question mentions into the ids of
  entities the caller may see.
- ``traverse`` walks from those ids up to ``max_hops`` and returns a
  ``Subgraph``.
- ``visible_entity_chunks`` gives the admitted source chunks of entities, so
  the strategy never writes its own access join.

**Access (ADR 0003, ruling R10).** A node is visible iff at least one
``entity_sources`` row points at a chunk the ``AccessFilter`` admits. An edge
is walkable iff at least one ``relationship_sources`` chunk is admitted and the
node it leads to is visible. Both predicates are correlated ``EXISTS``
subqueries built with ``access_clause`` and placed in the SQL itself: in the
match, in the anchor of the recursive CTE, in its recursive term, and in the
edge fetch. Nothing is filtered after the walk, because a walk that crosses a
denied edge and drops it afterwards has already used it to reach whatever lies
beyond, and matching an entity at all reveals that it exists.

**Direction (ruling R2).** Every known ``RelationType`` walks forwards. It
walks backwards only when it is in ``INVERSES``, and is then reported under
``INVERSES[rel]`` with ``reversed=True``. A relation not in ``INVERSES``
(``REPORTS_TO`` today) is never walked backwards, since reading it that way
asserts the opposite fact. A stored relation type that is not a
``RelationType`` has no direction rule and is not walked at all.

**Shape of the walk (ruling R12).** The recursive CTE produces
``(node_id, depth)`` rows combined with ``UNION``, which deduplicates, and
stops at ``depth = max_hops``. That bounds the work by visible nodes times
hops, where tracking a path per row would grow as degree to the power of hops.
Cycles terminate because a revisited ``(node, depth)`` pair adds no row and
depth is bounded. The minimum depth per node is then taken, and the edges are
fetched in a second statement: an edge is reported only when the walk could
have used it, that is, it joins two reached nodes, the node it leaves was
reached below the hop limit, it is walkable in that direction under the rules
above, and it has an admitted source chunk and a visible destination.
``max_hops`` must lie in ``1..MAX_HOPS_CEILING``.

**Guidance and budget (ruling R16).** ``relation_types`` narrows the walk to
the types a question implies. It is part of the join condition in the recursive
term and of the edge fetch, so a node reachable only through an unimplied type
is never reached, and an unimplied edge between reached nodes is never
reported. A relation walked backwards is filtered by its stored type, so
implying ``BELONGS_TO`` admits it read as ``CONTAINS``. ``None`` or an empty
collection means every type. ``node_budget`` caps the reached set: nodes are
ordered by (minimum depth, id), which puts the seeds (depth 0) first, and only
the first ``node_budget`` are kept, seeds included, so the cap is hard. Edges
are fetched among the kept nodes only, so every edge touching a dropped node is
dropped, and ``truncated`` is True exactly when more nodes were reached than
kept. Because every node at a smaller depth is kept before any node at a larger
one, each kept node still has a kept node one hop nearer the seeds. The budget
bounds the result, not the CTE's own work, which R12's hop ceiling bounds; a
``LIMIT`` inside the recursive term is not allowed on PostgreSQL.

Everything is portable SQL (``WITH RECURSIVE``, ``UNION``, ``CASE``,
``EXISTS``) and runs unchanged on SQLite and PostgreSQL. The recursive term has
a single reference to the CTE joined with one ``OR`` condition, rather than two
branches, because PostgreSQL allows the recursive table to be referenced only
once in the recursive term.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence

from sqlalchemy import Integer, Text, and_, case, cast, func, literal_column, or_, select
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

MAX_HOPS_CEILING = 4

# Enough for a focused answer over a few hops, small enough that a dense
# neighbourhood cannot flood the context. Callers pass their own when they know better.
DEFAULT_NODE_BUDGET = 50

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

    Aliases are compared in Python over rows the SQL has already restricted to
    visible entities, because ``normalise`` (casefold, Unicode punctuation) has
    no portable SQL equivalent. An ``aliases`` value that is not a list, and an
    alias that is not a string, are skipped rather than failing the match.
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

    # Comparing the serialised text skips the common empty list without any
    # JSON function, which would raise on a row whose value is not an array.
    candidates = db.execute(
        select(Entity.id, Entity.entity_type, Entity.aliases).where(
            visible, cast(Entity.aliases, Text) != "[]"
        )
    ).all()
    for entity_id, entity_type, aliases in candidates:
        if not isinstance(aliases, list):
            continue
        names = {normalise(alias) for alias in aliases if isinstance(alias, str)}
        for name, want_type in wanted:
            if name in names and (want_type is None or want_type.value == entity_type):
                matched.add(entity_id)
                break
    return sorted(matched)


def visible_entity_chunks(
    db: Session, entity_ids: Iterable[int], access: AccessFilter
) -> dict[int, list[int]]:
    """Admitted source chunk ids per entity, sorted. An entity with none is absent."""
    ids = sorted(set(entity_ids))
    if not ids:
        return {}
    query = (
        select(EntitySource.entity_id, EntitySource.chunk_id)
        .join(Chunk, Chunk.id == EntitySource.chunk_id)
        .where(EntitySource.entity_id.in_(ids))
    )
    clause = access_clause(access, Chunk.document_id, Chunk.collection_id)
    if clause is not None:
        query = query.where(clause)
    out: dict[int, list[int]] = {}
    for entity_id, chunk_id in db.execute(query):
        out.setdefault(entity_id, []).append(chunk_id)
    return {entity_id: sorted(chunks) for entity_id, chunks in sorted(out.items())}


def _walkable_types(
    relation_types: Collection[RelationType] | None,
) -> tuple[list[str], list[str]]:
    """Stored type values walkable forwards and backwards. ``None`` or empty means all."""
    if not relation_types:
        return _KNOWN, _INVERTIBLE
    wanted = {RelationType(rel).value for rel in relation_types}
    return sorted(wanted), [value for value in _INVERTIBLE if value in wanted]


def _reach_statement(
    seed_ids: list[int],
    access: AccessFilter,
    max_hops: int,
    known: list[str],
    invertible: list[str],
    limit: int,
):
    """The recursive CTE, then the minimum depth of each node, nearest first, up to ``limit``.

    Ordered by (minimum depth, id), so seeds come first and the order is stable.
    """
    anchor = select(
        Entity.id.label("node_id"),
        cast(literal_column("0"), Integer).label("depth"),
    ).where(Entity.id.in_(seed_ids), _entity_visible(Entity.id, access))
    reach = anchor.cte("reach", recursive=True)

    rel = aliased(Relationship)
    forwards = and_(rel.source_entity_id == reach.c.node_id, rel.relation_type.in_(known))
    backwards = and_(rel.target_entity_id == reach.c.node_id, rel.relation_type.in_(invertible))
    # A row joined through ``forwards`` leads to the target; otherwise it came
    # through ``backwards`` and leads to the source. A self-loop reads as forwards.
    next_node = case(
        (rel.source_entity_id == reach.c.node_id, rel.target_entity_id),
        else_=rel.source_entity_id,
    )
    step = (
        select(next_node.label("node_id"), (reach.c.depth + 1).label("depth"))
        .select_from(reach.join(rel, or_(forwards, backwards)))
        .where(
            reach.c.depth < max_hops,
            _relationship_visible(rel.id, access),
            _entity_visible(next_node, access),
        )
    )
    reach = reach.union(step)
    min_depth = func.min(reach.c.depth)
    return (
        select(reach.c.node_id, min_depth)
        .group_by(reach.c.node_id)
        .order_by(min_depth, reach.c.node_id)
        .limit(limit)
    )


def _edge_statement(
    depths: dict[int, int],
    access: AccessFilter,
    max_hops: int,
    known: list[str],
    invertible: list[str],
):
    """Edges the walk could have used among the reached nodes, with their direction flag.

    ``goes_forwards`` is 1 when the edge is walkable forwards from its source,
    which is preferred when both directions are; otherwise it is walkable
    backwards from its target, which only an invertible relation can be.
    """
    reached = sorted(depths)
    expandable = sorted(node for node, depth in depths.items() if depth < max_hops)
    rel = aliased(Relationship)
    forwards = and_(
        rel.relation_type.in_(known),
        rel.source_entity_id.in_(expandable),
        rel.target_entity_id.in_(reached),
        _entity_visible(rel.target_entity_id, access),
    )
    backwards = and_(
        rel.relation_type.in_(invertible),
        rel.target_entity_id.in_(expandable),
        rel.source_entity_id.in_(reached),
        _entity_visible(rel.source_entity_id, access),
    )
    return (
        select(
            rel.id,
            rel.source_entity_id,
            rel.target_entity_id,
            rel.relation_type,
            rel.confidence,
            case((forwards, 1), else_=0).label("goes_forwards"),
        )
        .where(or_(forwards, backwards), _relationship_visible(rel.id, access))
        .order_by(rel.id)
    )


def traverse(
    db: Session,
    seed_ids: Iterable[int],
    access: AccessFilter,
    *,
    max_hops: int,
    node_budget: int = DEFAULT_NODE_BUDGET,
    relation_types: Collection[RelationType] | None = None,
) -> Subgraph:
    """Walk from ``seed_ids`` up to ``max_hops`` edges, under ``access``.

    ``max_hops`` must be in ``1..MAX_HOPS_CEILING`` and ``node_budget`` at
    least 1; anything else raises ``ValueError``. Only ``relation_types`` are
    walked (``None`` or empty: all), and at most ``node_budget`` nodes are kept,
    nearest first by (depth, id) with seeds first; ``truncated`` says whether
    any reached node was dropped. Seeds are re-checked for visibility inside the anchor, so
    passing an id the caller may not see yields no node for it. Each edge
    appears once; when it is walkable in both directions the forward reading is
    reported. An edge's ``source_chunk_ids`` are its admitted source chunks only.

    ``empty_reason`` is set only as far as the ``Subgraph`` contract requires:
    ``NO_ENTITY_MATCHED`` when no seed is visible, ``NO_WALKABLE_EDGES`` when
    seeds are visible but no edge was walked. Telling a graph with no coverage
    apart from a non-match is the strategy's job. A truncated walk that kept no
    edge still reports ``NO_WALKABLE_EDGES``, as the contract requires.
    """
    if not 1 <= max_hops <= MAX_HOPS_CEILING:
        raise ValueError(f"max_hops must be between 1 and {MAX_HOPS_CEILING}, got {max_hops}")
    if node_budget < 1:
        raise ValueError(f"node_budget must be at least 1, got {node_budget}")
    known, invertible = _walkable_types(relation_types)
    seeds = sorted(set(seed_ids))
    if not seeds:
        return Subgraph(
            nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_ENTITY_MATCHED
        )

    # One row past the budget tells a walk that was cut short from one that fitted.
    reached = db.execute(
        _reach_statement(seeds, access, max_hops, known, invertible, node_budget + 1)
    ).all()
    truncated = len(reached) > node_budget
    depths = {node_id: depth for node_id, depth in reached[:node_budget]}
    if not depths:
        return Subgraph(
            nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_ENTITY_MATCHED
        )

    nodes = [
        GraphNode(id=entity_id, name=name, entity_type=EntityType(entity_type))
        for entity_id, name, entity_type in db.execute(
            select(Entity.id, Entity.name, Entity.entity_type)
            .where(Entity.id.in_(sorted(depths)))
            .order_by(Entity.id)
        )
    ]
    edges = _edges(db, depths, access, max_hops, known, invertible)
    return Subgraph(
        nodes=nodes,
        edges=edges,
        truncated=truncated,
        empty_reason=None if edges else EmptyReason.NO_WALKABLE_EDGES,
    )


def _edges(
    db: Session,
    depths: dict[int, int],
    access: AccessFilter,
    max_hops: int,
    known: list[str],
    invertible: list[str],
) -> list[GraphEdge]:
    rows = db.execute(_edge_statement(depths, access, max_hops, known, invertible)).all()
    if not rows:
        return []
    ids = [row.id for row in rows]
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
    for row in rows:
        relation = RelationType(row.relation_type)
        is_reversed = not row.goes_forwards
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
