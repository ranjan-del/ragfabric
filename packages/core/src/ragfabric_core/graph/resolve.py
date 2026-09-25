"""Entity resolution: recorded, reversible merges.

Entity resolution is the main failure mode of graph RAG. Merge too eagerly
and two people named Sharma become one node that the graph then asserts
things about; merge too little and a traversal that should connect two facts
finds no path. Neither can be tuned without measurement (Phase 8), so what
this module guarantees is not a perfect threshold but that every merge is
**recorded with the evidence that justified it** and can be **undone**.

Three stages run in order of confidence. Entities of different types never
merge at any stage: every candidate pair passes through one predicate,
``_may_merge``, and that predicate requires equal ``entity_type``.

1. ``exact``: the two entities' names normalise equal (``normalise(name)`` or
   the normalised stored ``normalized_name``). The unique constraint already
   folds most of these at write time (R5); this stage catches rows whose
   stored key was written differently, for example by an older normaliser.
2. ``alias``: a normalised alias of one entity equals the other's normalised
   name or one of its normalised aliases. A malformed ``aliases`` value (not
   a list) or a non-string item is skipped, never fatal (R14).
3. ``embedding``: cosine similarity between ``"name: description"`` (just the
   name when there is no description) of two same-type entities is at or
   above ``similarity_threshold``. Only entities in a type with at least two
   live candidates are embedded, in batches of ``batch_size`` per call, and
   the number of calls is reported. Pairs are merged greedily from the most
   similar down; a pair whose member was already merged away in this stage is
   skipped rather than chained through the survivor, because the similarity
   was measured against the entity that no longer exists.

Stages 1 and 2 repeat until no pair is left, because a merge appends the
merged name to the survivor's aliases and that can create a new alias match.

**Survivor choice** is deterministic: the entity with a measured confidence
beats one without; then the higher confidence; then more ``EntitySource``
rows; then the lower id.

**A merge** (R24) moves the merged-away entity's ``EntitySource`` rows onto
the survivor with their per-chunk confidence (a chunk that sourced both keeps
the higher of the two reports on the survivor's row), repoints its
relationships to the survivor, folds any edge that now duplicates another
(same source, target and relation type) by moving its ``RelationshipSource``
rows, drops any edge that would become a self-loop, appends the merged name to
the survivor's aliases (unless it is already there), hands any merge records
the merged-away entity had absorbed to the survivor, and recomputes
confidences from the remaining sources (R20). Every row is deleted through the
ORM, never left to the database cascade (R23), so the session never holds a
row the database has already dropped.

**The ``EntityMerge`` row** records ``merged_source_chunk_ids`` (the chunks
that sourced the merged-away entity), ``method`` (the stage), ``model`` (the
embedding model for an ``embedding`` merge, ``None`` otherwise) and
``evidence``:

- ``stage``, and ``survivor`` / ``merged`` as ``{"id", "name", "type"}``;
- ``exact``: ``normalized_name``, the key both names normalise to;
- ``alias``: ``alias`` (as stored), ``normalized_alias``, ``alias_of``
  (``"survivor"`` or ``"merged"``, whose alias it was) and ``matched``
  (``"name"`` or ``"alias"``, what it matched on the other entity);
- ``embedding``: ``similarity`` as measured, ``threshold`` and
  ``embedding_model``;
- ``restore``: not justification but what makes the unmerge exact: the
  merged entity's stored ``normalized_name`` and ``description``, whether its
  name was ``alias_appended`` to the survivor, the ``shared_sources`` (chunks
  that sourced both entities, with both confidences), the
  ``dropped_self_loops`` (with their source rows) and the
  ``repointed_merge_ids``.

**An unmerge** recreates the entity (under a new id) with its name, type,
aliases and description, moves back exactly the ``EntitySource`` rows for
``merged_source_chunk_ids`` (a shared chunk gets both original reports back),
moves back every relationship of the survivor whose every source chunk is in
``merged_source_chunk_ids``, recreates the dropped self-loops whose chunks
still exist, removes the merged name from the survivor's aliases if the merge
added it, hands back the merge records it had absorbed, recomputes confidences
and deletes the ``EntityMerge`` row. A relationship with a source chunk on
each side stays with the survivor, and ``UnmergeResult`` lists it: provenance,
not guesswork, decides what belongs to whom. Unmerge the most recent merge
first when merges chain; an older merge's chunks may since have moved.

Public API (called by ingestion in Task 11 and the CLI in Task 12)::

    resolve_entities(db, embedder, *, similarity_threshold, entity_ids=None,
                     batch_size=64) -> ResolutionReport
    unmerge(db, merge_id) -> UnmergeResult

``embedder`` may be ``None``, which skips stage 3 (0 embedding calls).
``entity_ids`` restricts resolution to pairs that involve at least one listed
entity (a survivor of such a merge joins the set); the rest of the graph is
still the pool they are compared against. Both functions flush; the caller
commits.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ragfabric_core.embeddings.normalise import normalise as unit_vector
from ragfabric_core.graph.contracts import normalise
from ragfabric_core.graph.extract import (
    _recompute_entity_confidence,
    _recompute_relationship_confidence,
)
from ragfabric_core.models.document import Chunk
from ragfabric_core.models.graph import (
    Entity,
    EntityMerge,
    EntitySource,
    Relationship,
    RelationshipSource,
)
from ragfabric_core.providers.base import EmbeddingProvider

METHOD_EXACT = "exact"
METHOD_ALIAS = "alias"
METHOD_EMBEDDING = "embedding"

DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class MergeOutcome:
    """One merge made by ``resolve_entities``. ``merged_id`` no longer exists."""

    merge_id: int
    survivor_id: int
    merged_id: int
    method: str


@dataclass(frozen=True)
class ResolutionReport:
    merges: list[MergeOutcome]
    embedding_calls: int


@dataclass(frozen=True)
class UnmergeResult:
    """What an unmerge did.

    ``shared_relationship_ids`` are the survivor's edges with source chunks
    on both sides of the split; they stay with the survivor (R24).
    ``restored_relationship_ids`` are the self-loops the merge had dropped,
    recreated between the two entities.
    """

    restored_entity_id: int
    survivor_entity_id: int
    moved_relationship_ids: list[int]
    shared_relationship_ids: list[int]
    restored_relationship_ids: list[int]


def resolve_entities(
    db: Session,
    embedder: EmbeddingProvider | None,
    *,
    similarity_threshold: float,
    entity_ids: Iterable[int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ResolutionReport:
    """Run the three resolution stages and record every merge. See the module docstring."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    resolver = _Resolver(db, None if entity_ids is None else set(entity_ids))
    while resolver.merge_next_exact() or resolver.merge_next_alias():
        pass
    embedding_calls = 0
    if embedder is not None:
        embedding_calls = resolver.merge_by_embedding(
            embedder, threshold=similarity_threshold, batch_size=batch_size
        )
    return ResolutionReport(merges=resolver.merges, embedding_calls=embedding_calls)


def _may_merge(a: Entity, b: Entity) -> bool:
    """The one predicate every stage's candidate pair must pass."""
    return a.id != b.id and a.entity_type == b.entity_type


def _aliases(entity: Entity) -> list[str]:
    """The entity's string aliases; a malformed value or item is skipped (R14)."""
    if not isinstance(entity.aliases, list):
        return []
    return [alias for alias in entity.aliases if isinstance(alias, str)]


def _name_keys(entity: Entity) -> set[str]:
    return {normalise(entity.name), normalise(entity.normalized_name)} - {""}


def _embedding_text(entity: Entity) -> str:
    return f"{entity.name}: {entity.description}" if entity.description else entity.name


def _describe(entity: Entity) -> dict[str, Any]:
    return {"id": entity.id, "name": entity.name, "type": entity.entity_type}


def _higher(candidate: float | None, current: float | None) -> bool:
    return candidate is not None and (current is None or candidate > current)


class _Resolver:
    def __init__(self, db: Session, scope: set[int] | None) -> None:
        self.db = db
        self.scope = scope
        self.merges: list[MergeOutcome] = []
        entities = db.execute(select(Entity).order_by(Entity.id)).scalars()
        self.live: dict[int, Entity] = {entity.id: entity for entity in entities}

    def _entities(self) -> list[Entity]:
        return [self.live[entity_id] for entity_id in sorted(self.live)]

    def _in_scope(self, a: Entity, b: Entity) -> bool:
        return self.scope is None or a.id in self.scope or b.id in self.scope

    def _candidate(self, a: Entity, b: Entity) -> bool:
        return _may_merge(a, b) and self._in_scope(a, b)

    def merge_next_exact(self) -> bool:
        index: dict[str, list[Entity]] = defaultdict(list)
        for entity in self._entities():
            for key in _name_keys(entity):
                index[key].append(entity)
        for key in sorted(index):
            group = index[key]
            for position, a in enumerate(group):
                for b in group[position + 1 :]:
                    if self._candidate(a, b):
                        self._merge(a, b, METHOD_EXACT, None, {"normalized_name": key})
                        return True
        return False

    def merge_next_alias(self) -> bool:
        # key -> (entity, "name" | "alias", the stored string)
        index: dict[str, list[tuple[Entity, str, str]]] = defaultdict(list)
        for entity in self._entities():
            for key in sorted(_name_keys(entity)):
                index[key].append((entity, "name", entity.name))
            for alias in _aliases(entity):
                key = normalise(alias)
                if key:
                    index[key].append((entity, "alias", alias))
        for key in sorted(index):
            entries = index[key]
            for position, (a, a_role, a_text) in enumerate(entries):
                for b, b_role, b_text in entries[position + 1 :]:
                    if "alias" not in (a_role, b_role) or not self._candidate(a, b):
                        continue
                    if a_role == "alias":
                        owner, alias, matched = a, a_text, b_role
                    else:
                        owner, alias, matched = b, b_text, a_role
                    self._merge(
                        a,
                        b,
                        METHOD_ALIAS,
                        None,
                        {"alias": alias, "normalized_alias": key, "matched": matched},
                        alias_owner=owner,
                    )
                    return True
        return False

    def merge_by_embedding(
        self, embedder: EmbeddingProvider, *, threshold: float, batch_size: int
    ) -> int:
        by_type: dict[str, list[Entity]] = defaultdict(list)
        for entity in self._entities():
            by_type[entity.entity_type].append(entity)
        # Embed only what could pair: a type with one live entity, or with
        # none in scope, has no candidate pair and is not worth a call.
        embedded = [
            entity
            for group in by_type.values()
            if len(group) > 1
            and (self.scope is None or any(entity.id in self.scope for entity in group))
            for entity in group
        ]
        if len(embedded) < 2:
            return 0

        vectors: dict[int, list[float]] = {}
        models: set[str] = set()
        calls = 0
        for start in range(0, len(embedded), batch_size):
            batch = embedded[start : start + batch_size]
            result = embedder.embed([_embedding_text(entity) for entity in batch])
            calls += 1
            models.add(result.model)
            if len(result.vectors) != len(batch):
                raise ValueError(
                    f"embedder returned {len(result.vectors)} vectors for {len(batch)} texts"
                )
            for entity, vector in zip(batch, result.vectors, strict=True):
                vectors[entity.id] = unit_vector(vector)
        if len(models) != 1:
            raise ValueError(f"embedding batches came from different models: {sorted(models)}")
        (model,) = models

        pairs: list[tuple[float, Entity, Entity]] = []
        for position, a in enumerate(embedded):
            for b in embedded[position + 1 :]:
                if not self._candidate(a, b):
                    continue
                similarity = sum(x * y for x, y in zip(vectors[a.id], vectors[b.id], strict=True))
                if similarity >= threshold:
                    pairs.append((similarity, a, b))
        pairs.sort(
            key=lambda pair: (-pair[0], min(pair[1].id, pair[2].id), max(pair[1].id, pair[2].id))
        )

        for similarity, a, b in pairs:
            if a.id not in self.live or b.id not in self.live:
                continue
            self._merge(
                a,
                b,
                METHOD_EMBEDDING,
                model,
                {"similarity": similarity, "threshold": threshold, "embedding_model": model},
            )
        return calls

    def _rank(self, entity: Entity) -> tuple[bool, float, int, int]:
        sources = self.db.execute(
            select(func.count())
            .select_from(EntitySource)
            .where(EntitySource.entity_id == entity.id)
        ).scalar_one()
        confidence = entity.confidence
        return (confidence is not None, confidence or 0.0, sources, -entity.id)

    def _merge(
        self,
        a: Entity,
        b: Entity,
        method: str,
        model: str | None,
        detail: dict[str, Any],
        *,
        alias_owner: Entity | None = None,
    ) -> None:
        survivor, merged = (a, b) if self._rank(a) > self._rank(b) else (b, a)
        evidence: dict[str, Any] = {
            "stage": method,
            "survivor": _describe(survivor),
            "merged": _describe(merged),
            **detail,
        }
        if alias_owner is not None:
            evidence["alias_of"] = "survivor" if alias_owner.id == survivor.id else "merged"
        record_id = _merge_entities(
            self.db, survivor, merged, method=method, model=model, evidence=evidence
        )
        self.merges.append(
            MergeOutcome(
                merge_id=record_id, survivor_id=survivor.id, merged_id=merged.id, method=method
            )
        )
        if self.scope is not None and (a.id in self.scope or b.id in self.scope):
            self.scope.add(survivor.id)
        del self.live[merged.id]


def _entity_sources(db: Session, entity_id: int) -> list[EntitySource]:
    return list(
        db.execute(
            select(EntitySource)
            .where(EntitySource.entity_id == entity_id)
            .order_by(EntitySource.chunk_id)
        ).scalars()
    )


def _relationship_sources(db: Session, relationship_id: int) -> list[RelationshipSource]:
    return list(
        db.execute(
            select(RelationshipSource)
            .where(RelationshipSource.relationship_id == relationship_id)
            .order_by(RelationshipSource.chunk_id)
        ).scalars()
    )


def _touching(db: Session, entity_id: int) -> list[Relationship]:
    return list(
        db.execute(
            select(Relationship)
            .where(
                or_(
                    Relationship.source_entity_id == entity_id,
                    Relationship.target_entity_id == entity_id,
                )
            )
            .order_by(Relationship.id)
        ).scalars()
    )


def _merge_entities(
    db: Session,
    survivor: Entity,
    merged: Entity,
    *,
    method: str,
    model: str | None,
    evidence: dict[str, Any],
) -> int:
    """Fold ``merged`` into ``survivor`` (R24) and record it; return the merge row id."""
    survivor_sources = {source.chunk_id: source for source in _entity_sources(db, survivor.id)}
    merged_sources = _entity_sources(db, merged.id)
    merged_chunk_ids = [source.chunk_id for source in merged_sources]

    shared_sources: list[dict[str, Any]] = []
    for source in merged_sources:
        existing = survivor_sources.get(source.chunk_id)
        if existing is None:
            db.add(
                EntitySource(
                    entity_id=survivor.id,
                    chunk_id=source.chunk_id,
                    confidence=source.confidence,
                    extraction_model=source.extraction_model,
                )
            )
        else:
            # One chunk named both: the survivor's row keeps the higher
            # report, and both are kept for the unmerge.
            shared_sources.append(
                {
                    "chunk_id": source.chunk_id,
                    "survivor_confidence": existing.confidence,
                    "survivor_extraction_model": existing.extraction_model,
                    "merged_confidence": source.confidence,
                    "merged_extraction_model": source.extraction_model,
                }
            )
            if _higher(source.confidence, existing.confidence):
                existing.confidence = source.confidence
                existing.extraction_model = source.extraction_model
        db.delete(source)
    db.flush()

    touched_relationship_ids: set[int] = set()
    dropped_self_loops: list[dict[str, Any]] = []
    for relationship in _touching(db, merged.id):
        new_source = (
            survivor.id
            if relationship.source_entity_id == merged.id
            else relationship.source_entity_id
        )
        new_target = (
            survivor.id
            if relationship.target_entity_id == merged.id
            else relationship.target_entity_id
        )
        sources = _relationship_sources(db, relationship.id)

        if new_source == new_target:
            dropped_self_loops.append(
                {
                    "source": "survivor"
                    if relationship.source_entity_id == survivor.id
                    else "merged",
                    "target": "survivor"
                    if relationship.target_entity_id == survivor.id
                    else "merged",
                    "relation_type": relationship.relation_type,
                    "description": relationship.description,
                    "weight": relationship.weight,
                    "sources": [
                        {
                            "chunk_id": source.chunk_id,
                            "confidence": source.confidence,
                            "extraction_model": source.extraction_model,
                        }
                        for source in sources
                    ],
                }
            )
            _delete_relationship(db, relationship, sources)
            continue

        duplicate = db.execute(
            select(Relationship).where(
                Relationship.source_entity_id == new_source,
                Relationship.target_entity_id == new_target,
                Relationship.relation_type == relationship.relation_type,
                Relationship.id != relationship.id,
            )
        ).scalar_one_or_none()
        if duplicate is None:
            relationship.source_entity_id = new_source
            relationship.target_entity_id = new_target
            touched_relationship_ids.add(relationship.id)
            continue

        kept = {source.chunk_id: source for source in _relationship_sources(db, duplicate.id)}
        for source in sources:
            existing = kept.get(source.chunk_id)
            if existing is None:
                db.add(
                    RelationshipSource(
                        relationship_id=duplicate.id,
                        chunk_id=source.chunk_id,
                        confidence=source.confidence,
                        extraction_model=source.extraction_model,
                    )
                )
            elif _higher(source.confidence, existing.confidence):
                existing.confidence = source.confidence
                existing.extraction_model = source.extraction_model
        _delete_relationship(db, relationship, sources)
        touched_relationship_ids.add(duplicate.id)
    db.flush()

    alias_appended = False
    if isinstance(survivor.aliases, list) and merged.name not in survivor.aliases:
        survivor.aliases = [*survivor.aliases, merged.name]
        alias_appended = True

    absorbed = list(
        db.execute(
            select(EntityMerge).where(EntityMerge.surviving_entity_id == merged.id)
        ).scalars()
    )
    for earlier in absorbed:
        earlier.surviving_entity_id = survivor.id

    evidence = {
        **evidence,
        "restore": {
            "normalized_name": merged.normalized_name,
            "description": merged.description,
            "alias_appended": alias_appended,
            "shared_sources": shared_sources,
            "dropped_self_loops": dropped_self_loops,
            "repointed_merge_ids": [earlier.id for earlier in absorbed],
        },
    }
    record = EntityMerge(
        surviving_entity_id=survivor.id,
        merged_name=merged.name,
        merged_entity_type=merged.entity_type,
        merged_aliases=list(merged.aliases) if isinstance(merged.aliases, list) else [],
        merged_source_chunk_ids=merged_chunk_ids,
        evidence=evidence,
        method=method,
        model=model,
    )
    db.add(record)
    db.flush()

    # Nothing points at the merged entity any more: its sources, edges and
    # absorbed merge records have all moved, so this delete cascades nothing.
    db.delete(merged)
    db.flush()

    _recompute_entity_confidence(db, survivor.id)
    for relationship_id in sorted(touched_relationship_ids):
        _recompute_relationship_confidence(db, relationship_id)
    db.flush()
    return record.id


def _delete_relationship(
    db: Session, relationship: Relationship, sources: list[RelationshipSource]
) -> None:
    """ORM-delete an edge's source rows, then the edge, never relying on the cascade."""
    for source in sources:
        db.delete(source)
    db.flush()
    db.delete(relationship)
    db.flush()


def unmerge(db: Session, merge_id: int) -> UnmergeResult:
    """Reverse one recorded merge. See the module docstring for exactly what moves back."""
    record = db.get(EntityMerge, merge_id)
    if record is None:
        raise LookupError(f"no entity merge with id {merge_id}")
    survivor = db.get(Entity, record.surviving_entity_id)
    if survivor is None:  # pragma: no cover - the FK cascade removes the record with it
        raise LookupError(f"merge {merge_id} has no surviving entity")
    restore: dict[str, Any] = record.evidence.get("restore", {})

    normalized_name = restore.get("normalized_name") or normalise(record.merged_name)
    clash = db.execute(
        select(Entity.id).where(
            Entity.normalized_name == normalized_name,
            Entity.entity_type == record.merged_entity_type,
        )
    ).first()
    if clash is not None:
        raise ValueError(
            f"cannot unmerge {merge_id}: entity {clash[0]} already holds "
            f"({normalized_name!r}, {record.merged_entity_type!r})"
        )
    restored = Entity(
        name=record.merged_name,
        normalized_name=normalized_name,
        entity_type=record.merged_entity_type,
        description=restore.get("description", ""),
        aliases=list(record.merged_aliases),
    )
    db.add(restored)
    db.flush()

    merged_chunks = set(record.merged_source_chunk_ids)
    shared = {item["chunk_id"]: item for item in restore.get("shared_sources", [])}
    for source in _entity_sources(db, survivor.id):
        if source.chunk_id not in merged_chunks:
            continue
        item = shared.get(source.chunk_id)
        if item is None:
            db.add(
                EntitySource(
                    entity_id=restored.id,
                    chunk_id=source.chunk_id,
                    confidence=source.confidence,
                    extraction_model=source.extraction_model,
                )
            )
            db.delete(source)
        else:
            db.add(
                EntitySource(
                    entity_id=restored.id,
                    chunk_id=source.chunk_id,
                    confidence=item["merged_confidence"],
                    extraction_model=item["merged_extraction_model"],
                )
            )
            source.confidence = item["survivor_confidence"]
            source.extraction_model = item["survivor_extraction_model"]
    db.flush()

    moved: list[int] = []
    shared_edges: list[int] = []
    for relationship in _touching(db, survivor.id):
        chunk_ids = {source.chunk_id for source in _relationship_sources(db, relationship.id)}
        if not chunk_ids & merged_chunks:
            continue
        if chunk_ids <= merged_chunks:
            if relationship.source_entity_id == survivor.id:
                relationship.source_entity_id = restored.id
            if relationship.target_entity_id == survivor.id:
                relationship.target_entity_id = restored.id
            moved.append(relationship.id)
        else:
            shared_edges.append(relationship.id)
    db.flush()

    restored_edges = _restore_self_loops(
        db, restore.get("dropped_self_loops", []), survivor.id, restored.id
    )

    if restore.get("alias_appended") and isinstance(survivor.aliases, list):
        aliases = list(survivor.aliases)
        if record.merged_name in aliases:
            last = len(aliases) - 1 - aliases[::-1].index(record.merged_name)
            del aliases[last]
            survivor.aliases = aliases

    for earlier_id in restore.get("repointed_merge_ids", []):
        earlier = db.get(EntityMerge, earlier_id)
        if earlier is not None and earlier.surviving_entity_id == survivor.id:
            earlier.surviving_entity_id = restored.id

    db.delete(record)
    db.flush()

    _recompute_entity_confidence(db, survivor.id)
    _recompute_entity_confidence(db, restored.id)
    for relationship_id in [*moved, *shared_edges, *restored_edges]:
        _recompute_relationship_confidence(db, relationship_id)
    db.flush()

    return UnmergeResult(
        restored_entity_id=restored.id,
        survivor_entity_id=survivor.id,
        moved_relationship_ids=moved,
        shared_relationship_ids=shared_edges,
        restored_relationship_ids=restored_edges,
    )


def _restore_self_loops(
    db: Session, loops: list[dict[str, Any]], survivor_id: int, restored_id: int
) -> list[int]:
    """Recreate the edges between the two entities the merge dropped as self-loops.

    A source chunk deleted since the merge is not restored; an edge left with
    none is not recreated, since nothing current supports it.
    """
    wanted = {source["chunk_id"] for loop in loops for source in loop["sources"]}
    existing = (
        set(db.execute(select(Chunk.id).where(Chunk.id.in_(wanted))).scalars()) if wanted else set()
    )
    side = {"survivor": survivor_id, "merged": restored_id}
    created: list[int] = []
    for loop in loops:
        sources = [source for source in loop["sources"] if source["chunk_id"] in existing]
        if not sources:
            continue
        relationship = Relationship(
            source_entity_id=side[loop["source"]],
            target_entity_id=side[loop["target"]],
            relation_type=loop["relation_type"],
            description=loop.get("description", ""),
            weight=loop.get("weight", 1.0),
        )
        db.add(relationship)
        db.flush()
        for source in sources:
            db.add(
                RelationshipSource(
                    relationship_id=relationship.id,
                    chunk_id=source["chunk_id"],
                    confidence=source["confidence"],
                    extraction_model=source["extraction_model"],
                )
            )
        created.append(relationship.id)
    db.flush()
    return created
