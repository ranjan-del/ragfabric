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
   similar down, and each entity takes part in at most one stage-3 merge per
   pass (R27): a pair touching an entity already merged in this stage, as
   survivor or as merged, is skipped. So every stage-3 merge joins exactly
   two entities whose own similarity met the threshold, and no pass clusters
   transitively; a legitimate larger cluster needs another pass.

Stages 1 and 2 repeat until no pair is left, because a merge adds the merged
name and aliases to the survivor's aliases and that can create a new match.

**Survivor choice** is deterministic: the entity with a measured confidence
beats one without; then the higher confidence; then more ``EntitySource``
rows; then the lower id.

**A merge** (R24, R26, R27) moves the merged-away entity's ``EntitySource`` rows onto
the survivor with their per-chunk confidence (a chunk that sourced both keeps
the higher of the two reports on the survivor's row), repoints its
relationships to the survivor, folds any edge that now duplicates another
(same source, target and relation type) by moving its ``RelationshipSource``
rows, drops any edge that would become a self-loop, adds the merged name and
the merged entity's own aliases to the survivor's aliases (those not already
there, R27), hands any merge records
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
  merged entity's stored ``normalized_name`` and ``description``, the
  ``aliases_appended`` to the survivor, the ``shared_sources`` (chunks that
  sourced both entities, with both confidences), the ``relationships`` the
  merge moved (R26: each ``repointed`` or ``folded``, with which endpoints
  were the merged entity's, its per-chunk reports, and for a fold the kept
  edge's own reports on the chunks both carried, before the fold overwrote
  them), the ``dropped_self_loops`` (with their reports) and the
  ``repointed_merge_ids``.

**An unmerge** recreates the entity (under a new id) with its name, type,
aliases and description, and moves back exactly the ``EntitySource`` rows for
``merged_source_chunk_ids`` (a shared chunk gets both original reports back).
Only the relationships the merge recorded are candidates (R26); the
survivor's own edges never are. A repointed edge points back at the restored
entity; a folded edge is recreated on it with its recorded reports, and the
survivor edge it was folded into gets back exactly its own reports. The
dropped self-loops whose chunks still exist are recreated. The aliases the
merge added are removed from the survivor, the merge records it had absorbed
are handed back, confidences are recomputed and the ``EntityMerge`` row is
deleted. With no change in between, merge then unmerge gives back the same
graph apart from the restored entity's and recreated edges' ids. Where the
graph did change since (a chunk re-extracted or deleted, an edge gone), the
current graph wins: nothing is brought back that no current row supports.
**Unmerge is globally last-in, first-out** (R30): only the most recent merge
still in place can be undone. ``unmerge`` raises ``UnmergeBlocked``, before
writing anything, while any live merge has a higher id, and lists every one of
them; undo those first, newest first, then retry. This blocks even merges of
completely unrelated entities, and that is the intended cost. Rules scoped to
the entities a merge touched were tried and were bypassable: an unmerge
recreates an entity under a new id, so a record of "which entities this merge
touched" goes stale, and a later merge on the recreated entity escaped the
check. The id order is a total order that no unmerge can change: a new merge
row always gets an id above every live one (a PostgreSQL sequence; SQLite
takes the current maximum plus one). Undone in that order, every merge gives
back the graph it started from. Recorded edges that nothing current supports any more are
listed in ``UnmergeResult.unrestored_relationship_ids`` rather than dropped
silently.

Public API (called by ingestion in Task 11 and the CLI in Task 12)::

    resolve_entities(db, embedder, *, similarity_threshold, entity_ids=None,
                     batch_size=64) -> ResolutionReport
    unmerge(db, merge_id) -> UnmergeResult     # raises UnmergeBlocked, LookupError

``embedder`` may be ``None``, which skips stage 3 (0 embedding calls).
``entity_ids`` restricts resolution to pairs that involve at least one listed
entity (a survivor of such a merge joins the set); the rest of the graph is
still the pool they are compared against. Both functions flush; the caller
commits.
"""

from __future__ import annotations

import copy
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


class UnmergeBlocked(Exception):
    """An unmerge refused because newer merges are still in place (R30).

    Unmerge is globally last-in, first-out: ``blocking_merge_ids`` are every
    live merge with a higher id, oldest first. Undo them newest first, then
    retry. Raised before anything is written.
    """

    def __init__(self, merge_id: int, blocking_merge_ids: list[int]) -> None:
        super().__init__(
            f"cannot unmerge {merge_id} while later merges {blocking_merge_ids} are in "
            "place; unmerge those first, newest first"
        )
        self.merge_id = merge_id
        self.blocking_merge_ids = blocking_merge_ids


@dataclass(frozen=True)
class UnmergeResult:
    """What an unmerge did.

    ``moved_relationship_ids`` are the repointed edges now pointing at the
    restored entity again. ``shared_relationship_ids`` are the survivor's
    edges that had absorbed one of the merged entity's edges (a fold); they
    stay with the survivor, now carrying only its own reports, plus any edge
    split off to hold reports a later extraction added to a repointed edge.
    ``restored_relationship_ids`` are edges recreated from the record:
    folded-away edges on the restored entity, and the self-loops the merge
    had dropped. ``unrestored_relationship_ids`` are the original ids of
    recorded edges that could not be restored because nothing current
    supports them any more (the edge, or every chunk it came from, was
    removed since, for example by re-extraction); nothing is recreated for
    them, and they are listed so the loss is never silent.
    """

    restored_entity_id: int
    survivor_entity_id: int
    moved_relationship_ids: list[int]
    shared_relationship_ids: list[int]
    restored_relationship_ids: list[int]
    unrestored_relationship_ids: list[int]


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

        # R27: one stage-3 merge per entity per pass. A pair touching an
        # entity already merged in this stage (on either side) is skipped, so
        # two entities whose own similarity never reached the threshold are
        # never joined through a third.
        involved: set[int] = set()
        for similarity, a, b in pairs:
            if a.id in involved or b.id in involved:
                continue
            involved.update((a.id, b.id))
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
    moved_relationships: list[dict[str, Any]] = []
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
                    "original_id": relationship.id,
                    "source": "survivor"
                    if relationship.source_entity_id == survivor.id
                    else "merged",
                    "target": "survivor"
                    if relationship.target_entity_id == survivor.id
                    else "merged",
                    "relation_type": relationship.relation_type,
                    "description": relationship.description,
                    "weight": relationship.weight,
                    "sources": [_report(source) for source in sources],
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
        entry: dict[str, Any] = {
            "original_id": relationship.id,
            "merged_source": relationship.source_entity_id == merged.id,
            "merged_target": relationship.target_entity_id == merged.id,
            "relation_type": relationship.relation_type,
            "description": relationship.description,
            "weight": relationship.weight,
            "sources": [_report(source) for source in sources],
        }
        if duplicate is None:
            relationship.source_entity_id = new_source
            relationship.target_entity_id = new_target
            touched_relationship_ids.add(relationship.id)
            moved_relationships.append(
                {"kind": "repointed", "relationship_id": relationship.id, **entry}
            )
            continue

        kept = {source.chunk_id: source for source in _relationship_sources(db, duplicate.id)}
        # The kept edge's own reports on the chunks both edges came from,
        # recorded before the fold may overwrite them.
        entry["kept_sources"] = [
            _report(kept[source.chunk_id]) for source in sources if source.chunk_id in kept
        ]
        moved_relationships.append({"kind": "folded", "relationship_id": duplicate.id, **entry})
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

    # The merged name and the merged entity's own aliases stay matchable on
    # the survivor (R27); exactly what was added is recorded for the unmerge.
    # The merged name is kept as its own surface spelling even when it
    # normalises to the survivor's name (an exact merge); one of the merged
    # entity's aliases is only worth adding when it matches nothing the
    # survivor already answers to.
    aliases_appended: list[str] = []
    if isinstance(survivor.aliases, list):
        if merged.name not in survivor.aliases:
            aliases_appended.append(merged.name)
        answers_to = _name_keys(survivor) | {normalise(alias) for alias in _aliases(survivor)}
        answers_to |= {normalise(alias) for alias in aliases_appended}
        for alias in _aliases(merged):
            key = normalise(alias)
            if key and key not in answers_to:
                aliases_appended.append(alias)
                answers_to.add(key)
        if aliases_appended:
            survivor.aliases = [*survivor.aliases, *aliases_appended]

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
            "aliases_appended": aliases_appended,
            "shared_sources": shared_sources,
            "relationships": moved_relationships,
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


def _report(source: RelationshipSource) -> dict[str, Any]:
    """One chunk's report on an edge, as recorded for an unmerge."""
    return {
        "chunk_id": source.chunk_id,
        "confidence": source.confidence,
        "extraction_model": source.extraction_model,
    }


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

    blocking = _later_merges(db, record.id)
    if blocking:
        raise UnmergeBlocked(merge_id, blocking)

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
    restored_edges: list[int] = []
    unrestored: list[int] = []
    # R26: only the relationships this merge moved are candidates; the
    # survivor's own edges never are.
    for entry in restore.get("relationships", []):
        if entry["kind"] == "repointed":
            outcome = _restore_repointed(db, entry, survivor.id, restored.id)
            if outcome is None:
                unrestored.append(entry["original_id"])
            else:
                moved.append(outcome[0])
                shared_edges.extend(outcome[1:])
        else:
            outcome = _restore_folded(db, entry, survivor.id, restored.id)
            if outcome is None:
                unrestored.append(entry["original_id"])
            else:
                if outcome[0] is not None:
                    shared_edges.append(outcome[0])
                restored_edges.append(outcome[1])
                _retarget_records(db, record.id, entry["original_id"], outcome[1])
    db.flush()

    loops_created, loops_lost = _restore_self_loops(
        db, restore.get("dropped_self_loops", []), survivor.id, restored.id
    )
    for original_id, new_id in loops_created:
        _retarget_records(db, record.id, original_id, new_id)
    loops_created = [new_id for _, new_id in loops_created]
    restored_edges += loops_created
    unrestored += loops_lost

    appended = restore.get("aliases_appended", [])
    if isinstance(survivor.aliases, list):
        survivor.aliases = _without(survivor.aliases, appended)

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
        unrestored_relationship_ids=unrestored,
    )


def _later_merges(db: Session, merge_id: int) -> list[int]:
    """Every live merge newer than ``merge_id``, oldest first (R30)."""
    return list(
        db.execute(
            select(EntityMerge.id).where(EntityMerge.id > merge_id).order_by(EntityMerge.id)
        ).scalars()
    )


def _without(values: Any, removed: list[str]) -> list:
    """``values`` with the last occurrence of each of ``removed`` taken out."""
    result = list(values) if isinstance(values, list) else []
    for value in removed:
        if value in result:
            del result[len(result) - 1 - result[::-1].index(value)]
    return result


def _other_side(relationship: Relationship, entry: dict[str, Any], survivor_id: int) -> bool:
    """Whether the edge's merged-side endpoints still point at the survivor."""
    return (not entry["merged_source"] or relationship.source_entity_id == survivor_id) and (
        not entry["merged_target"] or relationship.target_entity_id == survivor_id
    )


def _restore_repointed(
    db: Session, entry: dict[str, Any], survivor_id: int, restored_id: int
) -> tuple[int, ...] | None:
    """Point a repointed edge's merged-side endpoints back at the restored entity.

    Returns ``(moved edge id, *split-off edge ids)``, or ``None`` when the edge
    no longer exists or no longer points at the survivor (it was deleted or
    moved since, and the current graph wins). A source row for a chunk the
    record does not list was added after the merge by a later extraction that
    found the edge under the survivor's name (a later merge would have
    blocked this unmerge, R30). It is a claim about the
    survivor, so it is split off onto a survivor-side edge rather than handed
    to the restored entity.
    """
    relationship = db.get(Relationship, entry["relationship_id"])
    if relationship is None or not _other_side(relationship, entry, survivor_id):
        return None
    recorded = {source["chunk_id"] for source in entry["sources"]}
    foreign = [
        source
        for source in _relationship_sources(db, relationship.id)
        if source.chunk_id not in recorded
    ]
    split: list[int] = []
    if foreign:
        survivor_side = Relationship(
            source_entity_id=relationship.source_entity_id,
            target_entity_id=relationship.target_entity_id,
            relation_type=relationship.relation_type,
            description=relationship.description,
            weight=relationship.weight,
        )
        db.add(survivor_side)
        db.flush()
        for source in foreign:
            db.add(
                RelationshipSource(
                    relationship_id=survivor_side.id,
                    chunk_id=source.chunk_id,
                    confidence=source.confidence,
                    extraction_model=source.extraction_model,
                )
            )
            db.delete(source)
        db.flush()
        split.append(survivor_side.id)
    if entry["merged_source"]:
        relationship.source_entity_id = restored_id
    if entry["merged_target"]:
        relationship.target_entity_id = restored_id
    db.flush()
    return (relationship.id, *split)


def _retarget_records(db: Session, record_id: int, old_id: int, new_id: int) -> None:
    """Point earlier merges' recorded edges at an edge this unmerge recreated.

    Needed in plain last-in, first-out order: an older merge may have
    repointed an edge (or folded one into it) that a newer merge then folded
    away or dropped as a self-loop. Undoing the newer merge recreates that
    edge under a new id, and the older merge's record must follow it, or its
    own unmerge would find nothing to restore.
    """
    for other in db.execute(select(EntityMerge).where(EntityMerge.id != record_id)).scalars():
        # Copied, never edited in place: an in-place edit would also change
        # the value SQLAlchemy compares against, and the update would be lost.
        restore = copy.deepcopy(other.evidence.get("restore", {}))
        entries = restore.get("relationships", [])
        changed = False
        for entry in entries:
            if entry["relationship_id"] == old_id:
                entry["relationship_id"] = new_id
                changed = True
        if changed:
            other.evidence = {**other.evidence, "restore": {**restore, "relationships": entries}}


def _restore_folded(
    db: Session, entry: dict[str, Any], survivor_id: int, restored_id: int
) -> tuple[int | None, int] | None:
    """Split a folded edge back out of the survivor's edge it was folded into.

    The survivor's edge gets back exactly its own reports (the recorded
    ``kept_sources``; rows for chunks only the folded edge had are removed),
    and the folded edge is recreated on the restored entity with its recorded
    reports, for every chunk the survivor's edge still carries (a chunk
    re-extracted or deleted since is not brought back). Returns ``(kept edge
    id, recreated edge id)``, or ``None`` when the kept edge is gone or no
    longer points at the survivor.
    """
    kept = db.get(Relationship, entry["relationship_id"])
    if kept is None or not _other_side(kept, entry, survivor_id):
        return None
    current = {source.chunk_id: source for source in _relationship_sources(db, kept.id)}
    own = {source["chunk_id"]: source for source in entry["kept_sources"]}
    carried = [source for source in entry["sources"] if source["chunk_id"] in current]
    if not carried:
        return None

    recreated = Relationship(
        source_entity_id=restored_id if entry["merged_source"] else kept.source_entity_id,
        target_entity_id=restored_id if entry["merged_target"] else kept.target_entity_id,
        relation_type=entry["relation_type"],
        description=entry.get("description", ""),
        weight=entry.get("weight", 1.0),
    )
    db.add(recreated)
    db.flush()
    for source in carried:
        db.add(
            RelationshipSource(
                relationship_id=recreated.id,
                chunk_id=source["chunk_id"],
                confidence=source["confidence"],
                extraction_model=source["extraction_model"],
            )
        )
        row = current[source["chunk_id"]]
        original = own.get(source["chunk_id"])
        if original is None:
            db.delete(row)
        else:
            row.confidence = original["confidence"]
            row.extraction_model = original["extraction_model"]
    db.flush()
    if not _relationship_sources(db, kept.id):
        # Its own sources were removed since the merge: nothing supports it.
        db.delete(kept)
        db.flush()
        return (None, recreated.id)
    return (kept.id, recreated.id)


def _restore_self_loops(
    db: Session, loops: list[dict[str, Any]], survivor_id: int, restored_id: int
) -> tuple[list[tuple[int, int]], list[int]]:
    """Recreate the edges between the two entities the merge dropped as self-loops.

    A source chunk deleted since the merge is not restored; an edge left with
    none is not recreated, since nothing current supports it. Returns
    ``([(original id, recreated id)], original ids of the loops not recreated)``.
    """
    wanted = {source["chunk_id"] for loop in loops for source in loop["sources"]}
    existing = (
        set(db.execute(select(Chunk.id).where(Chunk.id.in_(wanted))).scalars()) if wanted else set()
    )
    side = {"survivor": survivor_id, "merged": restored_id}
    created: list[tuple[int, int]] = []
    lost: list[int] = []
    for loop in loops:
        sources = [source for source in loop["sources"] if source["chunk_id"] in existing]
        if not sources:
            lost.append(loop["original_id"])
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
        created.append((loop["original_id"], relationship.id))
    db.flush()
    return created, lost
