"""Relational mirror of the knowledge graph, for the console and audit. PostgreSQL answers queries.

``Entity`` and ``Relationship`` carry an extraction ``confidence``. It is
nullable on purpose (ADR 0004): a row written before Phase 6, or one an
extractor never scored, has no confidence, and ``None`` says "not measured"
where ``0.0`` would say "measured as worthless". When a confidence is
present it must fall inside [0, 1]; that range is enforced by a database
``CheckConstraint`` rather than trusted to callers, so it holds regardless of
which code path writes the row.

``EntitySource`` and ``RelationshipSource`` are the single source of truth
for provenance: which chunks an entity or relationship was extracted from.
They replace the JSON ``source_chunk_ids`` columns migration 0008 drops, so
that the access filter a traversal query applies can be a join inside the
recursive term rather than a second, driftable copy of the same list. Each
link row also carries the confidence and extraction model that one chunk's
extraction call reported (R20), so the parent row's own confidence can be
recomputed as the max over its current sources whenever a source is added,
removed or moved, rather than drifting once the chunk that supplied the
number is re-extracted or merged away. An ``EntitySource`` row also records
the ``surface_name`` that chunk used for the entity (R40), because the name a
caller sees must come from a chunk that caller may read.

``EntityMerge`` records every entity resolution decision (Task 5): the
surviving entity, the merged-away entity's name, type, aliases and source
chunk ids (kept so an unmerge can restore it), the evidence that justified
the merge, and the method and model that made the call. A merge without its
evidence cannot be inspected or trusted, so ``evidence`` is required.

``entities``, ``relationships`` and ``entity_merges`` never reuse an id
(``sqlite_autoincrement``; PostgreSQL sequences already do not). A merge
record names edges and entities by id, and SQLite's default rowid would hand
a deleted row's id to the next insert, so a record could silently name a
different row (R31).
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base, utcnow

_CONFIDENCE_RANGE = "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)"


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("normalized_name", "entity_type", name="uq_entity_name_type"),
        CheckConstraint(_CONFIDENCE_RANGE, name="ck_entity_confidence_range"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    normalized_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    aliases: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_model: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class Relationship(Base):
    __tablename__ = "relationships"
    __table_args__ = (
        CheckConstraint(_CONFIDENCE_RANGE, name="ck_relationship_confidence_range"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relation_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_model: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class EntitySource(Base):
    """One row per (entity, chunk) the entity was extracted from.

    The composite primary key is the provenance record itself: there is
    nothing to update, only rows to add or cascade-delete. Indexed on
    ``chunk_id`` because the access filter walks from a visible chunk set to
    the entities and relationships it justifies, not the other way round.

    ``confidence`` and ``extraction_model`` (R20) are what *that chunk's*
    extraction call reported for this entity, nullable for the same ADR 0004
    reason as the parent row. They are what ``Entity.confidence`` /
    ``Entity.extraction_model`` are recomputed from (the max over every
    surviving source) whenever a source is added, removed (Task 4 cleanup) or
    moved (Task 5 merge/unmerge), so the parent row never carries a number no
    current source supports.

    ``surface_name`` (R40) is the spelling *that chunk's* extraction reported
    for the entity. A name is text from a chunk, exactly like a description,
    so what a caller sees is the spelling of the best source they may read
    (``GraphNode.name``), and a question matches only admitted sources'
    spellings. ``Entity.name`` and ``Entity.aliases`` are resolution's own
    bookkeeping and never reach a caller. It moves with the row on merge and
    unmerge, and is required: every source row was written by an extraction
    that named the entity.
    """

    __tablename__ = "entity_sources"
    __table_args__ = (
        CheckConstraint(_CONFIDENCE_RANGE, name="ck_entity_sources_confidence_range"),
    )

    entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_model: Mapped[str | None] = mapped_column(String, nullable=True)
    surface_name: Mapped[str] = mapped_column(String, nullable=False)


class RelationshipSource(Base):
    """One row per (relationship, chunk) the relationship was extracted from. See ``EntitySource``."""

    __tablename__ = "relationship_sources"
    __table_args__ = (
        CheckConstraint(_CONFIDENCE_RANGE, name="ck_relationship_sources_confidence_range"),
    )

    relationship_id: Mapped[int] = mapped_column(
        ForeignKey("relationships.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_model: Mapped[str | None] = mapped_column(String, nullable=True)


class EntityMerge(Base):
    __tablename__ = "entity_merges"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    surviving_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    merged_name: Mapped[str] = mapped_column(String, nullable=False)
    merged_entity_type: Mapped[str] = mapped_column(String, nullable=False)
    merged_aliases: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    merged_source_chunk_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    evidence: Mapped[dict] = mapped_column(JSON, nullable=False)
    method: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
