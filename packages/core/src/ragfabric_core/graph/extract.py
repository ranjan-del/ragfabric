"""Extraction behind a confidence floor.

One LLM call reads a chunk against the extraction contract (``graph.contracts``)
and returns the entities and relationships it found, each carrying the
confidence the model reported. Everything below the configured floor is
discarded rather than stored: without a floor the graph would silently
accumulate the model's guesses as facts, and a hallucinated edge would be
indistinguishable from a real one the moment it was written. Reporting how
many were dropped is what lets an operator see a corpus or a model producing
junk.

An edge whose source or target entity fell below the floor is discarded too,
even when the edge's own confidence clears it: an edge cannot point at a row
that was never stored (R9).

A response that fails the extraction contract (``ContractViolation``) stores
nothing from that chunk. It is reported, not raised, matching every other
contract in this codebase: a malformed model response is an ordinary outcome,
not a bug.

One LLM call per chunk is the dominant cost of this whole strategy, so a
chunk whose text has not changed since its last extraction is skipped
entirely: no LLM call, counted in the report rather than silently dropped.
``Chunk.extraction_hash`` holds the sha256 hex digest of the chunk's text as
of its last successful extraction; ``None`` means never extracted, which is
unambiguously "changed". A chunk whose text *has* changed is not simply
re-extracted on top of the old rows: its previous ``EntitySource`` /
``RelationshipSource`` links are removed first, any ``Relationship`` or
``Entity`` left with no source at all is deleted (an orphaned entity cascades
to any edge still pointing at it, even one another chunk supports), and every
row that survives has its confidence and extraction_model recomputed from
its remaining sources (R20). Without this, a corrected or deleted sentence
would leave its old, now-false entities and edges in the graph forever. That
cleanup and the re-extraction call happen inside one nested transaction
(``Session.begin_nested``, a SAVEPOINT): if the new extraction hits a
contract violation, the cleanup rolls back too, so a bad response never
trades a real, previously-verified fact for nothing. The hash is written
only once a chunk's extraction actually completes; a contract violation
leaves it exactly as it was, so the next pass retries the same chunk instead
of silently treating it as up to date.

Every stored entity and edge records the chunk it came from
(``EntitySource`` / ``RelationshipSource``) and the model that produced it, so
a wrong fact can be traced back to the sentence and the model behind it. Each
link row also carries the confidence and model *that chunk's* extraction
reported (R20); the parent ``Entity`` / ``Relationship`` row's own confidence
and extraction_model are recomputed after every link write as the maximum
over all of its current sources, never a value nobody currently measured
(ADR 0004). A later chunk that re-reports an existing entity, or an existing
edge (same source entity, target entity and relation type), adds a link row
for that chunk rather than a duplicate; the aggregate recomputation is what
implements R9's "keep the maximum reported" rule.
Entity identity is the exact match required by the unique constraint,
``(normalise(name), entity_type)`` (R5); anything beyond that, aliasing or
embedding similarity, is Task 5's job, not this one's.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ragfabric_core.graph.contracts import (
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    RelationType,
    normalise,
    parse_extraction,
)
from ragfabric_core.json_contract import ContractViolation
from ragfabric_core.models.document import Chunk
from ragfabric_core.models.graph import Entity, EntitySource, Relationship, RelationshipSource
from ragfabric_core.providers.base import LLMProvider, Message

# An entity's identity for upsert purposes: the exact key the unique
# constraint enforces (R5).
EntityKey = tuple[str, EntityType]

_ENTITY_TYPES = ", ".join(entity_type.value for entity_type in EntityType)
_RELATION_TYPES = ", ".join(relation_type.value for relation_type in RelationType)

EXTRACTION_SYSTEM = (
    "You are the extraction step of a knowledge graph builder. "
    "You read one chunk of text and report the entities and relationships it "
    "states. You reply with one JSON object and nothing else."
)

# A module constant, not built per call: the enum lists are fixed for the
# process lifetime, and a small model needs the exact values spelled out
# rather than a description of the schema.
EXTRACTION_PROMPT = f"""Text:
{{text}}

Entity types (use exactly one of these, lowercase): {_ENTITY_TYPES}
Relation types (use exactly one of these, uppercase): {_RELATION_TYPES}

Report every entity the text names under "entities": each needs a "name", an
"entity_type" from the list above, an optional "description", and a
"confidence" between 0 and 1 for how sure you are the text actually names it.

Report every relationship the text states under "relationships": each needs a
"source" name, a "target" name, a "relation_type" from the list above, an
optional "description", and a "confidence" between 0 and 1.

Rules the response must follow:
- Every "source" and "target" must be the "name" of an entity also listed
  under "entities" (matched after trimming surrounding whitespace and
  punctuation).
- A name must map to exactly one "entity_type". Never list the same name
  under two different types.
- A relationship's "source" and "target" must be two different entities;
  never report an entity related to itself.
- "confidence" is required on every entity and every relationship, and must
  be a plain JSON number, never a string, never true or false, never omitted.
- Only report what the text actually states. If you are not sure something is
  true, give it a low confidence rather than leaving it out.

Reply with JSON of the form {{{{"entities": [...], "relationships": [...]}}}}.
Both keys are required; an empty list is fine.
"""


@dataclass(frozen=True)
class ExtractionReport:
    """Counts from one extraction call, small enough to sum across a batch.

    The stored counts are distinct upserted keys (an entity's
    ``(normalise(name), entity_type)``, an edge's ``(source_entity_id,
    target_entity_id, relation_type)``), not the raw number of kept items:
    the contract tolerates the same entity or edge listed twice in one
    response, and both collapse into one stored row.
    """

    entities_stored: int = 0
    relationships_stored: int = 0
    entities_discarded: int = 0
    relationships_discarded: int = 0
    contract_violation: bool = False
    chunks_skipped: int = 0

    def __add__(self, other: ExtractionReport) -> ExtractionReport:
        if not isinstance(other, ExtractionReport):
            return NotImplemented
        return ExtractionReport(
            entities_stored=self.entities_stored + other.entities_stored,
            relationships_stored=self.relationships_stored + other.relationships_stored,
            entities_discarded=self.entities_discarded + other.entities_discarded,
            relationships_discarded=self.relationships_discarded + other.relationships_discarded,
            contract_violation=self.contract_violation or other.contract_violation,
            chunks_skipped=self.chunks_skipped + other.chunks_skipped,
        )


def _hash_chunk_text(text: str) -> str:
    """sha256 hex digest of exactly ``chunk.text``, encoded as UTF-8.

    Nothing else about the chunk (page, position, embedding, section) enters
    this hash: text is the only field an extraction call reads, so it is the
    only field whose change should trigger one.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_chunk(
    db: Session,
    chunk: Chunk,
    llm: LLMProvider,
    *,
    floor: float,
    model: str,
) -> ExtractionReport:
    """Extract ``chunk`` unless its text is unchanged since its last extraction.

    Unchanged means ``chunk.extraction_hash`` equals the hash of the current
    text (a ``None`` hash, never extracted, is always "changed"): no LLM call
    is made, and the skip is counted (``chunks_skipped``) rather than silent.

    A changed chunk has its previous contributions removed before
    re-extracting (see module docstring), atomically with the re-extraction
    call: a contract violation rolls both back, leaving the old graph intact
    and the hash untouched so the chunk is retried next time.
    """
    chunk_hash = _hash_chunk_text(chunk.text)
    if chunk.extraction_hash == chunk_hash:
        return ExtractionReport(chunks_skipped=1)

    is_re_extraction = chunk.extraction_hash is not None

    with db.begin_nested() as nested:
        if is_re_extraction:
            _remove_chunk_contributions(db, chunk.id)

        report = _extract_and_store(db, chunk, llm, floor=floor, model=model)

        if report.contract_violation:
            # The cleanup above already ran inside this savepoint; rolling it
            # back here undoes it too, so a bad response never costs the
            # chunk its real, previously-verified contributions. The hash
            # stays whatever it was before this call.
            nested.rollback()
            return report

        chunk.extraction_hash = chunk_hash
        db.flush()

    return report


def _extract_and_store(
    db: Session,
    chunk: Chunk,
    llm: LLMProvider,
    *,
    floor: float,
    model: str,
) -> ExtractionReport:
    """Make one extraction call for ``chunk`` and write what clears the floor.

    Everything below ``floor`` is discarded, never stored (kept iff
    ``confidence >= floor``). A contract violation writes nothing and comes
    back reported on the return value rather than raised.
    """
    completion = llm.complete(
        [
            Message(role="system", content=EXTRACTION_SYSTEM),
            Message(role="user", content=EXTRACTION_PROMPT.format(text=chunk.text)),
        ],
        model=model,
        temperature=0.0,
    )
    parsed = parse_extraction(completion.text)
    if isinstance(parsed, ContractViolation):
        return ExtractionReport(contract_violation=True)

    kept_entities = [entity for entity in parsed.entities if entity.confidence >= floor]
    entities_discarded = len(parsed.entities) - len(kept_entities)

    entity_ids, entities_stored = _upsert_entities(
        db, chunk, kept_entities, extraction_model=completion.model
    )

    # The contract only guarantees a name maps to exactly one type across ALL
    # entities in the response, not just the ones that cleared the floor, so
    # relationship endpoints resolve their type against the full list.
    type_by_name = {normalise(entity.name): entity.entity_type for entity in parsed.entities}

    kept_relationships = [
        relationship for relationship in parsed.relationships if relationship.confidence >= floor
    ]
    relationships_discarded = len(parsed.relationships) - len(kept_relationships)

    relationships_stored, endpoint_discarded = _upsert_relationships(
        db,
        chunk,
        kept_relationships,
        entity_ids=entity_ids,
        type_by_name=type_by_name,
        extraction_model=completion.model,
    )
    relationships_discarded += endpoint_discarded

    return ExtractionReport(
        entities_stored=entities_stored,
        relationships_stored=relationships_stored,
        entities_discarded=entities_discarded,
        relationships_discarded=relationships_discarded,
    )


def extract_chunks(
    db: Session,
    chunks: Iterable[Chunk],
    llm: LLMProvider,
    *,
    floor: float,
    model: str,
) -> ExtractionReport:
    """Run ``extract_chunk`` over several chunks and sum the reports."""
    report = ExtractionReport()
    for chunk in chunks:
        report = report + extract_chunk(db, chunk, llm, floor=floor, model=model)
    return report


def _upsert_entities(
    db: Session,
    chunk: Chunk,
    entities: list[ExtractedEntity],
    *,
    extraction_model: str,
) -> tuple[dict[EntityKey, int], int]:
    """Store or link every kept entity; return (key -> id, count stored).

    Duplicate ``(normalised name, type)`` entries in the same response are
    tolerated by the contract; the one with the highest confidence is the
    representative written to the row.
    """
    grouped: dict[EntityKey, list[ExtractedEntity]] = defaultdict(list)
    for entity in entities:
        grouped[(normalise(entity.name), entity.entity_type)].append(entity)

    entity_ids: dict[EntityKey, int] = {}
    for key, group in grouped.items():
        normalized_name, entity_type = key
        representative = max(group, key=lambda candidate: candidate.confidence)

        existing = db.execute(
            select(Entity).where(
                Entity.normalized_name == normalized_name,
                Entity.entity_type == entity_type.value,
            )
        ).scalar_one_or_none()

        if existing is None:
            row = Entity(
                name=representative.name,
                normalized_name=normalized_name,
                entity_type=entity_type.value,
                description=representative.description,
            )
            db.add(row)
            db.flush()
            entity_id = row.id
        else:
            entity_id = existing.id

        _link_entity_source(
            db,
            entity_id,
            chunk.id,
            confidence=representative.confidence,
            extraction_model=extraction_model,
        )
        _recompute_entity_confidence(db, entity_id)
        entity_ids[key] = entity_id

    return entity_ids, len(grouped)


def _upsert_relationships(
    db: Session,
    chunk: Chunk,
    relationships: list[ExtractedRelationship],
    *,
    entity_ids: dict[EntityKey, int],
    type_by_name: dict[str, EntityType],
    extraction_model: str,
) -> tuple[int, int]:
    """Store or link every relationship whose endpoints both survived the floor.

    Returns (count stored, count discarded for a missing endpoint). A
    relationship whose source or target was discarded below the floor is
    itself discarded here and counted, even though its own confidence
    cleared the floor (R9): an edge cannot point at a row that was never
    stored.
    """
    grouped: dict[tuple[int, int, RelationType], list[ExtractedRelationship]] = defaultdict(list)
    endpoint_discarded = 0

    for relationship in relationships:
        source_id = entity_ids.get(
            (normalise(relationship.source), type_by_name[normalise(relationship.source)])
        )
        target_id = entity_ids.get(
            (normalise(relationship.target), type_by_name[normalise(relationship.target)])
        )
        if source_id is None or target_id is None:
            endpoint_discarded += 1
            continue
        grouped[(source_id, target_id, relationship.relation_type)].append(relationship)

    for (source_id, target_id, relation_type), group in grouped.items():
        representative = max(group, key=lambda candidate: candidate.confidence)

        existing = db.execute(
            select(Relationship).where(
                Relationship.source_entity_id == source_id,
                Relationship.target_entity_id == target_id,
                Relationship.relation_type == relation_type.value,
            )
        ).scalar_one_or_none()

        if existing is None:
            row = Relationship(
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=relation_type.value,
                description=representative.description,
            )
            db.add(row)
            db.flush()
            relationship_id = row.id
        else:
            relationship_id = existing.id

        _link_relationship_source(
            db,
            relationship_id,
            chunk.id,
            confidence=representative.confidence,
            extraction_model=extraction_model,
        )
        _recompute_relationship_confidence(db, relationship_id)

    return len(grouped), endpoint_discarded


def _remove_chunk_contributions(db: Session, chunk_id: int) -> None:
    """Undo everything ``chunk_id`` previously contributed to the graph.

    Called before re-extracting a changed chunk, since its existing
    ``EntitySource`` / ``RelationshipSource`` links describe text that no
    longer exists. Order: drop this chunk's link rows; delete any
    ``Relationship`` left with no source at all; delete any ``Entity`` left
    with no source at all (its FK ``ondelete="CASCADE"`` then removes any
    ``Relationship`` still pointing at it, even one another chunk supports,
    because an edge cannot survive the loss of an endpoint); finally
    recompute confidence/extraction_model on every row that is still there,
    from whatever sources it has left (R20).
    """
    relationship_ids = set(
        db.execute(
            select(RelationshipSource.relationship_id).where(
                RelationshipSource.chunk_id == chunk_id
            )
        ).scalars()
    )
    entity_ids = set(
        db.execute(
            select(EntitySource.entity_id).where(EntitySource.chunk_id == chunk_id)
        ).scalars()
    )

    db.execute(delete(RelationshipSource).where(RelationshipSource.chunk_id == chunk_id))
    db.execute(delete(EntitySource).where(EntitySource.chunk_id == chunk_id))
    db.flush()

    for relationship_id in relationship_ids:
        still_sourced = db.execute(
            select(RelationshipSource.relationship_id).where(
                RelationshipSource.relationship_id == relationship_id
            )
        ).first()
        if still_sourced is None:
            relationship = db.get(Relationship, relationship_id)
            if relationship is not None:
                db.delete(relationship)
    db.flush()

    for entity_id in entity_ids:
        still_sourced = db.execute(
            select(EntitySource.entity_id).where(EntitySource.entity_id == entity_id)
        ).first()
        if still_sourced is None:
            entity = db.get(Entity, entity_id)
            if entity is not None:
                db.delete(entity)  # cascades to any relationship still pointing at it
    db.flush()

    for relationship_id in relationship_ids:
        if db.get(Relationship, relationship_id) is not None:
            _recompute_relationship_confidence(db, relationship_id)
    for entity_id in entity_ids:
        if db.get(Entity, entity_id) is not None:
            _recompute_entity_confidence(db, entity_id)
    db.flush()


def _link_entity_source(
    db: Session, entity_id: int, chunk_id: int, *, confidence: float, extraction_model: str
) -> None:
    """Add or refresh the (entity, chunk) provenance row with what this chunk reported."""
    link = db.get(EntitySource, (entity_id, chunk_id))
    if link is None:
        db.add(
            EntitySource(
                entity_id=entity_id,
                chunk_id=chunk_id,
                confidence=confidence,
                extraction_model=extraction_model,
            )
        )
    else:
        link.confidence = confidence
        link.extraction_model = extraction_model


def _link_relationship_source(
    db: Session, relationship_id: int, chunk_id: int, *, confidence: float, extraction_model: str
) -> None:
    """Add or refresh the (relationship, chunk) provenance row. See ``_link_entity_source``."""
    link = db.get(RelationshipSource, (relationship_id, chunk_id))
    if link is None:
        db.add(
            RelationshipSource(
                relationship_id=relationship_id,
                chunk_id=chunk_id,
                confidence=confidence,
                extraction_model=extraction_model,
            )
        )
    else:
        link.confidence = confidence
        link.extraction_model = extraction_model


def _recompute_entity_confidence(db: Session, entity_id: int) -> None:
    """Set ``Entity.confidence``/``extraction_model`` to the max its current sources report.

    Recomputed from ``EntitySource`` rows rather than compared incrementally
    against the old value, so that removing a source (Task 4 cleanup) or
    moving one (Task 5 merge/unmerge) cannot leave a number behind that no
    surviving source actually reported (R20). ``None`` iff no surviving
    source has a measured confidence (ADR 0004: never fabricate a number).
    """
    entity = db.get(Entity, entity_id)
    if entity is None:
        return
    sources = db.execute(select(EntitySource).where(EntitySource.entity_id == entity_id)).scalars()
    measured = [source for source in sources if source.confidence is not None]
    if not measured:
        entity.confidence = None
        entity.extraction_model = None
        return
    best = max(measured, key=lambda source: source.confidence)
    entity.confidence = best.confidence
    entity.extraction_model = best.extraction_model


def _recompute_relationship_confidence(db: Session, relationship_id: int) -> None:
    """Relationship counterpart of ``_recompute_entity_confidence``."""
    relationship = db.get(Relationship, relationship_id)
    if relationship is None:
        return
    sources = db.execute(
        select(RelationshipSource).where(RelationshipSource.relationship_id == relationship_id)
    ).scalars()
    measured = [source for source in sources if source.confidence is not None]
    if not measured:
        relationship.confidence = None
        relationship.extraction_model = None
        return
    best = max(measured, key=lambda source: source.confidence)
    relationship.confidence = best.confidence
    relationship.extraction_model = best.extraction_model
