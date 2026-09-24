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

Every stored entity and edge records the chunk it came from
(``EntitySource`` / ``RelationshipSource``) and the model that produced it, so
a wrong fact can be traced back to the sentence and the model behind it. A
later chunk that re-reports an existing entity, or an existing edge (same
source entity, target entity and relation type), adds a link row for that
chunk rather than a duplicate, and keeps confidence at the maximum the model
has actually reported for it, with the extraction model of that maximum (R9).
Entity identity is the exact match required by the unique constraint,
``(normalise(name), entity_type)`` (R5); anything beyond that, aliasing or
embedding similarity, is Task 5's job, not this one's.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import select
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
    """Counts from one extraction call, small enough to sum across a batch."""

    entities_stored: int = 0
    relationships_stored: int = 0
    entities_discarded: int = 0
    relationships_discarded: int = 0
    contract_violation: bool = False

    def __add__(self, other: ExtractionReport) -> ExtractionReport:
        if not isinstance(other, ExtractionReport):
            return NotImplemented
        return ExtractionReport(
            entities_stored=self.entities_stored + other.entities_stored,
            relationships_stored=self.relationships_stored + other.relationships_stored,
            entities_discarded=self.entities_discarded + other.entities_discarded,
            relationships_discarded=self.relationships_discarded + other.relationships_discarded,
            contract_violation=self.contract_violation or other.contract_violation,
        )


def extract_chunk(
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
                confidence=representative.confidence,
                extraction_model=extraction_model,
            )
            db.add(row)
            db.flush()
            entity_id = row.id
        else:
            entity_id = existing.id
            if existing.confidence is None or representative.confidence > existing.confidence:
                existing.confidence = representative.confidence
                existing.extraction_model = extraction_model

        _link_entity_source(db, entity_id, chunk.id)
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
                confidence=representative.confidence,
                extraction_model=extraction_model,
            )
            db.add(row)
            db.flush()
            relationship_id = row.id
        else:
            relationship_id = existing.id
            if existing.confidence is None or representative.confidence > existing.confidence:
                existing.confidence = representative.confidence
                existing.extraction_model = extraction_model

        _link_relationship_source(db, relationship_id, chunk.id)

    return len(grouped), endpoint_discarded


def _link_entity_source(db: Session, entity_id: int, chunk_id: int) -> None:
    if db.get(EntitySource, (entity_id, chunk_id)) is None:
        db.add(EntitySource(entity_id=entity_id, chunk_id=chunk_id))


def _link_relationship_source(db: Session, relationship_id: int, chunk_id: int) -> None:
    if db.get(RelationshipSource, (relationship_id, chunk_id)) is None:
        db.add(RelationshipSource(relationship_id=relationship_id, chunk_id=chunk_id))
