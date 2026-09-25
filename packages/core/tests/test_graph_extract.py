"""Tests for extraction behind a confidence floor.

A single LLM call reads one chunk and returns entities and relationships, each
carrying the confidence the model reported. Anything below the configured
floor is discarded, never stored, and the discard count is reported so an
operator can see when a corpus or a model is producing junk. Every item that
does survive records its source chunk and the model that produced it, because
a wrong edge has to be traceable to the sentence and the model behind it.
"""

import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from ragfabric_core.graph.extract import extract_chunk, extract_chunks
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import Entity, EntitySource, Relationship, RelationshipSource
from ragfabric_core.providers.base import Completion
from ragfabric_core.providers.offline import ScriptedLLMProvider

FLOOR = 0.5


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'extract.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        yield session


def _make_chunk(db, text: str = "Ada Lovelace works on Analytical Engine.") -> Chunk:
    collection = Collection(name="c")
    db.add(collection)
    db.flush()
    document = Document(filename="a.txt", format="txt", collection_id=collection.id, status="ready")
    db.add(document)
    db.flush()
    chunk = Chunk(
        document_id=document.id,
        collection_id=collection.id,
        chunk_index=0,
        text=text,
        embedding=[0.0],
    )
    db.add(chunk)
    db.flush()
    return chunk


def _provider(payload: dict) -> ScriptedLLMProvider:
    return ScriptedLLMProvider(responses=[json.dumps(payload)], model="test-extractor")


class _RelabelingProvider:
    """A double for a provider that serves a different model than requested.

    A real provider can do this (a router, a fallback after a rate limit), so
    the writer has to record what the completion says was actually used, not
    what the caller asked for.
    """

    name = "relabeling"
    default_model = "requested-model"

    def __init__(self, response: str, actual_model: str) -> None:
        self._response = response
        self._actual_model = actual_model

    def complete(
        self,
        messages,
        *,
        model=None,
        max_tokens=1024,
        temperature=0.0,
        json_schema=None,
    ) -> Completion:
        return Completion(
            text=self._response,
            model=self._actual_model,
            provider=self.name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )


def test_an_edge_below_the_floor_is_not_stored(db):
    chunk = _make_chunk(db)
    payload = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": 0.9},
        ],
        "relationships": [
            {
                "source": "Ada Lovelace",
                "target": "Analytical Engine",
                "relation_type": "WORKS_ON",
                "confidence": 0.3,
            }
        ],
    }
    report = extract_chunk(db, chunk, _provider(payload), floor=FLOOR, model="test-extractor")

    assert report.relationships_stored == 0
    assert report.relationships_discarded == 1
    assert db.execute(select(Relationship)).scalars().all() == []


def test_an_edge_whose_endpoint_fell_below_the_floor_is_discarded_and_counted(db):
    """R9: an edge cannot point at a row that was never stored.

    The relationship's own confidence clears the floor here; its target
    entity does not. The edge must be discarded and counted anyway, and the
    endpoint that did clear the floor must still be stored normally.
    """
    chunk = _make_chunk(db)
    payload = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": 0.2},
        ],
        "relationships": [
            {
                "source": "Ada Lovelace",
                "target": "Analytical Engine",
                "relation_type": "WORKS_ON",
                "confidence": 0.9,
            }
        ],
    }
    report = extract_chunk(db, chunk, _provider(payload), floor=FLOOR, model="test-extractor")

    assert report.relationships_stored == 0
    assert report.relationships_discarded == 1
    assert db.execute(select(Relationship)).scalars().all() == []

    assert report.entities_stored == 1
    assert report.entities_discarded == 1
    ada = db.execute(select(Entity).where(Entity.normalized_name == "ada lovelace")).scalar_one()
    assert ada.confidence == 0.9


def test_extraction_model_records_what_the_completion_reports(db):
    """completion.model, not the requested model, is what gets stored.

    A router or a fallback can serve a different model than the one asked
    for; the row has to be traceable to the model that actually answered.
    """
    chunk = _make_chunk(db)
    payload = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    provider = _RelabelingProvider(json.dumps(payload), actual_model="actually-served-model")

    extract_chunk(db, chunk, provider, floor=FLOOR, model="requested-model")

    ada = db.execute(select(Entity).where(Entity.normalized_name == "ada lovelace")).scalar_one()
    assert ada.extraction_model == "actually-served-model"


def test_the_number_discarded_is_reported(db):
    chunk = _make_chunk(db)
    payload = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
            {"name": "Rumour Mill", "entity_type": "organisation", "confidence": 0.1},
        ],
        "relationships": [],
    }
    report = extract_chunk(db, chunk, _provider(payload), floor=FLOOR, model="test-extractor")

    assert report.entities_stored == 1
    assert report.entities_discarded == 1
    assert report.contract_violation is False


def test_every_stored_item_records_its_source_chunk_and_model(db):
    chunk = _make_chunk(db)
    payload = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": 0.8},
        ],
        "relationships": [
            {
                "source": "Ada Lovelace",
                "target": "Analytical Engine",
                "relation_type": "WORKS_ON",
                "confidence": 0.7,
            }
        ],
    }
    report = extract_chunk(db, chunk, _provider(payload), floor=FLOOR, model="test-extractor")

    assert report.entities_stored == 2
    assert report.relationships_stored == 1

    ada = db.execute(select(Entity).where(Entity.normalized_name == "ada lovelace")).scalar_one()
    assert ada.confidence == 0.9
    assert ada.extraction_model == "test-extractor"
    entity_source = db.get(EntitySource, (ada.id, chunk.id))
    assert entity_source is not None

    relationship = db.execute(select(Relationship)).scalar_one()
    assert relationship.confidence == 0.7
    assert relationship.extraction_model == "test-extractor"
    relationship_source = db.get(RelationshipSource, (relationship.id, chunk.id))
    assert relationship_source is not None


def test_a_contract_violation_stores_nothing_from_that_chunk(db):
    chunk = _make_chunk(db)
    provider = ScriptedLLMProvider(responses=["not a json object at all"], model="test-extractor")

    report = extract_chunk(db, chunk, provider, floor=FLOOR, model="test-extractor")

    assert report.contract_violation is True
    assert report.entities_stored == 0
    assert report.relationships_stored == 0
    assert report.entities_discarded == 0
    assert report.relationships_discarded == 0
    assert db.execute(select(Entity)).scalars().all() == []
    assert db.execute(select(Relationship)).scalars().all() == []


def test_confidence_exactly_at_the_floor_is_kept(db):
    chunk = _make_chunk(db)
    payload = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": FLOOR}],
        "relationships": [],
    }
    report = extract_chunk(db, chunk, _provider(payload), floor=FLOOR, model="test-extractor")

    assert report.entities_stored == 1
    assert report.entities_discarded == 0


def test_re_reporting_an_existing_entity_adds_a_source_link_not_a_duplicate_row(db):
    chunk_one = _make_chunk(db, text="Ada Lovelace is a mathematician.")
    chunk_two = _make_chunk(db, text="Ada Lovelace worked with Babbage.")
    payload = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.6}],
        "relationships": [],
    }
    extract_chunk(db, chunk_one, _provider(payload), floor=FLOOR, model="model-a")

    higher_payload = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.95}],
        "relationships": [],
    }
    report_two = extract_chunk(
        db, chunk_two, _provider(higher_payload), floor=FLOOR, model="model-b"
    )

    assert report_two.entities_stored == 1
    entities = (
        db.execute(select(Entity).where(Entity.normalized_name == "ada lovelace")).scalars().all()
    )
    assert len(entities) == 1
    assert entities[0].confidence == 0.95
    assert entities[0].extraction_model == "model-b"
    assert db.get(EntitySource, (entities[0].id, chunk_one.id)) is not None
    assert db.get(EntitySource, (entities[0].id, chunk_two.id)) is not None


def test_the_entity_link_row_records_the_confidence_that_chunk_reported(db):
    """R20: EntitySource carries the confidence and model that one chunk's

    extraction reported, distinct from the entity's own aggregate confidence
    (the max across every chunk that reports it).
    """
    chunk_one = _make_chunk(db, text="Ada Lovelace is a mathematician.")
    chunk_two = _make_chunk(db, text="Ada Lovelace worked with Babbage.")
    low_payload = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.6}],
        "relationships": [],
    }
    high_payload = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.95}],
        "relationships": [],
    }

    extract_chunk(db, chunk_one, _provider(low_payload), floor=FLOOR, model="model-a")
    extract_chunk(db, chunk_two, _provider(high_payload), floor=FLOOR, model="model-b")

    ada = db.execute(select(Entity).where(Entity.normalized_name == "ada lovelace")).scalar_one()
    assert ada.confidence == 0.95
    assert ada.extraction_model == "model-b"

    link_one = db.get(EntitySource, (ada.id, chunk_one.id))
    link_two = db.get(EntitySource, (ada.id, chunk_two.id))
    assert link_one.confidence == 0.6
    assert link_one.extraction_model == "model-a"
    assert link_two.confidence == 0.95
    assert link_two.extraction_model == "model-b"


def test_the_relationship_link_row_records_the_confidence_that_chunk_reported(db):
    """R20, relationship side: same rule as the entity link row."""
    chunk_one = _make_chunk(db, text="Ada Lovelace works on Analytical Engine, barely.")
    chunk_two = _make_chunk(db, text="Ada Lovelace clearly works on Analytical Engine.")
    low_payload = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": 0.9},
        ],
        "relationships": [
            {
                "source": "Ada Lovelace",
                "target": "Analytical Engine",
                "relation_type": "WORKS_ON",
                "confidence": 0.55,
            }
        ],
    }
    high_payload = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": 0.9},
        ],
        "relationships": [
            {
                "source": "Ada Lovelace",
                "target": "Analytical Engine",
                "relation_type": "WORKS_ON",
                "confidence": 0.9,
            }
        ],
    }

    extract_chunk(db, chunk_one, _provider(low_payload), floor=FLOOR, model="model-a")
    extract_chunk(db, chunk_two, _provider(high_payload), floor=FLOOR, model="model-b")

    relationship = db.execute(select(Relationship)).scalar_one()
    assert relationship.confidence == 0.9
    assert relationship.extraction_model == "model-b"

    link_one = db.get(RelationshipSource, (relationship.id, chunk_one.id))
    link_two = db.get(RelationshipSource, (relationship.id, chunk_two.id))
    assert link_one.confidence == 0.55
    assert link_one.extraction_model == "model-a"
    assert link_two.confidence == 0.9
    assert link_two.extraction_model == "model-b"


def test_extract_chunks_sums_reports(db):
    chunk_one = _make_chunk(db, text="Ada Lovelace is a mathematician.")
    chunk_two = _make_chunk(db, text="Charles Babbage designed engines.")
    payload_one = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    payload_two = {
        "entities": [{"name": "Charles Babbage", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    provider = ScriptedLLMProvider(
        responses=[json.dumps(payload_one), json.dumps(payload_two)], model="test-extractor"
    )

    report = extract_chunks(
        db, [chunk_one, chunk_two], provider, floor=FLOOR, model="test-extractor"
    )

    assert report.entities_stored == 2
    assert report.contract_violation is False


def test_a_disabled_entity_type_is_left_out_of_the_prompt_discarded_and_counted(db):
    from ragfabric_core.graph.contracts import EntityType
    from ragfabric_core.graph.extract import build_extraction_prompt

    assert "product" not in build_extraction_prompt([EntityType.PERSON])
    assert "product" in build_extraction_prompt()

    chunk = _make_chunk(db)
    payload = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": 0.9},
        ],
        "relationships": [
            {
                "source": "Ada Lovelace",
                "target": "Analytical Engine",
                "relation_type": "WORKS_ON",
                "confidence": 0.9,
            }
        ],
    }
    report = extract_chunk(
        db,
        chunk,
        _provider(payload),
        floor=FLOOR,
        model="test-extractor",
        entity_types=[EntityType.PERSON],
    )

    assert report.entities_stored == 1 and report.entities_discarded == 1
    # The edge's target was never stored, so the edge is discarded and counted (R9).
    assert report.relationships_stored == 0 and report.relationships_discarded == 1
    assert [entity.name for entity in db.execute(select(Entity)).scalars()] == ["Ada Lovelace"]


def test_a_disabled_relation_type_is_left_out_of_the_prompt_discarded_and_counted(db):
    from ragfabric_core.graph.contracts import RelationType
    from ragfabric_core.graph.extract import build_extraction_prompt

    assert "WORKS_ON" not in build_extraction_prompt(None, [RelationType.MEMBER_OF])

    chunk = _make_chunk(db)
    payload = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": 0.9},
        ],
        "relationships": [
            {
                "source": "Ada Lovelace",
                "target": "Analytical Engine",
                "relation_type": "WORKS_ON",
                "confidence": 0.9,
            }
        ],
    }
    report = extract_chunk(
        db,
        chunk,
        _provider(payload),
        floor=FLOOR,
        model="test-extractor",
        relation_types=[RelationType.MEMBER_OF],
    )

    assert report.entities_stored == 2
    assert report.relationships_stored == 0 and report.relationships_discarded == 1
    assert db.execute(select(Relationship)).scalars().all() == []
