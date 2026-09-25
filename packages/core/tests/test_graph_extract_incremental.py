"""Tests for skipping re-extraction of unchanged chunks.

One LLM call per chunk is the dominant cost of graph extraction. Without a
way to tell "this chunk's text has not changed since it was last extracted",
re-ingesting a corpus to fix one document pays for every chunk in it again.
A chunk's ``extraction_hash`` records the hash of the text as of its last
successful extraction; a chunk whose current text still hashes to that value
is skipped, no LLM call made, and the skip is counted.

A chunk whose text *has* changed must not just re-extract: its previous
contributions (the entities and edges it used to justify) have to be removed
first, or stale assertions from the old text survive forever alongside the
new ones. That cleanup, and the atomicity between it and the re-extraction
call, is exercised in detail below.
"""

import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

# Importing this registers `PRAGMA foreign_keys=ON` for every SQLite
# connection in this process (see db/session.py); the cascade-delete
# assertions below depend on it and this makes that true whether this file
# runs alone or as part of the full suite.
import ragfabric_core.db.session  # noqa: F401
from ragfabric_core.graph.extract import extract_chunk, extract_chunks
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import Entity, EntitySource, Relationship, RelationshipSource
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


def _provider(*payloads: dict) -> ScriptedLLMProvider:
    return ScriptedLLMProvider(
        responses=[json.dumps(payload) for payload in payloads], model="test-extractor"
    )


def test_an_unchanged_chunk_is_not_re_extracted(db):
    chunk = _make_chunk(db)
    payload = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    provider = _provider(payload)

    first = extract_chunk(db, chunk, provider, floor=FLOOR, model="test-extractor")
    assert first.entities_stored == 1
    assert first.chunks_skipped == 0
    assert chunk.extraction_hash is not None

    # Only one response was scripted; a second real call would raise.
    second = extract_chunk(db, chunk, provider, floor=FLOOR, model="test-extractor")

    assert second.chunks_skipped == 1
    assert second.entities_stored == 0
    assert provider.calls == 1


def test_a_changed_chunk_is_re_extracted(db):
    chunk = _make_chunk(db, text="Ada Lovelace is a mathematician.")
    payload_one = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    payload_two = {
        "entities": [{"name": "Charles Babbage", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    provider = _provider(payload_one, payload_two)

    extract_chunk(db, chunk, provider, floor=FLOOR, model="test-extractor")
    old_hash = chunk.extraction_hash

    chunk.text = "Charles Babbage designed engines."
    report = extract_chunk(db, chunk, provider, floor=FLOOR, model="test-extractor")

    assert provider.calls == 2
    assert report.chunks_skipped == 0
    assert report.entities_stored == 1
    assert chunk.extraction_hash != old_hash
    babbage = db.execute(
        select(Entity).where(Entity.normalized_name == "charles babbage")
    ).scalar_one()
    assert db.get(EntitySource, (babbage.id, chunk.id)) is not None


def test_a_chunk_with_no_prior_hash_is_extracted(db):
    """NULL extraction_hash (never extracted) is unambiguously "changed"."""
    chunk = _make_chunk(db)
    assert chunk.extraction_hash is None
    payload = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }

    report = extract_chunk(db, chunk, _provider(payload), floor=FLOOR, model="test-extractor")

    assert report.chunks_skipped == 0
    assert report.entities_stored == 1
    assert chunk.extraction_hash is not None


def test_the_number_skipped_is_reported(db):
    unchanged = _make_chunk(db, text="Ada Lovelace is a mathematician.")
    changed = _make_chunk(db, text="original text about Babbage.")
    payload_one = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    payload_two = {
        "entities": [{"name": "Charles Babbage", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    payload_three = {
        "entities": [{"name": "Ada King", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    extract_chunks(
        db, [unchanged, changed], _provider(payload_one, payload_two), floor=FLOOR, model="m"
    )

    changed.text = "new text about Ada King."
    report = extract_chunks(
        db, [unchanged, changed], _provider(payload_three), floor=FLOOR, model="m"
    )

    assert report.chunks_skipped == 1
    assert report.entities_stored == 1


def test_re_extraction_removes_the_edges_the_old_text_produced(db):
    """The subtle half: a changed chunk's stale contributions must not survive.

    An entity ("Steam Loom") and an edge (Ada WORKS_ON Analytical Engine) are
    each shared between two chunks. Only chunk one's text changes. After
    re-extraction: the shared entity and edge survive, but with their
    confidence recomputed from chunk two's contribution alone (chunk one's
    was higher, so this also proves the recompute actually re-ran rather than
    coincidentally keeping the old number); the entity chunk one alone
    supported ("Steam Loom") is gone entirely.
    """
    chunk_one = _make_chunk(
        db, text="Ada Lovelace works on the Analytical Engine and a Steam Loom."
    )
    chunk_two = _make_chunk(db, text="Ada Lovelace also works on the Analytical Engine.")

    payload_one_first = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": 0.9},
            {"name": "Steam Loom", "entity_type": "product", "confidence": 0.9},
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
    payload_two = {
        "entities": [
            {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.6},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": 0.6},
        ],
        "relationships": [
            {
                "source": "Ada Lovelace",
                "target": "Analytical Engine",
                "relation_type": "WORKS_ON",
                "confidence": 0.6,
            }
        ],
    }
    extract_chunk(db, chunk_one, _provider(payload_one_first), floor=FLOOR, model="model-a")
    extract_chunk(db, chunk_two, _provider(payload_two), floor=FLOOR, model="model-b")

    ada = db.execute(select(Entity).where(Entity.normalized_name == "ada lovelace")).scalar_one()
    engine = db.execute(
        select(Entity).where(Entity.normalized_name == "analytical engine")
    ).scalar_one()
    edge = db.execute(select(Relationship)).scalar_one()
    assert ada.confidence == 0.9
    assert edge.confidence == 0.9

    # Chunk one's text changes to something that no longer mentions any of it.
    chunk_one.text = "The weather that winter was unusually cold."
    payload_one_second = {
        "entities": [{"name": "Bread Recipe", "entity_type": "document", "confidence": 0.9}],
        "relationships": [],
    }
    report = extract_chunk(
        db, chunk_one, _provider(payload_one_second), floor=FLOOR, model="model-c"
    )

    assert report.entities_stored == 1
    assert report.chunks_skipped == 0

    # Shared rows survive, recomputed from chunk two's contribution alone.
    db.refresh(ada)
    db.refresh(engine)
    db.refresh(edge)
    assert ada.confidence == 0.6
    assert ada.extraction_model == "model-b"
    assert engine.confidence == 0.6
    assert edge.confidence == 0.6
    assert edge.extraction_model == "model-b"
    assert db.get(EntitySource, (ada.id, chunk_one.id)) is None
    assert db.get(EntitySource, (ada.id, chunk_two.id)) is not None
    assert db.get(RelationshipSource, (edge.id, chunk_one.id)) is None
    assert db.get(RelationshipSource, (edge.id, chunk_two.id)) is not None

    # The entity only chunk one ever supported is gone entirely.
    steam_loom = db.execute(
        select(Entity).where(Entity.normalized_name == "steam loom")
    ).scalar_one_or_none()
    assert steam_loom is None

    # The new text's own contribution is stored and linked to chunk one.
    bread = db.execute(select(Entity).where(Entity.normalized_name == "bread recipe")).scalar_one()
    assert db.get(EntitySource, (bread.id, chunk_one.id)) is not None


def test_a_contract_violation_on_re_extraction_leaves_the_old_graph_intact(db):
    """Atomicity: cleanup and re-extraction succeed or fail together.

    If the cleanup ran and then the new extraction hit a contract violation,
    a non-atomic implementation would have deleted the chunk's real old
    contributions for nothing. The nested transaction rolls the cleanup back
    too, and the hash is left unset so the chunk is retried next time.
    """
    chunk = _make_chunk(db, text="Ada Lovelace is a mathematician.")
    payload = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    extract_chunk(db, chunk, _provider(payload), floor=FLOOR, model="model-a")
    old_hash = chunk.extraction_hash
    ada = db.execute(select(Entity).where(Entity.normalized_name == "ada lovelace")).scalar_one()
    ada_id = ada.id

    chunk.text = "Something entirely different now."
    violating_provider = ScriptedLLMProvider(
        responses=["not a json object at all"], model="model-b"
    )
    report = extract_chunk(db, chunk, violating_provider, floor=FLOOR, model="model-b")

    assert report.contract_violation is True
    assert chunk.extraction_hash == old_hash

    db.refresh(ada)
    assert db.get(Entity, ada_id) is not None
    assert ada.confidence == 0.9
    assert db.get(EntitySource, (ada_id, chunk.id)) is not None
