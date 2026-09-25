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
    """Atomicity: link removal and re-extraction succeed or fail together.

    Two chunks share an entity and an edge. If chunk one's link removal ran
    and the new extraction then hit a contract violation, a non-atomic
    implementation would have deleted chunk one's real old contributions for
    nothing, even though the shared edge still has chunk two behind it. The
    nested transaction rolls the link removal back too, and the hash is left
    unset so the chunk is retried next time.
    """
    chunk_one = _make_chunk(db, text="Ada Lovelace works on the Analytical Engine.")
    chunk_two = _make_chunk(db, text="Ada Lovelace also works on the Analytical Engine.")
    payload_one = {
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
    extract_chunk(db, chunk_one, _provider(payload_one), floor=FLOOR, model="model-a")
    extract_chunk(db, chunk_two, _provider(payload_two), floor=FLOOR, model="model-b")
    old_hash = chunk_one.extraction_hash

    ada = db.execute(select(Entity).where(Entity.normalized_name == "ada lovelace")).scalar_one()
    ada_id = ada.id
    edge = db.execute(select(Relationship)).scalar_one()
    edge_id = edge.id
    assert ada.confidence == 0.9
    assert edge.confidence == 0.9
    assert edge.extraction_model == "model-a"

    chunk_one.text = "Something entirely different now."
    violating_provider = ScriptedLLMProvider(
        responses=["not a json object at all"], model="model-c"
    )
    report = extract_chunk(db, chunk_one, violating_provider, floor=FLOOR, model="model-c")

    assert report.contract_violation is True
    assert chunk_one.extraction_hash == old_hash

    db.refresh(ada)
    db.refresh(edge)
    assert db.get(Entity, ada_id) is not None
    assert ada.confidence == 0.9
    assert db.get(EntitySource, (ada_id, chunk_one.id)) is not None

    # The shared edge, its link to chunk two, and its aggregate confidence
    # and model all survive the violating re-extraction of chunk one too.
    assert db.get(Relationship, edge_id) is not None
    assert edge.confidence == 0.9
    assert edge.extraction_model == "model-a"
    assert db.get(RelationshipSource, (edge_id, chunk_one.id)) is not None
    link_two = db.get(RelationshipSource, (edge_id, chunk_two.id))
    assert link_two is not None
    assert link_two.confidence == 0.6


def test_a_re_reported_entity_keeps_its_id_and_aliases_across_re_extraction(db):
    """R23: link removal, not deletion, comes first, so identity survives.

    An entity sourced only by the chunk being re-extracted must not be
    deleted and recreated under a new id when the new text still reports it:
    that would silently drop any alias set on it and, once merge history
    exists, the merge rows an ``ON DELETE CASCADE`` from
    ``entity_merges.surviving_entity_id`` would take down with it. Removing
    the chunk's link first, then letting re-extraction's ordinary upsert
    find the still-existing row by its (normalised name, type) key, is what
    keeps the id (and anything hung off it) stable.
    """
    chunk = _make_chunk(db, text="Ada Lovelace works on the Analytical Engine.")
    payload_one = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9}],
        "relationships": [],
    }
    extract_chunk(db, chunk, _provider(payload_one), floor=FLOOR, model="model-a")

    ada = db.execute(select(Entity).where(Entity.normalized_name == "ada lovelace")).scalar_one()
    ada_id = ada.id
    ada.aliases = ["Countess of Lovelace"]
    db.flush()

    chunk.text = "Ada Lovelace, also known as the Countess of Lovelace, designed algorithms."
    payload_two = {
        "entities": [{"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.7}],
        "relationships": [],
    }
    report = extract_chunk(db, chunk, _provider(payload_two), floor=FLOOR, model="model-b")

    assert report.chunks_skipped == 0
    ada_after = db.execute(
        select(Entity).where(Entity.normalized_name == "ada lovelace")
    ).scalar_one()
    assert ada_after.id == ada_id
    assert ada_after.aliases == ["Countess of Lovelace"]
    assert ada_after.confidence == 0.7
    assert ada_after.extraction_model == "model-b"


def test_re_extraction_does_not_crash_when_an_edges_other_source_outlives_its_endpoint(db):
    """Regression for the stale-cascade crash: ORM-delete the edge before its endpoint.

    Chunk one and chunk two both sourced an edge (Ada WORKS_ON Engine), but
    only chunk one ever sourced the "Engine" endpoint. (Built directly here:
    the extraction contract cannot produce this asymmetry from a single
    extraction call, since a chunk that reports an edge must also report
    both its endpoints in that same response; this is the shape a graph is
    left in after enough independent extractions and re-extractions.)
    Re-extracting chunk one with text that no longer mentions Engine at all
    must not raise ``StaleDataError``: the edge has to be deleted through the
    ORM, explicitly, before its now-orphaned endpoint is, even though the
    edge still has a source of its own on chunk two.
    """
    chunk_one = _make_chunk(db, text="Ada Lovelace works on the Analytical Engine.")
    chunk_two = _make_chunk(db, text="Ada Lovelace works on something else too.")

    ada = Entity(
        name="Ada Lovelace",
        normalized_name="ada lovelace",
        entity_type="person",
        confidence=0.9,
        extraction_model="model-a",
    )
    engine = Entity(
        name="Analytical Engine",
        normalized_name="analytical engine",
        entity_type="product",
        confidence=0.9,
        extraction_model="model-a",
    )
    db.add_all([ada, engine])
    db.flush()

    edge = Relationship(
        source_entity_id=ada.id,
        target_entity_id=engine.id,
        relation_type="WORKS_ON",
        confidence=0.9,
        extraction_model="model-a",
    )
    db.add(edge)
    db.flush()

    db.add_all(
        [
            EntitySource(
                entity_id=ada.id,
                chunk_id=chunk_one.id,
                confidence=0.9,
                extraction_model="model-a",
                surface_name="Ada Lovelace",
            ),
            EntitySource(
                entity_id=ada.id,
                chunk_id=chunk_two.id,
                confidence=0.7,
                extraction_model="model-b",
                surface_name="Ada Lovelace",
            ),
            EntitySource(
                entity_id=engine.id,
                chunk_id=chunk_one.id,
                confidence=0.9,
                extraction_model="model-a",
                surface_name="Analytical Engine",
            ),
            RelationshipSource(
                relationship_id=edge.id,
                chunk_id=chunk_one.id,
                confidence=0.9,
                extraction_model="model-a",
            ),
            RelationshipSource(
                relationship_id=edge.id,
                chunk_id=chunk_two.id,
                confidence=0.7,
                extraction_model="model-b",
            ),
        ]
    )
    chunk_one.extraction_hash = "fake-old-hash"
    db.flush()
    edge_id = edge.id
    engine_id = engine.id

    payload = {
        "entities": [{"name": "Bread Recipe", "entity_type": "document", "confidence": 0.9}],
        "relationships": [],
    }
    report = extract_chunk(db, chunk_one, _provider(payload), floor=FLOOR, model="model-c")

    assert report.contract_violation is False
    assert db.get(Entity, engine_id) is None
    assert db.get(Relationship, edge_id) is None
    db.refresh(ada)
    assert ada.confidence == 0.7
    assert ada.extraction_model == "model-b"


def test_a_survivor_whose_remaining_sources_are_all_unmeasured_has_no_confidence(db):
    """A recomputed confidence is never fabricated (ADR 0004): all-``None`` sources stay ``None``.

    Set up directly: an entity with two sources, neither carrying a measured
    confidence (as a merge, rather than an extraction call, can leave
    behind). Re-extracting the chunk that supplies one of them, with new
    text that drops the entity, must leave the entity's confidence and
    extraction_model at ``None`` rather than inventing a number from a source
    that never measured one.
    """
    chunk_one = _make_chunk(db, text="Ada Lovelace works on the Analytical Engine.")
    chunk_two = _make_chunk(db, text="Ada Lovelace works on something else too.")

    ada = Entity(
        name="Ada Lovelace",
        normalized_name="ada lovelace",
        entity_type="person",
        confidence=None,
        extraction_model=None,
    )
    db.add(ada)
    db.flush()

    db.add_all(
        [
            EntitySource(
                entity_id=ada.id,
                chunk_id=chunk_one.id,
                confidence=None,
                extraction_model=None,
                surface_name="Ada Lovelace",
            ),
            EntitySource(
                entity_id=ada.id,
                chunk_id=chunk_two.id,
                confidence=None,
                extraction_model=None,
                surface_name="Ada Lovelace",
            ),
        ]
    )
    chunk_one.extraction_hash = "fake-old-hash"
    db.flush()
    ada_id = ada.id

    payload = {
        "entities": [{"name": "Bread Recipe", "entity_type": "document", "confidence": 0.9}],
        "relationships": [],
    }
    report = extract_chunk(db, chunk_one, _provider(payload), floor=FLOOR, model="model-b")

    assert report.contract_violation is False
    ada_after = db.get(Entity, ada_id)
    assert ada_after is not None
    assert ada_after.confidence is None
    assert ada_after.extraction_model is None
