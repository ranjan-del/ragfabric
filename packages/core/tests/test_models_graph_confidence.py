"""Model tests for graph confidence, extraction hashes and merge records.

Phase 6 adds an extraction confidence to entities and relationships. ADR 0004
holds: a confidence the model never reported is `None`, not a plausible
looking `0.0`, so the default has to be `None` and the valid range (when a
value is present) is enforced by the database rather than trusted to callers.
`entity_merges` records every resolution decision so a merge can be inspected
and undone (Task 5); a merge row without its evidence is meaningless and is
rejected the same way.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import Entity, EntityMerge


@pytest.fixture()
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'graph_confidence.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        yield db


def _make_chunk(db) -> Chunk:
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
        text="hello",
        embedding=[0.0],
    )
    db.add(chunk)
    db.flush()
    return chunk


def test_confidence_defaults_to_none_not_zero(session):
    entity = Entity(name="Ada Lovelace", normalized_name="ada lovelace", entity_type="person")
    session.add(entity)
    session.commit()
    assert entity.confidence is None


def test_confidence_outside_zero_to_one_is_rejected(session):
    entity = Entity(
        name="Ada Lovelace",
        normalized_name="ada lovelace",
        entity_type="person",
        confidence=1.5,
    )
    session.add(entity)
    with pytest.raises(IntegrityError):
        session.commit()


def test_extraction_hash_starts_null(session):
    chunk = _make_chunk(session)
    assert chunk.extraction_hash is None


def test_a_merge_record_requires_its_evidence(session):
    entity = Entity(name="Ada Lovelace", normalized_name="ada lovelace", entity_type="person")
    session.add(entity)
    session.flush()

    merge = EntityMerge(
        surviving_entity_id=entity.id,
        merged_name="A. Lovelace",
        merged_entity_type="person",
        method="exact_normalisation",
    )
    session.add(merge)
    with pytest.raises(IntegrityError):
        session.commit()
