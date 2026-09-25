"""An entity's name is text from a chunk, so a caller sees only the spellings they may read (R40).

The review probe: "Ravi Sharma" is sourced only by a denied HR document (0.95)
and "R. Sharma" only by a public one (0.7). The embedding stage merges them
and the higher-confidence spelling survives as ``Entity.name``. Before R40 a
restricted caller asking about "R. Sharma" got a node named "Ravi Sharma",
and asking "Ravi Sharma" matched too, which is an existence oracle for the
denied document. ``entity_sources.surface_name`` records the spelling each
chunk's extraction reported; ``GraphNode.name`` is the best admitted source's
spelling and matching uses only admitted sources' spellings.

Every test runs on SQLite and on PostgreSQL. The PostgreSQL parameter skips
without ``RAGFABRIC_TEST_DATABASE_URL``, and a skip is not a pass.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.graph.contracts import EntityMention, EntityType
from ragfabric_core.graph.extract import extract_chunk
from ragfabric_core.graph.resolve import resolve_entities, unmerge
from ragfabric_core.graph.traverse import match_entities, traverse
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import Entity, EntityMerge, EntitySource
from ragfabric_core.providers.base import Completion, EmbeddingResult

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")

ALL = AccessFilter.unrestricted()

HR_TEXT = "Ravi Sharma is on a performance plan."
PUBLIC_TEXT = "R. Sharma is a member of the Platform Team."

PAYLOADS = {
    HR_TEXT: {
        "entities": [{"name": "Ravi Sharma", "entity_type": "person", "confidence": 0.95}],
        "relationships": [],
    },
    PUBLIC_TEXT: {
        "entities": [
            {"name": "R. Sharma", "entity_type": "person", "confidence": 0.7},
            {"name": "Platform Team", "entity_type": "team", "confidence": 0.9},
        ],
        "relationships": [
            {
                "source": "R. Sharma",
                "target": "Platform Team",
                "relation_type": "MEMBER_OF",
                "confidence": 0.8,
            }
        ],
    },
}

VECTORS = {"Ravi Sharma": [1.0, 0.0], "R. Sharma": [0.99, 0.14], "Platform Team": [0.0, 1.0]}


class _ChunkLLM:
    name = "chunk"
    default_model = "chunk-model"

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        prompt = messages[-1].content
        (payload,) = [p for text, p in PAYLOADS.items() if text in prompt]
        return Completion(
            text=json.dumps(payload),
            model=model or self.default_model,
            provider=self.name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )

    def stream(self, *args, **kwargs):
        raise AssertionError("extraction never streams")


class _Embedder:
    name = "scripted"
    model = "scripted"
    dim = 2

    def embed(self, texts):
        return EmbeddingResult(
            vectors=[VECTORS[text] for text in texts],
            model=self.model,
            provider=self.name,
            input_tokens=0,
            latency_ms=0,
        )


@dataclass
class Merged:
    db: Session
    hr_document: int
    hr_chunk: int
    public_chunk: int
    survivor: int

    @property
    def restricted(self) -> AccessFilter:
        return AccessFilter(denied_document_ids=frozenset({self.hr_document}))

    def ask(self, name: str, access: AccessFilter) -> list[int]:
        return match_entities(
            self.db, [EntityMention(name=name, entity_type=EntityType.PERSON)], access
        )


@pytest.fixture(params=["sqlite", "postgresql"])
def merged(request, tmp_path):
    if request.param == "postgresql":
        if not URL:
            pytest.skip("needs RAGFABRIC_TEST_DATABASE_URL")
        downgrade(URL)
        upgrade(URL)
        engine = create_engine(URL)
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'names.db'}")
        Base.metadata.create_all(engine)

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        collection = Collection(name="c")
        db.add(collection)
        db.flush()
        chunks = {}
        for label, text in (("hr", HR_TEXT), ("public", PUBLIC_TEXT)):
            document = Document(
                filename=f"{label}.txt", format="txt", collection_id=collection.id, status="ready"
            )
            db.add(document)
            db.flush()
            chunk = Chunk(
                document_id=document.id,
                collection_id=collection.id,
                chunk_index=0,
                text=text,
                embedding=[],
            )
            db.add(chunk)
            db.flush()
            chunks[label] = (document.id, chunk)
        for _, chunk in chunks.values():
            extract_chunk(db, chunk, _ChunkLLM(), floor=0.5, model="chunk-model")
        report = resolve_entities(db, _Embedder(), similarity_threshold=0.9)
        db.commit()
        # Not vacuous: the two spellings really were merged, and the denied
        # document's spelling is the one the stored entity carries.
        [merge] = report.merges
        survivor = db.get(Entity, merge.survivor_id)
        assert survivor.name == "Ravi Sharma"
        yield Merged(
            db=db,
            hr_document=chunks["hr"][0],
            hr_chunk=chunks["hr"][1].id,
            public_chunk=chunks["public"][1].id,
            survivor=survivor.id,
        )
    engine.dispose()


def test_extraction_records_the_spelling_each_chunk_reported(merged):
    names = dict(
        merged.db.execute(
            select(EntitySource.chunk_id, EntitySource.surface_name).where(
                EntitySource.entity_id == merged.survivor
            )
        ).all()
    )
    assert names == {merged.hr_chunk: "Ravi Sharma", merged.public_chunk: "R. Sharma"}


def test_a_restricted_caller_sees_only_the_spelling_their_documents_use(merged):
    assert merged.ask("R. Sharma", merged.restricted) == [merged.survivor]
    subgraph = traverse(merged.db, [merged.survivor], merged.restricted, max_hops=1)
    names = {node.id: node.name for node in subgraph.nodes}
    assert names[merged.survivor] == "R. Sharma"
    assert "Ravi Sharma" not in subgraph.model_dump_json()
    # Not vacuous: the walk still reaches the public side of the graph.
    assert "Platform Team" in names.values()


def test_a_restricted_caller_cannot_match_a_denied_spelling(merged):
    """Matching the denied spelling would confirm the HR document names this person."""
    assert merged.ask("Ravi Sharma", merged.restricted) == []
    assert merged.ask("Ravi Sharma", ALL) == [merged.survivor]


def test_an_unrestricted_caller_sees_the_higher_confidence_spelling(merged):
    subgraph = traverse(merged.db, [merged.survivor], ALL, max_hops=1)
    names = {node.id: node.name for node in subgraph.nodes}
    assert names[merged.survivor] == "Ravi Sharma"


def test_entity_aliases_are_not_caller_facing(merged):
    """An alias is resolution's record of a merged name, not text a caller may read."""
    survivor = merged.db.get(Entity, merged.survivor)
    assert "R. Sharma" in survivor.aliases
    survivor.aliases = [*survivor.aliases, "The Sharma From HR"]
    merged.db.flush()
    assert merged.ask("The Sharma From HR", ALL) == []


def test_unmerge_moves_each_spelling_back_with_its_row(merged):
    [record] = merged.db.execute(select(EntityMerge)).scalars().all()
    result = unmerge(merged.db, record.id)
    merged.db.flush()

    rows = merged.db.execute(
        select(EntitySource.entity_id, EntitySource.chunk_id, EntitySource.surface_name)
    ).all()
    by_chunk = {chunk_id: (entity_id, name) for entity_id, chunk_id, name in rows}
    assert by_chunk[merged.hr_chunk] == (merged.survivor, "Ravi Sharma")
    assert by_chunk[merged.public_chunk][0] == result.restored_entity_id
    person_rows = {
        chunk_id: name
        for entity_id, chunk_id, name in rows
        if entity_id in {merged.survivor, result.restored_entity_id}
    }
    assert person_rows == {merged.hr_chunk: "Ravi Sharma", merged.public_chunk: "R. Sharma"}
