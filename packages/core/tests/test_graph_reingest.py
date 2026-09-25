"""Re-ingesting a document carries its graph across the chunk replacement (R41, R42).

``reingest_document`` replaces every chunk row. Before R41 the new rows came
with ``extraction_hash=None`` and no graph links, so an identical re-ingest
re-extracted every chunk (the review probe: 2 LLM calls became 3 for a
one-chunk re-ingest) and the old links went with the cascade, leaving parent
confidences computed from sources that no longer existed (R42).

Now each new chunk whose text equals an old chunk's (in ``chunk_index`` order
when texts repeat) inherits that chunk's hash and graph links; only a chunk
with new text is extracted, and whatever only the replaced text sourced is
garbage collected and the rest recomputed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from ragfabric_core.config_file import GraphStoreConfig
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.graph.extract import carry_graph_to_new_chunks, detach_documents
from ragfabric_core.ingest import indexing, pipeline
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import Entity, EntitySource, Relationship, RelationshipSource
from ragfabric_core.providers.base import Completion, EmbeddingResult
from ragfabric_core.workers import handlers

ADA = "Ada Lovelace works on the Analytical Engine."
GRACE = "Grace Hopper works on the Analytical Engine."
LINUS = "Linus Torvalds works on the Analytical Engine."

CONFIDENCE = {ADA: 0.95, GRACE: 0.6, LINUS: 0.7}


def _payload(text: str) -> dict:
    person = text.split(" works on")[0]
    confidence = CONFIDENCE[text]
    return {
        "entities": [
            {"name": person, "entity_type": "person", "confidence": confidence},
            {"name": "Analytical Engine", "entity_type": "product", "confidence": confidence},
        ],
        "relationships": [
            {
                "source": person,
                "target": "Analytical Engine",
                "relation_type": "WORKS_ON",
                "confidence": confidence,
            }
        ],
    }


class _TextLLM:
    """Answers each extraction call from the one known sentence the prompt contains."""

    name = "text"
    default_model = "text-model"

    def __init__(self) -> None:
        self.extracted: list[str] = []

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        prompt = messages[-1].content
        (text,) = [text for text in CONFIDENCE if text in prompt]
        self.extracted.append(text)
        return Completion(
            text=json.dumps(_payload(text)),
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

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        # Orthogonal per name: nothing merges by embedding in these tests.
        vectors = {
            "Ada Lovelace": [1.0, 0.0],
            "Grace Hopper": [0.0, 1.0],
            "Linus Torvalds": [-1.0, 0.0],
            "Analytical Engine": [0.0, -1.0],
        }
        return EmbeddingResult(
            vectors=[vectors[text] for text in texts],
            model=self.model,
            provider=self.name,
            input_tokens=0,
            latency_ms=0,
        )


class _NoStorage:
    def save(self, document_id, filename, data):
        return f"mem://{document_id}/{filename}"

    def delete(self, document_id):
        return None


def _paragraphs(text, *, chunk_size, overlap, sections):
    """One chunk per paragraph, so a test controls exactly which chunk text changes."""
    out, start = [], 0
    for index, paragraph in enumerate(text.split("\n\n")):
        begin = text.index(paragraph, start)
        out.append(
            {
                "chunk_index": index,
                "page": 1,
                "char_start": begin,
                "char_end": begin + len(paragraph),
                "text": paragraph,
                "section": None,
            }
        )
        start = begin + len(paragraph)
    return out


@pytest.fixture()
def env(tmp_path: Path, monkeypatch):
    llm, embedder = _TextLLM(), _Embedder()
    monkeypatch.setattr(pipeline, "build_queue", lambda cfg: None)
    monkeypatch.setattr(pipeline, "get_storage", lambda: _NoStorage())
    monkeypatch.setattr(pipeline, "chunk_text", _paragraphs)
    monkeypatch.setattr(indexing, "index_inline", lambda db, document: 0)
    monkeypatch.setattr(indexing, "_embedding_provider", lambda: embedder)

    def extract(db, document_id, **_configured):
        # The configured graph is off in tests; run the real handler enabled,
        # with the doubles, exactly as a worker would.
        return handlers.extract_graph(
            db, document_id, settings=GraphStoreConfig(enabled=True), llm=llm, embedder=embedder
        )

    monkeypatch.setattr(indexing, "extract_graph", extract)

    engine = create_engine(f"sqlite:///{tmp_path / 'reingest.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory, llm, embedder
    engine.dispose()


def _ingest(factory, *paragraphs: str) -> int:
    with factory() as db:
        document = pipeline.ingest_document(
            db, filename="a.txt", data="\n\n".join(paragraphs).encode()
        )
        assert document.status == "ready", document.error
        return document.id


def _reingest(factory, document_id: int, *paragraphs: str) -> None:
    with factory() as db:
        document = pipeline.reingest_document(
            db,
            db.get(Document, document_id),
            filename="a.txt",
            data="\n\n".join(paragraphs).encode(),
        )
        assert document.status == "ready", document.error


def _graph(factory) -> dict:
    """The whole graph, with chunks named by their text so a re-ingest compares equal."""
    with factory() as db:
        text = dict(db.execute(select(Chunk.id, Chunk.text)).all())
        return {
            "entities": sorted(
                db.execute(
                    select(Entity.id, Entity.name, Entity.entity_type, Entity.confidence)
                ).all()
            ),
            "relationships": sorted(
                db.execute(
                    select(
                        Relationship.id,
                        Relationship.source_entity_id,
                        Relationship.target_entity_id,
                        Relationship.relation_type,
                        Relationship.confidence,
                    )
                ).all()
            ),
            "entity_sources": sorted(
                (row.entity_id, text[row.chunk_id], row.confidence, row.surface_name)
                for row in db.execute(select(EntitySource)).scalars()
            ),
            "relationship_sources": sorted(
                (row.relationship_id, text[row.chunk_id], row.confidence)
                for row in db.execute(select(RelationshipSource)).scalars()
            ),
        }


def test_an_identical_re_ingest_makes_no_model_or_embedding_call_and_keeps_the_graph(env):
    factory, llm, embedder = env
    document_id = _ingest(factory, ADA, GRACE)
    assert llm.extracted == [ADA, GRACE]
    assert embedder.calls == 1
    before = _graph(factory)
    with factory() as db:
        old_chunks = set(db.execute(select(Chunk.id)).scalars())

    _reingest(factory, document_id, ADA, GRACE)

    assert llm.extracted == [ADA, GRACE], "no chunk is extracted again"
    assert embedder.calls == 1, "nothing is resolved again"
    assert _graph(factory) == before
    with factory() as db:
        chunks = db.execute(select(Chunk)).scalars().all()
        # Not vacuous: the chunk rows really were replaced.
        assert old_chunks.isdisjoint({chunk.id for chunk in chunks})
        assert all(chunk.extraction_hash is not None for chunk in chunks)


def test_a_re_ingest_changing_one_chunk_re_extracts_only_that_chunk(env):
    factory, llm, _ = env
    document_id = _ingest(factory, ADA, GRACE)

    _reingest(factory, document_id, ADA, LINUS)

    assert llm.extracted == [ADA, GRACE, LINUS]
    with factory() as db:
        names = set(db.execute(select(Entity.name)).scalars())
        engine = db.execute(select(Entity).where(Entity.name == "Analytical Engine")).scalar_one()
        # Grace was sourced only by the replaced text: collected, with her edge.
        assert names == {"Ada Lovelace", "Linus Torvalds", "Analytical Engine"}
        edges = db.execute(select(Relationship)).scalars().all()
        assert len(edges) == 2
        # The engine's confidence comes from the sources it has now (0.95, 0.7).
        assert engine.confidence == 0.95


def test_a_re_ingest_recomputes_a_confidence_the_replaced_text_supplied(env):
    """R42 on re-ingest: the 0.95 came from text that is gone, so it must not stay."""
    factory, llm, _ = env
    document_id = _ingest(factory, ADA, GRACE)
    with factory() as db:
        assert (
            db.execute(
                select(Entity.confidence).where(Entity.name == "Analytical Engine")
            ).scalar_one()
            == 0.95
        )

    _reingest(factory, document_id, GRACE)

    with factory() as db:
        assert (
            db.execute(
                select(Entity.confidence).where(Entity.name == "Analytical Engine")
            ).scalar_one()
            == 0.6
        )
        assert set(db.execute(select(Entity.name)).scalars()) == {
            "Grace Hopper",
            "Analytical Engine",
        }


def test_repeated_text_is_matched_in_chunk_order(env):
    """Two chunks with the same text both inherit a hash; neither is re-extracted."""
    factory, llm, _ = env
    document_id = _ingest(factory, ADA, ADA, GRACE)
    assert llm.extracted == [ADA, ADA, GRACE]
    before = _graph(factory)

    _reingest(factory, document_id, ADA, ADA, GRACE)

    assert llm.extracted == [ADA, ADA, GRACE]
    assert _graph(factory) == before


# --- the graph helpers themselves, on both dialects -------------------------------------

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")


@pytest.fixture(params=["sqlite", "postgresql"])
def db(request, tmp_path):
    if request.param == "postgresql":
        if not URL:
            pytest.skip("needs RAGFABRIC_TEST_DATABASE_URL")
        downgrade(URL)
        upgrade(URL)
        engine = create_engine(URL)
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'helpers.db'}")
        Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as session:
        yield session
    engine.dispose()


def _two_documents(db) -> dict:
    """An edge and its target sourced 0.95 by document one and 0.6 by document two;
    a second edge and entity sourced only by document one."""
    collection = Collection(name="c")
    db.add(collection)
    db.flush()
    chunks = {}
    for label, text in (("one", ADA), ("two", GRACE)):
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
            extraction_hash="h",
        )
        db.add(chunk)
        db.flush()
        chunks[label] = chunk

    def entity(name: str, sources: dict[str, float]) -> Entity:
        row = Entity(
            name=name,
            normalized_name=name.casefold(),
            entity_type="person",
            confidence=max(sources.values()),
        )
        db.add(row)
        db.flush()
        for label, value in sources.items():
            db.add(
                EntitySource(
                    entity_id=row.id,
                    chunk_id=chunks[label].id,
                    confidence=value,
                    surface_name=name,
                )
            )
        return row

    def edge(source: Entity, target: Entity, sources: dict[str, float]) -> Relationship:
        row = Relationship(
            source_entity_id=source.id,
            target_entity_id=target.id,
            relation_type="WORKS_ON",
            confidence=max(sources.values()),
        )
        db.add(row)
        db.flush()
        for label, value in sources.items():
            db.add(
                RelationshipSource(
                    relationship_id=row.id, chunk_id=chunks[label].id, confidence=value
                )
            )
        return row

    ada = entity("Ada", {"one": 0.95, "two": 0.6})
    engine = entity("Engine", {"one": 0.95, "two": 0.6})
    babbage = entity("Babbage", {"one": 0.9})
    shared = edge(ada, engine, {"one": 0.95, "two": 0.6})
    only_one = edge(babbage, engine, {"one": 0.9})
    db.flush()
    return {
        "chunks": chunks,
        "ada": ada.id,
        "engine": engine.id,
        "babbage": babbage.id,
        "shared": shared.id,
        "only_one": only_one.id,
    }


def test_detaching_a_document_recomputes_confidence_and_collects_orphans(db):
    graph = _two_documents(db)
    doomed = graph["chunks"]["one"]

    detach_documents(db, [doomed.document_id])
    db.delete(doomed)
    db.flush()

    assert db.get(Relationship, graph["shared"]).confidence == 0.6
    assert db.get(Entity, graph["ada"]).confidence == 0.6
    assert db.get(Entity, graph["engine"]).confidence == 0.6
    assert db.get(Relationship, graph["only_one"]) is None
    assert db.get(Entity, graph["babbage"]) is None


def test_carrying_moves_links_and_hash_to_the_chunk_with_the_same_text(db):
    graph = _two_documents(db)
    old = graph["chunks"]["one"]
    new = Chunk(
        document_id=old.document_id,
        collection_id=old.collection_id,
        chunk_index=0,
        text=old.text,
        embedding=[],
    )
    db.add(new)
    db.flush()

    carry_graph_to_new_chunks(db, [old], [new])
    db.delete(old)
    db.flush()

    assert new.extraction_hash == "h"
    assert db.get(Relationship, graph["shared"]).confidence == 0.95
    assert db.get(Entity, graph["babbage"]) is not None
    assert set(
        db.execute(select(EntitySource.entity_id).where(EntitySource.chunk_id == new.id)).scalars()
    ) == {graph["ada"], graph["engine"], graph["babbage"]}
    assert set(
        db.execute(
            select(RelationshipSource.relationship_id).where(RelationshipSource.chunk_id == new.id)
        ).scalars()
    ) == {graph["shared"], graph["only_one"]}
