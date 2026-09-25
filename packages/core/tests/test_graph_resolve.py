"""Tests for entity resolution: recorded, reversible merges.

Entity resolution is the main failure mode of graph RAG. Merge too eagerly and
two people named Sharma become one node that the graph then asserts things
about; merge too little and a traversal that should connect two facts finds no
path. Neither can be tuned without measurement, so what these tests pin down is
that every merge is justified by recorded evidence and can be undone, with the
separated entity getting back exactly what its own chunks said about it.
"""

import json
import math
import os

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

# Registers `PRAGMA foreign_keys=ON` for every SQLite connection in this
# process, so SQLite enforces the same foreign keys PostgreSQL does.
import ragfabric_core.db.session  # noqa: F401
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.graph.extract import extract_chunk
from ragfabric_core.graph.resolve import UnmergeBlocked, resolve_entities, unmerge
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import (
    Entity,
    EntityMerge,
    EntitySource,
    Relationship,
    RelationshipSource,
)
from ragfabric_core.providers.base import EmbeddingResult
from ragfabric_core.providers.offline import ScriptedLLMProvider

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
THRESHOLD = 0.9


@pytest.fixture(scope="module")
def postgres_engine():
    if not URL:
        pytest.skip("needs RAGFABRIC_TEST_DATABASE_URL")
    downgrade(URL)
    upgrade(URL)
    engine = create_engine(URL)
    yield engine
    engine.dispose()


@pytest.fixture(params=["sqlite", "postgresql"])
def db(request, tmp_path):
    if request.param == "postgresql":
        engine = request.getfixturevalue("postgres_engine")
        # Every test runs inside one outer transaction that is rolled back,
        # so tests share the migrated schema without seeing each other's rows.
        connection = engine.connect()
        outer = connection.begin()
        session = Session(bind=connection, join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            session.close()
            outer.rollback()
            connection.close()
        return

    engine = create_engine(f"sqlite:///{tmp_path / 'resolve.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        yield session
    engine.dispose()


class ScriptedEmbedder:
    """An embedder that returns fixed vectors per text and counts its calls.

    An unknown text raises, so a test notices when resolution embeds
    something it did not expect to.
    """

    name = "scripted"
    model = "scripted-embedder"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.dim = len(next(iter(vectors.values()))) if vectors else 2
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        return EmbeddingResult(
            vectors=[self.vectors[text] for text in texts],
            model=self.model,
            provider=self.name,
            input_tokens=0,
            latency_ms=0,
        )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _chunks(db, count: int) -> list[int]:
    collection = Collection(name="c")
    db.add(collection)
    db.flush()
    document = Document(filename="a.txt", format="txt", collection_id=collection.id, status="ready")
    db.add(document)
    db.flush()
    chunks = [
        Chunk(
            document_id=document.id,
            collection_id=collection.id,
            chunk_index=index,
            text=f"chunk {index}",
            embedding=[0.0],
        )
        for index in range(count)
    ]
    db.add_all(chunks)
    db.flush()
    return [chunk.id for chunk in chunks]


def _entity(
    db,
    name: str,
    entity_type: str,
    sources: dict[int, float | None],
    *,
    normalized_name: str | None = None,
    aliases: list | None = None,
    description: str = "",
    model: str = "test-extractor",
) -> Entity:
    """An entity with one EntitySource per chunk, confidence as that chunk reported."""
    from ragfabric_core.graph.contracts import normalise

    entity = Entity(
        name=name,
        normalized_name=normalized_name if normalized_name is not None else normalise(name),
        entity_type=entity_type,
        description=description,
        aliases=list(aliases or []),
    )
    db.add(entity)
    db.flush()
    for chunk_id, confidence in sources.items():
        db.add(
            EntitySource(
                entity_id=entity.id,
                chunk_id=chunk_id,
                confidence=confidence,
                extraction_model=None if confidence is None else model,
                surface_name=name,
            )
        )
    measured = [value for value in sources.values() if value is not None]
    entity.confidence = max(measured) if measured else None
    entity.extraction_model = model if measured else None
    db.flush()
    return entity


def _edge(db, source: Entity, target: Entity, relation_type: str, sources: dict[int, float]):
    relationship = Relationship(
        source_entity_id=source.id, target_entity_id=target.id, relation_type=relation_type
    )
    db.add(relationship)
    db.flush()
    for chunk_id, confidence in sources.items():
        db.add(
            RelationshipSource(
                relationship_id=relationship.id,
                chunk_id=chunk_id,
                confidence=confidence,
                extraction_model="test-extractor",
            )
        )
    relationship.confidence = max(sources.values())
    relationship.extraction_model = "test-extractor"
    db.flush()
    return relationship


def _entity_sources(db, entity_id: int) -> dict[int, float | None]:
    rows = db.execute(select(EntitySource).where(EntitySource.entity_id == entity_id)).scalars()
    return {row.chunk_id: row.confidence for row in rows}


def _edge_sources(db, relationship_id: int) -> dict[int, float | None]:
    rows = db.execute(
        select(RelationshipSource).where(RelationshipSource.relationship_id == relationship_id)
    ).scalars()
    return {row.chunk_id: row.confidence for row in rows}


def _edges(db) -> list[Relationship]:
    return list(db.execute(select(Relationship).order_by(Relationship.id)).scalars())


def _entities(db) -> list[Entity]:
    return list(db.execute(select(Entity).order_by(Entity.id)).scalars())


# Stage 1: exact normalised match


def test_an_exact_normalised_match_merges(db):
    c1, c2 = _chunks(db, 2)
    # Two surface spellings of one team that normalise equal. The unique
    # constraint keeps them apart only because the second row's stored key
    # was written by an older normaliser; resolution compares normalise(name).
    first = _entity(db, "Platform Team", "team", {c1: 0.9})
    second = _entity(db, "Platform Team.", "team", {c2: 0.7}, normalized_name="platform team.")
    embedder = ScriptedEmbedder({})

    report = resolve_entities(db, embedder, similarity_threshold=THRESHOLD)

    assert [entity.id for entity in _entities(db)] == [first.id]
    assert len(report.merges) == 1
    assert report.merges[0].method == "exact"
    assert report.merges[0].survivor_id == first.id
    assert report.merges[0].merged_id == second.id
    assert _entity_sources(db, first.id) == {c1: 0.9, c2: 0.7}
    assert "Platform Team." in first.aliases
    # One entity left: nothing to compare, so no embedding call was made.
    assert report.embedding_calls == 0
    assert embedder.calls == []


# Stage 2: alias match


def test_an_alias_match_merges(db):
    c1, c2 = _chunks(db, 2)
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.8})

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)

    assert [entity.id for entity in _entities(db)] == [robert.id]
    assert [merge.method for merge in report.merges] == ["alias"]
    assert _entity_sources(db, robert.id) == {c1: 0.9, c2: 0.8}
    # The merged name was already an alias, so it is not added twice.
    assert robert.aliases == ["Bob Sharma"]
    record = db.get(EntityMerge, report.merges[0].merge_id)
    assert record.evidence["alias"] == "Bob Sharma"
    assert record.evidence["normalized_alias"] == "bob sharma"
    assert record.evidence["alias_of"] == "survivor"
    assert record.evidence["matched"] == "name"
    assert bob.id == report.merges[0].merged_id


def test_an_alias_shared_by_two_entities_merges_them(db):
    c1, c2 = _chunks(db, 2)
    first = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["R. Sharma"])
    _entity(db, "Rob Sharma", "person", {c2: 0.8}, aliases=["r sharma", "R. Sharma"])

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)

    assert [entity.id for entity in _entities(db)] == [first.id]
    record = db.get(EntityMerge, report.merges[0].merge_id)
    assert record.evidence["matched"] == "alias"


def test_a_malformed_aliases_value_is_skipped_not_fatal(db):
    c1, c2, c3 = _chunks(db, 3)
    _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=[None, 7, {"x": 1}])
    _entity(db, "Bob Sharma", "person", {c2: 0.8})
    broken = _entity(db, "Anil Sharma", "person", {c3: 0.8})
    broken.aliases = {"not": "a list"}
    db.flush()

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)

    assert report.merges == []
    assert len(_entities(db)) == 3


# Stage 3: embedding similarity


def test_similarity_below_the_threshold_does_not_merge(db):
    c1, c2 = _chunks(db, 2)
    _entity(db, "Ada Lovelace", "person", {c1: 0.9})
    _entity(db, "Ada King", "person", {c2: 0.8})
    # cosine 0.8, below the 0.9 threshold.
    embedder = ScriptedEmbedder({"Ada Lovelace": [1.0, 0.0], "Ada King": [0.8, 0.6]})

    report = resolve_entities(db, embedder, similarity_threshold=THRESHOLD)

    assert report.merges == []
    assert len(_entities(db)) == 2
    assert report.embedding_calls == 1


def test_similarity_at_or_above_the_threshold_merges(db):
    c1, c2 = _chunks(db, 2)
    lovelace = _entity(db, "Ada Lovelace", "person", {c1: 0.9})
    _entity(db, "Ada King", "person", {c2: 0.8})
    # cosine 0.96, above the 0.9 threshold.
    embedder = ScriptedEmbedder({"Ada Lovelace": [1.0, 0.0], "Ada King": [0.96, 0.28]})

    report = resolve_entities(db, embedder, similarity_threshold=THRESHOLD)

    assert [entity.id for entity in _entities(db)] == [lovelace.id]
    assert [merge.method for merge in report.merges] == ["embedding"]


def test_the_embedded_text_is_name_plus_description(db):
    c1, c2 = _chunks(db, 2)
    _entity(db, "Ada Lovelace", "person", {c1: 0.9}, description="Mathematician")
    _entity(db, "Ada King", "person", {c2: 0.8})
    embedder = ScriptedEmbedder({"Ada Lovelace: Mathematician": [1.0, 0.0], "Ada King": [0.0, 1.0]})

    resolve_entities(db, embedder, similarity_threshold=THRESHOLD)

    assert sorted(embedder.calls[0]) == ["Ada King", "Ada Lovelace: Mathematician"]


def test_embedding_calls_are_batched_and_counted(db):
    chunk_ids = _chunks(db, 5)
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
    for chunk_id, name in zip(chunk_ids, names, strict=True):
        _entity(db, name, "project", {chunk_id: 0.9})
    # Orthogonal vectors: nothing is similar, so every entity stays.
    vectors = {name: [1.0 if i == j else 0.0 for j in range(5)] for i, name in enumerate(names)}
    embedder = ScriptedEmbedder(vectors)

    report = resolve_entities(db, embedder, similarity_threshold=THRESHOLD, batch_size=2)

    assert report.embedding_calls == 3
    assert [len(batch) for batch in embedder.calls] == [2, 2, 1]
    assert report.merges == []


def test_a_merged_away_entity_is_not_chained_through_by_similarity(db):
    c1, c2, c3 = _chunks(db, 3)
    # B is close to A and to C, but A and C are not close to each other. B
    # merges into A (the closest pair); C must then not merge into A just
    # because it was similar to B, which no longer exists.
    a = _entity(db, "A", "product", {c1: 0.9})
    _entity(db, "B", "product", {c2: 0.8})
    c = _entity(db, "C", "product", {c3: 0.7})
    s = math.sqrt(0.5)
    embedder = ScriptedEmbedder({"A": [1.0, 0.0], "B": [0.95, 0.312], "C": [s, s]})
    assert _cosine([1.0, 0.0], [0.95, 0.312]) > _cosine([0.95, 0.312], [s, s]) > 0.8
    assert _cosine([1.0, 0.0], [s, s]) < 0.8

    report = resolve_entities(db, embedder, similarity_threshold=0.8)

    assert [merge.method for merge in report.merges] == ["embedding"]
    assert [entity.id for entity in _entities(db)] == [a.id, c.id]


# Evidence


def test_a_merge_records_the_evidence_that_justified_it(db):
    c1, c2, c3 = _chunks(db, 3)
    lovelace = _entity(db, "Ada Lovelace", "person", {c1: 0.9})
    king = _entity(
        db,
        "Ada King",
        "person",
        {c2: 0.8, c3: 0.6},
        aliases=["Countess of Lovelace"],
        description="Countess",
    )
    vectors = {"Ada Lovelace": [1.0, 0.0], "Ada King: Countess": [0.96, 0.28]}
    embedder = ScriptedEmbedder(vectors)

    report = resolve_entities(db, embedder, similarity_threshold=THRESHOLD)

    record = db.get(EntityMerge, report.merges[0].merge_id)
    assert record.surviving_entity_id == lovelace.id
    assert record.merged_name == "Ada King"
    assert record.merged_entity_type == "person"
    assert record.merged_aliases == ["Countess of Lovelace"]
    assert sorted(record.merged_source_chunk_ids) == [c2, c3]
    assert record.method == "embedding"
    assert record.model == "scripted-embedder"
    evidence = record.evidence
    assert evidence["stage"] == "embedding"
    assert evidence["survivor"] == {"id": lovelace.id, "name": "Ada Lovelace", "type": "person"}
    assert evidence["merged"] == {"id": king.id, "name": "Ada King", "type": "person"}
    # The score as measured, not a default or a rounded placeholder.
    assert evidence["similarity"] == pytest.approx(_cosine([1.0, 0.0], [0.96, 0.28]))
    assert evidence["threshold"] == THRESHOLD


def test_an_exact_merge_records_its_matching_key(db):
    c1, c2 = _chunks(db, 2)
    _entity(db, "Platform Team", "team", {c1: 0.9})
    _entity(db, "Platform Team.", "team", {c2: 0.7}, normalized_name="platform team.")

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)

    record = db.get(EntityMerge, report.merges[0].merge_id)
    assert record.method == "exact"
    assert record.model is None
    assert record.evidence["stage"] == "exact"
    assert record.evidence["normalized_name"] == "platform team"
    assert record.evidence["merged"]["name"] == "Platform Team."


def test_the_survivor_is_chosen_deterministically(db):
    c1, c2, c3 = _chunks(db, 3)
    # Higher confidence wins over more sources and over a lower id.
    low = _entity(db, "Bob Sharma", "person", {c1: 0.6, c2: 0.6})
    high = _entity(db, "Robert Sharma", "person", {c3: 0.9}, aliases=["Bob Sharma"])

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)

    assert report.merges[0].survivor_id == high.id
    assert report.merges[0].merged_id == low.id


def test_the_survivor_tie_breaks_on_sources_then_lower_id(db):
    c1, c2, c3, c4 = _chunks(db, 4)
    first = _entity(db, "Bob Sharma", "person", {c1: 0.8})
    second = _entity(db, "Robert Sharma", "person", {c2: 0.8, c3: 0.5}, aliases=["Bob Sharma"])
    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)
    assert report.merges[0].survivor_id == second.id

    third = _entity(db, "Ann Rao", "person", {c4: 0.7})
    fourth = _entity(db, "Annie Rao", "person", {c4: 0.7}, aliases=["Ann Rao"])
    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)
    assert report.merges[0].survivor_id == third.id
    assert report.merges[0].merged_id == fourth.id
    assert db.get(Entity, first.id) is None


# Merge mechanics


def test_a_merge_repoints_folds_and_drops_self_loops(db):
    c1, c2, c3 = _chunks(db, 3)
    survivor = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    merged = _entity(db, "Bob Sharma", "person", {c2: 0.7, c3: 0.5})
    project = _entity(db, "Apollo", "project", {c1: 0.9, c2: 0.9})
    team = _entity(db, "Platform", "team", {c1: 0.9, c2: 0.9})
    works_on = _edge(db, merged, project, "WORKS_ON", {c2: 0.8})
    survivor_member = _edge(db, survivor, team, "MEMBER_OF", {c1: 0.6})
    merged_member = _edge(db, merged, team, "MEMBER_OF", {c2: 0.95})
    self_loop = _edge(db, survivor, merged, "RELATED_TO", {c3: 0.7})

    resolve_entities(db, None, similarity_threshold=THRESHOLD)

    assert db.get(Entity, merged.id) is None
    assert works_on.source_entity_id == survivor.id
    # The two MEMBER_OF edges fold into the survivor's, with both chunks.
    assert db.get(Relationship, merged_member.id) is None
    assert _edge_sources(db, survivor_member.id) == {c1: 0.6, c2: 0.95}
    assert survivor_member.confidence == 0.95
    # A RELATED_TO between the two would become a self-loop, so it is dropped.
    assert db.get(Relationship, self_loop.id) is None
    assert {edge.id for edge in _edges(db)} == {works_on.id, survivor_member.id}
    assert survivor.confidence == 0.9


def test_a_merge_of_unmeasured_entities_invents_no_confidence(db):
    c1, c2, c3 = _chunks(db, 3)
    survivor = _entity(db, "Robert Sharma", "person", {c1: None, c2: None}, aliases=["Bob Sharma"])
    _entity(db, "Bob Sharma", "person", {c3: None})

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)

    # Neither side was measured: the survivor (more sources) stays unmeasured.
    assert report.merges[0].survivor_id == survivor.id
    assert survivor.confidence is None
    assert survivor.extraction_model is None


def test_a_merge_keeps_the_merged_entitys_own_merge_history(db):
    c1, c2, c3 = _chunks(db, 3)
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.7}, aliases=["Bobby"])
    bobby = _entity(db, "Bobby", "person", {c3: 0.5})

    # Scoped to Bobby: Bobby merges into Bob, and Bob (now in scope as the
    # survivor) then merges into Robert.
    report = resolve_entities(db, None, similarity_threshold=THRESHOLD, entity_ids=[bobby.id])

    assert [(m.survivor_id, m.merged_id) for m in report.merges] == [
        (bob.id, bobby.id),
        (robert.id, bob.id),
    ]
    # The Bobby record is not lost to the cascade when Bob is merged away:
    # it moves with Bob's sources to Robert.
    inner = db.get(EntityMerge, report.merges[0].merge_id)
    assert inner is not None
    assert inner.surviving_entity_id == robert.id
    assert _entity_sources(db, robert.id) == {c1: 0.9, c2: 0.7, c3: 0.5}


def test_entity_ids_restrict_resolution_to_pairs_touching_them(db):
    c1, c2, c3, c4 = _chunks(db, 4)
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.8})
    ann = _entity(db, "Ann Rao", "person", {c3: 0.9}, aliases=["Annie Rao"])
    _entity(db, "Annie Rao", "person", {c4: 0.8})

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD, entity_ids=[bob.id])

    assert [(m.survivor_id, m.merged_id) for m in report.merges] == [(robert.id, bob.id)]
    assert len(_entities(db)) == 3
    assert db.get(Entity, ann.id) is not None


def test_entity_ids_restrict_embedding_pairs_to_ones_touching_them(db):
    """Stage 3 embeds the in-scope entity's whole type group, but only pairs
    with an in-scope member may merge: two similar out-of-scope entities stay."""
    c1, c2, c3 = _chunks(db, 3)
    ada = _entity(db, "Ada Lovelace", "person", {c1: 0.9})
    grace = _entity(db, "Grace Hopper", "person", {c2: 0.9})
    admiral = _entity(db, "Admiral Hopper", "person", {c3: 0.9})
    embedder = ScriptedEmbedder(
        {"Ada Lovelace": [1.0, 0.0], "Grace Hopper": [0.0, 1.0], "Admiral Hopper": [0.1, 0.995]}
    )

    report = resolve_entities(db, embedder, similarity_threshold=THRESHOLD, entity_ids=[ada.id])
    assert report.merges == [] and report.embedding_calls == 1
    assert {e.id for e in _entities(db)} == {ada.id, grace.id, admiral.id}

    report = resolve_entities(db, embedder, similarity_threshold=THRESHOLD)
    assert [(m.survivor_id, m.merged_id) for m in report.merges] != []


def test_entities_of_different_types_never_merge(db):
    c1, c2, c3, c4 = _chunks(db, 4)
    # Same name, alias crossover and identical vectors: every stage would
    # merge these if type were ignored.
    _entity(db, "Jordan", "person", {c1: 0.9}, aliases=["Jordan River"])
    _entity(db, "Jordan", "location", {c2: 0.9})
    _entity(db, "Jordan River", "location", {c3: 0.8})
    _entity(db, "Jordan River Project", "project", {c4: 0.8}, aliases=["Jordan"])
    # Same-type pairs (the two locations) get orthogonal vectors; every
    # cross-type pair is identical, so only a type check keeps them apart.
    embedder = ScriptedEmbedder(
        {
            "Jordan": [1.0, 0.0],
            "Jordan River": [0.0, 1.0],
            "Jordan River Project": [1.0, 0.0],
        }
    )

    report = resolve_entities(db, embedder, similarity_threshold=0.5)

    assert report.merges == []
    assert len(_entities(db)) == 4


def test_a_threshold_outside_zero_to_one_is_refused(db):
    with pytest.raises(ValueError):
        resolve_entities(db, None, similarity_threshold=1.5)


# Unmerge


def test_unmerging_restores_the_entity_and_its_edges(db):
    c1, c2, c3 = _chunks(db, 3)
    survivor = _entity(db, "Robert Sharma", "person", {c1: 0.9})
    merged = _entity(
        db,
        "Bob Sharma",
        "person",
        {c2: 0.7, c3: 0.5},
        aliases=["Robert Sharma"],
        description="On call engineer",
    )
    project = _entity(db, "Apollo", "project", {c1: 0.9, c2: 0.9})
    team = _entity(db, "Platform", "team", {c1: 0.9, c2: 0.9})
    works_on = _edge(db, merged, project, "WORKS_ON", {c2: 0.8})
    survivor_member = _edge(db, survivor, team, "MEMBER_OF", {c1: 0.6})
    _edge(db, merged, team, "MEMBER_OF", {c2: 0.95})
    _edge(db, survivor, merged, "RELATED_TO", {c3: 0.7})

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)
    merge_id = report.merges[0].merge_id
    assert db.get(EntityMerge, merge_id).evidence["alias_of"] == "merged"
    assert survivor.aliases == ["Bob Sharma"]

    result = unmerge(db, merge_id)

    restored = db.get(Entity, result.restored_entity_id)
    assert restored.name == "Bob Sharma"
    assert restored.entity_type == "person"
    assert restored.aliases == ["Robert Sharma"]
    assert restored.description == "On call engineer"
    assert restored.normalized_name == "bob sharma"
    # Exactly its own chunks come back, each with the confidence it reported.
    assert _entity_sources(db, restored.id) == {c2: 0.7, c3: 0.5}
    assert _entity_sources(db, survivor.id) == {c1: 0.9}
    assert restored.confidence == 0.7
    assert survivor.confidence == 0.9
    # The merged name the merge added is taken back off the survivor.
    assert survivor.aliases == []
    # An edge sourced only by its chunks moves back.
    assert works_on.source_entity_id == restored.id
    assert works_on.id in result.moved_relationship_ids
    # The folded MEMBER_OF edge splits back out (R26): the survivor's edge
    # keeps only its own report, and the merged entity's edge is recreated.
    assert survivor_member.source_entity_id == survivor.id
    assert result.shared_relationship_ids == [survivor_member.id]
    assert _edge_sources(db, survivor_member.id) == {c1: 0.6}
    assert survivor_member.confidence == 0.6
    member = [
        edge
        for edge in _edges(db)
        if edge.relation_type == "MEMBER_OF" and edge.source_entity_id == restored.id
    ]
    assert len(member) == 1
    assert member[0].target_entity_id == team.id
    assert _edge_sources(db, member[0].id) == {c2: 0.95}
    assert member[0].id in result.restored_relationship_ids
    # The self-loop the merge dropped comes back between the two entities.
    related = [edge for edge in _edges(db) if edge.relation_type == "RELATED_TO"]
    assert len(related) == 1
    assert (related[0].source_entity_id, related[0].target_entity_id) == (
        survivor.id,
        restored.id,
    )
    assert _edge_sources(db, related[0].id) == {c3: 0.7}
    assert related[0].id in result.restored_relationship_ids
    # The merge record is gone: the decision it recorded has been reversed.
    assert db.get(EntityMerge, merge_id) is None


def test_unmerging_a_chunk_both_entities_came_from_restores_both_confidences(db):
    c1, c2, c3 = _chunks(db, 3)
    survivor = _entity(db, "Robert Sharma", "person", {c1: 0.6, c3: 0.95}, aliases=["Bob Sharma"])
    _entity(db, "Bob Sharma", "person", {c1: 0.9, c2: 0.4})

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)
    assert report.merges[0].survivor_id == survivor.id
    # One chunk named both: after the merge it reports the higher of the two.
    assert _entity_sources(db, survivor.id) == {c1: 0.9, c2: 0.4, c3: 0.95}

    result = unmerge(db, report.merges[0].merge_id)

    assert _entity_sources(db, survivor.id) == {c1: 0.6, c3: 0.95}
    assert _entity_sources(db, result.restored_entity_id) == {c1: 0.9, c2: 0.4}
    assert db.get(Entity, result.restored_entity_id).confidence == 0.9
    assert survivor.confidence == 0.95


def _spellings(db, entity_id: int) -> dict[int, str]:
    return dict(
        db.execute(
            select(EntitySource.chunk_id, EntitySource.surface_name).where(
                EntitySource.entity_id == entity_id
            )
        ).all()
    )


def test_a_merge_moves_each_spelling_with_its_row_and_unmerge_restores_it(db):
    """R40: the spelling is part of what a chunk reported. On a chunk that named
    both, the survivor's row keeps the higher report and that report's spelling,
    and the unmerge hands both spellings back."""
    c1, c2, c3 = _chunks(db, 3)
    survivor = _entity(db, "Robert Sharma", "person", {c1: 0.6, c3: 0.95}, aliases=["Bob Sharma"])
    _entity(db, "Bob Sharma", "person", {c1: 0.9, c2: 0.4})

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)
    assert report.merges[0].survivor_id == survivor.id
    assert _spellings(db, survivor.id) == {
        c1: "Bob Sharma",
        c2: "Bob Sharma",
        c3: "Robert Sharma",
    }

    result = unmerge(db, report.merges[0].merge_id)

    assert _spellings(db, survivor.id) == {c1: "Robert Sharma", c3: "Robert Sharma"}
    assert _spellings(db, result.restored_entity_id) == {c1: "Bob Sharma", c2: "Bob Sharma"}


def test_unmerging_hands_back_the_merges_the_restored_entity_had_absorbed(db):
    c1, c2, c3 = _chunks(db, 3)
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    _entity(db, "Bob Sharma", "person", {c2: 0.7}, aliases=["Bobby"])
    bobby = _entity(db, "Bobby", "person", {c3: 0.5})
    report = resolve_entities(db, None, similarity_threshold=THRESHOLD, entity_ids=[bobby.id])
    inner, outer = report.merges

    restored_bob = unmerge(db, outer.merge_id)

    # Bob comes back with every chunk he held when he was merged, Bobby's
    # included, and the Bobby record points at him again.
    assert _entity_sources(db, restored_bob.restored_entity_id) == {c2: 0.7, c3: 0.5}
    assert _entity_sources(db, robert.id) == {c1: 0.9}
    record = db.get(EntityMerge, inner.merge_id)
    assert record.surviving_entity_id == restored_bob.restored_entity_id

    restored_bobby = unmerge(db, inner.merge_id)

    assert _entity_sources(db, restored_bobby.restored_entity_id) == {c3: 0.5}
    assert _entity_sources(db, restored_bob.restored_entity_id) == {c2: 0.7}
    assert len(_entities(db)) == 3


def _snapshot(db) -> dict:
    """The whole graph by name, so it compares across a change of ids."""
    names = {entity.id: entity.name for entity in _entities(db)}
    entities = {}
    for entity in _entities(db):
        rows = db.execute(select(EntitySource).where(EntitySource.entity_id == entity.id)).scalars()
        entities[(entity.name, entity.entity_type)] = (
            entity.normalized_name,
            entity.description,
            tuple(entity.aliases),
            entity.confidence,
            entity.extraction_model,
            frozenset((row.chunk_id, row.confidence, row.extraction_model) for row in rows),
        )
    edges = {}
    for edge in _edges(db):
        rows = db.execute(
            select(RelationshipSource).where(RelationshipSource.relationship_id == edge.id)
        ).scalars()
        key = (names[edge.source_entity_id], names[edge.target_entity_id], edge.relation_type)
        assert key not in edges
        edges[key] = (
            edge.description,
            edge.weight,
            edge.confidence,
            edge.extraction_model,
            frozenset((row.chunk_id, row.confidence, row.extraction_model) for row in rows),
        )
    return {"entities": entities, "edges": edges}


def test_merge_then_unmerge_gives_back_the_graph_it_started_from(db):
    c1, c2, c3 = _chunks(db, 3)
    # c1 names both Robert and Bob, which is exactly where provenance alone
    # cannot say whose edge is whose.
    survivor = _entity(db, "Robert Sharma", "person", {c1: 0.6, c3: 0.95}, aliases=["Bob Sharma"])
    merged = _entity(
        db,
        "Bob Sharma",
        "person",
        {c1: 0.9, c2: 0.5},
        aliases=["Bobby"],
        description="On call engineer",
    )
    project = _entity(db, "Apollo", "project", {c1: 0.9})
    team = _entity(db, "Platform", "team", {c1: 0.9, c2: 0.8})
    city = _entity(db, "Pune", "location", {c2: 0.8})
    wiki = _entity(db, "Runbook", "document", {c2: 0.8})
    # The reviewer's probe: Robert's own edge, sourced only by the shared
    # chunk. The merge never touches it, so the unmerge must not either.
    own_edge = _edge(db, survivor, project, "WORKS_ON", {c1: 0.7})
    # Folded on the shared chunk: Bob's 0.9 overwrites Robert's 0.6 on c1.
    _edge(db, survivor, team, "MEMBER_OF", {c1: 0.6, c3: 0.8})
    _edge(db, merged, team, "MEMBER_OF", {c1: 0.9, c2: 0.5})
    # Repointed on either end.
    _edge(db, merged, city, "LOCATED_IN", {c2: 0.7})
    _edge(db, wiki, merged, "MENTIONS", {c2: 0.6})
    # Would become a self-loop.
    _edge(db, survivor, merged, "RELATED_TO", {c1: 0.5})
    db.flush()
    before = _snapshot(db)

    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)
    assert [(m.survivor_id, m.merged_id) for m in report.merges] == [(survivor.id, merged.id)]
    during = _snapshot(db)
    assert ("Bob Sharma", "person") not in during["entities"]
    assert during["edges"][("Robert Sharma", "Platform", "MEMBER_OF")][-1] == frozenset(
        {(c1, 0.9, "test-extractor"), (c2, 0.5, "test-extractor"), (c3, 0.8, "test-extractor")}
    )
    # R27: Bob's own alias stays matchable on the survivor.
    assert survivor.aliases == ["Bob Sharma", "Bobby"]

    result = unmerge(db, report.merges[0].merge_id)

    assert _snapshot(db) == before
    assert own_edge.source_entity_id == survivor.id
    assert own_edge.id not in result.moved_relationship_ids
    assert own_edge.id not in result.shared_relationship_ids


def test_an_absorbed_alias_matches_a_later_entity(db):
    c1, c2, c3 = _chunks(db, 3)
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    _entity(db, "Bob Sharma", "person", {c2: 0.8}, aliases=["Bobby"])
    first = resolve_entities(db, None, similarity_threshold=THRESHOLD)
    assert len(first.merges) == 1
    assert robert.aliases == ["Bob Sharma", "Bobby"]
    record = db.get(EntityMerge, first.merges[0].merge_id)
    assert record.evidence["restore"]["aliases_appended"] == ["Bobby"]

    # A later extraction names Bobby on his own.
    bobby = _entity(db, "Bobby", "person", {c3: 0.5})
    second = resolve_entities(db, None, similarity_threshold=THRESHOLD, entity_ids=[bobby.id])

    assert [(m.survivor_id, m.merged_id, m.method) for m in second.merges] == [
        (robert.id, bobby.id, "alias")
    ]
    evidence = db.get(EntityMerge, second.merges[0].merge_id).evidence
    assert (evidence["alias"], evidence["alias_of"]) == ("Bobby", "survivor")


def test_stage_three_does_not_chain_when_the_middle_entity_survives(db):
    c1, c2, c3 = _chunks(db, 3)
    # Same geometry as the chaining test, but B is the higher-confidence
    # survivor, so B stays live after absorbing A. A and C never met the
    # threshold with each other, so B must not take C in the same pass.
    a = _entity(db, "A", "product", {c1: 0.7})
    b = _entity(db, "B", "product", {c2: 0.9})
    c = _entity(db, "C", "product", {c3: 0.6})
    s = math.sqrt(0.5)
    embedder = ScriptedEmbedder({"A": [1.0, 0.0], "B": [0.95, 0.312], "C": [s, s]})

    report = resolve_entities(db, embedder, similarity_threshold=0.8)

    assert [(m.survivor_id, m.merged_id) for m in report.merges] == [(b.id, a.id)]
    assert [entity.id for entity in _entities(db)] == [b.id, c.id]


def _merge_ids(db) -> list[int]:
    return list(db.execute(select(EntityMerge.id).order_by(EntityMerge.id)).scalars())


def _assert_blocked(db, merge_id: int, blocking: list[int]) -> None:
    """``unmerge`` refuses, names the blockers, and writes nothing."""
    graph, merges = _snapshot(db), _merge_ids(db)
    with pytest.raises(UnmergeBlocked) as refused:
        unmerge(db, merge_id)
    assert refused.value.merge_id == merge_id
    assert refused.value.blocking_merge_ids == blocking
    db.flush()
    assert _snapshot(db) == graph
    assert _merge_ids(db) == merges


def test_unmerge_is_blocked_while_a_later_merge_folded_its_edge_away(db):
    c1, c2, c3 = _chunks(db, 3)
    # The reviewer's fold-away probe. M1: Bob absorbs Bobby, whose edge to
    # Apollo is repointed onto Bob. M2: Robert absorbs Bob, and that edge
    # folds into Robert's own edge to Apollo. Undoing M1 first would find
    # its edge gone and bring Bobby back without it.
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    _entity(db, "Bob Sharma", "person", {c2: 0.7}, aliases=["Bobby"])
    bobby = _entity(db, "Bobby", "person", {c3: 0.5})
    apollo = _entity(db, "Apollo", "project", {c1: 0.9, c3: 0.9})
    _edge(db, bobby, apollo, "WORKS_ON", {c3: 0.8})
    _edge(db, robert, apollo, "WORKS_ON", {c1: 0.6})
    before = _snapshot(db)
    report = resolve_entities(db, None, similarity_threshold=THRESHOLD, entity_ids=[bobby.id])
    first, second = report.merges
    assert (first.merged_id, second.survivor_id) == (bobby.id, robert.id)

    _assert_blocked(db, first.merge_id, [second.merge_id])

    unmerge(db, second.merge_id)
    unmerge(db, first.merge_id)
    assert _snapshot(db) == before


def test_unmerge_is_blocked_while_a_later_merge_shares_its_survivor(db):
    c1, c2, c3 = _chunks(db, 3)
    # The reviewer's alias probe. M1: Robert absorbs Bob Sharma, gaining
    # Bob's alias Quill. M2: Robert absorbs an entity named Quill, which that
    # alias justified. Undoing M1 first would strip the alias M2 relies on.
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    _entity(db, "Bob Sharma", "person", {c2: 0.7}, aliases=["Quill"])
    _entity(db, "Quill", "person", {c3: 0.5})
    before = _snapshot(db)
    report = resolve_entities(db, None, similarity_threshold=THRESHOLD)
    first, second = report.merges
    assert first.survivor_id == second.survivor_id == robert.id
    assert robert.aliases == ["Bob Sharma", "Quill"]

    _assert_blocked(db, first.merge_id, [second.merge_id])
    assert robert.aliases == ["Bob Sharma", "Quill"]

    unmerge(db, second.merge_id)
    unmerge(db, first.merge_id)
    assert _snapshot(db) == before


def test_unmerge_is_blocked_while_a_later_merge_moved_the_far_end_of_its_edge(db):
    c1, c2, c3 = _chunks(db, 3)
    # M1 repoints Bob's edge to Apollo onto Robert; M2 then merges Apollo
    # Program into Apollo, the far end of that edge. Under the global rule
    # (R30) any later merge blocks, and this one also genuinely overlaps.
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.7})
    apollo = _entity(db, "Apollo", "project", {c2: 0.9}, aliases=["Apollo Program"])
    program = _entity(db, "Apollo Program", "project", {c3: 0.8})
    _edge(db, bob, apollo, "WORKS_ON", {c2: 0.8})
    _edge(db, robert, program, "WORKS_ON", {c3: 0.4})
    before = _snapshot(db)
    (first,) = resolve_entities(
        db, None, similarity_threshold=THRESHOLD, entity_ids=[bob.id]
    ).merges
    (second,) = resolve_entities(
        db, None, similarity_threshold=THRESHOLD, entity_ids=[program.id]
    ).merges
    assert (second.survivor_id, second.merged_id) == (apollo.id, program.id)
    # Robert's edges to Apollo and to Apollo Program folded together.
    assert len([e for e in _edges(db) if e.relation_type == "WORKS_ON"]) == 1

    _assert_blocked(db, first.merge_id, [second.merge_id])

    unmerge(db, second.merge_id)
    unmerge(db, first.merge_id)
    assert _snapshot(db) == before


def test_a_far_end_entity_recreated_by_an_unmerge_cannot_bypass_the_block(db):
    c1, c2, c3, c4 = _chunks(db, 4)
    # The reviewer's sequence. M1: Robert absorbs Bob, whose edge to Apollo
    # Program is repointed onto Robert. M2: Apollo absorbs Apollo Program and
    # the edge folds into Robert's own edge to Apollo. Undo M2 (in order):
    # Apollo Program comes back under a new id. M3: Apollo absorbs it again,
    # folding the edge again. An entity-scoped block keyed on the old id let
    # M1 through here; the global rule does not.
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.7})
    apollo = _entity(db, "Apollo", "project", {c4: 0.9}, aliases=["Apollo Program"])
    program = _entity(db, "Apollo Program", "project", {c3: 0.8})
    _edge(db, bob, program, "WORKS_ON", {c2: 0.8})
    _edge(db, robert, apollo, "WORKS_ON", {c4: 0.5})
    before = _snapshot(db)

    (m1,) = resolve_entities(db, None, similarity_threshold=THRESHOLD, entity_ids=[bob.id]).merges
    (m2,) = resolve_entities(
        db, None, similarity_threshold=THRESHOLD, entity_ids=[program.id]
    ).merges
    # A new id on PostgreSQL. SQLite may hand back the freed id, which is
    # what masked the entity-scoped bug on SQLite.
    recreated = unmerge(db, m2.merge_id).restored_entity_id
    (m3,) = resolve_entities(
        db, None, similarity_threshold=THRESHOLD, entity_ids=[recreated]
    ).merges
    assert (m3.survivor_id, m3.merged_id) == (apollo.id, recreated)
    assert m3.merge_id > m1.merge_id

    _assert_blocked(db, m1.merge_id, [m3.merge_id])

    unmerge(db, m3.merge_id)
    result = unmerge(db, m1.merge_id)
    assert result.unrestored_relationship_ids == []
    assert _snapshot(db) == before


def test_merges_of_unrelated_entities_still_block_out_of_order(db):
    c1, c2, c3, c4 = _chunks(db, 4)
    # The intended cost of the global rule (R30): these two merges share no
    # entity and no edge, and the older one still cannot be undone first.
    _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.7})
    _entity(db, "Apollo", "project", {c3: 0.9}, aliases=["Apollo Program"])
    program = _entity(db, "Apollo Program", "project", {c4: 0.8})
    (first,) = resolve_entities(
        db, None, similarity_threshold=THRESHOLD, entity_ids=[bob.id]
    ).merges
    (second,) = resolve_entities(
        db, None, similarity_threshold=THRESHOLD, entity_ids=[program.id]
    ).merges

    _assert_blocked(db, first.merge_id, [second.merge_id])


def test_the_most_recent_merge_is_always_undoable(db):
    chunk_ids = _chunks(db, 6)
    pairs = [("Robert Sharma", "Bob Sharma"), ("Apollo", "Apollo Program"), ("Ann Rao", "Annie")]
    for index, (keep, alias) in enumerate(pairs):
        _entity(db, keep, "person", {chunk_ids[2 * index]: 0.9}, aliases=[alias])
        _entity(db, alias, "person", {chunk_ids[2 * index + 1]: 0.5})
    before = _snapshot(db)
    merges = resolve_entities(db, None, similarity_threshold=THRESHOLD).merges
    ids = [merge.merge_id for merge in merges]
    assert len(ids) == 3

    # The oldest is blocked by every newer live merge, listed oldest first.
    _assert_blocked(db, ids[0], ids[1:])
    for merge_id in reversed(ids):
        unmerge(db, merge_id)

    assert _merge_ids(db) == []
    assert _snapshot(db) == before


def test_lifo_unmerge_follows_an_edge_a_later_merge_dropped_as_a_self_loop(db):
    c1, c2, c3 = _chunks(db, 3)
    # M1: Bob absorbs Bobby, and Bobby's edge to Robert becomes Bob's. M2:
    # Robert absorbs Bob, so that edge would be Robert to Robert and is
    # dropped. Undoing M2 recreates it under a new id, and M1 must follow it.
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    _entity(db, "Bob Sharma", "person", {c2: 0.7}, aliases=["Bobby"])
    bobby = _entity(db, "Bobby", "person", {c3: 0.5})
    _edge(db, bobby, robert, "REPORTS_TO", {c3: 0.6})
    before = _snapshot(db)
    first, second = resolve_entities(
        db, None, similarity_threshold=THRESHOLD, entity_ids=[bobby.id]
    ).merges
    assert _edges(db) == []

    unmerge(db, second.merge_id)
    result = unmerge(db, first.merge_id)

    assert result.unrestored_relationship_ids == []
    assert _snapshot(db) == before


def test_a_report_added_to_a_moved_edge_after_the_merge_stays_with_the_survivor(db):
    c1, c2, c3 = _chunks(db, 3)
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.7})
    apollo = _entity(db, "Apollo", "project", {c1: 0.9})
    edge = _edge(db, bob, apollo, "WORKS_ON", {c2: 0.8})
    (merge,) = resolve_entities(db, None, similarity_threshold=THRESHOLD).merges
    # A later extraction of c3 reports Robert works on Apollo; the upsert
    # finds the repointed edge and links c3 to it.
    db.add(
        RelationshipSource(
            relationship_id=edge.id, chunk_id=c3, confidence=0.4, extraction_model="later"
        )
    )
    db.flush()

    result = unmerge(db, merge.merge_id)

    assert edge.source_entity_id == result.restored_entity_id
    assert _edge_sources(db, edge.id) == {c2: 0.8}
    (split,) = [e for e in _edges(db) if e.source_entity_id == robert.id]
    assert split.target_entity_id == apollo.id
    assert _edge_sources(db, split.id) == {c3: 0.4}
    assert split.id in result.shared_relationship_ids
    assert result.unrestored_relationship_ids == []


def test_a_recorded_edge_removed_since_the_merge_is_reported_not_silently_lost(db):
    c1, c2 = _chunks(db, 2)
    _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.7})
    apollo = _entity(db, "Apollo", "project", {c2: 0.9})
    edge = _edge(db, bob, apollo, "WORKS_ON", {c2: 0.8})
    edge_id = edge.id
    (merge,) = resolve_entities(db, None, similarity_threshold=THRESHOLD).merges
    # Re-extraction since the merge garbage collected the edge.
    for row in db.execute(
        select(RelationshipSource).where(RelationshipSource.relationship_id == edge_id)
    ).scalars():
        db.delete(row)
    db.flush()
    db.delete(edge)
    db.flush()

    result = unmerge(db, merge.merge_id)

    assert result.unrestored_relationship_ids == [edge_id]
    assert result.moved_relationship_ids == []
    assert _edges(db) == []


def test_unmerging_an_unknown_merge_raises(db):
    with pytest.raises(LookupError):
        unmerge(db, 987654)


# R31: an id is not an identity, and ownership follows the text there now.


def _extract(db, chunk_id: int, text: str, entities: list, relationships: list) -> None:
    """Run the real extraction path on one chunk with a scripted model response."""
    chunk = db.get(Chunk, chunk_id)
    chunk.text = text
    provider = ScriptedLLMProvider(
        responses=[json.dumps({"entities": entities, "relationships": relationships})],
        model="test-extractor",
    )
    extract_chunk(db, chunk, provider, floor=0.5, model="test-extractor")


def _person(name: str, confidence: float) -> dict:
    return {"name": name, "entity_type": "person", "confidence": confidence}


def _bob_works_on_apollo(db) -> tuple[list[int], Entity, Relationship, int]:
    """Robert (alias Bob Sharma) from c1; Bob works on Apollo from c2; Bob merged into Robert.

    The Bob to Apollo edge is the only edge, so on SQLite without
    AUTOINCREMENT the next edge inserted after it is deleted takes its id.
    """
    chunk_ids = _chunks(db, 3)
    c1, c2, _ = chunk_ids
    _extract(db, c1, "Robert Sharma leads.", [_person("Robert Sharma", 0.9)], [])
    _extract(
        db,
        c2,
        "Bob Sharma works on Apollo.",
        [
            _person("Bob Sharma", 0.7),
            {"name": "Apollo", "entity_type": "project", "confidence": 0.9},
        ],
        [
            {
                "source": "Bob Sharma",
                "target": "Apollo",
                "relation_type": "WORKS_ON",
                "confidence": 0.8,
            }
        ],
    )
    robert = db.execute(select(Entity).where(Entity.name == "Robert Sharma")).scalar_one()
    robert.aliases = ["Bob Sharma"]
    db.flush()
    (edge,) = _edges(db)
    (merge,) = resolve_entities(db, None, similarity_threshold=THRESHOLD).merges
    assert merge.survivor_id == robert.id
    assert edge.source_entity_id == robert.id
    return chunk_ids, robert, edge, merge.merge_id


def test_a_merge_records_each_edges_far_end_and_each_chunks_text_hash(db):
    (c1, c2, _), _, edge, merge_id = _bob_works_on_apollo(db)
    restore = db.get(EntityMerge, merge_id).evidence["restore"]
    assert restore["chunk_hashes"] == {str(c2): db.get(Chunk, c2).extraction_hash}
    assert restore["chunk_hashes"][str(c2)] is not None
    assert restore["relationships"][0]["far_end_id"] == edge.target_entity_id


def test_an_edge_that_took_a_recorded_edges_id_is_not_moved(db):
    # Probe 1. c2 is re-extracted to nothing, so extraction garbage collects
    # the recorded edge. A new chunk then adds Robert REPORTS_TO Ann, which
    # SQLite used to give the freed id. The unmerge must not take it for
    # Bob's WORKS_ON edge, and must report that edge as lost.
    (c1, c2, c4), robert, edge, merge_id = _bob_works_on_apollo(db)
    edge_id = edge.id
    _extract(db, c2, "Nothing to see here.", [], [])
    assert db.get(Relationship, edge_id) is None
    _extract(
        db,
        c4,
        "Robert Sharma reports to Ann Rao.",
        [_person("Robert Sharma", 0.9), _person("Ann Rao", 0.8)],
        [
            {
                "source": "Robert Sharma",
                "target": "Ann Rao",
                "relation_type": "REPORTS_TO",
                "confidence": 0.85,
            }
        ],
    )
    # On SQLite before R31(b) this new edge took the freed id; the unmerge
    # below must be right either way (R31(a)).
    assert len(_edges(db)) == 1
    before = _snapshot(db)

    result = unmerge(db, merge_id)

    assert result.moved_relationship_ids == []
    assert result.unrestored_relationship_ids == [edge_id]
    assert result.changed_chunk_ids == [c2]
    assert result.restored_without_sources is True
    # Robert's REPORTS_TO edge is untouched and not split.
    graph = _snapshot(db)
    assert graph["edges"] == before["edges"]
    assert (
        graph["entities"][("Robert Sharma", "person")]
        == before["entities"][("Robert Sharma", "person")]
    )
    assert _entity_sources(db, result.restored_entity_id) == {}


def test_a_chunk_re_extracted_under_the_survivors_name_stays_with_the_survivor(db):
    # Probe 2. c2 now says Robert works on Apollo; extraction keeps the edge
    # id and links c2 to it again. That is Robert's claim from today's text,
    # so the unmerge must leave it, and Robert's c2 source, with Robert.
    (c1, c2, _), robert, edge, merge_id = _bob_works_on_apollo(db)
    _extract(
        db,
        c2,
        "Robert Sharma works on Apollo.",
        [
            _person("Robert Sharma", 0.85),
            {"name": "Apollo", "entity_type": "project", "confidence": 0.9},
        ],
        [
            {
                "source": "Robert Sharma",
                "target": "Apollo",
                "relation_type": "WORKS_ON",
                "confidence": 0.95,
            }
        ],
    )
    assert edge.source_entity_id == robert.id
    assert _edge_sources(db, edge.id) == {c2: 0.95}
    before = _snapshot(db)

    result = unmerge(db, merge_id)

    assert result.moved_relationship_ids == []
    assert result.unrestored_relationship_ids == [edge.id]
    assert result.changed_chunk_ids == [c2]
    assert result.restored_without_sources is True
    assert edge.source_entity_id == robert.id
    assert _edge_sources(db, edge.id) == {c2: 0.95}
    assert _entity_sources(db, robert.id) == {c1: 0.9, c2: 0.85}
    assert _entity_sources(db, result.restored_entity_id) == {}
    graph = _snapshot(db)
    assert graph["edges"] == before["edges"]
    assert (
        graph["entities"][("Robert Sharma", "person")]
        == before["entities"][("Robert Sharma", "person")]
    )


def test_an_unchanged_chunk_still_moves_back_beside_a_changed_one(db):
    # Bob came from c2 and c3; only c3 was re-extracted since the merge. c2
    # and its edge go back to Bob, c3's new report stays with Robert.
    c1, c2, c3 = _chunks(db, 3)
    _extract(db, c1, "Robert Sharma leads.", [_person("Robert Sharma", 0.9)], [])
    works_on = {
        "source": "Bob Sharma",
        "target": "Apollo",
        "relation_type": "WORKS_ON",
        "confidence": 0.8,
    }
    apollo = {"name": "Apollo", "entity_type": "project", "confidence": 0.9}
    _extract(
        db, c2, "Bob Sharma works on Apollo.", [_person("Bob Sharma", 0.7), apollo], [works_on]
    )
    _extract(db, c3, "Bob Sharma again.", [_person("Bob Sharma", 0.6), apollo], [works_on])
    robert = db.execute(select(Entity).where(Entity.name == "Robert Sharma")).scalar_one()
    robert.aliases = ["Bob Sharma"]
    db.flush()
    (edge,) = _edges(db)
    (merge,) = resolve_entities(db, None, similarity_threshold=THRESHOLD).merges
    _extract(
        db,
        c3,
        "Robert Sharma works on Apollo too.",
        [_person("Robert Sharma", 0.65), apollo],
        [{**works_on, "source": "Robert Sharma", "confidence": 0.75}],
    )

    result = unmerge(db, merge.merge_id)

    assert result.changed_chunk_ids == [c3]
    assert result.restored_without_sources is False
    assert _entity_sources(db, result.restored_entity_id) == {c2: 0.7}
    assert _entity_sources(db, robert.id) == {c1: 0.9, c3: 0.65}
    assert result.moved_relationship_ids == [edge.id]
    assert edge.source_entity_id == result.restored_entity_id
    assert _edge_sources(db, edge.id) == {c2: 0.8}
    (split,) = result.shared_relationship_ids
    assert db.get(Relationship, split).source_entity_id == robert.id
    assert _edge_sources(db, split) == {c3: 0.75}


@pytest.mark.parametrize("change", ["relation_type", "far_end", "sources"])
def test_an_edge_that_is_no_longer_the_recorded_edge_is_not_taken(db, change):
    # R31(a) on its own, independent of id reuse: the recorded edge keeps its
    # id but is no longer the same edge. Nothing is taken from it.
    c1, c2, c3 = _chunks(db, 3)
    _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.7})
    apollo = _entity(db, "Apollo", "project", {c2: 0.9})
    other = _entity(db, "Hermes", "project", {c3: 0.9})
    edge = _edge(db, bob, apollo, "WORKS_ON", {c2: 0.8})
    (merge,) = resolve_entities(db, None, similarity_threshold=THRESHOLD).merges
    if change == "relation_type":
        edge.relation_type = "OWNS"
    elif change == "far_end":
        edge.target_entity_id = other.id
    else:
        db.delete(db.get(RelationshipSource, (edge.id, c2)))
        db.add(RelationshipSource(relationship_id=edge.id, chunk_id=c3, confidence=0.5))
    db.flush()
    before = _edge_sources(db, edge.id), edge.source_entity_id, edge.target_entity_id

    result = unmerge(db, merge.merge_id)

    assert result.unrestored_relationship_ids == [edge.id]
    assert result.moved_relationship_ids == []
    assert result.shared_relationship_ids == []
    assert (_edge_sources(db, edge.id), edge.source_entity_id, edge.target_entity_id) == before
    assert len(_edges(db)) == 1
    # The entity itself still comes back with its unchanged chunk.
    assert _entity_sources(db, result.restored_entity_id) == {c2: 0.7}


def test_a_record_follows_only_a_recreated_edge_that_matches_it(db):
    # R31(a) for _retarget_records: an older record's entry follows a new id
    # only when the edge there is the edge it recorded.
    from ragfabric_core.graph.resolve import _retarget_records

    c1, c2 = _chunks(db, 2)
    robert = _entity(db, "Robert Sharma", "person", {c1: 0.9}, aliases=["Bob Sharma"])
    bob = _entity(db, "Bob Sharma", "person", {c2: 0.7})
    apollo = _entity(db, "Apollo", "project", {c2: 0.9})
    edge = _edge(db, bob, apollo, "WORKS_ON", {c2: 0.8})
    (merge,) = resolve_entities(db, None, similarity_threshold=THRESHOLD).merges

    def recorded_id() -> int:
        db.flush()
        db.expire_all()
        return db.get(EntityMerge, merge.merge_id).evidence["restore"]["relationships"][0][
            "relationship_id"
        ]

    wrong_type = _edge(db, robert, apollo, "OWNS", {c2: 0.8})
    _retarget_records(db, 0, edge.id, wrong_type.id)
    assert recorded_id() == edge.id

    wrong_chunk = _edge(db, robert, apollo, "WORKS_ON", {c1: 0.8})
    _retarget_records(db, 0, edge.id, wrong_chunk.id)
    assert recorded_id() == edge.id

    same = _edge(db, robert, apollo, "WORKS_ON", {c2: 0.8})
    _retarget_records(db, 0, edge.id, same.id)
    assert recorded_id() == same.id


def test_graph_and_merge_ids_are_never_reused(db):
    # R31(b): SQLite's default rowid hands the highest freed id to the next
    # insert; AUTOINCREMENT (and PostgreSQL's sequences) never do.
    (c1,) = _chunks(db, 1)
    a = _entity(db, "A", "person", {c1: 0.9})
    b = _entity(db, "B", "person", {c1: 0.9})
    edge = _edge(db, a, b, "RELATED_TO", {c1: 0.9})
    record = EntityMerge(
        surviving_entity_id=a.id,
        merged_name="X",
        merged_entity_type="person",
        evidence={"stage": "test"},
        method="exact",
    )
    db.add(record)
    db.flush()
    freed = (b.id, edge.id, record.id)
    db.delete(db.get(RelationshipSource, (edge.id, c1)))
    db.flush()
    for row in (record, edge):
        db.delete(row)
        db.flush()
    db.delete(db.get(EntitySource, (b.id, c1)))
    db.flush()
    db.delete(b)
    db.flush()

    c = _entity(db, "C", "person", {c1: 0.9})
    new_edge = _edge(db, a, c, "RELATED_TO", {c1: 0.9})
    new_record = EntityMerge(
        surviving_entity_id=a.id,
        merged_name="Y",
        merged_entity_type="person",
        evidence={"stage": "test"},
        method="exact",
    )
    db.add(new_record)
    db.flush()

    assert (c.id, new_edge.id, new_record.id) > freed
    assert c.id > freed[0] and new_edge.id > freed[1] and new_record.id > freed[2]


def test_unmerge_refuses_when_re_extraction_recreated_the_merged_name(db):
    # Documented ValueError: a re-extraction after the merge named Bob again
    # and the upsert created a fresh Bob row, so the unique key is taken.
    (c1, c2, c4), _, _, merge_id = _bob_works_on_apollo(db)
    _extract(db, c4, "Bob Sharma is back.", [_person("Bob Sharma", 0.6)], [])
    graph = _snapshot(db)

    with pytest.raises(ValueError, match="already holds"):
        unmerge(db, merge_id)

    db.flush()
    assert _snapshot(db) == graph
    assert db.get(EntityMerge, merge_id) is not None
