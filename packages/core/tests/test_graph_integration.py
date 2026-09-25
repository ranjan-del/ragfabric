"""One real graph extraction run, against a local model, over a corpus small enough to read.

**This is a record, not a benchmark.** Per ADR 0004, what this module reports is what
happened on this fixture corpus, with the model named in ``RAGFABRIC_TEST_OLLAMA_MODEL``,
on the day it was run, on one machine. Extraction is the whole risk of this phase: a local
model inventing a relationship, missing one, or getting a direction backwards is a finding
to write down here and in ``docs/learning/graph-extraction-first-run.md``, never a reason to
edit this corpus until the run looks clean.

The offline suite (``test_graph_extract.py``, ``test_graph_resolve.py``,
``test_strategy_graph.py``, ``test_graph_citations.py``) proves every branch of extraction,
resolution, traversal and the citation contract by feeding each one JSON a model is *told*
to produce. This module finds out whether a real small model produces that JSON at all, what
it gets wrong when it tries, and whether the graph built from its output still holds the
invariants those offline tests assume. So the assertions below are about things that must
hold regardless of model quality (every stored row carries a real source chunk and a real
confidence, no self loop, the access filter is respected, a kept answer never cites an edge
that is not in the traversed sub-graph). Everything about the model's own judgement, what it
invented, what it missed, what it got backwards, what resolution merged and did not merge, is
printed rather than asserted: a test that asserted the model extracted every fact correctly
would be a test that fails the moment the finding changes, and the finding is the point.

Run it with::

    RAGFABRIC_TEST_OLLAMA=1 \\
    RAGFABRIC_TEST_DATABASE_URL=postgresql+psycopg://... \\
    uv run pytest packages/core/tests/test_graph_integration.py -s
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.config_file import GraphStoreConfig, RagFabricConfig
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.generate.cited import GraphAnswer, generate_graph_answer
from ragfabric_core.graph.citations import EDGE_MARKER_RE, RelationshipDropReason
from ragfabric_core.graph.contracts import EntityType, RelationType, Subgraph, normalise
from ragfabric_core.graph.extract import ExtractionReport, extract_chunks
from ragfabric_core.graph.resolve import ResolutionReport, resolve_entities
from ragfabric_core.graph.traverse import traverse
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import (
    Entity,
    EntityMerge,
    EntitySource,
    Relationship,
    RelationshipSource,
)
from ragfabric_core.providers.base import EmbeddingProvider, LLMProvider
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider
from ragfabric_core.strategies.base import RetrievalContext, StrategyName, StrategyParams
from ragfabric_core.strategies.registry_defaults import default_registry

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
OLLAMA = os.environ.get("RAGFABRIC_TEST_OLLAMA", "")
BASE_URL = os.environ.get("RAGFABRIC_TEST_OLLAMA_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("RAGFABRIC_TEST_OLLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL = os.environ.get("RAGFABRIC_TEST_OLLAMA_EMBED_MODEL", "nomic-embed-text")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not OLLAMA, reason="needs RAGFABRIC_TEST_OLLAMA"),
    pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL"),
]

# The floor and threshold the brief asks for are the ones GraphStoreConfig ships as its
# defaults (0.5 and 0.9); read from there rather than repeated as bare literals, so this
# module always exercises whatever "the default" currently means.
DEFAULT_FLOOR = GraphStoreConfig().confidence_floor
DEFAULT_THRESHOLD = GraphStoreConfig().similarity_threshold

# ---------------------------------------------------------------------------------------
# Ground truth: written down before anything is run, so a miss and an invention are both
# visible against a fixed target rather than judged by eye against the model's own output.
# ---------------------------------------------------------------------------------------

# Three documents. "profile.txt" holds exactly one fact and nothing else, so it can be
# denied on its own for the access-filter proof without also hiding the rest of the corpus.
CORPUS: dict[str, list[str]] = {
    "profile.txt": [
        "Ravi Sharma is a software engineer on the Platform Team. Ravi Sharma reports to "
        "Meera Iyer.",
    ],
    "team.txt": [
        "R. Sharma presented the migration plan at the weekly Platform Team sync meeting.",
        "Anjali Sharma is a product designer who works on Project Atlas. Anjali Sharma is a "
        "different person from Ravi Sharma, despite sharing a surname.",
        "The Platform Team belongs to the Engineering organisation.",
        "Project Atlas is owned by the Engineering organisation.",
    ],
    "management.txt": [
        "Meera Iyer manages the Platform Team and reports to Kabir Rao, the head of the "
        "Engineering organisation.",
    ],
}

GROUND_TRUTH_ENTITIES: dict[str, EntityType] = {
    "Ravi Sharma": EntityType.PERSON,
    "Anjali Sharma": EntityType.PERSON,
    "Meera Iyer": EntityType.PERSON,
    "Kabir Rao": EntityType.PERSON,
    "Platform Team": EntityType.TEAM,
    "Engineering": EntityType.ORGANISATION,
    "Project Atlas": EntityType.PROJECT,
}

# (source, relation, target), exactly as the corpus states it. Two hops of REPORTS_TO
# (Ravi Sharma -> Meera Iyer -> Kabir Rao), three invertible relation types (MEMBER_OF,
# BELONGS_TO, OWNS).
GROUND_TRUTH_RELATIONSHIPS: list[tuple[str, RelationType, str]] = [
    ("Ravi Sharma", RelationType.REPORTS_TO, "Meera Iyer"),
    ("Meera Iyer", RelationType.REPORTS_TO, "Kabir Rao"),
    ("Ravi Sharma", RelationType.MEMBER_OF, "Platform Team"),
    ("Platform Team", RelationType.BELONGS_TO, "Engineering"),
    ("Anjali Sharma", RelationType.WORKS_ON, "Project Atlas"),
    ("Engineering", RelationType.OWNS, "Project Atlas"),
]

# Same person, two surface spellings: resolution should merge these if it works.
ALIAS_PAIR = ("Ravi Sharma", "R. Sharma")

# Two different people who happen to share a surname: resolution must NOT merge these.
SHARED_SURNAME_PAIR = ("Ravi Sharma", "Anjali Sharma")

QUESTION = "Who does Ravi Sharma report to, and who does that manager report to in turn?"


@dataclass
class GraphRun:
    factory: sessionmaker
    document_ids: dict[str, int]
    extraction_report: ExtractionReport
    resolution_report: ResolutionReport
    llm: LLMProvider
    embedder: EmbeddingProvider


def _config() -> RagFabricConfig:
    return RagFabricConfig(
        llm={"provider": "ollama", "model": MODEL, "base_url": BASE_URL},
        embeddings={
            "provider": "ollama",
            "model": EMBED_MODEL,
            "dim": 768,
            "base_url": BASE_URL,
        },
    )


def _context() -> RetrievalContext:
    return RetrievalContext(
        principal=Principal(user_id=1, email="engineer@example.com"),
        access_filter=AccessFilter.unrestricted(),
        params=StrategyParams(top_k=10),
    )


def _lookup(db: Session, name: str, entity_type: EntityType) -> Entity | None:
    return (
        db.query(Entity)
        .filter(Entity.normalized_name == normalise(name), Entity.entity_type == entity_type.value)
        .one_or_none()
    )


def _run_pipeline() -> GraphRun:
    """Ingest the fixture, then run real extraction and real resolution once.

    A plain function rather than a fixture: everything in this module reads one run, inside
    one test, so there is exactly one skip when ``RAGFABRIC_TEST_OLLAMA`` is unset rather than
    one per assertion group.
    """
    downgrade(URL)
    upgrade(URL)
    engine = create_engine(URL)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    cfg = _config()
    llm = build_llm_provider(cfg.llm)
    embedder = build_embedding_provider(cfg.embeddings)

    document_ids: dict[str, int] = {}
    with factory() as db:
        collection = Collection(name="graph-fixture")
        db.add(collection)
        db.flush()
        all_chunks: list[Chunk] = []
        for filename, texts in CORPUS.items():
            document = Document(
                filename=filename, format="txt", collection_id=collection.id, status="processing"
            )
            db.add(document)
            db.flush()
            document_ids[filename] = document.id
            for index, text in enumerate(texts):
                chunk = Chunk(
                    document_id=document.id,
                    collection_id=collection.id,
                    chunk_index=index,
                    page=1,
                    char_start=0,
                    char_end=len(text),
                    text=text,
                    embedding=[],
                )
                db.add(chunk)
                all_chunks.append(chunk)
        db.commit()

        extraction_report = extract_chunks(db, all_chunks, llm, floor=DEFAULT_FLOOR, model=MODEL)
        db.commit()

        resolution_report = resolve_entities(db, embedder, similarity_threshold=DEFAULT_THRESHOLD)
        db.commit()

    return GraphRun(
        factory=factory,
        document_ids=document_ids,
        extraction_report=extraction_report,
        resolution_report=resolution_report,
        llm=llm,
        embedder=embedder,
    )


def _classify_relationships(
    actual: list[tuple[str, str, str, float | None]],
    ground_truth: list[tuple[str, RelationType, str]],
) -> tuple[list, list, list, list]:
    """Compare what was stored against ground truth, by normalised endpoint name and type.

    Returns (exact, direction_errors, missing, invented). A relationship whose endpoints
    match a ground truth pair but in the reverse order is a direction error, not an
    invention: the model found the right two entities and the right relation type, only the
    direction is wrong.
    """
    gt_pairs = {(normalise(s), rel.value, normalise(t)): (s, rel, t) for s, rel, t in ground_truth}
    exact: list[tuple[str, str, str, float | None]] = []
    direction_errors: list[tuple[tuple[str, str, str, float | None], tuple]] = []
    invented: list[tuple[str, str, str, float | None]] = []
    matched_keys: set[tuple[str, str, str]] = set()
    for row in actual:
        source, relation, target, _confidence = row
        key = (normalise(source), relation, normalise(target))
        reverse_key = (normalise(target), relation, normalise(source))
        if key in gt_pairs:
            exact.append(row)
            matched_keys.add(key)
        elif reverse_key in gt_pairs:
            direction_errors.append((row, gt_pairs[reverse_key]))
            matched_keys.add(reverse_key)
        else:
            invented.append(row)
    missing = [gt for key, gt in gt_pairs.items() if key not in matched_keys]
    return exact, direction_errors, missing, invented


def _classify_merge(row: EntityMerge) -> str:
    survivor_name = row.evidence.get("survivor", {}).get("name", "?")
    pair = frozenset({normalise(survivor_name), normalise(row.merged_name)})
    if pair == frozenset(normalise(n) for n in ALIAS_PAIR):
        return "expected: the alias pair"
    if pair == frozenset(normalise(n) for n in SHARED_SURNAME_PAIR):
        return "WRONG: merged the two distinct same-surname people"
    return "unexpected (not one of the two pairs this corpus is designed to probe)"


def _report_ground_truth() -> str:
    lines = [
        "ground truth (written before anything ran):",
        "  entities:",
    ]
    for name, entity_type in GROUND_TRUTH_ENTITIES.items():
        lines.append(f"    {name} ({entity_type.value})")
    lines.append("  relationships:")
    for source, relation, target in GROUND_TRUTH_RELATIONSHIPS:
        lines.append(f"    {source} --{relation.value}--> {target}")
    lines.append(f"  alias pair (should merge): {ALIAS_PAIR[0]!r} / {ALIAS_PAIR[1]!r}")
    lines.append(
        f"  distinct people, shared surname (must not merge): "
        f"{SHARED_SURNAME_PAIR[0]!r} / {SHARED_SURNAME_PAIR[1]!r}"
    )
    return "\n".join(lines)


def _report_extraction(
    run: GraphRun, entities: list[Entity], actual_relationships: list[tuple]
) -> str:
    lines = [
        "",
        "=" * 78,
        f"graph extraction against {MODEL} (embeddings {EMBED_MODEL})",
        "a record of one run on this fixture corpus, not a benchmark (ADR 0004)",
        "=" * 78,
        _report_ground_truth(),
        "",
        f"extraction report: {run.extraction_report}",
        "",
        "entities actually stored:",
    ]
    for entity in entities:
        lines.append(
            f"  [{entity.id}] {entity.name!r} type={entity.entity_type} "
            f"confidence={entity.confidence} aliases={entity.aliases}"
        )
    lines.append("")
    lines.append("relationships actually stored:")
    for source, relation, target, confidence in actual_relationships:
        lines.append(f"  {source} --{relation}--> {target} (confidence={confidence})")

    exact, direction_errors, missing, invented = _classify_relationships(
        actual_relationships, GROUND_TRUTH_RELATIONSHIPS
    )
    lines.append("")
    lines.append(f"matched exactly ({len(exact)}):")
    for row in exact:
        lines.append(f"  {row[0]} --{row[1]}--> {row[2]}")
    lines.append(f"direction errors ({len(direction_errors)}):")
    for row, expected in direction_errors:
        lines.append(
            f"  stored {row[0]} --{row[1]}--> {row[2]}, expected the reverse: "
            f"{expected[0]} --{expected[1].value}--> {expected[2]}"
        )
    lines.append(f"missed ({len(missing)}):")
    for source, relation, target in missing:
        lines.append(f"  {source} --{relation.value}--> {target}")
    lines.append(f"invented / not in ground truth ({len(invented)}):")
    for row in invented:
        lines.append(f"  {row[0]} --{row[1]}--> {row[2]} (confidence={row[3]})")

    lines.append("")
    lines.append(f"resolution report: embedding_calls={run.resolution_report.embedding_calls}")
    lines.append(f"merges made ({len(run.resolution_report.merges)}):")
    return "\n".join(lines)


def _report_merges(db: Session, run: GraphRun) -> str:
    lines = []
    rows = db.query(EntityMerge).order_by(EntityMerge.id).all()
    for row in rows:
        survivor_name = row.evidence.get("survivor", {}).get("name", "?")
        lines.append(
            f"  [{row.id}] {row.merged_name!r} -> {survivor_name!r} "
            f"(method={row.method}, evidence={ {k: v for k, v in row.evidence.items() if k in ('similarity', 'threshold', 'embedding_model')} }) "
            f"-- {_classify_merge(row)}"
        )
    alias_merged = any(
        frozenset(
            {
                normalise(row.evidence.get("survivor", {}).get("name", "")),
                normalise(row.merged_name),
            }
        )
        == frozenset(normalise(n) for n in ALIAS_PAIR)
        for row in rows
    )
    if not alias_merged:
        lines.append(
            f"  NOT MERGED: {ALIAS_PAIR[0]!r} and {ALIAS_PAIR[1]!r} remained separate entities"
        )
    return "\n".join(lines) if lines else "  (none)"


def _report_query(result) -> str:
    lines = [
        "",
        "graph strategy query:",
        f"  question: {QUESTION}",
        f"  llm_calls={result.llm_calls} embedding_calls={result.embedding_calls} "
        f"retrieval_calls={result.retrieval_calls}",
    ]
    subgraph: Subgraph | None = result.subgraph
    if subgraph is None:
        lines.append("  subgraph: None (the question-extraction call was not usable)")
    else:
        lines.append(
            f"  subgraph.empty_reason={subgraph.empty_reason} truncated={subgraph.truncated}"
        )
        lines.append("  nodes:")
        for node in subgraph.nodes:
            lines.append(
                f"    [{node.id}] {node.name} ({node.entity_type.value}) depth={node.depth}"
            )
        lines.append("  edges:")
        for edge in subgraph.edges:
            lines.append(
                f"    [{edge.id}] {edge.source_id} --{edge.walked_as}--> {edge.target_id} "
                f"reversed={edge.reversed} confidence={edge.confidence} "
                f"source_chunk_ids={edge.source_chunk_ids}"
            )
    lines.append("  chunks pooled:")
    for chunk in result.chunks:
        lines.append(
            f"    [{chunk.chunk_id}] doc={chunk.document_id} depth={chunk.metadata.get('graph_depth')} "
            f"{chunk.text[:70]!r}"
        )
    return "\n".join(lines)


def _report_answer(answer: GraphAnswer) -> str:
    lines = [
        "",
        "generated answer:",
        f"  text: {answer.text!r}",
        f"  model={answer.model} generator={answer.generator} "
        f"input_tokens={answer.input_tokens} output_tokens={answer.output_tokens}",
        f"  dropped_claims (Phase 3 contract, {len(answer.dropped_claims)}):",
    ]
    for dropped in answer.dropped_claims:
        lines.append(f"    {dropped.reason}: {dropped.text!r}")
    lines.append(
        f"  dropped_relationship_claims (graph contract, {len(answer.dropped_relationship_claims)}):"
    )
    for dropped in answer.dropped_relationship_claims:
        lines.append(f"    {dropped.reason}: {dropped.text!r}")
    return "\n".join(lines)


def test_a_real_extraction_run_against_ollama() -> None:
    """Run real extraction, real resolution, a real graph query and real generation once.

    Assertions here are the ones that must hold no matter what the model produced: every
    stored row has a real, in-range confidence and at least one recorded source chunk
    (R1/R9/R20), no relationship is a self loop (the extraction contract forbids it, this
    checks the database agrees), the strategy makes no query-time embedding call (R19), any
    relationship claim the citation contract keeps cites an edge that actually exists in the
    traversed sub-graph, and the access filter excludes an entity sourced only by a denied
    document (ADR 0003) from a real traversal over this run's own output. Everything about
    what the model actually said -- what matched, what was backwards, what was missed, what
    was invented, what resolution merged or failed to merge -- is printed for
    docs/learning/graph-extraction-first-run.md, not asserted: a test that asserted the model
    got the corpus right would fail the moment the finding changed, and the finding is the
    point (ADR 0004).

    Everything runs inside one test, in one migrated database, so there is exactly one
    ``RAGFABRIC_TEST_OLLAMA`` skip for this whole real-model run rather than one per phase of
    it.
    """
    run = _run_pipeline()
    try:
        with run.factory() as db:
            entities = db.query(Entity).order_by(Entity.id).all()
            entities_by_id = {entity.id: entity for entity in entities}
            relationships = db.query(Relationship).order_by(Relationship.id).all()
            actual_relationships = [
                (
                    entities_by_id[relationship.source_entity_id].name,
                    relationship.relation_type,
                    entities_by_id[relationship.target_entity_id].name,
                    relationship.confidence,
                )
                for relationship in relationships
            ]

            print(_report_extraction(run, entities, actual_relationships))
            print(_report_merges(db, run))

            # --- Invariants: must hold regardless of what the model produced. ---
            for entity in entities:
                assert entity.confidence is not None, f"{entity.name!r} has no measured confidence"
                assert DEFAULT_FLOOR <= entity.confidence <= 1.0
                source_count = (
                    db.query(EntitySource).filter(EntitySource.entity_id == entity.id).count()
                )
                assert source_count >= 1, f"{entity.name!r} has no recorded source chunk"

            for relationship in relationships:
                assert relationship.confidence is not None, (
                    f"relationship {relationship.id} has no measured confidence"
                )
                assert DEFAULT_FLOOR <= relationship.confidence <= 1.0
                assert relationship.source_entity_id != relationship.target_entity_id, (
                    "a self-loop relationship was persisted; the extraction contract forbids this"
                )
                source_count = (
                    db.query(RelationshipSource)
                    .filter(RelationshipSource.relationship_id == relationship.id)
                    .count()
                )
                assert source_count >= 1, (
                    f"relationship {relationship.id} has no recorded source chunk"
                )

            # --- Access filter (ADR 0003), proven over this run's own real graph. ---
            # "profile.txt" is the only source of "Ravi Sharma reports to Meera Iyer" and, in
            # this corpus, the only mention of Meera Iyer at all. An unrestricted traversal
            # from Ravi Sharma must reach her; one that denies profile.txt must not, whatever
            # else the model did or did not extract elsewhere. The unrestricted run is the
            # control that keeps this a real proof rather than a vacuous one.
            ravi = _lookup(db, "Ravi Sharma", EntityType.PERSON)
            assert ravi is not None, (
                "Ravi Sharma was not extracted in this run; see the entity list printed above "
                "for what was actually stored. The access-filter proof needs a real seed to "
                "walk from."
            )
            profile_document_id = run.document_ids["profile.txt"]
            unrestricted_walk = traverse(db, [ravi.id], AccessFilter.unrestricted(), max_hops=2)
            denied_walk = traverse(
                db,
                [ravi.id],
                AccessFilter(denied_document_ids=frozenset({profile_document_id})),
                max_hops=2,
            )
            print(
                "\naccess filter proof (profile.txt denied):\n"
                f"  unrestricted nodes: {[n.name for n in unrestricted_walk.nodes]}\n"
                f"  denied nodes:       {[n.name for n in denied_walk.nodes]}"
            )
            unrestricted_names = {node.name for node in unrestricted_walk.nodes}
            assert "Meera Iyer" in unrestricted_names, (
                "the control run (unrestricted access) did not reach Meera Iyer from Ravi "
                "Sharma in this run's actual graph, so the access-filter proof cannot be "
                "constructed from this run's data -- see the extraction report above for what "
                "happened to that relationship"
            )
            denied_names = {node.name for node in denied_walk.nodes}
            assert "Meera Iyer" not in denied_names, (
                "Meera Iyer, sourced only by the denied document, leaked past the access filter"
            )

        # --- One real graph query, through the real strategy wiring. ---
        registry = default_registry(_config(), run.factory)
        strategy = registry.get(StrategyName.GRAPH)
        result = strategy.retrieve(QUESTION, _context())
        print(_report_query(result))

        assert result.strategy is StrategyName.GRAPH
        # R19: entity mentions in a question match by name/alias only, never a query-time embed.
        assert result.embedding_calls == 0
        # The strategy makes at most one LLM call (the question-extraction call), and none at
        # all when there is no visible entity to answer from.
        assert result.llm_calls <= 1

        answer = generate_graph_answer(QUESTION, result.chunks, result.subgraph, llm=run.llm)
        print(_report_answer(answer))

        edge_count = len(result.subgraph.edges) if result.subgraph is not None else 0
        for match in EDGE_MARKER_RE.finditer(answer.text):
            cited = int(match.group(1))
            assert 1 <= cited <= edge_count, (
                f"the kept answer cites edge {cited}, but the traversed sub-graph only has "
                f"{edge_count} edges -- the graph citation contract let an unbacked edge "
                f"reference through"
            )

        valid_reasons = {reason.value for reason in RelationshipDropReason}
        for dropped in answer.dropped_relationship_claims:
            assert dropped.reason in valid_reasons
    finally:
        downgrade(URL)
