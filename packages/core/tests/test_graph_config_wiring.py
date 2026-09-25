"""Graph configuration: strict validation, and every key read by something.

Phase 5 shipped four typed agent limits that were loaded, validated and echoed
back by ``config validate`` while nothing enforced them. The central test here,
``test_every_graph_setting_reaches_the_code_that_uses_it``, is parametrised over
the fields of both typed graph sections, so a field added later without a probe
proving it reaches the extractor, the resolver, the strategy or the handler
fails by name instead of shipping as a number nobody reads.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from ragfabric_core.config_file import (
    GraphStoreConfig,
    GraphStrategyConfig,
    RagFabricConfig,
    load_config,
)
from ragfabric_core.graph.contracts import EntityType, RelationType
from ragfabric_core.graph.traverse import DEFAULT_NODE_BUDGET, MAX_HOPS_CEILING
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.graph import Entity, EntityMerge, Relationship
from ragfabric_core.providers.base import Completion, EmbeddingResult
from ragfabric_core.queue.base import Job
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.strategies import registry_defaults
from ragfabric_core.workers import handlers
from ragfabric_core.workers.runner import default_handlers

EXAMPLE = Path(__file__).resolve().parents[3] / "ragfabric.example.yaml"

ADA_AND_ENGINE = {
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

# Two people whose names neither normalise equal nor alias each other, so
# only the embedding stage can join them, and only at a threshold at or below
# their measured similarity of 0.8.
TWO_NAMES_FOR_ONE_PERSON = {
    "entities": [
        {"name": "Ada Lovelace", "entity_type": "person", "confidence": 0.9},
        {"name": "Countess Lovelace", "entity_type": "person", "confidence": 0.9},
    ],
    "relationships": [],
}


class RecordingLLM:
    """Answers every extraction call with one payload and records each request."""

    name = "recording"
    default_model = "provider-default"

    def __init__(self, payload: dict) -> None:
        self._text = json.dumps(payload)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls.append({"messages": list(messages), "model": model})
        return Completion(
            text=self._text,
            model=model or self.default_model,
            provider=self.name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )

    def stream(self, *args, **kwargs):
        raise AssertionError("extraction never streams")

    def prompt(self) -> str:
        return "\n".join(message.content for message in self.calls[0]["messages"])


class RefusingLLM(RecordingLLM):
    """A model that must never be called."""

    def __init__(self) -> None:
        super().__init__({"entities": [], "relationships": []})

    def complete(self, *args, **kwargs):
        raise AssertionError("graph extraction is disabled; no model call may be made")


class ScriptedEmbedder:
    name = "scripted"
    model = "scripted-embedder"
    dim = 2

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
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


def _factory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path / 'graph.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _document(factory, text: str = "Ada Lovelace works on the Analytical Engine.") -> int:
    with factory() as db:
        collection = Collection(name="c")
        db.add(collection)
        db.flush()
        document = Document(
            filename="a.txt", format="txt", collection_id=collection.id, status="indexing"
        )
        db.add(document)
        db.flush()
        db.add(
            Chunk(
                document_id=document.id,
                collection_id=collection.id,
                chunk_index=0,
                text=text,
                embedding=[],
            )
        )
        db.commit()
        return document.id


def _run_extract_graph(path: Path, settings: GraphStoreConfig, llm, embedder=None):
    """Drive the real worker dispatch for one document; return the session factory."""
    factory = _factory(path)
    document_id = _document(factory)
    by_kind = default_handlers(
        embedding_provider=embedder or ScriptedEmbedder(),
        vector_store=PgVectorStore(factory),
        lexical_store=PostgresLexicalStore(factory),
        graph_settings=settings,
        llm=llm,
    )
    with factory() as db:
        by_kind["extract_graph"](
            db, Job(id="g", kind="extract_graph", payload={"document_id": document_id})
        )
    return factory


def _rows(factory, model) -> list:
    with factory() as db:
        return db.execute(select(model)).scalars().all()


# --- probes: one per typed field, each proving the value reaches its reader ---


def _probe_enabled(path: Path, settings: GraphStoreConfig) -> None:
    llm = RecordingLLM(ADA_AND_ENGINE)
    factory = _run_extract_graph(path, settings, llm)
    assert len(llm.calls) == 1
    assert {entity.name for entity in _rows(factory, Entity)} == {
        "Ada Lovelace",
        "Analytical Engine",
    }


def _probe_kind(path: Path, settings: GraphStoreConfig) -> None:
    # One accepted value: the graph lives in the relational tables. The
    # handler writing Entity and Relationship rows through the SQL session is
    # the code that implements it; anything else is refused at validation.
    assert get_args(GraphStoreConfig.model_fields["kind"].annotation) == ("postgres",)
    factory = _run_extract_graph(path, settings, RecordingLLM(ADA_AND_ENGINE))
    assert len(_rows(factory, Relationship)) == 1


def _probe_extraction_model(path: Path, settings: GraphStoreConfig) -> None:
    llm = RecordingLLM(ADA_AND_ENGINE)
    factory = _run_extract_graph(path, settings, llm)
    assert llm.calls[0]["model"] == settings.extraction_model
    assert {entity.extraction_model for entity in _rows(factory, Entity)} == {
        settings.extraction_model
    }


def _probe_confidence_floor(path: Path, settings: GraphStoreConfig) -> None:
    # Everything in the payload reports 0.9; the default floor keeps it all.
    factory = _run_extract_graph(path, settings, RecordingLLM(ADA_AND_ENGINE))
    assert _rows(factory, Entity) == []
    assert _rows(factory, Relationship) == []


def _probe_similarity_threshold(path: Path, settings: GraphStoreConfig) -> None:
    embedder = ScriptedEmbedder({"Ada Lovelace": [1.0, 0.0], "Countess Lovelace": [0.8, 0.6]})
    factory = _run_extract_graph(
        path, settings, RecordingLLM(TWO_NAMES_FOR_ONE_PERSON), embedder=embedder
    )
    merges = _rows(factory, EntityMerge)
    assert len(merges) == 1
    assert merges[0].evidence["threshold"] == settings.similarity_threshold
    assert len(_rows(factory, Entity)) == 1


def _probe_entity_types(path: Path, settings: GraphStoreConfig) -> None:
    llm = RecordingLLM(ADA_AND_ENGINE)
    factory = _run_extract_graph(path, settings, llm)
    assert "product" not in llm.prompt()
    assert {entity.entity_type for entity in _rows(factory, Entity)} == {"person"}
    # The edge's target was a disabled type, so the edge has nothing to point at.
    assert _rows(factory, Relationship) == []


def _probe_relation_types(path: Path, settings: GraphStoreConfig) -> None:
    llm = RecordingLLM(ADA_AND_ENGINE)
    factory = _run_extract_graph(path, settings, llm)
    assert "WORKS_ON" not in llm.prompt()
    assert "MEMBER_OF" in llm.prompt()
    assert len(_rows(factory, Entity)) == 2
    assert _rows(factory, Relationship) == []


STORE_PROBES: dict[str, tuple[Any, Callable[[Path, GraphStoreConfig], None]]] = {
    "enabled": (True, _probe_enabled),
    "kind": ("postgres", _probe_kind),
    "extraction_model": ("custom-extractor", _probe_extraction_model),
    "confidence_floor": (0.95, _probe_confidence_floor),
    "similarity_threshold": (0.7, _probe_similarity_threshold),
    "entity_types": (["person"], _probe_entity_types),
    "relation_types": (["MEMBER_OF"], _probe_relation_types),
}

STRATEGY_PROBES: dict[str, Any] = {
    "max_hops": 3,
    "node_budget": 7,
}

FIELDS = [("graph_store", name) for name in GraphStoreConfig.model_fields] + [
    ("strategies.graph", name) for name in GraphStrategyConfig.model_fields
]


def _only_one_value(model, name: str) -> bool:
    return len(get_args(model.model_fields[name].annotation)) == 1


@pytest.mark.parametrize(("section", "name"), FIELDS, ids=[f"{s}.{n}" for s, n in FIELDS])
def test_every_graph_setting_reaches_the_code_that_uses_it(tmp_path, section, name):
    if section == "graph_store":
        assert name in STORE_PROBES, f"graph_store.{name} has no probe proving anything reads it"
        value, probe = STORE_PROBES[name]
        field = GraphStoreConfig.model_fields[name]
        if not _only_one_value(GraphStoreConfig, name):
            assert value != field.get_default(call_default_factory=True), (
                f"the probe for graph_store.{name} must use a non-default value"
            )
        cfg = RagFabricConfig.model_validate({"graph_store": {"enabled": True, name: value}})
        probe(tmp_path, cfg.graph_store)
        return

    assert name in STRATEGY_PROBES, f"strategies.graph.{name} has no probe proving it is read"
    value = STRATEGY_PROBES[name]
    assert value != GraphStrategyConfig.model_fields[name].get_default()
    cfg = RagFabricConfig.model_validate({"strategies": {"graph": {name: value}}})
    strategy = registry_defaults.default_registry(cfg, _factory(tmp_path)).get("graph")
    assert getattr(strategy, name) == value


def test_graph_extraction_is_skipped_when_disabled(tmp_path, monkeypatch):
    # The handler's own logger is recorded directly, so the assertion does not
    # depend on whatever logging configuration an earlier test left behind.
    logged: list[str] = []
    monkeypatch.setattr(handlers.log, "info", lambda message, *args: logged.append(message % args))
    factory = _run_extract_graph(tmp_path, GraphStoreConfig(), RefusingLLM())
    assert _rows(factory, Entity) == []
    assert [chunk.extraction_hash for chunk in _rows(factory, Chunk)] == [None]
    assert any("disabled" in line for line in logged)


def test_a_disabled_graph_needs_no_model_at_all(tmp_path):
    """A deployment that does not want a graph builds no LLM for it."""
    factory = _run_extract_graph(tmp_path, GraphStoreConfig(), llm=None)
    assert _rows(factory, Entity) == []


def test_an_enabled_graph_without_a_model_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="LLM"):
        _run_extract_graph(tmp_path, GraphStoreConfig(enabled=True), llm=None)


def test_no_extraction_model_means_the_provider_default(tmp_path):
    llm = RecordingLLM(ADA_AND_ENGINE)
    _run_extract_graph(tmp_path, GraphStoreConfig(enabled=True), llm)
    assert llm.calls[0]["model"] == "provider-default"


def test_the_handler_commits_what_it_extracted(tmp_path):
    factory = _run_extract_graph(
        tmp_path, GraphStoreConfig(enabled=True), RecordingLLM(ADA_AND_ENGINE)
    )
    # _rows opens a fresh session, so it only sees committed rows.
    assert len(_rows(factory, Relationship)) == 1
    assert all(chunk.extraction_hash for chunk in _rows(factory, Chunk))


# --- validation ---


def test_an_unknown_graph_key_is_rejected():
    with pytest.raises(ValidationError, match="bogus"):
        RagFabricConfig.model_validate({"graph_store": {"bogus": 1}})
    with pytest.raises(ValidationError, match="max_nodes"):
        RagFabricConfig.model_validate({"strategies": {"graph": {"max_nodes": 200}}})


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_a_confidence_floor_outside_zero_to_one_is_rejected(value):
    with pytest.raises(ValidationError, match="confidence_floor"):
        GraphStoreConfig(confidence_floor=value)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_a_similarity_threshold_outside_zero_to_one_is_rejected(value):
    with pytest.raises(ValidationError, match="similarity_threshold"):
        GraphStoreConfig(similarity_threshold=value)


def test_neo4j_is_rejected_citing_the_decision_that_removed_it():
    with pytest.raises(ValidationError, match="ADR 0011"):
        GraphStoreConfig(kind="neo4j")


def test_graph_store_defaults():
    settings = GraphStoreConfig()
    assert settings.enabled is False
    assert settings.kind == "postgres"
    assert settings.extraction_model is None
    assert settings.confidence_floor == 0.5
    assert settings.similarity_threshold == 0.9
    assert settings.entity_types == list(EntityType)
    assert settings.relation_types == list(RelationType)


def test_graph_strategy_defaults():
    settings = GraphStrategyConfig()
    assert settings.max_hops == 2
    assert settings.node_budget == DEFAULT_NODE_BUDGET


@pytest.mark.parametrize("field", ["entity_types", "relation_types"])
def test_an_empty_type_list_is_rejected(field):
    with pytest.raises(ValidationError, match=field):
        GraphStoreConfig(**{field: []})


def test_an_unknown_type_is_rejected():
    with pytest.raises(ValidationError, match="entity_types"):
        GraphStoreConfig(entity_types=["spaceship"])
    with pytest.raises(ValidationError, match="relation_types"):
        GraphStoreConfig(relation_types=["LOVES"])


def test_a_duplicated_type_is_rejected():
    with pytest.raises(ValidationError, match="once"):
        GraphStoreConfig(entity_types=["person", "person"])


@pytest.mark.parametrize("value", [0, MAX_HOPS_CEILING + 1])
def test_max_hops_outside_one_to_the_ceiling_is_rejected(value):
    with pytest.raises(ValidationError, match="max_hops"):
        GraphStrategyConfig(max_hops=value)


def test_max_hops_at_the_ceiling_is_accepted():
    assert GraphStrategyConfig(max_hops=MAX_HOPS_CEILING).max_hops == MAX_HOPS_CEILING


def test_a_node_budget_below_one_is_rejected():
    with pytest.raises(ValidationError, match="node_budget"):
        GraphStrategyConfig(node_budget=0)


# --- the example file ---


def test_the_example_file_sets_the_typed_graph_sections():
    cfg = load_config(EXAMPLE)
    assert cfg.graph_store == GraphStoreConfig()
    assert cfg.strategies.graph == GraphStrategyConfig()


def test_every_key_in_the_example_file_has_a_comment():
    key = re.compile(r"^\s*[a-z_]+:")
    uncommented = [
        line
        for line in EXAMPLE.read_text(encoding="utf-8").splitlines()
        if key.match(line) and "#" not in line
    ]
    assert uncommented == []


# --- the inline ingestion path ---


def test_inline_indexing_extracts_the_graph_with_the_configured_settings(tmp_path, monkeypatch):
    """Inline is the default indexing mode; enabling the graph must not need a worker."""
    from ragfabric_core.ingest import indexing

    cfg = RagFabricConfig.model_validate(
        {"graph_store": {"enabled": True, "extraction_model": "inline-extractor"}}
    )
    llm = RecordingLLM(ADA_AND_ENGINE)
    monkeypatch.setattr(indexing, "get_config", lambda: cfg)
    monkeypatch.setattr(indexing, "build_llm_provider", lambda llm_cfg: llm)
    monkeypatch.setattr(indexing, "_embedding_provider", ScriptedEmbedder)
    factory = _factory(tmp_path)
    document_id = _document(factory)
    with factory() as db:
        indexing.extract_graph_inline(db, db.get(Document, document_id))
    assert llm.calls[0]["model"] == "inline-extractor"
    assert len(_rows(factory, Relationship)) == 1


def test_inline_indexing_builds_no_model_when_the_graph_is_disabled(tmp_path, monkeypatch):
    from ragfabric_core.ingest import indexing

    def refuse(llm_cfg):
        raise AssertionError("a disabled graph must not build an LLM provider")

    monkeypatch.setattr(indexing, "get_config", RagFabricConfig)
    monkeypatch.setattr(indexing, "build_llm_provider", refuse)
    monkeypatch.setattr(indexing, "_embedding_provider", ScriptedEmbedder)
    factory = _factory(tmp_path)
    document_id = _document(factory)
    with factory() as db:
        indexing.extract_graph_inline(db, db.get(Document, document_id))
    assert _rows(factory, Entity) == []


def test_inline_scheduling_indexes_then_extracts(tmp_path, monkeypatch):
    from ragfabric_core.ingest import indexing

    order: list[str] = []
    monkeypatch.setattr(indexing, "index_inline", lambda db, doc: order.append("index"))
    monkeypatch.setattr(indexing, "extract_graph_inline", lambda db, doc: order.append("graph"))
    factory = _factory(tmp_path)
    document_id = _document(factory)
    with factory() as db:
        assert indexing.schedule_indexing(db, db.get(Document, document_id), None) == "ready"
    assert order == ["index", "graph"]
