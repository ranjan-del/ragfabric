"""One real routing run, against a local model, on a corpus small enough to read.

**This is a record, not a benchmark.** Per ADR 0004, what this module reports is what
happened on this fixture corpus, with the model named in ``RAGFABRIC_TEST_OLLAMA_MODEL``,
on the day it was run, on one machine. It is a single run of a non deterministic system
over fifteen chunks and four questions. Nothing here may be quoted as a measurement of the
router, compared against another strategy, or used to claim that any model routes well or
badly in general. The corpus and the questions were fixed before the run and are not to be
edited until the output looks good: a surprising decision is a finding to write down in
``docs/learning/routing-first-run.md``, never a reason to change a question.

The offline suite proves every branch of the router by feeding it JSON a model is told to
produce. This module finds out what a real small model does when ``strategy: auto`` sends
it a real question: which strategy the rules or the classifier pick, what the agent plans
when it is the strategy picked, and whether the tool check (which runs after the plan)
corrects a tool the planner chose badly. The assertions are therefore only about what must
hold whatever the model does. Everything about the model's judgement is printed, as JSON,
between two marker lines, so the record is reproducible and can be pasted as it is.

Run it with::

    RAGFABRIC_TEST_OLLAMA=1 \\
    RAGFABRIC_TEST_DATABASE_URL=postgresql+psycopg://... \\
    uv run pytest packages/core/tests/test_router_integration.py -s

Set ``RAGFABRIC_ROUTING_RECORD`` to a path to also write the JSON record to a file.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.config_file import GraphStoreConfig, RagFabricConfig
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.graph.extract import extract_chunks
from ragfabric_core.graph.resolve import resolve_entities
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.providers.base import Completion, LLMProvider, Message
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider
from ragfabric_core.router.decision import MAX_REASONING_CHARS
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.strategies import registry_defaults
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    StrategyName,
    StrategyParams,
)
from ragfabric_core.strategies.registry_defaults import default_registry
from ragfabric_core.workers.handlers import index_document

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
OLLAMA = os.environ.get("RAGFABRIC_TEST_OLLAMA", "")
BASE_URL = os.environ.get("RAGFABRIC_TEST_OLLAMA_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("RAGFABRIC_TEST_OLLAMA_MODEL", "llama3.1:8b")
EMBED_MODEL = os.environ.get("RAGFABRIC_TEST_OLLAMA_EMBED_MODEL", "nomic-embed-text")
RECORD_PATH = os.environ.get("RAGFABRIC_ROUTING_RECORD", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not OLLAMA, reason="needs RAGFABRIC_TEST_OLLAMA"),
    pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL"),
]

DEFAULT_FLOOR = GraphStoreConfig().confidence_floor
DEFAULT_THRESHOLD = GraphStoreConfig().similarity_threshold

# The Phase 5 fixture (agentic-first-run) and the Phase 6 graph fixture
# (graph-extraction-first-run), copied verbatim, in one collection.
CORPUS: dict[str, list[str]] = {
    "ingestion-runbook.txt": [
        "The ingestion worker picks up one document at a time from the job queue.",
        "A failed indexing job is retried four times. After the fourth retry the job "
        "is moved to the dead letter queue and the document is left in the indexing state.",
        "Error RF-4312 is raised when the embedding provider refuses a batch larger "
        "than the configured limit.",
    ],
    "leave-policy.txt": [
        "Full time employees accrue twenty two days of annual leave per calendar year.",
        "Unused annual leave carries forward into the next year up to a maximum of "
        "five days. Anything above five days lapses on the thirty first of December.",
        "Leave requests are approved by the reporting manager.",
    ],
    "onboarding-notes.txt": [
        "New engineers are shown the ingestion runbook and the leave policy in their first week.",
        "Retries and leave carry forward are the two questions new joiners ask most, "
        "so both documents are linked from the handbook index.",
        "The handbook index is regenerated every Monday.",
    ],
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

# Fixed before the run. The first is the Phase 5 question verbatim. The last is long and
# names nothing, so no rule has anything to match and it has to reach the classifier.
QUESTIONS: list[tuple[str, str]] = [
    (
        "phase5_compound",
        "How many times is a failed indexing job retried before it is dead lettered, "
        "and how many days of unused annual leave carry forward?",
    ),
    ("relationship", "Who does Ravi Sharma report to?"),
    ("identifier", "What does error RF-4312 mean?"),
    (
        "long_vague",
        "I have been wondering about how things generally work around here when somebody "
        "new joins the engineering group, and what kind of things they are expected to "
        "read and get familiar with during the first few days of settling in.",
    ),
]

DENIED_FILENAME = "leave-policy.txt"


def _config() -> RagFabricConfig:
    return RagFabricConfig(
        llm={"provider": "ollama", "model": MODEL, "base_url": BASE_URL},
        embeddings={
            "provider": "ollama",
            "model": EMBED_MODEL,
            "dim": 768,
            "base_url": BASE_URL,
        },
        graph_store={"enabled": True},
        strategies={
            "agentic": {
                "tools": ["semantic_search", "lexical_search", "fetch_document", "graph_search"]
            }
        },
    )


class _CountingLLM:
    """Wraps a real provider and counts the calls the strategies make through it."""

    def __init__(self, inner: LLMProvider, counter: list[int]) -> None:
        self._inner = inner
        self._counter = counter
        self.name = inner.name
        self.default_model = inner.default_model

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        self._counter[0] += 1
        return self._inner.complete(messages, **kwargs)

    def stream(self, messages: list[Message], **kwargs):
        return self._inner.stream(messages, **kwargs)


@dataclass
class Fixture:
    factory: sessionmaker
    denied_document_id: int
    denied_chunk_ids: set[int]
    extraction: str
    resolution: str


@pytest.fixture()
def ingested():
    downgrade(URL)
    upgrade(URL)
    engine = create_engine(URL)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    cfg = _config()
    embedder = build_embedding_provider(cfg.embeddings)
    llm = build_llm_provider(cfg.llm)
    vector_store = PgVectorStore(factory, model=embedder.model)
    lexical_store = PostgresLexicalStore(factory)

    with factory() as db:
        collection = Collection(name="router-fixture")
        db.add(collection)
        db.flush()
        document_ids: dict[str, int] = {}
        chunks: list[Chunk] = []
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
                chunks.append(chunk)
        db.commit()

        for document_id in document_ids.values():
            index_document(
                db,
                document_id,
                embedding_provider=embedder,
                vector_store=vector_store,
                lexical_store=lexical_store,
            )

        # graph_store.enabled is true for this run: extract and resolve the whole corpus.
        extraction = extract_chunks(db, chunks, llm, floor=DEFAULT_FLOOR, model=MODEL)
        db.commit()
        resolution = resolve_entities(db, embedder, similarity_threshold=DEFAULT_THRESHOLD)
        db.commit()

        denied_id = document_ids[DENIED_FILENAME]
        denied_chunks = {
            row.id for row in db.query(Chunk).filter(Chunk.document_id == denied_id).all()
        }

    yield Fixture(
        factory=factory,
        denied_document_id=denied_id,
        denied_chunk_ids=denied_chunks,
        extraction=str(extraction),
        resolution=str(resolution),
    )
    downgrade(URL)


def _context(denied: int) -> RetrievalContext:
    return RetrievalContext(
        principal=Principal(user_id=1, email="engineer@example.com"),
        access_filter=AccessFilter(denied_document_ids=frozenset({denied})),
        params=StrategyParams(top_k=4),
    )


def _spans(result: RetrievalResult, name: str):
    return [span for span in result.trace if span.name == name]


def _record(label: str, question: str, result: RetrievalResult, counted: int, wall_ms: int):
    """Everything the run did for one question, as plain JSON-able data."""
    router = result.router
    plan_spans = _spans(result, "plan")
    repairs = _spans(result, "repair")
    finalize = _spans(result, "finalize")
    router_span = _spans(result, "router")[0]
    return {
        "label": label,
        "question": question,
        "decision": router.model_dump(mode="json") if router else None,
        "strategy_that_ran": str(result.strategy),
        "fallback_from": str(result.fallback_from) if result.fallback_from else None,
        "fallback_reason": router_span.attributes.get("fallback_reason"),
        "classifier_violation": router_span.attributes.get("classifier_violation"),
        "classifier_error": router_span.attributes.get("classifier_error"),
        "planned": [
            {"sub_questions": s.attributes.get("sub_questions"), "tools": s.attributes.get("tools")}
            for s in plan_spans
        ],
        "tool_check": [
            {"overrides": s.attributes.get("overrides"), "detail": s.attributes.get("detail")}
            for s in _spans(result, "tool_check")
        ],
        "repairs": [dict(s.attributes) for s in repairs],
        "switch_strategy": [
            dict(s.attributes) for s in repairs if s.attributes.get("move") == "switch_strategy"
        ],
        "stop": [
            {"stop_reason": s.attributes.get("stop_reason"), "detail": s.attributes.get("detail")}
            for s in finalize
        ],
        "sub_questions": [
            {"status": r.status, "text": r.text, "reason": r.reason, "chunk_ids": r.chunk_ids}
            for r in result.sub_questions
        ],
        "counters": {
            "llm_calls": result.llm_calls,
            "llm_calls_counted_by_wrapper": counted,
            "retrieval_calls": result.retrieval_calls,
            "embedding_calls": result.embedding_calls,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "wall_ms": wall_ms,
        },
        "trace": [
            {
                "name": s.name,
                "started_ms": s.started_ms,
                "duration_ms": s.duration_ms,
                "attributes": {k: v for k, v in s.attributes.items() if v not in (None, "")},
            }
            for s in result.trace
        ],
        "chunks": [
            {"chunk_id": c.chunk_id, "document_id": c.document_id, "text": c.text[:80]}
            for c in result.chunks
        ],
    }


def test_auto_routes_real_questions_against_a_local_model(ingested, monkeypatch):
    """Route four fixed questions, print everything, assert only the invariants."""
    counter = [0]
    real_build = registry_defaults.build_llm_provider
    monkeypatch.setattr(
        registry_defaults,
        "build_llm_provider",
        lambda llm_cfg: _CountingLLM(real_build(llm_cfg), counter),
    )
    cfg = _config()
    registry = default_registry(cfg, ingested.factory)
    auto = registry.get(StrategyName.AUTO)
    max_calls = cfg.strategies.agentic.max_llm_calls

    records = []
    for label, question in QUESTIONS:
        counter[0] = 0
        began = time.perf_counter()
        result = auto.retrieve(question, _context(ingested.denied_document_id))
        wall_ms = int((time.perf_counter() - began) * 1000)
        record = _record(label, question, result, counter[0], wall_ms)
        records.append(record)

        # ADR 0002: auto returns what the routed strategy returned, never itself.
        assert result.strategy is not StrategyName.AUTO
        assert result.router is not None
        assert result.router.reasoning
        assert len(result.router.reasoning) <= MAX_REASONING_CHARS
        assert result.router.selected_strategy in {"traditional", "vectorless", "agentic", "graph"}

        # ADR 0003: the caller's filter held on every path, fallbacks included.
        pooled = {chunk.chunk_id for chunk in result.chunks}
        assert not (pooled & ingested.denied_chunk_ids), f"denied chunk returned for {label}"

        # A result that fell back says which strategy it fell back from.
        if result.fallback_from is not None:
            assert record["fallback_reason"], f"fallback with no reason for {label}"

        # llm_calls is what the strategies really spent. It is exact unless a routed
        # strategy raised, in which case it is a documented lower bound.
        failed_attempt = _spans(result, "router")[0].attributes.get("failed_attempt_calls")
        if failed_attempt == "unknown":
            assert result.llm_calls <= counter[0]
        else:
            assert result.llm_calls == counter[0], f"llm_calls mismatch for {label}"

        assert result.llm_calls <= max_calls + 1  # the classifier's one call is on top
        assert result.latency_ms >= 0

    text = json.dumps(
        {
            "model": MODEL,
            "embedding_model": EMBED_MODEL,
            "denied_document": DENIED_FILENAME,
            "extraction": ingested.extraction,
            "resolution": ingested.resolution,
            "records": records,
        },
        indent=2,
        default=str,
    )
    print("\n=== ROUTING RECORD BEGIN (one run, not a benchmark, ADR 0004) ===")
    print(text)
    print("=== ROUTING RECORD END ===")
    if RECORD_PATH:
        with open(RECORD_PATH, "w") as handle:
            handle.write(text)
