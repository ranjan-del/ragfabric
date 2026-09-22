"""One real agent run, against a local model, on a corpus small enough to read.

**This is a record, not a benchmark.** Per ADR 0004, what this module reports
is what happened on this fixture corpus, with the model named in
``RAGFABRIC_TEST_OLLAMA_MODEL``, on the day it was run, on one machine. It is a
single run of a non deterministic system over nine chunks. Nothing here may be
quoted as a measurement of the agentic strategy, compared against another
strategy, or used to claim that any model plans well or badly in general. The
benchmark numbers in this repository come from ``make eval`` and from nowhere
else.

What the test is actually for is the thing the scripted doubles cannot do. The
offline tests prove every branch of the loop by feeding it JSON that a model is
told to produce. This one finds out whether a real small model produces that
JSON at all, whether its plan is sane, and whether the loop still stops
honestly when it is not. So the assertions are about the agent's invariants,
which must hold whatever the model does, and everything about the model's
judgement is printed rather than asserted. A test that asserted the model
planned two sub-questions would be a test that fails when the finding changes,
and the finding is the output.

Run it with::

    RAGFABRIC_TEST_OLLAMA=1 \\
    RAGFABRIC_TEST_DATABASE_URL=postgresql+psycopg://... \\
    uv run pytest packages/core/tests/test_agentic_integration.py -s
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.providers.registry import build_embedding_provider
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.strategies.base import RetrievalContext, StrategyName, StrategyParams
from ragfabric_core.strategies.registry_defaults import default_registry
from ragfabric_core.workers.handlers import index_document

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

# Two facts in two different documents, plus a document that talks about both
# subjects without answering either. The distractor is the point: a corpus
# where every chunk answers something cannot show whether the assessment is
# strict, because anything retrieved would do.
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
}

QUESTION = (
    "How many times is a failed indexing job retried before it is dead lettered, "
    "and how many days of unused annual leave carry forward?"
)


@pytest.fixture()
def ingested():
    """The corpus, embedded and lexically indexed through the real code paths.

    Chunks are written directly rather than through the file parser: the parser
    has its own tests, and what is under test here is the agent over a populated
    index, not the shape of a text file.
    """
    downgrade(URL)
    upgrade(URL)
    engine = create_engine(URL)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    cfg = _config()
    embedder = build_embedding_provider(cfg.embeddings)
    vector_store = PgVectorStore(factory, model=embedder.model)
    lexical_store = PostgresLexicalStore(factory)

    with factory() as db:
        collection = Collection(name="fixtures")
        db.add(collection)
        db.flush()
        document_ids = []
        for filename, texts in CORPUS.items():
            document = Document(
                filename=filename,
                format="txt",
                collection_id=collection.id,
                status="processing",
            )
            db.add(document)
            db.flush()
            db.add_all(
                [
                    Chunk(
                        document_id=document.id,
                        collection_id=collection.id,
                        chunk_index=index,
                        page=1,
                        char_start=0,
                        char_end=len(text),
                        text=text,
                        embedding=[],
                    )
                    for index, text in enumerate(texts)
                ]
            )
            document_ids.append(document.id)
        db.commit()

        for document_id in document_ids:
            index_document(
                db,
                document_id,
                embedding_provider=embedder,
                vector_store=vector_store,
                lexical_store=lexical_store,
            )

    yield factory
    downgrade(URL)


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
        params=StrategyParams(top_k=4),
    )


def _report(result, question: str) -> str:
    """Everything the run actually did, in the order somebody would read it."""
    lines = [
        "",
        "=" * 78,
        f"agentic run against {MODEL} (embeddings {EMBED_MODEL})",
        "a record of one run on this fixture corpus, not a benchmark (ADR 0004)",
        "=" * 78,
        f"question: {question}",
        "",
        "counters:",
        f"  llm_calls        {result.llm_calls}",
        f"  retrieval_calls  {result.retrieval_calls}",
        f"  embedding_calls  {result.embedding_calls}",
        f"  input_tokens     {result.input_tokens}",
        f"  output_tokens    {result.output_tokens}",
        f"  latency_ms       {result.latency_ms}",
        f"  chunks pooled    {len(result.chunks)}",
        "",
        f"sub-questions planned: {len(result.sub_questions)}",
    ]
    for index, report in enumerate(result.sub_questions):
        lines.append(f"  [{index}] {report.status}: {report.text}")
        if report.reason:
            lines.append(f"        reason: {report.reason}")
        lines.append(f"        chunks: {report.chunk_ids}")
    lines.append("")
    lines.append("trace:")
    for span in result.trace:
        attributes = " ".join(
            f"{key}={value}" for key, value in span.attributes.items() if value not in (None, "")
        )
        lines.append(
            f"  {span.started_ms:>6}ms +{span.duration_ms:<6}ms {span.name:<9} {attributes}"
        )
    lines.append("")
    lines.append("pooled evidence:")
    for chunk in result.chunks:
        lines.append(f"  [{chunk.chunk_id}] doc {chunk.document_id} {chunk.text[:70]}")
    lines.append("=" * 78)
    return "\n".join(lines)


def test_the_agent_runs_end_to_end_against_a_local_model(ingested):
    """Run it, print what happened, and assert only what must hold regardless.

    Every assertion below is an invariant of the agent, not a judgement of the
    model. The model is free to plan one sub-question or six, to pick the wrong
    tool, and to leave a part of the question unanswered; the loop still has to
    stop for one of three named reasons, report a real count of what it did,
    and explain every gap it is handing back.
    """
    registry = default_registry(_config(), ingested)
    agent = registry.get(StrategyName.AGENTIC)

    result = agent.retrieve(QUESTION, _context())
    print(_report(result, QUESTION))

    assert result.strategy is StrategyName.AGENTIC
    assert result.llm_calls >= 1
    # The deployment's configured cap, which the strategy now actually enforces.
    assert result.llm_calls <= _config().strategies.agentic.max_llm_calls
    assert result.retrieval_calls >= 1
    assert result.input_tokens > 0

    # The counters describe this run rather than the shape of the strategy.
    tool_calls = sum(1 for span in result.trace if span.name == "retrieve")
    assert tool_calls >= 1
    assert result.retrieval_calls >= tool_calls

    finalize = next(span for span in result.trace if span.name == "finalize")
    assert finalize.attributes["stop_reason"] in {"resolved", "budget", "no_progress"}
    assert finalize.attributes["detail"]

    # Local inference is priced at zero in pricing.yaml, with a source saying
    # why, so the cost of this run is known to be nothing rather than unknown.
    assert finalize.attributes["cost_known"] is True
    assert finalize.attributes["cost_usd"] == 0.0

    assert result.sub_questions, "a run that planned nothing would report nothing"
    for report in result.sub_questions:
        assert report.status in {"answered", "abandoned", "open"}
        if report.status != "answered":
            assert report.reason, f"{report.status} sub-question with no reason: {report.text}"


def test_the_access_filter_reaches_the_stores_on_a_real_run(ingested):
    """A filter that excludes a document must exclude it from the pooled evidence.

    Asserted against a live index rather than a fake tool, because ADR 0003
    puts the filter inside the store query and the agent pools evidence from
    several calls. One call made under the wrong filter contaminates an answer
    that is later generated from the whole pool.
    """
    with ingested() as db:
        leave = db.query(Document).filter(Document.filename == "leave-policy.txt").one()
        leave_id = leave.id
        leave_chunk_ids = {
            row.id for row in db.query(Chunk).filter(Chunk.document_id == leave_id).all()
        }

    registry = default_registry(_config(), ingested)
    agent = registry.get(StrategyName.AGENTIC)
    context = RetrievalContext(
        principal=Principal(user_id=1, email="engineer@example.com"),
        access_filter=AccessFilter(denied_document_ids=frozenset({leave_id})),
        params=StrategyParams(top_k=4),
    )

    result = agent.retrieve(QUESTION, context)
    print(_report(result, QUESTION + "  [leave-policy.txt denied]"))

    pooled = {chunk.chunk_id for chunk in result.chunks}
    assert not (pooled & leave_chunk_ids)
