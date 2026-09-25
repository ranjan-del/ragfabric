import os
import signal
import threading
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.config_file import GraphStoreConfig
from ragfabric_core.ingest.indexing import schedule_indexing
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.index import ChunkEmbedding, ChunkSearch
from ragfabric_core.providers.offline import HashingEmbeddingProvider
from ragfabric_core.queue.base import Job
from ragfabric_core.queue.memory_queue import MemoryJobQueue
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.workers.handlers import index_document, reconcile_stuck_indexing
from ragfabric_core.workers.runner import Worker, default_handlers, install_sigterm_handler


@pytest.fixture()
def db_and_doc(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'w.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        c = Collection(name="c")
        db.add(c)
        db.flush()
        d = Document(filename="a.txt", format="txt", collection_id=c.id, status="processing")
        db.add(d)
        db.flush()
        db.add_all(
            [
                Chunk(
                    document_id=d.id,
                    collection_id=c.id,
                    chunk_index=i,
                    page=1,
                    char_start=0,
                    char_end=5,
                    text=t,
                    embedding=[],
                )
                for i, t in enumerate(["leave policy", "rollout guide"])
            ]
        )
        db.commit()
        doc_id = d.id
    return factory, doc_id


def test_index_document_writes_both_indexes_and_marks_ready(db_and_doc):
    factory, doc_id = db_and_doc
    with factory() as db:
        n = index_document(
            db,
            doc_id,
            embedding_provider=HashingEmbeddingProvider(dim=16),
            vector_store=PgVectorStore(factory),
            lexical_store=PostgresLexicalStore(factory),
        )
        assert n == 2
        assert db.query(ChunkEmbedding).count() == 2 and db.query(ChunkSearch).count() == 2
        assert db.get(Document, doc_id).status == "ready"
        assert {e.model for e in db.query(ChunkEmbedding)} == {"hashing-16"}


def test_schedule_indexing_inline_runs_now_and_queue_mode_enqueues(db_and_doc, monkeypatch):
    factory, doc_id = db_and_doc
    monkeypatch.setattr(
        "ragfabric_core.ingest.indexing._embedding_provider",
        lambda: HashingEmbeddingProvider(dim=16),
    )
    monkeypatch.setattr(
        "ragfabric_core.ingest.indexing._stores",
        lambda: (PgVectorStore(factory), PostgresLexicalStore(factory)),
    )
    with factory() as db:
        doc = db.get(Document, doc_id)
        assert schedule_indexing(db, doc, queue=None) == "ready"
        assert db.query(ChunkEmbedding).count() == 2
    q = MemoryJobQueue()
    with factory() as db:
        doc = db.get(Document, doc_id)
        assert schedule_indexing(db, doc, queue=q) == "indexing"
        assert db.get(Document, doc_id).status == "indexing"
    kinds = [j.kind for j in list(q._items)]
    assert kinds == ["index_document", "extract_graph"]


def test_worker_runs_jobs_and_survives_a_failing_handler(db_and_doc, monkeypatch):
    factory, doc_id = db_and_doc
    q = MemoryJobQueue()
    q.enqueue(Job(id="bad", kind="explode", payload={}))
    q.enqueue(Job(id="ok", kind="index_document", payload={"document_id": doc_id}))
    handlers = default_handlers(
        embedding_provider=HashingEmbeddingProvider(dim=16),
        vector_store=PgVectorStore(factory),
        lexical_store=PostgresLexicalStore(factory),
        graph_settings=GraphStoreConfig(),
        llm=None,
    )

    def explode(db, job):
        raise RuntimeError("boom")

    handlers["explode"] = explode
    w = Worker(q, factory, handlers)
    assert w.run_once(timeout_seconds=0) is True  # bad job consumed, error logged, worker alive
    assert w.run_once(timeout_seconds=0) is True
    assert w.run_once(timeout_seconds=0) is False
    with factory() as db:
        assert db.get(Document, doc_id).status == "ready"
    assert w.failed == 1 and w.processed == 2


def test_install_sigterm_handler_sets_the_stop_event():
    stop = threading.Event()
    original = signal.getsignal(signal.SIGTERM)
    install_sigterm_handler(stop)
    try:
        assert not stop.is_set()
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.is_set(), "SIGTERM should set the stop event rather than kill the process"
    finally:
        signal.signal(signal.SIGTERM, original)


def test_run_forever_finishes_the_in_flight_job_before_a_sigterm_stops_it(db_and_doc):
    """A SIGTERM mid job must not abandon it: the job already dequeued keeps
    running to completion, and only the *next* dequeue is skipped.
    """
    factory, doc_id = db_and_doc
    q = MemoryJobQueue()
    q.enqueue(Job(id="j1", kind="index_document", payload={"document_id": doc_id}))
    q.enqueue(Job(id="j2", kind="never_runs", payload={}))
    stop = threading.Event()

    handlers = default_handlers(
        embedding_provider=HashingEmbeddingProvider(dim=16),
        vector_store=PgVectorStore(factory),
        lexical_store=PostgresLexicalStore(factory),
        graph_settings=GraphStoreConfig(),
        llm=None,
    )
    real_index = handlers["index_document"]

    def index_then_signal(db, job):
        # Simulate SIGTERM arriving while this job is still running: the
        # handler only sets the event, so this call is left to finish.
        stop.set()
        real_index(db, job)

    handlers["index_document"] = index_then_signal
    handlers["never_runs"] = lambda db, job: pytest.fail("must not run after stop was set")

    w = Worker(q, factory, handlers)
    w.run_forever(stop, timeout_seconds=0)

    assert w.processed == 1 and w.failed == 0
    with factory() as db:
        assert db.get(Document, doc_id).status == "ready"
    assert len(list(q._items)) == 1  # j2 was never dequeued


def test_queued_job_failure_marks_the_document_failed(db_and_doc):
    factory, doc_id = db_and_doc
    q = MemoryJobQueue()
    q.enqueue(Job(id="j1", kind="index_document", payload={"document_id": doc_id}))
    handlers = dict(
        default_handlers(
            embedding_provider=HashingEmbeddingProvider(dim=16),
            vector_store=PgVectorStore(factory),
            lexical_store=PostgresLexicalStore(factory),
            graph_settings=GraphStoreConfig(),
            llm=None,
        )
    )

    def raising_index(db, job):
        raise RuntimeError("embedding provider unavailable")

    handlers["index_document"] = raising_index
    w = Worker(q, factory, handlers)
    assert w.run_once(timeout_seconds=0) is True
    with factory() as db:
        doc = db.get(Document, doc_id)
        assert doc.status == "failed"
        assert "embedding provider unavailable" in doc.error


def test_inline_indexing_failure_marks_the_document_failed(db_and_doc, monkeypatch):
    factory, doc_id = db_and_doc
    from ragfabric_core.ingest import pipeline

    class Boom:
        name = "boom"

        def upsert(self, *a, **k):
            raise RuntimeError("store down")

        def query(self, *a, **k):
            return []

        def delete_document(self, *a): ...

        def count(self):
            return 0

    monkeypatch.setattr(
        "ragfabric_core.ingest.indexing._embedding_provider",
        lambda: HashingEmbeddingProvider(dim=16),
    )
    monkeypatch.setattr(
        "ragfabric_core.ingest.indexing._stores",
        lambda: (Boom(), PostgresLexicalStore(factory)),
    )
    monkeypatch.setattr("ragfabric_core.ingest.pipeline.build_queue", lambda cfg: None)
    with factory() as db:
        doc = db.get(Document, doc_id)
        result = pipeline._index_content(db, doc, b"annual leave is twelve days")
        assert result.status == "failed" and "store down" in result.error


def test_index_document_is_idempotent_when_called_twice(db_and_doc):
    """The fan out writes the vector store and the lexical store as two
    separate commits (a deferred item: a crash between them leaves a
    document indexed in one store only). The reconciliation approach for
    that gap only works if re-running index_document for an already-indexed
    document is safe, i.e. it never produces duplicate rows and always ends
    in the same state one clean run would have. This checks that property
    directly, independent of the reconcile helper itself.
    """
    factory, doc_id = db_and_doc
    with factory() as db:
        first = index_document(
            db,
            doc_id,
            embedding_provider=HashingEmbeddingProvider(dim=16),
            vector_store=PgVectorStore(factory),
            lexical_store=PostgresLexicalStore(factory),
        )
    with factory() as db:
        second = index_document(
            db,
            doc_id,
            embedding_provider=HashingEmbeddingProvider(dim=16),
            vector_store=PgVectorStore(factory),
            lexical_store=PostgresLexicalStore(factory),
        )
    assert first == second == 2
    with factory() as db:
        assert db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc_id).count() == 2
        assert db.query(ChunkSearch).filter(ChunkSearch.document_id == doc_id).count() == 2
        assert db.get(Document, doc_id).status == "ready"


def test_reconcile_recovers_a_document_stuck_by_a_crash_between_the_two_writes(db_and_doc):
    """Simulates the exact crash the fan out cannot make atomic: the vector
    store commit succeeded, the process died before the lexical store commit,
    so the document is left in "indexing" forever with no exception raised
    anywhere for the worker to have caught. reconcile_stuck_indexing must
    find it and finish the job.
    """
    factory, doc_id = db_and_doc
    with factory() as db:
        document = db.get(Document, doc_id)
        # Half of the crash: the vector store write already landed...
        PgVectorStore(factory).upsert(
            [c.id for c in document.chunks],
            [[0.0] * 16 for _ in document.chunks],
            [
                {
                    "document_id": doc_id,
                    "collection_id": document.collection_id,
                    "model": "m",
                    "dim": 16,
                }
                for _ in document.chunks
            ],
        )
        # ...but the lexical write never ran, and the document is stuck where
        # schedule_indexing leaves it while a queued job is in flight, well
        # past any reasonable grace period.
        document.status = "indexing"
        document.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
        db.commit()

    with factory() as db:
        assert db.query(ChunkSearch).filter(ChunkSearch.document_id == doc_id).count() == 0
        reconciled = reconcile_stuck_indexing(
            db,
            embedding_provider=HashingEmbeddingProvider(dim=16),
            vector_store=PgVectorStore(factory),
            lexical_store=PostgresLexicalStore(factory),
            older_than_seconds=300,
        )
        assert reconciled == 1

    with factory() as db:
        assert db.get(Document, doc_id).status == "ready"
        assert db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc_id).count() == 2
        assert db.query(ChunkSearch).filter(ChunkSearch.document_id == doc_id).count() == 2


def test_reconcile_leaves_a_recently_stuck_document_alone(db_and_doc):
    """A document only just enqueued for indexing is not a crash, it is
    normal in-flight work; retrying it before the grace period elapses would
    race a worker that is still legitimately processing it.
    """
    factory, doc_id = db_and_doc
    with factory() as db:
        document = db.get(Document, doc_id)
        document.status = "indexing"
        db.commit()

    with factory() as db:
        reconciled = reconcile_stuck_indexing(
            db,
            embedding_provider=HashingEmbeddingProvider(dim=16),
            vector_store=PgVectorStore(factory),
            lexical_store=PostgresLexicalStore(factory),
            older_than_seconds=300,
        )
        assert reconciled == 0

    with factory() as db:
        assert db.get(Document, doc_id).status == "indexing"
        assert db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc_id).count() == 0


# --- R44: job order must not erase a graph failure ------------------------------------


class _GraphLLM:
    """An extraction model that either fails every call or returns one entity."""

    name = "graph-double"
    default_model = "graph-double-model"

    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        from ragfabric_core.providers.base import Completion

        if self.fail:
            raise RuntimeError("model down")
        payload = '{"entities": [{"name": "Leave", "entity_type": "concept", "confidence": 0.9}],'
        payload += ' "relationships": []}'
        return Completion(
            text=payload,
            model=model or self.default_model,
            provider=self.name,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )

    def stream(self, *args, **kwargs):
        raise AssertionError("extraction never streams")


def _run(factory, llm: _GraphLLM, doc_id: int, *kinds: str) -> None:
    queue = MemoryJobQueue()
    for number, kind in enumerate(kinds):
        queue.enqueue(Job(id=f"j{number}", kind=kind, payload={"document_id": doc_id}))
    worker = Worker(
        queue,
        factory,
        default_handlers(
            embedding_provider=HashingEmbeddingProvider(dim=16),
            vector_store=PgVectorStore(factory),
            lexical_store=PostgresLexicalStore(factory),
            graph_settings=GraphStoreConfig(enabled=True),
            llm=llm,
        ),
    )
    while worker.run_once(timeout_seconds=0):
        pass


def _state(factory, doc_id: int) -> tuple[str, str]:
    with factory() as db:
        document = db.get(Document, doc_id)
        return document.status, document.error


def test_a_graph_failure_after_indexing_marks_the_document_failed(db_and_doc):
    factory, doc_id = db_and_doc
    _run(factory, _GraphLLM(fail=True), doc_id, "index_document", "extract_graph")
    assert _state(factory, doc_id) == ("failed", "extract_graph failed: model down")


def test_indexing_after_a_graph_failure_does_not_erase_it(db_and_doc):
    """Workers can run a document's two jobs in either order. index_document used to
    set ready and clear the error unconditionally, so a graph failure recorded first
    vanished and the document looked healthy with no graph."""
    factory, doc_id = db_and_doc
    _run(factory, _GraphLLM(fail=True), doc_id, "extract_graph", "index_document")
    assert _state(factory, doc_id) == ("failed", "extract_graph failed: model down")


def test_indexing_still_clears_its_own_earlier_failure(db_and_doc):
    factory, doc_id = db_and_doc
    with factory() as db:
        document = db.get(Document, doc_id)
        document.status, document.error = "failed", "index_document failed: store down"
        db.commit()
    _run(factory, _GraphLLM(fail=False), doc_id, "index_document")
    assert _state(factory, doc_id) == ("ready", "")


def test_a_later_graph_success_clears_its_own_failure(db_and_doc):
    factory, doc_id = db_and_doc
    _run(factory, _GraphLLM(fail=True), doc_id, "extract_graph", "index_document")
    _run(factory, _GraphLLM(fail=False), doc_id, "extract_graph")
    assert _state(factory, doc_id) == ("ready", "")


def test_a_graph_success_before_indexing_clears_the_failure_but_not_to_ready(db_and_doc):
    """Nothing has indexed the document yet, so clearing the graph failure must not
    claim it is searchable; it goes back to waiting for its index job."""
    factory, doc_id = db_and_doc
    _run(factory, _GraphLLM(fail=True), doc_id, "extract_graph")
    _run(factory, _GraphLLM(fail=False), doc_id, "extract_graph")
    assert _state(factory, doc_id) == ("indexing", "")
