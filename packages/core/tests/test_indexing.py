import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.ingest.indexing import schedule_indexing
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.index import ChunkEmbedding, ChunkSearch
from ragfabric_core.providers.offline import HashingEmbeddingProvider
from ragfabric_core.queue.base import Job
from ragfabric_core.queue.memory_queue import MemoryJobQueue
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.workers.handlers import index_document
from ragfabric_core.workers.runner import Worker, default_handlers


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
