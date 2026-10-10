"""The connector sync engine over an in-memory connector (Phase 10, Task 4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ragfabric_core.connectors.base import SourceDocument
from ragfabric_core.connectors.sync import sync_connector
from ragfabric_core.models.connector import ConnectorItem
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.testing.fixtures import make_pdf

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


class FakeConnector:
    """Holds files in a dict and counts fetches, so a test can see what was read."""

    name = "fake"

    def __init__(self) -> None:
        self.files: dict[str, tuple[bytes, datetime, str]] = {}
        self.fetches: list[str] = []
        self.broken: set[str] = set()
        self.not_ready: set[str] = set()

    def put(self, source_id: str, data: bytes, modified: datetime = T0, version: str = "") -> None:
        self.files[source_id] = (data, modified, version)

    def list_documents(self):
        for source_id, (data, modified, version) in sorted(self.files.items()):
            yield SourceDocument(
                source_id=source_id,
                name=source_id.rsplit("/", 1)[-1],
                content_type="",
                size_bytes=len(data),
                modified_at=modified,
                version=version,
                ready=source_id not in self.not_ready,
            )

    def fetch(self, source_id: str) -> bytes:
        self.fetches.append(source_id)
        if source_id in self.broken:
            raise OSError(f"cannot read {source_id}")
        return self.files[source_id][0]


@pytest.fixture()
def db(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import ragfabric_core.db.session as session_module
    from ragfabric_core import runtime
    from ragfabric_core.db.migrate import upgrade

    url = f"sqlite:///{tmp_path / 'sync.db'}"
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text(
        "embeddings:\n  provider: offline\n  dim: 16\ncache:\n  kind: memory\n"
        f"ingestion:\n  uploads_dir: {tmp_path / 'blobs'}\n"
    )
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(cfg))
    runtime.reset_config()
    upgrade(url)
    engine = create_engine(url)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", factory)
    with factory() as session:
        yield session
    runtime.reset_config()
    engine.dispose()


def _sync(db, connector, **kwargs):
    return sync_connector(db, connector, name="inbox", **kwargs)


def _docs(db) -> list[Document]:
    return db.query(Document).order_by(Document.id).all()


def test_a_new_source_is_ingested_and_remembered(db):
    fake = FakeConnector()
    fake.put("policies/leave.txt", b"annual leave is twenty days")
    report = _sync(db, fake)
    assert (report.added, report.updated, report.unchanged) == (1, 0, 0)
    [doc] = _docs(db)
    assert doc.filename == "leave.txt" and doc.status == "ready"
    item = db.query(ConnectorItem).one()
    assert (item.connector, item.source_id, item.document_id) == (
        "inbox",
        "policies/leave.txt",
        doc.id,
    )
    assert item.content_hash and item.size_bytes == len(b"annual leave is twenty days")


def test_unchanged_metadata_is_skipped_without_fetching(db):
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    _sync(db, fake)
    fake.fetches.clear()
    report = _sync(db, fake)
    assert report.unchanged == 1 and report.added == 0
    assert fake.fetches == []


def test_touched_but_identical_bytes_are_not_reingested(db):
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    _sync(db, fake)
    fake.put("a.txt", b"annual leave is twenty days", modified=T0 + timedelta(hours=1))
    report = _sync(db, fake)
    assert report.unchanged == 1 and report.updated == 0
    [doc] = _docs(db)
    assert doc.version == 1
    item = db.query(ConnectorItem).one()
    db.refresh(item)
    assert item.modified_at.replace(tzinfo=UTC) == T0 + timedelta(hours=1)


def test_changed_bytes_reingest_in_place(db):
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    _sync(db, fake)
    [before] = _docs(db)
    fake.put("a.txt", b"annual leave is thirty days", modified=T0 + timedelta(hours=1))
    report = _sync(db, fake)
    assert report.updated == 1
    [after] = _docs(db)
    assert after.id == before.id and after.version == 2
    texts = [c.text for c in db.query(Chunk).filter(Chunk.document_id == after.id)]
    assert any("thirty" in t for t in texts) and not any("twenty" in t for t in texts)


def test_a_removed_source_deletes_its_document(db):
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    fake.put("b.txt", b"sick leave is ten days")
    _sync(db, fake)
    del fake.files["a.txt"]
    report = _sync(db, fake)
    assert report.deleted == 1
    assert [d.filename for d in _docs(db)] == ["b.txt"]
    assert [i.source_id for i in db.query(ConnectorItem)] == ["b.txt"]


def test_on_delete_keep_leaves_the_document(db):
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    fake.put("b.txt", b"sick leave is ten days")
    _sync(db, fake)
    del fake.files["a.txt"]
    report = _sync(db, fake, on_delete="keep")
    assert report.deleted == 0
    assert len(_docs(db)) == 2


def test_an_empty_listing_never_deletes_everything(db):
    """An unmounted volume lists nothing. Treating that as 'every file was
    removed' would wipe the collection, so the engine refuses and says why."""
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    _sync(db, fake)
    fake.files.clear()
    report = _sync(db, fake)
    assert report.deleted == 0
    assert len(_docs(db)) == 1
    assert any("listed nothing" in w for w in report.warnings)


def test_documents_the_connector_did_not_create_are_never_deleted(db):
    from ragfabric_core.ingest.pipeline import ingest_document

    uploaded = ingest_document(db, filename="manual.txt", data=b"uploaded by hand")
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    fake.put("b.txt", b"sick leave is ten days")
    _sync(db, fake)
    other = FakeConnector()
    other.put("c.txt", b"travel policy")
    sync_connector(db, other, name="other")
    del fake.files["a.txt"]
    _sync(db, fake)
    names = sorted(d.filename for d in _docs(db))
    assert names == ["b.txt", "c.txt", "manual.txt"]
    assert db.get(Document, uploaded.id) is not None


def test_a_rejected_source_is_skipped_with_its_reason_and_not_refetched(db):
    fake = FakeConnector()
    fake.put("fake.pdf", b"not a pdf")
    report = _sync(db, fake)
    assert report.skipped == 1 and report.added == 0
    assert "not a PDF" in report.problems[0]
    assert _docs(db) == []
    item = db.query(ConnectorItem).one()
    assert item.document_id is None and "not a PDF" in item.rejected_reason
    fake.fetches.clear()
    report = _sync(db, fake)
    assert fake.fetches == [] and report.unchanged == 1
    # Fixed at the source: picked up on the next scan.
    fake.put("fake.pdf", make_pdf(["annual leave"]), modified=T0 + timedelta(minutes=5))
    report = _sync(db, fake)
    assert report.added == 1
    assert db.query(ConnectorItem).one().rejected_reason is None


def test_a_fetch_that_fails_is_counted_and_the_rest_still_sync(db):
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    fake.put("b.txt", b"sick leave is ten days")
    fake.broken.add("a.txt")
    report = _sync(db, fake)
    assert report.failed == 1 and report.added == 1
    assert "a.txt" in report.problems[0]
    assert [i.source_id for i in db.query(ConnectorItem)] == ["b.txt"]


def test_a_source_version_change_alone_triggers_a_fetch(db):
    fake = FakeConnector()
    fake.put("a.pdf", make_pdf(["annual leave"]), version="md5-a")
    _sync(db, fake)
    fake.fetches.clear()
    fake.put("a.pdf", make_pdf(["annual leave"]), version="md5-b")
    _sync(db, fake)
    assert fake.fetches == ["a.pdf"]


def test_deleting_the_document_by_hand_makes_the_source_new_again(db):
    from ragfabric_core.ingest.delete import delete_document_everywhere

    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    _sync(db, fake)
    [doc] = _docs(db)
    delete_document_everywhere(db, doc.id)
    assert db.query(ConnectorItem).count() == 0
    report = _sync(db, fake)
    assert report.added == 1


def test_collection_and_owner_are_applied(db):
    from ragfabric_core.models.document import Collection

    collection = Collection(name="handbook")
    db.add(collection)
    db.commit()
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    _sync(db, fake, collection_id=collection.id)
    [doc] = _docs(db)
    assert doc.collection_id == collection.id


def test_a_source_that_is_not_ready_is_neither_fetched_nor_deleted(db):
    """The folder connector lists a file still being written as not ready. A
    known file being re-saved must not disappear from the index meanwhile, and
    a new one must not be ingested half-written."""
    fake = FakeConnector()
    fake.put("a.txt", b"annual leave is twenty days")
    _sync(db, fake)
    fake.fetches.clear()
    fake.put("a.txt", b"annual leave is thirty", modified=T0 + timedelta(hours=1))
    fake.put("b.txt", b"half written")
    fake.not_ready = {"a.txt", "b.txt"}
    report = _sync(db, fake)
    assert fake.fetches == []
    assert report.pending == 2 and report.deleted == 0 and report.added == 0
    assert len(_docs(db)) == 1
