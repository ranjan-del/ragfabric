"""Vector index behaviour, with the restart path as the headline case.

The index is a NumPy matrix in process memory; the durable copy is the JSON
``chunks.embedding`` column. That split is the riskiest part of the design,
because a wrong rebuild produces a system that looks healthy (documents listed,
no errors) while search silently returns nothing. These tests pin down the
rebuild, the delete path, and idempotency of ``upsert``.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from ragfabric_core.db.session import SessionLocal
from ragfabric_core.ingest.embed import HashingEmbedder, get_embedder
from ragfabric_core.models.document import Chunk
from ragfabric_core.store.vector_store import InMemoryVectorStore, get_store
from ragfabric_server.main import app

HANDBOOK = (
    b"Company Leave Policy. Full time employees receive twenty five paid vacation "
    b"days each year. Unused leave may be carried over to the next year up to a "
    b"maximum of five days. Managers approve leave requests two weeks in advance."
)

RUNBOOK = (
    b"Incident Runbook. Page the on call engineer through the alerting channel. "
    b"Severity one incidents require a written postmortem within five working days."
)


def _record(chunk_id, text, embedder, **overrides):
    record = {
        "vector": embedder.embed_one(text).tolist(),
        "chunk_id": chunk_id,
        "document_id": 1,
        "collection_id": None,
        "filename": "kb.txt",
        "format": "txt",
        "page": 1,
        "chunk_index": chunk_id,
        "text": text,
    }
    record.update(overrides)
    return record


def _upload(client, headers, filename, data, collection_id=None):
    payload = {"collection_id": str(collection_id)} if collection_id else {}
    return client.post(
        "/api/documents/upload",
        files={"file": (filename, data, "text/plain")},
        data=payload,
        headers=headers,
    )


# --- store unit behaviour -----------------------------------------------------


def test_upsert_replaces_instead_of_duplicating():
    embedder = HashingEmbedder(dim=64)
    store = InMemoryVectorStore(dim=64)
    store.upsert([_record(1, "first revision about vacation", embedder)])
    store.upsert([_record(1, "second revision about pensions", embedder)])

    # One chunk_id must occupy exactly one row, or a single source would be
    # cited twice and the index would grow on every re-index.
    assert len(store) == 1
    assert store.all_meta()[0]["text"] == "second revision about pensions"
    results = store.search(embedder.embed_one("pensions"), top_k=5)
    assert len(results) == 1


def test_delete_document_keeps_the_chunk_lookup_consistent():
    embedder = HashingEmbedder(dim=64)
    store = InMemoryVectorStore(dim=64)
    store.upsert([_record(1, "alpha topic", embedder, document_id=1)])
    store.upsert([_record(2, "beta topic", embedder, document_id=2)])
    store.upsert([_record(3, "gamma topic", embedder, document_id=3)])

    store.delete_document(2)
    assert len(store) == 2

    # After the rows shift, re-upserting an existing chunk must still land on
    # the right row rather than clobbering a neighbour.
    store.upsert([_record(3, "gamma revised", embedder, document_id=3)])
    assert len(store) == 2
    texts = {m["text"] for m in store.all_meta()}
    assert texts == {"alpha topic", "gamma revised"}


def test_metadata_filters_are_anded():
    embedder = HashingEmbedder(dim=64)
    store = InMemoryVectorStore(dim=64)
    store.upsert([_record(1, "budget notes", embedder, collection_id=7, format="pdf")])
    store.upsert([_record(2, "budget notes", embedder, collection_id=7, format="csv")])
    store.upsert([_record(3, "budget notes", embedder, collection_id=8, format="pdf")])

    query = embedder.embed_one("budget")
    assert len(store.search(query, top_k=10, collection_id=7)) == 2
    assert len(store.search(query, top_k=10, format="pdf")) == 2
    assert len(store.search(query, top_k=10, collection_id=7, format="pdf")) == 1
    assert store.search(query, top_k=10, collection_id=99) == []


def test_search_is_stable_for_tied_scores():
    embedder = HashingEmbedder(dim=64)
    store = InMemoryVectorStore(dim=64)
    # Identical text means identical vectors and therefore tied scores.
    for chunk_id in (30, 10, 20):
        store.upsert([_record(chunk_id, "identical text", embedder)])

    order = [r["chunk_id"] for r in store.search(embedder.embed_one("identical"), top_k=3)]
    assert order == [10, 20, 30]  # tie broken deterministically on chunk_id


def test_upsert_of_a_batch_leaves_the_row_lookup_pointing_at_the_right_rows():
    """Every chunk_id must map to the row that actually holds its vector.

    A batch upsert is the normal case (one call per document, one call for the
    whole startup rebuild), so an off-by-N here corrupts the index on the very
    next re-index instead of failing loudly.
    """
    embedder = HashingEmbedder(dim=64)
    store = InMemoryVectorStore(dim=64)
    store.upsert([_record(i, f"topic number {i}", embedder) for i in (1, 2, 3)])

    meta = store.all_meta()
    for chunk_id, row in store._row_by_chunk.items():
        assert meta[row]["chunk_id"] == chunk_id

    # Re-index one chunk from the middle of the batch: it must replace its own
    # row and leave its neighbours untouched.
    store.upsert([_record(2, "topic number two revised", embedder)])
    assert len(store) == 3
    assert [m["text"] for m in store.all_meta()] == [
        "topic number 1",
        "topic number two revised",
        "topic number 3",
    ]


def test_upsert_tolerates_the_same_chunk_id_twice_in_one_batch():
    # The second record wins, and no extra row is created. The row for the
    # first copy is still staged (not yet in the matrix) when the duplicate
    # arrives, which is the case a naive matrix write would index past.
    embedder = HashingEmbedder(dim=64)
    store = InMemoryVectorStore(dim=64)
    store.upsert(
        [
            _record(1, "first copy", embedder),
            _record(2, "other chunk", embedder),
            _record(1, "second copy", embedder),
        ]
    )
    assert len(store) == 2
    assert [m["text"] for m in store.all_meta()] == ["second copy", "other chunk"]
    # The winning vector really is the one stored, not just the metadata.
    top = store.search(embedder.embed_one("second copy"), top_k=1)[0]
    assert top["chunk_id"] == 1
    assert top["score"] > 0.9


# --- rebuild from the database ------------------------------------------------


def test_rebuild_restores_the_index_after_a_simulated_restart(client, auth_headers):
    """``InMemoryVectorStore.rebuild_from_db`` restores a cleared index.

    Task 10 moved ``/api/search/query`` onto the durable pgvector-backed
    table, so it no longer round-trips through this in-memory index and can
    no longer serve as this test's observation point (a restart no longer
    loses anything for that endpoint, which is the point of that task). The
    in-memory store and its rebuild-on-restart path are still real and still
    used at process startup (see ``ragfabric_server.main``), so this test now
    observes the store directly instead of through an HTTP endpoint that no
    longer touches it.
    """
    _upload(client, auth_headers, "handbook.txt", HANDBOOK)
    _upload(client, auth_headers, "runbook.txt", RUNBOOK)

    query_vector = get_embedder().embed_one("how many vacation days do employees receive")
    before = get_store().search(query_vector, top_k=5)
    assert before

    indexed = len(get_store())
    assert indexed > 0

    # Simulate a process restart: memory is gone, only the database survives.
    get_store().clear()
    assert len(get_store()) == 0
    assert get_store().search(query_vector, top_k=5) == []  # proves the assertion below is not vacuous

    with SessionLocal() as db:
        restored = get_store().rebuild_from_db(db)
    assert restored == indexed

    after = get_store().search(query_vector, top_k=5)
    assert [c["chunk_id"] for c in after] == [c["chunk_id"] for c in before]
    assert [c["score"] for c in after] == [c["score"] for c in before]


def test_rebuild_is_idempotent(client, auth_headers):
    _upload(client, auth_headers, "handbook.txt", HANDBOOK)
    size = len(get_store())

    with SessionLocal() as db:
        get_store().rebuild_from_db(db)
        get_store().rebuild_from_db(db)

    # Rebuilding twice must not double the index.
    assert len(get_store()) == size


def test_rebuild_excludes_deleted_documents(client, auth_headers):
    keep = _upload(client, auth_headers, "handbook.txt", HANDBOOK).json()
    drop = _upload(client, auth_headers, "runbook.txt", RUNBOOK).json()

    client.delete(f"/api/documents/{drop['id']}", headers=auth_headers)

    with SessionLocal() as db:
        get_store().rebuild_from_db(db)

    document_ids = {m["document_id"] for m in get_store().all_meta()}
    assert document_ids == {keep["id"]}


def test_rebuild_repairs_embeddings_of_the_wrong_width(client, auth_headers):
    """A changed EMBEDDING_DIM must not brick startup.

    Stored vectors are re-embedded at the current width instead of raising,
    which would otherwise take the whole application down on boot.
    """
    _upload(client, auth_headers, "handbook.txt", HANDBOOK)

    with SessionLocal() as db:
        chunk = db.query(Chunk).order_by(Chunk.id).first()
        chunk.embedding = [0.1, 0.2, 0.3]  # wrong width, as if dim had changed
        db.commit()
        chunk_id = chunk.id

        restored = get_store().rebuild_from_db(db)
        assert restored > 0

        repaired = db.get(Chunk, chunk_id)
        # Written back to the database, so the next restart does not redo it.
        assert len(repaired.embedding) == get_store().dim
        assert abs(float(np.linalg.norm(repaired.embedding)) - 1.0) < 1e-5

    # The repaired chunk is genuinely searchable, not just present.
    answer = client.post(
        "/api/search/query",
        json={"query": "how many vacation days"},
        headers=auth_headers,
    ).json()
    assert answer["citations"]


def test_application_startup_rebuilds_the_index_by_itself(client, auth_headers):
    """The rebuild has to happen on its own, not because a test invoked it.

    Every other rebuild test calls ``rebuild_from_db`` directly, which would
    still pass if the lifespan handler had been deleted. This one wipes the
    index and then boots a SECOND application instance over the same database,
    so only the startup hook can put the vectors back.
    """
    _upload(client, auth_headers, "handbook.txt", HANDBOOK)
    _upload(client, auth_headers, "runbook.txt", RUNBOOK)
    question = {"query": "how many vacation days do employees receive"}
    before = client.post("/api/search/query", json=question, headers=auth_headers).json()
    indexed = len(get_store())
    assert indexed > 0

    get_store().clear()  # the process restarted; only the database survived

    with TestClient(app) as restarted:
        assert len(get_store()) == indexed
        # The same token still works, so the query goes through the normal path.
        after = restarted.post("/api/search/query", json=question, headers=auth_headers).json()

    assert after["answer"] == before["answer"]
    assert [c["chunk_id"] for c in after["citations"]] == [
        c["chunk_id"] for c in before["citations"]
    ]


def test_admin_can_rebuild_the_index_over_http(client, admin_headers):
    _upload(client, admin_headers, "handbook.txt", HANDBOOK)
    size = len(get_store())
    assert size > 0

    get_store().clear()
    assert len(get_store()) == 0

    resp = client.post("/api/admin/index/rebuild", headers=admin_headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["vectors"] == size
    assert len(get_store()) == size

    # And the repaired index answers questions again.
    answer = client.post(
        "/api/search/query",
        json={"query": "how many vacation days"},
        headers=admin_headers,
    ).json()
    assert answer["citations"]


def test_index_rebuild_requires_admin(client, auth_headers):
    assert client.post("/api/admin/index/rebuild", headers=auth_headers).status_code == 403


def test_health_reports_the_live_index_size(client, auth_headers):
    _upload(client, auth_headers, "handbook.txt", HANDBOOK)
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["index"]["vectors"] == len(get_store())
    assert health["index"]["documents"] == 1


def test_analytics_reports_index_and_chunk_counts_together(client, auth_headers):
    _upload(client, auth_headers, "handbook.txt", HANDBOOK)
    overview = client.get("/api/analytics/overview", headers=auth_headers).json()
    # The two counters come from different places (SQL vs. memory); if they
    # disagree the index has drifted.
    assert overview["chunks"] == overview["indexed_vectors"]


def test_store_applies_the_access_filter_before_ranking():
    from ragfabric_core.auth.principal import AccessFilter
    from ragfabric_core.store.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore(dim=2)
    store.upsert(
        [
            {
                "vector": [1.0, 0.0],
                "chunk_id": 1,
                "document_id": 10,
                "collection_id": 100,
                "filename": "a",
                "format": "txt",
                "page": 1,
                "chunk_index": 0,
                "text": "a",
            },
            {
                "vector": [0.9, 0.1],
                "chunk_id": 2,
                "document_id": 20,
                "collection_id": 200,
                "filename": "b",
                "format": "txt",
                "page": 1,
                "chunk_index": 0,
                "text": "b",
            },
        ]
    )
    open_only = AccessFilter(
        document_ids=frozenset(), collection_ids=frozenset({100}), denied_document_ids=frozenset()
    )
    hits = store.search([1.0, 0.0], top_k=5, access=open_only)
    assert [h["chunk_id"] for h in hits] == [1]
    assert store.access_stats({}, open_only) == (2, 1)
    assert (
        store.search([1.0, 0.0], top_k=5, access=None)
        and len(store.search([1.0, 0.0], top_k=5)) == 2
    )
