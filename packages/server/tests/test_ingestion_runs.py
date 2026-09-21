"""GET /api/runs/ingestion/{id} and the rows the pipeline/worker write for it (Task 16 C1).

Before this, ``ragfabric_core.telemetry.tracing.trace()`` only recorded a span
when a ``TraceContext`` was active, and only ``/api/ask``/``/api/search/*``
ever opened one, so every ingestion span (parse, chunk, embed, vector upsert,
lexical index) was silently dropped. These tests prove the pipeline and the
inline indexing call each now open their own trace and persist it.
"""

from __future__ import annotations

from ragfabric_core.models.document import IngestionRun
from ragfabric_core.testing.fixtures import make_txt


def test_ingesting_a_document_writes_an_ingestion_run_for_each_phase(db_session, ingested_doc):
    runs = (
        db_session.query(IngestionRun)
        .filter(IngestionRun.document_id == ingested_doc["id"])
        .order_by(IngestionRun.id)
        .all()
    )
    phases = {r.phase for r in runs}
    assert phases == {"ingest", "index"}, runs
    for run in runs:
        assert run.status == "ready"
        assert run.latency_ms >= 0
        assert run.trace, "spans must be non-empty now that a trace context is open"

    ingest_run = next(r for r in runs if r.phase == "ingest")
    index_run = next(r for r in runs if r.phase == "index")
    assert {s["name"] for s in ingest_run.trace} >= {"parse", "clean", "chunk", "persist_chunks"}
    assert {s["name"] for s in index_run.trace} >= {"embed", "vector_upsert", "lexical_index"}
    # The pipeline's own embedder has no named model; the real indexing
    # provider's does (ADR 0004: null when not genuinely available).
    assert ingest_run.embedding_model is None
    assert index_run.embedding_model is not None


def test_get_ingestion_run_returns_the_row(client, admin_token, ingested_doc, db_session):
    run = (
        db_session.query(IngestionRun)
        .filter(IngestionRun.document_id == ingested_doc["id"], IngestionRun.phase == "index")
        .one()
    )
    res = client.get(
        f"/api/runs/ingestion/{run.id}", headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["id"] == run.id
    assert body["document_id"] == ingested_doc["id"]
    assert body["phase"] == "index"
    assert body["trace"]


def test_get_ingestion_run_404s_for_an_unknown_id(client, admin_token):
    res = client.get(
        "/api/runs/ingestion/999999", headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert res.status_code == 404


def test_get_ingestion_run_is_gated_like_the_document_it_belongs_to(
    client, admin_headers, auth_headers, db_session
):
    """A viewer without a grant on the run's collection cannot read the run,
    mirroring GET /api/documents/{id}'s own access check."""
    hr = client.post("/api/collections", json={"name": "hr"}, headers=admin_headers).json()
    doc = client.post(
        "/api/documents/upload",
        files={"file": ("p.txt", make_txt("annual leave is twelve days"), "text/plain")},
        data={"collection_id": str(hr["id"])},
        headers=admin_headers,
    ).json()
    group = client.post("/api/admin/groups", json={"name": "hr-only"}, headers=admin_headers).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": hr["id"], "permission": "read"},
        headers=admin_headers,
    )
    run = db_session.query(IngestionRun).filter(IngestionRun.document_id == doc["id"]).first()
    assert run is not None

    denied = client.get(f"/api/runs/ingestion/{run.id}", headers=auth_headers)
    assert denied.status_code == 404

    allowed = client.get(f"/api/runs/ingestion/{run.id}", headers=admin_headers)
    assert allowed.status_code == 200
