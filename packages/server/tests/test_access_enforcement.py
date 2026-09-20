"""Phase 2 exit criterion: a viewer without a grant gets zero chunks from a restricted collection."""

from ragfabric_core.db.session import SessionLocal
from ragfabric_core.models.index import ChunkEmbedding
from ragfabric_core.models.runs import RetrievalRun, Source
from ragfabric_core.testing.fixtures import make_txt


def _upload(client, headers, name, text, collection_id):
    r = client.post(
        "/api/documents/upload",
        files={"file": (name, make_txt(text), "text/plain")},
        data={"collection_id": str(collection_id)},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_viewer_without_grant_gets_nothing_from_a_restricted_collection(
    client, admin_headers, auth_headers
):
    hr = client.post("/api/collections", json={"name": "hr"}, headers=admin_headers).json()
    doc = _upload(
        client, admin_headers, "policy.txt", "annual leave is twelve days for everyone", hr["id"]
    )
    # Before any grant exists the collection is open (v1 behaviour).
    assert client.post(
        "/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers
    ).json()["results"]
    group = client.post("/api/admin/groups", json={"name": "hr-only"}, headers=admin_headers).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": hr["id"], "permission": "read"},
        headers=admin_headers,
    )
    # Now the collection is restricted to the group, which the viewer is not in.
    r = client.post("/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers)
    assert r.status_code == 200 and r.json()["results"] == []
    answer = client.post(
        "/api/search/query", json={"query": "how many days of annual leave"}, headers=auth_headers
    ).json()
    # The wording comes from generate.cited's no-chunks path (Task 10 moved
    # /query onto generate_cited_answer); the point being verified is that
    # the viewer got zero citations, not the exact sentence.
    assert answer["citations"] == [] and "could not find" in answer["answer"]
    assert client.get(f"/api/documents/{doc['id']}", headers=auth_headers).status_code == 404
    assert all(
        d["id"] != doc["id"]
        for d in client.get("/api/documents", headers=auth_headers).json()["items"]
    )
    assert (
        client.get(f"/api/documents/{doc['id']}/download", headers=auth_headers).status_code == 404
    )
    # Admin still sees it, and the audit row records what the filter removed for the viewer.
    assert client.get(f"/api/documents/{doc['id']}", headers=admin_headers).status_code == 200
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.access import AuditLog

    with SessionLocal() as db:
        rows = (
            db.query(AuditLog).filter(AuditLog.action == "query").order_by(AuditLog.id.desc()).all()
        )
    assert rows and rows[0].sources_returned == 0 and rows[0].sources_filtered >= 1
    # Adding the viewer to the group restores access.
    me = client.get("/api/auth/me", headers=auth_headers).json()
    client.post(
        f"/api/admin/groups/{group['id']}/members",
        json={"user_id": me["id"]},
        headers=admin_headers,
    )
    assert client.post(
        "/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers
    ).json()["results"]


def test_viewer_cannot_upload_into_a_collection_they_cannot_read(
    client, admin_headers, auth_headers
):
    hr = client.post("/api/collections", json={"name": "hr"}, headers=admin_headers).json()
    group = client.post("/api/admin/groups", json={"name": "hr-only"}, headers=admin_headers).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": hr["id"], "permission": "read"},
        headers=admin_headers,
    )
    r = client.post(
        "/api/documents/upload",
        files={"file": ("policy.txt", make_txt("annual leave is twelve days"), "text/plain")},
        data={"collection_id": str(hr["id"])},
        headers=auth_headers,
    )
    assert r.status_code == 404
    # Admins are unrestricted, so their upload into the same collection still succeeds.
    admin_doc = _upload(
        client, admin_headers, "policy.txt", "annual leave is twelve days", hr["id"]
    )
    assert admin_doc["id"]
    # The viewer can still upload when no collection is named.
    loose = client.post(
        "/api/documents/upload",
        files={"file": ("loose.txt", make_txt("no collection here"), "text/plain")},
        headers=auth_headers,
    )
    assert loose.status_code == 201


def test_query_records_a_retrieval_run_with_sources_and_is_readable_by_its_owner(
    client, auth_headers, admin_headers
):
    col = client.post("/api/collections", json={"name": "c"}, headers=auth_headers).json()
    _upload(client, auth_headers, "n.txt", "kubernetes rollout guide with three steps", col["id"])
    client.post("/api/search/query", json={"query": "rollout steps"}, headers=auth_headers)
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.runs import RetrievalRun

    with SessionLocal() as db:
        run = db.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert (
        run.selected_strategy == "traditional" and run.mode == "manual" and run.retrieval_calls == 1
    )
    assert run.embedding_model.startswith("hashing-") and run.latency_ms >= 0
    r = client.get(f"/api/runs/{run.id}", headers=auth_headers)
    assert r.status_code == 200 and r.json()["sources"] and r.json()["question"] == "rollout steps"
    names = [s["name"] for s in r.json()["trace"]]
    # Task 10 moved retrieval onto TraditionalRAGStrategy, whose spans are
    # named "embed_query"/"vector_search"/... rather than the legacy
    # in-memory retriever's "semantic_search"/"hybrid_search".
    assert "vector_search" in names
    assert "answer" in names
    assert client.get(f"/api/runs/{run.id}", headers=admin_headers).status_code == 200
    client.post(
        "/api/auth/register", json={"email": "other@example.com", "password": "password123"}
    )
    tok = client.post(
        "/api/auth/login", data={"username": "other@example.com", "password": "password123"}
    ).json()["access_token"]
    assert (
        client.get(f"/api/runs/{run.id}", headers={"Authorization": f"Bearer {tok}"}).status_code
        == 404
    )


def test_a_restricted_user_never_sees_a_forbidden_chunk_on_the_real_vector_path(
    client, admin_headers, auth_headers
):
    """Task 10 Step 7: a restricted principal must never get a forbidden chunk back
    through the real vector path (/api/search/query -> TraditionalRAGStrategy ->
    store.query(access=...) -> generate_cited_answer), the same property
    test_viewer_without_grant_gets_nothing_from_a_restricted_collection already
    proves for /api/search/semantic.

    The corpus is built so the property is checked, not assumed:
      (a) the forbidden document is proven to be genuinely indexed and genuinely
          retrievable for the exact query used below, two independent ways, while
          it is still unrestricted;
      (b) the restricted principal's own request for the same query returns a
          real, non-empty answer (from a second, allowed document), so a leak
          cannot hide behind an empty/"could not find" response and enforcement
          cannot hide behind a corpus that never matched anything;
      (c) neither the AnswerResponse citations nor the persisted Source rows for
          that request reference the forbidden document.
    """
    hr = client.post("/api/collections", json={"name": "payroll"}, headers=admin_headers).json()
    forbidden_text = (
        "Confidential salary band memo: executive base pay ranges from two hundred "
        "thousand to four hundred thousand dollars per year, reviewed each quarter."
    )
    forbidden = _upload(client, admin_headers, "payroll-bands.txt", forbidden_text, hr["id"])
    forbidden_id = forbidden["id"]
    assert forbidden["status"] == "ready"

    query_text = "confidential salary band pay ranges"

    # (a) Proof #1: real chunk_embeddings rows exist for the forbidden document.
    with SessionLocal() as db:
        embedding_rows = (
            db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == forbidden_id).all()
        )
    assert embedding_rows, (
        "forbidden document has no chunk_embeddings rows; the property below "
        "would hold vacuously (nothing was ever indexed for it)"
    )

    # (a) Proof #2: an unrestricted principal (admin, still no grant exists yet)
    # genuinely retrieves it for this exact query through the same endpoint and
    # strategy the restricted principal below will use.
    admin_answer = client.post(
        "/api/search/query", json={"query": query_text, "top_k": 10}, headers=admin_headers
    ).json()
    assert any(c["document_id"] == forbidden_id for c in admin_answer["citations"]), (
        "admin (unrestricted) did not retrieve the forbidden document for this "
        "query; the property below would hold vacuously (the query never "
        "matched the forbidden chunk in the first place)"
    )

    # Now restrict the collection to a group the ordinary user is not a member of.
    group = client.post(
        "/api/admin/groups", json={"name": "payroll-only"}, headers=admin_headers
    ).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": hr["id"], "permission": "read"},
        headers=admin_headers,
    )

    # Give the restricted user something else real to find for the same query
    # (uploaded outside any collection, so v1's "no collection -> open" rule
    # keeps it visible to them), so that a genuine, non-empty answer is
    # possible without ever touching the forbidden document.
    allowed_text = (
        "Public salary band overview: entry level roles start near fifty thousand "
        "dollars, and pay bands are published on the intranet every year."
    )
    allowed = client.post(
        "/api/documents/upload",
        files={"file": ("public-bands.txt", make_txt(allowed_text), "text/plain")},
        headers=auth_headers,
    ).json()
    assert allowed["status"] == "ready"

    res = client.post(
        "/api/search/query",
        json={"query": query_text, "top_k": 10},
        headers=auth_headers,
    )
    assert res.status_code == 200
    answer = res.json()

    # (b) A genuine, non-empty answer, not the no-chunks short circuit that
    # test_viewer_without_grant_gets_nothing_from_a_restricted_collection checks
    # separately (there, the ONLY matching document is the forbidden one, so an
    # empty answer is itself the correct, meaningful outcome; here a second,
    # allowed document guarantees a real answer is possible either way, so an
    # empty one here would signal a broken query path rather than enforcement).
    assert answer["citations"], "restricted user's query returned no citations at all"
    assert "could not find" not in answer["answer"]

    # (c) No citation, and no source document, names the forbidden document.
    assert all(c["document_id"] != forbidden_id for c in answer["citations"])
    if answer["source_document"] is not None:
        assert answer["source_document"]["document_id"] != forbidden_id

    # (c) The persisted Source rows for this exact run (all retrieved chunks,
    # cited or not) also never name the forbidden document, confirming the
    # access filter kept it out of retrieval itself rather than merely out of
    # the rendered answer.
    me = client.get("/api/auth/me", headers=auth_headers).json()
    with SessionLocal() as db:
        run = (
            db.query(RetrievalRun)
            .filter(RetrievalRun.user_id == me["id"])
            .order_by(RetrievalRun.id.desc())
            .first()
        )
        assert run is not None and run.question == query_text
        sources = db.query(Source).filter(Source.retrieval_run_id == run.id).all()
    assert sources, "restricted user's run recorded no Source rows at all"
    assert all(s.document_id != forbidden_id for s in sources)


def test_moving_a_document_updates_denormalised_collection_ids(client, admin_headers, auth_headers):
    """chunk_embeddings and chunk_search each carry their own denormalised
    collection_id so the access filter can apply inside the store query
    (ADR 0003). A document moved to a new collection must update both, in the
    same transaction as the move, or a principal granted only the OLD
    collection could keep retrieving it out of the index forever.
    """
    old = client.post("/api/collections", json={"name": "old"}, headers=admin_headers).json()
    new = client.post("/api/collections", json={"name": "new"}, headers=admin_headers).json()
    group = client.post(
        "/api/admin/groups", json={"name": "old-only"}, headers=admin_headers
    ).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": old["id"], "permission": "read"},
        headers=admin_headers,
    )
    me = client.get("/api/auth/me", headers=auth_headers).json()
    client.post(
        f"/api/admin/groups/{group['id']}/members",
        json={"user_id": me["id"]},
        headers=admin_headers,
    )
    # A grant on "new" that the viewer is NOT part of. Without this, "new"
    # would carry no grant at all and, per the v1 default, would be open to
    # everyone regardless of the index rows, which would make the assertion
    # below pass for the wrong reason (an open destination collection) rather
    # than because the stale index rows were actually fixed up.
    admin_me = client.get("/api/auth/me", headers=admin_headers).json()
    decoy = client.post(
        "/api/admin/groups", json={"name": "new-owner"}, headers=admin_headers
    ).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": decoy["id"], "collection_id": new["id"], "permission": "read"},
        headers=admin_headers,
    )
    client.post(
        f"/api/admin/groups/{decoy['id']}/members",
        json={"user_id": admin_me["id"]},
        headers=admin_headers,
    )

    doc = _upload(
        client, admin_headers, "policy.txt", "annual leave is twelve days for everyone", old["id"]
    )

    # Before the move: the viewer, granted only the OLD collection, can see it.
    # This makes the test non-vacuous: it proves the grant actually reaches
    # this document before checking that the move revokes it.
    before = client.post(
        "/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers
    ).json()
    assert any(r["document_id"] == doc["id"] for r in before["results"]), before

    moved = client.post(
        f"/api/documents/{doc['id']}/move",
        json={"collection_id": new["id"]},
        headers=admin_headers,
    )
    assert moved.status_code == 200
    assert moved.json()["collection_id"] == new["id"]

    # After the move: the same viewer, still granted only the OLD collection,
    # sees nothing, even though the document and its chunks still exist.
    after = client.post(
        "/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers
    ).json()
    assert all(r["document_id"] != doc["id"] for r in after["results"]), after

    from ragfabric_core.models.document import Chunk
    from ragfabric_core.models.index import ChunkSearch

    with SessionLocal() as db:
        embeddings = db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc["id"]).all()
        searches = db.query(ChunkSearch).filter(ChunkSearch.document_id == doc["id"]).all()
        chunks = db.query(Chunk).filter(Chunk.document_id == doc["id"]).all()
    assert embeddings and all(row.collection_id == new["id"] for row in embeddings)
    assert searches and all(row.collection_id == new["id"] for row in searches)
    assert chunks and all(row.collection_id == new["id"] for row in chunks)

    # Granted only the NEW collection, the viewer can now retrieve it.
    group2 = client.post(
        "/api/admin/groups", json={"name": "new-only"}, headers=admin_headers
    ).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group2["id"], "collection_id": new["id"], "permission": "read"},
        headers=admin_headers,
    )
    client.post(
        f"/api/admin/groups/{group2['id']}/members",
        json={"user_id": me["id"]},
        headers=admin_headers,
    )
    after_regrant = client.post(
        "/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers
    ).json()
    assert any(r["document_id"] == doc["id"] for r in after_regrant["results"]), after_regrant
