"""End-to-end API tests via FastAPI's TestClient (fully offline).

Covers the full journey: register/login, upload + ingest, semantic + hybrid
search, cited answers, collections, analytics, and admin controls.
"""

from __future__ import annotations

VACATION_DOC = (
    b"Company Leave Policy. Full time employees receive twenty five paid "
    b"vacation days each year. Unused leave may be carried over to the next "
    b"year up to a maximum of five days. Sick leave is separate and unlimited."
)

SECURITY_DOC = (
    b"Security Guidelines. All administrator accounts must use multi factor "
    b"authentication. Passwords rotate every ninety days. Report incidents to "
    b"the security team immediately."
)


def _upload(client, headers, filename, data, collection_id=None):
    files = {"file": (filename, data, "text/plain")}
    payload = {}
    if collection_id is not None:
        payload["collection_id"] = str(collection_id)
    return client.post("/api/documents/upload", files=files, data=payload, headers=headers)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_reports_the_real_vector_count(client, auth_headers):
    """health()["index"]["vectors"] must be a measured count, not a vacuous
    or hardcoded one.

    The expected count comes from the upload responses' own ``num_chunks``
    field, a value produced independently of the health endpoint's own
    ``build_vector_store(...).count()`` call, so a regression that always
    reported 0 (or any other wrong number) would be caught here.
    """
    first = _upload(client, auth_headers, "one.txt", VACATION_DOC).json()
    second = _upload(client, auth_headers, "two.txt", SECURITY_DOC).json()
    expected = first["num_chunks"] + second["num_chunks"]
    assert expected > 0

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["index"]["vectors"] == expected


def test_register_login_and_me(client):
    assert (
        client.post(
            "/api/auth/register", json={"email": "jane@example.com", "password": "secret123"}
        ).status_code
        == 201
    )
    # Duplicate registration is rejected.
    assert (
        client.post(
            "/api/auth/register", json={"email": "jane@example.com", "password": "secret123"}
        ).status_code
        == 409
    )

    login = client.post(
        "/api/auth/login", data={"username": "jane@example.com", "password": "secret123"}
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "jane@example.com"
    assert me.json()["role"] == "user"


def test_login_rejects_bad_password(client):
    client.post("/api/auth/register", json={"email": "bob@example.com", "password": "secret123"})
    resp = client.post("/api/auth/login", data={"username": "bob@example.com", "password": "wrong"})
    assert resp.status_code == 401


def test_protected_routes_require_auth(client):
    assert client.get("/api/documents").status_code == 401
    assert client.post("/api/search/query", json={"query": "x"}).status_code == 401


def test_upload_ingest_and_query_flow(client, auth_headers):
    up = _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    assert up.status_code == 201, up.text
    doc = up.json()
    assert doc["status"] == "ready"
    assert doc["num_chunks"] >= 1
    assert doc["format"] == "txt"

    # Document appears in the list.
    listing = client.get("/api/documents", headers=auth_headers)
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    # Ask a question -> cited answer.
    resp = client.post(
        "/api/search/query",
        json={"query": "How many vacation days do employees get?"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    answer = resp.json()
    assert answer["answer"]
    assert answer["confidence"] > 0
    assert answer["citations"]
    assert answer["source_document"]["filename"] == "leave.txt"
    assert "vacation" in answer["answer"].lower()


def test_semantic_and_hybrid_search(client, auth_headers):
    _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    _upload(client, auth_headers, "security.txt", SECURITY_DOC)

    semantic = client.post(
        "/api/search/semantic",
        json={"query": "multi factor authentication for admins", "mode": "semantic"},
        headers=auth_headers,
    )
    assert semantic.status_code == 200
    results = semantic.json()["results"]
    assert results
    assert results[0]["filename"] == "security.txt"

    hybrid = client.post(
        "/api/search/hybrid",
        json={"query": "multi factor authentication", "mode": "hybrid"},
        headers=auth_headers,
    )
    assert hybrid.status_code == 200
    assert hybrid.json()["results"][0]["filename"] == "security.txt"


def test_unsupported_format_rejected(client, auth_headers):
    resp = _upload(client, auth_headers, "image.png", b"\x89PNG")
    assert resp.status_code == 400


def test_upload_rejects_a_type_outside_the_configured_allow_list(client, auth_headers):
    """limits.allowed_types (ragfabric.yaml) is a narrower, operator-set
    allow list on top of SUPPORTED_FORMATS (the parser's fixed capability
    list): csv is one of SUPPORTED_FORMATS, so this asserts the deployment's
    own configuration is what rejects it here, not the parser.
    """
    from ragfabric_core.runtime import get_config

    cfg = get_config()
    original = cfg.limits.allowed_types
    cfg.limits.allowed_types = [t for t in original if t != "csv"]
    try:
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
            headers=auth_headers,
        )
        assert resp.status_code == 400
        assert "not allowed" in resp.json()["detail"]
    finally:
        cfg.limits.allowed_types = original


def test_upload_rejects_a_file_over_the_configured_size_limit(client, auth_headers):
    from ragfabric_core.runtime import get_config

    cfg = get_config()
    original = cfg.limits.max_upload_mb
    cfg.limits.max_upload_mb = 1
    try:
        oversized = b"x" * (2 * 1024 * 1024)  # 2 MB against a 1 MB cap
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("big.txt", oversized, "text/plain")},
            headers=auth_headers,
        )
        assert resp.status_code == 413
        assert "1 MB" in resp.json()["detail"]

        within_limit = b"well under the cap"  # a few bytes, under the same 1 MB cap
        ok = client.post(
            "/api/documents/upload",
            files={"file": ("small.txt", within_limit, "text/plain")},
            headers=auth_headers,
        )
        assert ok.status_code == 201, ok.text
    finally:
        cfg.limits.max_upload_mb = original


def test_collections_scope_search(client, auth_headers):
    created = client.post(
        "/api/collections",
        json={"name": "HR", "description": "People docs"},
        headers=auth_headers,
    )
    assert created.status_code == 201
    collection_id = created.json()["id"]

    _upload(client, auth_headers, "leave.txt", VACATION_DOC, collection_id=collection_id)
    _upload(client, auth_headers, "security.txt", SECURITY_DOC)  # outside collection

    detail = client.get(f"/api/collections/{collection_id}", headers=auth_headers)
    assert detail.status_code == 200
    assert detail.json()["document_count"] == 1

    scoped = client.post(
        "/api/search/query",
        json={"query": "authentication", "collection_id": collection_id},
        headers=auth_headers,
    )
    assert scoped.status_code == 200
    # Only HR docs are searchable in this collection, so every citation is from it.
    for citation in scoped.json()["citations"]:
        assert citation["filename"] == "leave.txt"


def test_delete_document_removes_from_index(client, auth_headers):
    up = _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    doc_id = up.json()["id"]

    delete = client.delete(f"/api/documents/{doc_id}", headers=auth_headers)
    assert delete.status_code == 200

    resp = client.post(
        "/api/search/query",
        json={"query": "vacation days"},
        headers=auth_headers,
    )
    assert resp.json()["citations"] == []
    assert resp.json()["confidence"] == 0.0


def test_delete_document_removes_vector_and_lexical_index_rows(client, auth_headers):
    """Regression test for the delete leak found while reviewing Task 4: the
    relational cascade drops `chunks` rows but `chunk_embeddings` and
    `chunk_search` carry their own document_id column and are never joined
    through `chunks`, so they must be deleted explicitly. Left alone, those
    orphaned rows would accumulate and, once a store resolves hits back to
    the chunks table (as ChromaVectorStore does), silently crowd out live
    results.
    """
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.index import ChunkEmbedding, ChunkSearch

    up = _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    doc_id = up.json()["id"]

    with SessionLocal() as db:
        assert db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc_id).count() > 0
        assert db.query(ChunkSearch).filter(ChunkSearch.document_id == doc_id).count() > 0

    delete = client.delete(f"/api/documents/{doc_id}", headers=auth_headers)
    assert delete.status_code == 200

    with SessionLocal() as db:
        assert db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc_id).count() == 0
        assert db.query(ChunkSearch).filter(ChunkSearch.document_id == doc_id).count() == 0


def test_admin_delete_document_removes_vector_and_lexical_index_rows(client, admin_headers):
    """The same delete leak as above, but through the admin hard-delete route
    (`DELETE /api/admin/documents/{id}`), which used to drop only the
    relational rows and never touch the configured vector/lexical stores.
    """
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.index import ChunkEmbedding, ChunkSearch

    up = _upload(client, admin_headers, "leave.txt", VACATION_DOC)
    doc_id = up.json()["id"]

    with SessionLocal() as db:
        assert db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc_id).count() > 0
        assert db.query(ChunkSearch).filter(ChunkSearch.document_id == doc_id).count() > 0

    delete = client.delete(f"/api/admin/documents/{doc_id}", headers=admin_headers)
    assert delete.status_code == 200

    with SessionLocal() as db:
        assert db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc_id).count() == 0
        assert db.query(ChunkSearch).filter(ChunkSearch.document_id == doc_id).count() == 0


def test_delete_collection_removes_vector_and_lexical_index_rows(client, auth_headers):
    """The same delete leak as above, but through a collection delete
    (`DELETE /api/collections/{id}`), which removes every document the
    collection carried and used to leave every one of them orphaned in the
    configured vector/lexical stores.
    """
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.index import ChunkEmbedding, ChunkSearch

    collection = client.post(
        "/api/collections", json={"name": "hr-docs"}, headers=auth_headers
    ).json()
    collection_id = collection["id"]

    first = _upload(client, auth_headers, "leave.txt", VACATION_DOC, collection_id=collection_id)
    second = _upload(
        client, auth_headers, "security.txt", SECURITY_DOC, collection_id=collection_id
    )
    doc_ids = [first.json()["id"], second.json()["id"]]

    with SessionLocal() as db:
        for doc_id in doc_ids:
            assert db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc_id).count() > 0
            assert db.query(ChunkSearch).filter(ChunkSearch.document_id == doc_id).count() > 0

    delete = client.delete(f"/api/collections/{collection_id}", headers=auth_headers)
    assert delete.status_code == 200

    with SessionLocal() as db:
        for doc_id in doc_ids:
            assert (
                db.query(ChunkEmbedding).filter(ChunkEmbedding.document_id == doc_id).count() == 0
            )
            assert db.query(ChunkSearch).filter(ChunkSearch.document_id == doc_id).count() == 0


def test_analytics_overview(client, auth_headers):
    _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    client.post("/api/search/query", json={"query": "vacation"}, headers=auth_headers)
    overview = client.get("/api/analytics/overview", headers=auth_headers)
    assert overview.status_code == 200
    data = overview.json()
    assert data["documents"] == 1
    assert data["chunks"] >= 1
    assert data["queries"] == 1


def test_admin_controls(client, admin_headers, auth_headers):
    # A regular user cannot list users.
    assert client.get("/api/admin/users", headers=auth_headers).status_code == 403

    users = client.get("/api/admin/users", headers=admin_headers)
    assert users.status_code == 200
    assert any(u["role"] == "admin" for u in users.json())

    # Admin can bump a document version.
    up = _upload(client, admin_headers, "leave.txt", VACATION_DOC)
    doc_id = up.json()["id"]
    bumped = client.post(f"/api/admin/documents/{doc_id}/versions", headers=admin_headers)
    assert bumped.status_code == 200
    assert bumped.json()["version"] == 2


# --- metadata filters ---------------------------------------------------------


def test_search_can_be_filtered_by_document_format(client, auth_headers):
    from ragfabric_core.testing.fixtures import make_csv

    _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    client.post(
        "/api/documents/upload",
        files={
            "file": ("staff.csv", make_csv(["name", "leave"], [["Ada", "vacation"]]), "text/csv")
        },
        headers=auth_headers,
    )

    everything = client.post(
        "/api/search/semantic",
        json={"query": "vacation", "top_k": 10},
        headers=auth_headers,
    ).json()["results"]
    assert {r["format"] for r in everything} == {"txt", "csv"}

    csv_only = client.post(
        "/api/search/semantic",
        json={"query": "vacation", "top_k": 10, "format": "csv"},
        headers=auth_headers,
    ).json()["results"]
    assert csv_only
    assert all(r["format"] == "csv" for r in csv_only)


def test_search_rejects_an_unknown_format_filter(client, auth_headers):
    resp = client.post(
        "/api/search/semantic",
        json={"query": "vacation", "format": "xlsx"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_documents_can_be_listed_by_format(client, auth_headers):
    _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    _upload(client, auth_headers, "security.txt", SECURITY_DOC)
    listing = client.get("/api/documents?format=txt", headers=auth_headers).json()
    assert listing["total"] == 2
    assert client.get("/api/documents?format=pdf", headers=auth_headers).json()["total"] == 0


def test_search_can_be_scoped_to_one_document(client, auth_headers):
    keep = _upload(client, auth_headers, "leave.txt", VACATION_DOC).json()
    _upload(client, auth_headers, "security.txt", SECURITY_DOC)

    results = client.post(
        "/api/search/semantic",
        json={"query": "policy", "top_k": 10, "document_id": keep["id"]},
        headers=auth_headers,
    ).json()["results"]
    assert results
    assert all(r["document_id"] == keep["id"] for r in results)


# --- cited answers over HTTP --------------------------------------------------


def test_query_response_carries_used_flags_and_snippet_highlights(client, auth_headers):
    _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    _upload(client, auth_headers, "security.txt", SECURITY_DOC)

    answer = client.post(
        "/api/search/query",
        json={"query": "how many vacation days do employees receive", "top_k": 5},
        headers=auth_headers,
    ).json()

    assert answer["confidence"] > 0.5
    used = [c for c in answer["citations"] if c["used"]]
    assert used
    assert used[0]["filename"] == "leave.txt"
    # Every marker flagged as used really does appear in the answer text.
    for citation in used:
        assert citation["marker"] in answer["answer"]
    # Highlight spans index into the snippet the UI renders.
    highlighted = [c for c in answer["citations"] if c["highlights"]]
    assert highlighted
    for citation in highlighted:
        for span in citation["highlights"]:
            assert citation["snippet"][span["start"] : span["end"]].lower() == span["term"]


def test_every_offset_in_the_response_indexes_into_the_response(client, auth_headers):
    """The payload has to be self-describing.

    Character offsets are only useful to a client if the string they index into
    travelled with them. This walks every span in the response and resolves it
    against the string it claims to describe.
    """
    _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    _upload(client, auth_headers, "security.txt", SECURITY_DOC)

    answer = client.post(
        "/api/search/query",
        json={"query": "how many vacation days do employees receive", "top_k": 5},
        headers=auth_headers,
    ).json()

    for span in answer["highlights"]:
        assert answer["answer"][span["start"] : span["end"]].lower() == span["term"]

    checked = 0
    for citation in answer["citations"]:
        snippet = citation["snippet"]
        for span in citation["highlights"]:
            assert snippet[span["start"] : span["end"]].lower() == span["term"]
            checked += 1
        support = citation["supporting_span"]
        if support is not None:
            assert snippet[support["start"] : support["end"]] == support["text"]
            checked += 1
    assert checked, "no spans were checked, so this test proved nothing"


def test_supporting_span_over_http_is_quoted_by_the_answer(client, auth_headers):
    _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    answer = client.post(
        "/api/search/query",
        json={"query": "how many vacation days do employees receive"},
        headers=auth_headers,
    ).json()

    used = [c for c in answer["citations"] if c["used"]]
    assert used
    collapsed_answer = " ".join(answer["answer"].split())
    for citation in used:
        support = citation["supporting_span"]
        assert support is not None
        assert " ".join(support["text"].split()) in collapsed_answer


def test_unanswerable_question_reports_low_confidence(client, auth_headers):
    _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    answer = client.post(
        "/api/search/query",
        json={"query": "what is the airspeed velocity of an unladen swallow"},
        headers=auth_headers,
    ).json()
    assert answer["confidence"] == 0.0


# --- versioning ---------------------------------------------------------------


def test_new_version_replaces_content_and_reindexes(client, admin_headers):
    original = _upload(client, admin_headers, "handbook.txt", VACATION_DOC).json()
    doc_id = original["id"]

    # The old content is searchable before the new version lands.
    before = client.post(
        "/api/search/query", json={"query": "vacation days"}, headers=admin_headers
    ).json()
    assert before["citations"]

    revised = (
        b"Company Leave Policy version two. Full time employees now receive "
        b"thirty paid sabbatical days each year."
    )
    bumped = client.post(
        f"/api/admin/documents/{doc_id}/versions",
        files={"file": ("handbook.txt", revised, "text/plain")},
        headers=admin_headers,
    )
    assert bumped.status_code == 200, bumped.text
    assert bumped.json()["version"] == 2
    assert bumped.json()["status"] == "ready"
    assert bumped.json()["id"] == doc_id  # same id, so citations still resolve

    # Superseded text must be gone from the index, not merely outranked.
    stale = client.post(
        "/api/search/query", json={"query": "vacation days"}, headers=admin_headers
    ).json()
    assert all("vacation days" not in c["snippet"] for c in stale["citations"])

    fresh = client.post(
        "/api/search/query", json={"query": "sabbatical days"}, headers=admin_headers
    ).json()
    assert fresh["citations"]
    assert "sabbatical" in fresh["answer"]

    # Only one document row survives; the chunk count matches the live index.
    assert client.get("/api/documents", headers=admin_headers).json()["total"] == 1
    overview = client.get("/api/analytics/overview", headers=admin_headers).json()
    assert overview["chunks"] == overview["indexed_vectors"]


def test_new_version_without_a_file_only_bumps_the_counter(client, admin_headers):
    doc = _upload(client, admin_headers, "handbook.txt", VACATION_DOC).json()
    bumped = client.post(f"/api/admin/documents/{doc['id']}/versions", headers=admin_headers)
    assert bumped.status_code == 200
    assert bumped.json()["version"] == 2
    assert bumped.json()["num_chunks"] == doc["num_chunks"]


def test_versioning_requires_admin(client, auth_headers):
    doc = _upload(client, auth_headers, "handbook.txt", VACATION_DOC).json()
    resp = client.post(f"/api/admin/documents/{doc['id']}/versions", headers=auth_headers)
    assert resp.status_code == 403


# --- analytics ----------------------------------------------------------------


def test_usage_counts_documents_the_answers_actually_cited(client, auth_headers):
    leave = _upload(client, auth_headers, "leave.txt", VACATION_DOC).json()
    _upload(client, auth_headers, "security.txt", SECURITY_DOC)

    for _ in range(3):
        client.post(
            "/api/search/query",
            json={"query": "how many vacation days", "top_k": 1},
            headers=auth_headers,
        )

    usage = client.get("/api/analytics/usage", headers=auth_headers).json()
    cited = usage["most_cited_documents"]
    assert cited
    assert cited[0]["document_id"] == leave["id"]
    assert cited[0]["citations"] == 3
    assert cited[0]["filename"] == "leave.txt"


def test_usage_lists_recent_questions(client, auth_headers):
    _upload(client, auth_headers, "leave.txt", VACATION_DOC)
    client.post("/api/search/query", json={"query": "carry over rules"}, headers=auth_headers)
    usage = client.get("/api/analytics/usage", headers=auth_headers).json()
    assert usage["recent_queries"][0]["question"] == "carry over rules"


# --- permissions --------------------------------------------------------------


def test_admin_can_disable_a_user_and_block_their_token(client, admin_headers):
    client.post("/api/auth/register", json={"email": "temp@example.com", "password": "secret123"})
    login = client.post(
        "/api/auth/login", data={"username": "temp@example.com", "password": "secret123"}
    )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/auth/me", headers=headers).status_code == 200

    users = client.get("/api/admin/users", headers=admin_headers).json()
    user_id = next(u["id"] for u in users if u["email"] == "temp@example.com")
    disabled = client.put(
        f"/api/admin/users/{user_id}/permissions",
        json={"is_active": False},
        headers=admin_headers,
    )
    assert disabled.status_code == 200

    # An already-issued token must stop working once the account is disabled.
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_admin_rejects_an_unknown_role(client, admin_headers):
    users = client.get("/api/admin/users", headers=admin_headers).json()
    user_id = users[0]["id"]
    resp = client.put(
        f"/api/admin/users/{user_id}/permissions",
        json={"role": "superuser"},
        headers=admin_headers,
    )
    assert resp.status_code == 400


def test_users_cannot_delete_another_users_document(client, auth_headers, admin_headers):
    owned_by_admin = _upload(client, admin_headers, "leave.txt", VACATION_DOC).json()
    resp = client.delete(f"/api/documents/{owned_by_admin['id']}", headers=auth_headers)
    assert resp.status_code == 403


def test_ingest_records_document_type_and_sections(client, auth_headers):
    from ragfabric_core.testing.fixtures import make_txt

    data = make_txt("1. Intro\n" + "alpha " * 200 + "\n2. Leave\n" + "beta " * 200)
    r = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", data, "text/plain")},
        headers=auth_headers,
    )
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["document_type"] == "text"
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.document import Chunk

    with SessionLocal() as db:
        sections = {c.section for c in db.query(Chunk).filter(Chunk.document_id == doc["id"]).all()}
    assert "1. Intro" in sections and "2. Leave" in sections


def test_upload_retains_the_original_and_download_returns_it(client, auth_headers):
    from ragfabric_core.testing.fixtures import make_txt

    data = make_txt("retained body")
    doc = client.post(
        "/api/documents/upload",
        files={"file": ("keep.txt", data, "text/plain")},
        headers=auth_headers,
    ).json()
    assert doc["storage_path"] == f"{doc['id']}/keep.txt"
    r = client.get(f"/api/documents/{doc['id']}/download", headers=auth_headers)
    assert r.status_code == 200 and r.content == data
    assert r.headers["content-disposition"].endswith('filename="keep.txt"')
    client.delete(f"/api/documents/{doc['id']}", headers=auth_headers)
    assert (
        client.get(f"/api/documents/{doc['id']}/download", headers=auth_headers).status_code == 404
    )


def test_download_pins_the_content_type_and_forces_a_save(client, auth_headers):
    """A caller can set an arbitrary multipart Content-Type on upload (here,
    text/html on a .txt file, exactly what a browser would render as a page
    rather than save if the download route trusted it back). The download
    route must ignore that stored value and serve a type pinned from the
    file's own extension, with the response forced to save rather than
    render in the browser origin.
    """
    from ragfabric_core.testing.fixtures import make_txt

    data = make_txt("<script>alert(document.domain)</script>")
    doc = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", data, "text/html")},
        headers=auth_headers,
    ).json()
    assert doc["content_type"] == "text/html"  # exactly the untrusted value stored at upload

    r = client.get(f"/api/documents/{doc['id']}/download", headers=auth_headers)
    assert r.status_code == 200
    assert r.headers["content-type"].split(";")[0] == "text/plain"
    assert r.headers["content-disposition"].startswith("attachment")
    assert r.headers["content-disposition"].endswith('filename="notes.txt"')
    assert r.headers["x-content-type-options"] == "nosniff"


def test_download_defaults_an_unmapped_extension_to_octet_stream(client, auth_headers):
    """Belt and braces: even a filename extension this deployment does not
    recognise must never fall back to the caller-supplied Content-Type.

    The upload route's own format allow-list means a file cannot arrive with
    an unrecognised extension through upload alone, so this renames the
    document's recorded filename after the fact (the actual bytes on disk are
    untouched) to exercise the download route's default branch directly.
    """
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.document import Document

    doc = client.post(
        "/api/documents/upload",
        files={"file": ("payload.txt", b"hello", "text/html")},
        headers=auth_headers,
    ).json()

    with SessionLocal() as db:
        row = db.get(Document, doc["id"])
        row.filename = "payload.unknownext"
        db.commit()

    r = client.get(f"/api/documents/{doc['id']}/download", headers=auth_headers)
    assert r.status_code == 200
    assert r.headers["content-type"].split(";")[0] == "application/octet-stream"
    assert r.headers["x-content-type-options"] == "nosniff"
