"""The query path runs on the pgvector table, not the in memory index."""

from __future__ import annotations


def test_query_answers_from_the_indexed_corpus_with_citations(client, admin_token, ingested_doc):
    res = client.post(
        "/api/search/query",
        json={"query": "how much annual leave", "top_k": 3},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["answer"]
    assert body["citations"], "an answer over an indexed corpus must carry citations"
    assert all(c["chunk_id"] is not None for c in body["citations"])
    for highlight in body["highlights"]:
        assert body["answer"][highlight["start"] : highlight["end"]] == highlight["term"]


def test_query_records_the_real_embedding_model_on_the_run(
    client, admin_token, ingested_doc, db_session
):
    """``embedding_model`` on the recorded run must equal the value the
    configured vector store genuinely reports (its public ``model``
    property), not a hardcoded string. A test that only checks the recorded
    value looks like ``hashing-<dim>`` cannot tell a real read from a
    literal that happens to match this config's dim, so it is compared
    directly against the store's own ``model`` attribute instead.
    """
    from ragfabric_core.models.runs import RetrievalRun
    from ragfabric_server.main import app

    client.post(
        "/api/search/query",
        json={"query": "leave", "top_k": 3},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    run = db_session.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run is not None
    expected = app.state.vector_store.model
    assert run.embedding_model == expected, (
        "embedding_model must be the provider's real model name, not a hardcoded string"
    )
    assert [s["name"] for s in run.trace][:2] == ["embed_query", "vector_search"]


def test_query_records_an_embedding_model_that_tracks_the_configured_dim(
    client, admin_token, monkeypatch, db_session
):
    """A hardcoded ``"hashing-128"`` in the route would still pass the test
    above, because this whole test session runs under one fixed ``dim: 128``
    config, and under that single config the real value genuinely IS
    ``"hashing-128"``. No assertion made against one fixed configuration can
    tell a computed read from a literal that happens to match it.

    This test changes the configured embedding dimension to 64 for a second,
    independent app lifespan (a fresh ``TestClient`` context manager, which
    reruns FastAPI's startup and rebuilds the embedder and vector store from
    whatever ``get_config()`` returns at that moment) and asserts the
    recorded ``embedding_model`` tracks that change, read again from the live
    store's own ``model`` property, not from a literal written in this test
    either. A route that always wrote ``"hashing-128"`` would fail here even
    though it would pass the fixed-dim test above; a literal can match at
    most one of the two configurations.
    """
    from ragfabric_core.models.runs import RetrievalRun
    from ragfabric_core.testing.fixtures import make_txt
    from ragfabric_server.main import app, get_config

    base_cfg = get_config()
    assert base_cfg.embeddings.dim == 128, (
        "this test assumes the session's fixed config is dim 128 (see conftest.py); "
        "if that ever changes, dim=64 below stops being a genuinely different value"
    )
    cfg64 = base_cfg.model_copy(
        update={"embeddings": base_cfg.embeddings.model_copy(update={"dim": 64})}
    )
    monkeypatch.setattr("ragfabric_server.main.get_config", lambda: cfg64)

    headers = {"Authorization": f"Bearer {admin_token}"}
    from fastapi.testclient import TestClient as _TestClient

    with _TestClient(app) as client64:
        assert app.state.vector_store.model == "hashing-64", (
            "the second lifespan did not pick up the patched dim=64 config"
        )
        upload = client64.post(
            "/api/documents/upload",
            files={
                "file": (
                    "dim64-policy.txt",
                    make_txt(
                        "Contractors receive fifteen days of paid leave per year, "
                        "separate from the employee leave policy."
                    ),
                    "text/plain",
                )
            },
            headers=headers,
        )
        assert upload.status_code == 201, upload.text
        assert upload.json()["status"] == "ready"

        client64.post(
            "/api/search/query",
            json={"query": "contractor leave", "top_k": 3},
            headers=headers,
        )
        expected = app.state.vector_store.model

    run = db_session.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run is not None
    assert expected == "hashing-64"
    assert run.embedding_model == expected, (
        "embedding_model must track the configured dim, not stay fixed at one literal"
    )
    assert run.embedding_model != "hashing-128"


def test_query_records_no_cost_estimate_on_the_run(client, admin_token, ingested_doc, db_session):
    """``estimated_cost_usd`` must stay ``None`` until real pricing is wired up.

    Phase 3 made the LLM and embedding calls real but has no price table
    plumbed in yet. Writing ``0.0`` would claim a computed, known-free cost,
    indistinguishable from a genuinely free Ollama run; ``None`` honestly
    records that no cost was computed.
    """
    from ragfabric_core.models.runs import RetrievalRun

    client.post(
        "/api/search/query",
        json={"query": "leave", "top_k": 3},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    run = db_session.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run is not None
    assert run.estimated_cost_usd is None


def test_semantic_returns_ranked_chunks_from_the_vector_store(client, admin_token, ingested_doc):
    res = client.post(
        "/api/search/semantic",
        json={"query": "leave", "top_k": 5},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 200
    results = res.json()["results"]
    assert results
    assert all(r["score"] is not None for r in results)


def test_hybrid_still_returns_results_after_the_rewire(client, admin_token, ingested_doc):
    res = client.post(
        "/api/search/hybrid",
        json={"query": "leave", "top_k": 5, "mode": "hybrid"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 200
    assert res.json()["results"]


def test_run_row_records_measured_llm_and_token_counts(
    client, admin_token, ingested_doc, db_session
):
    """RULING A: llm_calls/input_tokens/output_tokens must be measured, not fabricated.

    The offline test double answers on its first attempt (see
    ``_FakeCitingLLM`` in conftest.py), so exactly one model call is made:
    zero from the strategy's reranker (none configured) plus one from
    ``generate_cited_answer``'s accepted first attempt.
    """
    from ragfabric_core.models.runs import RetrievalRun

    client.post(
        "/api/search/query",
        json={"query": "how much annual leave", "top_k": 3},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    run = db_session.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run is not None
    assert run.llm_calls == 1
    assert run.input_tokens > 0
    assert run.output_tokens > 0
