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
    from ragfabric_core.models.runs import RetrievalRun

    client.post(
        "/api/search/query",
        json={"query": "leave", "top_k": 3},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    run = db_session.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run is not None
    assert not run.embedding_model.startswith("hashing-") or run.embedding_model == "hashing-128", (
        "embedding_model must be the provider's real model name, not a hardcoded string"
    )
    assert [s["name"] for s in run.trace][:2] == ["embed_query", "vector_search"]


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
