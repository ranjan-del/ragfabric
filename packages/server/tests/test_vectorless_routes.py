"""The vectorless strategy over the HTTP surface.

The plan writes these against ``POST /v1/search`` and ``POST /v1/ask``. Those
paths do not exist: the real routes are ``/api/search/{query,semantic,hybrid}``
and ``/api/ask``. The plan also assumes ``SearchRequest.strategy``,
``SearchResults.strategy`` and a ``usage`` block on the answer, none of which
existed either. All three are added by this task, which is what the plan's own
"a Phase 3 defect to avoid repeating" note asks for: confirm the field exists,
and if it does not, add it and say so.
"""

from __future__ import annotations


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_search_accepts_the_vectorless_strategy(client, admin_token, ingested_doc):
    r = client.post(
        "/api/search/semantic",
        json={"query": "annual leave", "strategy": "vectorless"},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    assert r.json()["strategy"] == "vectorless"


def test_search_still_defaults_to_traditional(client, admin_token, ingested_doc):
    r = client.post(
        "/api/search/semantic", json={"query": "annual leave"}, headers=auth(admin_token)
    )
    assert r.status_code == 200, r.text
    assert r.json()["strategy"] == "traditional"


def test_the_query_endpoint_answers_through_the_vectorless_strategy(
    client, admin_token, ingested_doc
):
    r = client.post(
        "/api/search/query",
        json={"query": "how much annual leave", "strategy": "vectorless", "top_k": 3},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"]
    assert body["citations"]
    assert body["usage"]["embedding_calls"] == 0


def test_vectorless_reports_zero_embedding_calls(client, admin_token, ingested_doc):
    r = client.post(
        "/api/ask",
        json={"query": "how much annual leave", "strategy": "vectorless", "stream": False},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    assert r.json()["usage"]["embedding_calls"] == 0


def test_traditional_reports_one_embedding_call(client, admin_token, ingested_doc):
    """The zero above has to be a measurement, not a constant (ADR 0004)."""
    r = client.post(
        "/api/ask",
        json={"query": "how much annual leave", "strategy": "traditional", "stream": False},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    assert r.json()["usage"]["embedding_calls"] == 1


def test_an_unknown_strategy_is_a_422_not_a_500(client, admin_token):
    assert (
        client.post(
            "/api/search/semantic",
            json={"query": "x", "strategy": "nope"},
            headers=auth(admin_token),
        ).status_code
        == 422
    )


def test_an_unknown_strategy_on_ask_is_a_422_not_a_500(client, admin_token):
    assert (
        client.post(
            "/api/ask",
            json={"query": "x", "strategy": "nope", "stream": False},
            headers=auth(admin_token),
        ).status_code
        == 422
    )


def test_hybrid_refuses_a_lexical_only_strategy_rather_than_fusing_lexical_with_itself(
    client, admin_token, ingested_doc
):
    r = client.post(
        "/api/search/hybrid",
        json={"query": "annual leave", "strategy": "vectorless"},
        headers=auth(admin_token),
    )
    assert r.status_code == 422, r.text
    assert "hybrid" in r.json()["detail"].lower()


def test_the_stream_reports_the_strategy_and_its_usage(client, admin_token, ingested_doc):
    import json

    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "how much annual leave", "strategy": "vectorless", "stream": True},
        headers=auth(admin_token),
    ) as res:
        assert res.status_code == 200
        events, name = {}, None
        for line in res.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and name:
                events[name] = json.loads(line.split(":", 1)[1].strip())
    assert events["retrieval"]["strategy"] == "vectorless"
    assert events["done"]["usage"]["embedding_calls"] == 0


def test_the_run_records_the_strategy_that_actually_served_it(
    client, admin_token, ingested_doc, db_session
):
    from ragfabric_core.models.runs import RetrievalRun

    client.post(
        "/api/search/query",
        json={"query": "annual leave", "strategy": "vectorless"},
        headers=auth(admin_token),
    )
    run = db_session.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run is not None
    assert run.requested_strategy == "vectorless"
    assert run.selected_strategy == "vectorless"
