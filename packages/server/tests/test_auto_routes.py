"""The auto strategy over the HTTP surface (Phase 7a, Task 11).

The strategy under test is a real ``AutoStrategy`` over the app's own stores,
registered in place of the default one so its classifier is a canned model
rather than whatever ``ragfabric.yaml`` would build. Questions are chosen for
the rule they trip: a short plain one routes by signals alone, a long open one
reaches the classifier, and a relationship one points at the graph.
"""

from __future__ import annotations

import json

import pytest

from ragfabric_core import runtime
from ragfabric_core.config_file import RouterConfig
from ragfabric_core.db.session import SessionLocal
from ragfabric_core.models.access import AuditLog
from ragfabric_core.models.runs import RetrievalRun
from ragfabric_core.providers.base import Completion, Message, ProviderError
from ragfabric_core.strategies.auto import AutoStrategy
from ragfabric_core.strategies.base import StrategyName
from ragfabric_core.strategies.graph import GraphRAGStrategy
from ragfabric_server.api.routes import ask as ask_route
from ragfabric_server.api.routes import search as search_route

PLAIN = "how much annual leave"
# Longer than the router's plain limit and firing no rule, so the classifier decides.
OPEN = (
    "could you explain what employees are entitled to when it comes to annual leave "
    "and public holidays during a normal working year"
)
RELATIONSHIP = "who does Platform Team report to"
CLASSIFIER_CONFIDENCE = 0.9
CLASSIFIER_REASONING = "The question asks for an exact entitlement."


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class _Classifier:
    """Answers the router's one call with a fixed, confident verdict."""

    name = "classifier"
    default_model = "classifier-test-model"

    def __init__(self, strategy: str = "vectorless") -> None:
        self.strategy = strategy
        self.calls = 0

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        self.calls += 1
        text = json.dumps(
            {
                "query_type": "exact_match",
                "strategy": self.strategy,
                "confidence": CLASSIFIER_CONFIDENCE,
                "reasoning": CLASSIFIER_REASONING,
            }
        )
        return Completion(
            text=text,
            model=self.default_model,
            provider=self.name,
            input_tokens=3,
            output_tokens=5,
            latency_ms=0,
            finish_reason="stop",
        )


class _NoEntities:
    """A graph question call that finds nothing to walk from."""

    name = "no-entities"
    default_model = "no-entities-test-model"

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        return Completion(
            text=json.dumps({"entities": [], "implied_relation_types": []}),
            model=self.default_model,
            provider=self.name,
            input_tokens=1,
            output_tokens=1,
            latency_ms=0,
            finish_reason="stop",
        )


@pytest.fixture()
def auto(client, ingested_doc):
    """A real AutoStrategy with the graph enabled and a canned classifier and graph model."""
    registry = client.app.state.strategy_registry
    registry.register(GraphRAGStrategy(llm=_NoEntities(), session_factory=SessionLocal))
    classifier = _Classifier()
    registry.register(
        AutoStrategy(
            registry=registry,
            llm=classifier,
            min_confidence=0.6,
            classifier_model=None,
            graph_enabled=True,
            relation_types=["REPORTS_TO", "MEMBER_OF"],
        )
    )
    return classifier


@pytest.fixture()
def router_mode():
    """Set router.mode for one test and put the configuration back afterwards."""

    def _set(mode: str) -> None:
        cfg = runtime.get_config()
        runtime._config = cfg.model_copy(update={"router": RouterConfig(mode=mode)})

    yield _set
    runtime.reset_config()


def _ask(client, token: str, **body):
    return client.post(
        "/api/ask", json={"query": PLAIN, "stream": False, **body}, headers=auth(token)
    )


def _query(client, token: str, **body):
    return client.post("/api/search/query", json={"query": PLAIN, **body}, headers=auth(token))


def _runs() -> list[RetrievalRun]:
    with SessionLocal() as db:
        rows = db.query(RetrievalRun).order_by(RetrievalRun.id).all()
        db.expunge_all()
        return rows


def _last_run() -> RetrievalRun:
    return _runs()[-1]


def test_auto_is_accepted_on_ask_and_query(client, admin_token, auto):
    not_streamed = _ask(client, admin_token, strategy="auto")
    streamed = _ask(client, admin_token, strategy="auto", stream=True)
    queried = _query(client, admin_token, strategy="auto")

    assert not_streamed.status_code == 200, not_streamed.text
    assert streamed.status_code == 200, streamed.text
    assert "event: done" in streamed.text
    assert queried.status_code == 200, queried.text
    assert [run.requested_strategy for run in _runs()] == ["auto", "auto", "auto"]
    assert (
        client.post(
            "/api/ask", json={"query": PLAIN, "strategy": "bogus"}, headers=auth(admin_token)
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    ("mode", "requested", "run_mode"),
    [("auto", "auto", "auto"), ("manual", "traditional", "manual")],
)
def test_an_unset_strategy_resolves_from_router_mode(
    client, admin_token, auto, router_mode, mode, requested, run_mode
):
    router_mode(mode)

    assert _ask(client, admin_token).status_code == 200
    assert _query(client, admin_token).status_code == 200

    runs = _runs()
    assert len(runs) == 2
    assert [run.requested_strategy for run in runs] == [requested, requested]
    assert [run.mode for run in runs] == [run_mode, run_mode]


def test_the_run_row_records_the_strategy_that_ran(client, admin_token, auto):
    r = _ask(client, admin_token, strategy="auto", query=OPEN)

    assert r.status_code == 200, r.text
    run = _last_run()
    assert run.mode == "auto"
    assert run.requested_strategy == "auto"
    assert run.selected_strategy == "vectorless"
    assert run.router_confidence == pytest.approx(CLASSIFIER_CONFIDENCE)
    assert run.router_reasoning == CLASSIFIER_REASONING
    assert run.fallback_from is None

    # The graph is chosen by signals, finds nothing to walk from, and traditional answers.
    r = _ask(client, admin_token, strategy="auto", query=RELATIONSHIP)

    assert r.status_code == 200, r.text
    run = _last_run()
    assert run.selected_strategy == "traditional"
    assert run.fallback_from == "graph"
    # A rule is not a measurement: no confidence is invented for it.
    assert run.router_confidence is None
    assert run.router_reasoning


def test_the_response_carries_the_decision(client, admin_token, auto):
    routed = _ask(client, admin_token, strategy="auto", query=OPEN).json()
    fell_back = _query(client, admin_token, strategy="auto", query=RELATIONSHIP).json()
    named = _ask(client, admin_token, strategy="traditional").json()

    assert routed["strategy"] == "vectorless"
    assert routed["router"]["selected_strategy"] == "vectorless"
    assert routed["router"]["source"] == "classifier"
    assert routed["router"]["confidence"] == pytest.approx(CLASSIFIER_CONFIDENCE)
    assert routed["fallback_from"] is None
    assert fell_back["strategy"] == "traditional"
    assert fell_back["fallback_from"] == "graph"
    assert fell_back["router"]["selected_strategy"] == "graph"
    assert named["strategy"] == "traditional"
    assert named["router"] is None
    assert named["fallback_from"] is None


def test_the_streamed_retrieval_event_carries_the_decision(client, admin_token, auto):
    r = _ask(client, admin_token, strategy="auto", query=OPEN, stream=True)

    retrieval = next(
        json.loads(block.split("data: ", 1)[1])
        for block in r.text.split("\n\n")
        if block.startswith("event: retrieval")
    )
    assert retrieval["strategy"] == "vectorless"
    assert retrieval["router"]["source"] == "classifier"
    assert retrieval["fallback_from"] is None


def test_hybrid_still_refuses_auto(client, admin_token, auto):
    r = client.post(
        "/api/search/hybrid", json={"query": PLAIN, "strategy": "auto"}, headers=auth(admin_token)
    )

    assert r.status_code == 422
    assert "hybrid" in r.text


def test_graph_named_with_a_document_filter_is_still_422(client, admin_token, auto, ingested_doc):
    by_document = _ask(client, admin_token, strategy="graph", document_id=ingested_doc["id"])
    by_format = _query(client, admin_token, strategy="graph", format="txt")

    assert by_document.status_code == 422
    assert by_format.status_code == 422
    assert "graph" in by_document.json()["detail"]
    assert _runs() == []


def test_auto_with_a_document_filter_never_walks_the_graph(client, admin_token, auto, ingested_doc):
    r = _ask(
        client,
        admin_token,
        strategy="auto",
        query=RELATIONSHIP,
        document_id=ingested_doc["id"],
    )

    assert r.status_code == 200, r.text
    run = _last_run()
    assert run.selected_strategy != "graph"
    assert run.fallback_from is None
    assert r.json()["router"]["selected_strategy"] != "graph"


def test_access_stats_are_measured_on_the_strategy_that_ran(client, admin_token, auto, monkeypatch):
    registry = client.app.state.strategy_registry
    counted: list[object] = []
    real = search_route._access_stats

    def spy(strategy, filters, access):
        counted.append(strategy)
        return real(strategy, filters, access)

    monkeypatch.setattr(ask_route, "_access_stats", spy)
    monkeypatch.setattr(search_route, "_access_stats", spy)

    assert _ask(client, admin_token, strategy="auto", query=OPEN).status_code == 200
    assert _query(client, admin_token, strategy="auto", query=OPEN).status_code == 200

    assert len(counted) == 2
    assert not any(isinstance(item, AutoStrategy) for item in counted)
    assert counted == [registry.get("vectorless")] * 2
    with SessionLocal() as db:
        assert db.query(AuditLog).filter(AuditLog.action == "query").count() == 2


@pytest.mark.parametrize("path", ["query", "semantic", "hybrid"])
def test_a_bad_strategy_name_is_a_422_on_every_search_route(client, admin_token, auto, path):
    r = client.post(
        f"/api/search/{path}",
        json={"query": PLAIN, "strategy": "bogus"},
        headers=auth(admin_token),
    )

    assert r.status_code == 422


def test_semantic_with_auto_reports_and_counts_the_strategy_that_ran(
    client, admin_token, auto, monkeypatch
):
    registry = client.app.state.strategy_registry
    counted: list[object] = []
    real = search_route._access_stats

    def spy(strategy, filters, access):
        counted.append(strategy)
        return real(strategy, filters, access)

    monkeypatch.setattr(search_route, "_access_stats", spy)

    r = client.post(
        "/api/search/semantic",
        json={"query": OPEN, "strategy": "auto"},
        headers=auth(admin_token),
    )

    assert r.status_code == 200, r.text
    assert r.json()["strategy"] == "vectorless"
    assert counted == [registry.get("vectorless")]
    assert not any(isinstance(item, AutoStrategy) for item in counted)


class _Unreachable:
    """A strategy whose model cannot be reached."""

    def __init__(self, name) -> None:
        self.name = name

    def retrieve(self, query, ctx):
        raise ProviderError("graph-llm", "connection refused")


def test_auto_contains_a_routed_strategys_failure(client, admin_token, auto, router_mode):
    router_mode("auto")
    client.app.state.strategy_registry.register(_Unreachable(StrategyName.GRAPH))

    asked = _ask(client, admin_token, query=RELATIONSHIP)
    queried = _query(client, admin_token, query=RELATIONSHIP)

    assert asked.status_code == 200, asked.text
    assert queried.status_code == 200, queried.text
    assert asked.json()["strategy"] == "traditional"
    assert asked.json()["fallback_from"] == "graph"
    assert queried.json()["strategy"] == "traditional"
    run = _last_run()
    assert run.selected_strategy == "traditional" and run.fallback_from == "graph"


def _semantic(client, token: str, **body):
    return client.post("/api/search/semantic", json={"query": PLAIN, **body}, headers=auth(token))


@pytest.mark.parametrize("send", [_ask, _query, _semantic])
def test_rerank_with_no_strategy_resolves_to_traditional_under_auto(
    client, admin_token, auto, router_mode, send
):
    # R31: a caller asking for a reranker is asking for the one strategy that
    # applies it, so an unset strategy must not be handed to the router.
    router_mode("auto")

    r = send(client, admin_token, query=OPEN, rerank="none")

    assert r.status_code == 200, r.text
    assert r.json()["strategy"] == "traditional"
    assert auto.calls == 0


def test_rerank_with_no_strategy_records_traditional_as_requested(
    client, admin_token, auto, router_mode
):
    router_mode("auto")

    assert _ask(client, admin_token, query=OPEN, rerank="none").status_code == 200
    assert _query(client, admin_token, query=OPEN, rerank="none").status_code == 200

    assert [run.requested_strategy for run in _runs()] == ["traditional", "traditional"]


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_semantic_with_no_strategy_is_traditional_whatever_the_router_mode(
    client, admin_token, auto, router_mode, mode
):
    # R30: /semantic, like /hybrid, keeps traditional when no strategy is named.
    router_mode(mode)

    r = _semantic(client, admin_token, query=OPEN)

    assert r.status_code == 200, r.text
    assert r.json()["strategy"] == "traditional"
    assert auto.calls == 0


def test_semantic_still_accepts_an_explicit_auto(client, admin_token, auto, router_mode):
    router_mode("manual")

    r = _semantic(client, admin_token, query=OPEN, strategy="auto")

    assert r.status_code == 200, r.text
    assert r.json()["strategy"] == "vectorless"
    assert auto.calls == 1


def test_a_stored_auto_run_reports_the_routers_confidence_and_reasoning(client, admin_token, auto):
    """GET /api/runs/{id} returns what the router decided, for runs nobody streamed.

    The Trace page reads a run cold, long after the stream that produced it,
    so the two router columns already written on every auto run have to be
    readable through the API, not only through the database.
    """
    r = _ask(client, admin_token, strategy="auto", query=OPEN)
    assert r.status_code == 200, r.text
    routed = client.get(f"/api/runs/{_last_run().id}", headers=auth(admin_token)).json()

    assert routed["router_confidence"] == pytest.approx(CLASSIFIER_CONFIDENCE)
    assert routed["router_reasoning"] == CLASSIFIER_REASONING

    r = _ask(client, admin_token, strategy="traditional")
    assert r.status_code == 200, r.text
    manual = client.get(f"/api/runs/{_last_run().id}", headers=auth(admin_token)).json()

    # Nothing routed a manual run, so nothing is reported, never a made up value.
    assert manual["router_confidence"] is None
    assert manual["router_reasoning"] is None
