"""The agentic strategy over the HTTP surface.

**The real schemas, read before anything was changed.** Recorded here because
Phase 4's notes say a plan that assumes a field exists is how a route task ends
up inventing one.

``AskRequest`` (``schemas/ask.py``): ``query``, ``top_k`` (1 to 50, default 8),
``similarity_threshold``, ``rerank`` (``none``/``llm``/``cross_encoder`` or
absent), ``collection_id``, ``document_id``, ``format``, ``strategy`` (a string
with the pattern ``^(traditional|vectorless)$``), ``stream`` (default true).

``SearchRequest`` (``schemas/search.py``): the same filters plus ``mode``
(``semantic``/``hybrid``) and ``strategy`` as a ``Literal["traditional",
"vectorless"]``.

``AnswerResponse``: ``question``, ``answer``, ``confidence``, ``citations``,
``highlights``, ``source_document``, ``usage``. It had no trace and no report
of the parts of a question, so this task adds both, defaulted to empty so the
two strategies that do not decompose are unchanged.

The ``/api/ask`` stream sends ``retrieval`` (``chunks``, ``strategy``,
``trace``), then ``token``, optionally ``superseded``, then ``citations`` and
``done`` (``run_id``, ``latency_ms``, ``usage``).

``/api/search/hybrid`` accepts ``traditional`` only, and that does not change:
it fuses a vector ranking with a lexical one, and the agent is neither.
"""

from __future__ import annotations

import json

import pytest

from ragfabric_core.agent.state import NodeName
from ragfabric_core.agent.tools import (
    FetchDocumentTool,
    LexicalSearchTool,
    SemanticSearchTool,
    build_tool_registry,
)
from ragfabric_core.db.session import SessionLocal
from ragfabric_core.providers.base import Completion, Message
from ragfabric_core.providers.offline import ScriptedLLMProvider
from ragfabric_core.stores.document_chunks import SqlDocumentChunkReader
from ragfabric_core.strategies.agentic import AgenticRAGStrategy
from ragfabric_server.deps import get_llm_provider
from ragfabric_server.main import app

QUESTION = "how much annual leave"

PLAN = {"sub_questions": [{"text": QUESTION, "tool": "semantic_search", "why": "a policy"}]}
ANSWERED = {"verdicts": [{"sub_question": QUESTION, "answered": True, "missing": None}]}


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def agentic(client):
    """Swap the registry's agent for one driven by a scripted model.

    The registered agent builds its own provider from configuration, which in
    this environment would reach for a local model server. The tools are the
    real ones over the real stores, so what is exercised here is the route, the
    strategy and the stores; only the model's three decisions are canned.
    """
    registry = client.app.state.strategy_registry
    traditional = registry.get("traditional")
    vectorless = registry.get("vectorless")
    llm = ScriptedLLMProvider(node_responses={NodeName.PLAN: [PLAN], NodeName.ASSESS: [ANSWERED]})
    registry.register(
        AgenticRAGStrategy(
            llm=llm,
            tools=build_tool_registry(
                [
                    SemanticSearchTool(traditional),
                    LexicalSearchTool(vectorless),
                    FetchDocumentTool(SqlDocumentChunkReader(SessionLocal)),
                ]
            ),
        )
    )
    return client


def test_the_ask_response_carries_the_sub_question_report(agentic, admin_token, ingested_doc):
    r = agentic.post(
        "/api/ask",
        json={"query": QUESTION, "strategy": "agentic", "stream": False},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    reports = r.json()["sub_questions"]
    assert [report["text"] for report in reports] == [QUESTION]
    assert reports[0]["status"] == "answered"
    assert reports[0]["reason"] is None
    assert reports[0]["chunk_ids"]


def test_the_ask_response_carries_the_agent_trace(agentic, admin_token, ingested_doc):
    r = agentic.post(
        "/api/ask",
        json={"query": QUESTION, "strategy": "agentic", "stream": False},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    names = [span["name"] for span in r.json()["trace"]]
    assert {"plan", "retrieve", "assess", "finalize"} <= set(names)


def test_the_query_endpoint_answers_through_the_agentic_strategy(
    agentic, admin_token, ingested_doc
):
    r = agentic.post(
        "/api/search/query",
        json={"query": QUESTION, "strategy": "agentic", "top_k": 3},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"]
    assert body["sub_questions"][0]["status"] == "answered"
    assert body["usage"]["retrieval_calls"] >= 1


def test_the_semantic_endpoint_accepts_the_agentic_strategy(agentic, admin_token, ingested_doc):
    r = agentic.post(
        "/api/search/semantic",
        json={"query": QUESTION, "strategy": "agentic"},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    assert r.json()["strategy"] == "agentic"


def test_hybrid_still_refuses_agentic(agentic, admin_token, ingested_doc):
    """Hybrid fuses a vector ranking with a lexical one. An agent is neither."""
    r = agentic.post(
        "/api/search/hybrid",
        json={"query": QUESTION, "strategy": "agentic"},
        headers=auth(admin_token),
    )
    assert r.status_code == 422, r.text
    assert "hybrid" in r.json()["detail"].lower()
    assert "agentic" in r.json()["detail"]


def test_the_stream_reports_the_sub_questions_alongside_the_trace(
    agentic, admin_token, ingested_doc
):
    with agentic.stream(
        "POST",
        "/api/ask",
        json={"query": QUESTION, "strategy": "agentic", "stream": True},
        headers=auth(admin_token),
    ) as res:
        assert res.status_code == 200
        events, name = {}, None
        for line in res.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and name:
                events[name] = json.loads(line.split(":", 1)[1].strip())

    assert events["retrieval"]["strategy"] == "agentic"
    assert events["retrieval"]["sub_questions"][0]["status"] == "answered"
    assert events["retrieval"]["trace"]


def test_the_run_records_the_agentic_strategy(agentic, admin_token, ingested_doc, db_session):
    from ragfabric_core.models.runs import RetrievalRun

    agentic.post(
        "/api/ask",
        json={"query": QUESTION, "strategy": "agentic", "stream": False},
        headers=auth(admin_token),
    )
    run = db_session.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run is not None
    assert run.requested_strategy == "agentic"
    assert run.selected_strategy == "agentic"


class _OverreachingLLM:
    """Answers with one supported claim and one the evidence does not carry."""

    name = "overreaching"
    default_model = "overreaching-test-model"

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        text = "Employees get twenty five days [1]. The board approved this in March."
        return Completion(
            text=text,
            model=self.default_model,
            provider=self.name,
            input_tokens=1,
            output_tokens=1,
            latency_ms=0,
            finish_reason="stop",
        )

    def stream(self, messages: list[Message], **kwargs):
        yield self.complete(messages).text


def test_an_unsupported_claim_is_dropped_on_the_ask_response(agentic, admin_token, ingested_doc):
    """Dropped, and not retried: the removal is reported instead of hidden."""
    app.dependency_overrides[get_llm_provider] = lambda: _OverreachingLLM()
    try:
        r = agentic.post(
            "/api/ask",
            json={"query": QUESTION, "strategy": "agentic", "stream": False},
            headers=auth(admin_token),
        )
    finally:
        app.dependency_overrides.pop(get_llm_provider, None)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Employees get twenty five days [1]."
    assert [claim["text"] for claim in body["dropped_claims"]] == [
        "The board approved this in March."
    ]


def test_the_other_strategies_report_no_sub_questions(client, admin_token, ingested_doc):
    """The new fields are additive: nothing about traditional or vectorless moves."""
    for strategy in ("traditional", "vectorless"):
        r = client.post(
            "/api/ask",
            json={"query": QUESTION, "strategy": strategy, "stream": False},
            headers=auth(admin_token),
        )
        assert r.status_code == 200, r.text
        assert r.json()["sub_questions"] == []
        assert r.json()["dropped_claims"] == []
