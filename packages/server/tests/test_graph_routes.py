"""The graph strategy over the HTTP surface (Task 12, ruling R37).

The strategy under test is the real ``GraphRAGStrategy`` over the app's own
database, with entities and relationships seeded on the chunks of documents
uploaded through the API: ingestion runs graph extraction only when
``graph_store.enabled`` is set, and a route test must not depend on a model
extracting anything. Only the two model calls are canned: the question call
(which entities the question names) and the answer, which the fake below
writes from the relationships the prompt actually lists, so the edge and
passage numbers it cites are the ones the route rendered.
"""

from __future__ import annotations

import json
import re

import pytest

from ragfabric_core.db.session import SessionLocal
from ragfabric_core.graph.contracts import normalise
from ragfabric_core.models.document import Chunk
from ragfabric_core.models.graph import Entity, EntitySource, Relationship, RelationshipSource
from ragfabric_core.providers.base import Completion, Message
from ragfabric_core.strategies.graph import GraphRAGStrategy
from ragfabric_core.testing.fixtures import make_txt
from ragfabric_server.deps import get_llm_provider
from ragfabric_server.main import app

QUESTION = "what is the platform team part of"
ALLOWED_TEXT = "The Platform Team is a member of Engineering."
SECRET_TEXT = "The Platform Team works on Nightjar, a confidential acquisition."


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class _QuestionLLM:
    """Answers every question call with the same entity mention."""

    name = "question"
    default_model = "question-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        self.calls += 1
        text = json.dumps(
            {
                "entities": [{"name": "Platform Team", "entity_type": "team"}],
                "implied_relation_types": [],
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


_EDGE_LINE = re.compile(r"^\[E (\d+)\] (.+?) ([A-Z][A-Z_]+) (.+?) \(source passages: (.*)\)$", re.M)


class _GraphAnswerLLM:
    """Writes one supported relationship claim and one that cites a missing edge.

    It reads the ``Relationships:`` block ``build_graph_prompt`` rendered, finds
    the MEMBER_OF edge, and cites it with its first listed source passage, so
    the supported claim is supported only if the route really passed the
    sub-graph and the edge's backing chunk to the generator.
    """

    name = "graph-answer"
    default_model = "graph-answer-test-model"

    def _text(self, messages: list[Message]) -> str:
        prompt = next(m.content for m in messages if m.role == "user")
        claims = []
        for marker, source, relation, target, passages in _EDGE_LINE.findall(prompt):
            cited = re.findall(r"\[\d+\]", passages)
            if relation == "MEMBER_OF" and cited:
                claims.append(f"The {source} is a member of {target} [E {marker}] {cited[0]}.")
        claims.append("Engineering reports to the Board [E 99] [1].")
        return " ".join(claims)

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        text = self._text(messages)
        return Completion(
            text=text,
            model=self.default_model,
            provider=self.name,
            input_tokens=7,
            output_tokens=11,
            latency_ms=0,
            finish_reason="stop",
        )

    def stream(self, messages: list[Message], **kwargs):
        words = self._text(messages).split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


def _upload(client, token: str, filename: str, text: str, collection_id: int | None = None):
    response = client.post(
        "/api/documents/upload",
        files={"file": (filename, make_txt(text), "text/plain")},
        data={"collection_id": str(collection_id)} if collection_id is not None else None,
        headers=auth(token),
    )
    assert response.status_code == 201, response.text
    document = response.json()
    assert document["status"] == "ready", document
    return document["id"]


def _chunk_ids(document_id: int) -> list[int]:
    with SessionLocal() as db:
        return [
            row.id
            for row in db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.id)
        ]


def _seed(entities: dict[str, tuple[str, list[int]]], edges: list[tuple]) -> dict[str, int]:
    """Entities as name -> (type, source chunk ids); edges as (source, relation, target, chunks)."""
    ids: dict[str, int] = {}
    with SessionLocal() as db:
        for name, (entity_type, chunks) in entities.items():
            entity = Entity(name=name, normalized_name=normalise(name), entity_type=entity_type)
            db.add(entity)
            db.flush()
            db.add_all(EntitySource(entity_id=entity.id, chunk_id=c) for c in chunks)
            ids[name] = entity.id
        for source, relation, target, chunks in edges:
            edge = Relationship(
                source_entity_id=ids[source],
                target_entity_id=ids[target],
                relation_type=relation,
                confidence=0.8,
            )
            db.add(edge)
            db.flush()
            db.add_all(RelationshipSource(relationship_id=edge.id, chunk_id=c) for c in chunks)
        db.commit()
    return ids


def _register_graph(client) -> _QuestionLLM:
    llm = _QuestionLLM()
    client.app.state.strategy_registry.register(
        GraphRAGStrategy(llm=llm, session_factory=SessionLocal)
    )
    return llm


@pytest.fixture()
def graph_answers():
    app.dependency_overrides[get_llm_provider] = lambda: _GraphAnswerLLM()
    yield
    app.dependency_overrides.pop(get_llm_provider, None)


@pytest.fixture()
def team_graph(client, admin_token, graph_answers) -> dict:
    """Platform Team MEMBER_OF Engineering, sourced by one open document."""
    document_id = _upload(client, admin_token, "org.txt", ALLOWED_TEXT)
    [chunk_id] = _chunk_ids(document_id)
    ids = _seed(
        {"Platform Team": ("team", [chunk_id]), "Engineering": ("organisation", [chunk_id])},
        [("Platform Team", "MEMBER_OF", "Engineering", [chunk_id])],
    )
    _register_graph(client)
    return {"ids": ids, "chunk_id": chunk_id, "document_id": document_id}


def _ask(client, token: str, **body):
    return client.post(
        "/api/ask",
        json={"query": QUESTION, "strategy": "graph", "stream": False, **body},
        headers=auth(token),
    )


def test_the_ask_response_carries_the_subgraph(client, admin_token, team_graph):
    r = _ask(client, admin_token)

    assert r.status_code == 200, r.text
    subgraph = r.json()["subgraph"]
    ids = team_graph["ids"]
    assert {(n["id"], n["name"], n["entity_type"], n["depth"]) for n in subgraph["nodes"]} == {
        (ids["Platform Team"], "Platform Team", "team", 0),
        (ids["Engineering"], "Engineering", "organisation", 1),
    }
    [edge] = subgraph["edges"]
    assert edge["source_id"] == ids["Platform Team"]
    assert edge["target_id"] == ids["Engineering"]
    assert edge["relation_type"] == "MEMBER_OF"
    assert edge["walked_as"] == "MEMBER_OF"
    assert edge["reversed"] is False
    assert edge["confidence"] == 0.8
    assert edge["source_chunk_ids"] == [team_graph["chunk_id"]]
    assert subgraph["truncated"] is False
    assert subgraph["empty_reason"] is None


def test_the_subgraph_carries_no_descriptions(client, admin_token, team_graph):
    """R6: a stored description may paraphrase a denied chunk, so none is sent."""
    with SessionLocal() as db:
        for entity in db.query(Entity).all():
            entity.description = "SECRET DESCRIPTION"
        for edge in db.query(Relationship).all():
            edge.description = "SECRET DESCRIPTION"
        db.commit()

    r = _ask(client, admin_token)

    assert r.status_code == 200, r.text
    assert "SECRET DESCRIPTION" not in r.text
    subgraph = r.json()["subgraph"]
    assert all("description" not in node for node in subgraph["nodes"])
    assert all("description" not in edge for edge in subgraph["edges"])


def test_the_graph_answer_goes_through_the_graph_contract(client, admin_token, team_graph):
    r = _ask(client, admin_token)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "The Platform Team is a member of Engineering [E 1] [1]."
    assert body["dropped_relationship_claims"] == [
        {"text": "Engineering reports to the Board [E 99] [1].", "reason": "edge_not_in_subgraph"}
    ]
    assert body["dropped_claims"] == []
    # One question call plus exactly one answer call, no retry.
    assert body["usage"]["llm_calls"] == 2
    assert body["usage"]["embedding_calls"] == 0
    assert [c["chunk_id"] for c in body["citations"] if c["used"]] == [team_graph["chunk_id"]]


def test_the_query_endpoint_returns_the_subgraph(client, admin_token, team_graph):
    r = client.post(
        "/api/search/query",
        json={"query": QUESTION, "strategy": "graph", "top_k": 3},
        headers=auth(admin_token),
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert [edge["relation_type"] for edge in body["subgraph"]["edges"]] == ["MEMBER_OF"]
    assert body["answer"] == "The Platform Team is a member of Engineering [E 1] [1]."
    assert [claim["reason"] for claim in body["dropped_relationship_claims"]] == [
        "edge_not_in_subgraph"
    ]


def test_hybrid_still_refuses_graph(client, admin_token, team_graph):
    r = client.post(
        "/api/search/hybrid",
        json={"query": QUESTION, "strategy": "graph"},
        headers=auth(admin_token),
    )
    assert r.status_code == 422, r.text
    assert "hybrid" in r.json()["detail"].lower()
    assert "graph" in r.json()["detail"]


@pytest.mark.parametrize("field,value", [("document_id", 1), ("format", "txt")])
def test_graph_refuses_a_filter_it_cannot_apply(client, admin_token, team_graph, field, value):
    """The walk honours access and collection scope, not document or format filters.

    Serving the request would return entities and chunks from documents the
    caller asked to exclude, so it is refused rather than silently widened.
    """
    for path, body in (
        ("/api/ask", {"stream": False}),
        ("/api/search/query", {}),
        ("/api/search/semantic", {}),
    ):
        r = client.post(
            path,
            json={"query": QUESTION, "strategy": "graph", field: value, **body},
            headers=auth(admin_token),
        )
        assert r.status_code == 422, (path, r.text)
        assert field in r.json()["detail"]


def test_the_other_strategies_report_no_subgraph(client, admin_token, team_graph):
    r = client.post(
        "/api/ask",
        json={"query": QUESTION, "strategy": "traditional", "stream": False},
        headers=auth(admin_token),
    )
    assert r.status_code == 200, r.text
    assert r.json()["subgraph"] is None
    assert r.json()["dropped_relationship_claims"] == []


def test_the_run_records_the_graph_strategy(client, admin_token, team_graph, db_session):
    from ragfabric_core.models.access import AuditLog
    from ragfabric_core.models.runs import RetrievalRun

    _ask(client, admin_token)

    run = db_session.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run.requested_strategy == "graph"
    assert run.selected_strategy == "graph"
    assert run.llm_calls == 2
    audit = db_session.query(AuditLog).filter(AuditLog.retrieval_run_id == run.id).one()
    assert audit.strategy == "graph"
    assert audit.sources_filtered == 0


def _events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.strip().split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        if name:
            out.append((name, data))
    return out


def test_the_stream_carries_the_subgraph_and_the_dropped_claims(client, admin_token, team_graph):
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": QUESTION, "strategy": "graph", "stream": True},
        headers=auth(admin_token),
    ) as res:
        assert res.status_code == 200
        events = _events("".join(res.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "retrieval" and names[-2:] == ["citations", "done"]
    by_name = dict(events)
    assert by_name["retrieval"]["strategy"] == "graph"
    assert [e["relation_type"] for e in by_name["retrieval"]["subgraph"]["edges"]] == ["MEMBER_OF"]

    superseded = by_name["superseded"]
    assert superseded["text"] == "The Platform Team is a member of Engineering [E 1] [1]."
    assert superseded["dropped_relationship_claims"] == [
        {"text": "Engineering reports to the Board [E 99] [1].", "reason": "edge_not_in_subgraph"}
    ]
    assert superseded["dropped_claims"] == []
    assert by_name["done"]["usage"]["llm_calls"] == 2

    run = client.get(f"/api/runs/{by_name['done']['run_id']}", headers=auth(admin_token)).json()
    assert run["answer"] == superseded["text"]


# ---------------------------------------------------------------------------
# Access: a denied document's entities and chunks never reach the response
# ---------------------------------------------------------------------------


@pytest.fixture()
def restricted_token(client) -> str:
    client.post(
        "/api/auth/register", json={"email": "restricted@example.com", "password": "password123"}
    )
    r = client.post(
        "/api/auth/login", data={"username": "restricted@example.com", "password": "password123"}
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture()
def split_graph(client, admin_token, restricted_token, graph_answers) -> dict:
    """Platform Team is sourced by an open and a secret chunk; Nightjar only by the secret one."""
    admin = auth(admin_token)
    secret = client.post("/api/collections", json={"name": "m-and-a"}, headers=admin).json()
    secret_document = _upload(client, admin_token, "nightjar.txt", SECRET_TEXT, secret["id"])
    open_document = _upload(client, restricted_token, "org.txt", ALLOWED_TEXT)
    [secret_chunk] = _chunk_ids(secret_document)
    [open_chunk] = _chunk_ids(open_document)
    ids = _seed(
        {
            "Platform Team": ("team", [open_chunk, secret_chunk]),
            "Engineering": ("organisation", [open_chunk]),
            "Nightjar": ("project", [secret_chunk]),
        },
        [
            ("Platform Team", "MEMBER_OF", "Engineering", [open_chunk]),
            ("Platform Team", "WORKS_ON", "Nightjar", [secret_chunk]),
        ],
    )
    _register_graph(client)

    # Not vacuous: before the collection is locked down, the secret side is
    # genuinely reachable through this same route and question.
    control = _ask(client, admin_token, top_k=10).json()
    assert ids["Nightjar"] in {node["id"] for node in control["subgraph"]["nodes"]}
    assert secret_chunk in {c["chunk_id"] for c in control["citations"]}

    group = client.post("/api/admin/groups", json={"name": "m-and-a-only"}, headers=admin).json()
    client.post(
        "/api/admin/grants",
        json={"group_id": group["id"], "collection_id": secret["id"], "permission": "read"},
        headers=admin,
    )
    return {
        "ids": ids,
        "secret_chunk": secret_chunk,
        "secret_document": secret_document,
        "open_chunk": open_chunk,
    }


def test_a_denied_documents_entities_and_chunks_never_reach_the_ask_response(
    client, restricted_token, split_graph
):
    r = _ask(client, restricted_token, top_k=10)

    assert r.status_code == 200, r.text
    body = r.json()
    secret_chunk = split_graph["secret_chunk"]
    node_ids = {node["id"] for node in body["subgraph"]["nodes"]}
    assert split_graph["ids"]["Nightjar"] not in node_ids
    assert all(edge["relation_type"] != "WORKS_ON" for edge in body["subgraph"]["edges"])
    assert all(secret_chunk not in edge["source_chunk_ids"] for edge in body["subgraph"]["edges"])
    assert all(c["chunk_id"] != secret_chunk for c in body["citations"])
    assert all(c["document_id"] != split_graph["secret_document"] for c in body["citations"])
    assert "Nightjar" not in r.text
    assert "confidential acquisition" not in r.text
    # Not vacuous: the restricted caller still walked the open side of the graph.
    assert split_graph["ids"]["Platform Team"] in node_ids
    assert split_graph["ids"]["Engineering"] in node_ids
    assert body["answer"] == "The Platform Team is a member of Engineering [E 1] [1]."


def test_a_denied_documents_entities_and_chunks_never_reach_the_streamed_ask_response(
    client, restricted_token, split_graph
):
    """The streaming generator is a separate code path from ``_ask``'s ``stream: False``.

    It builds its own prompt (``build_graph_prompt``) and emits the sub-graph on the
    ``retrieval`` event rather than in the final JSON body, so the route-level
    non-streaming test above does not exercise it. ``split_graph`` already proved, via
    its own admin control run, that the secret side is reachable through this same
    question before the collection is locked down; this test checks every SSE event a
    restricted caller receives afterwards, not just the final response body.
    """
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": QUESTION, "strategy": "graph", "stream": True, "top_k": 10},
        headers=auth(restricted_token),
    ) as res:
        assert res.status_code == 200
        raw = "".join(res.iter_text())

    events = _events(raw)
    by_name = dict(events)
    secret_chunk = split_graph["secret_chunk"]
    subgraph = by_name["retrieval"]["subgraph"]
    node_ids = {node["id"] for node in subgraph["nodes"]}

    assert split_graph["ids"]["Nightjar"] not in node_ids
    assert all(edge["relation_type"] != "WORKS_ON" for edge in subgraph["edges"])
    assert all(secret_chunk not in edge["source_chunk_ids"] for edge in subgraph["edges"])
    citations = by_name["citations"]["citations"]
    assert all(c["chunk_id"] != secret_chunk for c in citations)
    assert all(c["document_id"] != split_graph["secret_document"] for c in citations)
    # Across every event on the wire, not just one of them: the streamed tokens,
    # any superseded repair, the retrieval event and the final citations.
    assert "Nightjar" not in raw
    assert "confidential acquisition" not in raw
    # Not vacuous: the restricted caller still walked the open side of the graph.
    assert split_graph["ids"]["Platform Team"] in node_ids
    assert split_graph["ids"]["Engineering"] in node_ids
    assert by_name["done"] is not None


class _NoDropGraphAnswerLLM:
    """Streams a leading fragment that carries no words and no ``[E k]``/``[n]`` marker.

    ``apply_graph_contract`` discards a claim like this without recording it in
    either drop list (its docstring: "one with no marker ... is discarded without a
    record"), which used to make the stream compare the checked text against the raw
    streamed text, see a difference, and announce a ``superseded`` repair with two
    empty drop lists. The event must be keyed on whether anything was actually
    dropped, not on whether the text changed shape.
    """

    name = "graph-answer-no-drop"
    default_model = "graph-answer-no-drop-test-model"

    TEXT = " . The Platform Team is a member of Engineering [E 1] [1]."

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        return Completion(
            text=self.TEXT,
            model=self.default_model,
            provider=self.name,
            input_tokens=7,
            output_tokens=11,
            latency_ms=0,
            finish_reason="stop",
        )

    def stream(self, messages: list[Message], **kwargs):
        words = self.TEXT.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


def test_a_contract_pass_with_no_drops_emits_no_superseded_event(client, admin_token, team_graph):
    app.dependency_overrides[get_llm_provider] = lambda: _NoDropGraphAnswerLLM()
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": QUESTION, "strategy": "graph", "stream": True},
        headers=auth(admin_token),
    ) as res:
        assert res.status_code == 200
        events = _events("".join(res.iter_text()))

    names = [name for name, _ in events]
    assert "superseded" not in names
    by_name = dict(events)
    # The leading marker-less fragment still vanishes from the recorded answer;
    # only the announcement is what changed, not the correction itself.
    run = client.get(f"/api/runs/{by_name['done']['run_id']}", headers=auth(admin_token)).json()
    assert run["answer"] == "The Platform Team is a member of Engineering [E 1] [1]."


def test_the_audit_row_counts_the_graph_chunks_the_filter_removed(
    client, restricted_token, split_graph, db_session
):
    from ragfabric_core.models.access import AuditLog

    _ask(client, restricted_token, top_k=10)

    audit = db_session.query(AuditLog).order_by(AuditLog.id.desc()).first()
    assert audit.strategy == "graph"
    # Two chunks source the graph; the secret one is filtered for this caller.
    assert audit.sources_filtered == 1
