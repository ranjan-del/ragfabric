"""GraphRAGStrategy: one question call, then a directed access-checked walk (Task 8).

Reuses the two-dialect seeded graph from ``test_graph_traverse`` so these tests
run on SQLite and on PostgreSQL exactly the way the traversal tests do. The
PostgreSQL parameter skips without ``RAGFABRIC_TEST_DATABASE_URL``, and a skip
is not a pass.
"""

from __future__ import annotations

import json

import test_graph_traverse as traversal_tests

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.graph.contracts import EmptyReason, EntityType, RelationType
from ragfabric_core.providers.offline import ScriptedLLMProvider
from ragfabric_core.stores.document_chunks import SqlDocumentChunkReader
from ragfabric_core.strategies.base import (
    Budget,
    RetrievalContext,
    RetrieverStrategy,
    StrategyName,
    StrategyParams,
    StrategyRegistry,
)
from ragfabric_core.strategies.graph import GraphRAGStrategy

graph = traversal_tests.graph
ALL = traversal_tests.ALL


def question_response(entities: list[dict], relation_types: list[str] | None = None) -> str:
    return json.dumps({"entities": entities, "implied_relation_types": relation_types or []})


def ctx(
    *,
    access: AccessFilter | None = None,
    top_k: int = 5,
    collection_ids: list[int] | None = None,
) -> RetrievalContext:
    return RetrievalContext(
        principal=Principal(user_id=1, email="engineer@example.com"),
        access_filter=access or AccessFilter.unrestricted(),
        collection_ids=collection_ids,
        params=StrategyParams(top_k=top_k),
        budget=Budget(),
    )


def session_factory_for(graph_fixture):
    """A one-off session factory over the seeded fixture's already open session.

    The traversal fixture hands back a single open ``Session`` rather than an
    engine-backed factory, so the strategy under test is given a factory that
    always returns that same session and is a no-op to close, which keeps this
    file from re-implementing engine setup the fixture already does.
    """

    class _NoCloseSession:
        def __init__(self, db):
            self._db = db

        def __enter__(self):
            return self._db

        def __exit__(self, *exc):
            return False

    return lambda: _NoCloseSession(graph_fixture.db)


def strategy(graph_fixture, llm, **kwargs) -> GraphRAGStrategy:
    sf = session_factory_for(graph_fixture)
    return GraphRAGStrategy(
        llm=llm, session_factory=sf, chunk_reader=SqlDocumentChunkReader(sf), **kwargs
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_strategy_is_resolvable_from_the_registry(graph):
    llm = ScriptedLLMProvider([question_response([{"name": "ada", "entity_type": "person"}])])
    registry = StrategyRegistry()
    instance = strategy(graph, llm)
    assert isinstance(instance, RetrieverStrategy)

    registry.register(instance)
    resolved = registry.get(StrategyName.GRAPH)
    assert resolved is instance
    assert resolved.name == StrategyName.GRAPH


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def test_counters_report_actual_calls(graph):
    graph.entity("ada")
    llm = ScriptedLLMProvider([question_response([{"name": "ada", "entity_type": "person"}])])

    result = strategy(graph, llm).retrieve("who is ada", ctx())

    assert result.llm_calls == 1
    assert result.embedding_calls == 0
    assert llm.calls == 1
    # check_coverage, match_entities, traverse, visible_entity_chunks and the
    # chunk fetch: all five are real round trips this run actually made.
    assert result.retrieval_calls == 5
    assert result.subgraph is not None
    assert {node.name for node in result.subgraph.nodes if node.id == graph.ids["ada"]} == {"ada"}


def test_a_contract_violation_is_reported_honestly_with_no_chunks(graph):
    graph.entity("ada")
    llm = ScriptedLLMProvider(["not json at all"])

    result = strategy(graph, llm).retrieve("who is ada", ctx())

    assert result.llm_calls == 1
    assert result.embedding_calls == 0
    # The coverage check ran (ada is visible) and made its one round trip
    # before the question call that then came back unusable.
    assert result.retrieval_calls == 1
    assert result.chunks == []
    assert result.subgraph is None
    violations = [span for span in result.trace if span.attributes.get("violation") is not None]
    assert len(violations) == 1
    assert violations[0].attributes["contract"] == "question"
    assert violations[0].attributes["violation"] == "no JSON object found in the response"


def test_no_entity_match_makes_no_traversal_calls_past_matching(graph):
    graph.entity("ada")
    llm = ScriptedLLMProvider([question_response([{"name": "nobody"}])])

    result = strategy(graph, llm).retrieve("who is nobody", ctx())

    assert result.chunks == []
    assert result.subgraph is not None
    assert result.subgraph.nodes == []
    # check_coverage and match_entities both run (the question named a mention,
    # even though it matched nothing); traverse makes no round trip when there
    # are no matched seeds to walk from (it returns without touching the
    # database), so only those two are counted.
    assert result.retrieval_calls == 2


# ---------------------------------------------------------------------------
# Chunks come from nodes and edges
# ---------------------------------------------------------------------------


def test_chunks_come_from_nodes_and_edges_on_the_path(graph):
    graph.entity("ada", chunks=(graph.open_chunk,))
    graph.entity("engine", chunks=(graph.open_chunk_2,))
    graph.edge("ada", RelationType.WORKS_ON, "engine", chunks=(graph.other_open_chunk,))
    llm = ScriptedLLMProvider([question_response([{"name": "ada", "entity_type": "person"}])])

    result = strategy(graph, llm, max_hops=2).retrieve("what does ada work on", ctx())

    assert {c.chunk_id for c in result.chunks} == {
        graph.open_chunk,
        graph.open_chunk_2,
        graph.other_open_chunk,
    }
    depths = {c.chunk_id: c.metadata["graph_depth"] for c in result.chunks}
    assert depths[graph.open_chunk] == 0
    assert depths[graph.open_chunk_2] == 1
    # An edge's depth is the max of its two endpoints (ruling R21): both ada
    # (depth 0) and engine (depth 1) must be reached before the edge between
    # them is usable, so its chunk carries depth 1, not 0.
    assert depths[graph.other_open_chunk] == 1
    # Ordered by graph_depth, then edge sources before node-only chunks (ruling
    # R37), then chunk id.
    edge_sources = {cid for edge in result.subgraph.edges for cid in edge.source_chunk_ids}
    assert [c.chunk_id for c in result.chunks] == sorted(
        depths, key=lambda cid: (depths[cid], cid not in edge_sources, cid)
    )
    assert [c.chunk_id for c in result.chunks] == [
        graph.open_chunk,
        graph.other_open_chunk,
        graph.open_chunk_2,
    ]
    assert all(c.score is None for c in result.chunks)


def test_chunks_are_capped_to_top_k(graph):
    graph.entity("ada", chunks=(graph.open_chunk,))
    graph.entity("engine", chunks=(graph.open_chunk_2,))
    graph.edge("ada", RelationType.WORKS_ON, "engine", chunks=(graph.other_open_chunk,))
    llm = ScriptedLLMProvider([question_response([{"name": "ada", "entity_type": "person"}])])

    result = strategy(graph, llm, max_hops=2).retrieve("what does ada work on", ctx(top_k=1))
    assert len(result.chunks) == 1


def test_the_top_k_cap_keeps_an_edges_source_before_a_node_only_chunk_at_the_same_depth(graph):
    """Ruling R37: a cited edge's backing chunk is not the first thing the cap cuts.

    ``engine``'s own chunk and the chunk behind ``ada --WORKS_ON--> engine``
    both sit at depth 1, and the node's chunk has the lower id. Ordered by id
    alone, a cap of two keeps the node's chunk and cuts the only passage a
    claim about the edge could cite.
    """
    graph.entity("ada", chunks=(graph.open_chunk,))
    graph.entity("engine", chunks=(graph.open_chunk_2,))
    graph.edge("ada", RelationType.WORKS_ON, "engine", chunks=(graph.other_open_chunk,))
    assert graph.open_chunk_2 < graph.other_open_chunk  # the order that exposes the cut
    llm = ScriptedLLMProvider([question_response([{"name": "ada", "entity_type": "person"}])])

    result = strategy(graph, llm, max_hops=2).retrieve("what does ada work on", ctx(top_k=2))

    assert [c.chunk_id for c in result.chunks] == [graph.open_chunk, graph.other_open_chunk]


# ---------------------------------------------------------------------------
# Depth reflects the walk, not edge direction (ruling R21)
# ---------------------------------------------------------------------------


def test_a_node_reached_only_backwards_still_contributes_its_chunk(graph):
    """A regression test for the bug this fix round closes.

    ``dept`` is the seed. ``team`` reaches it only by walking BELONGS_TO
    backwards (there is no other path to ``team``). Because ``team`` sits at
    depth 1 with a hop still to spare (``max_hops=2``), ``graph.traverse``'s
    ``goes_forwards`` preference reports the edge in the forwards reading
    (``reversed is False``) even though the walk only ever reached ``team``
    backwards. A depth reconstruction keyed off edge direction misses this
    node entirely; reading ``GraphNode.depth`` from the contract does not.
    """
    graph.entity("dept", EntityType.ORGANISATION, chunks=(graph.open_chunk,))
    graph.entity("team", EntityType.TEAM, chunks=(graph.open_chunk_2,))
    graph.edge("team", RelationType.BELONGS_TO, "dept")
    llm = ScriptedLLMProvider(
        [question_response([{"name": "dept", "entity_type": "organisation"}])]
    )

    result = strategy(graph, llm, max_hops=2).retrieve("what belongs to dept", ctx())

    [edge] = result.subgraph.edges
    assert edge.reversed is False
    assert edge.walked_as == "BELONGS_TO"
    node_depths = {node.id: node.depth for node in result.subgraph.nodes}
    assert node_depths[graph.ids["team"]] == 1

    chunk_depths = {c.chunk_id: c.metadata["graph_depth"] for c in result.chunks}
    assert graph.open_chunk_2 in chunk_depths
    assert chunk_depths[graph.open_chunk_2] == 1


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------


def _ada_question_llm() -> ScriptedLLMProvider:
    return ScriptedLLMProvider([question_response([{"name": "ada", "entity_type": "person"}])])


def test_the_access_filter_reaches_the_traversal(graph):
    graph.entity("ada", chunks=(graph.open_chunk,))
    graph.entity("grace", chunks=(graph.secret_chunk,))
    graph.edge("ada", RelationType.REPORTS_TO, "grace", chunks=(graph.secret_chunk,))

    # Control: unrestricted, the denied-under-restriction node, edge and chunk
    # are all genuinely reachable, so the assertion below is about access and
    # not about a graph that never had anything to hide.
    control = strategy(graph, _ada_question_llm(), max_hops=2).retrieve("who reports to ada", ctx())
    assert graph.ids["grace"] in {node.id for node in control.subgraph.nodes}
    assert graph.secret_chunk in {c.chunk_id for c in control.chunks}

    restricted = strategy(graph, _ada_question_llm(), max_hops=2).retrieve(
        "who reports to ada", ctx(access=graph.restricted)
    )
    node_ids = {node.id for node in restricted.subgraph.nodes}
    edge_endpoints = {(e.source_id, e.target_id) for e in restricted.subgraph.edges}
    chunk_ids = {c.chunk_id for c in restricted.chunks}

    assert graph.ids["grace"] not in node_ids
    assert (graph.ids["ada"], graph.ids["grace"]) not in edge_endpoints
    assert graph.secret_chunk not in chunk_ids
    assert restricted.subgraph.nodes and restricted.subgraph.nodes[0].id == graph.ids["ada"]


# ---------------------------------------------------------------------------
# Request scope (ctx.collection_ids, ruling R25): fix round 1 regression
# tests. Coverage alone honoring the scope is not enough: every subsequent
# call in the same run must honor it too, or a collection-scoped request can
# still receive nodes, edges and chunks from collections outside that scope.
# ---------------------------------------------------------------------------


def test_collection_ids_scopes_the_whole_run_not_only_coverage(graph):
    """The reviewer's demonstrated leak: ada in an open collection, grace in a
    secret collection, unrestricted access, ctx.collection_ids scoped to the
    open collection. Grace must be absent from nodes, edges and chunks, not
    only excluded from the coverage check.
    """
    graph.entity("ada", chunks=(graph.open_chunk,))
    graph.entity("grace", chunks=(graph.secret_chunk,))
    graph.edge("ada", RelationType.REPORTS_TO, "grace", chunks=(graph.secret_chunk,))

    # Control: with no collection scope, grace is genuinely reachable, so the
    # assertion below is about scoping and not about a graph with nothing to leak.
    control = strategy(graph, _ada_question_llm(), max_hops=2).retrieve("who reports to ada", ctx())
    assert graph.ids["grace"] in {node.id for node in control.subgraph.nodes}
    assert graph.secret_chunk in {c.chunk_id for c in control.chunks}

    scoped = strategy(graph, _ada_question_llm(), max_hops=2).retrieve(
        "who reports to ada", ctx(collection_ids=[graph.open_collection])
    )
    node_ids = {node.id for node in scoped.subgraph.nodes}
    chunk_ids = {c.chunk_id for c in scoped.chunks}

    assert graph.ids["grace"] not in node_ids
    assert scoped.subgraph.edges == []
    assert graph.secret_chunk not in chunk_ids
    assert node_ids == {graph.ids["ada"]}


def test_collection_ids_never_widens_past_the_access_filter(graph):
    """A request scope naming a collection the access filter itself denies
    must not resurrect it: collection_ids only narrows.
    """
    graph.entity("ada", chunks=(graph.open_chunk,))
    graph.entity("grace", chunks=(graph.secret_chunk,))

    result = strategy(graph, _ada_question_llm(), max_hops=2).retrieve(
        "who reports to ada", ctx(access=graph.restricted, collection_ids=[graph.secret_collection])
    )

    assert result.subgraph is not None
    assert result.subgraph.empty_reason == EmptyReason.NO_GRAPH_COVERAGE
    assert result.chunks == []


# ---------------------------------------------------------------------------
# The LLM call holds no database session (fix round 1: the coverage check's
# session must close before the model call, not wrap it)
# ---------------------------------------------------------------------------


class _RecordingSession:
    def __init__(self, inner, events: list[str]) -> None:
        self._inner = inner
        self._events = events

    def __enter__(self):
        self._events.append("session_open")
        return self._inner.__enter__()

    def __exit__(self, *exc):
        result = self._inner.__exit__(*exc)
        self._events.append("session_close")
        return result


class _RecordingSessionFactory:
    def __init__(self, real_factory, events: list[str]) -> None:
        self._real = real_factory
        self._events = events

    def __call__(self):
        return _RecordingSession(self._real(), self._events)


class _RecordingLLM(ScriptedLLMProvider):
    def __init__(self, responses: list[str], events: list[str]) -> None:
        super().__init__(responses)
        self._events = events

    def complete(self, *args, **kwargs):
        self._events.append("llm_call")
        return super().complete(*args, **kwargs)


def test_the_llm_call_runs_with_no_database_session_open(graph):
    graph.entity("ada")
    events: list[str] = []
    real_factory = session_factory_for(graph)
    recording_factory = _RecordingSessionFactory(real_factory, events)
    llm = _RecordingLLM([question_response([{"name": "ada", "entity_type": "person"}])], events)
    instance = GraphRAGStrategy(
        llm=llm,
        session_factory=recording_factory,
        chunk_reader=SqlDocumentChunkReader(recording_factory),
    )

    instance.retrieve("who is ada", ctx())

    assert events.count("llm_call") == 1
    llm_index = events.index("llm_call")
    before = events[:llm_index]
    # Any session opened before the model call was also closed before it: a
    # session left open across the call would leave an unmatched open here.
    assert before.count("session_open") == before.count("session_close")
    assert before.count("session_open") >= 1
    # And a session was opened again afterwards, for the traversal proper.
    assert events[llm_index + 1 :].count("session_open") >= 1
