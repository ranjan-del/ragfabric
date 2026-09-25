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
from ragfabric_core.graph.contracts import EntityType, RelationType
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


def ctx(*, access: AccessFilter | None = None, top_k: int = 5) -> RetrievalContext:
    return RetrievalContext(
        principal=Principal(user_id=1, email="engineer@example.com"),
        access_filter=access or AccessFilter.unrestricted(),
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
    # Ordered by (graph_depth, chunk id).
    assert [c.chunk_id for c in result.chunks] == sorted(depths, key=lambda cid: (depths[cid], cid))
    assert all(c.score is None for c in result.chunks)


def test_chunks_are_capped_to_top_k(graph):
    graph.entity("ada", chunks=(graph.open_chunk,))
    graph.entity("engine", chunks=(graph.open_chunk_2,))
    graph.edge("ada", RelationType.WORKS_ON, "engine", chunks=(graph.other_open_chunk,))
    llm = ScriptedLLMProvider([question_response([{"name": "ada", "entity_type": "person"}])])

    result = strategy(graph, llm, max_hops=2).retrieve("what does ada work on", ctx(top_k=1))
    assert len(result.chunks) == 1


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
