"""The honest empty-graph path (ruling R22, Task 9).

Three distinct empty cases, told apart rather than collapsed into a single
"no chunks" result with no reason: nothing visible to this caller at all
(``NO_GRAPH_COVERAGE``, checked first, no LLM call spent); a question whose
mentions matched no visible node (``NO_ENTITY_MATCHED``, including a question
that named nothing usable); and a match with nothing walkable from it
(``NO_WALKABLE_EDGES``, which also covers a walk truncated before it could
keep any edge, distinguished from true isolation by ``truncated``).

Reuses ``test_strategy_graph``'s helpers (``strategy``, ``ctx``,
``question_response``) and the two-dialect seeded graph fixture from
``test_graph_traverse``, so these tests run on SQLite and on PostgreSQL the
same way the rest of the graph suite does. The PostgreSQL parameter skips
without ``RAGFABRIC_TEST_DATABASE_URL``, and a skip is not a pass.
"""

from __future__ import annotations

import test_strategy_graph as strategy_tests

from ragfabric_core.graph.contracts import EmptyReason, RelationType, Subgraph
from ragfabric_core.providers.offline import ScriptedLLMProvider

graph = strategy_tests.graph
ctx = strategy_tests.ctx
strategy = strategy_tests.strategy
question_response = strategy_tests.question_response

# ---------------------------------------------------------------------------
# NO_ENTITY_MATCHED
# ---------------------------------------------------------------------------


def test_no_matched_entity_reports_that_specifically(graph):
    graph.entity("ada")
    llm = ScriptedLLMProvider([question_response([{"name": "nobody"}])])

    result = strategy(graph, llm).retrieve("who is nobody", ctx())

    assert result.subgraph is not None
    assert result.subgraph.empty_reason == EmptyReason.NO_ENTITY_MATCHED
    assert result.subgraph.nodes == []
    assert result.subgraph.edges == []
    assert result.llm_calls == 1


def test_no_chunks_are_returned_when_nothing_matched(graph):
    graph.entity("ada", chunks=(graph.open_chunk,))
    llm = ScriptedLLMProvider([question_response([{"name": "nobody"}])])

    result = strategy(graph, llm).retrieve("who is nobody", ctx())

    assert result.chunks == []


def test_zero_usable_mentions_is_no_entity_matched_and_costs_only_coverage(graph):
    graph.entity("ada")
    llm = ScriptedLLMProvider([question_response([])])

    result = strategy(graph, llm).retrieve("hello there", ctx())

    assert result.subgraph is not None
    assert result.subgraph.empty_reason == EmptyReason.NO_ENTITY_MATCHED
    assert result.chunks == []
    assert result.llm_calls == 1
    # The question named nothing usable, so match_entities returns without a
    # query and traverse returns without one too (no seeds): only the
    # coverage check made a round trip.
    assert result.retrieval_calls == 1


# ---------------------------------------------------------------------------
# NO_WALKABLE_EDGES: isolated vs truncated
# ---------------------------------------------------------------------------


def test_matched_but_isolated_entities_report_no_walkable_edges(graph):
    graph.entity("ada")
    llm = ScriptedLLMProvider([question_response([{"name": "ada", "entity_type": "person"}])])

    result = strategy(graph, llm).retrieve("who is ada", ctx())

    assert result.subgraph is not None
    assert result.subgraph.nodes != []
    assert result.subgraph.edges == []
    assert result.subgraph.empty_reason == EmptyReason.NO_WALKABLE_EDGES
    assert result.subgraph.truncated is False


def test_truncation_is_distinguishable_from_true_isolation(graph):
    """Both cases report ``NO_WALKABLE_EDGES`` (the contract has no separate
    reason for a walk cut short), but a truncated walk must never be
    reported as if its seed were simply isolated: ``truncated`` is how a
    caller tells the two apart, and it must be true in the trace as well as
    on the returned subgraph.
    """
    graph.entity("hub")
    graph.entity("leaf")
    graph.edge("hub", RelationType.OWNS, "leaf")
    graph.entity("loner")

    truncated_llm = ScriptedLLMProvider(
        [question_response([{"name": "hub", "entity_type": "person"}])]
    )
    truncated = strategy(graph, truncated_llm, node_budget=1).retrieve("what does hub own", ctx())
    assert truncated.subgraph is not None
    assert truncated.subgraph.edges == []
    assert truncated.subgraph.empty_reason == EmptyReason.NO_WALKABLE_EDGES
    assert truncated.subgraph.truncated is True
    truncated_span = next(span for span in truncated.trace if span.name == "traverse")
    assert truncated_span.attributes["truncated"] is True
    assert truncated_span.attributes["empty_reason"] == "no_walkable_edges"

    isolated_llm = ScriptedLLMProvider(
        [question_response([{"name": "loner", "entity_type": "person"}])]
    )
    isolated = strategy(graph, isolated_llm).retrieve("who is loner", ctx())
    assert isolated.subgraph is not None
    assert isolated.subgraph.edges == []
    assert isolated.subgraph.empty_reason == EmptyReason.NO_WALKABLE_EDGES
    assert isolated.subgraph.truncated is False
    isolated_span = next(span for span in isolated.trace if span.name == "traverse")
    assert isolated_span.attributes["truncated"] is False

    # Same reason string, opposite truncated flag: the distinction the
    # ruling requires is carried, not collapsed.
    assert truncated.subgraph.truncated != isolated.subgraph.truncated


# ---------------------------------------------------------------------------
# NO_GRAPH_COVERAGE
# ---------------------------------------------------------------------------


def test_an_empty_graph_reports_no_coverage(graph):
    llm = ScriptedLLMProvider([])

    result = strategy(graph, llm).retrieve("who is ada", ctx())

    assert result.chunks == []
    assert result.llm_calls == 0
    assert result.embedding_calls == 0
    assert result.retrieval_calls == 1
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert llm.calls == 0
    assert result.subgraph == Subgraph(
        nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_GRAPH_COVERAGE
    )


def test_denied_entities_report_no_coverage_same_as_a_truly_empty_graph(graph):
    """NO_GRAPH_COVERAGE must never reveal that invisible entities exist: a
    caller denied every entity in the graph must see exactly the same result
    shape and counters as a caller of a graph with no entities at all.
    """
    graph.entity("ada", chunks=(graph.secret_chunk,))
    graph.entity("grace", chunks=(graph.secret_chunk,))

    # Control: unrestricted, the graph is genuinely not empty, so the
    # restricted assertion below is about visibility and not about a graph
    # that never had anything to hide.
    control_llm = ScriptedLLMProvider(
        [question_response([{"name": "ada", "entity_type": "person"}])]
    )
    control = strategy(graph, control_llm).retrieve("who is ada", ctx())
    assert control.subgraph is not None
    assert control.subgraph.empty_reason != EmptyReason.NO_GRAPH_COVERAGE
    assert control_llm.calls == 1

    llm = ScriptedLLMProvider([])
    result = strategy(graph, llm).retrieve("who is ada", ctx(access=graph.restricted))

    # Exactly the shape and counters test_an_empty_graph_reports_no_coverage
    # asserts for a graph with no entities at all.
    assert result.chunks == []
    assert result.llm_calls == 0
    assert result.embedding_calls == 0
    assert result.retrieval_calls == 1
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert llm.calls == 0
    assert result.subgraph == Subgraph(
        nodes=[], edges=[], truncated=False, empty_reason=EmptyReason.NO_GRAPH_COVERAGE
    )
