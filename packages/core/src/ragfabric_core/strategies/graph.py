"""Graph RAG: one LLM call to read the question, then a directed, access-checked walk.

Read this next to ``strategies/traditional.py`` and ``strategies/vectorless.py``.
Those embed or search text; this one asks a model which entities the question
names and which relation types it is asking about, resolves those names to
visible entities (``graph.traverse.match_entities``), walks outward from them
(``graph.traverse.traverse``), and returns the chunks that justify every node
and edge the walk kept.

**One call, two things extracted (ruling R3).** The question contract
(``QuestionExtraction``) asks for entity mentions and implied relation types
together, in the JSON shape ``graph.contracts.parse_question`` validates, so
this strategy never makes a second call to ask for the other half.

**Access stays inside the traversal (ADR 0003).** Every join that decides
whether a node, an edge or a chunk is visible lives in ``graph.traverse`` and
``stores.document_chunks``. This module only calls those, in the caller's
``access_filter``, and never writes a query of its own over ``entities``,
``relationships`` or their source tables.

**No query-time embedding (ruling R19).** Entity mentions match by normalised
name and alias only; ``embedding_calls`` is always 0, because none are made.

**Counting honestly (ADR 0004).** ``llm_calls`` is 1 whenever the question call
was made, whether or not the model's answer could be used: a call that came
back as a contract violation still spent one call, and reporting 0 would hide
that spend. It is 0 when coverage fails and no question call is made at all
(ruling R22). Nothing here calls a model a second time to retry.
``retrieval_calls`` counts one per store round trip this run actually made,
not a fixed number per call shape: the coverage check always runs one query,
first; ``match_entities`` adds one only when the question named at least one
entity, since it returns without a query when there is nothing to look for;
``traverse`` adds one only when there were matched seeds to walk from, since
it too returns without touching the database when its seed list is empty;
``visible_entity_chunks`` and the chunk fetch each add one only when there was
something to ask about.

**The honest empty-graph path (ruling R22).** Three distinct empty cases are
told apart rather than collapsed into "no chunks" with no reason:

1. ``NO_GRAPH_COVERAGE``: nothing is visible to this caller at all, checked
   first with ``graph.traverse.has_visible_entities`` before any LLM call is
   made, so the one query this costs is cheaper than the call it saves. The
   result carries an empty ``Subgraph`` with this reason, ``llm_calls=0`` and
   ``retrieval_calls=1`` (the coverage check itself), and never differs,
   in shape or counters, from what a truly empty graph would report: the
   check only ever answers "is anything visible", never "does anything
   exist".
2. ``NO_ENTITY_MATCHED``: the question named no usable entity, or named some
   but none matched a visible node. ``graph.traverse.traverse`` reports this
   on an empty seed list, whichever way it became empty.
3. ``NO_WALKABLE_EDGES``: at least one entity matched, but the walk kept no
   edge. This also covers a walk that hit the node budget before it could
   keep any edge: ``Subgraph.truncated`` is ``True`` in that case, and the
   trace's ``traverse`` span records it, so a caller can tell a genuinely
   isolated node from one whose neighbours existed but were cut off by the
   budget. Neither is reported as the other; ``empty_reason`` is the same
   string for both because the contract has no separate reason for it, but
   ``truncated`` is what distinguishes them.

Each case appears in the trace as itself, never folded into a generic empty
result: the ``check_coverage`` span records whether coverage held, and the
``extract_question`` and ``traverse`` spans record their own outcome, so the
trace alone tells which of the three happened without inspecting ``subgraph``.

**Ordering (ruling R18, depth per ruling R21).** A chunk carries no score: a
traversal is not a similarity search and ADR 0004 forbids inventing one.
Chunks are ordered by the minimum hop depth of whichever node or edge sourced
them, then by chunk id, and that depth is recorded in
``metadata["graph_depth"]``. Node depth is read straight from
``GraphNode.depth``, which ``traverse`` fills from the walk's own minimum
depth, never re-derived here from edge direction: a node reached only
backwards through an invertible relation can still have its sole edge
reported in the forwards reading, which is exactly why the contract carries
depth instead of leaving callers to reconstruct it. An edge's depth is the
max of its two endpoints' depths, since both ends of an edge must already be
reached for the edge to be usable as a citation.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.graph.contracts import (
    ContractViolation,
    EmptyReason,
    EntityType,
    QuestionExtraction,
    RelationType,
    Subgraph,
    parse_question,
)
from ragfabric_core.graph.traverse import (
    DEFAULT_NODE_BUDGET,
    has_visible_entities,
    match_entities,
    traverse,
    visible_entity_chunks,
)
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.stores.document_chunks import SqlDocumentChunkReader
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    RetrievedChunk,
    StrategyName,
    TraceSpan,
)

GRAPH_QUESTION_SYSTEM = (
    "You are the entity extraction step of a graph retrieval system. "
    "You read a question and name the entities it mentions and the relation "
    "types it is asking about. You reply with one JSON object and nothing else."
)

_ENTITY_TYPES = ", ".join(sorted(t.value for t in EntityType))
_RELATION_TYPES = ", ".join(sorted(r.value for r in RelationType))

GRAPH_QUESTION_RULES = """Name every entity the question mentions, by its surface name in the \
question. Give each one an entity_type from this exact list when the question makes the type \
clear, or null when it does not: {entity_types}.

List every relation type from this exact list that the question is asking about, or an empty \
list when the question does not imply any particular relation: {relation_types}.

Never invent an entity_type or a relation_type outside these lists."""


def build_question_prompt(question: str) -> str:
    rules = GRAPH_QUESTION_RULES.format(entity_types=_ENTITY_TYPES, relation_types=_RELATION_TYPES)
    return (
        f"Question: {question}\n\n"
        f"{rules}\n\n"
        'Reply with JSON of the form {"entities": [{"name": "...", "entity_type": "..." or null}], '
        '"implied_relation_types": ["..."]}'
    )


class GraphRAGStrategy:
    name = StrategyName.GRAPH

    def __init__(
        self,
        *,
        llm: LLMProvider,
        session_factory: Callable[[], Session],
        chunk_reader: SqlDocumentChunkReader | None = None,
        max_hops: int = 2,
        node_budget: int = DEFAULT_NODE_BUDGET,
    ) -> None:
        self._llm = llm
        self._sf = session_factory
        self._chunks = chunk_reader or SqlDocumentChunkReader(session_factory)
        self._max_hops = max_hops
        self._node_budget = node_budget

    @property
    def max_hops(self) -> int:
        """The hop bound this strategy was built with (read only)."""
        return self._max_hops

    @property
    def node_budget(self) -> int:
        """The node budget this strategy was built with (read only)."""
        return self._node_budget

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        started = time.perf_counter()
        spans: list[TraceSpan] = []

        def span(name: str, begin: float, **attributes) -> None:
            spans.append(
                TraceSpan(
                    name=name,
                    started_ms=int((begin - started) * 1000),
                    duration_ms=int((time.perf_counter() - begin) * 1000),
                    attributes=attributes,
                )
            )

        with self._sf() as db:
            mark = time.perf_counter()
            covered = has_visible_entities(db, ctx.access_filter, ctx.collection_ids)
            span("check_coverage", mark, covered=covered)

            if not covered:
                return RetrievalResult(
                    strategy=self.name,
                    chunks=[],
                    retrieval_calls=1,
                    embedding_calls=0,
                    llm_calls=0,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    trace=spans,
                    subgraph=Subgraph(
                        nodes=[],
                        edges=[],
                        truncated=False,
                        empty_reason=EmptyReason.NO_GRAPH_COVERAGE,
                    ),
                )

            mark = time.perf_counter()
            completion = self._llm.complete(
                [
                    Message(role="system", content=GRAPH_QUESTION_SYSTEM),
                    Message(role="user", content=build_question_prompt(query)),
                ],
                temperature=0.0,
            )
            parsed = parse_question(completion.text)
            span(
                "extract_question",
                mark,
                model=completion.model,
                violation=parsed.error if isinstance(parsed, ContractViolation) else None,
                contract=parsed.contract if isinstance(parsed, ContractViolation) else None,
            )

            if isinstance(parsed, ContractViolation):
                return RetrievalResult(
                    strategy=self.name,
                    chunks=[],
                    retrieval_calls=1,
                    embedding_calls=0,
                    llm_calls=1,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    trace=spans,
                    subgraph=None,
                )

            assert isinstance(parsed, QuestionExtraction)
            retrieval_calls = 1  # the coverage check above already made one round trip

            mark = time.perf_counter()
            # match_entities returns without a query when there is nothing usable
            # to look for, so that call is not counted as a round trip either.
            matched_ids = match_entities(db, parsed.entities, ctx.access_filter)
            if parsed.entities:
                retrieval_calls += 1
            span("match_entities", mark, mentions=len(parsed.entities), matched=len(matched_ids))

            mark = time.perf_counter()
            subgraph = traverse(
                db,
                matched_ids,
                ctx.access_filter,
                max_hops=self._max_hops,
                node_budget=self._node_budget,
                relation_types=parsed.implied_relation_types or None,
            )
            # traverse() returns without a query when there are no seeds to walk
            # from, so that call is not counted as a round trip past matching.
            if matched_ids:
                retrieval_calls += 1
            span(
                "traverse",
                mark,
                nodes=len(subgraph.nodes),
                edges=len(subgraph.edges),
                truncated=subgraph.truncated,
                empty_reason=subgraph.empty_reason.value if subgraph.empty_reason else None,
            )

            entity_chunks: dict[int, list[int]] = {}
            if subgraph.nodes:
                mark = time.perf_counter()
                entity_chunks = visible_entity_chunks(
                    db, [node.id for node in subgraph.nodes], ctx.access_filter
                )
                retrieval_calls += 1
                span("visible_entity_chunks", mark, entities=len(entity_chunks))

            depth_by_chunk = _chunk_depths(subgraph, entity_chunks)

            chunks: list[RetrievedChunk] = []
            if depth_by_chunk:
                mark = time.perf_counter()
                fetched = self._chunks.chunks_by_ids(list(depth_by_chunk), ctx.access_filter)
                retrieval_calls += 1
                span("fetch_chunks", mark, requested=len(depth_by_chunk), returned=len(fetched))
                for chunk in fetched:
                    chunk.metadata["graph_depth"] = depth_by_chunk[chunk.chunk_id]
                chunks = sorted(fetched, key=lambda c: (c.metadata["graph_depth"], c.chunk_id))

        top_k = ctx.params.top_k
        return RetrievalResult(
            strategy=self.name,
            chunks=chunks[:top_k],
            retrieval_calls=retrieval_calls,
            embedding_calls=0,
            llm_calls=1,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            trace=spans,
            subgraph=subgraph,
        )


def _chunk_depths(subgraph: Subgraph, entity_chunks: dict[int, list[int]]) -> dict[int, int]:
    """The minimum graph depth of every chunk that sources a kept node or edge (ruling R18).

    Node depth comes straight from ``GraphNode.depth`` (ruling R21), never
    re-derived from edge direction: an edge walked backwards through an
    invertible relation can still be reported in the forwards reading, which
    makes edge direction unreliable for exactly the nodes only reachable that
    way. An edge's own depth is the max of its two endpoints, since both ends
    must already be reached before the edge is usable as a citation.
    """
    depths = {node.id: node.depth for node in subgraph.nodes}
    by_chunk: dict[int, int] = {}

    def offer(chunk_id: int, depth: int) -> None:
        if chunk_id not in by_chunk or depth < by_chunk[chunk_id]:
            by_chunk[chunk_id] = depth

    for entity_id, chunk_ids in entity_chunks.items():
        depth = depths.get(entity_id)
        if depth is None:
            continue
        for chunk_id in chunk_ids:
            offer(chunk_id, depth)

    for edge in subgraph.edges:
        depth = max(depths[edge.source_id], depths[edge.target_id])
        for chunk_id in edge.source_chunk_ids:
            offer(chunk_id, depth)

    return by_chunk
