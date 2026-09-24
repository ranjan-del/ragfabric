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
that spend. Nothing here calls a model a second time to retry.

**Ordering (ruling R18).** A chunk carries no score: a traversal is not a
similarity search and ADR 0004 forbids inventing one. Chunks are ordered by
the minimum hop depth of whichever node or edge sourced them, then by chunk
id, and that depth is recorded in ``metadata["graph_depth"]``. Depth is not
part of the ``Subgraph`` contract (``GraphNode`` and ``GraphEdge`` do not carry
it), so it is recomputed here by walking the returned edges outward from the
matched seeds, in the direction each edge was walked; that reconstruction is
exact because ``traverse`` guarantees every kept node has a kept node one hop
nearer the seeds.

**The three empty cases are not this task's.** A question with no matched
entity, a match with no walkable edge and a walk with no visible chunk all
currently fall out of the same code path and simply return no chunks with the
``Subgraph`` traversal already produced (or ``None`` when the question call
itself failed). Telling those apart for the caller is Task 9's job; this
module only leaves the seam, which is that ``subgraph`` and the trace already
carry everything that distinction needs.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.graph.contracts import (
    ContractViolation,
    EntityType,
    QuestionExtraction,
    RelationType,
    Subgraph,
    parse_question,
)
from ragfabric_core.graph.traverse import (
    DEFAULT_NODE_BUDGET,
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
            violation=isinstance(parsed, ContractViolation),
        )

        if isinstance(parsed, ContractViolation):
            return RetrievalResult(
                strategy=self.name,
                chunks=[],
                retrieval_calls=0,
                embedding_calls=0,
                llm_calls=1,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
                trace=spans,
                subgraph=None,
            )

        assert isinstance(parsed, QuestionExtraction)
        retrieval_calls = 0

        with self._sf() as db:
            mark = time.perf_counter()
            matched_ids = match_entities(db, parsed.entities, ctx.access_filter)
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
            retrieval_calls += 1
            span(
                "traverse",
                mark,
                nodes=len(subgraph.nodes),
                edges=len(subgraph.edges),
                truncated=subgraph.truncated,
                empty_reason=subgraph.empty_reason.value if subgraph.empty_reason else None,
            )

            depths = _hop_depths(subgraph, matched_ids)

            entity_chunks: dict[int, list[int]] = {}
            if subgraph.nodes:
                mark = time.perf_counter()
                entity_chunks = visible_entity_chunks(
                    db, [node.id for node in subgraph.nodes], ctx.access_filter
                )
                retrieval_calls += 1
                span("visible_entity_chunks", mark, entities=len(entity_chunks))

            depth_by_chunk = _chunk_depths(subgraph, depths, entity_chunks)

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


def _hop_depths(subgraph: Subgraph, seed_ids: list[int]) -> dict[int, int]:
    """Minimum hops from a seed to each node, by walking the returned edges forward.

    ``traverse`` computes this internally but the ``Subgraph`` contract does
    not carry it (``GraphNode`` has no depth field), so it is reconstructed
    here rather than exposed as a private detail of the traversal module. The
    edge set ``traverse`` returns is exactly the one that makes the
    reconstruction exact: every kept node has a kept node one hop nearer the
    seeds, so a breadth first walk in the direction each edge was walked
    recovers the same minimum depth ``traverse`` used to decide the budget.
    """
    node_ids = {node.id for node in subgraph.nodes}
    forward: dict[int, list[int]] = {}
    for edge in subgraph.edges:
        start, end = (
            (edge.target_id, edge.source_id)
            if edge.reversed
            else (
                edge.source_id,
                edge.target_id,
            )
        )
        forward.setdefault(start, []).append(end)

    depths: dict[int, int] = {}
    queue: deque[int] = deque()
    for seed in seed_ids:
        if seed in node_ids and seed not in depths:
            depths[seed] = 0
            queue.append(seed)
    while queue:
        node_id = queue.popleft()
        for neighbour in forward.get(node_id, ()):
            if neighbour not in depths:
                depths[neighbour] = depths[node_id] + 1
                queue.append(neighbour)
    return depths


def _chunk_depths(
    subgraph: Subgraph, depths: dict[int, int], entity_chunks: dict[int, list[int]]
) -> dict[int, int]:
    """The minimum graph depth of every chunk that sources a kept node or edge (ruling R18)."""
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
        endpoints = [
            depths[node_id] for node_id in (edge.source_id, edge.target_id) if node_id in depths
        ]
        if not endpoints:
            continue
        depth = min(endpoints)
        for chunk_id in edge.source_chunk_ids:
            offer(chunk_id, depth)

    return by_chunk
