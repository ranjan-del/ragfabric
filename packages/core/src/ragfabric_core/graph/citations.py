"""The graph citation contract: what a relationship claim must satisfy, mechanically.

The Phase 3 contract (``generate.contract``) checks a claim against the text of
the chunks it cites. It cannot see a relationship claim. "The platform team
reports to the CTO" may appear in no chunk while reading as a fair narration of
two walked edges, and a model that narrates a path as prose will happily assert
a relationship no source sentence ever stated. The edges are real, the walk is
correct, and the claim is still fabricated. This module is what catches that.
It is additive: chunk claims still go through the Phase 3 contract unchanged,
and nothing here replaces or loosens it.

**The citation form (ruling R32).** The graph prompt numbers the passages
``[n]``, as Phase 3 does, and numbers the traversed sub-graph's edges ``[E k]``,
1-based over ``Subgraph.edges`` in order, each shown in its walked reading
(``walked_as`` between the endpoint names). ``[E1]`` and ``[e 1]`` are read the
same as ``[E 1]``.

**What makes a claim a relationship claim.** It names two or more distinct
sub-graph node names, or it carries an ``[E k]`` marker. Distinct names are
counted, not node ids: an Atlas project and an Atlas person share one name, so
"Atlas shipped in May" is one mention and stays a chunk claim (ruling R33). Names are matched on the
``normalise()`` form of both the claim and the node name, as whole words or
phrases: "Atlas" is not named by "Atlassian". Longer names win over names
nested inside them ("Platform Team" is one name, not also "Platform"), so an
overlap cannot inflate the count. Markers are removed before matching, so the
``E`` of ``[E 1]`` can never name a node called "E".

**What a relationship claim must satisfy.** Checked in this order, and the
first failure is the recorded reason:

1. ``edge_not_in_subgraph``  every ``[E k]`` resolves to an edge of the
   traversed sub-graph. A number outside it, or any marker when there is no
   sub-graph, fails.
2. ``no_edge_cited``  at least one cited edge joins two entities the claim
   names. A claim naming two entities with no edge marker fails here, and so
   does a claim citing only edges between other entities.
3. ``chunk_not_edge_source``  every cited edge that joins two named entities
   has at least one cited ``[n]`` chunk among its ``source_chunk_ids``.
4. ``chunk_does_not_name_endpoints``  and at least one such chunk's text names
   both of that edge's endpoints (normalised, whole words).
5. ``uncovered_entity``  every entity the claim names is an endpoint of at
   least one of those cited, chunk-backed edges (ruling R33). For a name two
   nodes share, one of them being an endpoint is enough.

Requiring every edge between named entities to be supported, rather than just
one of them, is what makes a path claim ("A is in B, which reports to C")
need a source chunk per hop. Rule 5 is what stops a claim from riding a real
edge to assert a fabricated one: "the Platform Team and Engineering both
report to the CTO", citing only the backed Engineering to CTO edge, is dropped
because no cited edge touches the Platform Team. "A reports to C, through B"
with backed A-B and B-C edges still passes: every entity it names is touched.

Known looseness #1, recorded rather than hidden: **direction and paraphrase
faithfulness are not checked.** A claim that cites a supported ``OWNS`` edge
from A to B while saying B owns A passes, and so does a claim whose verb has
nothing to do with the edge's relation. Deciding either needs a semantic
reading of the sentence, which cannot be asserted (ADR 0004); Phase 8 measures
it.

Known looseness #2: rule 5 checks that every named entity is touched by a
backed edge, not which pairs the sentence relates. With backed A-B and B-C
edges cited, "A reports to C" in a sentence that also names B passes, although
no edge joins A to C: reading which pairs a sentence asserts, and in which
direction, is the semantic judgement looseness #1 leaves to Phase 8. A claim
naming only the two ends of a path is caught (rule 2), and a claim naming an
entity no cited edge touches is caught (rule 5).

Known looseness #3: a cited edge that joins no two named entities is
tolerated, not refused, provided it resolves (rule 1). Such a marker adds no
support (rules 2 to 4 ignore it) and refusing it would drop sentences over a
redundant citation.

Known looseness #4: names are matched only on the node's own name. A chunk
that names an endpoint by an alias or a pronoun does not count as naming it,
so a true claim can be dropped (over-strict, never over-lenient). The rendered
sub-graph carries no aliases, so there is nothing else to match on here.

Known looseness #5: a single-entity claim with no ``[E k]`` marker is not a
relationship claim, even if it relates that entity to something outside the
sub-graph ("Engineering reports to the board"). It goes through the Phase 3
contract, which checks its citation against chunk text, the same as any claim
did before this phase.
"""

from __future__ import annotations

import re
from enum import StrEnum

from ragfabric_core.generate.contract import cited_markers
from ragfabric_core.graph.contracts import GraphEdge, GraphNode, Subgraph, normalise
from ragfabric_core.strategies.base import RetrievedChunk

EDGE_MARKER_RE = re.compile(r"\[E\s*(\d+)\]", re.IGNORECASE)
_ANY_MARKER_RE = re.compile(r"\[(?:E\s*)?\d+\]", re.IGNORECASE)


class RelationshipDropReason(StrEnum):
    EDGE_NOT_IN_SUBGRAPH = "edge_not_in_subgraph"
    NO_EDGE_CITED = "no_edge_cited"
    CHUNK_NOT_EDGE_SOURCE = "chunk_not_edge_source"
    CHUNK_DOES_NOT_NAME_ENDPOINTS = "chunk_does_not_name_endpoints"
    UNCOVERED_ENTITY = "uncovered_entity"


def edge_marker(k: int) -> str:
    """The marker text for the k-th edge (1-based), as the prompt renders it."""
    return f"[E {k}]"


def edge_markers(claim: str) -> list[int]:
    """Edge marker numbers in first appearance order, without duplicates."""
    seen: list[int] = []
    for raw in EDGE_MARKER_RE.findall(claim):
        k = int(raw)
        if k not in seen:
            seen.append(k)
    return seen


def strip_markers(text: str) -> str:
    """The text with every ``[n]`` and ``[E k]`` marker removed."""
    return _ANY_MARKER_RE.sub(" ", text)


def _mentions(text: str, nodes: list[GraphNode]) -> list[set[int]]:
    """One entry per distinct node name ``text`` mentions: the ids carrying that name.

    Names are the normalised form, matched as whole phrases. Longest names are
    matched first and claim their span, so a name nested inside a longer one
    that was already matched there is not counted again. A name mentioned
    twice is one entry, and a name two nodes share is one entry holding both
    ids. A node whose name normalises to nothing is never matched.
    """
    haystack = normalise(strip_markers(text))
    by_name: dict[str, list[int]] = {}
    for node in nodes:
        key = normalise(node.name)
        if key:
            by_name.setdefault(key, []).append(node.id)

    taken: list[tuple[int, int]] = []
    found: list[set[int]] = []
    for name in sorted(by_name, key=len, reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")
        matched = False
        for match in pattern.finditer(haystack):
            start, end = match.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            taken.append((start, end))
            matched = True
        if matched:
            found.append(set(by_name[name]))
    return found


def _names_in(text: str, nodes: list[GraphNode]) -> set[int]:
    """Ids of the nodes whose normalised name ``text`` mentions."""
    return set().union(*_mentions(text, nodes))


def named_node_ids(claim: str, subgraph: Subgraph | None) -> set[int]:
    """Ids of the sub-graph nodes a claim names."""
    if subgraph is None:
        return set()
    return _names_in(claim, subgraph.nodes)


def is_relationship_claim(claim: str, subgraph: Subgraph | None) -> bool:
    """True iff the claim names two or more distinct sub-graph node names or carries ``[E k]``."""
    if edge_markers(claim):
        return True
    if subgraph is None:
        return False
    return len(_mentions(claim, subgraph.nodes)) >= 2


def _chunk_names_endpoints(chunk: RetrievedChunk, edge: GraphEdge, subgraph: Subgraph) -> bool:
    endpoints = [node for node in subgraph.nodes if node.id in (edge.source_id, edge.target_id)]
    return {edge.source_id, edge.target_id} <= _names_in(chunk.text, endpoints)


def relationship_claim_violation(
    claim: str, subgraph: Subgraph | None, chunks: list[RetrievedChunk]
) -> RelationshipDropReason | None:
    """Why a relationship claim fails the contract, or ``None`` when it satisfies it.

    Call this only for a claim ``is_relationship_claim`` accepts; the rules are
    listed, in the order they are applied, in the module docstring. ``chunks``
    are the passages as numbered in the prompt, ``[1]`` first.
    """
    edges = subgraph.edges if subgraph is not None else []
    cited_edges: list[GraphEdge] = []
    for k in edge_markers(claim):
        if k < 1 or k > len(edges):
            return RelationshipDropReason.EDGE_NOT_IN_SUBGRAPH
        cited_edges.append(edges[k - 1])

    named = named_node_ids(claim, subgraph)
    joining = [e for e in cited_edges if {e.source_id, e.target_id} <= named]
    if not joining or subgraph is None:
        return RelationshipDropReason.NO_EDGE_CITED

    cited_chunks = [chunks[n - 1] for n in cited_markers(claim) if 1 <= n <= len(chunks)]
    for cited_edge in joining:
        sources = [c for c in cited_chunks if c.chunk_id in cited_edge.source_chunk_ids]
        if not sources:
            return RelationshipDropReason.CHUNK_NOT_EDGE_SOURCE
        if not any(_chunk_names_endpoints(c, cited_edge, subgraph) for c in sources):
            return RelationshipDropReason.CHUNK_DOES_NOT_NAME_ENDPOINTS

    # Every edge in ``joining`` is now cited and chunk-backed, so its endpoints
    # are exactly the entities the claim's evidence covers.
    covered = {node_id for e in joining for node_id in (e.source_id, e.target_id)}
    if any(not (ids & covered) for ids in _mentions(claim, subgraph.nodes)):
        return RelationshipDropReason.UNCOVERED_ENTITY
    return None


def render_edges(subgraph: Subgraph | None, chunks: list[RetrievedChunk]) -> str:
    """The numbered edge list the graph prompt shows, one ``[E k]`` line per edge.

    Each edge is written in walk order: when it was walked backwards the stored
    target comes first, read through ``walked_as``, so the line always reads
    left to right as the walk went. Only names and relation names are shown,
    never a stored description (ruling R6). The passages that sourced the edge
    are listed by their ``[n]`` so the model can cite them; an edge none of
    whose source chunks was retrieved says so rather than listing nothing.
    """
    if subgraph is None or not subgraph.edges:
        return ""
    names = {node.id: node.name for node in subgraph.nodes}
    position = {c.chunk_id: n for n, c in enumerate(chunks, start=1)}
    lines: list[str] = []
    for k, e in enumerate(subgraph.edges, start=1):
        first, second = (e.target_id, e.source_id) if e.reversed else (e.source_id, e.target_id)
        markers = sorted(position[c] for c in set(e.source_chunk_ids) if c in position)
        sources = ", ".join(f"[{n}]" for n in markers) if markers else "none provided"
        lines.append(
            f"{edge_marker(k)} {names[first]} {e.walked_as} {names[second]} "
            f"(source passages: {sources})"
        )
    return "\n".join(lines)
