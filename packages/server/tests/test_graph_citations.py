"""The graph citation contract: a relationship claim cites its edge and that edge's chunk.

The Phase 3 contract checks a claim against retrieved chunk text. It cannot see
a relationship claim, because "the platform team reports to the CTO" may appear
in no chunk while reading as a plausible narration of two walked edges. So a
claim that names two sub-graph entities, or cites an ``[E k]`` marker, must cite
an edge of the traversed sub-graph that joins two entities it names, and a
``[n]`` chunk that is one of that edge's source chunks and names both of its
endpoints. Anything else is dropped and the drop is recorded with its reason.

The fixture is a small walk from the Platform Team:

    E1  Platform Team  MEMBER_OF   Engineering    source chunk 101
    E2  Engineering    REPORTS_TO  CTO            source chunk 102
    E3  Platform Team  CONTAINS    Atlas          stored Atlas BELONGS_TO Platform Team, 103
    E4  Platform Team  OWNS        Atlas          source chunk 105 (says "It", not the name)

There is deliberately no edge between the Platform Team and the CTO: a claim
that the one reports to the other narrates the path, and no source says it.
"""

from __future__ import annotations

from ragfabric_core.generate.cited import (
    NO_EVIDENCE_ANSWER,
    apply_graph_contract,
    build_graph_prompt,
    generate_graph_answer,
)
from ragfabric_core.generate.contract import CitationViolation, assert_citation_contract
from ragfabric_core.graph.citations import (
    RelationshipDropReason,
    is_relationship_claim,
    relationship_claim_violation,
)
from ragfabric_core.graph.contracts import (
    EntityType,
    GraphEdge,
    GraphNode,
    RelationType,
    Subgraph,
)
from ragfabric_core.providers.offline import ScriptedLLMProvider
from ragfabric_core.strategies.base import RetrievedChunk

PLATFORM, ENGINEERING, CTO, ATLAS = 1, 2, 3, 4


def chunk(chunk_id: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, document_id=1, collection_id=None, text=text)


def edge(
    edge_id: int,
    source_id: int,
    target_id: int,
    relation: RelationType,
    chunk_ids: list[int],
    *,
    reversed_as: str | None = None,
) -> GraphEdge:
    return GraphEdge(
        id=edge_id,
        source_id=source_id,
        target_id=target_id,
        relation_type=relation,
        walked_as=reversed_as or relation.value,
        reversed=reversed_as is not None,
        confidence=0.9,
        source_chunk_ids=chunk_ids,
    )


SUBGRAPH = Subgraph(
    nodes=[
        GraphNode(id=PLATFORM, name="Platform Team", entity_type=EntityType.TEAM, depth=0),
        GraphNode(id=ENGINEERING, name="Engineering", entity_type=EntityType.TEAM, depth=1),
        GraphNode(id=CTO, name="CTO", entity_type=EntityType.PERSON, depth=2),
        GraphNode(id=ATLAS, name="Atlas", entity_type=EntityType.PROJECT, depth=1),
    ],
    edges=[
        edge(10, PLATFORM, ENGINEERING, RelationType.MEMBER_OF, [101]),
        edge(11, ENGINEERING, CTO, RelationType.REPORTS_TO, [102]),
        edge(12, ATLAS, PLATFORM, RelationType.BELONGS_TO, [103], reversed_as="CONTAINS"),
        edge(13, PLATFORM, ATLAS, RelationType.OWNS, [105]),
    ],
    truncated=False,
    empty_reason=None,
)

CHUNKS = [
    chunk(101, "The Platform Team is a member of Engineering."),
    chunk(102, "Engineering reports to the CTO."),
    chunk(103, "Atlas belongs to the Platform Team."),
    chunk(104, "Engineering and the Platform Team share an office."),
    chunk(105, "It owns the roadmap for the project."),
]


def checked(text: str):
    return apply_graph_contract(text, CHUNKS, SUBGRAPH)


# ---------------------------------------------------------------------------
# The five tests the brief names.
# ---------------------------------------------------------------------------


def test_a_claim_about_an_edge_not_in_the_subgraph_is_dropped() -> None:
    """The traversal-narration fabrication: every edge is real, the claim is not.

    Both claims assert a REPORTS_TO between the Platform Team and the CTO. The
    relation type exists in the graph (E2 is one), but no traversed edge joins
    those two entities. The first cites an edge number the sub-graph does not
    have; the second cites a real edge that joins different entities.
    """
    result = checked(
        "The Platform Team reports to the CTO [E 7] [1]. "
        "The Platform Team reports to the CTO [E 2] [2]."
    )

    assert result.text == NO_EVIDENCE_ANSWER
    assert [(d.text, d.reason) for d in result.dropped_relationship_claims] == [
        (
            "The Platform Team reports to the CTO [E 7] [1].",
            RelationshipDropReason.EDGE_NOT_IN_SUBGRAPH.value,
        ),
        (
            "The Platform Team reports to the CTO [E 2] [2].",
            RelationshipDropReason.NO_EDGE_CITED.value,
        ),
    ]


def test_a_claim_whose_edge_exists_but_whose_chunk_does_not_support_it_is_dropped() -> None:
    """E1 is cited and joins the two named entities, but chunk [4] is not E1's source.

    Chunk [4] even names both endpoints, so the only thing wrong is provenance:
    it is not a chunk the edge was extracted from.
    """
    result = checked("The Platform Team is a member of Engineering [E 1] [4].")

    assert result.text == NO_EVIDENCE_ANSWER
    assert [d.reason for d in result.dropped_relationship_claims] == [
        RelationshipDropReason.CHUNK_NOT_EDGE_SOURCE.value
    ]


def test_a_supported_relationship_claim_survives_with_both_citations() -> None:
    text = "The Platform Team is a member of Engineering [E 1] [1]."

    result = checked(text)

    assert result.text == text
    assert result.dropped_relationship_claims == []
    assert result.dropped_claims == []


def test_a_dropped_claim_is_recorded() -> None:
    """The kept text loses the claim; the record keeps its exact words and the reason."""
    result = checked(
        "Engineering reports to the CTO [E 2] [2]. The Platform Team reports to the CTO [2]."
    )

    assert result.text == "Engineering reports to the CTO [E 2] [2]."
    assert result.dropped_claims == []
    assert len(result.dropped_relationship_claims) == 1
    dropped = result.dropped_relationship_claims[0]
    assert dropped.text == "The Platform Team reports to the CTO [2]."
    assert dropped.reason == RelationshipDropReason.NO_EDGE_CITED.value


def test_the_phase_3_contract_is_still_applied_to_chunk_claims() -> None:
    """A claim naming at most one entity goes through the Phase 3 contract, unchanged."""
    quoted = 'Engineering "has nine hundred engineers" [2].'
    uncited_range = "The office has a gym [9]."
    fine = "Engineering reports upward [2]."

    result = checked(f"{fine} {quoted} {uncited_range}")

    assert result.text == fine
    assert result.dropped_relationship_claims == []
    reasons = []
    for claim in (quoted, uncited_range):
        try:
            assert_citation_contract(claim, CHUNKS)
        except CitationViolation as exc:
            reasons.append(exc.reason)
    assert [(d.text, d.reason) for d in result.dropped_claims] == list(
        zip([quoted, uncited_range], reasons, strict=True)
    )


# ---------------------------------------------------------------------------
# The rest of the definition (ruling R32).
# ---------------------------------------------------------------------------


def test_a_claim_naming_two_entities_without_any_edge_marker_is_dropped() -> None:
    """Naming two entities makes a relationship claim; citing only a chunk is not enough."""
    result = checked("The Platform Team is a member of Engineering [1].")

    assert [d.reason for d in result.dropped_relationship_claims] == [
        RelationshipDropReason.NO_EDGE_CITED.value
    ]


def test_a_source_chunk_that_does_not_name_both_endpoints_is_not_support() -> None:
    """Chunk [5] is E4's source, but it says "It", so it names neither endpoint."""
    result = checked("The Platform Team owns Atlas [E 4] [5].")

    assert [d.reason for d in result.dropped_relationship_claims] == [
        RelationshipDropReason.CHUNK_DOES_NOT_NAME_ENDPOINTS.value
    ]


def test_every_cited_edge_between_named_entities_must_be_supported() -> None:
    """A path claim needs a source chunk per edge; one supported edge does not carry both."""
    both = "The Platform Team belongs to Engineering, which reports to the CTO [E 1] [E 2] [1] [2]."
    one = "The Platform Team belongs to Engineering, which reports to the CTO [E 1] [E 2] [1]."

    assert checked(both).text == both
    assert [d.reason for d in checked(one).dropped_relationship_claims] == [
        RelationshipDropReason.CHUNK_NOT_EDGE_SOURCE.value
    ]


def test_an_edge_marker_alone_makes_a_relationship_claim() -> None:
    """One named entity plus an [E k] marker is still checked as a relationship claim."""
    claim = "It reports to the CTO [E 2] [2]."

    assert is_relationship_claim(claim, SUBGRAPH)
    assert relationship_claim_violation(claim, SUBGRAPH, CHUNKS) == (
        RelationshipDropReason.NO_EDGE_CITED
    )


def test_names_match_whole_words_not_substrings_of_longer_words() -> None:
    """ "Atlassian" is not the Atlas project, so this names one entity and is a chunk claim."""
    claim = "The Platform Team uses Atlassian tools [1]."

    assert not is_relationship_claim(claim, SUBGRAPH)
    assert checked(claim).text == claim


def test_names_match_after_normalisation() -> None:
    """Case and spacing differ from the stored names; the claim still names both."""
    claim = "the PLATFORM   team is a member of engineering [E1] [1]."

    assert is_relationship_claim(claim, SUBGRAPH)
    assert checked(claim).text == claim


def test_a_reversed_edge_is_cited_under_its_walked_reading() -> None:
    result = checked("The Platform Team contains Atlas [E 3] [3].")

    assert result.dropped_relationship_claims == []


def test_the_prompt_renders_passages_and_numbered_edges_in_walk_direction() -> None:
    prompt = build_graph_prompt("who does the platform team report to", CHUNKS, SUBGRAPH)

    assert "[1] The Platform Team is a member of Engineering." in prompt
    assert "[E 1] Platform Team MEMBER_OF Engineering (source passages: [1])" in prompt
    assert "[E 2] Engineering REPORTS_TO CTO (source passages: [2])" in prompt
    assert "[E 3] Platform Team CONTAINS Atlas (source passages: [3])" in prompt
    assert "[E 4] Platform Team OWNS Atlas (source passages: [5])" in prompt


def test_an_edge_whose_source_chunks_were_not_retrieved_says_so_in_the_prompt() -> None:
    prompt = build_graph_prompt("q", CHUNKS[:1], SUBGRAPH)

    assert "[E 2] Engineering REPORTS_TO CTO (source passages: none provided)" in prompt


def test_generation_makes_one_call_and_drops_rather_than_retries() -> None:
    llm = ScriptedLLMProvider(
        [
            "The Platform Team is a member of Engineering [E 1] [1]. "
            "The Platform Team reports to the CTO [E 2] [2]."
        ]
    )

    answer = generate_graph_answer("who", CHUNKS, SUBGRAPH, llm)

    assert llm.calls == 1
    assert answer.text == "The Platform Team is a member of Engineering [E 1] [1]."
    assert [d.reason for d in answer.dropped_relationship_claims] == [
        RelationshipDropReason.NO_EDGE_CITED.value
    ]
    assert answer.generator == "llm"
    assert answer.input_tokens > 0 and answer.output_tokens > 0


def test_generation_with_no_chunks_makes_no_call() -> None:
    llm = ScriptedLLMProvider([])

    answer = generate_graph_answer("who", [], SUBGRAPH, llm)

    assert llm.calls == 0
    assert answer.text == NO_EVIDENCE_ANSWER
    assert (answer.input_tokens, answer.output_tokens) == (0, 0)


def test_a_missing_subgraph_leaves_every_edge_marker_unresolvable() -> None:
    result = apply_graph_contract("Engineering reports to the CTO [E 1] [2].", CHUNKS, None)

    assert [d.reason for d in result.dropped_relationship_claims] == [
        RelationshipDropReason.EDGE_NOT_IN_SUBGRAPH.value
    ]


def test_a_trailing_edge_marker_stays_with_its_claim() -> None:
    text = "The Platform Team is a member of Engineering. [E 1] [1]"

    assert checked(text).text == text


# ---------------------------------------------------------------------------
# Ruling R33: coverage, distinct name counting, marker-only fragments.
# ---------------------------------------------------------------------------


def test_a_named_entity_no_cited_edge_touches_is_dropped() -> None:
    """E2 is real and backed, but nothing cited touches the Platform Team.

    Without the coverage rule this sentence would fabricate exactly the
    relationship the brief uses as its example, riding on a real edge.
    """
    result = checked("The Platform Team and Engineering both report to the CTO [E 2] [2].")

    assert result.text == NO_EVIDENCE_ANSWER
    assert [(d.text, d.reason) for d in result.dropped_relationship_claims] == [
        (
            "The Platform Team and Engineering both report to the CTO [E 2] [2].",
            RelationshipDropReason.UNCOVERED_ENTITY.value,
        )
    ]


def test_a_path_claim_with_every_hop_cited_and_backed_still_survives() -> None:
    claim = "The Platform Team is in Engineering, which reports to the CTO [E 1] [E 2] [1] [2]."

    assert checked(claim).text == claim


def test_a_leading_marker_only_fragment_is_dropped_and_does_not_block_no_evidence() -> None:
    result = checked("[E 7]. The Platform Team reports to the CTO [E 2] [2].")

    assert result.text == NO_EVIDENCE_ANSWER
    assert [(d.text, d.reason) for d in result.dropped_relationship_claims] == [
        ("[E 7].", RelationshipDropReason.EDGE_NOT_IN_SUBGRAPH.value),
        (
            "The Platform Team reports to the CTO [E 2] [2].",
            RelationshipDropReason.NO_EDGE_CITED.value,
        ),
    ]


def test_a_leading_chunk_marker_fragment_is_dropped_too() -> None:
    result = checked("[1]. Engineering reports upward [2].")

    assert result.text == "Engineering reports upward [2]."
    assert [d.text for d in result.dropped_claims] == ["[1]."]


def test_one_mention_of_a_name_two_nodes_share_is_a_chunk_claim() -> None:
    """An Atlas project and an Atlas person: one mention, one name, not a relationship."""
    shared = SUBGRAPH.model_copy(
        update={
            "nodes": [
                *SUBGRAPH.nodes,
                GraphNode(id=5, name="Atlas", entity_type=EntityType.PERSON, depth=2),
            ]
        }
    )
    claim = "Atlas shipped in May [1]."

    assert not is_relationship_claim(claim, shared)
    assert apply_graph_contract(claim, CHUNKS, shared).text == claim
