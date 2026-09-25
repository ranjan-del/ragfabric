"""The contracts between graph extraction, traversal and whatever model drives them.

Forgiving about packaging, strict about content, the same discipline as the
agent contracts: a fenced block or surrounding prose is tolerated, a missing
confidence or an invented relation type is a ``ContractViolation``. The
sub-graph shape is checked here too, because two tracks build against it and a
reversed edge carrying the forward name is a fabricated fact.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ragfabric_core.agent import contracts as agent_contracts
from ragfabric_core.graph.contracts import (
    INVERSES,
    SYMMETRIC,
    ContractViolation,
    EmptyReason,
    EntityType,
    ExtractionResponse,
    GraphEdge,
    GraphNode,
    QuestionExtraction,
    RelationType,
    Subgraph,
    normalise,
    parse_extraction,
    parse_question,
)


def _extraction(**overrides) -> str:
    payload = {
        "entities": [
            {
                "name": "Priya Raman",
                "entity_type": "person",
                "description": "leads the platform team",
                "confidence": 0.9,
            },
            {
                "name": "Platform Team",
                "entity_type": "team",
                "description": "owns the ingest service",
                "confidence": 0.8,
            },
        ],
        "relationships": [
            {
                "source": "Priya Raman",
                "target": "Platform Team",
                "relation_type": "MEMBER_OF",
                "description": "Priya is on the platform team",
                "confidence": 0.7,
            }
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _relationship(**overrides) -> dict:
    relationship = {
        "source": "Priya Raman",
        "target": "Platform Team",
        "relation_type": "MEMBER_OF",
        "description": "",
        "confidence": 0.7,
    }
    relationship.update(overrides)
    return relationship


# Extraction: packaging is forgiven.


def test_a_clean_extraction_is_parsed():
    result = parse_extraction(_extraction())
    assert isinstance(result, ExtractionResponse)
    assert [e.name for e in result.entities] == ["Priya Raman", "Platform Team"]
    assert result.entities[0].entity_type is EntityType.PERSON
    assert result.entities[0].confidence == 0.9
    assert result.relationships[0].relation_type is RelationType.MEMBER_OF
    assert result.relationships[0].confidence == 0.7


def test_fenced_json_is_accepted():
    result = parse_extraction(f"```json\n{_extraction()}\n```")
    assert isinstance(result, ExtractionResponse)
    assert len(result.relationships) == 1


def test_surrounding_prose_is_tolerated():
    result = parse_extraction(f"Here is what I found:\n{_extraction()}\nHope that helps.")
    assert isinstance(result, ExtractionResponse)


def test_a_chunk_with_nothing_to_extract_is_not_a_violation():
    """Most chunks name no relationship. Refusing an empty answer would make
    extraction retry on every one of them."""
    result = parse_extraction('{"entities": [], "relationships": []}')
    assert isinstance(result, ExtractionResponse)
    assert result.entities == []


def test_a_description_may_be_omitted():
    result = parse_extraction(
        json.dumps(
            {
                "entities": [{"name": "Atlas", "entity_type": "project", "confidence": 0.6}],
                "relationships": [],
            }
        )
    )
    assert isinstance(result, ExtractionResponse)
    assert result.entities[0].description == ""


# Extraction: content is strict.


def test_text_with_no_json_is_a_violation_carrying_the_raw_text():
    result = parse_extraction("I could not find any entities.")
    assert isinstance(result, ContractViolation)
    assert result.contract == "extraction"
    assert result.raw == "I could not find any entities."
    assert result.error


def test_an_unknown_relation_type_is_a_violation():
    """A relation the traversal has no direction rule for must never be stored."""
    result = parse_extraction(_extraction(relationships=[_relationship(relation_type="LOVES")]))
    assert isinstance(result, ContractViolation)
    assert "relation_type" in result.error


def test_an_unknown_entity_type_is_a_violation():
    result = parse_extraction(
        _extraction(
            entities=[{"name": "Mars", "entity_type": "planet", "confidence": 0.9}],
            relationships=[],
        )
    )
    assert isinstance(result, ContractViolation)
    assert "entity_type" in result.error


def test_a_missing_confidence_is_a_violation_not_a_default():
    """ADR 0004: a confidence is what the model reported, never a plausible default."""
    no_entity_confidence = json.loads(_extraction())
    del no_entity_confidence["entities"][0]["confidence"]
    result = parse_extraction(json.dumps(no_entity_confidence))
    assert isinstance(result, ContractViolation)
    assert "confidence" in result.error

    no_edge_confidence = json.loads(_extraction())
    del no_edge_confidence["relationships"][0]["confidence"]
    result = parse_extraction(json.dumps(no_edge_confidence))
    assert isinstance(result, ContractViolation)
    assert "confidence" in result.error


def test_a_null_confidence_is_a_violation():
    result = parse_extraction(_extraction(relationships=[_relationship(confidence=None)]))
    assert isinstance(result, ContractViolation)


@pytest.mark.parametrize("confidence", [-0.1, 1.5, "0.8", True])
def test_a_confidence_that_is_not_a_number_in_range_is_a_violation(confidence):
    """A quoted number or a boolean is not a reported confidence."""
    result = parse_extraction(_extraction(relationships=[_relationship(confidence=confidence)]))
    assert isinstance(result, ContractViolation)


def test_a_nan_confidence_is_a_violation():
    """Python's json module accepts a bare NaN, so the schema has to refuse it."""
    text = _extraction().replace('"confidence": 0.7', '"confidence": NaN')
    assert "NaN" in text
    assert isinstance(parse_extraction(text), ContractViolation)


def test_confidence_at_the_bounds_is_accepted():
    result = parse_extraction(
        _extraction(relationships=[_relationship(confidence=0), _relationship(confidence=1)])
    )
    assert isinstance(result, ExtractionResponse)


def test_an_edge_naming_an_entity_not_in_the_list_is_a_violation():
    """An edge to an entity the model never listed has no type, so it cannot be stored
    without inventing one."""
    result = parse_extraction(_extraction(relationships=[_relationship(target="Finance Team")]))
    assert isinstance(result, ContractViolation)
    assert "Finance Team" in result.error

    result = parse_extraction(_extraction(relationships=[_relationship(source="Someone Else")]))
    assert isinstance(result, ContractViolation)
    assert "Someone Else" in result.error


def test_an_edge_endpoint_matches_its_entity_after_normalisation():
    """Task 3 stores entities by normalised name, so the endpoint check agrees with it."""
    result = parse_extraction(_extraction(relationships=[_relationship(target=" platform team.")]))
    assert isinstance(result, ExtractionResponse)


def test_an_edge_endpoint_that_names_two_entity_types_is_a_violation():
    """If "Atlas" is both a project and a product, the edge cannot say which."""
    result = parse_extraction(
        _extraction(
            entities=[
                {"name": "Priya Raman", "entity_type": "person", "confidence": 0.9},
                {"name": "Atlas", "entity_type": "project", "confidence": 0.9},
                {"name": "Atlas", "entity_type": "product", "confidence": 0.9},
            ],
            relationships=[_relationship(target="Atlas", relation_type="WORKS_ON")],
        )
    )
    assert isinstance(result, ContractViolation)
    assert "Atlas" in result.error


def test_an_edge_from_an_entity_to_itself_is_a_violation():
    result = parse_extraction(_extraction(relationships=[_relationship(target="priya raman")]))
    assert isinstance(result, ContractViolation)


def test_a_blank_entity_name_is_a_violation():
    result = parse_extraction(
        _extraction(
            entities=[{"name": "  ...  ", "entity_type": "person", "confidence": 0.9}],
            relationships=[],
        )
    )
    assert isinstance(result, ContractViolation)


def test_a_missing_relationships_list_is_a_violation():
    """Omitting the list is not the same as saying there are none."""
    payload = json.loads(_extraction())
    del payload["relationships"]
    assert isinstance(parse_extraction(json.dumps(payload)), ContractViolation)


def test_the_violation_type_is_the_one_the_agent_contracts_return():
    """One ContractViolation across the codebase, so callers check a single type."""
    assert ContractViolation is agent_contracts.ContractViolation


# The fixed vocabularies and their direction rules.


def test_inverses_are_keyed_only_by_relation_types():
    assert INVERSES
    for relation in INVERSES:
        assert isinstance(relation, RelationType)


def test_no_relation_is_its_own_inverse_unless_it_is_symmetric():
    for relation, inverse in INVERSES.items():
        if relation in SYMMETRIC:
            assert inverse == relation.value
        else:
            assert inverse != relation.value


def test_every_symmetric_relation_is_walkable_both_ways():
    assert SYMMETRIC
    for relation in SYMMETRIC:
        assert INVERSES[relation] == relation.value


def test_an_inverse_name_never_collides_with_another_relation_type():
    """A reversed BELONGS_TO read as CONTAINS must not be mistaken for a stored
    CONTAINS edge, so no inverse name may also be a forward relation type."""
    forward = {relation.value for relation in RelationType}
    for relation, inverse in INVERSES.items():
        if relation not in SYMMETRIC:
            assert inverse not in forward, inverse
    assert len(set(INVERSES.values())) == len(INVERSES)


def test_reports_to_is_directed():
    """Reading "B reports to A" backwards asserts a fact the corpus never stated."""
    assert RelationType.REPORTS_TO not in INVERSES


def test_belongs_to_read_backwards_is_contains():
    assert INVERSES[RelationType.BELONGS_TO] == "CONTAINS"


# normalise


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Platform Team", "platform team"),
        ("  Platform   Team  ", "platform team"),
        ("Platform\t\nTeam", "platform team"),
        ('"Platform Team."', "platform team"),
        ("(Platform Team)", "platform team"),
        ("“Platform Team”", "platform team"),
        ("ACME, Inc.", "acme, inc"),
        ("O'Brien", "o'brien"),
        ("C++", "c++"),
        ("Straße", "strasse"),
        (" . ", ""),
        ("C#", "c#"),
        ("F#", "f#"),
        (".NET", ".net"),
        ("100%", "100%"),
        ("@ispf", "@ispf"),
        ("Atlas (beta)", "atlas (beta)"),
        ("(a) and (b)", "(a) and (b)"),
        ("[Atlas]", "atlas"),
        ("Who is Priya?", "who is priya"),
        ("Atlas;", "atlas"),
        ("ＡＣＭＥ", "acme"),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


@pytest.mark.parametrize(
    ("one", "other"),
    [("C#", "C"), ("F#", "F"), (".NET", "NET"), ("100%", "100")],
)
def test_normalise_keeps_distinct_names_distinct(one, other):
    """R5: normalise defines entity identity, so a symbol that is part of a name
    must survive it."""
    assert normalise(one) != normalise(other)


def test_normalise_treats_composed_and_decomposed_forms_as_one_name():
    composed = "Café"
    decomposed = "Café"
    assert composed != decomposed
    assert normalise(composed) == normalise(decomposed) == "café"


def test_normalise_is_idempotent():
    for raw in ['  "ACME,   Inc."  ', "Platform Team", "(( x ))", "Café", "[C#]."]:
        once = normalise(raw)
        assert normalise(once) == once


# The question contract.


def test_a_question_extraction_is_parsed():
    result = parse_question(
        '{"entities": [{"name": "Priya Raman", "entity_type": "person"}, {"name": "Atlas"}],'
        ' "implied_relation_types": ["REPORTS_TO", "WORKS_ON"]}'
    )
    assert isinstance(result, QuestionExtraction)
    assert result.entities[0].entity_type is EntityType.PERSON
    assert result.entities[1].entity_type is None
    assert result.implied_relation_types == [RelationType.REPORTS_TO, RelationType.WORKS_ON]


def test_no_implied_relation_types_is_accepted_as_no_restriction():
    result = parse_question(
        '```json\n{"entities": [{"name": "Atlas"}], "implied_relation_types": []}\n```'
    )
    assert isinstance(result, QuestionExtraction)
    assert result.implied_relation_types == []


def test_an_unknown_implied_relation_type_is_a_violation():
    result = parse_question(
        '{"entities": [{"name": "Atlas"}], "implied_relation_types": ["HATES"]}'
    )
    assert isinstance(result, ContractViolation)
    assert result.contract == "question"


def test_an_unknown_question_entity_type_is_a_violation():
    result = parse_question(
        '{"entities": [{"name": "Mars", "entity_type": "planet"}], "implied_relation_types": []}'
    )
    assert isinstance(result, ContractViolation)


def test_a_missing_implied_relation_types_list_is_a_violation():
    """Absent is not the same as empty: empty means walk every type on purpose."""
    result = parse_question('{"entities": [{"name": "Atlas"}]}')
    assert isinstance(result, ContractViolation)


def test_a_blank_question_entity_name_is_a_violation():
    result = parse_question('{"entities": [{"name": " "}], "implied_relation_types": []}')
    assert isinstance(result, ContractViolation)


# The sub-graph shape.


def _node(
    node_id: int, name: str = "n", entity_type: EntityType = EntityType.TEAM, depth: int = 0
) -> GraphNode:
    return GraphNode(id=node_id, name=name, entity_type=entity_type, depth=depth)


def _edge(**overrides) -> GraphEdge:
    fields = {
        "id": 10,
        "source_id": 1,
        "target_id": 2,
        "relation_type": RelationType.BELONGS_TO,
        "walked_as": "BELONGS_TO",
        "reversed": False,
        "confidence": 0.8,
        "source_chunk_ids": [100],
    }
    fields.update(overrides)
    return GraphEdge(**fields)


def test_a_forward_edge_is_walked_under_its_own_name():
    edge = _edge()
    assert edge.walked_as == "BELONGS_TO"
    assert edge.reversed is False


def test_a_reversed_invertible_edge_is_walked_under_its_inverse_name():
    edge = _edge(reversed=True, walked_as="CONTAINS")
    assert edge.walked_as == "CONTAINS"


def test_a_reversed_edge_carrying_the_forward_name_is_refused():
    """The fabricated-fact failure: B BELONGS_TO A walked backwards and still called
    BELONGS_TO asserts that A belongs to B."""
    with pytest.raises(ValidationError):
        _edge(reversed=True, walked_as="BELONGS_TO")


def test_a_directed_relation_cannot_be_reversed():
    with pytest.raises(ValidationError):
        _edge(relation_type=RelationType.REPORTS_TO, reversed=True, walked_as="REPORTS_TO")


def test_a_forward_edge_under_another_name_is_refused():
    with pytest.raises(ValidationError):
        _edge(walked_as="CONTAINS")


def test_a_symmetric_edge_may_be_walked_backwards_under_its_own_name():
    edge = _edge(relation_type=RelationType.RELATED_TO, walked_as="RELATED_TO", reversed=True)
    assert edge.reversed is True


def test_an_unknown_edge_confidence_is_none_not_a_default():
    assert _edge(confidence=None).confidence is None
    with pytest.raises(ValidationError):
        _edge(confidence=1.2)


def test_an_edge_with_no_visible_source_chunk_is_refused():
    """R6: an edge is visible only through a visible source chunk, so a sub-graph edge
    with none is an access leak."""
    with pytest.raises(ValidationError):
        _edge(source_chunk_ids=[])


def test_nodes_and_edges_carry_no_description():
    """R6: a stored description may paraphrase a denied chunk."""
    assert "description" not in GraphNode.model_fields
    assert "description" not in GraphEdge.model_fields
    with pytest.raises(ValidationError):
        GraphNode(id=1, name="x", entity_type=EntityType.TEAM, depth=0, description="leak")


def test_a_subgraph_with_edges_has_no_empty_reason():
    graph = Subgraph(nodes=[_node(1), _node(2)], edges=[_edge()], truncated=True, empty_reason=None)
    assert graph.truncated is True
    assert graph.empty_reason is None


def test_a_subgraph_without_edges_must_say_why():
    with pytest.raises(ValidationError):
        Subgraph(nodes=[_node(1)], edges=[], truncated=False, empty_reason=None)


def test_a_subgraph_with_edges_cannot_claim_to_be_empty():
    with pytest.raises(ValidationError):
        Subgraph(
            nodes=[_node(1), _node(2)],
            edges=[_edge()],
            truncated=False,
            empty_reason=EmptyReason.NO_WALKABLE_EDGES,
        )


def test_matched_nodes_with_nothing_walkable_is_reported():
    graph = Subgraph(
        nodes=[_node(1)], edges=[], truncated=False, empty_reason=EmptyReason.NO_WALKABLE_EDGES
    )
    assert graph.empty_reason is EmptyReason.NO_WALKABLE_EDGES


@pytest.mark.parametrize("reason", [EmptyReason.NO_ENTITY_MATCHED, EmptyReason.NO_GRAPH_COVERAGE])
def test_an_empty_reason_that_means_nothing_matched_carries_no_nodes(reason):
    assert Subgraph(nodes=[], edges=[], truncated=False, empty_reason=reason).nodes == []
    with pytest.raises(ValidationError):
        Subgraph(nodes=[_node(1)], edges=[], truncated=False, empty_reason=reason)


def test_the_empty_reasons_are_exactly_the_three_ruled():
    assert {reason.value for reason in EmptyReason} == {
        "no_entity_matched",
        "no_walkable_edges",
        "no_graph_coverage",
    }


def test_an_edge_to_a_node_outside_the_subgraph_is_refused():
    with pytest.raises(ValidationError):
        Subgraph(nodes=[_node(1)], edges=[_edge()], truncated=False, empty_reason=None)


def test_duplicate_node_ids_are_refused():
    with pytest.raises(ValidationError):
        Subgraph(
            nodes=[_node(1), _node(1), _node(2)],
            edges=[_edge()],
            truncated=False,
            empty_reason=None,
        )
