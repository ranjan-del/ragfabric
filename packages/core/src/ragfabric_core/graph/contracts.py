"""What graph extraction asks a model for, and the sub-graph shape traversal returns.

Three contracts live here so that neither extraction nor traversal can invent a
shape of its own:

* ``ExtractionResponse``: the entities and relationships a model found in one
  chunk, each with the confidence the model reported.
* ``QuestionExtraction``: the entities a question mentions and the relation
  types it implies, from the single call the graph strategy makes per question.
* ``Subgraph``: what a traversal hands back, with every edge labelled by the
  name it was walked under.

The model-facing parsers follow the agent contracts: forgiving about packaging
(a fenced block or surrounding prose is tolerated), strict about content (a
missing confidence, an unknown type or an edge to an unlisted entity is a
``ContractViolation``). Entity and relation types are fixed enums, so a model
cannot introduce a relation the traversal has no direction rule for.

Nothing here calls a model or touches the database.
"""

from __future__ import annotations

import unicodedata
from enum import StrEnum
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from ragfabric_core.json_contract import ContractViolation, parse_contract


class EntityType(StrEnum):
    PERSON = "person"
    TEAM = "team"
    ORGANISATION = "organisation"
    PROJECT = "project"
    PRODUCT = "product"
    DOCUMENT = "document"
    POLICY = "policy"
    LOCATION = "location"


class RelationType(StrEnum):
    REPORTS_TO = "REPORTS_TO"
    MEMBER_OF = "MEMBER_OF"
    BELONGS_TO = "BELONGS_TO"
    OWNS = "OWNS"
    WORKS_ON = "WORKS_ON"
    LOCATED_IN = "LOCATED_IN"
    AUTHORED = "AUTHORED"
    MENTIONS = "MENTIONS"
    RELATED_TO = "RELATED_TO"


# Relations read the same in both directions: walked backwards under their own name.
SYMMETRIC: frozenset[RelationType] = frozenset({RelationType.RELATED_TO})

# The name a relation is read under when an edge is walked backwards. A relation
# absent from this table is directed and is walked forwards only: REPORTS_TO is
# absent because "B reports to A" read backwards is not a fact the corpus stated
# (a dotted line is not management). No inverse name is also a forward type.
INVERSES: dict[RelationType, str] = {
    # "a team belongs to a department" is "the department contains the team".
    RelationType.BELONGS_TO: "CONTAINS",
    # Membership is one fact seen from either side: the person or the group.
    RelationType.MEMBER_OF: "HAS_MEMBER",
    # Ownership is one fact: the owner owns it, it is owned by the owner.
    RelationType.OWNS: "OWNED_BY",
    # Working on a project is one fact: the project has that contributor.
    RelationType.WORKS_ON: "HAS_CONTRIBUTOR",
    # "the office is located in Pune" is "Pune is the location of the office".
    RelationType.LOCATED_IN: "LOCATION_OF",
    # Authorship is one fact: the person authored it, it was authored by them.
    RelationType.AUTHORED: "AUTHORED_BY",
    # A document mentioning an entity is the entity being mentioned in it.
    RelationType.MENTIONS: "MENTIONED_IN",
    # Symmetric: the same name in both directions.
    RelationType.RELATED_TO: RelationType.RELATED_TO.value,
}


def normalise(name: str) -> str:
    """The form entity names are matched and stored under.

    This defines entity identity (the unique key is ``(normalise(name),
    entity_type)``), so it only removes what cannot be part of a name:

    * Unicode NFKC normalisation, then casefolding, so composed and decomposed
      accents and fullwidth letters compare equal (``"Straße"`` is ``"strasse"``);
    * surrounding whitespace trimmed, internal whitespace collapsed to one space;
    * quotation marks at either end (ASCII ``"`` ``'`` and backtick, and Unicode
      opening and closing quotes);
    * a bracket pair ``()``, ``[]`` or ``{}`` only when it encloses the whole
      name (``"(Platform Team)"`` loses it, ``"atlas (beta)"`` keeps it);
    * trailing sentence punctuation ``. , ; : ! ?``.

    Everything else is kept, at either end: ``"c#"``, ``".net"``, ``"100%"``,
    ``"@ispf"``, ``"c++"``, and punctuation inside a name (``"acme, inc"``).
    """
    text = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", name).casefold())
    text = " ".join(text.split())
    previous = None
    while text != previous:
        previous = text
        text = text.rstrip(_TRAILING).strip()
        while text and _is_quote(text[0]):
            text = text[1:]
        while text and _is_quote(text[-1]):
            text = text[:-1]
        if _enclosed(text):
            text = text[1:-1]
        text = text.strip()
    return text


_TRAILING = ".,;:!? "
_PAIRS = {"(": ")", "[": "]", "{": "}"}


def _is_quote(char: str) -> bool:
    return char in "\"'`" or unicodedata.category(char) in ("Pi", "Pf")


def _enclosed(text: str) -> bool:
    """Whether the first character opens a bracket that the last character closes."""
    if len(text) < 2 or _PAIRS.get(text[0]) != text[-1]:
        return False
    opening, closing = text[0], text[-1]
    depth = 0
    for index, char in enumerate(text):
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index == len(text) - 1
    return False


def _named(name: str) -> str:
    if not normalise(name):
        raise ValueError("a name must contain something other than whitespace and punctuation")
    return name


Name = Annotated[str, AfterValidator(_named)]

# A confidence is a JSON number the model reported. A quoted number, a boolean or
# NaN is not one, and a missing value is never defaulted (ADR 0004).
Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)]


class ExtractedEntity(BaseModel):
    name: Name
    entity_type: EntityType
    description: str = ""
    confidence: Confidence


class ExtractedRelationship(BaseModel):
    """``source`` and ``target`` name entities in the same response, by ``normalise``."""

    source: Name
    target: Name
    relation_type: RelationType
    description: str = ""
    confidence: Confidence


class ExtractionResponse(BaseModel):
    """Entities and relationships from one chunk. Both lists may be empty; neither
    may be omitted.

    Duplicate entities (the same normalised name and type listed twice) are
    tolerated; the writer decides how to combine them.
    """

    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]

    @model_validator(mode="after")
    def _edges_name_listed_entities(self) -> Self:
        types_by_name: dict[str, set[EntityType]] = {}
        for entity in self.entities:
            types_by_name.setdefault(normalise(entity.name), set()).add(entity.entity_type)
        for relationship in self.relationships:
            for endpoint in (relationship.source, relationship.target):
                types = types_by_name.get(normalise(endpoint))
                if not types:
                    raise ValueError(f"relationship names {endpoint!r}, which is not listed")
                if len(types) > 1:
                    raise ValueError(
                        f"relationship names {endpoint!r}, which is listed under more than "
                        f"one entity type"
                    )
            if normalise(relationship.source) == normalise(relationship.target):
                raise ValueError(f"relationship from {relationship.source!r} to itself")
        return self


class EntityMention(BaseModel):
    """An entity a question mentions. The type is ``None`` when the question does not say."""

    name: Name
    entity_type: EntityType | None = None


class QuestionExtraction(BaseModel):
    """``implied_relation_types`` empty means no restriction; it may not be omitted."""

    entities: list[EntityMention]
    implied_relation_types: list[RelationType]


def parse_extraction(text: str) -> ExtractionResponse | ContractViolation:
    return parse_contract(text, "extraction", ExtractionResponse)


def parse_question(text: str) -> QuestionExtraction | ContractViolation:
    return parse_contract(text, "question", QuestionExtraction)


class EmptyReason(StrEnum):
    NO_ENTITY_MATCHED = "no_entity_matched"
    NO_WALKABLE_EDGES = "no_walkable_edges"
    NO_GRAPH_COVERAGE = "no_graph_coverage"


class GraphNode(BaseModel):
    """A node as the caller sees it. No description: a stored description may
    paraphrase a chunk the caller cannot see.

    ``depth`` is the minimum number of hops from a seed, as the walk itself
    measured it (``min(depth)`` over every path the recursive query found); a
    seed is depth 0. It is carried here, on the contract, rather than
    recomputed by a caller from the returned edges, because an edge walked
    backwards through an invertible relation can still be reported in the
    forwards reading (``graph.traverse``'s ``goes_forwards`` preference, when
    both directions are walkable), which makes a caller-side reconstruction
    from edge direction silently wrong for exactly the nodes only reachable
    that way (ruling R21).
    """

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    entity_type: EntityType
    depth: int = Field(ge=0)


class GraphEdge(BaseModel):
    """An edge as walked. ``source_id`` and ``target_id`` are the stored direction;
    ``walked_as`` is the relation read in the walk direction, which is the inverse
    name when ``reversed``. ``source_chunk_ids`` are only the caller-visible ones,
    and there is at least one, or the edge would not be visible at all."""

    model_config = ConfigDict(extra="forbid")

    id: int
    source_id: int
    target_id: int
    relation_type: RelationType
    walked_as: str
    reversed: bool
    confidence: Confidence | None
    source_chunk_ids: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _walked_as_matches_direction(self) -> Self:
        if not self.reversed:
            expected = self.relation_type.value
        elif self.relation_type in INVERSES:
            expected = INVERSES[self.relation_type]
        else:
            raise ValueError(f"{self.relation_type.value} is directed and cannot be reversed")
        if self.walked_as != expected:
            raise ValueError(f"walked_as is {self.walked_as!r}, expected {expected!r}")
        return self


class Subgraph(BaseModel):
    """What a traversal returns. ``empty_reason`` is set exactly when there are no
    edges; a reason meaning nothing matched also carries no nodes."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool
    empty_reason: EmptyReason | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        node_ids = [node.id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node ids must be unique")
        if len({edge.id for edge in self.edges}) != len(self.edges):
            raise ValueError("edge ids must be unique")
        known = set(node_ids)
        for edge in self.edges:
            if edge.source_id not in known or edge.target_id not in known:
                raise ValueError(f"edge {edge.id} joins a node outside the sub-graph")
        if self.edges and self.empty_reason is not None:
            raise ValueError("a sub-graph with edges has no empty_reason")
        if not self.edges and self.empty_reason is None:
            raise ValueError("a sub-graph with no edges must give an empty_reason")
        if self.nodes and self.empty_reason in (
            EmptyReason.NO_ENTITY_MATCHED,
            EmptyReason.NO_GRAPH_COVERAGE,
        ):
            raise ValueError(f"{self.empty_reason.value} means there are no nodes")
        return self
