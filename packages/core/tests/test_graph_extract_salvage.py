"""Fix wave: one bad relationship must not throw away every entity in the response.

Phase 9's first real run extracted ``team.md`` ("Asha Rao reports to Ravi
Sharma") with llama3.1:8b and stored none of the people. Reproduced on
2026-10-12: the model listed Ravi Sharma, Asha Rao and the Platform Team, then
added ``Ravi Sharma REPORTS_TO chief technology officer`` without listing the
officer. The contract rejected the whole response for that one edge, so the
chunk stored nothing. Now each relationship that breaks a rule is discarded
and counted, and everything valid in the response is kept. A reply that is
not a JSON object, or has no entity list, is still a contract violation.
"""

from __future__ import annotations

import json

from sqlalchemy import select
from test_graph_extract import FLOOR, _make_chunk, _provider, db  # noqa: F401

from ragfabric_core.graph.contracts import salvage_extraction
from ragfabric_core.graph.extract import extract_chunk
from ragfabric_core.json_contract import ContractViolation
from ragfabric_core.models.graph import Entity, Relationship
from ragfabric_core.providers.offline import ScriptedLLMProvider

TEAM = "Ravi Sharma reports to the chief technology officer. Asha Rao reports to Ravi Sharma."


def _person(name, t="person", c=0.95):
    return {"name": name, "entity_type": t, "description": "", "confidence": c}


def _rel(s, t, r="REPORTS_TO", c=0.9):
    return {"source": s, "target": t, "relation_type": r, "description": "", "confidence": c}


REPLY = {
    "entities": [_person("Ravi Sharma"), _person("Asha Rao"), _person("Platform Team", "team")],
    "relationships": [
        _rel("Ravi Sharma", "chief technology officer"),  # target never listed
        _rel("Asha Rao", "Ravi Sharma"),
        _rel("Asha Rao", "Asha Rao"),  # to itself
        _rel("Ravi Sharma", "Platform Team", "LEADS"),  # not a relation type
    ],
}


def test_one_unlisted_endpoint_no_longer_loses_the_people(db):  # noqa: F811
    chunk = _make_chunk(db, text=TEAM)
    report = extract_chunk(db, chunk, _provider(REPLY), floor=FLOOR, model="m")
    assert report.contract_violation is False
    names = {e.normalized_name for e in db.execute(select(Entity)).scalars()}
    assert {"ravi sharma", "asha rao", "platform team"} <= names
    # Both reporting lines are kept; the self edge and the invented LEADS are not.
    assert report.relationships_stored == 2
    assert report.relationships_discarded == 2
    assert {e.relation_type for e in db.execute(select(Relationship)).scalars()} == {"REPORTS_TO"}


def test_an_unlisted_reports_to_endpoint_is_adopted_as_a_person(db):  # noqa: F811
    """D3: the model names "the chief technology officer" as the target of an edge but
    does not list it under entities. The officer used to vanish with the edge; the edge
    only makes sense with a person at the end of it, so the endpoint is adopted."""
    chunk = _make_chunk(db, text=TEAM)
    report = extract_chunk(db, chunk, _provider(REPLY), floor=FLOOR, model="m")
    officer = db.execute(
        select(Entity).where(Entity.normalized_name == "chief technology officer")
    ).scalar_one()
    assert officer.entity_type == "person"
    assert report.entities_adopted == 1
    assert report.entities_stored == 4  # three listed plus the adopted officer
    ravi = db.execute(select(Entity).where(Entity.normalized_name == "ravi sharma")).scalar_one()
    edge = db.execute(
        select(Relationship).where(Relationship.source_entity_id == ravi.id)
    ).scalar_one()
    assert edge.target_entity_id == officer.id


def test_an_unlisted_endpoint_of_a_relation_with_no_clear_type_is_still_discarded(db):  # noqa: F811
    """OWNS and MENTIONS could end at anything, so inventing a type would store junk."""
    reply = {
        "entities": [_person("Ravi Sharma")],
        "relationships": [_rel("Ravi Sharma", "the team roadmap and hiring", "OWNS")],
    }
    reply["relationships"].append(_rel("Ravi Sharma", "BW-7731", "AUTHORED"))
    chunk = _make_chunk(db, text=TEAM)
    report = extract_chunk(db, chunk, _provider(reply), floor=FLOOR, model="m")
    assert (report.entities_adopted, report.relationships_stored) == (0, 0)
    assert report.relationships_discarded == 2
    assert db.execute(select(Entity)).scalars().all()[0].normalized_name == "ravi sharma"
    assert len(db.execute(select(Entity)).scalars().all()) == 1


def test_an_adopted_endpoint_that_looks_like_a_sentence_is_discarded(db):  # noqa: F811
    reply = {
        "entities": [_person("Ravi Sharma")],
        "relationships": [
            _rel("Ravi Sharma", "the person who handles all of the escalations in 2026")
        ],
    }
    chunk = _make_chunk(db, text=TEAM)
    report = extract_chunk(db, chunk, _provider(reply), floor=FLOOR, model="m")
    assert (report.entities_adopted, report.relationships_stored) == (0, 0)


def test_salvage_keeps_valid_entities_and_drops_invalid_ones():
    reply = {
        "entities": [
            _person("Ravi Sharma"),
            {"name": "x", "entity_type": "planet", "confidence": 0.9},
            _person("Asha Rao", c="high"),
        ],
        "relationships": [],
    }
    parsed, dropped_entities, dropped_relationships, adopted = salvage_extraction(json.dumps(reply))
    assert [e.name for e in parsed.entities] == ["Ravi Sharma"]
    assert (dropped_entities, dropped_relationships, adopted) == (2, 0, 0)


def test_a_name_under_two_types_keeps_both_entities_but_no_edge_through_it():
    reply = {
        "entities": [_person("Mercury"), _person("Mercury", "product"), _person("Asha Rao")],
        "relationships": [_rel("Asha Rao", "Mercury", "WORKS_ON")],
    }
    parsed, _, dropped_relationships, _ = salvage_extraction(json.dumps(reply))
    assert len(parsed.entities) == 3 and parsed.relationships == []
    assert dropped_relationships == 1


def test_a_reply_that_is_not_an_object_or_lacks_entities_is_still_a_violation():
    for raw in ("not json", json.dumps([1, 2]), json.dumps({"relationships": []})):
        assert isinstance(salvage_extraction(raw), ContractViolation)


def test_a_clean_reply_is_unchanged_by_salvage(db):  # noqa: F811
    clean = {
        "entities": [_person("Ravi Sharma"), _person("Asha Rao")],
        "relationships": [_rel("Asha Rao", "Ravi Sharma")],
    }
    chunk = _make_chunk(db, text=TEAM)
    provider = ScriptedLLMProvider(responses=[json.dumps(clean)], model="m")
    report = extract_chunk(db, chunk, provider, floor=FLOOR, model="m")
    assert (report.entities_stored, report.relationships_stored) == (2, 1)
    assert (report.entities_discarded, report.relationships_discarded) == (0, 0)
