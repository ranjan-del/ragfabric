from sqlalchemy import Column, Integer, MetaData, Table, select

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.stores.access_sql import access_clause

t = Table("t", MetaData(), Column("document_id", Integer), Column("collection_id", Integer))


def compile_(clause) -> str:
    return str(select(t).where(clause).compile(compile_kwargs={"literal_binds": True}))


def test_unrestricted_yields_no_clause():
    assert access_clause(AccessFilter.unrestricted(), t.c.document_id, t.c.collection_id) is None


def test_allow_lists_are_or_ed_and_deny_list_is_and_not():
    f = AccessFilter(
        document_ids=frozenset({1, 2}),
        collection_ids=frozenset({9}),
        denied_document_ids=frozenset({2}),
    )
    sql = compile_(access_clause(f, t.c.document_id, t.c.collection_id))
    assert "document_id IN (1, 2)" in sql and "collection_id IN (9)" in sql and "OR" in sql
    assert "document_id NOT IN (2)" in sql or "NOT IN (2)" in sql


def test_deny_only_filter_is_not_unrestricted_and_excludes():
    f = AccessFilter(denied_document_ids=frozenset({5}))
    sql = compile_(access_clause(f, t.c.document_id, t.c.collection_id))
    assert "NOT IN (5)" in sql and "IN (5)" in sql


def test_unrestricted_allow_axes_with_an_empty_deny_set_yields_no_clause():
    f = AccessFilter(document_ids=None, collection_ids=None, denied_document_ids=frozenset())
    assert access_clause(f, t.c.document_id, t.c.collection_id) is None


def test_empty_allow_lists_match_nothing():
    f = AccessFilter(document_ids=frozenset(), collection_ids=frozenset())
    sql = compile_(access_clause(f, t.c.document_id, t.c.collection_id))
    assert "false" in sql.lower() or "1 != 1" in sql


def test_chroma_where_is_none_when_unrestricted_and_no_model_pinned():
    from ragfabric_core.stores.chroma_store import chroma_where

    assert chroma_where(AccessFilter.unrestricted(), None) is None


def test_chroma_where_pins_the_model_even_when_access_is_unrestricted():
    from ragfabric_core.stores.chroma_store import chroma_where

    assert chroma_where(AccessFilter.unrestricted(), "nomic-embed-text") == {
        "model": {"$eq": "nomic-embed-text"}
    }


def test_chroma_where_ors_the_two_allow_axes_and_ands_the_deny_list():
    from ragfabric_core.stores.chroma_store import chroma_where

    access = AccessFilter(
        document_ids=frozenset({1, 2}),
        collection_ids=frozenset({7}),
        denied_document_ids=frozenset({2}),
    )
    where = chroma_where(access, None)
    assert where == {
        "$and": [
            {
                "$or": [
                    {"document_id": {"$in": [1, 2]}},
                    {"collection_id": {"$in": [7]}},
                ]
            },
            {"document_id": {"$nin": [2]}},
        ]
    }


def test_chroma_where_maps_an_empty_allow_set_to_a_predicate_that_matches_nothing():
    from ragfabric_core.stores.chroma_store import chroma_where

    where = chroma_where(AccessFilter(document_ids=frozenset(), collection_ids=None), None)
    assert where == {"document_id": {"$in": []}}, (
        "an empty allow set must match nothing, never everything"
    )
