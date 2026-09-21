import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth import service
from ragfabric_core.auth.policy import compute_access_filter
from ragfabric_core.auth.principal import Principal
from ragfabric_core.models import Base
from ragfabric_core.models.access import ApiKey, CollectionGrant
from ragfabric_core.models.document import Collection, Document
from ragfabric_core.models.user import User


@pytest.fixture()
def world(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'p.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    alice, bob, admin = (
        User(email="a@x", hashed_password="h"),
        User(email="b@x", hashed_password="h"),
        User(email="root@x", hashed_password="h", role="admin"),
    )
    db.add_all([alice, bob, admin])
    db.flush()
    open_c = Collection(name="open", owner_id=bob.id)
    hr = Collection(name="hr", owner_id=bob.id)
    mine = Collection(name="mine", owner_id=alice.id)
    db.add_all([open_c, hr, mine])
    db.flush()
    docs = {
        "open": Document(
            filename="o.txt", format="txt", collection_id=open_c.id, owner_id=bob.id, status="ready"
        ),
        "hr": Document(
            filename="h.txt", format="txt", collection_id=hr.id, owner_id=bob.id, status="ready"
        ),
        "hr_secret": Document(
            filename="s.txt", format="txt", collection_id=hr.id, owner_id=bob.id, status="ready"
        ),
        "mine": Document(
            filename="m.txt", format="txt", collection_id=mine.id, owner_id=alice.id, status="ready"
        ),
        "loose": Document(
            filename="l.txt", format="txt", collection_id=None, owner_id=bob.id, status="ready"
        ),
    }
    db.add_all(docs.values())
    db.flush()
    hr_group = service.create_group(db, "hr-team")
    service.grant_collection(db, hr_group.id, hr.id, "read")
    db.commit()
    yield (
        db,
        dict(alice=alice, bob=bob, admin=admin),
        dict(open=open_c, hr=hr, mine=mine),
        docs,
        hr_group,
    )
    db.close()


def principal(user, groups=(), api_key_id=None):
    return Principal(
        user_id=user.id,
        email=user.email,
        role=user.role,
        group_ids=list(groups),
        api_key_id=api_key_id,
    )


def test_admin_is_unrestricted(world):
    db, users, *_ = world
    assert compute_access_filter(db, principal(users["admin"])).is_unrestricted


def test_user_without_grant_sees_open_collections_own_collection_and_loose_documents_only(world):
    db, users, cols, docs, _ = world
    f = compute_access_filter(db, principal(users["alice"]))
    assert not f.is_unrestricted
    assert f.collection_ids == frozenset({cols["open"].id, cols["mine"].id})
    assert docs["loose"].id in f.document_ids and docs["mine"].id in f.document_ids
    assert f.allows(docs["hr"].id, cols["hr"].id) is False
    assert f.allows(docs["open"].id, cols["open"].id) is True


def test_group_grant_opens_the_collection(world):
    db, users, cols, docs, hr_group = world
    service.add_member(db, hr_group.id, users["alice"].id)
    db.commit()
    groups = service.group_ids_for_user(db, users["alice"].id)
    f = compute_access_filter(db, principal(users["alice"], groups))
    assert cols["hr"].id in f.collection_ids
    assert f.allows(docs["hr"].id, cols["hr"].id)


def test_deny_override_beats_a_collection_grant(world):
    db, users, cols, docs, hr_group = world
    service.add_member(db, hr_group.id, users["alice"].id)
    service.set_document_override(
        db, docs["hr_secret"].id, user_id=users["alice"].id, permission="deny"
    )
    db.commit()
    f = compute_access_filter(
        db, principal(users["alice"], service.group_ids_for_user(db, users["alice"].id))
    )
    assert f.allows(docs["hr"].id, cols["hr"].id) and not f.allows(
        docs["hr_secret"].id, cols["hr"].id
    )


def test_read_override_opens_one_document_in_a_restricted_collection(world):
    db, users, cols, docs, _ = world
    service.set_document_override(db, docs["hr"].id, user_id=users["alice"].id, permission="read")
    db.commit()
    f = compute_access_filter(db, principal(users["alice"]))
    assert f.allows(docs["hr"].id, cols["hr"].id) and not f.allows(
        docs["hr_secret"].id, cols["hr"].id
    )


def test_api_key_scopes_intersect(world):
    db, users, cols, docs, hr_group = world
    service.add_member(db, hr_group.id, users["alice"].id)
    key = ApiKey(
        name="k",
        key_prefix="rf_abc",
        key_hash="x",
        principal_user_id=users["alice"].id,
        collection_ids=[cols["hr"].id],
        strategies=[],
    )
    db.add(key)
    db.commit()
    f = compute_access_filter(
        db,
        principal(
            users["alice"], service.group_ids_for_user(db, users["alice"].id), api_key_id=key.id
        ),
    )
    assert f.collection_ids == frozenset({cols["hr"].id})
    assert docs["mine"].id not in f.document_ids and docs["loose"].id not in f.document_ids


def test_admin_owned_api_key_is_limited_to_its_scopes(world):
    db, users, cols, docs, hr_group = world
    key = ApiKey(
        name="k",
        key_prefix="rf_abc",
        key_hash="x",
        principal_user_id=users["admin"].id,
        collection_ids=[cols["hr"].id],
        strategies=[],
    )
    db.add(key)
    db.commit()
    f = compute_access_filter(db, principal(users["admin"], api_key_id=key.id))
    assert f.collection_ids == frozenset({cols["hr"].id})
    assert f.is_unrestricted is False
    assert not f.allows(docs["open"].id, cols["open"].id)


def test_dangling_or_inactive_api_key_fails_closed(world):
    db, users, cols, docs, hr_group = world
    f = compute_access_filter(db, principal(users["alice"], api_key_id=999999))
    assert f.collection_ids == frozenset() and f.document_ids == frozenset()
    key = ApiKey(
        name="k",
        key_prefix="rf_abc",
        key_hash="y",
        principal_user_id=users["alice"].id,
        collection_ids=[],
        strategies=[],
        is_active=False,
    )
    db.add(key)
    db.commit()
    g = compute_access_filter(db, principal(users["alice"], api_key_id=key.id))
    assert g.collection_ids == frozenset() and g.document_ids == frozenset()
    assert (
        compute_access_filter(db, principal(users["admin"], api_key_id=999999)).collection_ids
        == frozenset()
    )


def test_grant_upsert_and_revoke(world):
    db, users, cols, docs, hr_group = world
    g1 = service.grant_collection(db, hr_group.id, cols["hr"].id, "write")
    assert g1.permission == "write"
    service.revoke_collection(db, hr_group.id, cols["hr"].id)
    db.commit()
    f = compute_access_filter(db, principal(users["alice"], [hr_group.id]))
    assert cols["hr"].id in f.collection_ids  # no grants left, so the collection is open again


def test_grant_collection_upsert_is_idempotent(world):
    """The upsert replaced a read-then-write pattern (select for an existing
    row, then insert or update) that left a race window between two
    concurrently granting callers. A genuinely concurrent test against a
    single SQLite connection in this fixture cannot exercise that race
    directly, so this instead asserts the property the atomic upsert is
    supposed to guarantee: granting the same (group, collection) pair twice
    never raises and never leaves two rows for the same pair, which a
    non-atomic insert racing with itself could produce (two inserts both
    passing a "no existing row" check, one then failing the unique
    constraint, or worse, succeeding as a duplicate without one).
    """
    db, users, cols, docs, hr_group = world
    new_group = service.create_group(db, "finance-team")
    db.flush()

    first = service.grant_collection(db, new_group.id, cols["hr"].id, "read")
    second = service.grant_collection(db, new_group.id, cols["hr"].id, "read")
    db.commit()

    assert first.id == second.id
    rows = (
        db.query(CollectionGrant)
        .filter(
            CollectionGrant.group_id == new_group.id,
            CollectionGrant.collection_id == cols["hr"].id,
        )
        .all()
    )
    assert len(rows) == 1
    assert rows[0].permission == "read"


def _count_statements(engine, fn):
    """Run ``fn()`` and return how many SQL statements it sent to ``engine``."""
    count = 0

    def _tick(*_args, **_kwargs):
        nonlocal count
        count += 1

    event.listen(engine, "before_cursor_execute", _tick)
    try:
        fn()
    finally:
        event.remove(engine, "before_cursor_execute", _tick)
    return count


def test_compute_access_filter_query_count_does_not_grow_with_corpus_size(world):
    """Task 16 C3: the collections and documents axes were each a single
    unscoped table read (every grant, then every collection; every uncollected
    document) whose result set, and whose query count next to the group/owner
    reads run alongside them, both scaled with total corpus size rather than
    with what the principal can actually see. This asserts the number of SQL
    statements ``compute_access_filter`` issues stays flat as the corpus grows
    by two orders of magnitude, and stays low in absolute terms: one query for
    the collections axis, one for the documents axis, one for overrides.
    """
    db, users, cols, docs, hr_group = world
    engine = db.get_bind()
    groups = service.group_ids_for_user(db, users["alice"].id)
    p = principal(users["alice"], groups)

    baseline = _count_statements(engine, lambda: compute_access_filter(db, p))
    assert baseline <= 4, (
        "compute_access_filter should need about one query per axis plus the "
        f"override lookup, not {baseline}"
    )

    # Grow the corpus by two orders of magnitude with rows alice has no
    # grant, ownership or override on, so none of it is data she is allowed
    # to see -- only the two axis queries' WHERE clauses should be touched by
    # this, never their count.
    for i in range(300):
        db.add(Collection(name=f"extra-collection-{i}", owner_id=users["bob"].id))
    for i in range(300):
        db.add(
            Document(
                filename=f"extra-{i}.txt",
                format="txt",
                collection_id=None,
                owner_id=users["bob"].id,
                status="ready",
            )
        )
    db.commit()

    scaled = _count_statements(engine, lambda: compute_access_filter(db, p))
    assert scaled == baseline, (
        "compute_access_filter issued a different number of SQL statements after "
        f"the corpus grew: {baseline} before, {scaled} after"
    )
