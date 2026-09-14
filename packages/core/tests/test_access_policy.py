import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth import service
from ragfabric_core.auth.policy import compute_access_filter
from ragfabric_core.auth.principal import Principal
from ragfabric_core.models import Base
from ragfabric_core.models.access import ApiKey
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
