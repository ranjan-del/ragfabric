from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.api_keys import (
    KEY_PREFIX,
    create_api_key,
    generate_api_key,
    hash_api_key,
    principal_for_api_key,
    verify_api_key,
)
from ragfabric_core.auth.ratelimit import check_rate_limit
from ragfabric_core.models import Base
from ragfabric_core.models.user import User
from ragfabric_core.stores.memory_cache import MemoryCache


def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'k.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_generate_and_hash():
    plain, prefix, digest = generate_api_key()
    assert (
        plain.startswith(KEY_PREFIX)
        and len(plain) > 30
        and plain.startswith(prefix)
        and len(prefix) == 12
    )
    assert digest == hash_api_key(plain) and digest != plain


def test_create_verify_expire_and_deactivate(tmp_path):
    db = session(tmp_path)
    user = User(email="u@x", hashed_password="h")
    db.add(user)
    db.commit()
    key, plain = create_api_key(
        db,
        name="ci",
        user_id=user.id,
        collection_ids=[1],
        strategies=["traditional"],
        rate_limit_per_minute=5,
    )
    db.commit()
    assert key.key_hash == hash_api_key(plain) and key.key_prefix == plain[:12]
    found = verify_api_key(db, plain)
    assert found is not None and found.id == key.id and found.last_used_at is not None
    assert verify_api_key(db, plain + "x") is None
    p = principal_for_api_key(db, found)
    assert p.user_id == user.id and p.api_key_id == key.id and p.role == "user"
    key.expires_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=1)
    db.commit()
    assert verify_api_key(db, plain) is None
    key.expires_at = None
    key.is_active = False
    db.commit()
    assert verify_api_key(db, plain) is None


def test_rate_limit_counts_per_minute():
    cache = MemoryCache()
    assert check_rate_limit(cache, "k1", 2, now=60.0) is True
    assert check_rate_limit(cache, "k1", 2, now=61.0) is True
    assert check_rate_limit(cache, "k1", 2, now=62.0) is False
    assert check_rate_limit(cache, "k1", 2, now=125.0) is True  # next minute
    assert check_rate_limit(cache, "k2", 2, now=62.0) is True  # other subject
