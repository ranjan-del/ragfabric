"""API key verification on real PostgreSQL, where timestamptz comes back offset-aware.

SQLite returns naive datetimes, so the unit tests in test_api_keys.py cannot
see a naive/aware mix. Needs RAGFABRIC_TEST_DATABASE_URL.
"""

import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.api_keys import create_api_key, verify_api_key
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models.user import User

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL"),
]


@pytest.fixture()
def db():
    downgrade(URL)
    upgrade(URL)
    engine = create_engine(URL)
    with sessionmaker(bind=engine, expire_on_commit=False)() as session:
        yield session


def _user(db) -> User:
    user = User(email="pg-key@example.com", hashed_password="h")
    db.add(user)
    db.commit()
    return user


def test_verify_api_key_twice_on_postgres(db):
    user = _user(db)
    _, plain = create_api_key(db, name="pg", user_id=user.id)
    db.commit()
    first = verify_api_key(db, plain)
    assert first is not None
    # A later request loads the row afresh, so last_used_at is read back from
    # PostgreSQL as an aware value.
    db.expire_all()
    assert verify_api_key(db, plain) is not None


def test_expired_and_future_keys_on_postgres(db):
    user = _user(db)
    _, expired = create_api_key(
        db, name="old", user_id=user.id, expires_at=datetime.now(UTC) - timedelta(days=1)
    )
    _, valid = create_api_key(
        db, name="new", user_id=user.id, expires_at=datetime.now(UTC) + timedelta(days=1)
    )
    db.commit()
    assert verify_api_key(db, expired) is None
    assert verify_api_key(db, valid) is not None
