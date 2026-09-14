"""API keys: generated once, stored hashed, resolved to a Principal.

A key is `rf_` plus 32 URL safe random bytes. Only its SHA 256 is stored; the
12 character prefix is kept in clear so an operator can recognise a key in a
list without ever seeing the secret again.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import Principal
from ragfabric_core.auth.service import group_ids_for_user
from ragfabric_core.models.access import ApiKey
from ragfabric_core.models.user import User

KEY_PREFIX = "rf_"
PREFIX_LEN = 12


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
    return plaintext, plaintext[:PREFIX_LEN], hash_api_key(plaintext)


def create_api_key(
    db: Session,
    *,
    name: str,
    user_id: int,
    collection_ids=(),
    strategies=(),
    rate_limit_per_minute: int = 60,
    expires_at: datetime | None = None,
) -> tuple[ApiKey, str]:
    plaintext, prefix, digest = generate_api_key()
    key = ApiKey(
        name=name,
        key_prefix=prefix,
        key_hash=digest,
        principal_user_id=user_id,
        collection_ids=list(collection_ids),
        strategies=list(strategies),
        rate_limit_per_minute=rate_limit_per_minute,
        expires_at=expires_at,
    )
    db.add(key)
    db.flush()
    return key, plaintext


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def verify_api_key(db: Session, plaintext: str) -> ApiKey | None:
    if not plaintext.startswith(KEY_PREFIX):
        return None
    key = db.execute(
        select(ApiKey).where(ApiKey.key_hash == hash_api_key(plaintext))
    ).scalar_one_or_none()
    if key is None or not key.is_active:
        return None
    if key.expires_at is not None and key.expires_at <= _now():
        return None
    key.last_used_at = _now()
    db.commit()
    return key


def principal_for_user(db: Session, user: User) -> Principal:
    return Principal(
        user_id=user.id,
        email=user.email,
        role=user.role,
        group_ids=group_ids_for_user(db, user.id),
        api_key_id=None,
    )


def principal_for_api_key(db: Session, key: ApiKey) -> Principal:
    user = db.get(User, key.principal_user_id) if key.principal_user_id is not None else None
    return Principal(
        user_id=user.id if user else None,
        email=user.email if user else None,
        role=user.role if user else "user",
        group_ids=group_ids_for_user(db, user.id) if user else [],
        api_key_id=key.id,
    )
