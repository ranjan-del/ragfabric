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
from ragfabric_core.runtime import get_config

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
    rate_limit_per_minute: int | None = None,
    expires_at: datetime | None = None,
) -> tuple[ApiKey, str]:
    """Create a key. ``rate_limit_per_minute`` left unset (``None``, the
    default for every caller that names no explicit limit) takes the
    deployment's configured ``limits.rate_limit_per_minute`` at the moment the
    key is created, rather than a value fixed in this function's signature: a
    config key that only affected keys created before it was ever read would
    not really be "wired in".
    """
    plaintext, prefix, digest = generate_api_key()
    if rate_limit_per_minute is None:
        rate_limit_per_minute = get_config().limits.rate_limit_per_minute
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


# How often verify_api_key is willing to write last_used_at for the same key.
# Every authenticated read used to write this column on every single call,
# putting a write (and a commit) in the path of every request purely to
# refresh a "last used" display value nothing else depends on. Coalescing to
# once a minute keeps the column meaningfully fresh (an operator looking at
# "last used" never sees it stale by more than this) while cutting the write
# rate under sustained traffic to at most one per key per minute, with no new
# config surface: unlike a flag that could be turned off and silently stop
# updating the column at all, this always stays on and merely batches it.
LAST_USED_AT_COALESCE_SECONDS = 60


def verify_api_key(db: Session, plaintext: str) -> ApiKey | None:
    if not plaintext.startswith(KEY_PREFIX):
        return None
    key = db.execute(
        select(ApiKey).where(ApiKey.key_hash == hash_api_key(plaintext))
    ).scalar_one_or_none()
    if key is None or not key.is_active:
        return None
    now = _now()
    if key.expires_at is not None and key.expires_at <= now:
        return None
    stale = (
        key.last_used_at is None
        or (now - key.last_used_at).total_seconds() >= LAST_USED_AT_COALESCE_SECONDS
    )
    if stale:
        key.last_used_at = now
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
