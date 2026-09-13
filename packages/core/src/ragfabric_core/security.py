"""Security helpers: password hashing (bcrypt) and JWT encode/decode.

Passwords are hashed with bcrypt (slow + salted) and never stored in plaintext.
Access tokens are signed JWTs carrying the user id (``sub``) and a unique
``jti``. Secrets come from the environment only (see ``ragfabric_core.config``).
"""

import uuid
from datetime import timedelta

import bcrypt
from jose import JWTError, jwt

from ragfabric_core.config import settings
from ragfabric_core.models.base import utcnow

ACCESS = "access"

# bcrypt only considers the first 72 bytes of a password; truncate explicitly so
# longer inputs are handled predictably instead of raising.
_BCRYPT_MAX_BYTES = 72


def _to_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password with bcrypt (includes a random salt)."""
    return bcrypt.hashpw(_to_bytes(plain_password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if the plaintext matches the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(_to_bytes(plain_password), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str | int) -> str:
    """Issue a signed access token for the given user id."""
    now = utcnow()
    claims = {
        "sub": str(subject),
        "type": ACCESS,
        "iat": int(now.timestamp()),
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    """Decode + validate a JWT. Raises ``JWTError`` on bad signature/expiry."""
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


__all__ = [
    "ACCESS",
    "JWTError",
    "create_access_token",
    "decode_token",
    "hash_password",
    "verify_password",
]
