"""Shared FastAPI dependencies: current-user resolution and role guards.

- ``get_current_user`` decodes the bearer access token and loads the user.
- ``require_role("admin")`` builds a dependency that also checks the role.
- ``get_principal`` resolves the caller as a Principal: an API key (header or
  bearer) or a signed in user, so route handlers can depend on one thing
  regardless of how the caller authenticated.
- ``get_access_filter`` computes the AccessFilter for that principal.
- ``get_cache`` builds the process wide Cache once from configuration.

401 Unauthorized -> we don't know who you are (bad/missing/expired token).
403 Forbidden    -> we know who you are, but you're not allowed.
429 Too Many Requests -> an API key went over its rate limit.
"""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ragfabric_core.auth.api_keys import (
    KEY_PREFIX,
    principal_for_api_key,
    principal_for_user,
    verify_api_key,
)
from ragfabric_core.auth.policy import compute_access_filter
from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.auth.ratelimit import check_rate_limit
from ragfabric_core.db.session import get_db
from ragfabric_core.models.user import User
from ragfabric_core.runtime import get_config
from ragfabric_core.security import ACCESS, JWTError, decode_token
from ragfabric_core.stores.base import Cache
from ragfabric_core.stores.registry import build_cache

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)

_credentials_error = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

_cache: Cache | None = None


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Return the active user for the request's access token."""
    if not token:
        raise _credentials_error
    try:
        payload = decode_token(token)
    except JWTError as exc:
        raise _credentials_error from exc

    if payload.get("type") != ACCESS:
        raise _credentials_error
    user_id = payload.get("sub")
    if user_id is None:
        raise _credentials_error

    user = db.get(User, int(user_id))
    if user is None or not user.is_active:
        raise _credentials_error
    return user


def require_role(role: str):
    """Dependency factory enforcing a required role (e.g. 'admin')."""

    def _guard(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user

    return _guard


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = build_cache(get_config().cache)
    return _cache


def _api_key_from_request(request: Request, token: str | None) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header
    if token and token.startswith(KEY_PREFIX):
        return token
    return None


def get_principal(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Principal:
    """Resolve the caller: an API key (header or bearer) or a JWT user."""
    plaintext = _api_key_from_request(request, token)
    if plaintext is not None:
        key = verify_api_key(db, plaintext)
        if key is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        if not check_rate_limit(get_cache(), f"key:{key.id}", key.rate_limit_per_minute):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded"
            )
        return principal_for_api_key(db, key)
    user = get_current_user(token=token, db=db)
    return principal_for_user(db, user)


def get_access_filter(
    principal: Principal = Depends(get_principal), db: Session = Depends(get_db)
) -> AccessFilter:
    return compute_access_filter(db, principal)
