"""Shared FastAPI dependencies: current-user resolution and role guards.

- ``get_current_user`` decodes the bearer access token and loads the user.
- ``require_role("admin")`` builds a dependency that also checks the role.

401 Unauthorized -> we don't know who you are (bad/missing/expired token).
403 Forbidden    -> we know who you are, but you're not allowed.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ragfabric_core.security import ACCESS, JWTError, decode_token
from ragfabric_core.db.session import get_db
from ragfabric_core.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login", auto_error=False)

_credentials_error = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Return the active user for the request's access token."""
    if not token:
        raise _credentials_error
    try:
        payload = decode_token(token)
    except JWTError:
        raise _credentials_error

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
