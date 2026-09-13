"""Auth + user management routes.

- ``POST /register`` creates a user with a bcrypt-hashed password (role ``user``).
- ``POST /login`` verifies credentials and returns a signed JWT access token. It
  accepts an OAuth2 password form (``username`` = email, ``password``) so the
  Swagger "Authorize" button and standard clients work out of the box.
- ``GET /me`` returns the current authenticated user.
- ``POST /logout`` is a no-op for stateless JWTs (kept for client symmetry).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ragfabric_core.db.session import get_db
from ragfabric_core.models.user import Role, User
from ragfabric_core.security import create_access_token, hash_password, verify_password
from ragfabric_server.deps import get_current_user
from ragfabric_server.schemas.user import Token, UserCreate, UserOut

router = APIRouter()


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> User:
    """Create a new user account with a hashed password."""
    email = payload.email.lower()
    if db.query(User).filter(User.email == email).first() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with that email already exists.",
        )
    user = User(
        email=email,
        hashed_password=hash_password(payload.password),
        role=Role.USER.value,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Token:
    """Authenticate with email + password and receive a JWT access token."""
    email = form.username.lower()
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled.")
    return Token(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)) -> User:
    """Return the currently authenticated user."""
    return current_user


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)) -> dict:
    """Stateless logout — the client simply discards its token."""
    return {"detail": "Logged out. Discard the access token on the client."}
