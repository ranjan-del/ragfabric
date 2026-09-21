"""Pydantic schemas for authentication and user management."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PermissionUpdate(BaseModel):
    role: str | None = None
    is_active: bool | None = None


class AdminUserCreate(UserCreate):
    """An admin creating a user outright, rather than a self-service register.

    ``register`` deliberately hardcodes the ``user`` role so nobody can sign
    themselves up as an admin. This schema is only reachable behind the admin
    guard, so it may name the role and the initial active state.
    """

    role: str = "user"
    is_active: bool = True
