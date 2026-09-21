from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""


class GroupOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: str
    created_at: datetime


class MemberAdd(BaseModel):
    user_id: int


class GrantCreate(BaseModel):
    group_id: int
    collection_id: int
    permission: str = Field(default="read", pattern="^(read|write)$")


class GrantOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    group_id: int
    collection_id: int
    permission: str


class OverrideCreate(BaseModel):
    document_id: int
    group_id: int | None = None
    user_id: int | None = None
    permission: str = Field(default="deny", pattern="^(deny|read)$")


class OverrideOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    document_id: int
    group_id: int | None
    user_id: int | None
    permission: str


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    user_id: int
    collection_ids: list[int] = []
    strategies: list[str] = []
    # None (the default: no explicit limit named) takes the deployment's
    # configured limits.rate_limit_per_minute at creation time instead of a
    # value fixed in this schema.
    rate_limit_per_minute: int | None = Field(default=None, ge=1, le=100000)
    expires_at: datetime | None = None


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    key_prefix: str
    principal_user_id: int | None
    collection_ids: list[int]
    strategies: list[str]
    rate_limit_per_minute: int
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None


class ApiKeyCreated(ApiKeyOut):
    key: str


class GroupUpdate(BaseModel):
    """Rename a group or change its description.

    Both fields are optional so the console can send only what the operator
    edited, and ``None`` means "leave this alone" rather than "clear it".
    """

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = None
