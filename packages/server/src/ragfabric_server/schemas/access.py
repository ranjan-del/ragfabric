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
    rate_limit_per_minute: int = Field(default=60, ge=1, le=100000)
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
