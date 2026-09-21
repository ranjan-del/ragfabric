"""Store interfaces (ADR 0003: every query takes an AccessFilter).

The filter is an argument, not an afterthought, so an implementation cannot
forget it: a store that ranks first and filters later cannot satisfy these
signatures without lying, and the contract tests in later phases catch that.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.strategies.base import RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def upsert(
        self, chunk_ids: list[int], vectors: list[list[float]], payloads: list[dict]
    ) -> None: ...
    def query(
        self, vector: list[float], top_k: int, access: AccessFilter, filters: dict | None = None
    ) -> list[RetrievedChunk]: ...
    def delete_document(self, document_id: int) -> None: ...
    def count(self) -> int: ...
    def access_stats(self, filters: dict, access: AccessFilter) -> tuple[int, int]: ...


@runtime_checkable
class LexicalStore(Protocol):
    name: str

    def index(self, chunk_ids: list[int], texts: list[str], payloads: list[dict]) -> None: ...
    def search(
        self, query: str, top_k: int, access: AccessFilter, filters: dict | None = None
    ) -> list[RetrievedChunk]: ...
    def delete_document(self, document_id: int) -> None: ...


@runtime_checkable
class GraphStore(Protocol):
    name: str

    def upsert_entities(self, entities: list[dict]) -> None: ...
    def upsert_relationships(self, relationships: list[dict]) -> None: ...
    def neighbours(
        self, entity_ids: list[int], hops: int, access: AccessFilter, max_nodes: int
    ) -> dict: ...
    def delete_document(self, document_id: int) -> None: ...


@runtime_checkable
class Cache(Protocol):
    name: str

    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None: ...
    def incr(self, key: str, ttl_seconds: int | None = None) -> int: ...
