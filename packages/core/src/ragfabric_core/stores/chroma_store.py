"""VectorStore over a Chroma collection.

Why a second vector store exists at all: the product claims every interface has
at least two real implementations, so an adopter can believe a third one is
possible. pgvector and Chroma are the two for VectorStore, and they are held to
the same contract, including ADR 0003: the access predicate goes into Chroma's
own `where` document so a forbidden vector is never ranked, not filtered out of
the results afterwards.

Why text is not stored here: RetrievedChunk needs text, page, section and
character spans, all of which live on the chunks row. Duplicating them into
Chroma metadata would create a second source of truth that drifts on the first
edit. Chroma holds vectors plus exactly the metadata the access predicate
needs, and hits are resolved back to chunks rows through the session factory.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.embeddings.normalise import normalise
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.strategies.base import RetrievedChunk

NO_COLLECTION = -1


def _cid(collection_id: int | None) -> int:
    return NO_COLLECTION if collection_id is None else int(collection_id)


def _prune_unsatisfiable(node: dict | None) -> tuple[dict | None, bool]:
    """Remove ``{"field": {"$in": []}}`` leaves before a where document reaches Chroma.

    ``chroma_where`` deliberately maps an empty allow set to exactly that leaf
    (ADR 0003: an empty allow set must match nothing, never everything), which is
    correct as a predicate, but Chroma's server rejects an empty ``$in`` operand
    as invalid input rather than treating it as "never true". So the leaf is
    evaluated here, in the store, before the request goes out: an ``$and``
    containing one is entirely unsatisfiable (short-circuits the query, no
    request sent), and inside an ``$or`` it is simply dropped, since one
    impossible branch does not change whether the other branches can match.

    Returns ``(pruned_node, always_false)``. ``always_false`` means the caller
    must return no results without querying Chroma at all.
    """
    if node is None:
        return None, False
    if "$and" in node:
        # `kept` cannot end up empty here: chroma_where never builds an `$and`
        # with fewer than two parts, and `query`'s filter merge only wraps an
        # `$and` around an existing, non-empty `where`. If that invariant were
        # ever violated by a future caller composing a where document by hand,
        # `kept[0]` below raises IndexError, which fails loud rather than
        # silently falling back to "no restriction" (an unfiltered query),
        # the dangerous, fail-open direction for access-control code.
        kept: list[dict] = []
        for child in node["$and"]:
            pruned, always_false = _prune_unsatisfiable(child)
            if always_false:
                return None, True
            if pruned is not None:
                kept.append(pruned)
        return ({"$and": kept} if len(kept) > 1 else kept[0]), False
    if "$or" in node:
        kept = []
        for child in node["$or"]:
            pruned, always_false = _prune_unsatisfiable(child)
            if not always_false and pruned is not None:
                kept.append(pruned)
        if not kept:
            return None, True
        return ({"$or": kept} if len(kept) > 1 else kept[0]), False
    ((_, op_value),) = node.items()
    ((op, value),) = op_value.items()
    if op == "$in" and value == []:
        return None, True
    return node, False


def chroma_where(access: AccessFilter, model: str | None) -> dict | None:
    """Translate an AccessFilter (and the pinned model) into a Chroma where document."""
    parts: list[dict] = []
    if not access.is_unrestricted:
        if access.document_ids is not None or access.collection_ids is not None:
            allows: list[dict] = []
            if access.document_ids is not None:
                allows.append({"document_id": {"$in": sorted(access.document_ids)}})
            if access.collection_ids is not None:
                allows.append({"collection_id": {"$in": sorted(access.collection_ids)}})
            parts.append({"$or": allows} if len(allows) > 1 else allows[0])
        if access.denied_document_ids:
            parts.append({"document_id": {"$nin": sorted(access.denied_document_ids)}})
    if model is not None:
        parts.append({"model": {"$eq": model}})
    if not parts:
        return None
    return {"$and": parts} if len(parts) > 1 else parts[0]


class ChromaVectorStore:
    name = "chroma"

    def __init__(
        self,
        client,
        collection_name: str = "ragfabric_chunks",
        model: str | None = None,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        self._client = client
        self._name = collection_name
        self._model = model
        self._sf = session_factory

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def _collection(self):
        # cosine, to match pgvector's operator class and the normalised vectors.
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self, chunk_ids: list[int], vectors: list[list[float]], payloads: list[dict]
    ) -> None:
        if not chunk_ids:
            return
        self._collection.upsert(
            ids=[str(cid) for cid in chunk_ids],
            embeddings=[normalise(v) for v in vectors],
            metadatas=[
                {
                    "chunk_id": int(cid),
                    "document_id": int(p["document_id"]),
                    "collection_id": _cid(p.get("collection_id")),
                    "model": str(p["model"]),
                    "dim": int(p.get("dim", len(v))),
                }
                for cid, v, p in zip(chunk_ids, vectors, payloads, strict=True)
            ],
        )

    def query(
        self, vector: list[float], top_k: int, access: AccessFilter, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        where = chroma_where(access, self._model)
        for key in ("document_id", "collection_id"):
            if filters and filters.get(key) is not None:
                clause = {
                    key: {"$eq": _cid(filters[key]) if key == "collection_id" else filters[key]}
                }
                where = {"$and": [where, clause]} if where else clause
        where, always_false = _prune_unsatisfiable(where)
        if always_false:
            return []
        res = self._collection.query(
            query_embeddings=[normalise(vector)],
            n_results=top_k,
            where=where or None,
        )
        ids = [int(i) for i in (res.get("ids") or [[]])[0]]
        distances = (res.get("distances") or [[]])[0]
        if not ids:
            return []
        scores = {cid: 1.0 - float(d) for cid, d in zip(ids, distances, strict=True)}
        rows = self._resolve(ids)
        fmt_filter = (filters or {}).get("format")
        out = [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                collection_id=chunk.collection_id,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                score=scores.get(chunk.id),
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                metadata={"filename": filename, "format": fmt},
            )
            for chunk, filename, fmt in rows
            if fmt_filter is None or fmt == fmt_filter
        ]
        out.sort(key=lambda c: (-(c.score or 0.0), c.chunk_id))
        return out[:top_k]

    def _resolve(self, chunk_ids: list[int]):
        if self._sf is None:
            raise RuntimeError("ChromaVectorStore needs a session_factory to resolve chunk text")
        with self._sf() as db:
            return db.execute(
                select(Chunk, Document.filename, Document.format)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id.in_(chunk_ids))
            ).all()

    def delete_document(self, document_id: int) -> None:
        self._collection.delete(where={"document_id": {"$eq": int(document_id)}})

    def count(self) -> int:
        return int(self._collection.count())

    def access_stats(self, filters: dict, access: AccessFilter) -> tuple[int, int]:
        """Candidate counts before and after the access filter.

        Chroma has no count-with-predicate primitive: ``count()`` takes no
        ``where`` at all, so there is still no way to ask Chroma for two
        different counts (unrestricted vs. access-restricted) in a single
        aggregate the way ``PgVectorStore`` folds both into one
        conditional-aggregate SQL query. What Chroma's ``get()`` *does* give
        us, though, is every candidate id's stored metadata, including the
        ``document_id`` and ``collection_id`` fields the access predicate is
        actually evaluated over (``chroma_where`` builds that same predicate
        as a Chroma ``where`` document elsewhere in this class). So rather
        than sending the metadata-only filters once and the metadata+access
        filters again, this makes a single ``get(where=..., include=
        ["metadatas"])`` call for the unrestricted candidate set, and decides
        "after" for each row in Python with ``AccessFilter.allows``, which
        encodes the identical allow/deny logic ``chroma_where`` sends to
        Chroma. That is one Chroma round trip total, down from two, and every
        number is still measured off real metadata, never estimated (ADR
        0004). When ``filters`` includes ``format`` (not stored in Chroma
        metadata, per this module's docstring) the surviving ids are resolved
        against the chunks/documents tables once, also down from the
        previous two such lookups, because "after" is always a subset of
        "before" and both can be read off the same resolved format map.
        """
        fmt = (filters or {}).get("format")
        where = chroma_where(AccessFilter.unrestricted(), self._model)
        for key in ("document_id", "collection_id"):
            if filters and filters.get(key) is not None:
                clause = {
                    key: {"$eq": _cid(filters[key]) if key == "collection_id" else filters[key]}
                }
                where = {"$and": [where, clause]} if where else clause
        where, always_false = _prune_unsatisfiable(where)
        if always_false:
            return 0, 0

        res = self._collection.get(where=where, include=["metadatas"])
        ids = [int(i) for i in res.get("ids", [])]
        metadatas = res.get("metadatas") or []

        fmt_by_id = None
        if fmt is not None and ids:
            fmt_by_id = {chunk.id: row_fmt for chunk, _, row_fmt in self._resolve(ids)}

        before = after = 0
        for chunk_id, meta in zip(ids, metadatas, strict=True):
            if fmt_by_id is not None and fmt_by_id.get(chunk_id) != fmt:
                continue
            before += 1
            document_id = int(meta["document_id"])
            raw_collection_id = meta.get("collection_id")
            collection_id = None if raw_collection_id == NO_COLLECTION else int(raw_collection_id)
            if access.allows(document_id, collection_id):
                after += 1
        return before, after
