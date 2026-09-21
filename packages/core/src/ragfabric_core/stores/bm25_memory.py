"""In-process BM25 over rank_bm25. Not the production path, and it says so.

**The memory model, stated plainly.** This store keeps the entire corpus in
the memory of the process that built it. Under an API server with four worker
processes there are four copies. Nothing synchronises them: a document ingested
through worker 1 is invisible to workers 2, 3 and 4 until they are restarted,
so the copies drift apart the moment the corpus changes. The ceiling is RAM,
not disk. Every one of those properties is a reason stores/bm25_sql.py exists
and is the default.

What this is for: small corpora where the operational simplicity is worth
more than the correctness of a shared index, and cross-checking the SQL
implementation against a reference (packages/core/tests/test_bm25_memory.py
scores the same corpus both ways and requires the numbers to match).

**The cap is the point.** Without one, this store does not fail; it slowly
becomes the reason a deployment falls over, at a size nobody chose. Exceeding
max_chunks raises StoreCapacityError naming both the limit and the store to
use instead. It never silently truncates the corpus, because a store that
quietly answers from half its documents is worse than one that refuses.

**Why the IDF is overridden.** rank_bm25's BM25Okapi uses Robertson's IDF with
an epsilon floor for terms in more than half the corpus. bm25_sql uses the
Lucene variant, ``ln((N - df + 0.5) / (df + 0.5) + 1)``. Left alone the two
would disagree by construction, and the differential test would then be
comparing two different formulas rather than two implementations of one. The
saturation and length-normalisation half of BM25Okapi.get_scores is already
identical to ours, so only _calc_idf is replaced.

**What this store does not do.** It tokenises with the portable tokeniser
(stores/postgres_fts.token_list), so it does not stem and does not drop stop
words. On a PostgreSQL deployment the SQL store ranks over to_tsvector's
stemmed lexemes, so the two will not return identical orderings there. That is
a real difference between the two stores, not a bug in either.
"""

from __future__ import annotations

import math

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.stores.bm25_sql import DEFAULT_B, DEFAULT_K1, SUPPORTED_FILTERS
from ragfabric_core.stores.postgres_fts import token_list
from ragfabric_core.strategies.base import RetrievedChunk

# 50k chunks of roughly 600 tokens is already a few hundred megabytes per
# worker once Python object overhead is counted. Past that the SQL store is
# not an optimisation, it is the only thing that works.
DEFAULT_MAX_CHUNKS = 50_000


class StoreCapacityError(RuntimeError):
    """Raised when an in-process store is asked to hold more than it may."""


def _import_rank_bm25():
    """Indirection so a test can simulate the ``bm25`` extra being missing."""
    import rank_bm25

    return rank_bm25


class InMemoryBm25Store:
    name = "bm25_memory"

    def __init__(
        self,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self._max_chunks = max_chunks
        self._k1 = k1
        self._b = b
        self._texts: dict[int, str] = {}
        self._payloads: dict[int, dict] = {}
        self._index = None

    def count(self) -> int:
        return len(self._texts)

    def index(self, chunk_ids: list[int], texts: list[str], payloads: list[dict]) -> None:
        incoming = {
            chunk_id: (text, payload)
            for chunk_id, text, payload in zip(chunk_ids, texts, payloads, strict=True)
        }
        # Counted against the corpus this call would produce, not against this
        # call alone: re-indexing an existing chunk is not growth.
        would_hold = len(self._texts | incoming)
        if would_hold > self._max_chunks:
            raise StoreCapacityError(
                f"in-memory BM25 would hold {would_hold} chunks, limit is {self._max_chunks}; "
                f"use lexical_store.kind: bm25 for larger corpora"
            )
        for chunk_id, (text, payload) in incoming.items():
            self._texts[chunk_id] = text
            self._payloads[chunk_id] = dict(payload)
        self._index = None

    def delete_document(self, document_id: int) -> None:
        for chunk_id in [
            chunk_id
            for chunk_id, payload in self._payloads.items()
            if payload.get("document_id") == document_id
        ]:
            self._texts.pop(chunk_id, None)
            self._payloads.pop(chunk_id, None)
        self._index = None

    def search(
        self, query: str, top_k: int, access: AccessFilter, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        unknown = sorted(set(filters or {}) - set(SUPPORTED_FILTERS))
        if unknown:
            raise ValueError(
                f"unknown filter {unknown} for the bm25_memory store; "
                f"supported keys are {list(SUPPORTED_FILTERS)}"
            )
        terms = token_list(query)
        if not terms or not self._texts:
            return []

        scorer, order = self._build()
        scores = scorer.get_scores(terms)

        # The access filter and the metadata filters decide membership BEFORE
        # anything is ranked or cut (ADR 0003). This store's query is this
        # loop, so "inside the query" means here, not after the sort.
        scored: list[tuple[float, int]] = []
        for position, chunk_id in enumerate(order):
            payload = self._payloads[chunk_id]
            document_id = payload.get("document_id")
            collection_id = payload.get("collection_id")
            if not access.allows(document_id, collection_id):
                continue
            if not self._passes(payload, filters):
                continue
            score = float(scores[position])
            if score > 0:
                scored.append((score, chunk_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=self._payloads[chunk_id]["document_id"],
                collection_id=self._payloads[chunk_id].get("collection_id"),
                text=self._texts[chunk_id],
                score=score,
                metadata={
                    key: value
                    for key, value in self._payloads[chunk_id].items()
                    if key in ("filename", "format")
                },
            )
            for score, chunk_id in scored[:top_k]
        ]

    @staticmethod
    def _passes(payload: dict, filters: dict | None) -> bool:
        for key, value in (filters or {}).items():
            if payload.get(key) != value:
                return False
        return True

    def _build(self):
        """Rebuild the rank_bm25 index, which has no incremental update path."""
        if self._index is None:
            rank_bm25 = _import_rank_bm25()

            class _LuceneIdfBm25(rank_bm25.BM25Okapi):
                """BM25Okapi with bm25_sql's IDF, so the two cannot disagree."""

                def _calc_idf(self, nd):
                    for word, freq in nd.items():
                        self.idf[word] = math.log(
                            (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
                        )
                    self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0

            order = sorted(self._texts)
            corpus = [token_list(self._texts[chunk_id]) for chunk_id in order]
            self._index = (
                _LuceneIdfBm25(corpus, k1=self._k1, b=self._b),
                order,
            )
        return self._index
