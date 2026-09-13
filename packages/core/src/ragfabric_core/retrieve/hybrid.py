"""Hybrid retrieval: blend lexical overlap with semantic similarity.

Pure vector search can miss exact keyword matches (rare terms, product codes,
acronyms) whose meaning the hashing embedder smears across buckets. Pure keyword
search misses paraphrases. Hybrid search fuses both:

    fused = alpha * semantic + (1 - alpha) * lexical

Both component scores are min-max normalised to ``[0, 1]`` over the candidate set
before fusing, so neither scale dominates the other. ``alpha`` comes from
``settings.hybrid_alpha`` (default 0.6, i.e. slightly semantic-leaning).

The lexical score is the overlap coefficient between the query tokens and the
chunk tokens — simple, dependency-free, and reproducible, which keeps the whole
path offline and testable.
"""

from __future__ import annotations

import numpy as np

from ragfabric_core.config import settings
from ragfabric_core.ingest.embed import content_tokens, get_embedder
from ragfabric_core.store.vector_store import get_store


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale an array to ``[0, 1]``; a flat array maps to all-zeros."""
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _lexical_score(query_tokens: set[str], text: str) -> float:
    """Overlap coefficient: fraction of query CONTENT tokens present in a chunk.

    Stopwords are excluded on both sides. Including them would push this toward
    1.0 for every chunk in the corpus, since "the"/"of"/"is" appear everywhere,
    which is exactly the failure mode hybrid search is supposed to avoid.
    """
    if not query_tokens:
        return 0.0
    doc_tokens = set(content_tokens(text))
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


class HybridRetriever:
    """Blend keyword overlap and vector similarity, then re-rank."""

    def __init__(self, store=None, embedder=None, alpha: float | None = None):
        self.store = store or get_store()
        self.embedder = embedder or get_embedder()
        self.alpha = settings.hybrid_alpha if alpha is None else alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        collection_id: int | None = None,
        document_id: int | None = None,
        format: str | None = None,
    ) -> list[dict]:
        """Return the ``top_k`` chunks ranked by the fused score."""
        meta = self.store.all_meta()
        if not meta:
            return []

        # Semantic scores for every stored chunk (cosine similarity).
        query_vector = self.embedder.embed_one(query)
        semantic = self.store.score_all(query_vector)

        # Restrict to the candidates that pass the metadata filters. The store
        # owns filter semantics so semantic and hybrid modes cannot disagree.
        candidates = self.store.candidate_rows(
            {
                "collection_id": collection_id,
                "document_id": document_id,
                "format": format,
            }
        )
        if not candidates:
            return []

        query_tokens = set(content_tokens(query))
        semantic_c = np.array([semantic[i] for i in candidates], dtype=np.float32)
        lexical_c = np.array(
            [_lexical_score(query_tokens, meta[i]["text"]) for i in candidates],
            dtype=np.float32,
        )

        fused = self.alpha * _minmax(semantic_c) + (1 - self.alpha) * _minmax(lexical_c)

        # Stable ordering: sort on (-fused, chunk_id) so equal scores resolve the
        # same way on every process, which matters after an index rebuild.
        order = sorted(
            range(len(candidates)),
            key=lambda r: (-float(fused[r]), meta[candidates[r]].get("chunk_id") or 0),
        )[:top_k]
        results: list[dict] = []
        for rank in order:
            idx = candidates[rank]
            record = dict(meta[idx])
            # ``score`` stays the raw cosine so it means the same thing in both
            # modes and can be compared across queries. The fused value is a
            # min-max rank within THIS result set (its top is always ~1.0), so it
            # is exposed separately rather than passed off as a similarity.
            record["score"] = float(semantic[idx])
            record["lexical_score"] = float(lexical_c[rank])
            record["hybrid_score"] = float(fused[rank])
            results.append(record)
        return results
