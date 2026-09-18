"""Rerank with a cross encoder model (optional extra: ragfabric[rerank]).

A cross encoder reads the question and one passage together and scores the pair
directly, which is more accurate than comparing two independently produced
embeddings. The cost is a forward pass per candidate and a torch install, which
is why it is an extra rather than a dependency.
"""

from __future__ import annotations

from ragfabric_core.providers.base import ProviderError
from ragfabric_core.strategies.base import RetrievedChunk

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _import_cross_encoder():
    """Indirection so the tests can simulate a missing extra."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            cross_encoder = _import_cross_encoder()
        except ImportError as exc:
            raise ProviderError(
                "cross_encoder",
                "sentence-transformers is not installed. Install the extra with "
                "`uv pip install 'ragfabric[rerank]'` or set reranker.kind to none or llm.",
            ) from exc
        self._model = cross_encoder(model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks])
        rescored = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        rescored.sort(key=lambda c: (-(c.score or 0.0), c.chunk_id))
        return rescored[:top_k]
