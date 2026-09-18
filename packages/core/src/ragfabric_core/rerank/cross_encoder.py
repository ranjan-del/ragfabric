"""Rerank with a cross encoder model (optional extra: ragfabric[rerank]).

A cross encoder reads the question and one passage together and scores the pair
directly, which is more accurate than comparing two independently produced
embeddings. The cost is a forward pass per candidate and a torch install, which
is why it is an extra rather than a dependency.

Why the import is validated eagerly but the model is loaded lazily: a missing
extra is a configuration error and must fail fast and loudly at process start,
not on some user's first query hours later. But a later task wires
`build_reranker` into server startup, and eagerly instantiating the model
means downloading and loading multi gigabyte weights during that startup.
That ties process liveness to the download: a transient network blip or a
cold cache turns a config choice into a failed liveness probe and a crash
restart loop. So `__init__` imports and validates `sentence_transformers`
right away, but only instantiates the model on the first `rerank()` call,
caching it after that so the cost is paid once, at the first real query, not
on every process start.
"""

from __future__ import annotations

from ragfabric_core.providers.base import ProviderError
from ragfabric_core.rerank.base import all_finite, rescore_and_sort
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
            self._cross_encoder_cls = _import_cross_encoder()
        except ImportError as exc:
            raise ProviderError(
                "cross_encoder",
                "sentence-transformers is not installed. Install the extra with "
                "`uv pip install 'ragfabric[rerank]'` or set reranker.kind to none or llm.",
            ) from exc
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            self._model = self._cross_encoder_cls(self._model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self._load_model()
        raw_scores = model.predict([(query, c.text) for c in chunks])
        scores = [float(s) for s in raw_scores]
        if not all_finite(scores):
            # A model can emit NaN or infinity for a pair just as an LLM can
            # emit one in its score list; the degrade is the same, retrieval
            # order, never a partial rescore with an arbitrarily placed chunk.
            return list(chunks[:top_k])
        return rescore_and_sort(chunks, scores, top_k)
