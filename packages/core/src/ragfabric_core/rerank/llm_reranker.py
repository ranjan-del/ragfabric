"""Rerank by asking the configured LLM to score each candidate.

Why scores rather than a reordered list: a model asked to reorder can drop or
duplicate items, and detecting that costs more than it saves. A fixed length
list of scores is checkable: wrong length, wrong type or unparseable output all
degrade to retrieval order, which is a worse ranking but never a wrong result
set. Retrieval already guaranteed the access filter, so a degraded rerank can
never leak a forbidden chunk.
"""

from __future__ import annotations

import json

from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.strategies.base import RetrievedChunk

_SYSTEM = (
    "You score how well each numbered passage answers the question. "
    "Reply with JSON only, in the form {\"scores\": [0.0, 1.0, ...]}, "
    "one score between 0 and 1 per passage, in the same order as the passages. "
    "Do not add commentary."
)
_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}}
    },
    "required": ["scores"],
}
_MAX_PASSAGE_CHARS = 1200


class LlmReranker:
    name = "llm"

    def __init__(self, llm: LLMProvider, model: str | None = None) -> None:
        self._llm = llm
        self._model = model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._score(query, chunks)
        if scores is None:
            return list(chunks[:top_k])
        rescored = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        rescored.sort(key=lambda c: (-(c.score or 0.0), c.chunk_id))
        return rescored[:top_k]

    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float] | None:
        passages = "\n\n".join(
            f"[{i}] {c.text[:_MAX_PASSAGE_CHARS]}" for i, c in enumerate(chunks, start=1)
        )
        messages = [
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=f"Question: {query}\n\nPassages:\n{passages}"),
        ]
        try:
            completion = self._llm.complete(
                messages, model=self._model, max_tokens=512, json_schema=_SCHEMA
            )
            scores = json.loads(completion.text)["scores"]
        except Exception:
            return None
        if not isinstance(scores, list) or len(scores) != len(chunks):
            return None
        try:
            return [float(s) for s in scores]
        except (TypeError, ValueError):
            return None
