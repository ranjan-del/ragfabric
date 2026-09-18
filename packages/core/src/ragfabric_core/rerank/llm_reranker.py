"""Rerank by asking the configured LLM to score each candidate.

Why scores rather than a reordered list: a model asked to reorder can drop or
duplicate items, and detecting that costs more than it saves. A fixed length
list of scores is checkable: wrong length, wrong type, a non-finite value (NaN
or infinity) or unparseable output all degrade to retrieval order, which is a
worse ranking but never a wrong result set. Retrieval already guaranteed the
access filter, so a degraded rerank can never leak a forbidden chunk.

Why there is a candidate cap: an unbounded prompt is not just an architectural
context limit risk. Ollama's runtime `num_ctx` commonly defaults to 2048 or
4096 tokens and, when the prompt exceeds it, silently truncates the prompt
rather than rejecting it. The model then returns a syntactically valid,
correct length score array computed against a partially truncated view of the
passages: a silent quality degradation that looks exactly like success and
that no response-shape check can catch. So the reranker refuses to build an
unbounded prompt in the first place: it caps how many candidates are ever
sent, independent of how many the caller hands it.
"""

from __future__ import annotations

import json

from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.rerank.base import all_finite, rescore_and_sort
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
_DEFAULT_MAX_CANDIDATES = 12


class LlmReranker:
    name = "llm"

    def __init__(
        self,
        llm: LLMProvider,
        model: str | None = None,
        max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self._llm = llm
        self._model = model
        self._max_candidates = max_candidates

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        # The cap can never reduce the result set below what the caller asked
        # for: it is at least top_k, so a caller requesting more than the
        # default cap (for example top_k=20 with max_candidates=12) still gets
        # every candidate it needs considered. Candidates beyond the cap are
        # dropped in retrieval order, which is safe precisely because the cut
        # to top_k happens anyway and the cap is never smaller than top_k, so
        # dropping them can never remove a chunk that would have been returned.
        effective_cap = max(self._max_candidates, top_k)
        candidates = chunks[:effective_cap]
        scores = self._score(query, candidates)
        if scores is None:
            return list(chunks[:top_k])
        return rescore_and_sort(candidates, scores, top_k)

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
            floats = [float(s) for s in scores]
        except (TypeError, ValueError):
            return None
        if not all_finite(floats):
            return None
        return floats
