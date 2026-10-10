"""Deterministic metrics: retrieval quality and citation correctness.

Relevance is decided by the question's ground truth (D3): a retrieved passage
is relevant to an expected source when it comes from that source's document
and, if the source names an evidence phrase, contains it (case and whitespace
insensitive). Nothing here calls a model.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from ragfabric_core.evaluation.dataset import ExpectedSource
from ragfabric_core.evaluation.target import ContextItem
from ragfabric_core.generate.contract import (
    NO_EVIDENCE,
    CitationViolation,
    assert_citation_contract,
    cited_markers,
)
from ragfabric_core.strategies.base import RetrievedChunk

# A cited sentence must share at least this share of its content words with
# the passages it cites (D7). Half is deliberately loose: it catches a marker
# pointing at the wrong passage, not a paraphrase.
SUPPORT_FLOOR = 0.5

_WORD = re.compile(r"[a-z0-9][a-z0-9\-.,]*[a-z0-9]|[a-z0-9]")
_MARKER = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
STOPWORDS = frozenset(
    """a an and are as at be been by can could did do does for from had has have he her his
    how i if in into is it its may might must no not of on or our per she should so than that
    the their them then there these they this those to under up was we were what when where
    which while who whom why will with would you your also each any all""".split()
)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content_words(text: str) -> set[str]:
    words = {w.strip(".,") for w in _WORD.findall(_MARKER.sub(" ", text.lower()))}
    return {w for w in words if w and w not in STOPWORDS}


def is_relevant(item: ContextItem, source: ExpectedSource) -> bool:
    if item.document != source.document:
        return False
    if source.evidence is None:
        return True
    return _flat(source.evidence) in _flat(item.text)


class RetrievalScores(BaseModel):
    precision: float | None
    recall: float | None
    hit: bool | None
    reciprocal_rank: float | None


def retrieval_scores(contexts: list[ContextItem], sources: list[ExpectedSource]) -> RetrievalScores:
    """Precision, recall, hit and reciprocal rank. All ``None`` without ground truth."""
    if not sources:
        return RetrievalScores(precision=None, recall=None, hit=None, reciprocal_rank=None)
    ranked = sorted(contexts, key=lambda c: c.rank)
    relevant = [any(is_relevant(c, s) for s in sources) for c in ranked]
    found = sum(1 for s in sources if any(is_relevant(c, s) for c in ranked))
    first = next((i for i, r in enumerate(relevant, start=1) if r), None)
    return RetrievalScores(
        precision=(sum(relevant) / len(ranked)) if ranked else None,
        recall=found / len(sources),
        hit=first is not None,
        reciprocal_rank=(1.0 / first) if first else 0.0,
    )


class CitationCheck(BaseModel):
    correct: bool | None
    reason: str | None = None


def citation_correct(answer: str, contexts: list[ContextItem]) -> CitationCheck:
    """The Phase 3 contract plus per-sentence support (D7).

    ``None`` for an empty answer: there is nothing to check. A no-evidence
    answer is correct only when nothing was retrieved; with passages in hand,
    declining to cite any of them is a citation failure the contract alone
    would let through.
    """
    if not answer.strip():
        return CitationCheck(correct=None, reason="empty answer")
    ranked = sorted(contexts, key=lambda c: c.rank)
    markers = cited_markers(answer)
    if not markers and NO_EVIDENCE in answer.lower():
        if ranked:
            return CitationCheck(
                correct=False, reason="no-evidence answer although passages were retrieved"
            )
        return CitationCheck(correct=True)
    chunks = [
        RetrievedChunk(chunk_id=i, document_id=0, collection_id=None, text=c.text)
        for i, c in enumerate(ranked, start=1)
    ]
    try:
        assert_citation_contract(answer, chunks)
    except CitationViolation as exc:
        return CitationCheck(correct=False, reason=exc.reason)
    if not markers:
        return CitationCheck(correct=False, reason="no [n] marker in the answer")
    for sentence in _SENTENCE.split(answer):
        cited = [int(n) for n in _MARKER.findall(sentence)]
        if not cited:
            continue
        words = content_words(sentence)
        if not words:
            continue
        support = set().union(*(content_words(ranked[n - 1].text) for n in cited))
        if len(words & support) / len(words) < SUPPORT_FLOOR:
            return CitationCheck(
                correct=False,
                reason=f"passage {cited} does not support: {sentence.strip()[:80]!r}",
            )
    return CitationCheck(correct=True)
