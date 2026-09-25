"""Answer assembly: answer text, confidence, citations, highlights, source doc.

Takes the ranked chunks from a retriever and packages the full answer payload the
UI renders:

- ``answer``          grounded answer text (offline extractive by default).
- ``confidence``      a composite score in ``[0, 1]`` (see :func:`_confidence`).
- ``citations``       one entry per retrieved chunk (``[1]``, ``[2]`` ...) with
                      source filename, page, score, snippet, per-snippet
                      highlight spans, the exact span the answer was taken from,
                      and whether the answer actually used it.
- ``highlights``      query terms located in the ANSWER text, with character
                      spans, so the frontend can mark them without re-deriving
                      the match positions in the browser.
- ``source_document`` the primary (top-ranked) source's metadata.

Everything here is derived from the retrieved data. Nothing is a fixed
placeholder: confidence comes from measured term coverage and cosine similarity,
highlight offsets come from real regex match positions, the supporting span is
the sentence the extractive answer actually quoted, and the ``used`` flag comes
from parsing the markers back out of the answer itself.

One invariant holds across the whole payload: every ``start``/``end`` pair
indexes into a string that is ALSO in the payload (``answer`` for the top-level
highlights, ``citation.snippet`` for the per-citation ones). Offsets into a
string the client never receives cannot be checked and cannot be rendered, so
they are worth no more than a placeholder.
"""

from __future__ import annotations

import math
import re

from ragfabric_core.config import settings
from ragfabric_core.generate import llm
from ragfabric_core.ingest.embed import content_tokens

_MAX_SNIPPET = 240

# Terms shorter than this are too noisy to be worth highlighting.
_MIN_HIGHLIGHT_LEN = 3

# Written where a snippet is cut out of a longer chunk.
_ELLIPSIS = "..."


def _assemble_context(chunks: list[dict], budget: int | None = None) -> str:
    """Concatenate ranked chunks into a ``[n]``-marked context within a budget."""
    budget = settings.max_context_chars if budget is None else budget
    parts: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        block = f"[{i}] {chunk.get('text', '')}"
        if used + len(block) > budget and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _find_term_spans(query: str, text: str) -> list[dict]:
    """Locate query content terms inside ``text`` as non-overlapping spans.

    Matching is word-boundary anchored. A plain substring search would highlight
    "cat" inside "category", which reads as a bug to anyone looking at the result
    and quietly inflates anything measured on top of it.
    """
    if not text:
        return []
    terms = {t for t in content_tokens(query) if len(t) >= _MIN_HIGHLIGHT_LEN}
    if not terms:
        return []

    # Longest first so "authentication" wins over a shorter term sharing a prefix.
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    pattern = re.compile(rf"\b({alternatives})\b", re.IGNORECASE)

    spans = [
        {"term": match.group(0).lower(), "start": match.start(), "end": match.end()}
        for match in pattern.finditer(text)
    ]
    # Sort by position and drop overlaps so the frontend can splice cleanly.
    spans.sort(key=lambda s: (s["start"], -s["end"]))
    deduped: list[dict] = []
    last_end = -1
    for span in spans:
        if span["start"] >= last_end:
            deduped.append(span)
            last_end = span["end"]
    return deduped


def _confidence(query: str, retrieved: list[dict]) -> float:
    """Composite confidence for the top result, in ``[0, 1]``.

    Raw cosine similarity is a poor confidence signal for a bag-of-words
    embedder because it is length-biased: a short question against an 800
    character chunk tops out well below 1.0 even for a perfect match, so every
    answer would render as "low confidence". Three measured signals are blended
    instead:

    ``coverage``  fraction of the question's content words that appear in the top
                  chunk. The most informative of the three: if the words you
                  asked about are not in the source, the answer is a guess.
    ``relevance`` cosine similarity divided by the best cosine a chunk of that
                  length could achieve. For unit-norm bag-of-words vectors, a
                  chunk of ``N`` distinct tokens that contains all ``n`` query
                  tokens scores about ``sqrt(n / N)``; dividing by that removes
                  the length bias.
    ``support``   fraction of the OTHER retrieved chunks that also cover at least
                  half the question, i.e. whether the corpus corroborates itself.

    The weights are a judgement call, not a calibration against labelled data.
    Read the number as a ranked heuristic, not a probability.

    A top chunk with no score (the graph strategy's, since a traversal
    measures no similarity, ruling R18) has no ``relevance`` to measure. It is
    left out and the other two weights are rescaled to sum to one, rather than
    counting the missing cosine as zero, which would report "measured as
    irrelevant" for a number nobody measured (ADR 0004).
    """
    if not retrieved:
        return 0.0

    query_tokens = set(content_tokens(query))
    if not query_tokens:
        return 0.0

    top = retrieved[0]
    top_tokens = set(content_tokens(top.get("text", "")))
    coverage = len(query_tokens & top_tokens) / len(query_tokens)

    top_score = top.get("score", 0.0)
    if top_score is None:
        relevance = None
    else:
        cosine = max(0.0, float(top_score))
        n = len(query_tokens)
        best_possible = math.sqrt(n / max(n, len(top_tokens) or n))
        relevance = min(1.0, cosine / best_possible) if best_possible > 0 else 0.0

    others = retrieved[1:]
    if others:
        corroborating = sum(
            1
            for chunk in others
            if len(query_tokens & set(content_tokens(chunk.get("text", ""))))
            >= len(query_tokens) / 2
        )
        support = corroborating / len(others)
    else:
        support = 0.0

    if relevance is None:
        score = (0.5 * coverage + 0.1 * support) / 0.6
    else:
        score = 0.5 * coverage + 0.4 * relevance + 0.1 * support
    return round(max(0.0, min(1.0, score)), 4)


def _snippet_window(text: str, span: tuple[int, int] | None) -> tuple[str, int]:
    """Cut the displayed snippet out of a chunk, centred on ``span``.

    Returns ``(snippet, shift)`` where ``shift`` converts an offset in ``text``
    into the matching offset in ``snippet``.

    A chunk is up to 800 characters but the UI shows 240, so a naive
    ``text[:240]`` regularly cuts away the very sentence the answer quoted,
    leaving a "supporting" snippet that does not support anything. Centring the
    window on the supporting span keeps the quoted sentence visible.
    """
    if len(text) <= _MAX_SNIPPET:
        return text, 0

    if span is None:
        start = 0
    else:
        span_start, span_end = span
        slack = max(0, _MAX_SNIPPET - (span_end - span_start))
        start = max(0, span_start - slack // 2)
        start = min(start, len(text) - _MAX_SNIPPET)

    body = text[start : start + _MAX_SNIPPET]
    prefix = _ELLIPSIS if start > 0 else ""
    suffix = _ELLIPSIS if start + len(body) < len(text) else ""
    # The leading ellipsis is part of the string the client renders, so it has
    # to be accounted for or every span would land a few characters early.
    return prefix + body + suffix, len(prefix) - start


def _clamp_span(span: tuple[int, int], shift: int, limit: int) -> dict | None:
    """Translate a chunk-relative span into a snippet-relative one, or drop it."""
    start = max(0, span[0] + shift)
    end = min(limit, span[1] + shift)
    if end <= start:
        return None
    return {"start": start, "end": end}


def _build_citations(
    query: str,
    chunks: list[dict],
    answer: str,
    support: list[dict],
) -> list[dict]:
    """Map each retrieved chunk back to its source metadata, in rank order.

    ``support`` is the selection the answer text was built from, keyed by the
    same 1-based marker used here, so ``supporting_span`` on citation ``[n]``
    is literally the sentence quoted after marker ``[n]`` in the answer.
    """
    used_markers = llm.cited_markers(answer)
    spans_by_marker = {item["marker"]: (item["start"], item["end"]) for item in support}

    citations: list[dict] = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        span = spans_by_marker.get(i)
        snippet, shift = _snippet_window(text, span)

        supporting = None
        if span is not None:
            bounds = _clamp_span(span, shift, len(snippet))
            if bounds is not None:
                supporting = {
                    **bounds,
                    "text": snippet[bounds["start"] : bounds["end"]],
                }

        citations.append(
            {
                "marker": f"[{i}]",
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "filename": chunk.get("filename"),
                "page": chunk.get("page"),
                # None when the retriever measured no score (the graph
                # strategy's chunks), never a 0.0 standing in for it.
                "score": None
                if chunk.get("score", 0.0) is None
                else round(float(chunk.get("score", 0.0)), 4),
                "snippet": snippet,
                # True when the answer text carries this marker, so the UI can
                # separate "the answer is built on this" from "also retrieved".
                "used": i in used_markers,
                # Offsets are relative to ``snippet``, which is what the UI shows.
                "highlights": _find_term_spans(query, snippet),
                # The exact sentence the answer lifted from this chunk, also
                # relative to ``snippet``. None when this chunk contributed no
                # sentence (it was retrieved but not quoted).
                "supporting_span": supporting,
            }
        )
    return citations


def build_answer(query: str, retrieved: list[dict], answer_text: str | None = None) -> dict:
    """Assemble the final answer payload returned to the UI.

    Args:
        query: the user's question.
        retrieved: ranked chunks from a retriever (each with ``text``, ``score``
            and citation metadata).
        answer_text: when given, used verbatim as the answer instead of running
            the offline extractive generator, so a caller that already produced
            an answer (an LLM completion, or an extractive fallback higher up
            the stack) doesn't have that text discarded and regenerated here.
            Everything downstream, confidence, citations, highlights and the
            source document, is derived exactly as before: all of it comes from
            the answer string and the retrieved rows, not from how that string
            was produced.

    Returns:
        ``{question, answer, confidence, citations, highlights, source_document,
        retrieved}``.
    """
    if not retrieved:
        return {
            "question": query,
            "answer": answer_text if answer_text is not None else llm.extractive_answer(query, []),
            "confidence": 0.0,
            "citations": [],
            "highlights": [],
            "source_document": None,
            "retrieved": [],
        }

    # Select the supporting sentences ONCE and reuse them for both the answer
    # text and the citation spans. Recomputing them separately would be two
    # sources of truth for the same fact, and any future change to the scoring
    # would silently desynchronise the quote from the span it is meant to mark.
    support = llm.select_support(query, retrieved)
    if answer_text is None:
        context = _assemble_context(retrieved)
        answer_text = llm.generate(query, context, chunks=retrieved, support=support)

    top = retrieved[0]
    source_document = {
        "document_id": top.get("document_id"),
        "filename": top.get("filename"),
        "page": top.get("page"),
        "collection_id": top.get("collection_id"),
    }

    return {
        "question": query,
        "answer": answer_text,
        "confidence": _confidence(query, retrieved),
        "citations": _build_citations(query, retrieved, answer_text, support),
        # Spans into the ANSWER string, which the client already has, rather
        # than into the top chunk's full text, which it never receives.
        "highlights": _find_term_spans(query, answer_text),
        "source_document": source_document,
        "retrieved": retrieved,
    }
