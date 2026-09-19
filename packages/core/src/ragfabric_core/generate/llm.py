"""Answer generation (offline-first).

Design goal for this repo: the pipeline must RUN OFFLINE with no API key. So the
DEFAULT backend is a deterministic *extractive* generator. It does not invent
prose; it selects the sentences from the retrieved chunks that best match the
question and stitches them together with ``[n]`` citation markers. That choice is
what makes the citation contract checkable: every clause in the answer is a
literal substring of the chunk its marker points at, and a test asserts exactly
that (see ``test_answer_text_is_lifted_from_the_cited_chunk``).

This extractive generator is also the citation contract's satisfying fallback:
it is what ``generate.cited.generate_cited_answer`` reaches for after an LLM
answer fails the contract twice, because its output is grounded and cited by
construction and needs no model call to verify. It is also what the
``generation: extractive`` config mode selects directly. Generation through a
configured LLM provider goes through ``generate.cited`` instead of this
module; there is no vendor-specific branch here, so there is exactly one route
to a model call and it is not hardcoded to any one vendor.
"""

from __future__ import annotations

import re

from ragfabric_core.ingest.embed import content_tokens

# Sentence boundary: end punctuation followed by whitespace, OR a newline. The
# newline arm matters because CSV and DOCX text arrives as unpunctuated lines,
# which would otherwise collapse into one enormous "sentence".
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Longest sentence we will quote verbatim. A chunk with no punctuation at all is
# one 800-character sentence; quoting all of it makes an unreadable answer.
_MAX_SENTENCE = 320

# Citation markers such as "[2]" inside an answer, used to work out which sources
# the answer actually leaned on.
_MARKER_RE = re.compile(r"\[(\d+)\]")

# How many of the top-ranked chunks may contribute a quoted sentence. Shared by
# the answer text and the citation layer so both describe the same selection.
MAX_SUPPORT_CHUNKS = 3

# Relevance floors. Without these, "top 3 chunks" was taken literally: a chunk
# that shared not one word with the question was still quoted and still marked
# used=true, because it happened to be third in a list of three. A retriever
# always returns its best candidates, even when its best is nothing at all, so
# rank on its own is not evidence of relevance.
#
# Two independent floors, because they catch different failures:
#
#   MIN_CHUNK_SCORE   - the retriever itself scored this chunk at zero, meaning
#                       it shares no vocabulary with the query. Quoting it is
#                       quoting a random passage.
#   MIN_TERM_OVERLAP  - the chunk scored, but the specific SENTENCE picked out
#                       of it covers none of the question's content words. The
#                       chunk may be about the right topic while that sentence
#                       is not, and it is the sentence that gets quoted.
#
# The cost of the floors is that a genuinely answerable question phrased in
# synonyms now returns "I don't have enough information" instead of a quote
# that happened to be right. That is the better failure: a citation the user
# cannot trust is worse than an admission, because it looks identical to one
# that is correct.
MIN_CHUNK_SCORE = 1e-9
MIN_TERM_OVERLAP = 1


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into sentence ``(start, end)`` offsets.

    Offsets are returned rather than strings so a caller can map a quoted
    sentence back to its exact position inside the source chunk, which is what
    the highlight layer needs.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        piece = text[cursor : match.start()]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            spans.append((cursor + lead, cursor + len(piece.rstrip())))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        spans.append((cursor + lead, cursor + len(tail.rstrip())))
    return spans


def _truncate_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shorten an over-long sentence span at a word boundary.

    A prefix of a sentence is still an exact substring of the chunk, so the
    "answer text is lifted from the cited chunk" contract survives truncation.
    """
    if end - start <= _MAX_SENTENCE:
        return start, end
    hard_end = start + _MAX_SENTENCE
    cut = text[start:hard_end].rfind(" ")
    return start, (start + cut if cut > _MAX_SENTENCE // 2 else hard_end)


def select_support(
    question: str, chunks: list[dict], max_chunks: int = MAX_SUPPORT_CHUNKS
) -> list[dict]:
    """Pick the best-matching sentence from each of the top ``max_chunks``.

    Sentences are scored by how many of the question's content words they
    contain, with a small length penalty so a short precise sentence beats a long
    rambling one that happens to cover the same words.

    Returns one entry per contributing chunk::

        {"marker": int, "text": str, "start": int, "end": int, "score": float}

    where ``start``/``end`` index into that chunk's own ``text``.
    """
    query_tokens = set(content_tokens(question))
    support: list[dict] = []

    for rank, chunk in enumerate(chunks[:max_chunks], start=1):
        text = chunk.get("text", "")
        if not text.strip():
            continue

        # Floor 1: the retriever found no vocabulary in common. Being third in
        # a list of three is a position, not a reason to quote something.
        #
        # A MISSING score is not the same as a zero one. Callers that select
        # support from chunks they assembled themselves never set the field,
        # and treating that absence as "scored zero" would reject everything.
        # Only an explicit zero is evidence, and floor 2 covers the rest.
        score = chunk.get("score")
        if score is not None and float(score) <= MIN_CHUNK_SCORE:
            continue

        best: tuple[float, int, int, int] | None = None
        for start, end in sentence_spans(text):
            sentence = text[start:end]
            tokens = content_tokens(sentence)
            if not tokens:
                continue
            hits = len(query_tokens & set(tokens))
            score = hits / (len(query_tokens) or 1) - 0.0005 * len(sentence)
            if best is None or score > best[0]:
                best = (score, start, end, hits)

        # Floor 2: the best sentence in this chunk covers none of the
        # question's content words, so quoting it would attach a citation to a
        # sentence that does not address what was asked.
        if best is None or best[3] < MIN_TERM_OVERLAP:
            continue
        _, start, end, _ = best
        start, end = _truncate_span(text, start, end)
        support.append(
            {
                "marker": rank,
                "text": text[start:end],
                "start": start,
                "end": end,
                "score": round(float(best[0]), 4),
            }
        )
    return support


def extractive_answer(
    question: str,
    chunks: list[dict],
    max_chunks: int = MAX_SUPPORT_CHUNKS,
    support: list[dict] | None = None,
) -> str:
    """Deterministic offline answer: the best sentence from each top chunk.

    This is NOT a generated answer. It quotes the retrieved text verbatim and
    tags each quote with the marker of the chunk it came from, so the answer is
    always grounded, always reproducible, and needs no LLM.

    ``support`` may be passed in by a caller that has already run
    :func:`select_support`, so the answer text and the spans the citation layer
    highlights are guaranteed to come from one single selection rather than two
    independent ones that could drift apart.
    """
    if not chunks:
        return "I don't have enough information in the knowledge base to answer that."

    if support is None:
        support = select_support(question, chunks, max_chunks=max_chunks)
    if not support:
        return "I don't have enough information in the knowledge base to answer that."

    parts = [f"{' '.join(item['text'].split())} [{item['marker']}]" for item in support]
    return "Based on the most relevant sources: " + " ".join(parts)


def cited_markers(answer: str) -> set[int]:
    """Marker numbers the answer text actually references, e.g. ``{1, 3}``."""
    return {int(m.group(1)) for m in _MARKER_RE.finditer(answer)}


def generate(
    question: str,
    context: str,
    chunks: list[dict] | None = None,
    model: str | None = None,
    support: list[dict] | None = None,
) -> str:
    """Produce an answer for ``question`` grounded in ``context``.

    Always returns the deterministic extractive answer. A vendor-backed model
    call, when the caller wants one, goes through ``generate.cited`` and a
    configured ``LLMProvider`` instead; this module has no vendor branch of its
    own, so there is exactly one route to a model call in this codebase.
    ``context`` and ``model`` are accepted for signature compatibility with
    :class:`LLMClient`'s callers but do not affect the offline path.
    """
    del context, model
    return extractive_answer(question, chunks or [], support=support)


class LLMClient:
    """Thin object-oriented wrapper around :func:`generate`.

    Kept for callers that prefer an injectable client. The default path is offline
    and requires no configuration.
    """

    def __init__(self, model: str | None = None):
        self.model = model

    def complete(
        self,
        question: str,
        context: str,
        chunks: list[dict] | None = None,
        support: list[dict] | None = None,
    ) -> str:
        return generate(question, context, chunks=chunks, model=self.model, support=support)
