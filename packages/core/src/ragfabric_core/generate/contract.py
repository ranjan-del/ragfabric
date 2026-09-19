"""The citation contract: what a cited answer must satisfy, mechanically.

Phase 2's extractive generator guaranteed that every clause was a literal
substring of its chunk. Fluent generation cannot satisfy that and a check that
forbids generation is not a check, it is a ban. These three rules are what can
be verified without a semantic judgement:

1. marker validity  every [n] refers to a chunk that was actually retrieved
2. quote fidelity   anything in double quotes really appears in a cited chunk
3. grounding        an answer with evidence available carries a citation

Faithfulness of a paraphrase is deliberately NOT checked here. It is a semantic
property, it cannot be asserted, and Phase 8 measures it. Asserting it would be
a fabricated guarantee (ADR 0004).

Known looseness #1, recorded rather than hidden: rule 3 is skipped for any
answer that merely contains the ``NO_EVIDENCE`` substring, wherever it
appears. A model that writes "I could not find a date, but the policy is X"
with no marker would pass, because the substring check cannot tell an
admission from an admission plus an uncited claim. ``NO_EVIDENCE`` is
imported by ``generate.cited`` rather than redefined there, so the sentinel
the contract checks for and the sentence the generator emits can never drift
apart.

Known looseness #2, recorded rather than hidden: quote fidelity (rule 2) does
not verify every quoted span, only the ones ``_should_verify`` selects. A
short quote (under 8 characters) that contains no digit is left unchecked. A
digit-bearing quote is always checked regardless of length, because a short
quote carrying a number is a factual claim a reader will rely on, exactly the
kind of thing a citation exists to protect and exactly what a model is most
likely to get wrong. A short quote with no digit is usually incidental
phrasing (a single word quoted for emphasis, or an inflection of a word the
chunk contains), and verifying it strictly would produce retries and
extractive fallbacks that buy no safety: nothing dangerous hides inside a
non-numeric fragment that short. So the gap that remains is narrow and named
here: a fabricated non-numeric short quote, such as "the CEO" not actually in
the cited chunk, would still pass.
"""

from __future__ import annotations

import re

from ragfabric_core.strategies.base import RetrievedChunk

_MARKER_RE = re.compile(r"\[(\d+)\]")
_QUOTE_RE = re.compile(r"\"([^\"]+)\"")

# The sentinel substring that marks an explicit no-evidence answer. Defined
# here, once, and imported by generate.cited so the sentence the generator
# emits and the substring the contract looks for cannot drift apart.
NO_EVIDENCE = "could not find"


class CitationViolation(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def cited_markers(answer: str) -> list[int]:
    """Marker numbers in first appearance order, without duplicates."""
    seen: list[int] = []
    for raw in _MARKER_RE.findall(answer):
        n = int(raw)
        if n not in seen:
            seen.append(n)
    return seen


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _should_verify(quote: str) -> bool:
    """Decide whether a quoted span is worth checking against the chunks.

    A quote of 8 or more characters is always checked. Below that floor, a
    quote is checked only if it contains a digit: a short quote carrying a
    number ("99 days") is a factual claim, exactly what a citation exists to
    protect, while a short quote with no digit is usually incidental phrasing
    that a strict check would flag with no safety benefit.
    """
    return len(quote) >= 8 or any(ch.isdigit() for ch in quote)


def assert_citation_contract(answer: str, chunks: list[RetrievedChunk]) -> None:
    markers = cited_markers(answer)

    for n in markers:
        if n < 1 or n > len(chunks):
            raise CitationViolation(
                f"marker [{n}] does not point at any of the {len(chunks)} retrieved chunks"
            )

    if chunks and not markers and NO_EVIDENCE not in answer.lower():
        raise CitationViolation(
            "uncited answer: evidence was supplied but the answer carries no [n] marker"
        )

    cited_text = " ".join(_flat(chunks[n - 1].text) for n in markers)
    for quote in _QUOTE_RE.findall(answer):
        if not _should_verify(quote):
            continue
        if _flat(quote) not in cited_text:
            raise CitationViolation(
                f"quote {quote[:60]!r} does not appear in any chunk the answer cites"
            )
