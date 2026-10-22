"""Which passage a number, code or quote is cited to, and moving it when that is wrong.

The citation contract checks a quote against all cited passages joined together and a number not at
all, so a fact cited to the wrong passage passed (eval answers, 22 October 2026: "claims submitted
later than 30 days [4]" with the rule in passage 3; "the product's code is BWA-410 [2]" with the code
in passage 4). Paraphrase is still not checked here: it is a semantic property (see ``contract``).

What is checked is narrow and mechanical. A claim is the text a marker group owns. Its *phrases* are
the quotes the contract already verifies, product or error codes (``BWA-410``), and a number with the
word after it (``30 days``). A phrase counts only when no cited passage holds it and some other
retrieved passage does; a number nothing holds, such as a computed ``54,000``, is left alone, and so
is a year (context, not a fact to look up). When the holding passage is clear, the marker is moved to
it deterministically. When it is not (two passages hold the phrase equally well) nothing is guessed:
``assert_number_attribution`` raises, and the existing repair retry shows the model where the phrase is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragfabric_core.generate.contract import (
    _QUOTE_RE,
    CitationViolation,
    _flat,
    _should_verify,
)
from ragfabric_core.strategies.base import RetrievedChunk

# [1], [1], [2] and [1] and [3]: one run of markers that cite one claim.
_GROUP = re.compile(r"\[\d+\](?:\s*(?:,|and|&)?\s*\[\d+\])*")
_NUMBER = re.compile(r"(?<![\w.,-])\d[\d,]*(?:\.\d+)?")
_CODE = re.compile(r"\b[A-Z]{2,}-\d+\b")
_NEXT_WORD = re.compile(r"\s+([A-Za-z]+)")
_YEAR = re.compile(r"(19|20)\d\d")
_WORD = re.compile(r"[a-z0-9]{4,}")
_REFERENCE = ("passage", "passages", "source", "sources", "according to", "per", "see", "from")
# A cited passage that shares this share of a claim's longer words keeps its marker.
_KEEP_OVERLAP = 0.34


@dataclass(frozen=True)
class Misattribution:
    phrase: str
    cited: tuple[int, ...]
    holders: tuple[int, ...]
    segment: str


@dataclass
class Repaired:
    text: str
    repairs: list[str] = field(default_factory=list)
    unresolved: list[Misattribution] = field(default_factory=list)


@dataclass
class _Group:
    start: int
    end: int
    markers: tuple[int, ...]
    segment: str


def _normal(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", _flat(text))


def _holds(normal_text: str, phrase: str) -> bool:
    if phrase.startswith('"'):
        return phrase[1:] in normal_text
    return re.search(rf"(?<![\d.,]){re.escape(phrase)}(?![\d])", normal_text) is not None


def _phrases(segment: str) -> list[str]:
    """Quotes, codes and number-with-unit phrases in a claim, normalised for matching."""
    found = ['"' + _normal(q) for q in _QUOTE_RE.findall(segment) if _should_verify(q)]
    outside = _QUOTE_RE.sub(" ", segment)
    for match in _CODE.finditer(outside):
        found.append(match.group(0).lower())
    for match in _NUMBER.finditer(outside):
        number = match.group(0).rstrip(",").replace(",", "")
        if _YEAR.fullmatch(number):
            continue
        word = _NEXT_WORD.match(outside, match.end())
        found.append(f"{number} {word.group(1).lower()}" if word else number)
    return list(dict.fromkeys(found))


def _words(text: str) -> set[str]:
    return set(_WORD.findall(_normal(text)))


def _groups(sentence: str, count: int) -> list[_Group]:
    groups: list[_Group] = []
    for m in _GROUP.finditer(sentence):
        markers = tuple(int(n) for n in re.findall(r"\d+", m.group(0)))
        if not all(1 <= n <= count for n in markers):
            continue
        groups.append(_Group(m.start(), m.end(), markers, ""))
    if not groups:
        return []
    bare = _GROUP.sub(" ", sentence)
    if sentence[: groups[0].start].strip() == "":
        union = tuple(dict.fromkeys(n for g in groups for n in g.markers))
        return [_Group(groups[0].start, groups[-1].end, union, bare)]
    for i, g in enumerate(groups):
        prev_end = groups[i - 1].end if i else 0
        next_start = groups[i + 1].start if i + 1 < len(groups) else len(sentence)
        before = sentence[prev_end : g.start]
        if before.strip().lower().endswith(_REFERENCE):
            g.segment = sentence[g.end : next_start]
        else:
            g.segment = before
    return groups


def _flagged(group: _Group, normals: list[str]) -> list[tuple[str, tuple[int, ...]]]:
    """Phrases of this claim that no cited passage holds but some other passage does."""
    cited_text = " ".join(normals[n - 1] for n in group.markers)
    out = []
    for phrase in _phrases(group.segment):
        if _holds(cited_text, phrase):
            continue
        holders = tuple(
            n
            for n, text in enumerate(normals, start=1)
            if n not in group.markers and _holds(text, phrase)
        )
        if holders:
            out.append((phrase, holders))
    return out


def _sentence_groups(answer: str, chunks: list[RetrievedChunk]):
    from ragfabric_core.generate.cited import split_claims

    for sentence in split_claims(answer):
        yield sentence, _groups(sentence, len(chunks))


def _misattribution(phrase: str, holders: tuple[int, ...], group: _Group) -> Misattribution:
    return Misattribution(phrase.lstrip('"'), group.markers, holders, group.segment.strip())


def find_misattributions(answer: str, chunks: list[RetrievedChunk]) -> list[Misattribution]:
    normals = [_normal(c.text) for c in chunks]
    return [
        _misattribution(phrase, holders, group)
        for _, groups in _sentence_groups(answer, chunks)
        for group in groups
        for phrase, holders in _flagged(group, normals)
    ]


def _pick_holder(m: Misattribution, chunks: list[RetrievedChunk]) -> int | None:
    """The one passage that holds the phrase, or the best match to the claim; None on a tie."""
    if len(m.holders) == 1:
        return m.holders[0]
    wanted = _words(m.segment)
    scores = sorted(
        ((len(wanted & _words(chunks[n - 1].text)), n) for n in m.holders), reverse=True
    )
    return scores[0][1] if scores[0][0] > scores[1][0] else None


def _markers(numbers) -> str:
    return ", ".join(f"[{n}]" for n in numbers)


def repair_attribution(answer: str, chunks: list[RetrievedChunk]) -> Repaired:
    """Move markers whose passage lacks the claim's number, code or quote to the one that holds it.

    The cited passage keeps its marker when it holds another of the claim's phrases or shares at
    least a third of the claim's longer words (it supports the rest of the sentence); otherwise
    the holder replaces it. Nothing is moved when the holder is ambiguous.
    """
    result = Repaired(text=answer)
    normals = [_normal(c.text) for c in chunks]
    pieces: list[str] = []
    for sentence, groups in _sentence_groups(answer, chunks):
        edits: list[tuple[int, int, str]] = []
        for group in groups:
            moved: dict[str, int] = {}
            for phrase, holders in _flagged(group, normals):
                m = _misattribution(phrase, holders, group)
                pick = _pick_holder(m, chunks)
                if pick is None:
                    result.unresolved.append(m)
                else:
                    moved[m.phrase] = pick
            if not moved:
                continue
            phrases = _phrases(group.segment)
            segment_words = _words(group.segment)
            keep = [
                n
                for n in group.markers
                if any(_holds(normals[n - 1], p) for p in phrases)
                or not segment_words
                or len(segment_words & _words(chunks[n - 1].text)) / len(segment_words)
                >= _KEEP_OVERLAP
            ]
            final = sorted(set(keep) | set(moved.values()))
            edits.append((group.start, group.end, _markers(final)))
            result.repairs.extend(
                f"{phrase}: {_markers(group.markers)} to {_markers(final)}" for phrase in moved
            )
        for start, end, replacement in sorted(edits, reverse=True):
            sentence = sentence[:start] + replacement + sentence[end:]
        pieces.append(sentence)
    if result.repairs:
        result.text = "".join(pieces)
    return result


def assert_number_attribution(answer: str, chunks: list[RetrievedChunk]) -> None:
    """Raise when a number, code or quote is cited to a passage that lacks it but another has it."""
    found = find_misattributions(answer, chunks)
    if not found:
        return
    m = found[0]
    where = " or ".join(f"passage [{n}]" for n in m.holders)
    raise CitationViolation(
        f"{m.phrase!r} does not appear in {_markers(m.cited)}, the passage cited for it; "
        f"it appears in {where}"
    )
