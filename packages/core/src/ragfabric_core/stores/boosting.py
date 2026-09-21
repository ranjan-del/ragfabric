"""Exact phrase and identifier boosts, the two cases BM25's IDF gets wrong.

IDF already favours rare tokens, which covers most of what a user means by "an
unusual word". Two cases it cannot cover:

A quoted phrase. BM25 is a bag of words. "retry limit" scores the same whether
the two words are adjacent or four paragraphs apart, because the model has no
notion of order. When a user puts quotation marks round something they are
stating that adjacency is the point.

An identifier. to_tsvector('english') stems and splits: ERR_QUOTA_4419 becomes
err, quota and 4419, get_user_by_id becomes get, user and id. The exact form is
gone by the time anything is ranked, so a chunk containing the real identifier
and a chunk containing those words scattered look identical to the index. The
raw text still has the answer, so that is where this looks.

**A boost multiplies a score. It never adds a chunk.** Retrieval decided
membership, and it did so with the access filter inside the store query (ADR
0003). A boost that could pull in a chunk on the strength of a text match
would be reaching past that filter and returning something the principal is
not allowed to read. Reordering cannot do that; introducing can.

**Identifier detection is a documented regex set, not a feeling.** A token is
an identifier if it matches any one of the four patterns below. Each exists
for a reason, and each has a negative case in the tests so the rule stays a
rule rather than drifting into "whatever made the last example pass".
"""

from __future__ import annotations

import re

from ragfabric_core.strategies.base import RetrievedChunk

# Tokens may carry dots, hyphens and underscores internally (v2.1.4,
# get_user_by_id) but must not start or end on one, so trailing sentence
# punctuation does not become part of the token.
_TOKEN = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_.\-]*[A-Za-z0-9_])?")

# 1. Underscore between word characters: snake_case names and error codes.
_UNDERSCORED = re.compile(r"^\w+_\w+$")

# 2. A digit and a letter in the same token: utf8, sha256, h2. A pure word has
#    no digit and a pure number no letter, so neither qualifies here.
_DIGIT_AND_LETTER = re.compile(r"^(?=[^0-9]*[0-9])(?=[^A-Za-z]*[A-Za-z])[A-Za-z0-9]+$")

# 3. CamelCase with an INTERNAL capital, meaning a lower case letter directly
#    followed by an upper case one. Requiring the transition is what keeps a
#    merely sentence-initial capital ("The") out.
_CAMEL = re.compile(r"^[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*$")

# 4. A dotted or hyphenated version: v2.1.4, 1.14.0-rc2. Digits separated by a
#    dot or a hyphen, optionally prefixed with v and suffixed with a tag.
_VERSION = re.compile(r"^v?\d+(?:[.\-]\d+)+(?:[.\-]?[A-Za-z0-9]+)*$")

_IDENTIFIER_PATTERNS = (_UNDERSCORED, _DIGIT_AND_LETTER, _CAMEL, _VERSION)

# Straight and typographic double quotes both count, because a query pasted
# out of a word processor should not silently lose its phrase.
_PHRASE = re.compile(r'"([^"]+)"|“([^”]+)”')


def is_identifier(token: str) -> bool:
    """True when a token is a name or code rather than an English word."""
    return any(pattern.match(token) for pattern in _IDENTIFIER_PATTERNS)


def identifiers(query: str) -> set[str]:
    """Every identifier-looking token in the query, in its original case.

    Case is preserved on purpose: the whole point is the exact form, and the
    match against chunk text is done case insensitively anyway.
    """
    return {token for token in _TOKEN.findall(query) if is_identifier(token)}


def phrases(query: str) -> list[str]:
    """The quoted phrases in the query, verbatim and in order.

    An unclosed quote yields nothing rather than swallowing the rest of the
    query as a phrase, which would make one stray character silently change
    what the whole query means.
    """
    return [straight or curly for straight, curly in _PHRASE.findall(query)]


def apply_boosts(
    hits: list[RetrievedChunk],
    query: str,
    phrase_boost: float,
    identifier_boost: float,
) -> list[RetrievedChunk]:
    """Reorder ``hits`` by multiplying the scores of exact matches.

    Returns exactly the chunks it was given, never more. A hit whose score is
    None is left at None: there is no number to multiply, and inventing one
    would be reporting a rank the ranker never produced (ADR 0004).
    """
    if not hits:
        return []

    wanted_phrases = [p.lower() for p in phrases(query)]
    wanted_identifiers = [i.lower() for i in identifiers(query)]

    boosted: list[RetrievedChunk] = []
    for hit in hits:
        if hit.score is None:
            boosted.append(hit)
            continue
        haystack = hit.text.lower()
        multiplier = 1.0
        if any(phrase in haystack for phrase in wanted_phrases):
            multiplier *= phrase_boost
        if any(identifier in haystack for identifier in wanted_identifiers):
            multiplier *= identifier_boost
        boosted.append(
            hit if multiplier == 1.0 else hit.model_copy(update={"score": hit.score * multiplier})
        )

    # Unscored hits sort last rather than first: an absent score is not a zero
    # and not a win. Ties break on chunk_id so the order is deterministic.
    boosted.sort(key=lambda h: (h.score is None, -(h.score or 0.0), h.chunk_id))
    return boosted
