"""Cheap, deterministic features of a question, and the strategy they point to.

No I/O and no model. Everything here is a rule that either fired or did not,
which is why a proposal says ``decisive`` and never carries a confidence
number (ADR 0004).
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from ragfabric_core.router.decision import QueryType
from ragfabric_core.stores.boosting import identifiers, is_identifier, phrases
from ragfabric_core.strategies.base import StrategyName

PLAIN_MAX_WORDS = 15

# Relation types a question can name in plain words. MENTIONS and RELATED_TO
# are left out on purpose: "related to" appears in questions that are not
# about the graph at all, and a signal that fires on everything is noise.
DEFAULT_RELATION_PHRASES: dict[str, tuple[str, ...]] = {
    "REPORTS_TO": ("report to", "reports to", "reporting to", "manager of", "managed by"),
    "MEMBER_OF": ("member of", "members of", "part of the team", "on the team"),
    "BELONGS_TO": ("belong to", "belongs to"),
    "OWNS": ("own", "owns", "owner of", "owned by"),
    "WORKS_ON": ("work on", "works on", "working on"),
    "LOCATED_IN": ("located in", "based in", "based out of"),
    "AUTHORED": ("wrote", "author of", "authored"),
}
_COMPARISON = (
    "compare",
    "comparison",
    "difference between",
    "differences between",
    "versus",
    " vs ",
    "differ from",
)
_AGGREGATION = ("how many", "total", "sum of", "count of", "list all", "list every")
_NOT_ENTITIES = frozenset(
    "who what which when where why how does do did is are was were can could should "
    "would will the a an list compare show tell give find i we our my".split()
)
_PUNCTUATION = re.compile(r"[^a-z0-9']+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

TOOL_FOR_STRATEGY: dict[StrategyName, str] = {
    StrategyName.TRADITIONAL: "semantic_search",
    StrategyName.VECTORLESS: "lexical_search",
    StrategyName.GRAPH: "graph_search",
}
_DEFAULT_ORDER = (
    StrategyName.TRADITIONAL,
    StrategyName.VECTORLESS,
    StrategyName.GRAPH,
    StrategyName.AGENTIC,
)


@dataclass(frozen=True)
class Signals:
    identifiers: tuple[str, ...]
    phrases: tuple[str, ...]
    entities: tuple[str, ...]
    relations: tuple[str, ...]
    comparison: bool
    aggregation: bool
    compound: bool
    words: int


@dataclass(frozen=True)
class Proposal:
    strategy: StrategyName
    decisive: bool
    reasons: tuple[str, ...]
    query_type: QueryType
    ranking: tuple[StrategyName, ...]


def entity_mentions(question: str) -> tuple[str, ...]:
    """Runs of capitalised words that are not question words or identifiers."""
    mentions: list[str] = []
    run: list[str] = []
    for word in _WORD.findall(question):
        if word[0].isupper() and word.lower() not in _NOT_ENTITIES and not is_identifier(word):
            run.append(word)
            continue
        if run:
            mentions.append(" ".join(run))
            run = []
    if run:
        mentions.append(" ".join(run))
    return tuple(mentions)


def extract_signals(question: str, *, relation_types: Collection[str]) -> Signals:
    # Punctuation becomes a space so a phrase at the end of a sentence still
    # matches: "report to?" must hit " report to ".
    text = f" {' '.join(_PUNCTUATION.sub(' ', question.lower()).split())} "
    configured = {name.upper() for name in relation_types}
    relations = tuple(
        sorted(
            name
            for name, spoken in DEFAULT_RELATION_PHRASES.items()
            if name in configured and any(f" {phrase} " in text for phrase in spoken)
        )
    )
    return Signals(
        identifiers=tuple(sorted(identifiers(question))),
        phrases=tuple(phrases(question)),
        entities=entity_mentions(question),
        relations=relations,
        comparison=any(marker in text for marker in _COMPARISON),
        aggregation=any(marker in text for marker in _AGGREGATION),
        compound=question.count("?") > 1,
        words=len(question.split()),
    )


def _fired(signals: Signals) -> list[tuple[StrategyName, QueryType, str]]:
    fired: list[tuple[StrategyName, QueryType, str]] = []
    if signals.identifiers or signals.phrases:
        fired.append(
            (
                StrategyName.VECTORLESS,
                "exact_match",
                "The question names an exact identifier or quoted phrase.",
            )
        )
    if signals.relations and signals.entities:
        fired.append(
            (
                StrategyName.GRAPH,
                "relationship",
                "The question asks how named things are related.",
            )
        )
    if signals.comparison:
        fired.append((StrategyName.AGENTIC, "comparison", "The question compares several things."))
    elif signals.aggregation:
        fired.append(
            (
                StrategyName.AGENTIC,
                "aggregation",
                "The question asks for a count or a total across sources.",
            )
        )
    elif signals.compound:
        fired.append(
            (StrategyName.AGENTIC, "compound", "The question asks several things at once.")
        )
    return fired


def propose(signals: Signals, *, available: Collection[StrategyName]) -> Proposal:
    fired = _fired(signals)
    reasons: list[str] = []
    usable = [item for item in fired if item[0] in available]
    for skipped, _, _ in fired:
        if skipped not in available:
            reasons.append(
                f"The {skipped.value} strategy is not available for this request, "
                "so it was not considered."
            )

    if len(usable) == 1:
        strategy, query_type, reason = usable[0]
        decisive = True
        reasons.insert(0, reason)
    elif not usable and (fired or signals.words <= PLAIN_MAX_WORDS):
        strategy, query_type, decisive = StrategyName.TRADITIONAL, "simple_factual", True
        reasons.insert(0, "A short question about one concept.")
    else:
        # Conflicting signals or a long question: the classifier decides. The
        # provisional choice is the most capable strategy that fired, for the
        # case where the classifier cannot be reached.
        strategy = next(
            (s for s, _, _ in usable if s is StrategyName.AGENTIC),
            usable[0][0] if usable else StrategyName.TRADITIONAL,
        )
        query_type, decisive = "ambiguous", False
        reasons.insert(0, "The signals disagree or the question is long and open.")

    ranking = [strategy] + [s for s, _, _ in usable if s is not strategy]
    ranking += [s for s in _DEFAULT_ORDER if s in available and s not in ranking]
    return Proposal(
        strategy=strategy,
        decisive=decisive,
        reasons=tuple(reasons),
        query_type=query_type,
        ranking=tuple(ranking),
    )
