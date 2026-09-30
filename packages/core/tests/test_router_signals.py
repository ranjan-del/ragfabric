import pytest

from ragfabric_core.router.signals import extract_signals, propose
from ragfabric_core.strategies.base import StrategyName as S

RELATIONS = [
    "REPORTS_TO",
    "MEMBER_OF",
    "BELONGS_TO",
    "OWNS",
    "WORKS_ON",
    "LOCATED_IN",
    "AUTHORED",
    "MENTIONS",
    "RELATED_TO",
]
ALL = {S.TRADITIONAL, S.VECTORLESS, S.AGENTIC, S.GRAPH}


def run(question, available=ALL):
    return propose(extract_signals(question, relation_types=RELATIONS), available=available)


@pytest.mark.parametrize(
    ("question", "strategy", "query_type"),
    [
        ("What does ERR_QUOTA_4419 mean?", S.VECTORLESS, "exact_match"),
        ('Where is "annual leave carry forward" defined?', S.VECTORLESS, "exact_match"),
        ("ERR_QUOTA_4419", S.VECTORLESS, "exact_match"),
        ("Who does Ravi Sharma report to?", S.GRAPH, "relationship"),
        ("Which team owns Billing?", S.GRAPH, "relationship"),
        ("Compare the leave policy for Pune and Delhi", S.AGENTIC, "comparison"),
        ("How many offices does the company have?", S.AGENTIC, "aggregation"),
        ("What is our refund policy?", S.TRADITIONAL, "simple_factual"),
    ],
)
def test_a_single_signal_is_decisive(question, strategy, query_type):
    proposal = run(question)
    assert (proposal.strategy, proposal.decisive, proposal.query_type) == (
        strategy,
        True,
        query_type,
    )


def test_conflicting_signals_are_not_decisive():
    proposal = run("Compare ERR_QUOTA_4419 with ERR_QUOTA_4420 for Billing")
    assert not proposal.decisive


def test_a_long_question_with_no_signal_is_not_decisive():
    question = " ".join(["policy"] * 20) + "?"
    assert not run(question).decisive


def test_a_relation_phrase_without_an_entity_is_not_a_graph_signal():
    assert run("who do people report to?").strategy is not S.GRAPH


def test_an_unconfigured_relation_type_is_not_a_graph_signal():
    signals = extract_signals("Who does Ravi Sharma report to?", relation_types=["OWNS"])
    assert propose(signals, available=ALL).strategy is S.TRADITIONAL


def test_graph_unavailable_resolves_to_traditional_and_says_why():
    proposal = run("Who does Ravi Sharma report to?", available=ALL - {S.GRAPH})
    assert proposal.strategy is S.TRADITIONAL
    assert any("graph" in reason for reason in proposal.reasons)
    assert S.GRAPH not in proposal.ranking


def test_the_ranking_puts_the_proposal_first_and_lists_every_available_strategy_once():
    proposal = run("What does ERR_QUOTA_4419 mean?")
    assert proposal.ranking[0] is S.VECTORLESS
    assert sorted(proposal.ranking) == sorted(ALL)


@pytest.mark.parametrize("question", ["", "   ", "?"])
def test_an_empty_question_proposes_traditional_without_raising(question):
    assert run(question).strategy is S.TRADITIONAL


def test_extraction_is_pure():
    first = extract_signals("Who owns Billing?", relation_types=RELATIONS)
    second = extract_signals("Who owns Billing?", relation_types=RELATIONS)
    assert first == second


def test_a_plain_question_falls_back_to_an_available_strategy():
    proposal = run("What is our refund policy?", available={S.VECTORLESS, S.AGENTIC})
    assert proposal.strategy is S.VECTORLESS
    assert S.TRADITIONAL not in proposal.ranking


@pytest.mark.parametrize(
    "question",
    [
        "What is our refund policy?",
        "Who does Ravi Sharma report to?",
        "What does ERR_QUOTA_4419 mean?",
        "Compare ERR_QUOTA_4419 with ERR_QUOTA_4420 for Billing",
        " ".join(["policy"] * 20) + "?",
        "",
    ],
)
@pytest.mark.parametrize(
    "available",
    [
        {S.VECTORLESS},
        {S.AGENTIC},
        {S.GRAPH, S.AGENTIC},
        {S.VECTORLESS, S.AGENTIC},
        {S.TRADITIONAL, S.GRAPH},
    ],
)
def test_the_proposal_and_ranking_stay_inside_the_available_set(question, available):
    proposal = run(question, available=available)
    assert proposal.strategy in available
    assert set(proposal.ranking) <= available
    assert proposal.ranking[0] is proposal.strategy


def test_no_available_strategy_raises():
    with pytest.raises(ValueError, match="at least one available strategy"):
        run("What is our refund policy?", available=set())


def test_the_bare_word_own_is_not_a_graph_signal():
    assert run("What is our own leave policy for Pune?").strategy is not S.GRAPH


def test_from_rule_is_true_for_exactly_one_fired_signal():
    assert run("What does ERR_QUOTA_4419 mean?").from_rule is True


def test_from_rule_is_false_for_the_plain_short_default():
    proposal = run("What is the retry limit?")
    assert proposal.decisive is True and proposal.from_rule is False


def test_from_rule_is_false_for_a_fired_but_unavailable_fallback():
    proposal = run("Who does Ravi Sharma report to?", available=ALL - {S.GRAPH})
    assert proposal.from_rule is False


def test_from_rule_is_false_when_not_decisive():
    long_text = " ".join(["onboarding"] * 20) + "?"
    proposal = run(long_text)
    assert proposal.decisive is False and proposal.from_rule is False


@pytest.mark.parametrize(
    "question",
    [
        "Is the VPN totally free for contractors?",
        "What is the subtotal line on an invoice?",
        "Where do I find the totals report?",
    ],
)
def test_a_marker_inside_a_longer_word_is_not_agentic(question):
    signals = extract_signals(question, relation_types=RELATIONS)
    assert (signals.comparison, signals.aggregation) == (False, False)
    assert run(question).strategy is not S.AGENTIC


@pytest.mark.parametrize(
    ("question", "query_type"),
    [
        ("Pune vs Delhi leave policy", "comparison"),
        ("How is the Pune policy compared with Delhi?", "comparison"),
        ("What is the difference between sick leave and casual leave?", "comparison"),
        ("How many offices does the company have?", "aggregation"),
        ("What is the total headcount?", "aggregation"),
    ],
)
def test_a_whole_word_marker_is_agentic(question, query_type):
    proposal = run(question)
    assert (proposal.strategy, proposal.decisive, proposal.query_type) == (
        S.AGENTIC,
        True,
        query_type,
    )


def test_an_unavailable_fired_signal_is_named_first_and_keeps_its_query_type():
    proposal = run("Who does Ravi Sharma report to?", available=ALL - {S.GRAPH})
    assert proposal.reasons[0] == (
        "The question asks how named things are related, but the graph is not "
        "available here, so meaning search was used."
    )
    assert proposal.query_type == "relationship"
    assert (proposal.strategy, proposal.decisive, proposal.from_rule) == (
        S.TRADITIONAL,
        True,
        False,
    )
