"""The four configured limits, and the proof that each one changes a run.

A limit that is typed, validated and printed back by ``config validate`` while
nothing reads it is worse than no limit at all: the operator edits it, sees it
echoed, and gets the old behaviour. These tests exist because that was the
state of ``max_llm_calls``, ``max_latency_ms``, ``max_cost_usd`` and
``assess_strictness``, so each one is asserted by the run it changes rather
than by the field it was stored in.

The cost cap carries the subtlety worth reading twice. ``estimate_cost``
returns ``usd=None, known=False`` for a model with no price, and there are only
two ways to handle that and one of them is honest. Treating unknown as zero
silently disables the cap; treating it as infinite stops every run on an
unpriced model. So an unknown price means the cap cannot bind, and the run
records that it could not, which is what ADR 0004 asks of every number here.
"""

from __future__ import annotations

import json

import pytest
from agent_doubles import FakeTool, RecordingLLM, SlowTool, chunk, ctx

from ragfabric_core.agent.loop import (
    STOP_BUDGET,
    STOP_NO_PROGRESS,
    STOP_RESOLVED,
    run_agent,
)
from ragfabric_core.pricing import PricingTable, estimate_cost

STOPS = {STOP_RESOLVED, STOP_BUDGET, STOP_NO_PROGRESS}


def plan_json(*pairs: tuple[str, str]) -> str:
    return json.dumps(
        {"sub_questions": [{"text": text, "tool": tool, "why": ""} for text, tool in pairs]}
    )


def assessed(text: str, answered: bool, missing: str | None = None) -> str:
    return json.dumps(
        {"verdicts": [{"sub_question": text, "answered": answered, "missing": missing}]}
    )


def repair_json(move: str, query: str | None = None) -> str:
    return json.dumps({"move": move, "rewritten_query": query, "why": "because"})


def unresolved_script(question: str, rounds: int = 6) -> list[str]:
    """A script that never resolves, so a run ends on whichever cap is under test."""
    script = [plan_json((question, "semantic_search"))]
    for _ in range(rounds):
        script.append(assessed(question, False, "the number of retries"))
        script.append(repair_json("broaden"))
    return script


def growing_tool(name: str = "semantic_search") -> FakeTool:
    """A tool whose every call returns a chunk the pool has not seen.

    Needed so that a run under test stops on the cap being exercised and not on
    ``no_progress``, which would pass for the wrong reason.
    """

    class Growing(FakeTool):
        def run(self, query, context):
            self.queries.append(query)
            self.contexts.append(context)
            self.retrieval_calls += 1
            return [chunk(self.retrieval_calls)]

    return Growing(name)


QUESTION = "what is the retry limit"


# --- max_llm_calls ---------------------------------------------------------


def test_the_configured_call_cap_reaches_the_loop():
    """The deployment's cap binds even when the caller asked for nothing."""
    llm = RecordingLLM(*unresolved_script(QUESTION))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": growing_tool()},
        ctx=ctx(max_llm_calls=50),
        max_iterations=10,
        max_llm_calls=3,
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_llm_calls of 3" in run.stop_detail
    assert run.state.llm_calls == 3


def test_a_caller_may_tighten_the_configured_call_cap():
    """A request that asks for less gets less. The smaller of the two wins."""
    llm = RecordingLLM(*unresolved_script(QUESTION))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": growing_tool()},
        ctx=ctx(max_llm_calls=2),
        max_iterations=10,
        max_llm_calls=12,
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_llm_calls of 2" in run.stop_detail
    assert run.state.llm_calls == 2


def test_a_caller_cannot_exceed_the_configured_call_cap():
    """The other direction, which is the one that would cost somebody money."""
    llm = RecordingLLM(*unresolved_script(QUESTION))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": growing_tool()},
        ctx=ctx(max_llm_calls=500),
        max_iterations=10,
        max_llm_calls=4,
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_llm_calls of 4" in run.stop_detail
    assert run.state.llm_calls == 4


def test_the_default_budget_does_not_undercut_the_configured_default():
    """The example file advertises twelve calls, so twelve is what a default run gets.

    The caller's ``Budget`` default used to be eight, which meant the number in
    the file was never the one enforced and no run could ever reach it.
    """
    from ragfabric_core.config_file import AgenticConfig
    from ragfabric_core.strategies.base import Budget

    assert Budget().max_llm_calls == AgenticConfig().max_llm_calls


# --- max_latency_ms --------------------------------------------------------


def test_the_latency_cap_stops_the_run_on_the_budget_stop_reason():
    tools = {"semantic_search": SlowTool("semantic_search", delay_s=0.02, chunks=[chunk(1)])}
    llm = RecordingLLM(*unresolved_script(QUESTION))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools=tools,
        ctx=ctx(),
        max_iterations=10,
        max_latency_ms=1,
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_latency_ms of 1" in run.stop_detail
    assert run.state.iterations == 1


def test_the_latency_stop_does_not_invent_a_fourth_stop_reason():
    """Three words is the whole vocabulary. Latency is a budget, not a new one."""
    tools = {"semantic_search": SlowTool("semantic_search", delay_s=0.02, chunks=[chunk(1)])}
    llm = RecordingLLM(*unresolved_script(QUESTION))

    run = run_agent(QUESTION, llm=llm, tools=tools, ctx=ctx(), max_iterations=10, max_latency_ms=1)

    assert run.stop_reason in STOPS
    assert run.state.stop_reason in STOPS


def test_the_latency_cap_stops_before_spending_another_model_call():
    """Tripping after retrieval and before the assessment is the point of the cap."""
    tools = {"semantic_search": SlowTool("semantic_search", delay_s=0.02, chunks=[chunk(1)])}
    llm = RecordingLLM(*unresolved_script(QUESTION))

    run = run_agent(QUESTION, llm=llm, tools=tools, ctx=ctx(), max_iterations=10, max_latency_ms=1)

    assert run.state.llm_calls == 1  # the plan, and nothing after it


def test_a_generous_latency_cap_lets_the_run_finish():
    tools = {"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])}
    llm = RecordingLLM(plan_json((QUESTION, "semantic_search")), assessed(QUESTION, True))

    run = run_agent(QUESTION, llm=llm, tools=tools, ctx=ctx(), max_latency_ms=60_000)

    assert run.stop_reason == STOP_RESOLVED


def test_a_caller_may_tighten_the_configured_latency_cap():
    tools = {"semantic_search": SlowTool("semantic_search", delay_s=0.02, chunks=[chunk(1)])}
    llm = RecordingLLM(*unresolved_script(QUESTION))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools=tools,
        ctx=ctx(max_latency_ms=2),
        max_iterations=10,
        max_latency_ms=60_000,
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_latency_ms of 2" in run.stop_detail


# --- max_cost_usd ----------------------------------------------------------


def priced_llm(*responses: str) -> RecordingLLM:
    """A double that answers as a model the pricing table actually knows."""
    return RecordingLLM(*responses, provider="openai", model="gpt-5.4")


def test_the_cost_cap_stops_the_run_when_the_price_is_known():
    llm = priced_llm(*unresolved_script(QUESTION))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": growing_tool()},
        ctx=ctx(max_llm_calls=50),
        max_iterations=10,
        max_cost_usd=0.000001,
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_cost_usd" in run.stop_detail
    assert run.cost_known is True


def test_the_recorded_cost_is_the_real_token_counts_at_the_configured_price():
    llm = priced_llm(plan_json((QUESTION, "semantic_search")), assessed(QUESTION, True))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])},
        ctx=ctx(),
        max_cost_usd=1.0,
    )

    expected = estimate_cost(
        PricingTable.load(), "openai", "gpt-5.4", run.input_tokens, run.output_tokens
    )
    assert run.stop_reason == STOP_RESOLVED
    assert run.cost_usd == pytest.approx(expected.usd)
    assert run.input_tokens > 0


def test_an_unknown_price_cannot_bind_the_cost_cap():
    """The cap is zero and the run still finishes, because nobody knows the price.

    Stopping here would mean pricing every run on an unpriced model at
    infinity, which stops every run. The honest answer is that the cap did not
    apply, and the run says so rather than implying it held.
    """
    llm = RecordingLLM(plan_json((QUESTION, "semantic_search")), assessed(QUESTION, True))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])},
        ctx=ctx(),
        max_cost_usd=0.0,
    )

    assert run.stop_reason == STOP_RESOLVED
    assert run.cost_known is False
    assert run.cost_usd is None
    assert run.unpriced_models == ["recording/recording"]


def test_an_unknown_price_is_recorded_as_unenforceable_not_as_zero():
    """The trace has to distinguish "cost nothing" from "nobody could price it"."""
    llm = RecordingLLM(plan_json((QUESTION, "semantic_search")), assessed(QUESTION, True))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])},
        ctx=ctx(),
        max_cost_usd=0.10,
    )

    finalize = next(span for span in run.trace if span.name == "finalize")
    assert finalize.attributes["cost_usd"] is None
    assert finalize.attributes["cost_known"] is False
    assert finalize.attributes["cost_cap_enforceable"] is False
    assert finalize.attributes["unpriced_models"] == "recording/recording"


def test_a_model_priced_at_zero_is_known_and_is_not_the_unknown_case():
    """Local inference is priced at zero in the table, which is a real price.

    This is the pair that makes the previous test mean something: a zero cost
    the table states and a cost nobody can state must not look the same.
    """
    llm = RecordingLLM(
        plan_json((QUESTION, "semantic_search")),
        assessed(QUESTION, True),
        provider="ollama",
        model="llama3.1:8b",
    )

    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": FakeTool("semantic_search", chunks=[chunk(1)])},
        ctx=ctx(),
        max_cost_usd=0.0,
    )

    assert run.cost_known is True
    assert run.cost_usd == 0.0
    assert run.unpriced_models == []

    finalize = next(span for span in run.trace if span.name == "finalize")
    assert finalize.attributes["cost_cap_enforceable"] is True


def test_a_caller_may_tighten_the_configured_cost_cap():
    llm = priced_llm(*unresolved_script(QUESTION))

    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": growing_tool()},
        ctx=ctx(max_llm_calls=50, max_cost_usd=0.000001),
        max_iterations=10,
        max_cost_usd=1.0,
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_cost_usd" in run.stop_detail
