"""Fix wave: a slow plan must not leave the agent with no evidence at all.

Phase 9's first real run: on a contended 8B model, planning alone took longer
than the default ``max_latency_ms`` (30 s), so the loop stopped on the budget
before its first retrieval, with every sub-question open and no evidence, and
the agentic strategy answered with the no-evidence sentence. Retrieval costs no
model call, so the latency and cost caps are checked from the second iteration
on, and between retrieval and assessment as before: the first retrieval always
runs, and a budget stop then answers from what it found.
"""

from __future__ import annotations

import json
import time

from agent_doubles import FakeTool, RecordingLLM, chunk, ctx

from ragfabric_core.agent.loop import STOP_BUDGET, run_agent

QUESTION = "How many days of casual leave do employees get?"


class SlowPlanLLM(RecordingLLM):
    def complete(self, messages, **kwargs):
        time.sleep(0.03)
        return super().complete(messages, **kwargs)


def _plan() -> str:
    return json.dumps({"sub_questions": [{"text": QUESTION, "tool": "semantic_search", "why": ""}]})


def test_a_plan_slower_than_the_latency_cap_still_gets_one_retrieval():
    tool = FakeTool("semantic_search", chunks=[chunk(1), chunk(2)])
    llm = SlowPlanLLM(_plan())

    run = run_agent(
        QUESTION, llm=llm, tools={"semantic_search": tool}, ctx=ctx(), max_latency_ms=10
    )

    assert run.stop_reason == STOP_BUDGET
    assert "max_latency_ms" in run.stop_detail
    assert run.state.iterations == 1
    assert len(run.tool_calls) == 1
    assert {c.chunk_id for c in run.chunks} == {1, 2}
    assert run.state.llm_calls == 1  # the plan only: no assessment after the cap


def test_the_cap_still_stops_before_a_second_retrieval():
    tool = FakeTool("semantic_search", chunks=[chunk(1)])
    llm = SlowPlanLLM(_plan())
    run = run_agent(
        QUESTION,
        llm=llm,
        tools={"semantic_search": tool},
        ctx=ctx(),
        max_latency_ms=10,
        max_iterations=5,
    )
    assert run.state.iterations == 1 and len(run.tool_calls) == 1
