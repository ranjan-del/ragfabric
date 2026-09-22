"""The tools the agent may call, and the contract they all honour.

Tools are thin. They do not re-implement retrieval; they call the strategies
that already exist and are already tested, so there is exactly one BM25
implementation and one vector implementation in the codebase.

**Every tool receives the caller's ``RetrievalContext`` and must pass its
``AccessFilter`` through unchanged.** This matters more here than anywhere else
in the system: the agent retrieves repeatedly and pools what it finds, so a
single tool that forgets the filter leaks evidence into a pool that is later
summarised into an answer. ADR 0003 requires the filter to run inside the store
query, and a tool is not an exception to that.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragfabric_core.strategies.base import RetrievalContext, RetrievedChunk


@runtime_checkable
class AgentTool(Protocol):
    """One retrieval capability the planner can choose.

    ``description`` is shown to the model when it plans, so it is written for
    that reader: it says what the tool is good at, not how it is implemented.
    """

    name: str
    description: str

    def run(self, query: str, ctx: RetrievalContext) -> list[RetrievedChunk]: ...


ToolRegistry = dict[str, AgentTool]
