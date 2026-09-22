"""Offline providers: the test doubles that keep CI free of keys and networks.

ScriptedLLMProvider replays canned responses so agent loops and routers can be
unit tested branch by branch. HashingEmbeddingProvider wraps the v1 hashing
embedder: deterministic, no download, useful for tests and for the "no key"
first run, not for production quality retrieval.

**Why the scripted double knows about nodes.** The agent asks a model three
different questions in one run: plan, then assess, then repair. A flat list of
responses only replays that correctly while the call order never varies, and
the whole point of the agent is that it varies: a branch that skips the repair
node would quietly hand the repair answer to the next assess call and the test
would pass for the wrong reason. So responses can be queued per node, and a
call is matched to the node whose name its prompt states.

Matching is deliberately conservative. A prompt that names no node, or names
two, falls back to the ordered script rather than guessing, because a double
that guesses wrong produces a green test over a path that was never taken.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping

from ragfabric_core.agent.state import NodeName
from ragfabric_core.ingest.embed import HashingEmbedder, tokenize
from ragfabric_core.providers.base import Completion, EmbeddingResult, Message, ProviderError

# The node names a prompt may identify itself by, matched on word boundaries so
# "planning" and "replan" do not count as the plan node.
_NODE_PATTERNS = {node.value: re.compile(rf"\b{node.value}\b", re.IGNORECASE) for node in NodeName}

# The key the ordered, node-less script is held under. Not a node name, so it
# can never collide with one.
_UNROUTED = ""


def detect_node(messages: list[Message]) -> str | None:
    """Which node this prompt belongs to, or ``None`` if it cannot be told.

    ``None`` when no node name appears and when more than one does. Half a
    signal is not a signal: a repair prompt that quotes the plan it is repairing
    names two nodes, and picking either one would be a coin toss decided inside
    a test double.
    """
    text = " ".join(message.content for message in messages)
    found = [name for name, pattern in _NODE_PATTERNS.items() if pattern.search(text)]
    return found[0] if len(found) == 1 else None


class ScriptedLLMProvider:
    name = "scripted"

    def __init__(
        self,
        responses: list[str] | None = None,
        model: str = "scripted",
        *,
        node_responses: Mapping[str, Iterable[str | dict | list]] | None = None,
    ) -> None:
        """``responses`` is the ordered script; ``node_responses`` queues per node.

        Both may be given: the per-node queues answer the calls that identify
        themselves, and the ordered script answers everything else. A queue
        entry that is not already a string is serialised as JSON, so a test
        writes the decision as a dict and the contract layer parses it back,
        while a string passes through untouched so malformed output can be
        scripted deliberately.
        """
        self._responses = list(responses or [])
        self._node_responses: dict[str, list[str]] = {}
        for node, queued in (node_responses or {}).items():
            for response in queued:
                self.queue(node, response)
        self.default_model = model
        self.calls = 0
        self.calls_by_node: dict[str, int] = {}

    def queue(self, node: NodeName | str, response: str | dict | list) -> None:
        """Add one response to a node's queue. An unknown node name is refused.

        Refused rather than accepted, because a mistyped key would sit in a
        queue that is never read while the node it was meant for falls through
        to the ordered script. The test would then pass having exercised a
        different path from the one it names.
        """
        name = str(node)
        if name not in _NODE_PATTERNS:
            known = ", ".join(sorted(_NODE_PATTERNS))
            raise ValueError(f"unknown agent node {name!r}: expected one of {known}")
        text = response if isinstance(response, str) else json.dumps(response)
        self._node_responses.setdefault(name, []).append(text)

    def pending(self) -> dict[str, int]:
        """What is queued and still unread, by node. ``""`` is the ordered script.

        A branch test that finishes with something pending did not take the path
        it was written for, which is worth being able to assert.
        """
        counts = {node: len(queue) for node, queue in self._node_responses.items() if queue}
        if self._responses:
            counts[_UNROUTED] = len(self._responses)
        return counts

    def _next_response(self, messages: list[Message]) -> str:
        node = detect_node(messages)
        queue = self._node_responses.get(node) if node else None
        if queue:
            self.calls_by_node[node] = self.calls_by_node.get(node, 0) + 1
            return queue.pop(0)
        if not self._responses:
            where = f" for node {node}" if node else ""
            raise ProviderError(self.name, f"script exhausted: no more responses{where}")
        if node:
            self.calls_by_node[node] = self.calls_by_node.get(node, 0) + 1
        return self._responses.pop(0)

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        started = time.perf_counter()
        text = self._next_response(messages)
        self.calls += 1
        return Completion(
            text=text,
            model=model or self.default_model,
            provider=self.name,
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(text.split()),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason="stop",
        )

    def stream(self, messages, *, model=None, max_tokens=1024, temperature=0.0):
        """Yield the scripted response word by word so tests exercise a real stream.

        An exhausted script yields nothing rather than raising, unlike
        ``complete``: a caller that streams is by definition prepared to see
        zero tokens (an empty answer is a valid outcome for a stream), so
        raising here would only make the offline double behave differently
        from a real provider under the one condition streaming exists to
        avoid surprising, namely "nothing came back".

        Node queues are read here too, so a streamed generate node can be
        scripted the same way the other nodes are.
        """
        try:
            text = self._next_response(messages)
        except ProviderError:
            return
        parts = text.split(" ")
        for i, word in enumerate(parts):
            yield word if i == len(parts) - 1 else word + " "


class HashingEmbeddingProvider:
    name = "offline"

    def __init__(self, dim: int | None = None) -> None:
        self._embedder = HashingEmbedder(dim=dim)
        self.dim = self._embedder.dim
        self.model = f"hashing-{self.dim}"

    def embed(self, texts: list[str]) -> EmbeddingResult:
        started = time.perf_counter()
        vectors = [row.tolist() for row in self._embedder.embed(texts)] if texts else []
        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            provider=self.name,
            input_tokens=sum(len(tokenize(t)) for t in texts),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
