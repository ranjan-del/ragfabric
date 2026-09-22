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

import re
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.strategies.base import RetrievalContext, RetrievedChunk
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy
from ragfabric_core.strategies.vectorless import VectorlessRAGStrategy


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


@runtime_checkable
class DocumentChunkReader(Protocol):
    """Whole-document reads, with the access predicate inside the query.

    A separate protocol from ``VectorStore`` and ``LexicalStore`` because this
    is not a search: nothing is ranked, and there is no query string to rank
    against. ``stores/document_chunks.py`` holds the SQL implementation.
    """

    name: str

    def chunks_for_document(
        self, document_id: int, access: AccessFilter
    ) -> list[RetrievedChunk]: ...


class SemanticSearchTool:
    """Meaning-based retrieval over the vector index.

    Wraps ``TraditionalRAGStrategy`` rather than embedding and querying itself,
    so there is one embed-then-search implementation in the codebase and the
    tool inherits its reranking and context budgeting unchanged.
    """

    name = "semantic_search"
    description = (
        "Find passages that mean the same thing as the question, even when they "
        "use none of its words. Best for concepts, paraphrases, policies "
        "described in other terms, and questions phrased the way a person would "
        "ask rather than the way a document would state it. Weak on exact "
        "strings: an error code or a part number can be missed entirely because "
        "a near-identical code looks almost the same to it."
    )

    def __init__(self, strategy: TraditionalRAGStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> TraditionalRAGStrategy:
        """The strategy this tool wraps (read only), for counters and reporting."""
        return self._strategy

    def run(self, query: str, ctx: RetrievalContext) -> list[RetrievedChunk]:
        # ctx is passed through untouched, filter included. A tool that rebuilt
        # the context could drop the filter by omission, which is the one
        # mistake in this file that would not show up as a failing feature.
        return self._strategy.retrieve(query, ctx).chunks


class LexicalSearchTool:
    """Exact-term retrieval over the two lexical rankings, no embeddings.

    Wraps ``VectorlessRAGStrategy``, so BM25, ts_rank_cd, fusion and the phrase
    and identifier boosts are the same ones the vectorless strategy ships.
    """

    name = "lexical_search"
    description = (
        "Find passages containing specific words exactly as written. Best for "
        "identifiers, error codes, part numbers, version strings, function or "
        "table names, quoted phrases, and any question where a character out of "
        "place changes the answer. Weak on paraphrase: a passage that answers "
        "the question in different words will not be found."
    )

    def __init__(self, strategy: VectorlessRAGStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> VectorlessRAGStrategy:
        """The strategy this tool wraps (read only), for counters and reporting."""
        return self._strategy

    def run(self, query: str, ctx: RetrievalContext) -> list[RetrievedChunk]:
        return self._strategy.retrieve(query, ctx).chunks


class FetchDocumentTool:
    """Read one whole document in order, when fragments are not enough.

    **This is the tool that has to be right about access.** Every other tool
    reaches chunks through a ranked search whose SQL already carries the
    predicate. This one names a document directly, so if it did not apply the
    same predicate the agent would have a way to read documents the principal
    cannot search for. The filter is handed to the reader on every call and the
    reader puts it inside the query; ``test_fetch_document_refuses_a_document_
    the_principal_cannot_read`` and the SQL reader tests hold that shut.

    The document is returned whole and in order, never cut to ``top_k``.
    Truncating it would defeat the reason the planner chose this tool over a
    search, and the context budget above the agent is where length is decided.
    """

    name = "fetch_document"
    description = (
        "Return one document in full, in reading order, given its id. Best when "
        "the answer needs the whole policy or procedure rather than the few "
        "passages that matched, when a search has already identified the right "
        "document but returned it in fragments, or when the question is about "
        "what a document says overall. Needs a document id; it cannot find a "
        "document by describing it."
    )

    def __init__(self, reader: DocumentChunkReader) -> None:
        self._reader = reader

    @property
    def reader(self) -> DocumentChunkReader:
        """The reader this tool fetches through (read only)."""
        return self._reader

    def run(self, query: str, ctx: RetrievalContext) -> list[RetrievedChunk]:
        document_id = parse_document_id(query)
        if document_id is None:
            # The query comes from a model, which can name no document at all.
            # Retrieving nothing is a dead end the loop already knows how to
            # handle (no new evidence, so no progress). Raising here would turn
            # a poor model choice into a failed request.
            return []
        return self._reader.chunks_for_document(document_id, ctx.access_filter)


_DOCUMENT_ID = re.compile(r"\d+")


def parse_document_id(query: str) -> int | None:
    """The document id inside a repair query, or ``None`` if there is not one.

    A model asked for a document id returns "42", "document 42" or
    "fetch_document(42)" depending on its size and mood, so the first integer in
    the string is taken. No integer means no document was named, which the
    caller treats as retrieving nothing rather than as an error.
    """
    found = _DOCUMENT_ID.search(query)
    return int(found.group()) if found else None


def build_tool_registry(
    tools: Iterable[AgentTool], enabled: Iterable[str] | None = None
) -> ToolRegistry:
    """Index tools by name, optionally keeping only the ones configuration enables.

    An enabled name that matches no tool raises. A typo in configuration that
    silently disabled a tool would leave the planner choosing from a smaller set
    than the operator believes it has, and the resulting answers would look like
    a model failure rather than a configuration one.
    """
    by_name: ToolRegistry = {tool.name: tool for tool in tools}
    if enabled is None:
        return by_name
    wanted = list(enabled)
    unknown = sorted(set(wanted) - set(by_name))
    if unknown:
        raise KeyError(f"unknown tool(s) enabled: {', '.join(unknown)}")
    return {name: by_name[name] for name in wanted}
