"""What every comparison adapter shares: RagFabric's own chunks, read from a collection (D14).

Competitors index exactly the chunks RagFabric made, with the embeddings
RagFabric stored for them (the same model, ``nomic-embed-text`` through
Ollama), so chunking and document embedding are not variables. Each tool
embeds the query itself, through its own Ollama integration and the same
model, and retrieves the same ``top_k``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ragfabric_core.evaluation.target import ContextItem, TargetAnswer


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    document: str
    collection_id: int | None
    text: str
    embedding: list[float] = field(repr=False)


def load_chunks(session_factory, collection_ids: Sequence[int]) -> list[ChunkRecord]:
    """Every chunk of the given collections, in id order, with its document's file name.

    The vector is the one the pgvector store holds (``chunk_embeddings``, the
    configured model), not the legacy ``chunks.embedding`` column.
    """
    from ragfabric_core.models.document import Chunk, Document
    from ragfabric_core.models.index import ChunkEmbedding

    with session_factory() as db:
        rows = (
            db.query(Chunk, Document.filename, ChunkEmbedding.embedding)
            .join(Document, Document.id == Chunk.document_id)
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
            .filter(Chunk.collection_id.in_(list(collection_ids)))
            .order_by(Chunk.id)
            .all()
        )
        return [
            ChunkRecord(
                chunk_id=c.id,
                document=filename,
                collection_id=c.collection_id,
                text=c.text,
                embedding=[float(x) for x in vector],
            )
            for c, filename, vector in rows
        ]


@dataclass(frozen=True)
class Passage:
    """One retrieved passage in a tool-neutral shape."""

    document: str
    text: str
    score: float | None
    collection_id: int | None = None
    chunk_id: int | None = None


def contexts(passages: Sequence[Passage]) -> list[ContextItem]:
    return [
        ContextItem(rank=i, document=p.document, text=p.text, score=p.score)
        for i, p in enumerate(passages, start=1)
    ]


class FrameworkTarget:
    """An ``EvalTarget`` over two callables a framework supplies.

    ``retrieve_fn(question) -> list[Passage]`` and
    ``query_fn(question) -> (answer, passages)``, the tool's own one-call
    question answering path (it retrieves again, as a user's call would). The tool's default prompt does
    not ask for ``[n]`` markers, so answers declare ``cites=False`` (D16).
    Token counts are not reported by the tools' high level APIs, so they are
    ``None`` rather than estimated (ADR 0004).
    """

    def __init__(
        self,
        name: str,
        retrieve_fn: Callable[[str], list[Passage]],
        query_fn: Callable[[str], tuple[str, list[Passage]]],
        *,
        llm_model: str | None = None,
    ) -> None:
        self.name = name
        self._retrieve = retrieve_fn
        self._query = query_fn
        self.llm_model = llm_model

    def retrieve(self, question: str) -> TargetAnswer:
        started = time.perf_counter()
        passages = self._retrieve(question)
        ms = int((time.perf_counter() - started) * 1000)
        return TargetAnswer(
            answer="",
            contexts=contexts(passages),
            strategy_used=self.name,
            retrieval_calls=1,
            embedding_calls=1,
            latency_ms=ms,
            retrieval_latency_ms=ms,
            llm_model=self.llm_model,
            cites=False,
        )

    def answer(self, question: str) -> TargetAnswer:
        started = time.perf_counter()
        text, passages = self._query(question)
        done = time.perf_counter()
        return TargetAnswer(
            answer=text,
            contexts=contexts(passages),
            strategy_used=self.name,
            llm_calls=None,  # the tool may refine over several calls; not reported
            retrieval_calls=1,
            embedding_calls=1,
            latency_ms=int((done - started) * 1000),
            llm_model=self.llm_model,
            estimated_cost_usd=0.0,
            cites=False,
        )
