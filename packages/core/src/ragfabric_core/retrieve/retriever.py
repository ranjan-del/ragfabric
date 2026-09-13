"""Semantic retriever.

Embed the user's question with the SAME offline embedder used at ingest time and
run a cosine-similarity search over the in-memory vector store. Using one
embedder for both sides is essential: the query and document vectors only live in
the same space if they were produced by the same function.

Metadata filters (collection / document) let a search be scoped to a single
collection or a single document, which is how the "Collections" feature and
per-document search are implemented.
"""

from __future__ import annotations

from ragfabric_core.ingest.embed import get_embedder
from ragfabric_core.store.vector_store import get_store


class Retriever:
    """Embed a query and fetch the most similar chunks from the vector store."""

    def __init__(self, store=None, embedder=None):
        self.store = store or get_store()
        self.embedder = embedder or get_embedder()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        collection_id: int | None = None,
        document_id: int | None = None,
        format: str | None = None,
    ) -> list[dict]:
        """Return the ``top_k`` chunks most similar to ``query``.

        Each returned chunk carries its cosine ``score`` plus the citation
        metadata (``filename``, ``page``, ``document_id`` ...) needed to trace it
        back to the source document.
        """
        query_vector = self.embedder.embed_one(query)
        return self.store.search(
            query_vector,
            top_k=top_k,
            collection_id=collection_id,
            document_id=document_id,
            format=format,
        )
