"""LlamaIndex as an evaluation target: VectorStoreIndex over RagFabric's chunks, default QA prompt.

Imported only when run (``uv run --with`` the pinned versions in
``benchmarks/requirements-competitors.txt``); never a workspace dependency (D15).
"""

from __future__ import annotations

from collections.abc import Sequence

from adapters.common import ChunkRecord, FrameworkTarget, Passage


def _passages(nodes) -> list[Passage]:
    out = []
    for item in nodes:
        node = item.node
        meta = node.metadata or {}
        out.append(
            Passage(
                document=str(meta.get("document", "")),
                text=node.get_content(),
                score=getattr(item, "score", None),
                collection_id=meta.get("collection_id"),
                chunk_id=meta.get("chunk_id"),
            )
        )
    return out


def build(
    chunks: Sequence[ChunkRecord],
    *,
    model: str,
    top_k: int,
    base_url: str = "http://localhost:11434",
    num_ctx: int = 8192,
    allowed_collections: set[int] | None = None,
) -> FrameworkTarget:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama

    embed = OllamaEmbedding(model_name="nomic-embed-text", base_url=base_url)
    llm = Ollama(
        model=model,
        base_url=base_url,
        context_window=num_ctx,
        temperature=0.0,
        request_timeout=900.0,
    )
    nodes = [
        TextNode(
            id_=str(c.chunk_id),
            text=c.text,
            embedding=c.embedding,
            metadata={
                "document": c.document,
                "collection_id": c.collection_id,
                "chunk_id": c.chunk_id,
            },
            excluded_embed_metadata_keys=["document", "collection_id", "chunk_id"],
            excluded_llm_metadata_keys=["collection_id", "chunk_id"],
        )
        for c in chunks
    ]
    index = VectorStoreIndex(nodes, embed_model=embed)
    filters = None
    if allowed_collections is not None:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="collection_id",
                    value=sorted(allowed_collections),
                    operator=FilterOperator.IN,
                )
            ]
        )
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    engine = index.as_query_engine(llm=llm, similarity_top_k=top_k, filters=filters)

    def retrieve(question: str) -> list[Passage]:
        return _passages(retriever.retrieve(question))

    def query(question: str) -> tuple[str, list[Passage]]:
        # The default query engine: compact synthesis with the default text QA prompt.
        response = engine.query(question)
        return str(response.response or ""), _passages(response.source_nodes)

    return FrameworkTarget("llamaindex", retrieve, query, llm_model=model)
