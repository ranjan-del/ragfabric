"""LangChain as an evaluation target: InMemoryVectorStore over RagFabric's chunks, a stuff chain.

LangChain ships no single default QA prompt any more; this uses the prompt of
its own RAG tutorial (``rlm/rag-prompt`` on the LangChain Hub), inlined so a
run needs no network. Imported only when run (D15).
"""

from __future__ import annotations

from collections.abc import Sequence

from adapters.common import ChunkRecord, FrameworkTarget, Passage

RAG_PROMPT = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved "
    "context to answer the question. If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \nContext: {context} \nAnswer:"
)


def _passages(pairs) -> list[Passage]:
    out = []
    for doc, score in pairs:
        meta = doc.metadata or {}
        out.append(
            Passage(
                document=str(meta.get("document", "")),
                text=doc.page_content,
                score=float(score) if score is not None else None,
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
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
    llm = ChatOllama(model=model, base_url=base_url, num_ctx=num_ctx, temperature=0.0)
    store = InMemoryVectorStore(embeddings)
    # RagFabric's stored vectors go straight into the store (same model), so
    # documents are not embedded twice; only the query is embedded by LangChain.
    for c in chunks:
        store.store[str(c.chunk_id)] = {
            "id": str(c.chunk_id),
            "vector": c.embedding,
            "text": c.text,
            "metadata": {
                "document": c.document,
                "collection_id": c.collection_id,
                "chunk_id": c.chunk_id,
            },
        }
    keep = None
    if allowed_collections is not None:
        allowed = set(allowed_collections)

        def keep(doc) -> bool:
            return doc.metadata.get("collection_id") in allowed

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    def retrieve(question: str) -> list[Passage]:
        return _passages(store.similarity_search_with_score(question, k=top_k, filter=keep))

    def query(question: str) -> tuple[str, list[Passage]]:
        passages = retrieve(question)
        context = "\n\n".join(p.text for p in passages)
        reply = (prompt | llm).invoke({"question": question, "context": context})
        return str(reply.content), passages

    return FrameworkTarget("langchain", retrieve, query, llm_model=model)
