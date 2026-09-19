"""Token counting and the context budget.

Exact where a tokeniser exists, an explicit estimate where one does not, never a
mixture presented as a measurement (ADR 0004). tiktoken covers the OpenAI
models; local models through Ollama have no published tokeniser we can rely on,
so those use four characters per token, which is documented and labelled.

The budget drops whole chunks from the tail. Splitting a chunk would invalidate
its stored character spans, which are what the citation offsets and the citation
contract are checked against.
"""

from __future__ import annotations

from functools import lru_cache

from ragfabric_core.strategies.base import RetrievedChunk

ESTIMATED_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=8)
def _encoding(model: str):
    import tiktoken

    return tiktoken.encoding_for_model(model)


def count_tokens(text: str, model: str | None = None) -> int:
    """Exact when the model has a tokeniser, otherwise a labelled estimate."""
    if not text:
        return 0
    if model:
        try:
            return len(_encoding(model).encode(text))
        except Exception:
            pass
    return max(1, len(text) // ESTIMATED_CHARS_PER_TOKEN)


def fit_to_budget(
    chunks: list[RetrievedChunk], max_tokens: int, model: str | None = None
) -> tuple[list[RetrievedChunk], int]:
    """Keep chunks in rank order while they fit. Returns (kept, tokens_used)."""
    kept: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = count_tokens(chunk.text, model)
        if used + cost > max_tokens:
            break
        kept.append(chunk)
        used += cost
    return kept, used
