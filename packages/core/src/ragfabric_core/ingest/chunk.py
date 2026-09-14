"""Text chunking for the ingestion pipeline.

Splits parsed document text into overlapping, character-based windows and keeps
the metadata (page, chunk index, character span) needed to cite each chunk back
to its source. Text produced by the parsers may contain form-feed page markers
(``\\f``); we split on those first so each chunk carries the correct page number.

A fixed-size sliding window is the simplest strategy to reason about: the window
advances by ``chunk_size - overlap`` characters so consecutive chunks share
``overlap`` characters of context, keeping facts that straddle a boundary whole
in at least one chunk.
"""

from __future__ import annotations

from ragfabric_core.ingest.clean import detect_sections
from ragfabric_core.ingest.parser import PAGE_BREAK


def _section_for(headings: list[tuple[int, str]], offset: int) -> str | None:
    current: str | None = None
    for start, title in headings:
        if start <= offset:
            current = title
        else:
            break
    return current


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    sections: bool = False,
) -> list[dict]:
    """Split text into overlapping chunks with page + span metadata.

    Args:
        text: parsed plain text (may contain ``\\f`` page markers).
        chunk_size: maximum characters per chunk (> 0).
        overlap: characters shared between consecutive chunks
            (``0 <= overlap < chunk_size``).
        sections: when true, each chunk also carries the title of the nearest
            preceding heading detected in its page, as ``section``.

    Returns:
        A list of dicts, each shaped::

            {
                "text": str,
                "page": int,          # 1-based
                "chunk_index": int,   # global across the whole document
                "char_start": int,    # offset within its page's text
                "char_end": int,
                "section": str | None,  # only present when sections=True
            }

        The span is exact: ``page_text[char_start:char_end] == text``. That
        matters because the answer layer reports highlight offsets relative to a
        chunk, and a UI that wants to jump back into the original page text can
        only do so if the recorded offsets really bound the stored text.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    step = chunk_size - overlap
    chunks: list[dict] = []
    chunk_index = 0

    pages = text.split(PAGE_BREAK) if PAGE_BREAK in text else [text]
    for page_number, page_text in enumerate(pages, start=1):
        headings = detect_sections(page_text) if sections else []
        start = 0
        length = len(page_text)
        while start < length:
            end = min(start + chunk_size, length)
            window = page_text[start:end]
            piece = window.strip()
            if piece:
                # Re-derive the span so it bounds the STRIPPED text rather than
                # the raw window; otherwise leading/trailing whitespace would
                # push every recorded offset out by a few characters.
                lead = len(window) - len(window.lstrip())
                chunk = {
                    "text": piece,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "char_start": start + lead,
                    "char_end": start + lead + len(piece),
                }
                if sections:
                    chunk["section"] = _section_for(headings, start + lead)
                chunks.append(chunk)
                chunk_index += 1
            if end == length:
                break
            start += step

    return chunks
