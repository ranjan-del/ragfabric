"""Text cleaning and light structure detection before chunking.

Why clean at all: PDF text extraction leaves hyphenated line breaks
("employ-\\nee"), runs of spaces and tabs, and headers or footers repeated on
every page. Each of those hurts retrieval: the broken word never matches a
query, the noise dilutes embeddings, and a repeated footer becomes the most
"similar" chunk to almost anything. Cleaning is deterministic and conservative:
it never reorders text and never touches page boundaries, so character spans
computed after cleaning stay exact for the cleaned text stored in the chunk.

Section detection is a heuristic over single lines: numbered headings ("1.",
"2.3"), short ALL CAPS lines, and markdown "#" headings. A chunk records the
nearest preceding heading as its ``section``, which later feeds citations and
metadata filters. Wrong guesses are cheap here because nothing is ranked by
section in this phase.
"""

from __future__ import annotations

import re
from collections import Counter

from ragfabric_core.ingest.parser import PAGE_BREAK

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_BLANK_RUNS = re.compile(r"\n{3,}")
_NUMBERED = re.compile(r"^\d+(\.\d+)*[.)]?\s+\S")
_MARKDOWN = re.compile(r"^#{1,6}\s+(.+)$")
_MAX_HEADING_LEN = 80
_MIN_REPEATS = 3

_DOCUMENT_TYPES = {
    "pdf": "document",
    "docx": "document",
    "pptx": "presentation",
    "csv": "table",
    "txt": "text",
    "md": "text",
}


def document_type_for(extension: str) -> str:
    return _DOCUMENT_TYPES.get(extension.lower(), "other")


def _clean_page(page: str) -> str:
    page = _HYPHEN_BREAK.sub(r"\1\2", page)
    page = _SPACES.sub(" ", page)
    page = "\n".join(line.strip() for line in page.split("\n"))
    page = _BLANK_RUNS.sub("\n\n", page)
    return page.strip()


def _drop_repeated_lines(pages: list[str]) -> list[str]:
    """Remove lines that appear on at least ``_MIN_REPEATS`` pages (headers and footers)."""
    if len(pages) < _MIN_REPEATS:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        for line in set(page.split("\n")):
            if line:
                counts[line] += 1
    repeated = {line for line, n in counts.items() if n >= _MIN_REPEATS}
    if not repeated:
        return pages
    return ["\n".join(ln for ln in page.split("\n") if ln not in repeated) for page in pages]


def clean_text(text: str) -> str:
    pages = text.split(PAGE_BREAK) if PAGE_BREAK in text else [text]
    pages = [_clean_page(p) for p in pages]
    pages = _drop_repeated_lines(pages)
    return PAGE_BREAK.join(pages)


def _is_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LEN:
        return None
    md = _MARKDOWN.match(stripped)
    if md:
        return md.group(1).strip()
    if stripped.endswith((".", ",", ";", ":")) and not _NUMBERED.match(stripped):
        return None
    if _NUMBERED.match(stripped):
        return stripped
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters):
        return stripped
    return None


def detect_sections(page_text: str) -> list[tuple[int, str]]:
    """Return ``(start_offset, title)`` for every heading line in ``page_text``."""
    found: list[tuple[int, str]] = []
    offset = 0
    for line in page_text.split("\n"):
        title = _is_heading(line)
        if title is not None:
            found.append((offset, title))
        offset += len(line) + 1
    return found
