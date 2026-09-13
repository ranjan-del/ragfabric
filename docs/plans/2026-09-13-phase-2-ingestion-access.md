# Phase 2: Shared Ingestion, Access Control Foundation, Workers, CLI, Tracing. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ingestion shared and real (cleaning, sections, configurable chunking, retained originals, fan out to a pgvector `VectorStore` and a PostgreSQL full text `LexicalStore` through Redis backed workers), make access control real (groups, grants, overrides, hashed API keys, an `AccessFilter` computed once per request and applied inside every store query, an audit row per query), extend the CLI (`init`, `ingest`, `users`, `groups`, `grants`, `keys`, `worker`), and add tracing spans on every ingestion and retrieval step, stored per run.

**Architecture:** Everything lands in `ragfabric_core` behind the Phase 1 interfaces. The v1 in memory index stays the live query path for one more phase (Phase 3 replaces it) but now honours the `AccessFilter`; the new stores are fed by the fan out so Phases 3 and 4 can query them. Indexing runs inline by default and through a Redis queue and a `ragfabric worker` process when `ingestion.indexing: queue`. Access policy lives in core (`auth/policy.py`) and the server only calls it.

**Tech Stack:** Python 3.13, uv, SQLAlchemy 2.0, Alembic, pgvector 0.5 (`pgvector.sqlalchemy.Vector`), PostgreSQL 18 `tsvector` with a GIN index, redis-py 8.1, opentelemetry-sdk 1.44 with the OTLP HTTP exporter, Typer, pytest.

**Spec:** `docs/design/2026-09-13-ragfabric-design.md` sections 5, 7, 8; ADR 0003. Roadmap issue #3.

## Global Constraints

- Python `>=3.13,<3.14`. No em dashes in any file written. No AI assistant references anywhere in code, docs or commit messages. Commits authored `Ranjan G <ranjan.g@ispf.ngo>` via `git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit ...`, conventional prefixes, NO trailers of any kind; verify with `git log -1 --format='%(trailers)'` after every commit.
- No secrets in the repository. API keys are stored as SHA 256 hashes only; the plaintext is shown exactly once at creation.
- `ragfabric_core` never imports `ragfabric_server` or `ragfabric_cli` (`uv run lint-imports` must keep 2 contracts).
- Every store query takes and applies an `AccessFilter` before ranking (ADR 0003). No post ranking filtering anywhere new.
- All 161 Phase 1 tests keep passing; the 1 skipped is the live OpenAI check. Only import paths and additive behaviour may change in existing tests; assertions are not weakened.
- Work on branch `feat/phase-2-ingestion-access` in a worktree at `~/AI/ragfabric-wt/phase-2`. Push to the feature branch is pre authorised; the PR opens only after the whole branch review; merging is the owner's action.
- Default configuration keeps the first run key free where possible: unit tests use the offline hashing embedder; the example yaml keeps `openai` as documented. Do not introduce silent fallbacks.
- Chunking defaults come from `ragfabric.yaml` (`ingestion.chunk_size` 600, `chunk_overlap` 80). The v1 `chunk_text()` function defaults stay 800/100 so its unit tests are unchanged; the pipeline passes the configured values.

---

## File structure after this phase (new or changed)

```
packages/core/src/ragfabric_core/
    runtime.py                          get_config(), reset_config(), get_session_factory()
    config_file.py                      IngestionConfig gains uploads_dir and indexing; TelemetryConfig unchanged
    ingest/clean.py                     clean_text(), detect_sections(), document_type_for()
    ingest/chunk.py                     chunk_text(..., sections=None) attaches "section"
    ingest/storage.py                   FileStorage, get_storage()
    ingest/indexing.py                  schedule_indexing(), index_inline()
    ingest/pipeline.py                  uses config, clean, sections, storage, schedule_indexing
    models/document.py                  Document.document_type, Document.storage_path, Chunk.section
    models/index.py                     ChunkEmbedding, ChunkSearch
    migrations/versions/0003_ingestion_and_indexes.py
    stores/access_sql.py                access_clause()
    stores/pgvector_store.py            PgVectorStore
    stores/postgres_fts.py              PostgresLexicalStore
    stores/memory_cache.py, redis_cache.py
    stores/registry.py                  build_vector_store(), build_lexical_store(), build_cache()
    store/vector_store.py               v1 store: candidate_rows(filters, access=None), search(..., access=None)
    retrieve/retriever.py, hybrid.py    retrieve(..., access=None)
    strategies/legacy.py                passes ctx.access_filter through
    queue/base.py, memory_queue.py, redis_queue.py, registry.py
    workers/handlers.py, runner.py
    auth/api_keys.py                    generate, hash, verify, principal_for_api_key, principal_for_user
    auth/service.py                     groups, members, grants, overrides
    auth/policy.py                      compute_access_filter()
    auth/ratelimit.py                   check_rate_limit()
    telemetry/tracing.py                trace(), start_trace(), collected_spans(), configure_otel()
packages/server/src/ragfabric_server/
    deps.py                             get_principal(), get_access_filter(), enforce_rate_limit()
    api/routes/search.py                access filter, RetrievalRun + AuditLog per query, trace
    api/routes/documents.py             access aware list/get, download, storage cleanup
    api/routes/access.py                admin: groups, members, grants, overrides, keys
    api/routes/runs.py                  GET /api/runs/{id}
packages/cli/src/ragfabric_cli/
    main.py                             registers sub apps; adds init and worker
    commands/ingest.py, users.py, access.py
docker-compose.yml                      worker service
.github/workflows/ci.yml                migrations job also runs integration tests on PostgreSQL 18
docs/                                   getting-started, configuration, architecture, providers, concepts/ingestion.md, concepts/access-control.md
```

---

### Task 1: Worktree, dependencies, runtime accessors and config fields

**Files:**
- Modify: `packages/core/pyproject.toml`, `packages/core/src/ragfabric_core/config_file.py`, `ragfabric.example.yaml`
- Create: `packages/core/src/ragfabric_core/runtime.py`
- Test: `packages/core/tests/test_runtime.py`

**Interfaces:**
- Produces: `IngestionConfig.uploads_dir: str = "data/uploads"`, `IngestionConfig.indexing: Literal["inline", "queue"] = "inline"`; `runtime.get_config() -> RagFabricConfig` (cached, reads `RAGFABRIC_CONFIG` or `./ragfabric.yaml`), `runtime.reset_config()`, `runtime.get_session_factory() -> sessionmaker` (returns `ragfabric_core.db.session.SessionLocal`).

- [ ] **Step 1: Worktree and branch**

```bash
cd ~/AI/ragfabric && git fetch -q origin && git checkout -q main && git pull -q --ff-only
git worktree add -b feat/phase-2-ingestion-access ~/AI/ragfabric-wt/phase-2 main
cd ~/AI/ragfabric-wt/phase-2 && uv sync -q && uv run pytest -q -p no:warnings 2>&1 | tail -1
```

Expected: 161 passed, 1 skipped.

- [ ] **Step 2: Add dependencies**

In `packages/core/pyproject.toml` add to `dependencies`:

```toml
  "pgvector>=0.5",
  "redis>=8.1",
  "opentelemetry-sdk>=1.44",
  "opentelemetry-exporter-otlp-proto-http>=1.44",
```

Run `uv sync` and `uv run python -c "import pgvector.sqlalchemy, redis, opentelemetry.sdk; print('deps ok')"`.

- [ ] **Step 3: Write the failing tests**

```python
# packages/core/tests/test_runtime.py
from ragfabric_core import runtime
from ragfabric_core.config_file import IngestionConfig


def test_ingestion_config_new_fields_have_defaults():
    cfg = IngestionConfig()
    assert cfg.uploads_dir == "data/uploads"
    assert cfg.indexing == "inline"


def test_get_config_is_cached_and_resettable(tmp_path, monkeypatch):
    p = tmp_path / "ragfabric.yaml"
    p.write_text("ingestion:\n  indexing: queue\n  chunk_size: 300\n")
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(p))
    runtime.reset_config()
    first = runtime.get_config()
    assert first.ingestion.indexing == "queue" and first.ingestion.chunk_size == 300
    p.write_text("ingestion:\n  indexing: inline\n")
    assert runtime.get_config() is first
    runtime.reset_config()
    assert runtime.get_config().ingestion.indexing == "inline"


def test_session_factory_is_the_shared_one():
    from ragfabric_core.db.session import SessionLocal

    assert runtime.get_session_factory() is SessionLocal
```

Run: `uv run pytest packages/core/tests/test_runtime.py -q` → expect `ModuleNotFoundError`.

- [ ] **Step 4: Implement**

Add to `IngestionConfig` in `config_file.py` (after `retain_originals`):

```python
    uploads_dir: str = "data/uploads"
    # inline: index in the request that ingests. queue: enqueue for `ragfabric worker`.
    indexing: Literal["inline", "queue"] = "inline"
```

```python
# packages/core/src/ragfabric_core/runtime.py
"""Process wide accessors for configuration and the database session factory.

`get_config()` loads ragfabric.yaml once (path from RAGFABRIC_CONFIG or the
working directory) and caches it; `reset_config()` exists for tests and for the
CLI after `ragfabric init` writes a new file. Keeping these here, rather than
importing module level singletons everywhere, lets tests swap configuration
without reloading modules.
"""

from __future__ import annotations

from ragfabric_core.config_file import RagFabricConfig, load_config

_config: RagFabricConfig | None = None


def get_config() -> RagFabricConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    global _config
    _config = None


def get_session_factory():
    from ragfabric_core.db.session import SessionLocal

    return SessionLocal
```

Add to `ragfabric.example.yaml` under `ingestion:` after `retain_originals`:

```yaml
  uploads_dir: data/uploads   # where retained originals are stored (a volume in compose)
  indexing: inline            # inline | queue (queue needs Redis and `ragfabric worker`)
```

- [ ] **Step 5: Run, lint, commit**

```bash
uv run pytest packages/core/tests/test_runtime.py packages/core/tests/test_config_file.py -q && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat(core): add runtime config accessors, indexing mode and uploads directory settings" && git log -1 --format='%h %(trailers)'
```

---

### Task 2: Cleaning, section detection, document type, chunk sections, migration 0003 part one

**Files:**
- Create: `packages/core/src/ragfabric_core/ingest/clean.py`
- Modify: `packages/core/src/ragfabric_core/ingest/chunk.py`, `models/document.py`, `ingest/pipeline.py`
- Create: `packages/core/src/ragfabric_core/migrations/versions/0003_ingestion_and_indexes.py` (this task adds the three columns; Task 4 adds the two index tables to the SAME revision before it is ever released)
- Test: `packages/core/tests/test_clean.py`, additions to `test_pipeline.py`, `test_migrations.py`

**Interfaces:**
- Produces: `clean_text(text: str) -> str`; `detect_sections(page_text: str) -> list[tuple[int, str]]` (start offset, heading title); `document_type_for(extension: str) -> str` returning `document` (pdf, docx), `presentation` (pptx), `table` (csv), `text` (txt, md), else `other`; `chunk_text(text, chunk_size=800, overlap=100, sections: bool = False)` where `sections=True` adds `"section": str | None` to every chunk dict; `Document.document_type: str` (default ""), `Document.storage_path: str | None`, `Chunk.section: str | None`.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_clean.py
from ragfabric_core.ingest.chunk import chunk_text
from ragfabric_core.ingest.clean import clean_text, detect_sections, document_type_for
from ragfabric_core.ingest.parser import PAGE_BREAK


def test_clean_joins_hyphenated_line_breaks_and_collapses_whitespace():
    raw = "The employ-\nee handbook   covers\tleave.\n\n\n\nNext paragraph."
    assert clean_text(raw) == "The employee handbook covers leave.\n\nNext paragraph."


def test_clean_removes_lines_repeated_on_three_or_more_pages():
    page = "ACME Corp Confidential\nBody {n}\nPage {n} of 3"
    raw = PAGE_BREAK.join(page.format(n=n) for n in (1, 2, 3))
    cleaned = clean_text(raw)
    assert "ACME Corp Confidential" not in cleaned
    assert "Body 2" in cleaned
    assert cleaned.count(PAGE_BREAK) == 2


def test_clean_keeps_a_line_that_repeats_on_only_two_pages():
    raw = PAGE_BREAK.join(["Header\nA", "Header\nB", "Other\nC"])
    assert clean_text(raw).count("Header") == 2


def test_detect_sections_finds_numbered_caps_and_markdown_headings():
    text = "1. Introduction\nsome text\nLEAVE POLICY\nmore text\n## Carry forward\nrules\nplain line"
    found = detect_sections(text)
    assert [title for _, title in found] == ["1. Introduction", "LEAVE POLICY", "Carry forward"]
    assert found[0][0] == 0 and text[found[1][0] :].startswith("LEAVE POLICY")


def test_detect_sections_ignores_long_or_sentence_like_lines():
    text = "This is a long sentence that ends with a period and is not a heading.\nOK"
    assert detect_sections(text) == []


def test_chunk_text_attaches_the_nearest_preceding_section_when_asked():
    text = "1. Intro\n" + ("a " * 50) + "\n2. Leave\n" + ("b " * 50)
    chunks = chunk_text(text, chunk_size=60, overlap=0, sections=True)
    assert chunks[0]["section"] == "1. Intro"
    assert chunks[-1]["section"] == "2. Leave"
    assert all("section" in c for c in chunks)


def test_chunk_text_default_does_not_add_section_key():
    assert "section" not in chunk_text("hello world")[0]


def test_document_type_mapping():
    assert document_type_for("pdf") == "document"
    assert document_type_for("pptx") == "presentation"
    assert document_type_for("csv") == "table"
    assert document_type_for("txt") == "text" and document_type_for("md") == "text"
    assert document_type_for("xyz") == "other"
```

Add to `packages/core/tests/test_migrations.py` `EXPECTED_TABLES` nothing yet (Task 4 adds tables); add a column test:

```python
def test_0003_adds_section_document_type_and_storage_path(tmp_path):
    engine, _ = _migrated_engine(tmp_path)
    cols = {c["name"] for c in inspect(engine).get_columns("chunks")}
    assert "section" in cols
    dcols = {c["name"] for c in inspect(engine).get_columns("documents")}
    assert {"document_type", "storage_path"} <= dcols
```

Run: `uv run pytest packages/core/tests/test_clean.py packages/core/tests/test_migrations.py -q` → expect ImportError and the new migration test failing.

- [ ] **Step 2: Implement cleaning and sections**

```python
# packages/core/src/ragfabric_core/ingest/clean.py
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
```

- [ ] **Step 3: Extend the chunker**

In `chunk.py`, change the signature to `def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100, sections: bool = False) -> list[dict]:`, import `detect_sections` from `ragfabric_core.ingest.clean`, and inside the page loop compute `headings = detect_sections(page_text) if sections else []` before the `while`. When appending a chunk, if `sections` is true add `"section": _section_for(headings, start + lead)`. Add the helper:

```python
def _section_for(headings: list[tuple[int, str]], offset: int) -> str | None:
    current: str | None = None
    for start, title in headings:
        if start <= offset:
            current = title
        else:
            break
    return current
```

Update the docstring's returned dict shape to mention the optional `section`.

- [ ] **Step 4: Models and migration 0003 (columns only)**

In `models/document.py` add to `Document` after `format`:

```python
    # document | presentation | table | text | other, derived from the extension
    document_type: Mapped[str] = mapped_column(String, default="", nullable=False)
    # Relative path of the retained original under ingestion.uploads_dir, if kept
    storage_path: Mapped[str | None] = mapped_column(String, nullable=True)
```

and to `Chunk` after `char_end`:

```python
    section: Mapped[str | None] = mapped_column(String, nullable=True)
```

Generate the migration against a database at 0002 and review it:

```bash
cd ~/AI/ragfabric-wt/phase-2 && rm -f /tmp/rf_autogen3.db
export DATABASE_URL=sqlite:////tmp/rf_autogen3.db JWT_SECRET=dev-only-secret-not-for-production-use-1234567890
uv run python -c "from ragfabric_core.db.migrate import upgrade; import os; upgrade(os.environ['DATABASE_URL'], '0002_platform_tables')"
uv run alembic -c packages/core/src/ragfabric_core/migrations/alembic.ini revision --autogenerate -m "ingestion and indexes" --rev-id 0003_ingestion_and_indexes
unset DATABASE_URL JWT_SECRET
```

If Alembic doubles the slug in the filename, rename the file to `0003_ingestion_and_indexes.py`. Set the docstring to:

```python
"""Ingestion metadata and index tables.

Adds documents.document_type and documents.storage_path, chunks.section, and
(in the same revision, added by the store task) chunk_embeddings and
chunk_search.

Revision ID: 0003_ingestion_and_indexes
Revises: 0002_platform_tables
"""
```

`documents.document_type` must be added with `server_default=""` so existing rows stay valid, then the server default may be dropped in the same script if the model has none; simplest is to keep `server_default=""` in the migration and add `server_default=""` to the model column too so the drift test agrees. Do the same for nothing else (the other two columns are nullable).

- [ ] **Step 5: Wire the pipeline**

In `ingest/pipeline.py`:
- import `from ragfabric_core.ingest.clean import clean_text, document_type_for` and `from ragfabric_core.runtime import get_config`
- in `_index_content`, after parsing: `text = clean_text(text)`, then `cfg = get_config().ingestion` and `chunks = chunk_text(text, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap, sections=True)`
- when building `Chunk(...)` pass `section=chunk_meta.get("section")`
- in `ingest_document` set `document_type=document_type_for(_extension(filename))` on the `Document(...)`; in `reingest_document` set `document.document_type = document_type_for(document.format)` after `document.format = ...`

Add to `packages/core/tests/test_pipeline.py` (find the existing ingestion test that uses a real session and follow its fixture pattern; if none is convenient, use the server `client` upload path in `packages/server/tests/test_api.py` instead):

```python
def test_ingest_records_document_type_and_sections(client, auth_headers):
    from ragfabric_core.testing.fixtures import make_txt

    data = make_txt("1. Intro\n" + "alpha " * 200 + "\n2. Leave\n" + "beta " * 200)
    r = client.post("/api/documents/upload", files={"file": ("notes.txt", data, "text/plain")}, headers=auth_headers)
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["document_type"] == "text"
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.document import Chunk

    with SessionLocal() as db:
        sections = {c.section for c in db.query(Chunk).filter(Chunk.document_id == doc["id"]).all()}
    assert "1. Intro" in sections and "2. Leave" in sections
```

(Place this test in `packages/server/tests/test_api.py`, since it needs the client.) Add `document_type: str = ""` to `DocumentOut` in `packages/server/src/ragfabric_server/schemas/document.py`.

- [ ] **Step 6: Run everything, lint, commit**

```bash
uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat(core): clean text, detect sections and document types, configurable chunking, migration 0003 columns" && git log -1 --format='%h %(trailers)'
```

Expected: previous count plus 9 new tests, all passing.

---

### Task 3: Retained originals and the download route

**Files:**
- Create: `packages/core/src/ragfabric_core/ingest/storage.py`
- Modify: `ingest/pipeline.py`, `packages/server/src/ragfabric_server/api/routes/documents.py`, `api/routes/admin.py`
- Test: `packages/core/tests/test_storage.py`, additions to `packages/server/tests/test_api.py`

**Interfaces:**
- Produces: `FileStorage(root: Path)` with `save(document_id: int, filename: str, data: bytes) -> str` (relative path `"<document_id>/<safe name>"`), `path_for(relative: str) -> Path`, `delete(document_id: int) -> None`; `get_storage() -> FileStorage` rooted at `get_config().ingestion.uploads_dir` (relative paths resolve against the current working directory); `GET /api/documents/{id}/download` streams the original when `retain_originals` is on, 404 otherwise.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_storage.py
from pathlib import Path

import pytest

from ragfabric_core.ingest.storage import FileStorage


def test_save_writes_under_document_id_and_sanitises_the_name(tmp_path):
    fs = FileStorage(tmp_path)
    rel = fs.save(7, "../../etc/passwd weird name.PDF", b"%PDF-1.4")
    assert rel == "7/etc_passwd_weird_name.PDF"
    assert fs.path_for(rel).read_bytes() == b"%PDF-1.4"
    assert fs.path_for(rel).resolve().is_relative_to(tmp_path.resolve())


def test_delete_removes_the_document_directory(tmp_path):
    fs = FileStorage(tmp_path)
    fs.save(3, "a.txt", b"a")
    fs.delete(3)
    assert not (tmp_path / "3").exists()
    fs.delete(3)  # idempotent


def test_path_for_rejects_escapes(tmp_path):
    fs = FileStorage(tmp_path)
    with pytest.raises(ValueError):
        fs.path_for("../outside")


def test_get_storage_uses_config(tmp_path, monkeypatch):
    from ragfabric_core import runtime
    from ragfabric_core.ingest.storage import get_storage

    p = tmp_path / "ragfabric.yaml"
    p.write_text(f"ingestion:\n  uploads_dir: {tmp_path / 'blobs'}\n")
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(p))
    runtime.reset_config()
    assert get_storage().root == Path(tmp_path / "blobs")
    runtime.reset_config()
```

Add to `packages/server/tests/test_api.py`:

```python
def test_upload_retains_the_original_and_download_returns_it(client, auth_headers):
    from ragfabric_core.testing.fixtures import make_txt

    data = make_txt("retained body")
    doc = client.post("/api/documents/upload", files={"file": ("keep.txt", data, "text/plain")}, headers=auth_headers).json()
    assert doc["storage_path"] == f"{doc['id']}/keep.txt"
    r = client.get(f"/api/documents/{doc['id']}/download", headers=auth_headers)
    assert r.status_code == 200 and r.content == data
    assert r.headers["content-disposition"].endswith('filename="keep.txt"')
    client.delete(f"/api/documents/{doc['id']}", headers=auth_headers)
    assert client.get(f"/api/documents/{doc['id']}/download", headers=auth_headers).status_code == 404
```

Add `storage_path: str | None = None` to `DocumentOut`. In `packages/server/tests/conftest.py`, before the app import, point uploads at a temp dir so tests never write into the repo: after the existing `os.environ[...]` lines add

```python
_UPLOADS = os.path.join(tempfile.gettempdir(), "ragfabric_test_uploads")
_CFG = os.path.join(tempfile.gettempdir(), "ragfabric_test_config.yaml")
with open(_CFG, "w") as fh:
    fh.write(f"ingestion:\n  uploads_dir: {_UPLOADS}\n")
os.environ["RAGFABRIC_CONFIG"] = _CFG
```

Run the two test files → expect failures on import and 404.

- [ ] **Step 2: Implement storage**

```python
# packages/core/src/ragfabric_core/ingest/storage.py
"""Retained originals on local disk.

Citations can only link back to a source if the source still exists. Files are
kept under ``<uploads_dir>/<document_id>/<sanitised name>`` so a document's
files are removed together and names from users can never escape the root.
Object storage is a later connector behind the same three methods.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from ragfabric_core.runtime import get_config

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    base = Path(name).name or "file"
    cleaned = _UNSAFE.sub("_", base).strip("._") or "file"
    return cleaned


class FileStorage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(self, document_id: int, filename: str, data: bytes) -> str:
        target_dir = self.root / str(document_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        relative = f"{document_id}/{safe_filename(filename)}"
        (self.root / relative).write_bytes(data)
        return relative

    def path_for(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root.resolve()):
            raise ValueError("storage path escapes the uploads root")
        return candidate

    def delete(self, document_id: int) -> None:
        shutil.rmtree(self.root / str(document_id), ignore_errors=True)


_storage: FileStorage | None = None


def get_storage() -> FileStorage:
    global _storage
    root = Path(get_config().ingestion.uploads_dir)
    if _storage is None or _storage.root != root:
        _storage = FileStorage(root)
    return _storage
```

- [ ] **Step 3: Wire pipeline and routes**

In `ingest/pipeline.py` `ingest_document`, after `db.flush()` and before `_index_content`: `if get_config().ingestion.retain_originals: document.storage_path = get_storage().save(document.id, filename, data)`. In `reingest_document`, after updating `filename`: same save (overwrites the directory contents after `get_storage().delete(document.id)`).

In `api/routes/documents.py`:
- both delete paths (documents and admin) call `get_storage().delete(document_id)` after `db.commit()`
- add:

```python
@router.get("/{document_id}/download")
def download_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    """Return the retained original file, if the deployment keeps originals."""
    document = db.get(Document, document_id)
    if document is None or not document.storage_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original not available.")
    try:
        path = get_storage().path_for(document.storage_path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original not available.") from exc
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original not available.")
    return FileResponse(path, filename=document.filename, media_type=document.content_type or "application/octet-stream")
```

with `from fastapi.responses import FileResponse` and `from ragfabric_core.ingest.storage import get_storage`. Task 8 adds the access check to this route.

- [ ] **Step 4: Run, lint, commit**

```bash
uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat: retain uploaded originals and serve them from the documents API" && git log -1 --format='%h %(trailers)'
```

---

### Task 4: Access aware SQL, pgvector `VectorStore`, PostgreSQL full text `LexicalStore`, caches, store registry, migration 0003 part two

**Files:**
- Create: `packages/core/src/ragfabric_core/models/index.py`, `stores/access_sql.py`, `stores/pgvector_store.py`, `stores/postgres_fts.py`, `stores/memory_cache.py`, `stores/redis_cache.py`, `stores/registry.py`
- Modify: `models/__init__.py`, `migrations/versions/0003_ingestion_and_indexes.py`, `packages/core/tests/test_migrations.py`
- Test: `packages/core/tests/test_access_sql.py`, `test_stores_sqlite.py`, `test_stores_postgres.py` (integration), `test_caches.py`

**Interfaces:**
- Produces: `ChunkEmbedding(chunk_id PK FK chunks CASCADE, model: str, dim: int, embedding: JSON | VECTOR)`, `ChunkSearch(chunk_id PK FK chunks CASCADE, document_id: int index, collection_id: int | None index, tsv: Text | TSVECTOR)` with `Index("ix_chunk_search_tsv", "tsv", postgresql_using="gin")`.
- `access_clause(access: AccessFilter, document_col, collection_col)` returns a SQLAlchemy boolean expression, or `None` when unrestricted.
- `PgVectorStore(session_factory, name="pgvector")` implementing `VectorStore`: `upsert(chunk_ids, vectors, payloads)` where each payload has `model` and `dim`; `query(vector, top_k, access, filters=None) -> list[RetrievedChunk]` using cosine distance `<=>` on PostgreSQL and a NumPy fallback on other dialects; `delete_document`, `count`.
- `PostgresLexicalStore(session_factory, name="postgres_fts")` implementing `LexicalStore`: `index(chunk_ids, texts, payloads)` (payload has `document_id`, `collection_id`), `search(query, top_k, access, filters=None)` using `plainto_tsquery('english', :q)` and `ts_rank_cd` on PostgreSQL and a token overlap fallback elsewhere; `delete_document`.
- `MemoryCache()`, `RedisCache(url)` implementing `Cache`; `build_vector_store(cfg, session_factory)`, `build_lexical_store(cfg, session_factory)`, `build_cache(cfg)`.
- Integration tests run only when `RAGFABRIC_TEST_DATABASE_URL` is set (PostgreSQL); they are marked `integration`.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_access_sql.py
from sqlalchemy import Column, Integer, MetaData, Table, select

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.stores.access_sql import access_clause

t = Table("t", MetaData(), Column("document_id", Integer), Column("collection_id", Integer))


def compile_(clause) -> str:
    return str(select(t).where(clause).compile(compile_kwargs={"literal_binds": True}))


def test_unrestricted_yields_no_clause():
    assert access_clause(AccessFilter.unrestricted(), t.c.document_id, t.c.collection_id) is None


def test_allow_lists_are_or_ed_and_deny_list_is_and_not():
    f = AccessFilter(document_ids=frozenset({1, 2}), collection_ids=frozenset({9}), denied_document_ids=frozenset({2}))
    sql = compile_(access_clause(f, t.c.document_id, t.c.collection_id))
    assert "document_id IN (1, 2)" in sql and "collection_id IN (9)" in sql and "OR" in sql
    assert "document_id NOT IN (2)" in sql or "NOT IN (2)" in sql


def test_deny_only_filter_is_not_unrestricted_and_excludes():
    f = AccessFilter(denied_document_ids=frozenset({5}))
    sql = compile_(access_clause(f, t.c.document_id, t.c.collection_id))
    assert "NOT IN (5)" in sql and "IN (5)" in sql


def test_empty_allow_lists_match_nothing():
    f = AccessFilter(document_ids=frozenset(), collection_ids=frozenset())
    sql = compile_(access_clause(f, t.c.document_id, t.c.collection_id))
    assert "false" in sql.lower() or "1 != 1" in sql
```

```python
# packages/core/tests/test_stores_sqlite.py
"""Store behaviour on SQLite (the dialect fallbacks). The PostgreSQL paths are in test_stores_postgres.py."""

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore


@pytest.fixture()
def sf(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 's.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        c1, c2 = Collection(name="open"), Collection(name="secret")
        db.add_all([c1, c2]); db.flush()
        d1 = Document(filename="a.txt", format="txt", collection_id=c1.id, status="ready")
        d2 = Document(filename="b.txt", format="txt", collection_id=c2.id, status="ready")
        db.add_all([d1, d2]); db.flush()
        rows = [
            Chunk(document_id=d1.id, collection_id=c1.id, chunk_index=0, page=1, char_start=0, char_end=10, text="leave policy twelve days", embedding=[]),
            Chunk(document_id=d1.id, collection_id=c1.id, chunk_index=1, page=1, char_start=0, char_end=10, text="kubernetes rollout guide", embedding=[]),
            Chunk(document_id=d2.id, collection_id=c2.id, chunk_index=0, page=1, char_start=0, char_end=10, text="secret leave bonus rules", embedding=[]),
        ]
        db.add_all(rows); db.commit()
        ids = [r.id for r in rows]
    return factory, ids, (d1.id, d2.id), (c1.id, c2.id)


def unit(v):
    a = np.asarray(v, dtype=float); return (a / np.linalg.norm(a)).tolist()


def test_pgvector_store_upsert_query_filter_and_delete(sf):
    factory, ids, (d1, d2), (c1, c2) = sf
    store = PgVectorStore(factory)
    vecs = [unit([1, 0, 0]), unit([0, 1, 0]), unit([0.9, 0.1, 0])]
    payloads = [{"model": "hashing-3", "dim": 3, "document_id": d, "collection_id": c} for d, c in ((d1, c1), (d1, c1), (d2, c2))]
    store.upsert(ids, vecs, payloads)
    assert store.count() == 3
    hits = store.query(unit([1, 0, 0]), top_k=2, access=AccessFilter.unrestricted())
    assert [h.chunk_id for h in hits] == [ids[0], ids[2]] and hits[0].score > hits[1].score
    only_open = store.query(unit([1, 0, 0]), top_k=3, access=AccessFilter(collection_ids=frozenset({c1})))
    assert {h.collection_id for h in only_open} == {c1}
    denied = store.query(unit([1, 0, 0]), top_k=3, access=AccessFilter(denied_document_ids=frozenset({d2})))
    assert all(h.document_id != d2 for h in denied)
    store.upsert([ids[0]], [unit([0, 0, 1])], [payloads[0]])  # replace
    assert store.count() == 3
    store.delete_document(d1)
    assert store.count() == 1


def test_lexical_store_index_search_filter_and_delete(sf):
    factory, ids, (d1, d2), (c1, c2) = sf
    store = PostgresLexicalStore(factory)
    texts = ["leave policy twelve days", "kubernetes rollout guide", "secret leave bonus rules"]
    payloads = [{"document_id": d, "collection_id": c} for d, c in ((d1, c1), (d1, c1), (d2, c2))]
    store.index(ids, texts, payloads)
    hits = store.search("leave", top_k=5, access=AccessFilter.unrestricted())
    assert {h.chunk_id for h in hits} == {ids[0], ids[2]}
    assert hits[0].text in texts
    scoped = store.search("leave", top_k=5, access=AccessFilter(collection_ids=frozenset({c1})))
    assert [h.chunk_id for h in scoped] == [ids[0]]
    assert store.search("nothingmatches", top_k=5, access=AccessFilter.unrestricted()) == []
    store.delete_document(d2)
    assert {h.chunk_id for h in store.search("leave", top_k=5, access=AccessFilter.unrestricted())} == {ids[0]}


def test_stores_satisfy_the_protocols(sf):
    from ragfabric_core.stores.base import LexicalStore, VectorStore

    factory, *_ = sf
    assert isinstance(PgVectorStore(factory), VectorStore)
    assert isinstance(PostgresLexicalStore(factory), LexicalStore)
```

```python
# packages/core/tests/test_stores_postgres.py
"""Real PostgreSQL paths: pgvector cosine distance and tsvector ranking. Needs RAGFABRIC_TEST_DATABASE_URL."""

import os

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.db.migrate import downgrade, upgrade
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore

URL = os.environ.get("RAGFABRIC_TEST_DATABASE_URL", "")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not URL, reason="needs RAGFABRIC_TEST_DATABASE_URL")]


@pytest.fixture()
def pg():
    downgrade(URL); upgrade(URL)
    engine = create_engine(URL)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        c = Collection(name="c"); db.add(c); db.flush()
        d = Document(filename="a.txt", format="txt", collection_id=c.id, status="ready"); db.add(d); db.flush()
        rows = [Chunk(document_id=d.id, collection_id=c.id, chunk_index=i, page=1, char_start=0, char_end=1, text=t, embedding=[]) for i, t in enumerate(["annual leave is twelve days", "the cluster rollout failed", "leave carry forward capped"])]
        db.add_all(rows); db.commit()
        ids = [r.id for r in rows]
    yield factory, ids, d.id, c.id
    downgrade(URL)


def unit(v):
    a = np.asarray(v, dtype=float); return (a / np.linalg.norm(a)).tolist()


def test_pgvector_uses_the_vector_type_and_orders_by_cosine_distance(pg):
    factory, ids, d, c = pg
    store = PgVectorStore(factory)
    store.upsert(ids, [unit([1, 0]), unit([0, 1]), unit([0.8, 0.6])], [{"model": "m", "dim": 2, "document_id": d, "collection_id": c}] * 3)
    with factory() as db:
        typ = db.execute(text("select udt_name from information_schema.columns where table_name='chunk_embeddings' and column_name='embedding'")).scalar()
    assert typ == "vector"
    hits = store.query(unit([1, 0]), top_k=3, access=AccessFilter.unrestricted())
    assert [h.chunk_id for h in hits] == [ids[0], ids[2], ids[1]]


def test_fts_ranks_with_ts_rank_cd_and_uses_the_gin_index(pg):
    factory, ids, d, c = pg
    store = PostgresLexicalStore(factory)
    store.index(ids, ["annual leave is twelve days", "the cluster rollout failed", "leave carry forward capped"], [{"document_id": d, "collection_id": c}] * 3)
    with factory() as db:
        idx = db.execute(text("select indexdef from pg_indexes where tablename='chunk_search' and indexname='ix_chunk_search_tsv'")).scalar()
    assert idx and "gin" in idx.lower()
    hits = store.search("leave", top_k=5, access=AccessFilter.unrestricted())
    assert {h.chunk_id for h in hits} == {ids[0], ids[2]} and all(h.score is not None for h in hits)
```

```python
# packages/core/tests/test_caches.py
from ragfabric_core.config_file import CacheConfig
from ragfabric_core.stores.base import Cache
from ragfabric_core.stores.memory_cache import MemoryCache
from ragfabric_core.stores.registry import build_cache


def test_memory_cache_get_set_incr_and_ttl(monkeypatch):
    c = MemoryCache()
    assert isinstance(c, Cache)
    assert c.get("k") is None
    c.set("k", b"v", ttl_seconds=60)
    assert c.get("k") == b"v"
    assert c.incr("n", ttl_seconds=60) == 1 and c.incr("n") == 2
    now = [1000.0]
    monkeypatch.setattr("ragfabric_core.stores.memory_cache.time.monotonic", lambda: now[0])
    c.set("t", b"x", ttl_seconds=1)
    now[0] += 2
    assert c.get("t") is None


def test_registry_builds_memory_and_redis_kinds():
    assert isinstance(build_cache(CacheConfig(kind="memory")), MemoryCache)
    redis_cache = build_cache(CacheConfig(kind="redis"), url="redis://localhost:6379/9")
    assert redis_cache.name == "redis"
```

Add to `packages/core/tests/test_migrations.py` `EXPECTED_TABLES`: `"chunk_embeddings", "chunk_search"`.

Run `uv run pytest packages/core/tests/test_access_sql.py packages/core/tests/test_stores_sqlite.py packages/core/tests/test_caches.py packages/core/tests/test_migrations.py -q` → expect import errors and the table test failing.

- [ ] **Step 2: Models and migration additions**

```python
# packages/core/src/ragfabric_core/models/index.py
"""Index tables fed by the ingestion fan out.

chunk_embeddings holds one vector per chunk for the configured embedding model.
The column is JSON on SQLite (tests, development) and pgvector's VECTOR on
PostgreSQL, through a dialect variant, so one migration serves both. The
dimension is stored per row; an HNSW index needs a fixed dimension and is
created by Phase 3 once the deployment's embedding model is known.

chunk_search holds the tsvector for full text search on PostgreSQL (Text on
SQLite) with a GIN index. document_id and collection_id are copied here so the
access filter can be applied inside the index query without a join.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base

EmbeddingType = JSON().with_variant(Vector(), "postgresql")
TsvType = Text().with_variant(TSVECTOR(), "postgresql")


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    collection_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding = mapped_column(EmbeddingType, nullable=False)


class ChunkSearch(Base):
    __tablename__ = "chunk_search"
    __table_args__ = (Index("ix_chunk_search_tsv", "tsv", postgresql_using="gin"),)

    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    collection_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    tsv = mapped_column(TsvType, nullable=False)
```

Add `index` to the imports in `models/__init__.py` (`from ragfabric_core.models import access, document, evaluation, graph, index, runs, user`).

Extend `0003_ingestion_and_indexes.py` by hand (do not regenerate, so the Task 2 columns stay): in `upgrade()` add

```python
    embedding_type = sa.JSON().with_variant(Vector(), "postgresql")
    tsv_type = sa.Text().with_variant(TSVECTOR(), "postgresql")
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("collection_id", sa.Integer(), nullable=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("embedding", embedding_type, nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    with op.batch_alter_table("chunk_embeddings", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_chunk_embeddings_collection_id"), ["collection_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_chunk_embeddings_document_id"), ["document_id"], unique=False)
    op.create_table(
        "chunk_search",
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("collection_id", sa.Integer(), nullable=True),
        sa.Column("tsv", tsv_type, nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    with op.batch_alter_table("chunk_search", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_chunk_search_collection_id"), ["collection_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_chunk_search_document_id"), ["document_id"], unique=False)
        batch_op.create_index("ix_chunk_search_tsv", ["tsv"], unique=False, postgresql_using="gin")
```

with imports `from pgvector.sqlalchemy import Vector` and `from sqlalchemy.dialects.postgresql import TSVECTOR`, and at the very start of `upgrade()`:

```python
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
```

In `downgrade()` drop the indexes and tables in reverse (chunk_search first, then chunk_embeddings) before the Task 2 column drops. Run the drift test; if `compare_metadata` reports a type difference for the variant columns on SQLite, add `compare_type=False` only for those two tables by checking the `include_object` hook is not needed: the JSON/Text base types match on SQLite, so the diff should be empty.

- [ ] **Step 3: Implement access_clause, stores, caches, registry**

```python
# packages/core/src/ragfabric_core/stores/access_sql.py
"""Translate an AccessFilter into a SQL predicate applied inside the store query (ADR 0003)."""

from __future__ import annotations

from sqlalchemy import and_, false, not_, or_

from ragfabric_core.auth.principal import AccessFilter


def access_clause(access: AccessFilter, document_col, collection_col):
    if access.is_unrestricted:
        return None
    parts = []
    if access.document_ids is None and access.collection_ids is None:
        allow = None
    else:
        allows = []
        if access.document_ids is not None:
            allows.append(document_col.in_(sorted(access.document_ids)) if access.document_ids else false())
        if access.collection_ids is not None:
            allows.append(collection_col.in_(sorted(access.collection_ids)) if access.collection_ids else false())
        allow = or_(*allows) if len(allows) > 1 else allows[0]
    if allow is not None:
        parts.append(allow)
    if access.denied_document_ids:
        parts.append(not_(document_col.in_(sorted(access.denied_document_ids))))
    return and_(*parts) if len(parts) > 1 else parts[0]
```

```python
# packages/core/src/ragfabric_core/stores/pgvector_store.py
"""VectorStore over the chunk_embeddings table.

On PostgreSQL the ranking is pgvector's cosine distance operator (`<=>`), with
the access predicate in the same WHERE clause, so a restricted caller never
has a forbidden row ranked. On SQLite (tests, development without Postgres)
the same table holds JSON vectors and the ranking is a NumPy dot product over
the permitted rows; behaviour is identical, only speed differs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.models.index import ChunkEmbedding
from ragfabric_core.stores.access_sql import access_clause
from ragfabric_core.strategies.base import RetrievedChunk


def _to_chunk(row, score: float) -> RetrievedChunk:
    chunk, filename, fmt = row
    return RetrievedChunk(
        chunk_id=chunk.id, document_id=chunk.document_id, collection_id=chunk.collection_id,
        text=chunk.text, page=chunk.page, section=chunk.section, score=score,
        char_start=chunk.char_start, char_end=chunk.char_end,
        metadata={"filename": filename, "format": fmt},
    )


class PgVectorStore:
    name = "pgvector"

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._sf = session_factory

    def upsert(self, chunk_ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
        with self._sf() as db:
            for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads, strict=True):
                row = db.get(ChunkEmbedding, chunk_id)
                values = dict(
                    document_id=payload["document_id"], collection_id=payload.get("collection_id"),
                    model=payload["model"], dim=payload.get("dim", len(vector)), embedding=list(map(float, vector)),
                )
                if row is None:
                    db.add(ChunkEmbedding(chunk_id=chunk_id, **values))
                else:
                    for k, v in values.items():
                        setattr(row, k, v)
            db.commit()

    def query(self, vector: list[float], top_k: int, access: AccessFilter, filters: dict | None = None) -> list[RetrievedChunk]:
        clause = access_clause(access, ChunkEmbedding.document_id, ChunkEmbedding.collection_id)
        with self._sf() as db:
            base = (
                select(Chunk, Document.filename, Document.format, ChunkEmbedding.embedding)
                .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
                .join(Document, Document.id == Chunk.document_id)
            )
            if clause is not None:
                base = base.where(clause)
            for key, value in (filters or {}).items():
                if key == "document_id":
                    base = base.where(ChunkEmbedding.document_id == value)
                elif key == "collection_id":
                    base = base.where(ChunkEmbedding.collection_id == value)
                elif key == "format":
                    base = base.where(Document.format == value)
            if db.bind.dialect.name == "postgresql":
                distance = ChunkEmbedding.embedding.cosine_distance(list(map(float, vector)))
                rows = db.execute(base.add_columns(distance.label("distance")).order_by(distance, Chunk.id).limit(top_k)).all()
                return [_to_chunk(r[:3], 1.0 - float(r[4])) for r in rows]
            rows = db.execute(base).all()
            if not rows:
                return []
            q = np.asarray(vector, dtype=np.float32)
            scored = []
            for r in rows:
                emb = np.asarray(r[3], dtype=np.float32)
                score = float(emb @ q) if emb.shape == q.shape else -1.0
                scored.append((score, r))
            scored.sort(key=lambda item: (-item[0], item[1][0].id))
            return [_to_chunk(r[:3], s) for s, r in scored[:top_k]]

    def delete_document(self, document_id: int) -> None:
        with self._sf() as db:
            db.execute(delete(ChunkEmbedding).where(ChunkEmbedding.document_id == document_id))
            db.commit()

    def count(self) -> int:
        with self._sf() as db:
            return int(db.execute(select(func.count()).select_from(ChunkEmbedding)).scalar() or 0)
```

```python
# packages/core/src/ragfabric_core/stores/postgres_fts.py
"""LexicalStore over the chunk_search table.

PostgreSQL: to_tsvector('english') at index time, plainto_tsquery at query
time, ts_rank_cd for ordering, a GIN index for speed, the access predicate in
the WHERE clause. Other dialects: a token overlap score computed in Python
over the permitted rows, so unit tests and SQLite development keep working.
Phase 4 adds BM25 and fusion on top; this store is the fan out target.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.models.index import ChunkSearch
from ragfabric_core.stores.access_sql import access_clause
from ragfabric_core.strategies.base import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _to_chunk(chunk: Chunk, filename: str, fmt: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.id, document_id=chunk.document_id, collection_id=chunk.collection_id,
        text=chunk.text, page=chunk.page, section=chunk.section, score=score,
        char_start=chunk.char_start, char_end=chunk.char_end, metadata={"filename": filename, "format": fmt},
    )


class PostgresLexicalStore:
    name = "postgres_fts"

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._sf = session_factory

    def index(self, chunk_ids: list[int], texts: list[str], payloads: list[dict]) -> None:
        with self._sf() as db:
            postgres = db.bind.dialect.name == "postgresql"
            for chunk_id, text, payload in zip(chunk_ids, texts, payloads, strict=True):
                tsv = func.to_tsvector("english", text) if postgres else " ".join(sorted(_tokens(text)))
                row = db.get(ChunkSearch, chunk_id)
                if row is None:
                    db.add(ChunkSearch(chunk_id=chunk_id, document_id=payload["document_id"], collection_id=payload.get("collection_id"), tsv=tsv))
                else:
                    row.document_id, row.collection_id, row.tsv = payload["document_id"], payload.get("collection_id"), tsv
            db.commit()

    def search(self, query: str, top_k: int, access: AccessFilter, filters: dict | None = None) -> list[RetrievedChunk]:
        clause = access_clause(access, ChunkSearch.document_id, ChunkSearch.collection_id)
        with self._sf() as db:
            base = (
                select(Chunk, Document.filename, Document.format, ChunkSearch.tsv)
                .join(ChunkSearch, ChunkSearch.chunk_id == Chunk.id)
                .join(Document, Document.id == Chunk.document_id)
            )
            if clause is not None:
                base = base.where(clause)
            for key, value in (filters or {}).items():
                if key == "document_id":
                    base = base.where(ChunkSearch.document_id == value)
                elif key == "collection_id":
                    base = base.where(ChunkSearch.collection_id == value)
                elif key == "format":
                    base = base.where(Document.format == value)
            if db.bind.dialect.name == "postgresql":
                tsq = func.plainto_tsquery("english", query)
                rank = func.ts_rank_cd(ChunkSearch.tsv, tsq)
                rows = db.execute(base.add_columns(rank.label("rank")).where(ChunkSearch.tsv.op("@@")(tsq)).order_by(rank.desc(), Chunk.id).limit(top_k)).all()
                return [_to_chunk(r[0], r[1], r[2], float(r[4])) for r in rows]
            q = _tokens(query)
            if not q:
                return []
            scored = []
            for chunk, filename, fmt, tsv in db.execute(base).all():
                overlap = len(q & set(str(tsv).split()))
                if overlap:
                    scored.append((overlap / len(q), chunk, filename, fmt))
            scored.sort(key=lambda item: (-item[0], item[1].id))
            return [_to_chunk(c, f, m, s) for s, c, f, m in scored[:top_k]]

    def delete_document(self, document_id: int) -> None:
        with self._sf() as db:
            db.execute(delete(ChunkSearch).where(ChunkSearch.document_id == document_id))
            db.commit()
```

```python
# packages/core/src/ragfabric_core/stores/memory_cache.py
"""In process Cache for tests and single process development."""

from __future__ import annotations

import time


class MemoryCache:
    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, float | None]] = {}

    def _live(self, key: str) -> bytes | None:
        item = self._data.get(key)
        if item is None:
            return None
        value, expires = item
        if expires is not None and time.monotonic() >= expires:
            del self._data[key]
            return None
        return value

    def get(self, key: str) -> bytes | None:
        return self._live(key)

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        self._data[key] = (value, time.monotonic() + ttl_seconds if ttl_seconds else None)

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        current = self._live(key)
        new = (int(current) if current else 0) + 1
        expires = self._data[key][1] if current is not None else (time.monotonic() + ttl_seconds if ttl_seconds else None)
        self._data[key] = (str(new).encode(), expires)
        return new
```

```python
# packages/core/src/ragfabric_core/stores/redis_cache.py
"""Cache over Redis: embeddings cache, answer cache and rate limit counters."""

from __future__ import annotations

from typing import Any


class RedisCache:
    name = "redis"

    def __init__(self, url: str, client: Any | None = None) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(url)
        self._r = client

    def get(self, key: str) -> bytes | None:
        value = self._r.get(key)
        return bytes(value) if value is not None else None

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        self._r.set(key, value, ex=ttl_seconds)

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        new = int(self._r.incr(key))
        if new == 1 and ttl_seconds:
            self._r.expire(key, ttl_seconds)
        return new
```

```python
# packages/core/src/ragfabric_core/stores/registry.py
"""Build stores and caches from configuration. The only place store kinds are switched on."""

from __future__ import annotations

import os
from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.config_file import CacheConfig, LexicalStoreConfig, VectorStoreConfig
from ragfabric_core.stores.base import Cache, LexicalStore, VectorStore
from ragfabric_core.stores.memory_cache import MemoryCache
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.stores.redis_cache import RedisCache


def build_vector_store(cfg: VectorStoreConfig, session_factory: Callable[[], Session]) -> VectorStore:
    if cfg.kind in ("pgvector", "memory"):
        # "memory" keeps the v1 in process index for queries in this phase; the
        # fan out still writes to the relational table so nothing is lost.
        return PgVectorStore(session_factory)
    raise NotImplementedError(f"vector store {cfg.kind!r} arrives in Phase 3")


def build_lexical_store(cfg: LexicalStoreConfig, session_factory: Callable[[], Session]) -> LexicalStore:
    if cfg.kind == "postgres_fts":
        return PostgresLexicalStore(session_factory)
    raise NotImplementedError(f"lexical store {cfg.kind!r} arrives in Phase 4")


def build_cache(cfg: CacheConfig, url: str | None = None) -> Cache:
    if cfg.kind == "redis":
        return RedisCache(url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    return MemoryCache()
```

- [ ] **Step 4: Run SQLite tests, then the PostgreSQL integration tests on OrbStack, then commit**

```bash
uv run pytest packages/core/tests/test_access_sql.py packages/core/tests/test_stores_sqlite.py packages/core/tests/test_caches.py packages/core/tests/test_migrations.py -q
docker run -d --rm --name rf-pg-p2 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragfabric -p 127.0.0.1:55432:5432 pgvector/pgvector:0.8.6-pg18 >/dev/null && sleep 6
RAGFABRIC_TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:55432/ragfabric JWT_SECRET=dev-only-secret-not-for-production-use-1234567890 uv run pytest packages/core/tests/test_stores_postgres.py -m integration -q
docker stop rf-pg-p2 >/dev/null
uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat(core): add pgvector and PostgreSQL full text stores with access aware queries, caches and migration 0003 index tables" && git log -1 --format='%h %(trailers)'
```

Expected: 2 integration tests pass on PostgreSQL; the full suite passes with the SQLite paths.

---

### Task 5: Job queue, workers, and the indexing fan out

**Files:**
- Create: `packages/core/src/ragfabric_core/queue/__init__.py`, `queue/base.py`, `queue/memory_queue.py`, `queue/redis_queue.py`, `queue/registry.py`, `workers/__init__.py`, `workers/handlers.py`, `workers/runner.py`, `ingest/indexing.py`
- Modify: `ingest/pipeline.py`
- Test: `packages/core/tests/test_queue.py`, `test_indexing.py`

**Interfaces:**
- Produces: `Job(id: str, kind: str, payload: dict, attempts: int = 0)`; `JobQueue` protocol: `enqueue(job) -> None`, `dequeue(timeout_seconds: float = 1.0) -> Job | None`, `size() -> int`; `MemoryJobQueue()` also has `drain(handler: Callable[[Job], None]) -> int`; `RedisJobQueue(url, key="ragfabric:jobs", client=None)`; `build_queue(cfg: RagFabricConfig) -> JobQueue | None` (None when `ingestion.indexing == "inline"`).
- `handlers.index_document(db, document_id, *, embedding_provider, vector_store, lexical_store) -> int`; `handlers.extract_graph(db, document_id) -> None` (records a `queued_for_phase_6` note in the document error field? No: it logs and returns; nothing is written); `Worker(queue, session_factory, handlers: dict[str, Callable[[Session, Job], None]])` with `run_once(timeout_seconds=1.0) -> bool` and `run_forever(stop: threading.Event)`.
- `indexing.index_inline(db, document) -> int` and `indexing.schedule_indexing(db, document, queue: JobQueue | None) -> str` returning the resulting status (`ready` or `indexing`).
- Document status values: `processing`, `indexing`, `ready`, `failed`.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_queue.py
from types import SimpleNamespace

from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.queue.base import Job, JobQueue
from ragfabric_core.queue.memory_queue import MemoryJobQueue
from ragfabric_core.queue.redis_queue import RedisJobQueue
from ragfabric_core.queue.registry import build_queue


def test_memory_queue_is_fifo_and_drains():
    q = MemoryJobQueue()
    assert isinstance(q, JobQueue)
    q.enqueue(Job(id="1", kind="a", payload={}))
    q.enqueue(Job(id="2", kind="b", payload={"x": 1}))
    assert q.size() == 2
    seen = []
    assert q.drain(lambda job: seen.append(job.kind)) == 2
    assert seen == ["a", "b"] and q.size() == 0 and q.dequeue(timeout_seconds=0) is None


def test_redis_queue_serialises_jobs():
    calls = []

    class FakeRedis:
        def __init__(self):
            self.items = []
        def rpush(self, key, value):
            calls.append(("rpush", key)); self.items.append(value)
        def blpop(self, keys, timeout):
            return (keys[0], self.items.pop(0)) if self.items else None
        def llen(self, key):
            return len(self.items)

    q = RedisJobQueue("redis://unused", client=FakeRedis())
    q.enqueue(Job(id="j1", kind="index_document", payload={"document_id": 5}))
    assert q.size() == 1 and calls[0] == ("rpush", "ragfabric:jobs")
    job = q.dequeue(timeout_seconds=1)
    assert job.id == "j1" and job.payload == {"document_id": 5} and job.attempts == 0
    assert q.dequeue(timeout_seconds=0) is None


def test_build_queue_returns_none_for_inline_and_redis_queue_for_queue_mode(monkeypatch):
    assert build_queue(RagFabricConfig()) is None
    cfg = RagFabricConfig.model_validate({"ingestion": {"indexing": "queue"}})
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/9")
    q = build_queue(cfg)
    assert isinstance(q, RedisJobQueue)
```

```python
# packages/core/tests/test_indexing.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.models import Base
from ragfabric_core.models.document import Chunk, Collection, Document
from ragfabric_core.models.index import ChunkEmbedding, ChunkSearch
from ragfabric_core.providers.offline import HashingEmbeddingProvider
from ragfabric_core.queue.base import Job
from ragfabric_core.queue.memory_queue import MemoryJobQueue
from ragfabric_core.stores.pgvector_store import PgVectorStore
from ragfabric_core.stores.postgres_fts import PostgresLexicalStore
from ragfabric_core.workers.handlers import index_document
from ragfabric_core.workers.runner import Worker, default_handlers
from ragfabric_core.ingest.indexing import schedule_indexing


@pytest.fixture()
def db_and_doc(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'w.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        c = Collection(name="c"); db.add(c); db.flush()
        d = Document(filename="a.txt", format="txt", collection_id=c.id, status="processing")
        db.add(d); db.flush()
        db.add_all([Chunk(document_id=d.id, collection_id=c.id, chunk_index=i, page=1, char_start=0, char_end=5, text=t, embedding=[]) for i, t in enumerate(["leave policy", "rollout guide"])])
        db.commit()
        doc_id = d.id
    return factory, doc_id


def test_index_document_writes_both_indexes_and_marks_ready(db_and_doc):
    factory, doc_id = db_and_doc
    with factory() as db:
        n = index_document(db, doc_id, embedding_provider=HashingEmbeddingProvider(dim=16), vector_store=PgVectorStore(factory), lexical_store=PostgresLexicalStore(factory))
        assert n == 2
        assert db.query(ChunkEmbedding).count() == 2 and db.query(ChunkSearch).count() == 2
        assert db.get(Document, doc_id).status == "ready"
        assert {e.model for e in db.query(ChunkEmbedding)} == {"hashing-16"}


def test_schedule_indexing_inline_runs_now_and_queue_mode_enqueues(db_and_doc, monkeypatch):
    factory, doc_id = db_and_doc
    monkeypatch.setattr("ragfabric_core.ingest.indexing._embedding_provider", lambda: HashingEmbeddingProvider(dim=16))
    monkeypatch.setattr("ragfabric_core.ingest.indexing._stores", lambda: (PgVectorStore(factory), PostgresLexicalStore(factory)))
    with factory() as db:
        doc = db.get(Document, doc_id)
        assert schedule_indexing(db, doc, queue=None) == "ready"
        assert db.query(ChunkEmbedding).count() == 2
    q = MemoryJobQueue()
    with factory() as db:
        doc = db.get(Document, doc_id)
        assert schedule_indexing(db, doc, queue=q) == "indexing"
        assert db.get(Document, doc_id).status == "indexing"
    kinds = [j.kind for j in list(q._items)]
    assert kinds == ["index_document", "extract_graph"]


def test_worker_runs_jobs_and_survives_a_failing_handler(db_and_doc, monkeypatch):
    factory, doc_id = db_and_doc
    q = MemoryJobQueue()
    q.enqueue(Job(id="bad", kind="explode", payload={}))
    q.enqueue(Job(id="ok", kind="index_document", payload={"document_id": doc_id}))
    handlers = default_handlers(embedding_provider=HashingEmbeddingProvider(dim=16), vector_store=PgVectorStore(factory), lexical_store=PostgresLexicalStore(factory))
    def explode(db, job):
        raise RuntimeError("boom")
    handlers["explode"] = explode
    w = Worker(q, factory, handlers)
    assert w.run_once(timeout_seconds=0) is True  # bad job consumed, error logged, worker alive
    assert w.run_once(timeout_seconds=0) is True
    assert w.run_once(timeout_seconds=0) is False
    with factory() as db:
        assert db.get(Document, doc_id).status == "ready"
    assert w.failed == 1 and w.processed == 2
```

Run → import errors.

- [ ] **Step 2: Implement the queue**

```python
# packages/core/src/ragfabric_core/queue/__init__.py
"""Background jobs: a small FIFO over Redis, with an in memory twin for tests."""
```

```python
# packages/core/src/ragfabric_core/queue/base.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Job(BaseModel):
    id: str
    kind: str
    payload: dict = Field(default_factory=dict)
    attempts: int = 0


@runtime_checkable
class JobQueue(Protocol):
    def enqueue(self, job: Job) -> None: ...
    def dequeue(self, timeout_seconds: float = 1.0) -> Job | None: ...
    def size(self) -> int: ...
```

```python
# packages/core/src/ragfabric_core/queue/memory_queue.py
from __future__ import annotations

from collections import deque
from collections.abc import Callable

from ragfabric_core.queue.base import Job


class MemoryJobQueue:
    def __init__(self) -> None:
        self._items: deque[Job] = deque()

    def enqueue(self, job: Job) -> None:
        self._items.append(job)

    def dequeue(self, timeout_seconds: float = 1.0) -> Job | None:
        return self._items.popleft() if self._items else None

    def size(self) -> int:
        return len(self._items)

    def drain(self, handler: Callable[[Job], None]) -> int:
        n = 0
        while self._items:
            handler(self._items.popleft())
            n += 1
        return n
```

```python
# packages/core/src/ragfabric_core/queue/redis_queue.py
"""FIFO job queue on a Redis list: RPUSH to enqueue, BLPOP to dequeue.

Why not Celery: one list and two commands cover ingestion fan out, and the
worker is a plain loop that any adopter can read in a minute. Retries and dead
letters arrive when a later phase needs them; the Job carries `attempts` so
that change is additive.
"""

from __future__ import annotations

from typing import Any

from ragfabric_core.queue.base import Job


class RedisJobQueue:
    def __init__(self, url: str, key: str = "ragfabric:jobs", client: Any | None = None) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(url)
        self._r = client
        self.key = key

    def enqueue(self, job: Job) -> None:
        self._r.rpush(self.key, job.model_dump_json())

    def dequeue(self, timeout_seconds: float = 1.0) -> Job | None:
        item = self._r.blpop([self.key], timeout=max(int(timeout_seconds), 0)) if timeout_seconds > 0 else None
        if item is None and timeout_seconds <= 0:
            raw = self._r.blpop([self.key], timeout=1) if self._r.llen(self.key) else None
            item = raw
        if item is None:
            return None
        _, raw = item
        return Job.model_validate_json(raw if isinstance(raw, str) else raw.decode("utf-8"))

    def size(self) -> int:
        return int(self._r.llen(self.key))
```

```python
# packages/core/src/ragfabric_core/queue/registry.py
from __future__ import annotations

import os

from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.queue.base import JobQueue
from ragfabric_core.queue.redis_queue import RedisJobQueue


def build_queue(cfg: RagFabricConfig) -> JobQueue | None:
    if cfg.ingestion.indexing != "queue":
        return None
    return RedisJobQueue(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
```

- [ ] **Step 3: Implement handlers, runner and indexing orchestration**

```python
# packages/core/src/ragfabric_core/workers/__init__.py
"""Background workers for the ingestion fan out."""
```

```python
# packages/core/src/ragfabric_core/workers/handlers.py
"""Job handlers. Each takes a Session and does one document's worth of work.

index_document embeds the document's chunks with the configured provider and
writes the vector and lexical indexes. It marks the document ready only after
both writes succeed, so a document is never searchable in one index and
missing from the other. extract_graph is the Phase 6 hook; here it only
records that the job ran.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.providers.base import EmbeddingProvider
from ragfabric_core.stores.base import LexicalStore, VectorStore

log = logging.getLogger(__name__)


def index_document(
    db: Session,
    document_id: int,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    lexical_store: LexicalStore,
) -> int:
    document = db.get(Document, document_id)
    if document is None:
        log.warning("index_document: document %s no longer exists", document_id)
        return 0
    chunks = db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index).all()
    if not chunks:
        document.status = "ready"
        db.commit()
        return 0
    result = embedding_provider.embed([c.text for c in chunks])
    ids = [c.id for c in chunks]
    payloads = [
        {"document_id": c.document_id, "collection_id": c.collection_id, "model": result.model, "dim": embedding_provider.dim}
        for c in chunks
    ]
    vector_store.upsert(ids, result.vectors, payloads)
    lexical_store.index(ids, [c.text for c in chunks], payloads)
    document = db.get(Document, document_id)
    document.status = "ready"
    document.error = ""
    db.commit()
    return len(chunks)


def extract_graph(db: Session, document_id: int) -> None:
    log.info("extract_graph: document %s queued; graph extraction arrives in Phase 6", document_id)
```

```python
# packages/core/src/ragfabric_core/workers/runner.py
"""The worker loop: dequeue, dispatch by kind, commit or log, repeat."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.providers.base import EmbeddingProvider
from ragfabric_core.queue.base import Job, JobQueue
from ragfabric_core.stores.base import LexicalStore, VectorStore
from ragfabric_core.workers import handlers

log = logging.getLogger(__name__)

Handler = Callable[[Session, Job], None]


def default_handlers(
    *, embedding_provider: EmbeddingProvider, vector_store: VectorStore, lexical_store: LexicalStore
) -> dict[str, Handler]:
    def _index(db: Session, job: Job) -> None:
        handlers.index_document(
            db, int(job.payload["document_id"]), embedding_provider=embedding_provider,
            vector_store=vector_store, lexical_store=lexical_store,
        )

    def _graph(db: Session, job: Job) -> None:
        handlers.extract_graph(db, int(job.payload["document_id"]))

    return {"index_document": _index, "extract_graph": _graph}


class Worker:
    def __init__(self, queue: JobQueue, session_factory: Callable[[], Session], handlers_by_kind: dict[str, Handler]) -> None:
        self.queue = queue
        self._sf = session_factory
        self.handlers = handlers_by_kind
        self.processed = 0
        self.failed = 0

    def run_once(self, timeout_seconds: float = 1.0) -> bool:
        job = self.queue.dequeue(timeout_seconds=timeout_seconds)
        if job is None:
            return False
        handler = self.handlers.get(job.kind)
        with self._sf() as db:
            try:
                if handler is None:
                    raise KeyError(f"no handler for job kind {job.kind!r}")
                handler(db, job)
                self.processed += 1
            except Exception:
                db.rollback()
                self.failed += 1
                self.processed += 1
                log.exception("job %s (%s) failed", job.id, job.kind)
        return True

    def run_forever(self, stop: threading.Event, timeout_seconds: float = 1.0) -> None:
        while not stop.is_set():
            self.run_once(timeout_seconds=timeout_seconds)
```

```python
# packages/core/src/ragfabric_core/ingest/indexing.py
"""Decide whether a freshly chunked document is indexed now or by a worker."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from ragfabric_core.models.document import Document
from ragfabric_core.providers.registry import build_embedding_provider
from ragfabric_core.queue.base import Job, JobQueue
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store
from ragfabric_core.workers.handlers import index_document


def _embedding_provider():
    return build_embedding_provider(get_config().embeddings)


def _stores():
    cfg = get_config()
    sf = get_session_factory()
    return build_vector_store(cfg.vector_store, sf), build_lexical_store(cfg.lexical_store, sf)


def index_inline(db: Session, document: Document) -> int:
    vector_store, lexical_store = _stores()
    return index_document(db, document.id, embedding_provider=_embedding_provider(), vector_store=vector_store, lexical_store=lexical_store)


def schedule_indexing(db: Session, document: Document, queue: JobQueue | None) -> str:
    if queue is None:
        index_inline(db, document)
        return "ready"
    document.status = "indexing"
    db.commit()
    for kind in ("index_document", "extract_graph"):
        queue.enqueue(Job(id=uuid.uuid4().hex, kind=kind, payload={"document_id": document.id}))
    return "indexing"
```

Note for the implementer on `test_schedule_indexing_inline_runs_now_and_queue_mode_enqueues`: it monkeypatches `_embedding_provider` and `_stores` in `ragfabric_core.ingest.indexing`; keep those two module level functions with exactly those names.

- [ ] **Step 4: Wire the pipeline**

In `ingest/pipeline.py` `_index_content`, after the successful `db.commit()` and the v1 `get_store().upsert(store_records)` (the v1 in memory index stays the live query path this phase), add:

```python
    from ragfabric_core.ingest.indexing import schedule_indexing
    from ragfabric_core.queue.registry import build_queue

    document.status = schedule_indexing(db, document, build_queue(get_config()))
    db.commit()
    db.refresh(document)
```

Keep the v1 embedding into `chunks.embedding` as is. Add a module docstring paragraph: "Two indexes are written during Phase 2: the v1 in memory index (still the live query path) and the pgvector plus full text tables that Phases 3 and 4 will query. The duplication ends when Phase 3 retires the in memory index."

In the server tests, the default config file written by `conftest.py` (Task 3) has `indexing` unset, so `inline` applies and the upload tests keep returning `status == "ready"`. The server `test_api.py` upload test asserts `doc["status"] == "ready"`; confirm it still passes.

- [ ] **Step 5: Run, lint, commit**

```bash
uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat(core): add the job queue, indexing workers and the ingestion fan out" && git log -1 --format='%h %(trailers)'
```

---

### Task 6: Access control services and the access policy

**Files:**
- Create: `packages/core/src/ragfabric_core/auth/service.py`, `auth/policy.py`
- Test: `packages/core/tests/test_access_policy.py`

**Interfaces:**
- Produces (`auth/service.py`): `create_group(db, name, description="") -> Group`; `add_member(db, group_id, user_id) -> None`; `remove_member(db, group_id, user_id) -> None`; `grant_collection(db, group_id, collection_id, permission="read") -> CollectionGrant` (upsert on the unique pair); `revoke_collection(db, group_id, collection_id) -> None`; `set_document_override(db, document_id, *, group_id=None, user_id=None, permission="deny") -> DocumentOverride`; `group_ids_for_user(db, user_id) -> list[int]`.
- Produces (`auth/policy.py`): `compute_access_filter(db, principal: Principal) -> AccessFilter` with these rules, in order:
  1. `principal.role == "admin"` returns `AccessFilter.unrestricted()`.
  2. Allowed collections = collections with a grant to any of `principal.group_ids`, plus collections owned by `principal.user_id`, plus collections that have no grants at all (open by default; documented as the v1 compatible rule).
  3. Allowed documents = documents owned by `principal.user_id`, plus documents with a `read` override for the user or the user's groups, plus documents with `collection_id IS NULL` (uncollected uploads stay open, as in v1).
  4. Denied documents = documents with a `deny` override for the user or the user's groups.
  5. If the principal came from an API key with non empty `collection_ids` scopes, allowed collections are intersected with the scopes and allowed documents are reduced to those whose collection is in scope.
  The result always has all three fields set to frozensets (never None) for non admins, so the filter is never accidentally unrestricted.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_access_policy.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.policy import compute_access_filter
from ragfabric_core.auth.principal import Principal
from ragfabric_core.auth import service
from ragfabric_core.models import Base
from ragfabric_core.models.access import ApiKey
from ragfabric_core.models.document import Collection, Document
from ragfabric_core.models.user import User


@pytest.fixture()
def world(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'p.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    alice, bob, admin = User(email="a@x", hashed_password="h"), User(email="b@x", hashed_password="h"), User(email="root@x", hashed_password="h", role="admin")
    db.add_all([alice, bob, admin]); db.flush()
    open_c = Collection(name="open", owner_id=bob.id)
    hr = Collection(name="hr", owner_id=bob.id)
    mine = Collection(name="mine", owner_id=alice.id)
    db.add_all([open_c, hr, mine]); db.flush()
    docs = {
        "open": Document(filename="o.txt", format="txt", collection_id=open_c.id, owner_id=bob.id, status="ready"),
        "hr": Document(filename="h.txt", format="txt", collection_id=hr.id, owner_id=bob.id, status="ready"),
        "hr_secret": Document(filename="s.txt", format="txt", collection_id=hr.id, owner_id=bob.id, status="ready"),
        "mine": Document(filename="m.txt", format="txt", collection_id=mine.id, owner_id=alice.id, status="ready"),
        "loose": Document(filename="l.txt", format="txt", collection_id=None, owner_id=bob.id, status="ready"),
    }
    db.add_all(docs.values()); db.flush()
    hr_group = service.create_group(db, "hr-team")
    service.grant_collection(db, hr_group.id, hr.id, "read")
    db.commit()
    yield db, dict(alice=alice, bob=bob, admin=admin), dict(open=open_c, hr=hr, mine=mine), docs, hr_group
    db.close()


def principal(user, groups=(), api_key_id=None):
    return Principal(user_id=user.id, email=user.email, role=user.role, group_ids=list(groups), api_key_id=api_key_id)


def test_admin_is_unrestricted(world):
    db, users, *_ = world
    assert compute_access_filter(db, principal(users["admin"])).is_unrestricted


def test_user_without_grant_sees_open_collections_own_collection_and_loose_documents_only(world):
    db, users, cols, docs, _ = world
    f = compute_access_filter(db, principal(users["alice"]))
    assert not f.is_unrestricted
    assert f.collection_ids == frozenset({cols["open"].id, cols["mine"].id})
    assert docs["loose"].id in f.document_ids and docs["mine"].id in f.document_ids
    assert f.allows(docs["hr"].id, cols["hr"].id) is False
    assert f.allows(docs["open"].id, cols["open"].id) is True


def test_group_grant_opens_the_collection(world):
    db, users, cols, docs, hr_group = world
    service.add_member(db, hr_group.id, users["alice"].id); db.commit()
    groups = service.group_ids_for_user(db, users["alice"].id)
    f = compute_access_filter(db, principal(users["alice"], groups))
    assert cols["hr"].id in f.collection_ids
    assert f.allows(docs["hr"].id, cols["hr"].id)


def test_deny_override_beats_a_collection_grant(world):
    db, users, cols, docs, hr_group = world
    service.add_member(db, hr_group.id, users["alice"].id)
    service.set_document_override(db, docs["hr_secret"].id, user_id=users["alice"].id, permission="deny"); db.commit()
    f = compute_access_filter(db, principal(users["alice"], service.group_ids_for_user(db, users["alice"].id)))
    assert f.allows(docs["hr"].id, cols["hr"].id) and not f.allows(docs["hr_secret"].id, cols["hr"].id)


def test_read_override_opens_one_document_in_a_restricted_collection(world):
    db, users, cols, docs, _ = world
    service.set_document_override(db, docs["hr"].id, user_id=users["alice"].id, permission="read"); db.commit()
    f = compute_access_filter(db, principal(users["alice"]))
    assert f.allows(docs["hr"].id, cols["hr"].id) and not f.allows(docs["hr_secret"].id, cols["hr"].id)


def test_api_key_scopes_intersect(world):
    db, users, cols, docs, hr_group = world
    service.add_member(db, hr_group.id, users["alice"].id)
    key = ApiKey(name="k", key_prefix="rf_abc", key_hash="x", principal_user_id=users["alice"].id, collection_ids=[cols["hr"].id], strategies=[])
    db.add(key); db.commit()
    f = compute_access_filter(db, principal(users["alice"], service.group_ids_for_user(db, users["alice"].id), api_key_id=key.id))
    assert f.collection_ids == frozenset({cols["hr"].id})
    assert docs["mine"].id not in f.document_ids and docs["loose"].id not in f.document_ids


def test_grant_upsert_and_revoke(world):
    db, users, cols, docs, hr_group = world
    g1 = service.grant_collection(db, hr_group.id, cols["hr"].id, "write")
    assert g1.permission == "write"
    service.revoke_collection(db, hr_group.id, cols["hr"].id); db.commit()
    f = compute_access_filter(db, principal(users["alice"], [hr_group.id]))
    assert cols["hr"].id in f.collection_ids  # no grants left, so the collection is open again
```

Run → import errors.

- [ ] **Step 2: Implement the service**

```python
# packages/core/src/ragfabric_core/auth/service.py
"""Groups, membership, collection grants and document overrides."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragfabric_core.models.access import CollectionGrant, DocumentOverride, Group, GroupMember

VALID_GRANTS = ("read", "write")
VALID_OVERRIDES = ("deny", "read")


def create_group(db: Session, name: str, description: str = "") -> Group:
    group = Group(name=name, description=description)
    db.add(group)
    db.flush()
    return group


def add_member(db: Session, group_id: int, user_id: int) -> None:
    if db.get(GroupMember, (group_id, user_id)) is None:
        db.add(GroupMember(group_id=group_id, user_id=user_id))
        db.flush()


def remove_member(db: Session, group_id: int, user_id: int) -> None:
    member = db.get(GroupMember, (group_id, user_id))
    if member is not None:
        db.delete(member)
        db.flush()


def grant_collection(db: Session, group_id: int, collection_id: int, permission: str = "read") -> CollectionGrant:
    if permission not in VALID_GRANTS:
        raise ValueError(f"permission must be one of {VALID_GRANTS}")
    grant = db.execute(
        select(CollectionGrant).where(CollectionGrant.group_id == group_id, CollectionGrant.collection_id == collection_id)
    ).scalar_one_or_none()
    if grant is None:
        grant = CollectionGrant(group_id=group_id, collection_id=collection_id, permission=permission)
        db.add(grant)
    else:
        grant.permission = permission
    db.flush()
    return grant


def revoke_collection(db: Session, group_id: int, collection_id: int) -> None:
    grant = db.execute(
        select(CollectionGrant).where(CollectionGrant.group_id == group_id, CollectionGrant.collection_id == collection_id)
    ).scalar_one_or_none()
    if grant is not None:
        db.delete(grant)
        db.flush()


def set_document_override(
    db: Session, document_id: int, *, group_id: int | None = None, user_id: int | None = None, permission: str = "deny"
) -> DocumentOverride:
    if permission not in VALID_OVERRIDES:
        raise ValueError(f"permission must be one of {VALID_OVERRIDES}")
    if (group_id is None) == (user_id is None):
        raise ValueError("exactly one of group_id or user_id is required")
    override = DocumentOverride(document_id=document_id, group_id=group_id, user_id=user_id, permission=permission)
    db.add(override)
    db.flush()
    return override


def group_ids_for_user(db: Session, user_id: int) -> list[int]:
    return list(db.execute(select(GroupMember.group_id).where(GroupMember.user_id == user_id)).scalars())
```

- [ ] **Step 3: Implement the policy**

```python
# packages/core/src/ragfabric_core/auth/policy.py
"""Compute the AccessFilter for a principal (ADR 0003).

The filter is computed once per request from grants and overrides and then
passed into every store query, so ranking only ever sees permitted rows.

Rules, applied in order:
1. Admins are unrestricted.
2. A collection is readable when one of the principal's groups holds a grant
   on it, when the principal owns it, or when it has no grants at all. The
   last rule keeps v1 behaviour (everyone sees everything) until an admin
   adds the first grant to a collection, at which point that collection is
   restricted to its grantees.
3. A document is readable when the principal owns it, when a read override
   names the principal or one of their groups, or when it belongs to no
   collection.
4. A deny override for the principal or one of their groups always wins.
5. An API key with collection scopes narrows everything above to those
   collections.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.models.access import ApiKey, CollectionGrant, DocumentOverride
from ragfabric_core.models.document import Collection, Document


def compute_access_filter(db: Session, principal: Principal) -> AccessFilter:
    if principal.role == "admin":
        return AccessFilter.unrestricted()

    groups = list(principal.group_ids)
    granted_ids = set(db.execute(select(CollectionGrant.collection_id).distinct()).scalars())
    allowed_collections: set[int] = set()
    if groups:
        allowed_collections |= set(
            db.execute(select(CollectionGrant.collection_id).where(CollectionGrant.group_id.in_(groups))).scalars()
        )
    if principal.user_id is not None:
        allowed_collections |= set(
            db.execute(select(Collection.id).where(Collection.owner_id == principal.user_id)).scalars()
        )
    all_collections = set(db.execute(select(Collection.id)).scalars())
    allowed_collections |= all_collections - granted_ids

    allowed_documents: set[int] = set(
        db.execute(select(Document.id).where(Document.collection_id.is_(None))).scalars()
    )
    if principal.user_id is not None:
        allowed_documents |= set(
            db.execute(select(Document.id).where(Document.owner_id == principal.user_id)).scalars()
        )

    override_filter = []
    if principal.user_id is not None:
        override_filter.append(DocumentOverride.user_id == principal.user_id)
    if groups:
        override_filter.append(DocumentOverride.group_id.in_(groups))
    denied: set[int] = set()
    if override_filter:
        from sqlalchemy import or_

        rows = db.execute(select(DocumentOverride).where(or_(*override_filter))).scalars().all()
        for row in rows:
            if row.permission == "deny":
                denied.add(row.document_id)
            elif row.permission == "read":
                allowed_documents.add(row.document_id)

    if principal.api_key_id is not None:
        key = db.get(ApiKey, principal.api_key_id)
        scopes = set(key.collection_ids or []) if key is not None else set()
        if scopes:
            allowed_collections &= scopes
            in_scope_docs = set(
                db.execute(select(Document.id).where(Document.collection_id.in_(sorted(scopes)))).scalars()
            )
            allowed_documents &= in_scope_docs

    return AccessFilter(
        document_ids=frozenset(allowed_documents - denied),
        collection_ids=frozenset(allowed_collections),
        denied_document_ids=frozenset(denied),
    )
```

- [ ] **Step 4: Run, lint, commit**

```bash
uv run pytest packages/core/tests/test_access_policy.py -q && uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat(core): add groups, grants, overrides and the access filter policy" && git log -1 --format='%h %(trailers)'
```

Expected: 7 new tests pass.

---

### Task 7: API keys, principals, rate limiting, and the admin access API

**Files:**
- Create: `packages/core/src/ragfabric_core/auth/api_keys.py`, `auth/ratelimit.py`, `packages/server/src/ragfabric_server/schemas/access.py`, `api/routes/access.py`
- Modify: `packages/server/src/ragfabric_server/deps.py`, `main.py`
- Test: `packages/core/tests/test_api_keys.py`, `packages/server/tests/test_access_api.py`

**Interfaces:**
- Produces (`auth/api_keys.py`): `KEY_PREFIX = "rf_"`; `generate_api_key() -> tuple[str, str, str]` returning `(plaintext, prefix12, sha256hex)`; `hash_api_key(plaintext) -> str`; `create_api_key(db, *, name, user_id, collection_ids=(), strategies=(), rate_limit_per_minute=60, expires_at=None) -> tuple[ApiKey, str]`; `verify_api_key(db, plaintext) -> ApiKey | None` (active, not expired, updates `last_used_at`); `principal_for_user(db, user) -> Principal` (groups from `group_ids_for_user`); `principal_for_api_key(db, key) -> Principal` (role and groups from the key's user; `api_key_id` set).
- Produces (`auth/ratelimit.py`): `check_rate_limit(cache: Cache, subject: str, limit_per_minute: int, now: float | None = None) -> bool` using key `ratelimit:{subject}:{minute}` with `incr(..., ttl_seconds=120)`.
- Produces (`deps.py`): `get_principal(request, token, db) -> Principal`: an `X-API-Key` header or a bearer token starting with `rf_` authenticates by API key (401 on failure, 429 when over the key's limit); otherwise the JWT path; `get_access_filter(principal, db) -> AccessFilter`; `get_cache() -> Cache` built once from config (`memory` in tests).
- Produces (`api/routes/access.py`, all `require_role("admin")`, mounted at `/api/admin`): `POST /groups`, `GET /groups`, `POST /groups/{group_id}/members`, `DELETE /groups/{group_id}/members/{user_id}`, `POST /grants`, `GET /grants`, `DELETE /grants/{grant_id}`, `POST /overrides`, `GET /overrides`, `POST /keys` (returns the plaintext once), `GET /keys`, `DELETE /keys/{key_id}` (deactivates).

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_api_keys.py
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragfabric_core.auth.api_keys import KEY_PREFIX, create_api_key, generate_api_key, hash_api_key, principal_for_api_key, verify_api_key
from ragfabric_core.auth.ratelimit import check_rate_limit
from ragfabric_core.models import Base
from ragfabric_core.models.user import User
from ragfabric_core.stores.memory_cache import MemoryCache


def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'k.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_generate_and_hash():
    plain, prefix, digest = generate_api_key()
    assert plain.startswith(KEY_PREFIX) and len(plain) > 30 and plain.startswith(prefix) and len(prefix) == 12
    assert digest == hash_api_key(plain) and digest != plain


def test_create_verify_expire_and_deactivate(tmp_path):
    db = session(tmp_path)
    user = User(email="u@x", hashed_password="h"); db.add(user); db.commit()
    key, plain = create_api_key(db, name="ci", user_id=user.id, collection_ids=[1], strategies=["traditional"], rate_limit_per_minute=5)
    db.commit()
    assert key.key_hash == hash_api_key(plain) and key.key_prefix == plain[:12]
    found = verify_api_key(db, plain)
    assert found is not None and found.id == key.id and found.last_used_at is not None
    assert verify_api_key(db, plain + "x") is None
    p = principal_for_api_key(db, found)
    assert p.user_id == user.id and p.api_key_id == key.id and p.role == "user"
    key.expires_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=1); db.commit()
    assert verify_api_key(db, plain) is None
    key.expires_at = None; key.is_active = False; db.commit()
    assert verify_api_key(db, plain) is None


def test_rate_limit_counts_per_minute():
    cache = MemoryCache()
    assert check_rate_limit(cache, "k1", 2, now=60.0) is True
    assert check_rate_limit(cache, "k1", 2, now=61.0) is True
    assert check_rate_limit(cache, "k1", 2, now=62.0) is False
    assert check_rate_limit(cache, "k1", 2, now=125.0) is True  # next minute
    assert check_rate_limit(cache, "k2", 2, now=62.0) is True  # other subject
```

```python
# packages/server/tests/test_access_api.py
from ragfabric_core.testing.fixtures import make_txt


def _upload(client, headers, name, text, collection_id=None):
    data = {"collection_id": str(collection_id)} if collection_id else {}
    r = client.post("/api/documents/upload", files={"file": (name, make_txt(text), "text/plain")}, data=data, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def test_admin_manages_groups_grants_and_keys(client, admin_headers, auth_headers):
    me = client.get("/api/auth/me", headers=auth_headers).json()
    g = client.post("/api/admin/groups", json={"name": "hr-team"}, headers=admin_headers)
    assert g.status_code == 201 and g.json()["name"] == "hr-team"
    gid = g.json()["id"]
    assert client.post(f"/api/admin/groups/{gid}/members", json={"user_id": me["id"]}, headers=admin_headers).status_code == 200
    col = client.post("/api/collections", json={"name": "hr"}, headers=admin_headers).json()
    grant = client.post("/api/admin/grants", json={"group_id": gid, "collection_id": col["id"], "permission": "read"}, headers=admin_headers)
    assert grant.status_code == 201
    assert any(x["collection_id"] == col["id"] for x in client.get("/api/admin/grants", headers=admin_headers).json())
    assert client.post("/api/admin/groups", json={"name": "x"}, headers=auth_headers).status_code == 403


def test_api_key_is_shown_once_and_authenticates(client, admin_headers):
    admin = client.get("/api/auth/me", headers=admin_headers).json()
    created = client.post("/api/admin/keys", json={"name": "ci", "user_id": admin["id"], "rate_limit_per_minute": 2}, headers=admin_headers)
    assert created.status_code == 201
    plain = created.json()["key"]
    assert plain.startswith("rf_")
    listed = client.get("/api/admin/keys", headers=admin_headers).json()
    assert listed and "key" not in listed[0] and listed[0]["key_prefix"] == plain[:12]
    _upload(client, admin_headers, "a.txt", "leave policy is twelve days")
    r = client.post("/api/search/semantic", json={"query": "leave"}, headers={"X-API-Key": plain})
    assert r.status_code == 200 and r.json()["results"]
    r = client.post("/api/search/semantic", json={"query": "leave"}, headers={"Authorization": f"Bearer {plain}"})
    assert r.status_code == 200
    r = client.post("/api/search/semantic", json={"query": "leave"}, headers={"X-API-Key": plain})
    assert r.status_code == 429
    assert client.post("/api/search/semantic", json={"query": "leave"}, headers={"X-API-Key": "rf_wrong"}).status_code == 401
    key_id = listed[0]["id"]
    assert client.delete(f"/api/admin/keys/{key_id}", headers=admin_headers).status_code == 200
    assert client.post("/api/search/semantic", json={"query": "leave"}, headers={"X-API-Key": plain}).status_code == 401
```

Run → failures (404 routes, import errors).

- [ ] **Step 2: Implement keys and rate limiting**

```python
# packages/core/src/ragfabric_core/auth/api_keys.py
"""API keys: generated once, stored hashed, resolved to a Principal.

A key is `rf_` plus 32 URL safe random bytes. Only its SHA 256 is stored; the
12 character prefix is kept in clear so an operator can recognise a key in a
list without ever seeing the secret again.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import Principal
from ragfabric_core.auth.service import group_ids_for_user
from ragfabric_core.models.access import ApiKey
from ragfabric_core.models.user import User

KEY_PREFIX = "rf_"
PREFIX_LEN = 12


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
    return plaintext, plaintext[:PREFIX_LEN], hash_api_key(plaintext)


def create_api_key(
    db: Session,
    *,
    name: str,
    user_id: int,
    collection_ids=(),
    strategies=(),
    rate_limit_per_minute: int = 60,
    expires_at: datetime | None = None,
) -> tuple[ApiKey, str]:
    plaintext, prefix, digest = generate_api_key()
    key = ApiKey(
        name=name, key_prefix=prefix, key_hash=digest, principal_user_id=user_id,
        collection_ids=list(collection_ids), strategies=list(strategies),
        rate_limit_per_minute=rate_limit_per_minute, expires_at=expires_at,
    )
    db.add(key)
    db.flush()
    return key, plaintext


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def verify_api_key(db: Session, plaintext: str) -> ApiKey | None:
    if not plaintext.startswith(KEY_PREFIX):
        return None
    key = db.execute(select(ApiKey).where(ApiKey.key_hash == hash_api_key(plaintext))).scalar_one_or_none()
    if key is None or not key.is_active:
        return None
    if key.expires_at is not None and key.expires_at <= _now():
        return None
    key.last_used_at = _now()
    db.commit()
    return key


def principal_for_user(db: Session, user: User) -> Principal:
    return Principal(user_id=user.id, email=user.email, role=user.role, group_ids=group_ids_for_user(db, user.id), api_key_id=None)


def principal_for_api_key(db: Session, key: ApiKey) -> Principal:
    user = db.get(User, key.principal_user_id) if key.principal_user_id is not None else None
    return Principal(
        user_id=user.id if user else None,
        email=user.email if user else None,
        role=user.role if user else "user",
        group_ids=group_ids_for_user(db, user.id) if user else [],
        api_key_id=key.id,
    )
```

```python
# packages/core/src/ragfabric_core/auth/ratelimit.py
"""Fixed window rate limit on the Cache interface: one counter per subject per minute."""

from __future__ import annotations

import time

from ragfabric_core.stores.base import Cache


def check_rate_limit(cache: Cache, subject: str, limit_per_minute: int, now: float | None = None) -> bool:
    current = time.time() if now is None else now
    minute = int(current // 60)
    count = cache.incr(f"ratelimit:{subject}:{minute}", ttl_seconds=120)
    return count <= limit_per_minute
```

- [ ] **Step 3: Server dependencies**

Rewrite `packages/server/src/ragfabric_server/deps.py` keeping `get_current_user` and `require_role` exactly as they are and adding:

```python
from fastapi import Request

from ragfabric_core.auth.api_keys import KEY_PREFIX, principal_for_api_key, principal_for_user, verify_api_key
from ragfabric_core.auth.policy import compute_access_filter
from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.auth.ratelimit import check_rate_limit
from ragfabric_core.runtime import get_config
from ragfabric_core.stores.base import Cache
from ragfabric_core.stores.registry import build_cache

_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = build_cache(get_config().cache)
    return _cache


def _api_key_from_request(request: Request, token: str | None) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header
    if token and token.startswith(KEY_PREFIX):
        return token
    return None


def get_principal(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Principal:
    """Resolve the caller: an API key (header or bearer) or a JWT user."""
    plaintext = _api_key_from_request(request, token)
    if plaintext is not None:
        key = verify_api_key(db, plaintext)
        if key is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        if not check_rate_limit(get_cache(), f"key:{key.id}", key.rate_limit_per_minute):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
        return principal_for_api_key(db, key)
    user = get_current_user(token=token, db=db)
    return principal_for_user(db, user)


def get_access_filter(
    principal: Principal = Depends(get_principal), db: Session = Depends(get_db)
) -> AccessFilter:
    return compute_access_filter(db, principal)
```

Note: the tests' `cache.kind` is `redis` by default; the server test config file written in `conftest.py` (Task 3) must add `cache:\n  kind: memory\n` so `get_cache()` needs no Redis. Update that fixture string accordingly.

- [ ] **Step 4: Schemas and routes**

```python
# packages/server/src/ragfabric_server/schemas/access.py
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""


class GroupOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: str
    created_at: datetime


class MemberAdd(BaseModel):
    user_id: int


class GrantCreate(BaseModel):
    group_id: int
    collection_id: int
    permission: str = Field(default="read", pattern="^(read|write)$")


class GrantOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    group_id: int
    collection_id: int
    permission: str


class OverrideCreate(BaseModel):
    document_id: int
    group_id: int | None = None
    user_id: int | None = None
    permission: str = Field(default="deny", pattern="^(deny|read)$")


class OverrideOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    document_id: int
    group_id: int | None
    user_id: int | None
    permission: str


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    user_id: int
    collection_ids: list[int] = []
    strategies: list[str] = []
    rate_limit_per_minute: int = Field(default=60, ge=1, le=100000)
    expires_at: datetime | None = None


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    key_prefix: str
    principal_user_id: int | None
    collection_ids: list[int]
    strategies: list[str]
    rate_limit_per_minute: int
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None


class ApiKeyCreated(ApiKeyOut):
    key: str
```

```python
# packages/server/src/ragfabric_server/api/routes/access.py
"""Admin API for groups, grants, overrides and API keys."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ragfabric_core.auth import service
from ragfabric_core.auth.api_keys import create_api_key
from ragfabric_core.db.session import get_db
from ragfabric_core.models.access import ApiKey, CollectionGrant, DocumentOverride, Group
from ragfabric_core.models.document import Collection, Document
from ragfabric_core.models.user import User
from ragfabric_server.deps import require_role
from ragfabric_server.schemas.access import (
    ApiKeyCreate, ApiKeyCreated, ApiKeyOut, GrantCreate, GrantOut, GroupCreate, GroupOut,
    MemberAdd, OverrideCreate, OverrideOut,
)

router = APIRouter(dependencies=[Depends(require_role("admin"))])


def _get_or_404(db: Session, model, ident, name: str):
    obj = db.get(model, ident)
    if obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{name} not found.")
    return obj


@router.post("/groups", response_model=GroupOut, status_code=status.HTTP_201_CREATED)
def create_group(payload: GroupCreate, db: Session = Depends(get_db)) -> Group:
    if db.query(Group).filter(Group.name == payload.name).first() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Group name already exists.")
    group = service.create_group(db, payload.name, payload.description)
    db.commit()
    db.refresh(group)
    return group


@router.get("/groups", response_model=list[GroupOut])
def list_groups(db: Session = Depends(get_db)) -> list[Group]:
    return db.query(Group).order_by(Group.name).all()


@router.post("/groups/{group_id}/members", status_code=status.HTTP_200_OK)
def add_member(group_id: int, payload: MemberAdd, db: Session = Depends(get_db)) -> dict:
    _get_or_404(db, Group, group_id, "Group")
    _get_or_404(db, User, payload.user_id, "User")
    service.add_member(db, group_id, payload.user_id)
    db.commit()
    return {"detail": "Member added.", "group_id": group_id, "user_id": payload.user_id}


@router.delete("/groups/{group_id}/members/{user_id}", status_code=status.HTTP_200_OK)
def remove_member(group_id: int, user_id: int, db: Session = Depends(get_db)) -> dict:
    service.remove_member(db, group_id, user_id)
    db.commit()
    return {"detail": "Member removed.", "group_id": group_id, "user_id": user_id}


@router.post("/grants", response_model=GrantOut, status_code=status.HTTP_201_CREATED)
def create_grant(payload: GrantCreate, db: Session = Depends(get_db)) -> CollectionGrant:
    _get_or_404(db, Group, payload.group_id, "Group")
    _get_or_404(db, Collection, payload.collection_id, "Collection")
    grant = service.grant_collection(db, payload.group_id, payload.collection_id, payload.permission)
    db.commit()
    db.refresh(grant)
    return grant


@router.get("/grants", response_model=list[GrantOut])
def list_grants(db: Session = Depends(get_db)) -> list[CollectionGrant]:
    return db.query(CollectionGrant).order_by(CollectionGrant.id).all()


@router.delete("/grants/{grant_id}", status_code=status.HTTP_200_OK)
def delete_grant(grant_id: int, db: Session = Depends(get_db)) -> dict:
    grant = _get_or_404(db, CollectionGrant, grant_id, "Grant")
    db.delete(grant)
    db.commit()
    return {"detail": "Grant removed.", "id": grant_id}


@router.post("/overrides", response_model=OverrideOut, status_code=status.HTTP_201_CREATED)
def create_override(payload: OverrideCreate, db: Session = Depends(get_db)) -> DocumentOverride:
    _get_or_404(db, Document, payload.document_id, "Document")
    try:
        override = service.set_document_override(
            db, payload.document_id, group_id=payload.group_id, user_id=payload.user_id, permission=payload.permission
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    db.commit()
    db.refresh(override)
    return override


@router.get("/overrides", response_model=list[OverrideOut])
def list_overrides(db: Session = Depends(get_db)) -> list[DocumentOverride]:
    return db.query(DocumentOverride).order_by(DocumentOverride.id).all()


@router.post("/keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
def create_key(payload: ApiKeyCreate, db: Session = Depends(get_db)) -> ApiKeyCreated:
    _get_or_404(db, User, payload.user_id, "User")
    key, plaintext = create_api_key(
        db, name=payload.name, user_id=payload.user_id, collection_ids=payload.collection_ids,
        strategies=payload.strategies, rate_limit_per_minute=payload.rate_limit_per_minute, expires_at=payload.expires_at,
    )
    db.commit()
    db.refresh(key)
    return ApiKeyCreated(**ApiKeyOut.model_validate(key).model_dump(), key=plaintext)


@router.get("/keys", response_model=list[ApiKeyOut])
def list_keys(db: Session = Depends(get_db)) -> list[ApiKey]:
    return db.query(ApiKey).order_by(ApiKey.id).all()


@router.delete("/keys/{key_id}", status_code=status.HTTP_200_OK)
def revoke_key(key_id: int, db: Session = Depends(get_db)) -> dict:
    key = _get_or_404(db, ApiKey, key_id, "API key")
    key.is_active = False
    db.commit()
    return {"detail": "API key revoked.", "id": key_id}
```

In `main.py` import `access` alongside the other route modules and add `app.include_router(access.router, prefix="/api/admin", tags=["access"])`. Change the search routes to depend on `get_principal` instead of `get_current_user` (Task 8 finishes the search route work; here just enough for the API key test): in `search.py` replace `current_user: User = Depends(get_current_user)` with `principal: Principal = Depends(get_principal)` in all three endpoints and use `principal.user_id` for the `QueryLog.user_id`.

- [ ] **Step 5: Run, lint, commit**

```bash
uv run pytest packages/core/tests/test_api_keys.py packages/server/tests/test_access_api.py -q && uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat: add API keys, principal resolution, rate limiting and the admin access API" && git log -1 --format='%h %(trailers)'
```

---

### Task 8: Apply the access filter on the live query path, record runs and audit rows

**Files:**
- Modify: `packages/core/src/ragfabric_core/store/vector_store.py`, `retrieve/retriever.py`, `retrieve/hybrid.py`, `strategies/legacy.py`, `packages/core/tests/test_strategy_interface.py`
- Modify: `packages/server/src/ragfabric_server/api/routes/search.py`, `api/routes/documents.py`
- Create: `packages/server/src/ragfabric_server/api/routes/runs.py`, `schemas/runs.py`
- Test: additions to `packages/core/tests/test_pipeline.py` or `packages/server/tests/test_index.py` for the store filter; `packages/server/tests/test_access_enforcement.py`

**Interfaces:**
- `InMemoryVectorStore.candidate_rows(filters, access: AccessFilter | None = None)`; `search(..., access=None)`; `access_stats(filters, access) -> tuple[int, int]` returning `(candidates_before_access, candidates_after_access)`.
- `Retriever.retrieve(..., access=None)`, `HybridRetriever.retrieve(..., access=None)`.
- `LegacyHybridStrategy.retrieve` passes `access=ctx.access_filter` and no longer post filters.
- `POST /api/search/query` writes one `RetrievalRun` (mode `manual`, requested and selected strategy `traditional`, `embedding_model` from the v1 embedder as `hashing-<dim>`, latency fields, `trace` from Task 10, empty list until then), one `Source` per retrieved chunk with `cited` from the answer's `used` citations, and one `AuditLog` row (`action="query"`, `sources_returned`, `sources_filtered`). `/semantic` and `/hybrid` write an `AuditLog` row with `action="search"`.
- `GET /api/documents` and `GET /api/documents/{id}` and `/download` apply the access filter (list uses `access_clause` on `Document.id` and `Document.collection_id`; get and download return 404 for documents outside the filter).
- `GET /api/runs/{run_id}` returns the run with its sources and trace; admins see all, others only their own runs.

- [ ] **Step 1: Write the failing tests**

Add to `packages/server/tests/test_index.py`:

```python
def test_store_applies_the_access_filter_before_ranking():
    from ragfabric_core.auth.principal import AccessFilter
    from ragfabric_core.store.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore(dim=2)
    store.upsert([
        {"vector": [1.0, 0.0], "chunk_id": 1, "document_id": 10, "collection_id": 100, "filename": "a", "format": "txt", "page": 1, "chunk_index": 0, "text": "a"},
        {"vector": [0.9, 0.1], "chunk_id": 2, "document_id": 20, "collection_id": 200, "filename": "b", "format": "txt", "page": 1, "chunk_index": 0, "text": "b"},
    ])
    open_only = AccessFilter(document_ids=frozenset(), collection_ids=frozenset({100}), denied_document_ids=frozenset())
    hits = store.search([1.0, 0.0], top_k=5, access=open_only)
    assert [h["chunk_id"] for h in hits] == [1]
    assert store.access_stats({}, open_only) == (2, 1)
    assert store.search([1.0, 0.0], top_k=5, access=None) and len(store.search([1.0, 0.0], top_k=5)) == 2
```

```python
# packages/server/tests/test_access_enforcement.py
"""Phase 2 exit criterion: a viewer without a grant gets zero chunks from a restricted collection."""

from ragfabric_core.testing.fixtures import make_txt


def _upload(client, headers, name, text, collection_id):
    r = client.post("/api/documents/upload", files={"file": (name, make_txt(text), "text/plain")}, data={"collection_id": str(collection_id)}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def test_viewer_without_grant_gets_nothing_from_a_restricted_collection(client, admin_headers, auth_headers):
    hr = client.post("/api/collections", json={"name": "hr"}, headers=admin_headers).json()
    doc = _upload(client, admin_headers, "policy.txt", "annual leave is twelve days for everyone", hr["id"])
    # Before any grant exists the collection is open (v1 behaviour).
    assert client.post("/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers).json()["results"]
    group = client.post("/api/admin/groups", json={"name": "hr-only"}, headers=admin_headers).json()
    client.post("/api/admin/grants", json={"group_id": group["id"], "collection_id": hr["id"], "permission": "read"}, headers=admin_headers)
    # Now the collection is restricted to the group, which the viewer is not in.
    r = client.post("/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers)
    assert r.status_code == 200 and r.json()["results"] == []
    answer = client.post("/api/search/query", json={"query": "how many days of annual leave"}, headers=auth_headers).json()
    assert answer["citations"] == [] and "enough information" in answer["answer"]
    assert client.get(f"/api/documents/{doc['id']}", headers=auth_headers).status_code == 404
    assert all(d["id"] != doc["id"] for d in client.get("/api/documents", headers=auth_headers).json()["items"])
    assert client.get(f"/api/documents/{doc['id']}/download", headers=auth_headers).status_code == 404
    # Admin still sees it, and the audit row records what the filter removed for the viewer.
    assert client.get(f"/api/documents/{doc['id']}", headers=admin_headers).status_code == 200
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.access import AuditLog

    with SessionLocal() as db:
        rows = db.query(AuditLog).filter(AuditLog.action == "query").order_by(AuditLog.id.desc()).all()
    assert rows and rows[0].sources_returned == 0 and rows[0].sources_filtered >= 1
    # Adding the viewer to the group restores access.
    me = client.get("/api/auth/me", headers=auth_headers).json()
    client.post(f"/api/admin/groups/{group['id']}/members", json={"user_id": me["id"]}, headers=admin_headers)
    assert client.post("/api/search/semantic", json={"query": "annual leave"}, headers=auth_headers).json()["results"]


def test_query_records_a_retrieval_run_with_sources_and_is_readable_by_its_owner(client, auth_headers, admin_headers):
    col = client.post("/api/collections", json={"name": "c"}, headers=auth_headers).json()
    _upload(client, auth_headers, "n.txt", "kubernetes rollout guide with three steps", col["id"])
    client.post("/api/search/query", json={"query": "rollout steps"}, headers=auth_headers)
    from ragfabric_core.db.session import SessionLocal
    from ragfabric_core.models.runs import RetrievalRun

    with SessionLocal() as db:
        run = db.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run.selected_strategy == "traditional" and run.mode == "manual" and run.retrieval_calls == 1
    assert run.embedding_model.startswith("hashing-") and run.latency_ms >= 0
    r = client.get(f"/api/runs/{run.id}", headers=auth_headers)
    assert r.status_code == 200 and r.json()["sources"] and r.json()["question"] == "rollout steps"
    assert client.get(f"/api/runs/{run.id}", headers=admin_headers).status_code == 200
    other = client.post("/api/auth/register", json={"email": "other@example.com", "password": "password123"})
    tok = client.post("/api/auth/login", data={"username": "other@example.com", "password": "password123"}).json()["access_token"]
    assert client.get(f"/api/runs/{run.id}", headers={"Authorization": f"Bearer {tok}"}).status_code == 404
```

Update `test_legacy_hybrid_strategy_meets_the_contract` in `packages/core/tests/test_strategy_interface.py`: the `FakeRetriever.retrieve` signature gains `access=None` and the test asserts the adapter passed the context's filter through: record the received `access` on the fake (`self.seen_access = access`) and assert `fake.seen_access is ctx.access_filter` after the call (adjust the monkeypatch to return a single fake instance you keep a reference to).

Run → failures.

- [ ] **Step 2: Store, retrievers and adapter**

In `store/vector_store.py`:
- `from ragfabric_core.auth.principal import AccessFilter`
- `candidate_rows(self, filters: dict, access: AccessFilter | None = None)`: after computing the metadata matches (the existing list), if `access is not None and not access.is_unrestricted`, return `[i for i in rows if access.allows(self._meta[i].get("document_id"), self._meta[i].get("collection_id"))]`.
- `search(..., access: AccessFilter | None = None)` passes `access` to `candidate_rows`.
- add `def access_stats(self, filters: dict, access: AccessFilter | None) -> tuple[int, int]: before = self.candidate_rows(filters); after = self.candidate_rows(filters, access); return len(before), len(after)`.

In `retrieve/retriever.py` and `retrieve/hybrid.py` add `access=None` as the last keyword parameter of `retrieve` and pass it to `self.store.search(...)` and `self.store.candidate_rows({...}, access)` respectively. Update docstrings: "access: the caller's AccessFilter, applied before ranking".

In `strategies/legacy.py`: call `HybridRetriever().retrieve(query, top_k=ctx.params.top_k, collection_id=collection_id, access=ctx.access_filter)`, remove the `if ctx.access_filter.allows(...)` post filter, and update the module docstring ("The access filter is passed into the v1 store, which applies it before ranking.").

- [ ] **Step 3: Search route with run and audit rows**

Rewrite `search.py`'s `query` endpoint:

```python
@router.post("/query", response_model=AnswerResponse)
def query(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
) -> AnswerResponse:
    started = time.perf_counter()
    retrieved = _retrieve(payload, access)
    retrieval_ms = int((time.perf_counter() - started) * 1000)
    result = build_answer(payload.query, retrieved)
    total_ms = int((time.perf_counter() - started) * 1000)
    cited = sorted({c["document_id"] for c in result["citations"] if c["used"] and c["document_id"] is not None})
    used_chunks = {c["chunk_id"] for c in result["citations"] if c["used"]}
    before, after = get_store().access_stats(
        {"collection_id": payload.collection_id, "document_id": payload.document_id, "format": payload.format}, access
    )
    run = RetrievalRun(
        user_id=principal.user_id, api_key_id=principal.api_key_id, question=payload.query, mode="manual",
        requested_strategy="traditional", selected_strategy="traditional", answer=result["answer"],
        latency_ms=total_ms, retrieval_latency_ms=retrieval_ms, generation_latency_ms=total_ms - retrieval_ms,
        llm_calls=0, retrieval_calls=1, input_tokens=0, output_tokens=0, estimated_cost_usd=0.0,
        embedding_model=f"hashing-{get_embedder().dim}", trace=[],
    )
    db.add(run)
    db.flush()
    for rank, row in enumerate(retrieved, start=1):
        db.add(Source(retrieval_run_id=run.id, chunk_id=row.get("chunk_id"), document_id=row.get("document_id"),
                      rank=rank, score=row.get("score"), cited=row.get("chunk_id") in used_chunks, page=row.get("page")))
    db.add(AuditLog(principal_user_id=principal.user_id, api_key_id=principal.api_key_id, action="query",
                    question=payload.query, strategy="traditional", retrieval_run_id=run.id,
                    sources_returned=len(retrieved), sources_filtered=max(before - after, 0),
                    details={"collection_id": payload.collection_id, "mode": payload.mode}))
    db.add(QueryLog(user_id=principal.user_id, collection_id=payload.collection_id, question=payload.query,
                    confidence=result["confidence"], cited_document_ids=cited))
    db.commit()
    return AnswerResponse(**result)
```

with `_retrieve(payload, access)` passing `access=access` to the retriever, and the `/semantic` and `/hybrid` endpoints taking `principal` and `access` too, passing `access`, and writing an `AuditLog(action="search", ...)` row with the same counts (no `RetrievalRun`). Imports: `time`, `Principal`, `AccessFilter`, `get_principal`, `get_access_filter`, `get_store`, `get_embedder`, `RetrievalRun`, `Source`, `AuditLog`.

- [ ] **Step 4: Documents routes and the runs route**

In `documents.py`: `list_documents` takes `access: AccessFilter = Depends(get_access_filter)` and applies `clause = access_clause(access, Document.id, Document.collection_id)`; `if clause is not None: query = query.filter(clause)`. `get_document` and `download_document` take `access` and return 404 when `not access.allows(document.id, document.collection_id)`. Keep `current_user` where it is used for ownership checks; where a route only needed authentication, `Depends(get_principal)` replaces it so API keys work.

```python
# packages/server/src/ragfabric_server/schemas/runs.py
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    rank: int
    chunk_id: int | None
    document_id: int | None
    score: float | None
    cited: bool
    page: int | None


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    question: str
    mode: str
    requested_strategy: str | None
    selected_strategy: str
    fallback_from: str | None
    answer: str | None
    latency_ms: int
    retrieval_latency_ms: int
    generation_latency_ms: int
    llm_calls: int
    retrieval_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    llm_model: str | None
    embedding_model: str | None
    trace: list
    created_at: datetime
    sources: list[SourceOut] = []
```

```python
# packages/server/src/ragfabric_server/api/routes/runs.py
"""Read one retrieval run with its sources and trace (the Trace page's data source)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.models.runs import RetrievalRun, Source
from ragfabric_server.deps import get_principal
from ragfabric_server.schemas.runs import RunOut, SourceOut

router = APIRouter()


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: int, db: Session = Depends(get_db), principal: Principal = Depends(get_principal)) -> RunOut:
    run = db.get(RetrievalRun, run_id)
    if run is None or (principal.role != "admin" and run.user_id != principal.user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
    sources = db.query(Source).filter(Source.retrieval_run_id == run.id).order_by(Source.rank).all()
    out = RunOut.model_validate(run)
    out.sources = [SourceOut.model_validate(s) for s in sources]
    return out
```

Mount it in `main.py`: `app.include_router(runs.router, prefix="/api/runs", tags=["runs"])`.

- [ ] **Step 5: Run, lint, commit**

```bash
uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat: apply the access filter inside retrieval, record retrieval runs, sources and audit rows" && git log -1 --format='%h %(trailers)'
```

Expected: the exit criterion test passes; every v1 API test still passes (v1 collections have no grants, so they stay open).

---

### Task 9: CLI: init, ingest, users, groups, grants, keys, worker

**Files:**
- Create: `packages/cli/src/ragfabric_cli/commands/__init__.py`, `commands/ingest.py`, `commands/users.py`, `commands/access.py`, `commands/worker.py`, `commands/common.py`
- Modify: `packages/cli/src/ragfabric_cli/main.py`
- Test: `packages/cli/tests/test_cli_phase2.py`

**Interfaces:**
- `ragfabric init [--force]`: copies `.env.example` to `.env` and `ragfabric.example.yaml` to `ragfabric.yaml` when missing (or with `--force`), then runs the same checks as `config validate`. Exit 1 if the examples are not found in the current directory.
- `ragfabric ingest PATH [--collection NAME] [--recursive] [--owner EMAIL]`: ingests one file or every supported file in a directory through `ingest_document`, creating the collection by name if needed, printing `filename: status (n chunks)` per file and a summary; exit 1 if any file failed.
- `ragfabric users create --email --password [--role user|admin]`, `users list`, `users set-role EMAIL ROLE`, `users deactivate EMAIL`.
- `ragfabric groups create NAME [--description]`, `groups add-member GROUP_NAME EMAIL`, `groups list`.
- `ragfabric grants add --group NAME --collection NAME [--permission read|write]`, `grants list`.
- `ragfabric keys create --name NAME --user EMAIL [--collection NAME ...] [--strategy NAME ...] [--rate-limit N]` prints the plaintext once, `keys list`, `keys revoke ID`.
- `ragfabric worker [--once]`: runs the worker against the configured queue; exit 1 with a clear message when `ingestion.indexing` is `inline`.
- `commands/common.py`: `session()` context manager over `get_session_factory()`, `user_by_email(db, email) -> User` (exits 1 if missing), `collection_by_name(db, name, create=False)`.

- [ ] **Step 1: Write the failing tests**

```python
# packages/cli/tests/test_cli_phase2.py
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from ragfabric_core import runtime

    url = f"sqlite:///{tmp_path / 'cli2.db'}"
    cfg = tmp_path / "ragfabric.yaml"
    cfg.write_text(f"embeddings:\n  provider: offline\n  dim: 16\ncache:\n  kind: memory\ningestion:\n  uploads_dir: {tmp_path / 'blobs'}\n")
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(cfg))
    monkeypatch.setenv("JWT_SECRET", "cli-test-secret")
    runtime.reset_config()
    # Fresh engine bound to this database: the session module caches settings at import.
    import ragfabric_core.db.session as session_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(url, connect_args={"check_same_thread": False})
    monkeypatch.setattr(session_module, "engine", engine)
    monkeypatch.setattr(session_module, "SessionLocal", sessionmaker(bind=engine, autoflush=False, autocommit=False))
    from ragfabric_core.store.vector_store import get_store

    get_store().clear()
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    yield tmp_path
    runtime.reset_config()


def test_init_copies_examples_and_validates(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    for name in (".env.example", "ragfabric.example.yaml"):
        (tmp_path / name).write_text((root / name).read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / ".env").exists() and (tmp_path / "ragfabric.yaml").exists()
    assert "configuration ok" in result.stdout
    again = runner.invoke(app, ["init"])
    assert "already exists" in again.stdout


def test_users_groups_grants_keys_flow(env):
    assert runner.invoke(app, ["users", "create", "--email", "a@x.io", "--password", "password123"]).exit_code == 0
    assert runner.invoke(app, ["users", "create", "--email", "root@x.io", "--password", "password123", "--role", "admin"]).exit_code == 0
    listed = runner.invoke(app, ["users", "list"])
    assert "a@x.io" in listed.stdout and "root@x.io" in listed.stdout
    assert runner.invoke(app, ["groups", "create", "hr"]).exit_code == 0
    assert runner.invoke(app, ["groups", "add-member", "hr", "a@x.io"]).exit_code == 0
    assert "hr" in runner.invoke(app, ["groups", "list"]).stdout
    from ragfabric_cli.commands.common import collection_by_name, session

    with session() as db:
        collection_by_name(db, "policies", create=True); db.commit()
    assert runner.invoke(app, ["grants", "add", "--group", "hr", "--collection", "policies", "--permission", "read"]).exit_code == 0
    assert "policies" in runner.invoke(app, ["grants", "list"]).stdout
    created = runner.invoke(app, ["keys", "create", "--name", "ci", "--user", "a@x.io", "--collection", "policies", "--rate-limit", "10"])
    assert created.exit_code == 0 and "rf_" in created.stdout
    listed = runner.invoke(app, ["keys", "list"])
    assert "ci" in listed.stdout and "rf_" in listed.stdout and created.stdout.split("rf_")[1].strip().split()[0] not in listed.stdout
    assert runner.invoke(app, ["keys", "revoke", "1"]).exit_code == 0
    assert runner.invoke(app, ["users", "set-role", "a@x.io", "admin"]).exit_code == 0
    assert runner.invoke(app, ["users", "deactivate", "a@x.io"]).exit_code == 0
    assert runner.invoke(app, ["users", "set-role", "nobody@x.io", "admin"]).exit_code == 1


def test_ingest_directory_creates_collection_and_indexes(env):
    docs = env / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("annual leave is twelve days")
    (docs / "b.md").write_text("# Rollout\nkubernetes rollout guide")
    (docs / "skip.xyz").write_text("ignored")
    result = runner.invoke(app, ["ingest", str(docs), "--collection", "handbook"])
    assert result.exit_code == 0, result.stdout
    assert "a.txt: ready" in result.stdout and "2 ingested" in result.stdout and "skip.xyz" not in result.stdout
    from ragfabric_cli.commands.common import session
    from ragfabric_core.models.document import Document
    from ragfabric_core.models.index import ChunkEmbedding

    with session() as db:
        assert db.query(Document).count() == 2 and db.query(ChunkEmbedding).count() >= 2


def test_worker_refuses_inline_mode(env):
    result = runner.invoke(app, ["worker", "--once"])
    assert result.exit_code == 1 and "indexing: queue" in result.stdout
```

Note: `.md` is not in the v1 `SUPPORTED_FORMATS`. Add `"md"` to `SUPPORTED_FORMATS` and to `PARSERS` in `ingest/parser.py` using `parse_txt`, and add `md` to the `SearchRequest.format` pattern in `schemas/search.py`; the `test_supported_formats_constant_matches_the_tested_set` parser test lists the formats it exercises, so add a `make_txt` based `md` case to that parametrised list too. Record this as a deliberate small scope addition in the report.

Run → failures.

- [ ] **Step 2: Implement common helpers and commands**

```python
# packages/cli/src/ragfabric_cli/commands/__init__.py
"""Sub commands of the ragfabric CLI."""
```

```python
# packages/cli/src/ragfabric_cli/commands/common.py
from __future__ import annotations

from contextlib import contextmanager

import typer
from sqlalchemy.orm import Session

from ragfabric_core.models.document import Collection
from ragfabric_core.models.user import User


@contextmanager
def session():
    from ragfabric_core.runtime import get_session_factory

    db: Session = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def user_by_email(db: Session, email: str) -> User:
    user = db.query(User).filter(User.email == email.lower()).first()
    if user is None:
        typer.echo(f"user not found: {email}")
        raise typer.Exit(code=1)
    return user


def collection_by_name(db: Session, name: str, create: bool = False) -> Collection:
    collection = db.query(Collection).filter(Collection.name == name).first()
    if collection is None:
        if not create:
            typer.echo(f"collection not found: {name}")
            raise typer.Exit(code=1)
        collection = Collection(name=name)
        db.add(collection)
        db.flush()
    return collection
```

```python
# packages/cli/src/ragfabric_cli/commands/users.py
from __future__ import annotations

import typer

from ragfabric_cli.commands.common import session, user_by_email
from ragfabric_core.models.user import Role, User
from ragfabric_core.security import hash_password

app = typer.Typer(help="Users.")


@app.command("create")
def create(email: str = typer.Option(...), password: str = typer.Option(..., prompt=False), role: str = typer.Option("user")) -> None:
    if role not in {Role.ADMIN.value, Role.USER.value}:
        typer.echo("role must be user or admin")
        raise typer.Exit(code=1)
    with session() as db:
        if db.query(User).filter(User.email == email.lower()).first() is not None:
            typer.echo(f"user already exists: {email}")
            raise typer.Exit(code=1)
        db.add(User(email=email.lower(), hashed_password=hash_password(password), role=role))
        db.commit()
    typer.echo(f"created {email} ({role})")


@app.command("list")
def list_users() -> None:
    with session() as db:
        for u in db.query(User).order_by(User.id).all():
            typer.echo(f"{u.id}\t{u.email}\t{u.role}\t{'active' if u.is_active else 'inactive'}")


@app.command("set-role")
def set_role(email: str, role: str) -> None:
    if role not in {Role.ADMIN.value, Role.USER.value}:
        typer.echo("role must be user or admin")
        raise typer.Exit(code=1)
    with session() as db:
        user = user_by_email(db, email)
        user.role = role
        db.commit()
    typer.echo(f"{email} is now {role}")


@app.command("deactivate")
def deactivate(email: str) -> None:
    with session() as db:
        user = user_by_email(db, email)
        user.is_active = False
        db.commit()
    typer.echo(f"{email} deactivated")
```

```python
# packages/cli/src/ragfabric_cli/commands/access.py
from __future__ import annotations

import typer

from ragfabric_cli.commands.common import collection_by_name, session, user_by_email
from ragfabric_core.auth import service
from ragfabric_core.auth.api_keys import create_api_key
from ragfabric_core.models.access import ApiKey, CollectionGrant, Group
from ragfabric_core.models.document import Collection

groups_app = typer.Typer(help="Groups and membership.")
grants_app = typer.Typer(help="Collection grants.")
keys_app = typer.Typer(help="API keys.")


def _group(db, name: str) -> Group:
    group = db.query(Group).filter(Group.name == name).first()
    if group is None:
        typer.echo(f"group not found: {name}")
        raise typer.Exit(code=1)
    return group


@groups_app.command("create")
def create_group(name: str, description: str = typer.Option("")) -> None:
    with session() as db:
        if db.query(Group).filter(Group.name == name).first() is not None:
            typer.echo(f"group already exists: {name}")
            raise typer.Exit(code=1)
        service.create_group(db, name, description)
        db.commit()
    typer.echo(f"created group {name}")


@groups_app.command("add-member")
def add_member(group_name: str, email: str) -> None:
    with session() as db:
        group = _group(db, group_name)
        user = user_by_email(db, email)
        service.add_member(db, group.id, user.id)
        db.commit()
    typer.echo(f"added {email} to {group_name}")


@groups_app.command("list")
def list_groups() -> None:
    with session() as db:
        for g in db.query(Group).order_by(Group.name).all():
            members = service_members(db, g.id)
            typer.echo(f"{g.id}\t{g.name}\t{members} member(s)")


def service_members(db, group_id: int) -> int:
    from ragfabric_core.models.access import GroupMember

    return db.query(GroupMember).filter(GroupMember.group_id == group_id).count()


@grants_app.command("add")
def add_grant(group: str = typer.Option(...), collection: str = typer.Option(...), permission: str = typer.Option("read")) -> None:
    with session() as db:
        g = _group(db, group)
        c = collection_by_name(db, collection)
        try:
            service.grant_collection(db, g.id, c.id, permission)
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None
        db.commit()
    typer.echo(f"granted {permission} on {collection} to {group}")


@grants_app.command("list")
def list_grants() -> None:
    with session() as db:
        rows = db.query(CollectionGrant, Group.name, Collection.name).join(Group, Group.id == CollectionGrant.group_id).join(Collection, Collection.id == CollectionGrant.collection_id).all()
        for grant, group_name, collection_name in rows:
            typer.echo(f"{grant.id}\t{group_name}\t{collection_name}\t{grant.permission}")


@keys_app.command("create")
def create_key(
    name: str = typer.Option(...),
    user: str = typer.Option(...),
    collection: list[str] = typer.Option([], "--collection"),
    strategy: list[str] = typer.Option([], "--strategy"),
    rate_limit: int = typer.Option(60, "--rate-limit"),
) -> None:
    with session() as db:
        owner = user_by_email(db, user)
        collection_ids = [collection_by_name(db, c).id for c in collection]
        key, plaintext = create_api_key(db, name=name, user_id=owner.id, collection_ids=collection_ids, strategies=strategy, rate_limit_per_minute=rate_limit)
        db.commit()
        typer.echo(f"created key {key.id} ({name}) for {user}")
    typer.echo("store this now, it is not shown again:")
    typer.echo(plaintext)


@keys_app.command("list")
def list_keys() -> None:
    with session() as db:
        for k in db.query(ApiKey).order_by(ApiKey.id).all():
            typer.echo(f"{k.id}\t{k.name}\t{k.key_prefix}...\t{'active' if k.is_active else 'revoked'}\tlimit {k.rate_limit_per_minute}/min")


@keys_app.command("revoke")
def revoke_key(key_id: int) -> None:
    with session() as db:
        key = db.get(ApiKey, key_id)
        if key is None:
            typer.echo(f"key not found: {key_id}")
            raise typer.Exit(code=1)
        key.is_active = False
        db.commit()
    typer.echo(f"revoked key {key_id}")
```

```python
# packages/cli/src/ragfabric_cli/commands/ingest.py
from __future__ import annotations

from pathlib import Path

import typer

from ragfabric_cli.commands.common import collection_by_name, session, user_by_email
from ragfabric_core.ingest.parser import SUPPORTED_FORMATS
from ragfabric_core.ingest.pipeline import ingest_document

app = typer.Typer(help="Ingestion.")


def _files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in path.glob(pattern) if p.is_file() and p.suffix.lower().lstrip(".") in SUPPORTED_FORMATS)


@app.callback(invoke_without_command=True)
def ingest(
    path: Path = typer.Argument(..., exists=True),
    collection: str | None = typer.Option(None, "--collection"),
    recursive: bool = typer.Option(False, "--recursive"),
    owner: str | None = typer.Option(None, "--owner", help="Email of the owning user."),
) -> None:
    """Ingest one file or every supported file in a directory."""
    files = _files(path, recursive)
    if not files:
        typer.echo("no supported files found")
        raise typer.Exit(code=1)
    failed = 0
    with session() as db:
        collection_id = collection_by_name(db, collection, create=True).id if collection else None
        owner_id = user_by_email(db, owner).id if owner else None
        db.commit()
        for file in files:
            doc = ingest_document(db, filename=file.name, data=file.read_bytes(), collection_id=collection_id, owner_id=owner_id)
            typer.echo(f"{file.name}: {doc.status} ({doc.num_chunks} chunks){' ' + doc.error if doc.error else ''}")
            failed += doc.status == "failed"
    typer.echo(f"{len(files) - failed} ingested, {failed} failed")
    if failed:
        raise typer.Exit(code=1)
```

```python
# packages/cli/src/ragfabric_cli/commands/worker.py
from __future__ import annotations

import threading

import typer

from ragfabric_core.providers.registry import build_embedding_provider
from ragfabric_core.queue.registry import build_queue
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store
from ragfabric_core.workers.runner import Worker, default_handlers


def worker(once: bool = typer.Option(False, "--once", help="Process at most one job and exit.")) -> None:
    """Run the indexing worker against the configured queue."""
    cfg = get_config()
    queue = build_queue(cfg)
    if queue is None:
        typer.echo("ingestion.indexing is inline; set `indexing: queue` in ragfabric.yaml to run a worker")
        raise typer.Exit(code=1)
    sf = get_session_factory()
    handlers = default_handlers(
        embedding_provider=build_embedding_provider(cfg.embeddings),
        vector_store=build_vector_store(cfg.vector_store, sf),
        lexical_store=build_lexical_store(cfg.lexical_store, sf),
    )
    w = Worker(queue, sf, handlers)
    if once:
        typer.echo("processed 1 job" if w.run_once(timeout_seconds=1.0) else "queue empty")
        return
    typer.echo("worker started; waiting for jobs")
    w.run_forever(threading.Event())
```

In `main.py`: import the sub apps and register them:

```python
from ragfabric_cli.commands import access as access_commands
from ragfabric_cli.commands import ingest as ingest_commands
from ragfabric_cli.commands import users as users_commands
from ragfabric_cli.commands.worker import worker as worker_command

app.add_typer(users_commands.app, name="users")
app.add_typer(access_commands.groups_app, name="groups")
app.add_typer(access_commands.grants_app, name="grants")
app.add_typer(access_commands.keys_app, name="keys")
app.add_typer(ingest_commands.app, name="ingest")
app.command("worker")(worker_command)
```

and add `init`:

```python
@app.command()
def init(force: bool = typer.Option(False, "--force", help="Overwrite existing .env and ragfabric.yaml.")) -> None:
    """Create .env and ragfabric.yaml from the examples, then validate."""
    import shutil
    from pathlib import Path

    for example, target in ((".env.example", ".env"), ("ragfabric.example.yaml", "ragfabric.yaml")):
        src, dst = Path(example), Path(target)
        if not src.exists():
            typer.echo(f"{example} not found in the current directory")
            raise typer.Exit(code=1)
        if dst.exists() and not force:
            typer.echo(f"{target} already exists (use --force to overwrite)")
            continue
        shutil.copyfile(src, dst)
        typer.echo(f"wrote {target}")
    from ragfabric_core.runtime import reset_config

    reset_config()
    config_validate(path=None, check_providers=False)
```

- [ ] **Step 3: Run, lint, commit**

```bash
uv sync -q && uv run pytest packages/cli/tests -q && uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat(cli): add init, ingest, users, groups, grants, keys and worker commands" && git log -1 --format='%h %(trailers)'
```

---

### Task 10: Tracing spans on ingestion and retrieval, stored per run

**Files:**
- Create: `packages/core/src/ragfabric_core/telemetry/__init__.py`, `telemetry/tracing.py`
- Modify: `ingest/pipeline.py`, `workers/handlers.py`, `retrieve/retriever.py`, `retrieve/hybrid.py`, `packages/server/src/ragfabric_server/api/routes/search.py`, `main.py`
- Test: `packages/core/tests/test_tracing.py`, addition to `packages/server/tests/test_access_enforcement.py`

**Interfaces:**
- `start_trace() -> TraceContext`; `TraceContext.spans -> list[TraceSpan]`; use as `with start_trace() as ctx:` then `ctx.spans` after; `trace(name, **attributes)` context manager records a `TraceSpan(name, started_ms relative to the trace start, duration_ms, attributes)` into the current context, no op when no trace is active; `configure_otel(endpoint: str | None) -> bool` sets up an OTLP HTTP exporter to `<endpoint>/v1/traces` when an endpoint is given and returns whether export is on; when on, `trace()` also opens a real OpenTelemetry span with the same name and attributes.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_tracing.py
import time

from ragfabric_core.strategies.base import TraceSpan
from ragfabric_core.telemetry.tracing import configure_otel, otel_enabled, start_trace, trace


def test_spans_are_recorded_with_relative_offsets_and_attributes():
    with start_trace() as ctx:
        with trace("parse", fmt="pdf"):
            time.sleep(0.005)
            with trace("chunk", count=3):
                pass
    names = [s.name for s in ctx.spans]
    assert names == ["chunk", "parse"]  # inner span closes first
    parse = next(s for s in ctx.spans if s.name == "parse")
    assert isinstance(parse, TraceSpan) and parse.attributes == {"fmt": "pdf"} and parse.duration_ms >= 5
    assert all(s.started_ms >= 0 for s in ctx.spans)


def test_trace_outside_a_context_is_a_no_op():
    with trace("orphan"):
        pass


def test_nested_traces_do_not_leak_between_contexts():
    with start_trace() as a:
        with trace("x"):
            pass
    with start_trace() as b:
        pass
    assert [s.name for s in a.spans] == ["x"] and b.spans == []


def test_otel_is_off_by_default_and_on_with_an_endpoint():
    assert configure_otel(None) is False and otel_enabled() is False
    assert configure_otel("http://localhost:4318") is True and otel_enabled() is True
    with start_trace() as ctx:
        with trace("exported", k=1):
            pass
    assert ctx.spans[0].name == "exported"
    configure_otel(None)
```

Add to `test_query_records_a_retrieval_run_with_sources_and_is_readable_by_its_owner` in `packages/server/tests/test_access_enforcement.py`, after the run is fetched over HTTP: `names = [s["name"] for s in r.json()["trace"]]; assert "hybrid_search" in names or "semantic_search" in names; assert "answer" in names`.

Run → import errors.

- [ ] **Step 2: Implement**

```python
# packages/core/src/ragfabric_core/telemetry/__init__.py
"""Tracing: one trace per request, spans stored with the run, optional OTLP export."""
```

```python
# packages/core/src/ragfabric_core/telemetry/tracing.py
"""Lightweight tracing that stores spans with the retrieval run.

Every request opens a trace; every stage opens a span. Spans are collected in
a context variable and written to retrieval_runs.trace, which is what the
Trace page reads. When telemetry.otlp_endpoint is configured, the same spans
are also emitted through the OpenTelemetry SDK to an OTLP HTTP collector, so
adopters can see them in Grafana, Jaeger or Datadog without any extra code.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager

from ragfabric_core.strategies.base import TraceSpan

_current: contextvars.ContextVar["TraceContext | None"] = contextvars.ContextVar("ragfabric_trace", default=None)
_otel_tracer = None


class TraceContext:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.spans: list[TraceSpan] = []

    def __enter__(self) -> TraceContext:
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc) -> None:
        _current.reset(self._token)


def start_trace() -> TraceContext:
    return TraceContext()


def otel_enabled() -> bool:
    return _otel_tracer is not None


def configure_otel(endpoint: str | None) -> bool:
    global _otel_tracer
    if not endpoint:
        _otel_tracer = None
        return False
    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "ragfabric"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")))
    _otel_tracer = provider.get_tracer("ragfabric")
    return True


@contextmanager
def trace(name: str, **attributes) -> Iterator[None]:
    ctx = _current.get()
    otel_span = _otel_tracer.start_span(name, attributes=attributes) if _otel_tracer is not None else None
    started = time.perf_counter()
    try:
        yield
    finally:
        if otel_span is not None:
            otel_span.end()
        if ctx is not None:
            ctx.spans.append(
                TraceSpan(
                    name=name,
                    started_ms=int((started - ctx.started) * 1000),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    attributes={k: v for k, v in attributes.items() if isinstance(v, (str, int, float, bool)) or v is None},
                )
            )
```

- [ ] **Step 3: Instrument**

- `ingest/pipeline.py` `_index_content`: wrap `parser.parse` in `with trace("parse", format=document.format):`, cleaning in `trace("clean")`, chunking in `trace("chunk", chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)`, the embed and persist block in `trace("persist_chunks", count=len(chunks))`, `schedule_indexing` in `trace("schedule_indexing", mode=get_config().ingestion.indexing)`.
- `workers/handlers.py` `index_document`: `trace("embed", model=result.model, count=len(chunks))` around the embed, `trace("vector_upsert")`, `trace("lexical_index")`.
- `retrieve/retriever.py`: wrap the body of `retrieve` in `trace("semantic_search", top_k=top_k)`; `retrieve/hybrid.py`: `trace("hybrid_search", top_k=top_k, alpha=self.alpha)`.
- `search.py` `query`: open `with start_trace() as ctx:` around retrieval and generation; wrap `build_answer` in `trace("answer", citations=len(result["citations"]))`; set `trace=[s.model_dump() for s in ctx.spans]` on the `RetrievalRun`.
- `main.py` lifespan start: `configure_otel(get_config().telemetry.otlp_endpoint)`; log whether export is on.

- [ ] **Step 4: Run, lint, commit**

```bash
uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run ruff check packages --fix -q && uv run ruff format packages -q && uv run lint-imports | tail -1
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "feat: add tracing spans on ingestion and retrieval, stored with each run, with optional OTLP export" && git log -1 --format='%h %(trailers)'
```

---

### Task 11: Compose worker service, CI integration tests on PostgreSQL, end to end verification

**Files:**
- Modify: `docker-compose.yml`, `.github/workflows/ci.yml`, `.env.example`, `docs/getting-started.md` (only the compose comment, full docs in Task 12)

- [ ] **Step 1: Compose worker**

Add after the `api` service:

```yaml
  worker:
    build:
      context: .
      dockerfile: deploy/docker/api.Dockerfile
    command: ["ragfabric", "worker"]
    env_file:
      - path: .env
        required: false
    environment:
      DATABASE_URL: postgresql+psycopg://${POSTGRES_USER:-ragfabric}:${POSTGRES_PASSWORD:-ragfabric}@postgres:5432/${POSTGRES_DB:-ragfabric}
      REDIS_URL: redis://redis:6379/0
      RAGFABRIC_CONFIG: /app/ragfabric.yaml
    volumes:
      - ./ragfabric.yaml:/app/ragfabric.yaml:ro
      - uploads:/app/data/uploads
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      api:
        condition: service_healthy
    restart: unless-stopped
```

The worker exits with code 1 when `ingestion.indexing` is `inline`; with `restart: unless-stopped` it would loop. Add `profiles: ["full", "workers"]` to the worker so the lite default does not start it, and document: `docker compose --profile workers up` (or `full`) when `indexing: queue`. Update the compose header comment accordingly.

- [ ] **Step 2: CI integration tests**

In `.github/workflows/ci.yml` `migrations` job, after the upgrade/downgrade/upgrade step add:

```yaml
      - name: Store integration tests on PostgreSQL 18
        env:
          JWT_SECRET: ci-test-secret
          DATABASE_URL: postgresql+psycopg://postgres:postgres@localhost:5432/ragfabric
          RAGFABRIC_TEST_DATABASE_URL: postgresql+psycopg://postgres:postgres@localhost:5432/ragfabric
        run: uv run pytest -m integration packages/core/tests/test_stores_postgres.py -q
```

Add `REDIS_URL` is already in `.env.example`; add `RAGFABRIC_TEST_DATABASE_URL=` with a comment "PostgreSQL URL for the store integration tests; leave empty to skip them".

- [ ] **Step 3: End to end verification on OrbStack with queue mode**

```bash
cd ~/AI/ragfabric-wt/phase-2 && cp -n .env.example .env; cp ragfabric.example.yaml ragfabric.yaml
python3 - <<'EOF'
import re,io
p='ragfabric.yaml'; s=open(p).read()
s=s.replace("  provider: openai            # openai | ollama | offline","  provider: offline            # verification run without a key",1)
s=s.replace("  indexing: inline","  indexing: queue")
open(p,'w').write(s)
EOF
docker compose --profile workers up -d --build 2>&1 | tail -2
for i in $(seq 1 40); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 3; done
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login -d "username=admin@example.com&password=adminpass123" | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
printf 'Annual leave is twelve days.\nCarry forward is capped at five days.\n' > /tmp/rf_policy.txt
DOC=$(curl -s -X POST http://localhost:8000/api/documents/upload -H "Authorization: Bearer $TOKEN" -F "file=@/tmp/rf_policy.txt" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['id'], d['status'])"); echo "uploaded: $DOC"
for i in $(seq 1 20); do ST=$(curl -s http://localhost:8000/api/documents/${DOC%% *} -H "Authorization: Bearer $TOKEN" | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])"); [ "$ST" = ready ] && break; sleep 2; done; echo "status after worker: $ST"
docker compose exec -T postgres psql -U ragfabric -d ragfabric -tAc "select (select count(*) from chunk_embeddings), (select count(*) from chunk_search), (select count(*) from documents where status='ready')"
docker compose logs worker 2>&1 | tail -3
docker compose --profile workers down
git checkout -q ragfabric.yaml 2>/dev/null || cp ragfabric.example.yaml ragfabric.yaml
```

Expected: `uploaded: <id> indexing`, `status after worker: ready`, counts `N|N|1` with N at least 1, worker log shows the job. If the status stays `indexing`, read `docker compose logs worker` and fix before continuing.

- [ ] **Step 4: Commit**

```bash
git add -A && git status --short | grep -E "^\?\? (\.env|ragfabric\.yaml)$" && echo "STOP: local files unignored" || git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "build: add the worker service, run store integration tests on PostgreSQL in CI" && git log -1 --format='%h %(trailers)'
```

---

### Task 12: Documentation sync, learning notes, tracking

**Files:**
- Modify: `README.md`, `ROADMAP.md`, `CHANGELOG.md`, `docs/README.md`, `docs/getting-started.md`, `docs/configuration.md`, `docs/architecture.md`, `docs/providers.md`, `docs/traditional-rag.md` (one line on the pgvector store)
- Create: `docs/concepts/ingestion.md`, `docs/concepts/access-control.md`
- Local only: `~/AI/ragfabric/MEMORY.md`

Steps 1 to 4 are the implementer's; Step 5 (push, PR, issue #3 and Discussion #13 comments) is the controller's after the whole branch review.

- [ ] **Step 1: README**

"What works today" gains: configurable cleaning, sections and chunking; retained originals with a download route; a pgvector vector index and a PostgreSQL full text index written by the ingestion fan out, inline or through Redis workers; groups, collection grants, document overrides and hashed API keys with rate limits; an `AccessFilter` computed per request and applied inside every store query; a retrieval run, sources and an audit row per query, readable at `/api/runs/{id}`; tracing spans with optional OTLP export; CLI commands `init`, `ingest`, `users`, `groups`, `grants`, `keys`, `worker`. State plainly that queries still run through the v1 in memory index until Phase 3 and that the new indexes are populated but not yet queried by the API. Add the open by default rule in one sentence under Access control: "A collection with no grants is open to every signed in user; the first grant restricts it to its grantees."

- [ ] **Step 2: ROADMAP, CHANGELOG**

Tick every Phase 2 line. Add to Phase 3: `- [ ] Query the pgvector index populated since Phase 2 and retire the in memory index`. CHANGELOG Unreleased: Added (each shipped item, one line), Changed (`md` accepted as a text format; pipeline chunk size and overlap come from `ragfabric.yaml`; document status gains `indexing`; `/api/search/*` accept API keys), Notes (queries still use the v1 index this phase).

- [ ] **Step 3: docs**

- `docs/getting-started.md`: real CLI flow `ragfabric init`, `ragfabric db upgrade`, `ragfabric users create`, `ragfabric ingest ./docs --collection handbook`, `ragfabric keys create`, `ragfabric worker` with `indexing: queue` and `docker compose --profile workers up`.
- `docs/configuration.md`: `ingestion.uploads_dir`, `ingestion.indexing`, `RAGFABRIC_TEST_DATABASE_URL`; the workers profile; the note that `llm` is still not read by the runtime but `embeddings`, `vector_store`, `lexical_store`, `cache` and `ingestion` now are.
- `docs/architecture.md`: add `queue/`, `workers/`, `telemetry/`, `stores/*` implementations, `auth/{service,policy,api_keys,ratelimit}.py`, `ingest/{clean,storage,indexing}.py`; the request flow gains "compute AccessFilter" and "write RetrievalRun, Source, AuditLog"; the ingestion diagram gains the fan out.
- `docs/providers.md`: pgvector and PostgreSQL full text rows become "Phase 2 (shipped, fed by ingestion; queried from Phase 3)"; Redis and memory cache "Phase 2 (shipped)"; API keys "Phase 2 (shipped)".
- `docs/concepts/ingestion.md` (learning notes): why clean before chunk, chunk size and overlap trade offs with the two numbers we chose and what changing them does, why metadata (page, span, section, document type) must survive every stage, inline versus queued indexing, why both indexes are written in one job.
- `docs/concepts/access-control.md` (learning notes): principals, groups, grants, overrides, API key scopes; the open by default rule and how to close a collection; filter before rank with the SQL predicate example; audit rows; what Phase 2 does not do yet (row level security in the database, OIDC).
- `docs/README.md`: add the two concept documents, update statuses.

Grep for em dashes and for claims of unshipped features (`ragfabric ask`, `Reranker`, `bm25`, `console`) before committing; both must be clean.

- [ ] **Step 4: Full verification and commit**

```bash
uv run ruff check packages && uv run ruff format --check packages && uv run lint-imports
uv run pytest -q -p no:warnings 2>&1 | tail -1
(cd apps/assistant && npm test 2>&1 | tail -2)
docker compose config --quiet && echo compose ok
git add -A && git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -q -m "docs: sync README, roadmap, changelog and docs with Phase 2, add ingestion and access control notes" && git log -1 --format='%h %(trailers)'
```

- [ ] **Step 5 (controller, after the whole branch review): push, PR, tracking**

Push `feat/phase-2-ingestion-access`, open the PR against `main` with the standard template (What, Phase / area, How it was tested with the real counts, Notes, Checklist; "closes #3"), comment on issue #3 with the PR number, comment on Discussion #13, update `~/AI/ragfabric/MEMORY.md`.

---

## Self review

**Spec coverage.** Design section 5 (interfaces): Tasks 4, 5 implement `VectorStore`, `LexicalStore`, `Cache` and add the queue protocol. Section 7 (access control): Tasks 6, 7, 8 cover roles, groups, grants, overrides, API keys, enforcement inside retrieval, audit. Section 8 (observability): Task 10 spans stored per run with OTLP export. ADR 0003: Tasks 4 and 8 (predicate in the store query, filter passed into the v1 store before ranking). Issue #3 checklist: cleaning and sections (T2), configurable chunking (T2), retained originals (T3), fan out via Redis workers (T5, T11), groups, grants, overrides, keys, audit (T6, T7, T8), filter inside every store query (T4, T8), CLI (T9), OTel spans (T10), learning notes (T12). Exit criteria: Task 8's enforcement test and Task 11's end to end check.

**Placeholder scan.** No TBD or TODO. Task 12 describes documentation edits in prose with the exact facts to state; that matches the Phase 1 pattern.

**Type consistency.** `AccessFilter.allows(document_id, collection_id)` and the three frozenset fields are used identically in Tasks 4, 6, 8. `RetrievedChunk` construction in the two stores uses the Phase 1 fields. `Job` and `JobQueue` names match between Tasks 5, 9 and 11. `index_document(db, document_id, *, embedding_provider, vector_store, lexical_store)` matches its callers in Tasks 5 and 9. `get_principal`, `get_access_filter`, `get_cache` are defined in Task 7 and consumed in Tasks 8 and 9's routes. `start_trace`, `trace`, `configure_otel` match between Task 10's tests and instrumentation. Document status values `processing`, `indexing`, `ready`, `failed` are used consistently in Tasks 5, 9, 11.
