# Phase 3: Traditional RAG, Real Embeddings, Reranking, Cited Generation, SDK. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer real questions from the pgvector index that Phase 2 has been filling, using a real embedding model, an optional reranker, a context budget and an LLM that writes a cited answer; retire the v1 in memory index; add Chroma as a second vector store; ship `POST /api/ask` with SSE streaming, the Python SDK, `ragfabric ask` and `ragfabric reindex`; and clear every deferred item from the Phase 1 and Phase 2 reviews.

**Architecture:** `TraditionalRAGStrategy` replaces `LegacyHybridStrategy` behind the unchanged `RetrieverStrategy` interface. It embeds the question with the configured `EmbeddingProvider`, queries the configured `VectorStore` (pgvector or Chroma) with the `AccessFilter` inside the query as ADR 0003 requires, applies a similarity threshold, reranks, fits the result to a token budget, and hands the chunks to a cited generator. The embedding model is pinned per deployment: migration 0004 fixes the vector column dimension and builds an HNSW index, every vector query filters on the active model name, and `ragfabric reindex` re-embeds the corpus when the model changes. The server keeps `POST /api/search/query` on its existing response shape and adds `POST /api/ask` for streaming.

**Tech Stack:** Python 3.13, uv, SQLAlchemy 2.0, Alembic, pgvector 0.5 with HNSW and `vector_cosine_ops`, chromadb 1.5, Ollama (`nomic-embed-text` 768d for embeddings, `llama3.2:3b` for generation), tiktoken, sentence-transformers behind the `rerank` extra, FastAPI with `StreamingResponse` for SSE, httpx for the SDK, Typer, pytest.

**Spec:** `docs/design/2026-09-13-ragfabric-design.md` sections 3, 5, 6; ADR 0002 (one result shape), ADR 0003 (access inside retrieval), ADR 0004 (no fabricated numbers), and the new ADR 0006 written in Task 3. Roadmap issue #4.

## Global Constraints

- Python `>=3.13,<3.14`. No em dashes in any file written. No AI assistant references anywhere in code, docs or commit messages. Commits authored `Ranjan G <ranjan.g@ispf.ngo>` via `git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit ...`, conventional prefixes, NO trailers of any kind; verify with `git log -1 --format='%(trailers)'` after every commit.
- No secrets in the repository. Provider keys come from the environment only.
- `ragfabric_core` never imports `ragfabric_server` or `ragfabric_cli`. `ragfabric_sdk` never imports `ragfabric_core`; it talks to the API over HTTP only. `uv run lint-imports` must end this phase with 3 contracts, not 2.
- Every store query takes and applies an `AccessFilter` before ranking (ADR 0003). No post ranking filtering anywhere new. The contract test `assert_strategy_contract` must pass for `TraditionalRAGStrategy`.
- All 225 Phase 2 tests keep passing unless this plan deletes the module under test. Deletions are explicit: a task that removes a module also removes its tests in the same commit and says so. Assertions in surviving tests are never weakened.
- The default `ragfabric.yaml` and `ragfabric.example.yaml` ship `embeddings.provider: ollama`, `embeddings.model: nomic-embed-text`, `embeddings.dim: 768`, `llm.provider: ollama`, `llm.model: llama3.2:3b`, `llm.base_url: http://localhost:11434/v1`. This is the no key path. OpenAI and Anthropic stay fully supported and are covered by tests that skip without a key.
- Unit tests never call a network. They use `HashingEmbeddingProvider` and `ScriptedLLMProvider` from `providers/offline.py`. Tests that need Ollama are marked `integration` and skip when `RAGFABRIC_TEST_OLLAMA` is unset. Tests that need PostgreSQL are marked `integration` and skip when `RAGFABRIC_TEST_DATABASE_URL` is unset. Tests that need Chroma are marked `integration` and skip when `RAGFABRIC_TEST_CHROMA_URL` is unset.
- The vector dimension pinned by migration 0004 is **768**. It is a constant in the migration, not a value read from config, so the migration is deterministic. Changing the embedding model to another dimension needs a new migration plus `ragfabric reindex`; the runbook for that is written in Task 17.
- Work on branch `feat/phase-3-traditional-rag` in a worktree at `~/AI/ragfabric-wt/phase-3`. Push to the feature branch is pre authorised; the PR opens only after the whole branch review; merging follows the owner's standing instruction of 2026-09-14.

---

## Decision log

Every non-obvious choice in this phase, the alternatives rejected, and what it costs if the
choice turns out to be wrong. This exists because a self hosted platform is forked and
customised by people who were not in the room: a decision without a recorded reason becomes
folklore, and folklore does not survive a fork. Anything here that changes during
implementation is recorded the same way in the ledger, and anything architectural graduates
to an ADR.

| # | Decision | Alternatives rejected | Cost if wrong |
|---|---|---|---|
| D1 | Ollama with `nomic-embed-text` and `llama3.2:3b` is the shipped default | OpenAI as the default (needs a paid key before the first query, so the project cannot be evaluated by a stranger in ten minutes); `offline` hashing as the default (returns results a new adopter would reasonably read as broken retrieval) | Two lines of yaml. Every provider stays configurable and tested |
| D2 | One pinned embedding model and dimension per deployment (ADR 0006) | See the ADR's own alternatives table | A migration plus a reindex to move to another dimension, which is the documented path anyway |
| D3 | Vectors stored L2 normalised | Normalise at query time only (leaves stored data non comparable across dialects); store raw and rely on pgvector cosine (makes the SQLite development path rank differently from production, so a developer cannot trust a local result) | One pass of `ragfabric reindex` |
| D4 | `/api/search/query` keeps its response shape and is rewired underneath; `/api/ask` is the new streaming endpoint | Deprecating `/query` now (breaks the shipped Angular app inside a phase that has no frontend budget); replacing it outright (a breaking public API change before v0.1.0 has shipped, for no functional gain) | Two endpoints to maintain until Phase 9 retires one |
| D5 | Both pgvector and Chroma ship in this phase | pgvector only (faster phase, but the pluggable store claim would rest on one real implementation plus a test double, which is not evidence a third party adapter is possible) | One module, one container in the `full` profile, one integration test file |
| D6 | All three reranker kinds ship, with `cross_encoder` behind the `rerank` extra | Interface only (leaves a documented config value raising NotImplementedError, which is the kind of half promise that erodes trust in a config file); cross encoder as a hard dependency (about 2 GB of torch in every install, including API containers that never rerank) | An extras line in `pyproject.toml` and a clear error message |
| D7 | The citation contract is redefined rather than kept as is | Keep "every clause is a substring of its chunk" (true only for extractive output, so it would forbid LLM generation entirely); drop the contract while an LLM writes the answer (removes the one mechanical check that a citation is not invented) | The contract's precise form, which Task 10 encodes in one assertion module |
| D8 | The extractive generator stays as a supported `generation: extractive` mode | Delete it now that a real LLM exists (loses the only generation path that needs no model at all, which is also the fixture the citation tests run against) | One config value and the module that already exists |
| D9 | `ragfabric reindex` re-embeds from `chunks` rather than re-ingesting files | Re-ingest everything (re-parses, re-chunks, renumbers chunk ids, and orphans every `Source` row from every answer already recorded) | The command exists either way; this is about which table it reads |
| D10 | All 20 deferred review items are cleared in one task in this phase | Defer again (the list has grown for two phases and two of its items already cause flaky parallel test runs) | A larger review diff in one task, flagged to the reviewer up front |

---

## File structure after this phase (new or changed)

```
packages/core/src/ragfabric_core/
  embeddings/
    normalise.py              NEW  L2 normalisation shared by every store and the reindexer
  models/
    index.py                  MOD  ChunkEmbedding gains nothing structural; docstring updated for the pinned dim
  migrations/versions/
    0004_pin_vector_dim_and_hnsw.py   NEW  pin vector(768), HNSW cosine index, index on model, drop stale rows
  stores/
    pgvector_store.py         MOD  model filter, normalised vectors, similarity returned as cosine on both dialects
    chroma_store.py           NEW  ChromaVectorStore over chromadb, access filter translated to a where clause
    registry.py               MOD  chroma kind, embedding_model passed through, Phase 4 message for bm25
  rerank/
    __init__.py               NEW
    base.py                   NEW  Reranker protocol
    noop.py                   NEW  NoopReranker
    llm_reranker.py           NEW  LlmReranker, scores each chunk with the configured LLM
    cross_encoder.py          NEW  CrossEncoderReranker, sentence-transformers, optional extra
    registry.py               NEW  build_reranker(cfg, llm_provider)
  tokens.py                   NEW  count_tokens and fit_to_budget
  strategies/
    traditional.py            NEW  TraditionalRAGStrategy
    legacy.py                 DEL  replaced by traditional.py
    registry_defaults.py      NEW  default_registry() building the strategy registry from config
  generate/
    cited.py                  NEW  generate_cited_answer over an LLMProvider, streaming and non streaming
    contract.py               NEW  assert_citation_contract
    answer.py                 MOD  packaging unchanged; answer text now comes from cited.py when an LLM is configured
    llm.py                    MOD  extractive path kept as the offline fallback, Claude specific branch removed
  retrieve/                   DEL  retriever.py and hybrid.py, replaced by the strategy
  store/                      DEL  vector_store.py, the v1 in memory index
  ingest/
    reindex.py                NEW  reindex_all, batched re-embedding with progress callbacks
  providers/
    base.py                   MOD  LLMProvider gains stream()
    openai_compat.py          MOD  stream() for OpenAI and Ollama
    anthropic_provider.py     MOD  stream()
    offline.py                MOD  stream() on ScriptedLLMProvider

packages/server/src/ragfabric_server/
  api/routes/search.py        MOD  /query and /semantic on the new path, /hybrid on pgvector plus postgres_fts
  api/routes/ask.py           NEW  POST /api/ask, SSE
  schemas/ask.py              NEW  AskRequest, AskEvent
  schemas/search.py           MOD  SearchRequest gains similarity_threshold and rerank
  deps.py                     MOD  get_strategy_registry, get_vector_store, get_llm_provider

packages/sdk-python/          NEW  ragfabric_sdk: Client, ask, search, ingest, documents, typed models
packages/cli/src/ragfabric_cli/commands/
  ask.py                      NEW  ragfabric ask
  reindex.py                  NEW  ragfabric reindex

docs/adr/0006-pinned-embedding-model-and-dimension.md   NEW
docs/concepts/reranking.md                              NEW
docs/traditional-rag.md, configuration.md, architecture.md, providers.md, getting-started.md, README.md, ROADMAP.md, CHANGELOG.md   MOD
```

---

### Task 1: Worktree, dependencies, and the no key default configuration

**Files:**
- Modify: `packages/core/pyproject.toml`, `ragfabric.example.yaml`, `ragfabric.yaml`, `.env.example`
- Test: `packages/core/tests/test_config_file.py` (additions)

**Interfaces:**
- Consumes: nothing; this is the first task.
- Produces: `chromadb>=1.5` and `tiktoken>=0.12` in core dependencies; an optional dependency group `rerank = ["sentence-transformers>=5.1"]`; the example and live yaml defaulting to Ollama as stated in the Global Constraints.

- [ ] **Step 1: Create the worktree and confirm the baseline**

```bash
cd ~/AI/ragfabric && git fetch -q origin && git checkout -q main && git pull -q --ff-only
git worktree add -b feat/phase-3-traditional-rag ~/AI/ragfabric-wt/phase-3 main
# This plan lives untracked in the main working tree. It becomes the first
# commit on the branch, so the worktree carries its own instructions and a
# fresh implementer needs nothing from outside the checkout.
mkdir -p ~/AI/ragfabric-wt/phase-3/docs/plans
cp ~/AI/ragfabric/docs/plans/2026-09-18-phase-3-traditional-rag.md \
   ~/AI/ragfabric-wt/phase-3/docs/plans/
cd ~/AI/ragfabric-wt/phase-3
git add docs/plans/2026-09-18-phase-3-traditional-rag.md
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" \
    commit -m "docs: add the Phase 3 traditional RAG implementation plan"
git log -1 --format='%(trailers)'
uv sync -q && uv run pytest -q -p no:warnings 2>&1 | tail -1
```

Expected: empty trailers, then `225 passed, 3 skipped`. If the test number differs, stop and report before changing anything: the baseline is what every later count in this plan is measured against.

- [ ] **Step 2: Pull the embedding model**

```bash
ollama pull nomic-embed-text
ollama list | grep nomic
```

Expected: a `nomic-embed-text` row. This is the only external download the phase needs and it is free.

- [ ] **Step 3: Write the failing config test**

Add to `packages/core/tests/test_config_file.py`:

```python
def test_default_example_config_is_the_no_key_ollama_path(tmp_path):
    from pathlib import Path

    from ragfabric_core.config_file import load_config

    example = Path(__file__).resolve().parents[3] / "ragfabric.example.yaml"
    cfg = load_config(example)
    assert cfg.embeddings.provider == "ollama"
    assert cfg.embeddings.model == "nomic-embed-text"
    assert cfg.embeddings.dim == 768
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "llama3.2:3b"
    assert cfg.llm.base_url == "http://localhost:11434/v1"
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv run pytest packages/core/tests/test_config_file.py::test_default_example_config_is_the_no_key_ollama_path -q`
Expected: FAIL, `assert 'openai' == 'ollama'`.

- [ ] **Step 5: Edit both yaml files**

In `ragfabric.example.yaml` and `ragfabric.yaml`, replace the `llm` and `embeddings` blocks with:

```yaml
llm:
  provider: ollama            # openai | anthropic | ollama | offline
  model: llama3.2:3b          # openai default: gpt-5.4-mini, anthropic default: claude-sonnet-5
  base_url: http://localhost:11434/v1   # any OpenAI compatible endpoint

embeddings:
  provider: ollama            # openai | ollama | offline
  model: nomic-embed-text     # 768 dimensions, pinned by migration 0004
  dim: 768                    # must match the migration; see docs/configuration.md
  base_url: http://localhost:11434/v1
```

While in `ragfabric.yaml`, add the two Phase 2 keys it is missing so the live file matches the example:

```yaml
ingestion:
  chunk_size: 600
  chunk_overlap: 80
  retain_originals: true
  uploads_dir: data/uploads
  indexing: inline
```

- [ ] **Step 6: Add the dependencies**

In `packages/core/pyproject.toml`, add to `dependencies`:

```toml
  "chromadb>=1.5",
  "tiktoken>=0.12",
```

and add a new block after `dependencies`:

```toml
[project.optional-dependencies]
rerank = ["sentence-transformers>=5.1"]
```

- [ ] **Step 7: Sync and run the test**

Run: `uv sync -q && uv run pytest packages/core/tests/test_config_file.py -q`
Expected: PASS.

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: `225 passed, 3 skipped`. The config change must not break anything, because nothing reads a live provider in unit tests.

- [ ] **Step 9: Commit**

```bash
git add packages/core/pyproject.toml uv.lock ragfabric.example.yaml ragfabric.yaml packages/core/tests/test_config_file.py
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "chore: default to the Ollama no key path and add chroma, tiktoken and the rerank extra"
git log -1 --format='%(trailers)'
```

Expected: the trailers output is empty.

---

### Task 2: Normalised vectors and the active model filter on the vector store

**Files:**
- Create: `packages/core/src/ragfabric_core/embeddings/__init__.py`, `packages/core/src/ragfabric_core/embeddings/normalise.py`
- Modify: `packages/core/src/ragfabric_core/stores/pgvector_store.py`, `packages/core/src/ragfabric_core/stores/registry.py`, `packages/core/src/ragfabric_core/ingest/indexing.py`, `packages/core/src/ragfabric_core/workers/handlers.py`
- Test: `packages/core/tests/test_normalise.py`, additions to `packages/core/tests/test_stores_sqlite.py` and `packages/core/tests/test_stores_postgres.py`

**Interfaces:**
- Consumes: Task 1's configuration defaults.
- Produces: `normalise(vector: list[float]) -> list[float]` (L2, returns the input unchanged when the norm is 0); `normalise_all(vectors: list[list[float]]) -> list[list[float]]`; `PgVectorStore(session_factory, model: str | None = None)` where a non None `model` adds `ChunkEmbedding.model == model` to every query; `build_vector_store(cfg, session_factory, embedding_model: str | None = None)`.

Why this task exists: `PgVectorStore.query` today ranks every row in `chunk_embeddings` regardless of which model wrote it, so the moment Task 4 re-embeds a corpus, vectors from two models would compete in one ranking. It also scores with a raw dot product on SQLite and cosine distance on PostgreSQL, so the same data ranks differently on the two dialects unless the vectors are unit length. Normalising at write time makes the dot product equal to cosine and makes the two paths agree.

- [ ] **Step 1: Write the failing normalisation test**

Create `packages/core/tests/test_normalise.py`:

```python
import math

from ragfabric_core.embeddings.normalise import normalise, normalise_all


def test_normalise_returns_a_unit_vector():
    out = normalise([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(v * v for v in out)), 1.0, rel_tol=1e-9)
    assert math.isclose(out[0], 0.6, rel_tol=1e-9)
    assert math.isclose(out[1], 0.8, rel_tol=1e-9)


def test_normalise_leaves_a_zero_vector_alone_rather_than_dividing_by_zero():
    assert normalise([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_normalise_all_maps_every_row():
    out = normalise_all([[3.0, 4.0], [0.0, 5.0]])
    assert math.isclose(out[1][1], 1.0, rel_tol=1e-9)
    assert len(out) == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest packages/core/tests/test_normalise.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragfabric_core.embeddings'`.

- [ ] **Step 3: Write the module**

Create `packages/core/src/ragfabric_core/embeddings/__init__.py` as an empty file, then `packages/core/src/ragfabric_core/embeddings/normalise.py`:

```python
"""L2 normalisation, applied once at write time.

Why here and not inside each store: pgvector ranks with cosine distance and the
SQLite fallback ranks with a dot product. Those two agree only when the vectors
are unit length. Normalising on the way in makes every store, every dialect and
the reindexer score identically, and makes a stored score directly comparable
across models.
"""

from __future__ import annotations

import math


def normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm == 0.0:
        return [float(v) for v in vector]
    return [float(v) / norm for v in vector]


def normalise_all(vectors: list[list[float]]) -> list[list[float]]:
    return [normalise(v) for v in vectors]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest packages/core/tests/test_normalise.py -q`
Expected: 3 passed.

- [ ] **Step 5: Write the failing model filter test**

Add to `packages/core/tests/test_stores_sqlite.py`:

```python
def test_query_ignores_rows_written_by_another_embedding_model(session_factory, seed_chunks):
    from ragfabric_core.auth.principal import AccessFilter
    from ragfabric_core.stores.pgvector_store import PgVectorStore

    old = PgVectorStore(session_factory, model="hashing-384")
    new = PgVectorStore(session_factory, model="nomic-embed-text")
    chunk_ids = seed_chunks
    old.upsert(
        [chunk_ids[0]],
        [[1.0, 0.0]],
        [{"document_id": 1, "collection_id": None, "model": "hashing-384", "dim": 2}],
    )
    new.upsert(
        [chunk_ids[1]],
        [[1.0, 0.0]],
        [{"document_id": 1, "collection_id": None, "model": "nomic-embed-text", "dim": 2}],
    )

    hits = new.query([1.0, 0.0], top_k=10, access=AccessFilter.unrestricted())

    assert [h.chunk_id for h in hits] == [chunk_ids[1]], (
        "a query for one model returned a row written by another model"
    )
```

If `seed_chunks` does not already exist in that file, add this fixture beside it:

```python
import pytest


@pytest.fixture
def seed_chunks(session_factory):
    """Two chunks on one document, returned as their ids."""
    from ragfabric_core.models.document import Chunk, Document

    with session_factory() as db:
        doc = Document(filename="f.txt", format="txt", status="ready", owner_id=None)
        db.add(doc)
        db.flush()
        rows = [
            Chunk(document_id=doc.id, collection_id=None, text="alpha", chunk_index=0),
            Chunk(document_id=doc.id, collection_id=None, text="beta", chunk_index=1),
        ]
        db.add_all(rows)
        db.commit()
        return [r.id for r in rows]
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest packages/core/tests/test_stores_sqlite.py -q -k another_embedding_model`
Expected: FAIL, `TypeError: PgVectorStore.__init__() got an unexpected keyword argument 'model'`.

- [ ] **Step 7: Add the model filter and normalisation to the store**

In `packages/core/src/ragfabric_core/stores/pgvector_store.py`:

Change the constructor:

```python
    def __init__(self, session_factory: Callable[[], Session], model: str | None = None) -> None:
        self._sf = session_factory
        self._model = model
```

In `upsert`, normalise before storing. Replace the `embedding=list(map(float, vector))` line with:

```python
                    embedding=normalise(vector),
```

and add the import at the top:

```python
from ragfabric_core.embeddings.normalise import normalise
```

In `query`, normalise the query vector and filter on the model. Immediately after `clause = access_clause(...)` add:

```python
        vector = normalise(vector)
```

and immediately after `if clause is not None: base = base.where(clause)` add:

```python
            if self._model is not None:
                base = base.where(ChunkEmbedding.model == self._model)
```

- [ ] **Step 8: Run the store tests**

Run: `uv run pytest packages/core/tests/test_stores_sqlite.py -q`
Expected: all pass, including the new one.

- [ ] **Step 9: Pass the model through the registry and the fan out**

In `packages/core/src/ragfabric_core/stores/registry.py`, change the vector builder:

```python
def build_vector_store(
    cfg: VectorStoreConfig,
    session_factory: Callable[[], Session],
    embedding_model: str | None = None,
) -> VectorStore:
    if cfg.kind in ("pgvector", "memory"):
        return PgVectorStore(session_factory, model=embedding_model)
    raise NotImplementedError(f"vector store {cfg.kind!r} arrives in Phase 3")
```

In `packages/core/src/ragfabric_core/ingest/indexing.py`, change `_stores` so the store knows the active model:

```python
def _stores():
    cfg = get_config()
    sf = get_session_factory()
    provider = _embedding_provider()
    return (
        build_vector_store(cfg.vector_store, sf, embedding_model=provider.model),
        build_lexical_store(cfg.lexical_store, sf),
    )
```

and change `index_inline` so the provider is built once rather than twice:

```python
def index_inline(db: Session, document: Document) -> int:
    cfg = get_config()
    sf = get_session_factory()
    provider = _embedding_provider()
    vector_store = build_vector_store(cfg.vector_store, sf, embedding_model=provider.model)
    lexical_store = build_lexical_store(cfg.lexical_store, sf)
    return index_document(
        db,
        document.id,
        embedding_provider=provider,
        vector_store=vector_store,
        lexical_store=lexical_store,
    )
```

- [ ] **Step 10: Run the whole suite**

Run: `uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: `229 passed, 3 skipped` (225 plus 3 normalisation tests plus 1 store test).

- [ ] **Step 11: Run the PostgreSQL integration tests**

```bash
docker compose --profile lite up -d postgres
export RAGFABRIC_TEST_DATABASE_URL="postgresql+psycopg://ragfabric:ragfabric@localhost:5432/ragfabric_test"
uv run pytest packages/core/tests/test_stores_postgres.py -q -m integration
```

Expected: the existing PostgreSQL store tests still pass. If the connection string differs in `.env.example`, use that one.

- [ ] **Step 12: Commit**

```bash
git add packages/core/src/ragfabric_core/embeddings packages/core/src/ragfabric_core/stores/pgvector_store.py packages/core/src/ragfabric_core/stores/registry.py packages/core/src/ragfabric_core/ingest/indexing.py packages/core/tests/test_normalise.py packages/core/tests/test_stores_sqlite.py
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: normalise stored vectors and filter vector queries by the active embedding model"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 3: ADR 0006, migration 0004, pinned dimension and the HNSW index

**Files:**
- Create: `docs/adr/0006-pinned-embedding-model-and-dimension.md`, `packages/core/src/ragfabric_core/migrations/versions/0004_pin_vector_dim_and_hnsw.py`
- Modify: `packages/core/src/ragfabric_core/models/index.py` (docstring only)
- Test: additions to `packages/core/tests/test_migrations.py`

**Interfaces:**
- Consumes: Task 2's normalised vectors.
- Produces: revision `0004_pin_vector_dim_and_hnsw`, down revision `0003_ingestion_and_indexes`. On PostgreSQL: deletes rows whose `dim` is not 768, alters `chunk_embeddings.embedding` to `vector(768)`, creates `ix_chunk_embeddings_hnsw` using HNSW with `vector_cosine_ops`, creates `ix_chunk_embeddings_model` on `model`. On SQLite: creates `ix_chunk_embeddings_model` only, because a JSON column has no vector index and no fixed width.
- Constant: `EMBEDDING_DIM = 768` inside the migration module.

Why the delete is in the migration and not left to the operator: `ALTER COLUMN ... TYPE vector(768)` fails outright on any row with a different width, so the upgrade either removes the stale hashing vectors or does not run at all. The rows are reproducible from `chunks` by `ragfabric reindex` (Task 4), so nothing unrecoverable is lost. This is stated in ADR 0006 and repeated in the migration docstring and the release notes.

- [ ] **Step 1: Write ADR 0006**

Create `docs/adr/0006-pinned-embedding-model-and-dimension.md`:

```markdown
# ADR 0006: One pinned embedding model and dimension per deployment

Status: accepted
Date: 2026-09-18
Supersedes: none
Related: ADR 0003 (access control inside retrieval)

## Context

Phase 2 wrote `chunk_embeddings` rows with a `model` name and a `dim` per row, and left
the vector column unconstrained so one migration could serve both PostgreSQL and SQLite.
That was right for a table nothing queried. Phase 3 queries it, and two facts now bite:

1. pgvector cannot build an HNSW or IVFFlat index on a `vector` column with no declared
   dimension. Without an index every query is a sequential scan over the whole corpus.
2. Vectors from two different models occupy different spaces. Ranking them against each
   other produces confident nonsense, not an error.

## Decision

A deployment pins one embedding model and one dimension.

- The vector column is `vector(768)`, fixed by migration 0004. 768 is `nomic-embed-text`,
  the default no key model.
- An HNSW index with `vector_cosine_ops` is created on that column.
- Every vector query filters on the active model name, so a row written by another model
  can never enter a ranking even if it is present.
- Vectors are stored L2 normalised, so cosine distance and dot product agree and a score
  means the same thing on PostgreSQL and on the SQLite development path.
- Changing the embedding model requires `ragfabric reindex`. Changing to a model with a
  different dimension additionally requires a one line migration. Both are documented in
  `docs/configuration.md`.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| Leave the column unconstrained and accept sequential scans | Honest but slow, and it makes the product's own benchmark numbers a function of corpus size rather than of the retrieval design. An adopter with 100k chunks would measure our scan, not our ranking. |
| Store a dimension per row and build one partial index per dimension | pgvector indexes cannot be built on an unconstrained column at all, partial or otherwise, so this does not exist as an option. |
| One table per embedding model | Supports two models at once, at the cost of a dynamic table name in every query, a migration per model, and an access predicate duplicated per table. The benefit is an evaluation feature, and Phase 8 can get it with a second index rather than a second schema. |
| Keep both models' rows and disambiguate only at query time | The model filter alone does make ranking correct, and this task ships it. It is not sufficient: the column still cannot be indexed, and stale rows grow the table forever with no process to remove them. |
| Read the dimension from `ragfabric.yaml` inside the migration | Makes `alembic upgrade head` produce different schemas on different machines from the same revision, which breaks the drift test, CI reproducibility and any support conversation about "what does your schema look like". |

## Consequences

- Migration 0004 deletes `chunk_embeddings` rows whose dimension is not 768, because the
  column type change cannot succeed otherwise. Those rows are derived data, rebuilt by
  `ragfabric reindex` from `chunks`, which is the source of truth. No user content is lost.
- A deployment cannot serve two embedding models at once. That is deliberate. Comparing
  embedding models is an evaluation concern and belongs to Phase 8, where it can be done
  against separate indexes with measured results rather than silently inside one ranking.
- The pinned dimension is a constant in the migration, not a value read from configuration,
  so `alembic upgrade head` produces the same schema on every machine.
```

- [ ] **Step 2: Write the failing migration tests**

Add to `packages/core/tests/test_migrations.py`:

```python
def test_0004_creates_the_model_index_on_every_dialect(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    migrate.upgrade(url, "head")
    engine = sa.create_engine(url)
    names = {ix["name"] for ix in sa.inspect(engine).get_indexes("chunk_embeddings")}
    assert "ix_chunk_embeddings_model" in names


def test_0004_downgrade_returns_to_0003(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    migrate.upgrade(url, "head")
    migrate.downgrade(url, "0003_ingestion_and_indexes")
    engine = sa.create_engine(url)
    names = {ix["name"] for ix in sa.inspect(engine).get_indexes("chunk_embeddings")}
    assert "ix_chunk_embeddings_model" not in names
    assert "chunk_embeddings" in sa.inspect(engine).get_table_names()
```

Use the same import style the file already uses for `sa` and `migrate`; do not add duplicate imports.

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest packages/core/tests/test_migrations.py -q -k 0004`
Expected: FAIL, the index is not there.

- [ ] **Step 4: Write the migration**

Create `packages/core/src/ragfabric_core/migrations/versions/0004_pin_vector_dim_and_hnsw.py`:

```python
"""Pin the vector dimension and build the HNSW index.

pgvector cannot index a vector column with no declared dimension, so Phase 3
fixes the column at 768 (nomic-embed-text, the default no key model) and builds
an HNSW index with cosine ops.

Rows whose dim is not 768 are deleted first, because the type change cannot
succeed while they exist. They are derived data: `ragfabric reindex` rebuilds
them from the chunks table, which is the source of truth. See ADR 0006.

Revision ID: 0004_pin_vector_dim_and_hnsw
Revises: 0003_ingestion_and_indexes
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_pin_vector_dim_and_hnsw"
down_revision: str | None = "0003_ingestion_and_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    bind = op.get_bind()
    op.create_index("ix_chunk_embeddings_model", "chunk_embeddings", ["model"])
    if bind.dialect.name != "postgresql":
        return
    op.execute(sa.text(f"DELETE FROM chunk_embeddings WHERE dim <> {EMBEDDING_DIM}"))
    op.execute(
        sa.text(
            f"ALTER TABLE chunk_embeddings "
            f"ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) "
            f"USING embedding::vector({EMBEDDING_DIM})"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX ix_chunk_embeddings_hnsw ON chunk_embeddings "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(sa.text("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw"))
        op.execute(sa.text("ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector"))
    op.drop_index("ix_chunk_embeddings_model", table_name="chunk_embeddings")
```

- [ ] **Step 5: Run the migration tests**

Run: `uv run pytest packages/core/tests/test_migrations.py -q`
Expected: all pass, including the drift test `test_migrations_match_the_models`. If drift fails, the model docstring is not the problem; check that no column definition changed. Nothing in this task changes a column on SQLite.

- [ ] **Step 6: Update the model docstring**

In `packages/core/src/ragfabric_core/models/index.py`, replace the sentence `The dimension is stored per row; an HNSW index needs a fixed dimension and is created by Phase 3 once the deployment's embedding model is known.` with:

```
The dimension is stored per row for provenance, but since migration 0004 the
PostgreSQL column is fixed at vector(768) with an HNSW cosine index, and every
query filters on the active model name. One deployment, one embedding model
(ADR 0006).
```

- [ ] **Step 7: Verify the index really exists on PostgreSQL**

```bash
docker compose --profile lite up -d postgres
export RAGFABRIC_TEST_DATABASE_URL="postgresql+psycopg://ragfabric:ragfabric@localhost:5432/ragfabric_test"
uv run python -c "
from ragfabric_core.db import migrate
import os, sqlalchemy as sa
url = os.environ['RAGFABRIC_TEST_DATABASE_URL']
migrate.upgrade(url, 'head')
e = sa.create_engine(url)
with e.connect() as c:
    rows = c.execute(sa.text(\"select indexname from pg_indexes where tablename='chunk_embeddings'\")).scalars().all()
print(sorted(rows))
"
```

Expected: the list contains `ix_chunk_embeddings_hnsw` and `ix_chunk_embeddings_model`.

- [ ] **Step 8: Commit**

```bash
git add docs/adr/0006-pinned-embedding-model-and-dimension.md packages/core/src/ragfabric_core/migrations/versions/0004_pin_vector_dim_and_hnsw.py packages/core/src/ragfabric_core/models/index.py packages/core/tests/test_migrations.py
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: pin the vector dimension at 768 and build the HNSW cosine index"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 4: `ragfabric reindex`, batched re-embedding

**Files:**
- Create: `packages/core/src/ragfabric_core/ingest/reindex.py`, `packages/cli/src/ragfabric_cli/commands/reindex.py`
- Modify: `packages/cli/src/ragfabric_cli/main.py`
- Test: `packages/core/tests/test_reindex.py`, additions to `packages/cli/tests/test_cli_phase3.py` (created here)

**Interfaces:**
- Consumes: Task 2's `build_vector_store(..., embedding_model=...)`, Task 3's pinned dimension.
- Produces: `reindex_all(db, *, embedding_provider, vector_store, lexical_store, batch_size: int = 64, on_progress: Callable[[int, int], None] | None = None) -> int` returning the number of chunks re-embedded; it processes every chunk of every document in id order, in batches, and deletes each document's stale vectors before writing new ones. `ragfabric reindex [--batch-size N] [--document-id ID] [--yes]`.

Why a dedicated command rather than re-running ingestion: re-ingesting would re-parse and re-chunk every file, which changes chunk ids and breaks every `Source` row pointing at them. Re-embedding reads the existing `chunks` rows and only replaces vectors, so retrieval runs recorded before the model change still resolve.

- [ ] **Step 1: Write the failing test**

Create `packages/core/tests/test_reindex.py`:

```python
import pytest

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.ingest.reindex import reindex_all
from ragfabric_core.providers.offline import HashingEmbeddingProvider
from ragfabric_core.stores.pgvector_store import PgVectorStore


@pytest.fixture
def seeded(session_factory):
    from ragfabric_core.models.document import Chunk, Document

    with session_factory() as db:
        doc = Document(filename="f.txt", format="txt", status="ready", owner_id=None)
        db.add(doc)
        db.flush()
        db.add_all(
            [
                Chunk(document_id=doc.id, collection_id=None, text=f"chunk {i}", chunk_index=i)
                for i in range(5)
            ]
        )
        db.commit()
        return doc.id


def test_reindex_writes_one_vector_per_chunk_under_the_new_model(session_factory, seeded):
    provider = HashingEmbeddingProvider(dim=8)
    store = PgVectorStore(session_factory, model=provider.model)
    with session_factory() as db:
        count = reindex_all(
            db, embedding_provider=provider, vector_store=store, lexical_store=None, batch_size=2
        )
    assert count == 5
    assert store.count() == 5
    hits = store.query([1.0] * 8, top_k=10, access=AccessFilter.unrestricted())
    assert len(hits) == 5


def test_reindex_replaces_vectors_from_the_previous_model(session_factory, seeded):
    old = HashingEmbeddingProvider(dim=4)
    old_store = PgVectorStore(session_factory, model=old.model)
    with session_factory() as db:
        reindex_all(
            db, embedding_provider=old, vector_store=old_store, lexical_store=None, batch_size=5
        )
    new = HashingEmbeddingProvider(dim=8)
    new_store = PgVectorStore(session_factory, model=new.model)
    with session_factory() as db:
        reindex_all(
            db, embedding_provider=new, vector_store=new_store, lexical_store=None, batch_size=5
        )

    assert new_store.count() == 5, "reindex must replace vectors, not accumulate them"
    assert old_store.query([1.0] * 4, top_k=10, access=AccessFilter.unrestricted()) == []


def test_reindex_reports_progress_in_batches(session_factory, seeded):
    provider = HashingEmbeddingProvider(dim=8)
    store = PgVectorStore(session_factory, model=provider.model)
    seen: list[tuple[int, int]] = []
    with session_factory() as db:
        reindex_all(
            db,
            embedding_provider=provider,
            vector_store=store,
            lexical_store=None,
            batch_size=2,
            on_progress=lambda done, total: seen.append((done, total)),
        )
    assert seen[-1] == (5, 5)
    assert all(total == 5 for _, total in seen)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest packages/core/tests/test_reindex.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragfabric_core.ingest.reindex'`.

- [ ] **Step 3: Write the module**

Create `packages/core/src/ragfabric_core/ingest/reindex.py`:

```python
"""Re-embed an existing corpus under the active embedding model.

Why this is not "ingest again": re-ingesting re-parses and re-chunks, which
changes chunk ids and orphans every Source row recorded against an earlier
answer. Reindexing reads the chunks table, which is the source of truth for
text, and replaces only the derived vectors. Retrieval runs recorded before a
model change still resolve to real chunks afterwards.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ragfabric_core.models.document import Chunk
from ragfabric_core.providers.base import EmbeddingProvider
from ragfabric_core.stores.base import LexicalStore, VectorStore


def reindex_all(
    db: Session,
    *,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    lexical_store: LexicalStore | None = None,
    batch_size: int = 64,
    document_id: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Re-embed every chunk (or one document's chunks) and return how many."""
    base = select(Chunk).order_by(Chunk.id)
    counter = select(func.count()).select_from(Chunk)
    if document_id is not None:
        base = base.where(Chunk.document_id == document_id)
        counter = counter.where(Chunk.document_id == document_id)
        vector_store.delete_document(document_id)
    total = int(db.execute(counter).scalar() or 0)

    done = 0
    batch: list[Chunk] = []
    for chunk in db.execute(base).scalars():
        batch.append(chunk)
        if len(batch) >= batch_size:
            done += _flush(batch, embedding_provider, vector_store, lexical_store)
            batch = []
            if on_progress is not None:
                on_progress(done, total)
    if batch:
        done += _flush(batch, embedding_provider, vector_store, lexical_store)
    if on_progress is not None:
        on_progress(done, total)
    return done


def _flush(
    batch: list[Chunk],
    provider: EmbeddingProvider,
    vector_store: VectorStore,
    lexical_store: LexicalStore | None,
) -> int:
    texts = [c.text for c in batch]
    result = provider.embed(texts)
    ids = [c.id for c in batch]
    payloads = [
        {
            "document_id": c.document_id,
            "collection_id": c.collection_id,
            "model": provider.model,
            "dim": provider.dim,
        }
        for c in batch
    ]
    vector_store.upsert(ids, result.vectors, payloads)
    if lexical_store is not None:
        lexical_store.index(ids, texts, payloads)
    return len(batch)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest packages/core/tests/test_reindex.py -q`
Expected: 3 passed. The second test relies on `PgVectorStore.upsert` overwriting the row for a chunk id, which it does, and on the model filter from Task 2 hiding the old model's rows.

- [ ] **Step 5: Write the failing CLI test**

Create `packages/cli/tests/test_cli_phase3.py`:

```python
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()


def test_reindex_reports_the_number_of_chunks(monkeypatch, tmp_path, cli_env):
    result = runner.invoke(app, ["reindex", "--yes", "--batch-size", "2"])
    assert result.exit_code == 0, result.output
    assert "re-embedded" in result.output


def test_reindex_without_yes_asks_for_confirmation(cli_env):
    result = runner.invoke(app, ["reindex"], input="n\n")
    assert result.exit_code == 1
    assert "aborted" in result.output.lower()
```

Reuse the `cli_env` fixture that `packages/cli/tests/test_cli_phase2.py` already defines for a temporary database and config; if it is local to that file, move it into `packages/cli/tests/conftest.py` unchanged in this step so both files share it.

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest packages/cli/tests/test_cli_phase3.py -q`
Expected: FAIL, `No such command 'reindex'`.

- [ ] **Step 7: Write the command**

Create `packages/cli/src/ragfabric_cli/commands/reindex.py`:

```python
"""ragfabric reindex: re-embed the corpus under the active embedding model."""

from __future__ import annotations

import typer

from ragfabric_cli.commands.common import session
from ragfabric_core.ingest.reindex import reindex_all
from ragfabric_core.providers.registry import build_embedding_provider
from ragfabric_core.runtime import get_config, get_session_factory
from ragfabric_core.stores.registry import build_lexical_store, build_vector_store


def reindex(
    batch_size: int = typer.Option(64, "--batch-size", min=1, help="Chunks per embedding call."),
    document_id: int | None = typer.Option(None, "--document-id", help="Limit to one document."),
    yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation."),
) -> None:
    """Re-embed every chunk with the configured embedding model."""
    cfg = get_config()
    provider = build_embedding_provider(cfg.embeddings)
    if not yes:
        typer.echo(
            f"This replaces every stored vector using {provider.model} ({provider.dim} dims)."
        )
        if not typer.confirm("Continue?"):
            typer.echo("aborted")
            raise typer.Exit(1)

    sf = get_session_factory()
    vector_store = build_vector_store(cfg.vector_store, sf, embedding_model=provider.model)
    lexical_store = build_lexical_store(cfg.lexical_store, sf)

    def progress(done: int, total: int) -> None:
        typer.echo(f"  {done}/{total}")

    with session() as db:
        count = reindex_all(
            db,
            embedding_provider=provider,
            vector_store=vector_store,
            lexical_store=lexical_store,
            batch_size=batch_size,
            document_id=document_id,
            on_progress=progress,
        )
    typer.echo(f"re-embedded {count} chunks with {provider.model}")
```

In `packages/cli/src/ragfabric_cli/main.py`, add the import beside the other command imports:

```python
from ragfabric_cli.commands.reindex import reindex as reindex_command
```

and register it beside `app.command("worker")(worker_command)`:

```python
app.command("reindex")(reindex_command)
```

- [ ] **Step 8: Run the CLI tests**

Run: `uv run pytest packages/cli/tests -q`
Expected: all pass.

- [ ] **Step 9: Run the whole suite**

Run: `uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: `234 passed, 3 skipped`.

- [ ] **Step 10: Commit**

```bash
git add packages/core/src/ragfabric_core/ingest/reindex.py packages/cli/src/ragfabric_cli/commands/reindex.py packages/cli/src/ragfabric_cli/main.py packages/core/tests/test_reindex.py packages/cli/tests/
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: add ragfabric reindex for batched re-embedding under the active model"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 5: `ChromaVectorStore`, the second real vector store

**Files:**
- Create: `packages/core/src/ragfabric_core/stores/chroma_store.py`, `packages/core/tests/test_stores_chroma.py`
- Modify: `packages/core/src/ragfabric_core/stores/registry.py`, `docker-compose.yml`, `.env.example`, `.github/workflows/ci.yml`
- Test: `packages/core/tests/test_stores_chroma.py` (integration), additions to `packages/core/tests/test_access_sql.py` for the where clause translator

**Interfaces:**
- Consumes: Task 2's `normalise`, the `VectorStore` protocol in `stores/base.py`.
- Produces: `chroma_where(access: AccessFilter, model: str | None) -> dict | None` translating an `AccessFilter` into a Chroma `where` document; `ChromaVectorStore(client, collection_name="ragfabric_chunks", model=None, session_factory=None)` implementing `VectorStore`; `build_vector_store` accepting `kind: chroma`.
- Metadata written per vector: `{"document_id": int, "collection_id": int, "model": str, "dim": int, "chunk_id": int}`. `collection_id` is `-1` when the chunk has no collection, because Chroma cannot match a null in a `where` clause; `NO_COLLECTION = -1` is a module constant and the translation maps `None` to it on both the write and the read side.

Why chunk text is read back from the database rather than from Chroma: `RetrievedChunk` carries `text`, `page`, `section`, `char_start`, `char_end` and the document filename. Storing all of that in Chroma metadata would duplicate the relational row and create a second source of truth that can drift. Chroma holds vectors and the minimum metadata the access predicate needs; the store resolves the hits back to `Chunk` rows through the session factory. That keeps one source of truth for text and makes the two vector stores return identical `RetrievedChunk` objects.

- [ ] **Step 1: Write the failing where clause tests**

Add to `packages/core/tests/test_access_sql.py`:

```python
def test_chroma_where_is_none_when_unrestricted_and_no_model_pinned():
    from ragfabric_core.stores.chroma_store import chroma_where

    assert chroma_where(AccessFilter.unrestricted(), None) is None


def test_chroma_where_pins_the_model_even_when_access_is_unrestricted():
    from ragfabric_core.stores.chroma_store import chroma_where

    assert chroma_where(AccessFilter.unrestricted(), "nomic-embed-text") == {
        "model": {"$eq": "nomic-embed-text"}
    }


def test_chroma_where_ors_the_two_allow_axes_and_ands_the_deny_list():
    from ragfabric_core.stores.chroma_store import chroma_where

    access = AccessFilter(
        document_ids=frozenset({1, 2}),
        collection_ids=frozenset({7}),
        denied_document_ids=frozenset({2}),
    )
    where = chroma_where(access, None)
    assert where == {
        "$and": [
            {
                "$or": [
                    {"document_id": {"$in": [1, 2]}},
                    {"collection_id": {"$in": [7]}},
                ]
            },
            {"document_id": {"$nin": [2]}},
        ]
    }


def test_chroma_where_maps_an_empty_allow_set_to_a_predicate_that_matches_nothing():
    from ragfabric_core.stores.chroma_store import chroma_where

    where = chroma_where(AccessFilter(document_ids=frozenset(), collection_ids=None), None)
    assert where == {"document_id": {"$in": []}}, (
        "an empty allow set must match nothing, never everything"
    )
```

That last test is the one that matters most. An access filter with an empty allow set means "this principal may read nothing", and a translator that drops an empty list would turn that into "no restriction". `access_clause` handles it with `false()`; Chroma has no `false()`, so an empty `$in` is the equivalent, and it is asserted rather than assumed.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest packages/core/tests/test_access_sql.py -q -k chroma`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragfabric_core.stores.chroma_store'`.

- [ ] **Step 3: Write the store**

Create `packages/core/src/ragfabric_core/stores/chroma_store.py`:

```python
"""VectorStore over a Chroma collection.

Why a second vector store exists at all: the product claims every interface has
at least two real implementations, so an adopter can believe a third one is
possible. pgvector and Chroma are the two for VectorStore, and they are held to
the same contract, including ADR 0003: the access predicate goes into Chroma's
own `where` document so a forbidden vector is never ranked, not filtered out of
the results afterwards.

Why text is not stored here: RetrievedChunk needs text, page, section and
character spans, all of which live on the chunks row. Duplicating them into
Chroma metadata would create a second source of truth that drifts on the first
edit. Chroma holds vectors plus exactly the metadata the access predicate
needs, and hits are resolved back to chunks rows through the session factory.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.embeddings.normalise import normalise
from ragfabric_core.models.document import Chunk, Document
from ragfabric_core.strategies.base import RetrievedChunk

NO_COLLECTION = -1


def _cid(collection_id: int | None) -> int:
    return NO_COLLECTION if collection_id is None else int(collection_id)


def chroma_where(access: AccessFilter, model: str | None) -> dict | None:
    """Translate an AccessFilter (and the pinned model) into a Chroma where document."""
    parts: list[dict] = []
    if not access.is_unrestricted:
        if access.document_ids is not None or access.collection_ids is not None:
            allows: list[dict] = []
            if access.document_ids is not None:
                allows.append({"document_id": {"$in": sorted(access.document_ids)}})
            if access.collection_ids is not None:
                allows.append({"collection_id": {"$in": sorted(access.collection_ids)}})
            parts.append({"$or": allows} if len(allows) > 1 else allows[0])
        if access.denied_document_ids:
            parts.append({"document_id": {"$nin": sorted(access.denied_document_ids)}})
    if model is not None:
        parts.append({"model": {"$eq": model}})
    if not parts:
        return None
    return {"$and": parts} if len(parts) > 1 else parts[0]


class ChromaVectorStore:
    name = "chroma"

    def __init__(
        self,
        client,
        collection_name: str = "ragfabric_chunks",
        model: str | None = None,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        self._client = client
        self._name = collection_name
        self._model = model
        self._sf = session_factory

    @property
    def _collection(self):
        # cosine, to match pgvector's operator class and the normalised vectors.
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self, chunk_ids: list[int], vectors: list[list[float]], payloads: list[dict]
    ) -> None:
        if not chunk_ids:
            return
        self._collection.upsert(
            ids=[str(cid) for cid in chunk_ids],
            embeddings=[normalise(v) for v in vectors],
            metadatas=[
                {
                    "chunk_id": int(cid),
                    "document_id": int(p["document_id"]),
                    "collection_id": _cid(p.get("collection_id")),
                    "model": str(p["model"]),
                    "dim": int(p.get("dim", len(v))),
                }
                for cid, v, p in zip(chunk_ids, vectors, payloads, strict=True)
            ],
        )

    def query(
        self, vector: list[float], top_k: int, access: AccessFilter, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        where = chroma_where(access, self._model)
        for key in ("document_id", "collection_id"):
            if filters and filters.get(key) is not None:
                clause = {key: {"$eq": _cid(filters[key]) if key == "collection_id" else filters[key]}}
                where = {"$and": [where, clause]} if where else clause
        res = self._collection.query(
            query_embeddings=[normalise(vector)],
            n_results=top_k,
            where=where or None,
        )
        ids = [int(i) for i in (res.get("ids") or [[]])[0]]
        distances = (res.get("distances") or [[]])[0]
        if not ids:
            return []
        scores = {cid: 1.0 - float(d) for cid, d in zip(ids, distances, strict=True)}
        rows = self._resolve(ids)
        fmt_filter = (filters or {}).get("format")
        out = [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                collection_id=chunk.collection_id,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                score=scores.get(chunk.id),
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                metadata={"filename": filename, "format": fmt},
            )
            for chunk, filename, fmt in rows
            if fmt_filter is None or fmt == fmt_filter
        ]
        out.sort(key=lambda c: (-(c.score or 0.0), c.chunk_id))
        return out[:top_k]

    def _resolve(self, chunk_ids: list[int]):
        if self._sf is None:
            raise RuntimeError("ChromaVectorStore needs a session_factory to resolve chunk text")
        with self._sf() as db:
            return db.execute(
                select(Chunk, Document.filename, Document.format)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.id.in_(chunk_ids))
            ).all()

    def delete_document(self, document_id: int) -> None:
        self._collection.delete(where={"document_id": {"$eq": int(document_id)}})

    def count(self) -> int:
        return int(self._collection.count())
```

- [ ] **Step 4: Run the where clause tests**

Run: `uv run pytest packages/core/tests/test_access_sql.py -q`
Expected: all pass.

- [ ] **Step 5: Wire the registry**

In `packages/core/src/ragfabric_core/stores/registry.py`, replace `build_vector_store` with:

```python
def build_vector_store(
    cfg: VectorStoreConfig,
    session_factory: Callable[[], Session],
    embedding_model: str | None = None,
) -> VectorStore:
    if cfg.kind in ("pgvector", "memory"):
        return PgVectorStore(session_factory, model=embedding_model)
    if cfg.kind == "chroma":
        import chromadb

        url = os.environ.get("CHROMA_URL", "http://localhost:8000")
        host, _, port = url.removeprefix("http://").removeprefix("https://").partition(":")
        client = chromadb.HttpClient(host=host, port=int(port or 8000))
        return ChromaVectorStore(
            client, model=embedding_model, session_factory=session_factory
        )
    raise NotImplementedError(f"unknown vector store kind {cfg.kind!r}")
```

Add the import `from ragfabric_core.stores.chroma_store import ChromaVectorStore` at the top. `chromadb` is imported inside the function on purpose: it is a heavy import and a pgvector deployment should not pay for it at startup.

- [ ] **Step 6: Write the integration test**

Create `packages/core/tests/test_stores_chroma.py`:

```python
"""Chroma integration. Skipped unless RAGFABRIC_TEST_CHROMA_URL is set.

Run with:
    docker compose --profile full up -d chroma
    RAGFABRIC_TEST_CHROMA_URL=http://localhost:8000 uv run pytest -m integration \
        packages/core/tests/test_stores_chroma.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.stores.chroma_store import ChromaVectorStore

pytestmark = pytest.mark.integration

URL = os.environ.get("RAGFABRIC_TEST_CHROMA_URL")


@pytest.fixture
def store(session_factory):
    if not URL:
        pytest.skip("RAGFABRIC_TEST_CHROMA_URL is not set")
    import chromadb

    host, _, port = URL.removeprefix("http://").partition(":")
    client = chromadb.HttpClient(host=host, port=int(port or 8000))
    name = f"test_{uuid.uuid4().hex[:8]}"
    yield ChromaVectorStore(
        client, collection_name=name, model="hashing-8", session_factory=session_factory
    )
    client.delete_collection(name)


@pytest.fixture
def two_collections(session_factory):
    """Two documents in two different collections, three chunks each."""
    from ragfabric_core.models.document import Chunk, Collection, Document

    with session_factory() as db:
        col_a = Collection(name="a", owner_id=None)
        col_b = Collection(name="b", owner_id=None)
        db.add_all([col_a, col_b])
        db.flush()
        made = {}
        for label, col in (("a", col_a), ("b", col_b)):
            doc = Document(
                filename=f"{label}.txt",
                format="txt",
                status="ready",
                owner_id=None,
                collection_id=col.id,
            )
            db.add(doc)
            db.flush()
            chunks = [
                Chunk(
                    document_id=doc.id,
                    collection_id=col.id,
                    text=f"{label} chunk {i}",
                    chunk_index=i,
                )
                for i in range(3)
            ]
            db.add_all(chunks)
            db.commit()
            made[label] = (doc.id, col.id, [c.id for c in chunks])
        return made


def test_chroma_returns_chunks_resolved_from_the_database(store, two_collections):
    doc_id, col_id, chunk_ids = two_collections["a"]
    store.upsert(
        chunk_ids,
        [[1.0] + [0.0] * 7, [0.9, 0.1] + [0.0] * 6, [0.0] * 7 + [1.0]],
        [{"document_id": doc_id, "collection_id": col_id, "model": "hashing-8", "dim": 8}] * 3,
    )
    hits = store.query([1.0] + [0.0] * 7, top_k=2, access=AccessFilter.unrestricted())
    assert [h.chunk_id for h in hits] == chunk_ids[:2]
    assert hits[0].text == "a chunk 0", "text must come from the chunks table"
    assert hits[0].score is not None and hits[0].score > hits[1].score


def test_chroma_applies_a_restrictive_access_filter_inside_the_query(store, two_collections):
    a_doc, a_col, a_chunks = two_collections["a"]
    b_doc, b_col, b_chunks = two_collections["b"]
    for doc_id, col_id, ids in ((a_doc, a_col, a_chunks), (b_doc, b_col, b_chunks)):
        store.upsert(
            ids,
            [[1.0] + [0.0] * 7] * 3,
            [{"document_id": doc_id, "collection_id": col_id, "model": "hashing-8", "dim": 8}] * 3,
        )

    access = AccessFilter(document_ids=None, collection_ids=frozenset({a_col}))
    hits = store.query([1.0] + [0.0] * 7, top_k=10, access=access)

    assert hits, "the permitted collection should still return chunks"
    assert {h.collection_id for h in hits} == {a_col}
    assert all(h.document_id == a_doc for h in hits)


def test_chroma_denies_everything_for_an_empty_allow_set(store, two_collections):
    doc_id, col_id, chunk_ids = two_collections["a"]
    store.upsert(
        chunk_ids,
        [[1.0] + [0.0] * 7] * 3,
        [{"document_id": doc_id, "collection_id": col_id, "model": "hashing-8", "dim": 8}] * 3,
    )
    access = AccessFilter(document_ids=frozenset(), collection_ids=None)
    assert store.query([1.0] + [0.0] * 7, top_k=10, access=access) == []
```

- [ ] **Step 7: Run it against the container**

```bash
docker compose --profile full up -d chroma
RAGFABRIC_TEST_CHROMA_URL=http://localhost:8000 uv run pytest -m integration packages/core/tests/test_stores_chroma.py -q
```

Expected: 3 passed. If `Collection` is not the model name used for collections, check `models/document.py` and use the real one; do not invent a model.

- [ ] **Step 8: Wire compose and CI**

In `docker-compose.yml`, add `CHROMA_URL: http://chroma:8000` to the `api` and `worker` service environments. In `.env.example`, add:

```
# Only needed when vector_store.kind is chroma (compose profile: full)
CHROMA_URL=http://localhost:8000
```

In `.github/workflows/ci.yml`, add a `chroma` service to the job that already runs the PostgreSQL integration tests, using image `chromadb/chroma:1.5.9` on port 8000, and set `RAGFABRIC_TEST_CHROMA_URL: http://localhost:8000` in that job's env so the Chroma tests run in CI rather than only locally.

- [ ] **Step 9: Run the whole suite and the import contracts**

Run: `uv run pytest -q -p no:warnings 2>&1 | tail -1 && uv run lint-imports`
Expected: `238 passed, 3 skipped` locally without the integration env vars, and the import contracts still pass.

- [ ] **Step 10: Commit**

```bash
git add packages/core/src/ragfabric_core/stores/chroma_store.py packages/core/src/ragfabric_core/stores/registry.py packages/core/tests/test_stores_chroma.py packages/core/tests/test_access_sql.py docker-compose.yml .env.example .github/workflows/ci.yml
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: add the Chroma vector store with the access predicate inside the query"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 6: The `Reranker` interface and three implementations

**Files:**
- Create: `packages/core/src/ragfabric_core/rerank/__init__.py`, `rerank/base.py`, `rerank/noop.py`, `rerank/llm_reranker.py`, `rerank/cross_encoder.py`, `rerank/registry.py`
- Test: `packages/core/tests/test_rerank.py`

**Interfaces:**
- Consumes: `RetrievedChunk` from `strategies/base.py`, `LLMProvider` and `Message` from `providers/base.py`.
- Produces: `Reranker` protocol with `name: str` and `rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]`; `NoopReranker()`; `LlmReranker(llm, model=None, batch_size=8)`; `CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L6-v2")`; `build_reranker(cfg: RerankerConfig, llm: LLMProvider | None = None) -> Reranker`.
- Contract every implementation obeys: the returned list is a subset of the input, never longer than `top_k`, never contains a chunk the input did not, and preserves each chunk's identity fields. A reranker rewrites `score` and may reorder; it may not invent, merge or edit chunks.

Why reranking is a separate interface rather than a flag on the strategy: the same reranker is used by Traditional RAG now and by Vectorless, Agentic and Graph RAG later, and Phase 8 needs to measure the same reranker across all four. A flag would force each strategy to reimplement it.

Why `cross_encoder` is behind an extra: `sentence-transformers` pulls in torch, roughly 2 GB. Most deployments, and every API container that does not rerank, should not carry it. Selecting the kind without the extra installed raises a `ProviderError` naming the install command, which is a better failure than an ImportError traceback.

- [ ] **Step 1: Write the failing tests**

Create `packages/core/tests/test_rerank.py`:

```python
import pytest

from ragfabric_core.providers.base import Completion, ProviderError
from ragfabric_core.rerank.llm_reranker import LlmReranker
from ragfabric_core.rerank.noop import NoopReranker
from ragfabric_core.rerank.registry import build_reranker
from ragfabric_core.strategies.base import RetrievedChunk


def chunk(cid: int, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=1, collection_id=None, text=text, score=score
    )


class StubLLM:
    name = "stub"
    default_model = "stub"

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls = 0

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls += 1
        import json

        payload = json.dumps({"scores": self._scores})
        return Completion(
            text=payload,
            model="stub",
            provider="stub",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
        )


def test_noop_reranker_truncates_but_does_not_reorder():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = NoopReranker().rerank("q", chunks, top_k=2)
    assert [c.chunk_id for c in out] == [1, 2]


def test_llm_reranker_reorders_by_the_returned_scores():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = LlmReranker(StubLLM([0.1, 0.95, 0.5])).rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [2, 3, 1]
    assert out[0].score == pytest.approx(0.95)


def test_llm_reranker_returns_a_subset_and_never_invents_a_chunk():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8)]
    out = LlmReranker(StubLLM([0.2, 0.4])).rerank("q", chunks, top_k=5)
    assert {c.chunk_id for c in out} <= {1, 2}
    assert len(out) == 2


def test_llm_reranker_falls_back_to_the_original_order_when_the_model_misbehaves():
    class Broken(StubLLM):
        def complete(self, messages, **kwargs):
            return Completion(
                text="not json at all",
                model="stub",
                provider="stub",
                input_tokens=1,
                output_tokens=1,
                latency_ms=1,
            )

    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8)]
    out = LlmReranker(Broken([])).rerank("q", chunks, top_k=2)
    assert [c.chunk_id for c in out] == [1, 2], (
        "a malformed rerank response must degrade to retrieval order, not drop results"
    )


def test_llm_reranker_handles_a_wrong_length_score_list():
    chunks = [chunk(1, "a", 0.9), chunk(2, "b", 0.8), chunk(3, "c", 0.7)]
    out = LlmReranker(StubLLM([0.5])).rerank("q", chunks, top_k=3)
    assert [c.chunk_id for c in out] == [1, 2, 3]


def test_build_reranker_none_gives_the_noop():
    from ragfabric_core.config_file import RerankerConfig

    assert build_reranker(RerankerConfig(kind="none")).name == "none"


def test_build_reranker_llm_without_a_provider_is_a_clear_error():
    from ragfabric_core.config_file import RerankerConfig

    with pytest.raises(ProviderError) as exc:
        build_reranker(RerankerConfig(kind="llm"), llm=None)
    assert "llm provider" in str(exc.value)


def test_build_reranker_cross_encoder_without_the_extra_names_the_install_command():
    from ragfabric_core.config_file import RerankerConfig

    import ragfabric_core.rerank.cross_encoder as ce

    original = ce._import_cross_encoder

    def boom():
        raise ImportError("no module named sentence_transformers")

    ce._import_cross_encoder = boom
    try:
        with pytest.raises(ProviderError) as exc:
            build_reranker(RerankerConfig(kind="cross_encoder"))
        assert "ragfabric[rerank]" in str(exc.value)
    finally:
        ce._import_cross_encoder = original
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest packages/core/tests/test_rerank.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragfabric_core.rerank'`.

- [ ] **Step 3: Write the protocol**

Create `packages/core/src/ragfabric_core/rerank/__init__.py` as an empty file and `packages/core/src/ragfabric_core/rerank/base.py`:

```python
"""The Reranker interface.

A reranker re-scores a shortlist that a store already ranked cheaply. It exists
as its own interface because all four retrieval strategies share it and Phase 8
measures the same reranker across all of them; a per strategy flag would mean
four implementations of one idea.

Contract, asserted in the tests of every implementation: the output is a subset
of the input, no longer than top_k, with identity fields untouched. A reranker
reorders and rewrites score. It never invents, merges or edits a chunk.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragfabric_core.strategies.base import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...
```

- [ ] **Step 4: Write the noop**

Create `packages/core/src/ragfabric_core/rerank/noop.py`:

```python
"""The reranker used when reranking is off. Truncation only, order preserved."""

from __future__ import annotations

from ragfabric_core.strategies.base import RetrievedChunk


class NoopReranker:
    name = "none"

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return list(chunks[:top_k])
```

- [ ] **Step 5: Write the LLM reranker**

Create `packages/core/src/ragfabric_core/rerank/llm_reranker.py`:

```python
"""Rerank by asking the configured LLM to score each candidate.

Why scores rather than a reordered list: a model asked to reorder can drop or
duplicate items, and detecting that costs more than it saves. A fixed length
list of scores is checkable: wrong length, wrong type or unparseable output all
degrade to retrieval order, which is a worse ranking but never a wrong result
set. Retrieval already guaranteed the access filter, so a degraded rerank can
never leak a forbidden chunk.
"""

from __future__ import annotations

import json

from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.strategies.base import RetrievedChunk

_SYSTEM = (
    "You score how well each numbered passage answers the question. "
    "Reply with JSON only, in the form {\"scores\": [0.0, 1.0, ...]}, "
    "one score between 0 and 1 per passage, in the same order as the passages. "
    "Do not add commentary."
)
_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}}
    },
    "required": ["scores"],
}
_MAX_PASSAGE_CHARS = 1200


class LlmReranker:
    name = "llm"

    def __init__(self, llm: LLMProvider, model: str | None = None) -> None:
        self._llm = llm
        self._model = model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._score(query, chunks)
        if scores is None:
            return list(chunks[:top_k])
        rescored = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        rescored.sort(key=lambda c: (-(c.score or 0.0), c.chunk_id))
        return rescored[:top_k]

    def _score(self, query: str, chunks: list[RetrievedChunk]) -> list[float] | None:
        passages = "\n\n".join(
            f"[{i}] {c.text[:_MAX_PASSAGE_CHARS]}" for i, c in enumerate(chunks, start=1)
        )
        messages = [
            Message(role="system", content=_SYSTEM),
            Message(role="user", content=f"Question: {query}\n\nPassages:\n{passages}"),
        ]
        try:
            completion = self._llm.complete(
                messages, model=self._model, max_tokens=512, json_schema=_SCHEMA
            )
            scores = json.loads(completion.text)["scores"]
        except Exception:
            return None
        if not isinstance(scores, list) or len(scores) != len(chunks):
            return None
        try:
            return [float(s) for s in scores]
        except (TypeError, ValueError):
            return None
```

- [ ] **Step 6: Write the cross encoder reranker**

Create `packages/core/src/ragfabric_core/rerank/cross_encoder.py`:

```python
"""Rerank with a cross encoder model (optional extra: ragfabric[rerank]).

A cross encoder reads the question and one passage together and scores the pair
directly, which is more accurate than comparing two independently produced
embeddings. The cost is a forward pass per candidate and a torch install, which
is why it is an extra rather than a dependency.
"""

from __future__ import annotations

from ragfabric_core.providers.base import ProviderError
from ragfabric_core.strategies.base import RetrievedChunk

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _import_cross_encoder():
    """Indirection so the tests can simulate a missing extra."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        try:
            cross_encoder = _import_cross_encoder()
        except ImportError as exc:
            raise ProviderError(
                "cross_encoder",
                "sentence-transformers is not installed. Install the extra with "
                "`uv pip install 'ragfabric[rerank]'` or set reranker.kind to none or llm.",
            ) from exc
        self._model = cross_encoder(model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks])
        rescored = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        rescored.sort(key=lambda c: (-(c.score or 0.0), c.chunk_id))
        return rescored[:top_k]
```

- [ ] **Step 7: Write the registry**

Create `packages/core/src/ragfabric_core/rerank/registry.py`:

```python
"""Build a reranker from configuration. The only place reranker kinds are switched on."""

from __future__ import annotations

from ragfabric_core.config_file import RerankerConfig
from ragfabric_core.providers.base import LLMProvider, ProviderError
from ragfabric_core.rerank.base import Reranker
from ragfabric_core.rerank.cross_encoder import CrossEncoderReranker
from ragfabric_core.rerank.llm_reranker import LlmReranker
from ragfabric_core.rerank.noop import NoopReranker


def build_reranker(cfg: RerankerConfig, llm: LLMProvider | None = None) -> Reranker:
    if cfg.kind == "llm":
        if llm is None:
            raise ProviderError("reranker", "reranker.kind is llm but no llm provider was given")
        return LlmReranker(llm)
    if cfg.kind == "cross_encoder":
        return CrossEncoderReranker()
    return NoopReranker()
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest packages/core/tests/test_rerank.py -q`
Expected: 8 passed.

- [ ] **Step 9: Run the whole suite**

Run: `uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: `246 passed, 3 skipped`.

- [ ] **Step 10: Commit**

```bash
git add packages/core/src/ragfabric_core/rerank packages/core/tests/test_rerank.py
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: add the Reranker interface with noop, llm and cross encoder implementations"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 7: Token counting and the context budget

**Files:**
- Create: `packages/core/src/ragfabric_core/tokens.py`
- Test: `packages/core/tests/test_tokens.py`

**Interfaces:**
- Consumes: nothing beyond `RetrievedChunk`.
- Produces: `count_tokens(text: str, model: str | None = None) -> int`; `fit_to_budget(chunks: list[RetrievedChunk], max_tokens: int, model: str | None = None) -> tuple[list[RetrievedChunk], int]` returning the chunks that fit in rank order and the token total; `ESTIMATED_CHARS_PER_TOKEN = 4`.

Why the budget truncates whole chunks and never splits one: a half chunk has a character span that no longer matches its stored `char_start` and `char_end`, which breaks the citation offsets the UI renders and the citation contract checks. Dropping the lowest ranked chunk loses the least relevant evidence; splitting the highest ranked one corrupts the most relevant citation.

Why the count is exact for tiktoken models and estimated otherwise, with no silent mixing: ADR 0004 forbids fabricated numbers, so an estimate is labelled. `count_tokens` returns an exact count when a tokeniser exists for the model and a documented four characters per token estimate otherwise, and `fit_to_budget` records which method it used so a run's trace does not present an estimate as a measurement.

- [ ] **Step 1: Write the failing tests**

Create `packages/core/tests/test_tokens.py`:

```python
from ragfabric_core.strategies.base import RetrievedChunk
from ragfabric_core.tokens import count_tokens, fit_to_budget


def chunk(cid: int, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=1, collection_id=None, text=text, score=score
    )


def test_count_tokens_is_exact_for_a_known_openai_model():
    assert count_tokens("hello world", "text-embedding-3-small") == 2


def test_count_tokens_estimates_for_an_unknown_model_without_raising():
    assert count_tokens("a" * 40, "llama3.2:3b") == 10


def test_count_tokens_of_empty_text_is_zero():
    assert count_tokens("", None) == 0


def test_fit_to_budget_keeps_rank_order_and_drops_the_tail():
    chunks = [chunk(1, "a" * 40, 0.9), chunk(2, "b" * 40, 0.8), chunk(3, "c" * 40, 0.7)]
    kept, total = fit_to_budget(chunks, max_tokens=20, model="llama3.2:3b")
    assert [c.chunk_id for c in kept] == [1, 2]
    assert total == 20


def test_fit_to_budget_never_splits_a_chunk():
    chunks = [chunk(1, "a" * 400, 0.9)]
    kept, total = fit_to_budget(chunks, max_tokens=10, model="llama3.2:3b")
    assert kept == [], "a chunk that does not fit is dropped whole, never truncated"
    assert total == 0


def test_fit_to_budget_with_a_generous_budget_keeps_everything():
    chunks = [chunk(1, "a" * 40, 0.9), chunk(2, "b" * 40, 0.8)]
    kept, total = fit_to_budget(chunks, max_tokens=10_000, model=None)
    assert len(kept) == 2
    assert total == 20
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest packages/core/tests/test_tokens.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragfabric_core.tokens'`.

- [ ] **Step 3: Write the module**

Create `packages/core/src/ragfabric_core/tokens.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest packages/core/tests/test_tokens.py -q`
Expected: 6 passed. If `count_tokens("hello world", "text-embedding-3-small")` is not 2 on the installed tiktoken, assert the real value rather than changing the implementation, and note it in the ledger.

- [ ] **Step 5: Commit**

```bash
git add packages/core/src/ragfabric_core/tokens.py packages/core/tests/test_tokens.py
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: add exact and estimated token counting with a whole chunk context budget"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 8: `TraditionalRAGStrategy`

**Files:**
- Create: `packages/core/src/ragfabric_core/strategies/traditional.py`, `packages/core/src/ragfabric_core/strategies/registry_defaults.py`
- Test: `packages/core/tests/test_traditional_strategy.py`

**Interfaces:**
- Consumes: Task 2's `build_vector_store(..., embedding_model=...)`, Task 6's `Reranker`, Task 7's `fit_to_budget`, the `RetrieverStrategy` protocol and `assert_strategy_contract`.
- Produces: `TraditionalRAGStrategy(embedding_provider, vector_store, reranker=None, max_context_tokens=6000, generation_model=None, candidate_multiplier=3)` with `name = StrategyName.TRADITIONAL`; `default_registry(cfg, session_factory) -> StrategyRegistry` registering the traditional strategy from configuration.
- Trace spans emitted, in order: `embed_query`, `vector_search`, `rerank` (only when a reranker other than the noop is configured), `context_budget`.

Why the store is asked for more candidates than `top_k`: a reranker can only improve an ordering it is given, so handing it exactly `top_k` candidates makes reranking almost pointless. The strategy fetches `top_k * candidate_multiplier` (default 3, so 24 for the shipped `top_k` of 8), applies the similarity threshold, reranks, then cuts to `top_k`. With the noop reranker the extra candidates are discarded by the same cut, so behaviour is unchanged and the only cost is a slightly wider store query.

Why the similarity threshold is applied before reranking: the threshold expresses "this passage is not about the question at all", which is a property of the retrieval score. Applying it after a rerank would compare an LLM's or cross encoder's score against a cosine threshold, which are different scales and would make the configured number meaningless.

- [ ] **Step 1: Write the failing tests**

Create `packages/core/tests/test_traditional_strategy.py`:

```python
import pytest

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.providers.offline import HashingEmbeddingProvider
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievedChunk,
    StrategyName,
    StrategyParams,
)
from ragfabric_core.strategies.contract import assert_strategy_contract
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy


class StubStore:
    name = "stub"

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_top_k: int | None = None
        self.last_access: AccessFilter | None = None
        self.last_filters: dict | None = None

    def upsert(self, chunk_ids, vectors, payloads):  # pragma: no cover
        raise NotImplementedError

    def query(self, vector, top_k, access, filters=None):
        self.last_top_k = top_k
        self.last_access = access
        self.last_filters = filters
        return [c for c in self._chunks if access.allows(c.document_id, c.collection_id)][:top_k]

    def delete_document(self, document_id):  # pragma: no cover
        raise NotImplementedError

    def count(self):
        return len(self._chunks)


def chunk(cid: int, score: float, doc: int = 1, col: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=doc,
        collection_id=col,
        text=f"passage {cid} about leave policy",
        score=score,
    )


def ctx(top_k: int = 3, threshold: float = 0.0, access: AccessFilter | None = None):
    return RetrievalContext(
        principal=Principal(user_id=1, email="a@b.c", role="user"),
        access_filter=access or AccessFilter.unrestricted(),
        params=StrategyParams(top_k=top_k, similarity_threshold=threshold),
    )


def strategy(store, reranker=None, max_context_tokens: int = 6000):
    return TraditionalRAGStrategy(
        embedding_provider=HashingEmbeddingProvider(dim=16),
        vector_store=store,
        reranker=reranker,
        max_context_tokens=max_context_tokens,
    )


def test_it_satisfies_the_strategy_contract():
    store = StubStore([chunk(1, 0.9), chunk(2, 0.8), chunk(3, 0.7)])
    result = assert_strategy_contract(strategy(store), "how much leave do I get", ctx(top_k=2))
    assert result.strategy == StrategyName.TRADITIONAL
    assert len(result.chunks) == 2


def test_it_asks_the_store_for_more_candidates_than_top_k():
    store = StubStore([chunk(i, 1.0 - i / 100) for i in range(1, 30)])
    strategy(store).retrieve("q", ctx(top_k=5))
    assert store.last_top_k == 15, "candidates must be top_k * candidate_multiplier"


def test_it_passes_the_access_filter_into_the_store_rather_than_filtering_after():
    store = StubStore([chunk(1, 0.9, doc=1), chunk(2, 0.8, doc=2)])
    access = AccessFilter(document_ids=frozenset({2}), collection_ids=None)
    result = strategy(store).retrieve("q", ctx(access=access))
    assert store.last_access is access
    assert [c.chunk_id for c in result.chunks] == [2]


def test_it_drops_candidates_below_the_similarity_threshold():
    store = StubStore([chunk(1, 0.9), chunk(2, 0.2), chunk(3, 0.05)])
    result = strategy(store).retrieve("q", ctx(top_k=3, threshold=0.25))
    assert [c.chunk_id for c in result.chunks] == [1]


def test_it_returns_no_chunks_when_everything_is_below_the_threshold():
    store = StubStore([chunk(1, 0.1)])
    result = strategy(store).retrieve("q", ctx(threshold=0.5))
    assert result.chunks == []
    assert result.retrieval_calls == 1


def test_it_applies_the_reranker_after_the_threshold_and_before_the_cut():
    from ragfabric_core.rerank.noop import NoopReranker

    class Reverse(NoopReranker):
        name = "reverse"

        def rerank(self, query, chunks, top_k):
            return list(reversed(chunks))[:top_k]

    store = StubStore([chunk(1, 0.9), chunk(2, 0.8), chunk(3, 0.7)])
    result = strategy(store, reranker=Reverse()).retrieve("q", ctx(top_k=2))
    assert [c.chunk_id for c in result.chunks] == [3, 2]


def test_it_records_the_expected_trace_spans():
    store = StubStore([chunk(1, 0.9)])
    result = strategy(store).retrieve("q", ctx())
    assert [s.name for s in result.trace] == [
        "embed_query",
        "vector_search",
        "context_budget",
    ]


def test_it_records_a_rerank_span_when_a_real_reranker_is_configured():
    from ragfabric_core.rerank.noop import NoopReranker

    class Named(NoopReranker):
        name = "llm"

    store = StubStore([chunk(1, 0.9)])
    result = strategy(store, reranker=Named()).retrieve("q", ctx())
    assert "rerank" in [s.name for s in result.trace]


def test_it_enforces_the_context_budget_by_dropping_the_tail():
    store = StubStore([chunk(1, 0.9), chunk(2, 0.8), chunk(3, 0.7)])
    result = strategy(store, max_context_tokens=9).retrieve("q", ctx(top_k=3))
    assert len(result.chunks) < 3
    budget_span = [s for s in result.trace if s.name == "context_budget"][0]
    assert budget_span.attributes["tokens_used"] <= 9


def test_metadata_filters_reach_the_store():
    store = StubStore([chunk(1, 0.9)])
    context = RetrievalContext(
        principal=Principal(user_id=1, email="a@b.c", role="user"),
        access_filter=AccessFilter.unrestricted(),
        collection_ids=[7],
        params=StrategyParams(top_k=3, metadata_filters={"format": "pdf"}),
    )
    strategy(store).retrieve("q", context)
    assert store.last_filters == {"collection_id": 7, "format": "pdf"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest packages/core/tests/test_traditional_strategy.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragfabric_core.strategies.traditional'`.

- [ ] **Step 3: Write the strategy**

Create `packages/core/src/ragfabric_core/strategies/traditional.py`:

```python
"""Traditional RAG: embed the question, search one vector index, rerank, fit a budget.

The order of operations is deliberate and is asserted by the tests:

  embed -> store query with the access filter inside it -> similarity threshold
  -> rerank -> cut to top_k -> context budget

The threshold runs before the rerank because it is expressed against the
retrieval score; comparing a reranker's score to a cosine threshold would make
the configured number meaningless. The store is asked for more candidates than
top_k so the reranker has something to improve, and the cut to top_k happens
after reranking, so with the noop reranker the behaviour is identical to asking
for top_k directly.

This strategy does not generate the answer. It returns a RetrievalResult, the
one shape every strategy returns (ADR 0002), and generation happens above it so
that all four strategies are generated and cited by identical code.
"""

from __future__ import annotations

import time

from ragfabric_core.providers.base import EmbeddingProvider
from ragfabric_core.rerank.base import Reranker
from ragfabric_core.stores.base import VectorStore
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    StrategyName,
    TraceSpan,
)
from ragfabric_core.tokens import fit_to_budget

DEFAULT_CANDIDATE_MULTIPLIER = 3


class TraditionalRAGStrategy:
    name = StrategyName.TRADITIONAL

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        max_context_tokens: int = 6000,
        generation_model: str | None = None,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        self._embedder = embedding_provider
        self._store = vector_store
        self._reranker = reranker
        self._max_context_tokens = max_context_tokens
        self._generation_model = generation_model
        self._multiplier = max(1, candidate_multiplier)

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        started = time.perf_counter()
        spans: list[TraceSpan] = []

        def span(name: str, begin: float, **attributes) -> None:
            spans.append(
                TraceSpan(
                    name=name,
                    started_ms=int((begin - started) * 1000),
                    duration_ms=int((time.perf_counter() - begin) * 1000),
                    attributes=attributes,
                )
            )

        mark = time.perf_counter()
        embedded = self._embedder.embed([query])
        span("embed_query", mark, model=self._embedder.model, dim=self._embedder.dim)

        filters: dict = {}
        if ctx.collection_ids:
            filters["collection_id"] = ctx.collection_ids[0]
        filters.update(dict(ctx.params.metadata_filters))

        mark = time.perf_counter()
        candidates = self._store.query(
            embedded.vectors[0],
            top_k=ctx.params.top_k * self._multiplier,
            access=ctx.access_filter,
            filters=filters or None,
        )
        span(
            "vector_search",
            mark,
            store=self._store.name,
            requested=ctx.params.top_k * self._multiplier,
            returned=len(candidates),
        )

        threshold = ctx.params.similarity_threshold
        if threshold > 0.0:
            candidates = [c for c in candidates if (c.score or 0.0) >= threshold]

        if self._reranker is not None and self._reranker.name != "none":
            mark = time.perf_counter()
            candidates = self._reranker.rerank(query, candidates, ctx.params.top_k)
            span("rerank", mark, reranker=self._reranker.name, kept=len(candidates))
        else:
            candidates = candidates[: ctx.params.top_k]

        mark = time.perf_counter()
        chunks, tokens_used = fit_to_budget(
            candidates, self._max_context_tokens, self._generation_model
        )
        span(
            "context_budget",
            mark,
            max_context_tokens=self._max_context_tokens,
            tokens_used=tokens_used,
            kept=len(chunks),
        )

        return RetrievalResult(
            strategy=self.name,
            chunks=chunks,
            retrieval_calls=1,
            llm_calls=1 if (self._reranker is not None and self._reranker.name == "llm") else 0,
            input_tokens=embedded.input_tokens,
            output_tokens=0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            trace=spans,
        )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest packages/core/tests/test_traditional_strategy.py -q`
Expected: 10 passed.

- [ ] **Step 5: Write the registry helper**

Create `packages/core/src/ragfabric_core/strategies/registry_defaults.py`:

```python
"""Build the strategy registry from configuration.

One place assembles a strategy from config so the API, the CLI and the tests all
get the same object graph. Phases 4 to 6 register their strategies here too.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session

from ragfabric_core.config_file import RagFabricConfig
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider
from ragfabric_core.rerank.registry import build_reranker
from ragfabric_core.stores.registry import build_vector_store
from ragfabric_core.strategies.base import StrategyRegistry
from ragfabric_core.strategies.traditional import TraditionalRAGStrategy


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def default_registry(
    cfg: RagFabricConfig, session_factory: Callable[[], Session]
) -> StrategyRegistry:
    embedder = build_embedding_provider(cfg.embeddings)
    store = build_vector_store(cfg.vector_store, session_factory, embedding_model=embedder.model)
    llm = build_llm_provider(cfg.llm) if cfg.reranker.kind == "llm" else None
    reranker = build_reranker(cfg.reranker, llm=llm)
    registry = StrategyRegistry()
    registry.register(
        TraditionalRAGStrategy(
            embedding_provider=embedder,
            vector_store=store,
            reranker=reranker,
            max_context_tokens=_int(cfg.strategies.traditional.get("max_context_tokens"), 6000),
            generation_model=cfg.llm.model,
        )
    )
    return registry
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: `256 passed, 3 skipped`.

- [ ] **Step 7: Commit**

```bash
git add packages/core/src/ragfabric_core/strategies/traditional.py packages/core/src/ragfabric_core/strategies/registry_defaults.py packages/core/tests/test_traditional_strategy.py
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: add TraditionalRAGStrategy with threshold, reranking and a context budget"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 9: Cited generation and the citation contract

**Files:**
- Create: `packages/core/src/ragfabric_core/generate/cited.py`, `packages/core/src/ragfabric_core/generate/contract.py`
- Modify: `packages/core/src/ragfabric_core/generate/llm.py` (remove the hardcoded Claude branch, keep the extractive path)
- Test: `packages/core/tests/test_cited_generation.py`, `packages/core/tests/test_citation_contract.py`

**Interfaces:**
- Consumes: `LLMProvider`, `Message`, `Completion` from `providers/base.py`; `RetrievedChunk`.
- Produces:
  - `CitationViolation(Exception)` with `.reason: str`.
  - `assert_citation_contract(answer: str, chunks: list[RetrievedChunk]) -> None` raising `CitationViolation`.
  - `cited_markers(answer: str) -> list[int]`.
  - `CitedAnswer` model: `text: str`, `model: str`, `input_tokens: int`, `output_tokens: int`, `latency_ms: int`, `generator: Literal["llm", "extractive"]`, `retried: bool`.
  - `generate_cited_answer(query, chunks, llm, *, model=None, max_tokens=800, extractive_fallback=True) -> CitedAnswer`.
- `generation` config value: `strategies.traditional["generation"]`, one of `llm` or `extractive`, default `llm`.

**The contract, stated precisely.** Phase 2's invariant was that every clause of an answer is a literal substring of its cited chunk. That is true of an extractive generator and impossible for fluent prose, so keeping it would forbid LLM generation and dropping it would leave nothing mechanical between a citation and an invention. The replacement has three parts, each independently checkable:

1. **Marker validity.** Every `[n]` in the answer satisfies `1 <= n <= len(chunks)`. A marker pointing at a chunk that was never retrieved is a fabricated citation.
2. **Quote fidelity.** Every span inside double quotation marks in the answer appears, after whitespace normalisation, in at least one chunk the answer cites. The model may paraphrase freely; the moment it claims to quote, the quote is verified.
3. **Grounding.** When chunks were supplied and the answer is not the explicit no evidence sentence, the answer carries at least one marker. An answer with evidence available and no citation is unattributable.

What the contract deliberately does not check is whether the paraphrase is faithful to the source. That is a semantic judgement, it cannot be asserted mechanically, and it is exactly what the Phase 8 faithfulness metric measures. Claiming otherwise would be a fabricated guarantee, which ADR 0004 forbids.

**On violation:** one retry with the violation named in the prompt, then, if `extractive_fallback` is on, the extractive generator, whose output satisfies the contract by construction. `CitedAnswer.generator` and `.retried` record which path produced the text, so a run's trace never presents a fallback as a model answer.

- [ ] **Step 1: Write the failing contract tests**

Create `packages/core/tests/test_citation_contract.py`:

```python
import pytest

from ragfabric_core.generate.contract import (
    CitationViolation,
    assert_citation_contract,
    cited_markers,
)
from ragfabric_core.strategies.base import RetrievedChunk


def chunks(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=i, document_id=1, collection_id=None, text=t, score=0.9)
        for i, t in enumerate(texts, start=1)
    ]


def test_cited_markers_reads_every_marker_in_order_without_duplicates():
    assert cited_markers("First [1] then [2] and again [1].") == [1, 2]


def test_a_valid_answer_passes():
    assert_citation_contract(
        'Staff get 24 days of annual leave [1], and it accrues monthly [2].',
        chunks("Employees receive 24 days of annual leave.", "Leave accrues monthly."),
    )


def test_a_marker_pointing_past_the_retrieved_set_is_a_violation():
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract("The policy says so [4].", chunks("a", "b"))
    assert "marker" in exc.value.reason


def test_a_quote_that_is_not_in_any_cited_chunk_is_a_violation():
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract(
            'The handbook says "you get forty days of leave" [1].',
            chunks("Employees receive 24 days of annual leave."),
        )
    assert "quote" in exc.value.reason


def test_a_quote_matching_the_chunk_apart_from_whitespace_passes():
    assert_citation_contract(
        'It says "24 days of annual leave" [1].',
        chunks("Employees receive 24   days of\nannual leave."),
    )


def test_a_paraphrase_without_quotation_marks_passes():
    assert_citation_contract(
        "Staff are entitled to just under five weeks off each year [1].",
        chunks("Employees receive 24 days of annual leave."),
    )


def test_an_answer_with_evidence_but_no_marker_is_a_violation():
    with pytest.raises(CitationViolation) as exc:
        assert_citation_contract("Staff get 24 days.", chunks("Employees receive 24 days."))
    assert "uncited" in exc.value.reason


def test_the_no_evidence_sentence_is_allowed_without_a_marker():
    assert_citation_contract(
        "I could not find an answer to that in the documents provided.", chunks("unrelated text")
    )


def test_an_empty_chunk_list_allows_an_uncited_answer():
    assert_citation_contract("I could not find an answer to that.", [])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest packages/core/tests/test_citation_contract.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragfabric_core.generate.contract'`.

- [ ] **Step 3: Write the contract module**

Create `packages/core/src/ragfabric_core/generate/contract.py`:

```python
"""The citation contract: what a cited answer must satisfy, mechanically.

Phase 2's extractive generator guaranteed that every clause was a literal
substring of its chunk. Fluent generation cannot satisfy that and a check that
forbids generation is not a check, it is a ban. These three rules are what can
be verified without a semantic judgement:

1. marker validity  every [n] refers to a chunk that was actually retrieved
2. quote fidelity   anything in double quotes really appears in a cited chunk
3. grounding        an answer with evidence available carries a citation

Faithfulness of a paraphrase is deliberately NOT checked here. It is a semantic
property, it cannot be asserted, and Phase 8 measures it. Asserting it would be
a fabricated guarantee (ADR 0004).
"""

from __future__ import annotations

import re

from ragfabric_core.strategies.base import RetrievedChunk

_MARKER_RE = re.compile(r"\[(\d+)\]")
_QUOTE_RE = re.compile(r"\"([^\"]{8,})\"")
NO_EVIDENCE = "could not find"


class CitationViolation(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def cited_markers(answer: str) -> list[int]:
    """Marker numbers in first appearance order, without duplicates."""
    seen: list[int] = []
    for raw in _MARKER_RE.findall(answer):
        n = int(raw)
        if n not in seen:
            seen.append(n)
    return seen


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def assert_citation_contract(answer: str, chunks: list[RetrievedChunk]) -> None:
    markers = cited_markers(answer)

    for n in markers:
        if n < 1 or n > len(chunks):
            raise CitationViolation(
                f"marker [{n}] does not point at any of the {len(chunks)} retrieved chunks"
            )

    if chunks and not markers and NO_EVIDENCE not in answer.lower():
        raise CitationViolation(
            "uncited answer: evidence was supplied but the answer carries no [n] marker"
        )

    cited_text = " ".join(_flat(chunks[n - 1].text) for n in markers)
    for quote in _QUOTE_RE.findall(answer):
        if _flat(quote) not in cited_text:
            raise CitationViolation(
                f"quote {quote[:60]!r} does not appear in any chunk the answer cites"
            )
```

- [ ] **Step 4: Run the contract tests**

Run: `uv run pytest packages/core/tests/test_citation_contract.py -q`
Expected: 9 passed.

- [ ] **Step 5: Write the failing generation tests**

Create `packages/core/tests/test_cited_generation.py`:

```python
from ragfabric_core.generate.cited import generate_cited_answer
from ragfabric_core.providers.base import Completion
from ragfabric_core.strategies.base import RetrievedChunk


def chunks(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=i, document_id=1, collection_id=None, text=t, score=0.9)
        for i, t in enumerate(texts, start=1)
    ]


class ScriptedLLM:
    name = "scripted"
    default_model = "scripted"

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.prompts: list[str] = []

    def complete(self, messages, *, model=None, max_tokens=1024, temperature=0.0, json_schema=None):
        self.prompts.append(messages[-1].content)
        text = self._texts.pop(0) if self._texts else ""
        return Completion(
            text=text,
            model="scripted",
            provider="scripted",
            input_tokens=10,
            output_tokens=5,
            latency_ms=3,
        )


def test_a_valid_first_answer_is_returned_as_is():
    llm = ScriptedLLM("Staff get 24 days of annual leave [1].")
    out = generate_cited_answer(
        "how much leave", chunks("Employees receive 24 days of annual leave."), llm
    )
    assert out.text == "Staff get 24 days of annual leave [1]."
    assert out.generator == "llm"
    assert out.retried is False
    assert out.input_tokens == 10 and out.output_tokens == 5


def test_the_numbered_passages_reach_the_prompt():
    llm = ScriptedLLM("Answer [1].")
    generate_cited_answer("q", chunks("first passage", "second passage"), llm)
    prompt = llm.prompts[0]
    assert "[1] first passage" in prompt
    assert "[2] second passage" in prompt


def test_a_contract_violation_is_retried_once_with_the_reason_in_the_prompt():
    llm = ScriptedLLM("The policy says so [9].", "Staff get 24 days [1].")
    out = generate_cited_answer("q", chunks("Employees receive 24 days."), llm)
    assert out.text == "Staff get 24 days [1]."
    assert out.retried is True
    assert "marker" in llm.prompts[1]


def test_two_violations_fall_back_to_the_extractive_generator():
    llm = ScriptedLLM("nonsense [9].", "still nonsense [9].")
    out = generate_cited_answer("leave", chunks("Employees receive 24 days of annual leave."), llm)
    assert out.generator == "extractive"
    assert out.retried is True
    assert "[1]" in out.text


def test_the_extractive_fallback_can_be_switched_off():
    import pytest

    from ragfabric_core.generate.contract import CitationViolation

    llm = ScriptedLLM("nonsense [9].", "still nonsense [9].")
    with pytest.raises(CitationViolation):
        generate_cited_answer("q", chunks("text"), llm, extractive_fallback=False)


def test_no_chunks_produces_the_no_evidence_sentence_without_calling_the_model():
    llm = ScriptedLLM()
    out = generate_cited_answer("q", [], llm)
    assert "could not find" in out.text.lower()
    assert llm.prompts == []
    assert out.generator == "extractive"
```

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run pytest packages/core/tests/test_cited_generation.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragfabric_core.generate.cited'`.

- [ ] **Step 7: Write the generator**

Create `packages/core/src/ragfabric_core/generate/cited.py`:

```python
"""Generate a cited answer from retrieved chunks, and verify it before returning.

The model is told to cite with [n], to quote only verbatim, and to say it cannot
find the answer rather than guess. The output is then checked against the
citation contract. A violation is retried once with the specific reason quoted
back, because a named failure is repairable and a generic "try again" is not. A
second violation falls back to the extractive generator, which satisfies the
contract by construction, and the result records which generator produced it so
nothing downstream mistakes a fallback for a model answer.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel

from ragfabric_core.generate import llm as extractive
from ragfabric_core.generate.contract import CitationViolation, assert_citation_contract
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.strategies.base import RetrievedChunk

NO_EVIDENCE_ANSWER = "I could not find an answer to that in the documents provided."

_SYSTEM = (
    "You answer strictly from the numbered passages provided. "
    "Cite every claim with the passage number in square brackets, for example [1]. "
    "Quote verbatim only, inside double quotation marks; paraphrase everything else. "
    "If the passages do not answer the question, reply exactly: "
    f"{NO_EVIDENCE_ANSWER}"
)


class CitedAnswer(BaseModel):
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    generator: Literal["llm", "extractive"] = "llm"
    retried: bool = False


def build_prompt(query: str, chunks: list[RetrievedChunk], violation: str | None = None) -> str:
    passages = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
    prompt = f"Question: {query}\n\nPassages:\n{passages}"
    if violation is not None:
        prompt += (
            "\n\nYour previous answer was rejected: "
            f"{violation}. Answer again, obeying the citation rules exactly."
        )
    return prompt


def generate_cited_answer(
    query: str,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
    *,
    model: str | None = None,
    max_tokens: int = 800,
    extractive_fallback: bool = True,
) -> CitedAnswer:
    started = time.perf_counter()
    if not chunks:
        return CitedAnswer(
            text=NO_EVIDENCE_ANSWER,
            model="none",
            generator="extractive",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    violation: str | None = None
    input_tokens = 0
    output_tokens = 0
    last_model = model or getattr(llm, "default_model", "unknown")

    for attempt in (1, 2):
        completion = llm.complete(
            [
                Message(role="system", content=_SYSTEM),
                Message(role="user", content=build_prompt(query, chunks, violation)),
            ],
            model=model,
            max_tokens=max_tokens,
        )
        input_tokens += completion.input_tokens
        output_tokens += completion.output_tokens
        last_model = completion.model
        try:
            assert_citation_contract(completion.text, chunks)
        except CitationViolation as exc:
            violation = exc.reason
            if attempt == 2:
                break
            continue
        return CitedAnswer(
            text=completion.text,
            model=completion.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            generator="llm",
            retried=attempt == 2,
        )

    if not extractive_fallback:
        raise CitationViolation(violation or "citation contract not satisfied")

    text = extractive.extractive_answer(query, [c.model_dump() for c in chunks])
    return CitedAnswer(
        text=text,
        model=last_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
        generator="extractive",
        retried=True,
    )
```

- [ ] **Step 8: Run the generation tests**

Run: `uv run pytest packages/core/tests/test_cited_generation.py -q`
Expected: 6 passed. If `extractive_answer` expects a different dict shape than `RetrievedChunk.model_dump()`, adapt the call to the real signature in `generate/llm.py` rather than changing that function, and note the adaptation in the ledger.

- [ ] **Step 9: Remove the hardcoded Claude branch**

In `packages/core/src/ragfabric_core/generate/llm.py`, delete the `CLAUDE_MODEL` constant and the optional Anthropic branch that reads `ANTHROPIC_API_KEY` directly. Generation now goes through the configured `LLMProvider`, so a second hardcoded vendor path in this module is a duplicate route with different behaviour. Keep `extractive_answer` and the marker helpers exactly as they are, update the module docstring to say that the extractive generator is the contract satisfying fallback and the `generation: extractive` mode, and delete any test that asserted the Anthropic branch existed.

- [ ] **Step 10: Run the whole suite**

Run: `uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: `271 passed, 3 skipped`, minus any Anthropic branch test deleted in step 9. Record the exact number in the ledger.

- [ ] **Step 11: Commit**

```bash
git add packages/core/src/ragfabric_core/generate/ packages/core/tests/test_citation_contract.py packages/core/tests/test_cited_generation.py
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: generate cited answers and verify them against a mechanical citation contract"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 10: Rewire `/query`, `/semantic` and `/hybrid` onto the real stores

**Files:**
- Modify: `packages/server/src/ragfabric_server/deps.py`, `packages/server/src/ragfabric_server/api/routes/search.py`, `packages/core/src/ragfabric_core/generate/answer.py`
- Test: `packages/server/tests/test_search_real_path.py`, additions to `packages/server/tests/test_access_enforcement.py`

**Interfaces:**
- Consumes: Task 8's `default_registry`, Task 9's `generate_cited_answer`, Task 6's `build_reranker`.
- Produces in `deps.py`: `get_strategy_registry() -> StrategyRegistry` (built once per process from config), `get_llm_provider() -> LLMProvider`, `get_vector_store() -> VectorStore`, `get_lexical_store() -> LexicalStore`.
- Produces in `answer.py`: `build_answer(query, retrieved, answer_text: str | None = None) -> dict`. When `answer_text` is given it is used verbatim as the answer and the rest of the payload (confidence, citations, highlights, source document) is derived exactly as before. When it is `None` the extractive path runs, unchanged.
- `AnswerResponse` keeps every field it has today. No response shape changes on any existing endpoint.

Why `build_answer` gains a parameter instead of being replaced: it produces the citation list, per snippet highlight offsets, the supporting span and the confidence score that the shipped Angular app renders, and its invariant that every offset indexes a string also present in the payload is worth keeping. Only the source of the answer text changes. Rewriting the packaging would risk the offsets for no gain.

Why `/hybrid` moves to pgvector plus `PostgresLexicalStore` now: the endpoint exists and is documented, and after Task 13 deletes the in memory hybrid retriever it would otherwise have no implementation. The fusion used here is the simple normalised sum the v1 code used. Phase 4 replaces it with BM25, phrase and identifier boosting, which is that phase's whole subject. Doing it here would pull Phase 4 forward; leaving the endpoint broken for a phase is worse than shipping simple fusion with a comment saying which phase improves it.

- [ ] **Step 1: Write the failing route test**

Create `packages/server/tests/test_search_real_path.py`:

```python
"""The query path runs on the pgvector table, not the in memory index."""

from __future__ import annotations


def test_query_answers_from_the_indexed_corpus_with_citations(client, admin_token, ingested_doc):
    res = client.post(
        "/api/search/query",
        json={"query": "how much annual leave", "top_k": 3},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["answer"]
    assert body["citations"], "an answer over an indexed corpus must carry citations"
    assert all(c["chunk_id"] is not None for c in body["citations"])
    for highlight in body["highlights"]:
        assert body["answer"][highlight["start"] : highlight["end"]] == highlight["term"]


def test_query_records_the_real_embedding_model_on_the_run(
    client, admin_token, ingested_doc, db_session
):
    from ragfabric_core.models.runs import RetrievalRun

    client.post(
        "/api/search/query",
        json={"query": "leave", "top_k": 3},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    run = db_session.query(RetrievalRun).order_by(RetrievalRun.id.desc()).first()
    assert run is not None
    assert not run.embedding_model.startswith("hashing-") or run.embedding_model == "hashing-384", (
        "embedding_model must be the provider's real model name, not a hardcoded string"
    )
    assert [s["name"] for s in run.trace][:2] == ["embed_query", "vector_search"]


def test_semantic_returns_ranked_chunks_from_the_vector_store(client, admin_token, ingested_doc):
    res = client.post(
        "/api/search/semantic",
        json={"query": "leave", "top_k": 5},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 200
    results = res.json()["results"]
    assert results
    assert all(r["score"] is not None for r in results)


def test_hybrid_still_returns_results_after_the_rewire(client, admin_token, ingested_doc):
    res = client.post(
        "/api/search/hybrid",
        json={"query": "leave", "top_k": 5, "mode": "hybrid"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 200
    assert res.json()["results"]
```

The `ingested_doc` fixture must upload a small text document through the API so the fan out writes real `chunk_embeddings` and `chunk_search` rows. If `packages/server/tests/conftest.py` has no such fixture, add one there that posts a text file with a sentence about annual leave and waits for status `ready`. The server test config already selects the offline hashing embedder, so no network is involved.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest packages/server/tests/test_search_real_path.py -q`
Expected: FAIL. The trace assertion fails first because the current route emits a `hybrid_search` span from the in memory retriever.

- [ ] **Step 3: Add the dependencies**

In `packages/server/src/ragfabric_server/deps.py`, add, following the existing single instance pattern used for the cache:

```python
@lru_cache(maxsize=1)
def _registry() -> StrategyRegistry:
    from ragfabric_core.runtime import get_config, get_session_factory
    from ragfabric_core.strategies.registry_defaults import default_registry

    return default_registry(get_config(), get_session_factory())


def get_strategy_registry() -> StrategyRegistry:
    return _registry()


@lru_cache(maxsize=1)
def _llm():
    from ragfabric_core.providers.registry import build_llm_provider
    from ragfabric_core.runtime import get_config

    return build_llm_provider(get_config().llm)


def get_llm_provider():
    return _llm()


@lru_cache(maxsize=1)
def _vector_store():
    from ragfabric_core.providers.registry import build_embedding_provider
    from ragfabric_core.runtime import get_config, get_session_factory
    from ragfabric_core.stores.registry import build_vector_store

    cfg = get_config()
    embedder = build_embedding_provider(cfg.embeddings)
    return build_vector_store(cfg.vector_store, get_session_factory(), embedding_model=embedder.model)


def get_vector_store():
    return _vector_store()


@lru_cache(maxsize=1)
def _lexical_store():
    from ragfabric_core.runtime import get_config, get_session_factory
    from ragfabric_core.stores.registry import build_lexical_store

    return build_lexical_store(get_config().lexical_store, get_session_factory())


def get_lexical_store():
    return _lexical_store()
```

Task 16 replaces these `lru_cache` singletons with a state object on the FastAPI app, which is one of the deferred items. Keep the same pattern the module already uses until then rather than inventing a third one here.

- [ ] **Step 4: Rewrite the route bodies**

In `packages/server/src/ragfabric_server/api/routes/search.py`:

Replace the `_retrieve` helper with one that builds a `RetrievalContext` and calls the strategy:

```python
def _context(payload: SearchRequest, principal: Principal, access: AccessFilter) -> RetrievalContext:
    filters: dict[str, str | int | float | bool] = {}
    if payload.document_id is not None:
        filters["document_id"] = payload.document_id
    if payload.format is not None:
        filters["format"] = payload.format
    return RetrievalContext(
        principal=principal,
        access_filter=access,
        collection_ids=[payload.collection_id] if payload.collection_id is not None else None,
        params=StrategyParams(
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
            metadata_filters=filters,
        ),
    )
```

In `query`, replace the retrieval and answer block with:

```python
    started = time.perf_counter()
    strategy = registry.get(StrategyName.TRADITIONAL)
    with start_trace() as tracing:
        result = strategy.retrieve(payload.query, _context(payload, principal, access))
        retrieval_ms = int((time.perf_counter() - started) * 1000)
        with trace("answer"):
            cited = generate_cited_answer(
                payload.query, result.chunks, llm, model=None, max_tokens=800
            )
        payload_rows = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "collection_id": c.collection_id,
                "text": c.text,
                "page": c.page,
                "score": c.score,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "filename": c.metadata.get("filename"),
                "format": c.metadata.get("format"),
            }
            for c in result.chunks
        ]
        answer = build_answer(payload.query, payload_rows, answer_text=cited.text)
    total_ms = int((time.perf_counter() - started) * 1000)
```

Then, in the `RetrievalRun` construction, replace the hardcoded fields with measured ones:

```python
        llm_calls=result.llm_calls + (0 if cited.generator == "extractive" else 1),
        retrieval_calls=result.retrieval_calls,
        input_tokens=result.input_tokens + cited.input_tokens,
        output_tokens=result.output_tokens + cited.output_tokens,
        embedding_model=embedding_model,
        trace=[s.model_dump() for s in result.trace] + [s.model_dump() for s in tracing.spans],
```

where `embedding_model` comes from the store's active model. Add a dependency for it rather than reconstructing the provider in the route:

```python
def get_embedding_model(store=Depends(get_vector_store)) -> str:
    return getattr(store, "_model", None) or "unknown"
```

Put that helper in `deps.py`, not in the route module, and export it. Accessing a private attribute is not acceptable in shipped code, so give `PgVectorStore` and `ChromaVectorStore` a public read only `model` property in this step and read that instead:

```python
    @property
    def model(self) -> str | None:
        return self._model
```

Add the same property to both stores and use `store.model` in the dependency.

For `/semantic`, call the strategy and map `RetrievedChunk` to `SearchResultItem`. For `/hybrid`, call the strategy for the vector side, call `lexical.search(...)` for the keyword side with the same access filter, and fuse with the normalised sum:

```python
def _fuse(
    vector_hits: list[RetrievedChunk], lexical_hits: list[RetrievedChunk], top_k: int
) -> list[tuple[RetrievedChunk, float, float]]:
    """Normalised score sum. Phase 4 replaces this with BM25 plus boosting."""
    def norm(hits: list[RetrievedChunk]) -> dict[int, float]:
        scores = [h.score or 0.0 for h in hits]
        top = max(scores, default=0.0)
        return {h.chunk_id: ((h.score or 0.0) / top if top > 0 else 0.0) for h in hits}

    v, lex = norm(vector_hits), norm(lexical_hits)
    by_id = {h.chunk_id: h for h in vector_hits} | {h.chunk_id: h for h in lexical_hits}
    fused = [
        (by_id[cid], v.get(cid, 0.0), lex.get(cid, 0.0))
        for cid in by_id
    ]
    fused.sort(key=lambda row: (-(row[1] + row[2]), row[0].chunk_id))
    return fused[:top_k]
```

Keep every `AuditLog`, `QueryLog` and `Source` write exactly as it is. Only the source of the chunks and the answer text changes.

- [ ] **Step 5: Add the `answer_text` parameter**

In `packages/core/src/ragfabric_core/generate/answer.py`, change the signature to `def build_answer(query: str, retrieved: list[dict], answer_text: str | None = None) -> dict:` and, where the answer text is currently produced by the extractive call, use `answer_text` when it is not None. Everything downstream of that line, confidence, citations, highlights and the source document, is unchanged, because all of it is derived from the answer string and the retrieved rows.

- [ ] **Step 6: Run the route tests**

Run: `uv run pytest packages/server/tests -q`
Expected: all pass, including the existing `test_access_enforcement.py` and `test_api.py`. Any failure in an existing test means the response shape moved, which this task must not do; fix the route, not the test.

- [ ] **Step 7: Add an access enforcement test on the new path**

Add to `packages/server/tests/test_access_enforcement.py`:

```python
def test_a_restricted_user_never_sees_a_forbidden_chunk_on_the_real_vector_path(
    client, restricted_token, restricted_setup
):
    res = client.post(
        "/api/search/query",
        json={"query": "confidential salary band", "top_k": 10},
        headers={"Authorization": f"Bearer {restricted_token}"},
    )
    assert res.status_code == 200
    forbidden = restricted_setup["forbidden_document_id"]
    assert all(c["document_id"] != forbidden for c in res.json()["citations"])
```

Use the fixtures that file already defines for the restricted principal; do not create a parallel set.

- [ ] **Step 8: Run the whole suite and commit**

Run: `uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: every test passes. Record the count in the ledger.

```bash
git add packages/server/src/ragfabric_server packages/core/src/ragfabric_core/generate/answer.py packages/core/src/ragfabric_core/stores/pgvector_store.py packages/core/src/ragfabric_core/stores/chroma_store.py packages/server/tests/
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: run the query, semantic and hybrid endpoints on the real stores"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 11: `LLMProvider.stream()` and `POST /api/ask` with SSE

**Files:**
- Modify: `packages/core/src/ragfabric_core/providers/base.py`, `providers/openai_compat.py`, `providers/anthropic_provider.py`, `providers/offline.py`
- Create: `packages/server/src/ragfabric_server/api/routes/ask.py`, `packages/server/src/ragfabric_server/schemas/ask.py`
- Modify: `packages/server/src/ragfabric_server/main.py`
- Test: `packages/core/tests/test_provider_streaming.py`, `packages/server/tests/test_ask_sse.py`

**Interfaces:**
- Produces on the protocol: `def stream(self, messages, *, model=None, max_tokens=1024, temperature=0.0) -> Iterator[str]` yielding text deltas, then returning. Every provider implements it. `ScriptedLLMProvider.stream` yields its scripted text word by word so tests need no network.
- Produces `AskRequest`: `query`, `top_k`, `similarity_threshold`, `collection_id`, `document_id`, `format`, `strategy` (default `traditional`), `stream` (default `true`).
- Produces `POST /api/ask`. With `stream: true` it returns `text/event-stream` with these events in order: `retrieval` (one JSON object with the chunk count and the trace so far), zero or more `token` events carrying `{"text": "..."}`, one `citations` event with the full citation list, one `done` event with `{"run_id": n, "latency_ms": n}`. With `stream: false` it returns the same JSON body as `/api/search/query`.

Why the citations arrive after the tokens rather than before: the `used` flag on a citation is derived from the markers the answer actually contains, so it is unknowable until the last token. Sending a citation list up front would mean sending a `used` value that is a guess, and a client that renders it would show a source as cited before the answer cites it.

Why the run row is written after streaming finishes: latency, token counts and the citation list are all only final at the end, and a row written up front would have to be updated, leaving a window in which `/api/runs/{id}` returns a half recorded run. The `done` event carries the `run_id` so a client can fetch the complete record immediately.

- [ ] **Step 1: Write the failing provider streaming test**

Create `packages/core/tests/test_provider_streaming.py`:

```python
from ragfabric_core.providers.base import LLMProvider, Message
from ragfabric_core.providers.offline import ScriptedLLMProvider


def test_the_scripted_provider_streams_its_text_in_pieces():
    provider = ScriptedLLMProvider(responses=["one two three"], model="scripted")
    pieces = list(provider.stream([Message(role="user", content="q")]))
    assert "".join(pieces) == "one two three"
    assert len(pieces) > 1, "a stream of one piece is not a stream"


def test_the_scripted_provider_still_satisfies_the_protocol_after_the_addition():
    assert isinstance(ScriptedLLMProvider(responses=[], model="scripted"), LLMProvider)


def test_streaming_an_exhausted_script_yields_nothing_rather_than_raising():
    provider = ScriptedLLMProvider(responses=[], model="scripted")
    assert list(provider.stream([Message(role="user", content="q")])) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest packages/core/tests/test_provider_streaming.py -q`
Expected: FAIL, `AttributeError: 'ScriptedLLMProvider' object has no attribute 'stream'`.

- [ ] **Step 3: Add `stream` to the protocol and every provider**

In `providers/base.py`, add to the `LLMProvider` protocol, with the import `from collections.abc import Iterator`:

```python
    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]: ...
```

In `providers/offline.py`, on `ScriptedLLMProvider`:

```python
    def stream(self, messages, *, model=None, max_tokens=1024, temperature=0.0):
        """Yield the scripted response word by word so tests exercise a real stream."""
        if not self._responses:
            return
        text = self._responses.pop(0)
        parts = text.split(" ")
        for i, word in enumerate(parts):
            yield word if i == len(parts) - 1 else word + " "
```

Match the real attribute name for the response list in that class rather than assuming `self._responses`.

In `providers/openai_compat.py`, on the shared client class, add a `stream` that sets `stream=True` on the chat completions call and yields `chunk.choices[0].delta.content` when it is not None. This one implementation covers both `OpenAIProvider` and `OllamaProvider`, which is why the module is shared.

In `providers/anthropic_provider.py`, add a `stream` using the SDK's streaming context manager, yielding each text delta.

- [ ] **Step 4: Run the provider tests**

Run: `uv run pytest packages/core/tests/test_provider_streaming.py packages/core/tests/test_providers.py -q`
Expected: all pass. If a `runtime_checkable` isinstance check now fails for a provider missing `stream`, that provider was missed; add it.

- [ ] **Step 5: Write the failing SSE test**

Create `packages/server/tests/test_ask_sse.py`:

```python
"""POST /api/ask streams retrieval, tokens, citations and a run id, in that order."""

from __future__ import annotations

import json


def _events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.strip().split("\n\n"):
        name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        if name:
            out.append((name, data))
    return out


def test_ask_streams_events_in_the_documented_order(client, admin_token, ingested_doc):
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "how much annual leave", "top_k": 3, "stream": True},
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        events = _events("".join(res.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "retrieval"
    assert "token" in names
    assert names[-2] == "citations"
    assert names[-1] == "done"
    assert names.index("citations") > max(i for i, n in enumerate(names) if n == "token"), (
        "citations must arrive after the last token, because used is only known then"
    )


def test_the_done_event_carries_a_run_id_that_resolves(client, admin_token, ingested_doc):
    with client.stream(
        "POST",
        "/api/ask",
        json={"query": "leave", "stream": True},
        headers={"Authorization": f"Bearer {admin_token}"},
    ) as res:
        events = _events("".join(res.iter_text()))
    run_id = dict(events)["done"]["run_id"]

    res = client.get(f"/api/runs/{run_id}", headers={"Authorization": f"Bearer {admin_token}"})
    assert res.status_code == 200
    assert res.json()["sources"]


def test_ask_without_streaming_returns_the_same_shape_as_query(client, admin_token, ingested_doc):
    res = client.post(
        "/api/ask",
        json={"query": "leave", "stream": False},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 200
    body = res.json()
    assert set(body) >= {"question", "answer", "confidence", "citations", "highlights"}


def test_ask_applies_the_access_filter(client, restricted_token, restricted_setup):
    res = client.post(
        "/api/ask",
        json={"query": "confidential salary band", "stream": False},
        headers={"Authorization": f"Bearer {restricted_token}"},
    )
    assert res.status_code == 200
    forbidden = restricted_setup["forbidden_document_id"]
    assert all(c["document_id"] != forbidden for c in res.json()["citations"])
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest packages/server/tests/test_ask_sse.py -q`
Expected: FAIL with 404, the route does not exist.

- [ ] **Step 7: Write the schemas**

Create `packages/server/src/ragfabric_server/schemas/ask.py`:

```python
"""Request and event schemas for POST /api/ask."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    collection_id: int | None = None
    document_id: int | None = None
    format: str | None = Field(default=None, pattern="^(pdf|docx|pptx|txt|csv|md)$")
    strategy: str = Field(default="traditional", pattern="^(traditional)$")
    stream: bool = True
```

The `strategy` pattern accepts only `traditional` in this phase and each later phase widens it. A field that accepts a strategy name the server cannot serve would be a promise the API does not keep.

- [ ] **Step 8: Write the route**

Create `packages/server/src/ragfabric_server/api/routes/ask.py` implementing the event order above. The generator retrieves first, emits `retrieval`, streams the model's deltas as `token` events while accumulating the full text, then validates the accumulated text against the citation contract, falling back exactly as Task 9 does when it fails, then emits `citations` built by `build_answer`, writes the `RetrievalRun`, `Source`, `AuditLog` and `QueryLog` rows in one transaction, and emits `done` with the run id. Use `fastapi.responses.StreamingResponse` with `media_type="text/event-stream"` and the headers `Cache-Control: no-cache` and `X-Accel-Buffering: no`, and format each event as `event: <name>\ndata: <json>\n\n`.

One constraint the implementer must respect: the database session must not be held open across the token stream. Retrieve and generate first, then open a session for the writes. A session held for the whole stream pins a connection for as long as the client reads, which exhausts the pool under any real concurrency.

Register the router in `main.py` with `prefix="/api"` so the path is exactly `/api/ask`, beside the existing routers.

- [ ] **Step 9: Run the tests**

Run: `uv run pytest packages/server/tests/test_ask_sse.py -q`
Expected: 4 passed.

- [ ] **Step 10: Run the whole suite and commit**

Run: `uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: all pass; record the count.

```bash
git add packages/core/src/ragfabric_core/providers packages/server/src/ragfabric_server packages/core/tests/test_provider_streaming.py packages/server/tests/test_ask_sse.py
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: add provider streaming and POST /api/ask with server sent events"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 12: Per request parameter overrides

**Files:**
- Modify: `packages/server/src/ragfabric_server/schemas/search.py`, `packages/server/src/ragfabric_server/api/routes/documents.py`, `packages/core/src/ragfabric_core/ingest/pipeline.py`
- Test: `packages/server/tests/test_overrides.py`

**Interfaces:**
- Produces: `SearchRequest.similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)` and `SearchRequest.rerank: Literal["none", "llm", "cross_encoder"] | None = None`; upload accepts optional form fields `chunk_size` and `chunk_overlap`; `ingest_document(..., chunk_size: int | None = None, chunk_overlap: int | None = None)` where `None` means "use the configured value".
- Defaults are unchanged when a field is absent, so every existing client keeps its current behaviour.

Why the request may override retrieval parameters but not the embedding model or the store: `top_k`, the threshold and the reranker change how a fixed index is searched, which is a per question decision and is exactly what a Compare page needs. The embedding model and the store determine what the index physically is; changing them per request would mean querying vectors that do not exist. This is the same boundary ADR 0006 draws, expressed in the API surface.

Why `chunk_size` is per upload rather than per query: chunking happens once, at ingestion. A per query chunk size would require re-chunking and re-embedding on every question, which is not a tuning knob but a full reindex.

- [ ] **Step 1: Write the failing tests**

Create `packages/server/tests/test_overrides.py`:

```python
def test_similarity_threshold_in_the_request_filters_results(client, admin_token, ingested_doc):
    loose = client.post(
        "/api/search/semantic",
        json={"query": "entirely unrelated aardvark", "top_k": 10, "similarity_threshold": 0.0},
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()["results"]
    strict = client.post(
        "/api/search/semantic",
        json={"query": "entirely unrelated aardvark", "top_k": 10, "similarity_threshold": 0.99},
        headers={"Authorization": f"Bearer {admin_token}"},
    ).json()["results"]
    assert len(strict) <= len(loose)
    assert all(r["score"] >= 0.99 for r in strict)


def test_top_k_is_honoured_per_request(client, admin_token, ingested_doc):
    res = client.post(
        "/api/search/semantic",
        json={"query": "leave", "top_k": 1},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert len(res.json()["results"]) <= 1


def test_an_out_of_range_threshold_is_rejected_rather_than_clamped(client, admin_token):
    res = client.post(
        "/api/search/semantic",
        json={"query": "leave", "similarity_threshold": 1.5},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 422


def test_upload_accepts_a_chunk_size_override(client, admin_token, tmp_path):
    body = ("sentence one. " * 200).encode()
    res = client.post(
        "/api/documents/upload",
        files={"file": ("big.txt", body, "text/plain")},
        data={"chunk_size": "200", "chunk_overlap": "20"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code in (200, 201), res.text
    small = res.json()["chunk_count"]

    res = client.post(
        "/api/documents/upload",
        files={"file": ("big2.txt", body, "text/plain")},
        data={"chunk_size": "2000", "chunk_overlap": "20"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    large = res.json()["chunk_count"]
    assert small > large, "a smaller chunk size must produce more chunks"


def test_an_invalid_chunk_overlap_is_rejected(client, admin_token):
    res = client.post(
        "/api/documents/upload",
        files={"file": ("x.txt", b"hello", "text/plain")},
        data={"chunk_size": "100", "chunk_overlap": "500"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res.status_code == 422, "overlap larger than the chunk size must be refused"
```

Use the upload route's real path and response field names; check `documents.py` and `test_api.py` for both rather than assuming `chunk_count`.

- [ ] **Step 2: Run to verify it fails, then implement**

Run: `uv run pytest packages/server/tests/test_overrides.py -q`
Expected: FAIL on the threshold field and the form fields.

Add the two fields to `SearchRequest`, thread `similarity_threshold` through `_context` (Task 10 already reads it), add `chunk_size` and `chunk_overlap` as optional `Form` fields on the upload route, validate `0 <= chunk_overlap < chunk_size` and raise `HTTPException(422, ...)` otherwise, and pass both into `ingest_document`, which falls back to `get_config().ingestion` values when either is `None`.

For `rerank`, build the named reranker per request through `build_reranker` and pass it to a per request copy of the strategy. Do not mutate the shared strategy instance: it is a process singleton and mutating it would leak one request's choice into every concurrent request. `TraditionalRAGStrategy` is cheap to construct around the shared store and provider, so construct one when `rerank` is present and use the registry's instance otherwise.

- [ ] **Step 3: Run the tests, the suite, and commit**

Run: `uv run pytest packages/server/tests -q && uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: all pass.

```bash
git add packages/server/src/ragfabric_server packages/core/src/ragfabric_core/ingest/pipeline.py packages/server/tests/test_overrides.py
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: accept per request retrieval overrides and per upload chunking overrides"
git log -1 --format='%(trailers)'
```

---

### Task 13: Retire the v1 in memory index

**Files:**
- Delete: `packages/core/src/ragfabric_core/store/vector_store.py`, `packages/core/src/ragfabric_core/store/__init__.py`, `packages/core/src/ragfabric_core/retrieve/retriever.py`, `packages/core/src/ragfabric_core/retrieve/hybrid.py`, `packages/core/src/ragfabric_core/strategies/legacy.py`
- Delete the tests that target only those modules; identify them with the grep in Step 1 rather than by guessing filenames.
- Modify: any remaining importer found by Step 1, `docs/architecture.md` (module list only; the full docs pass is Task 17)

**Interfaces:**
- Consumes: Tasks 8, 10, 11 and 12, which must all be complete, because this task removes the only other implementation.
- Produces: nothing new. This task is a deletion, and it is a separate task precisely so the diff is reviewable as a deletion rather than hidden inside a feature.

Why this is not merged into Task 10: a task that both adds the new path and deletes the old one produces a diff where a reviewer cannot tell whether a behaviour was moved or lost. Splitting them means Task 10's review answers "is the new path correct" and this review answers "is anything still depending on the old one".

- [ ] **Step 1: Find every importer**

```bash
grep -rn "ragfabric_core.store\b\|ragfabric_core\.store\.\|retrieve\.retriever\|retrieve\.hybrid\|strategies\.legacy\|LegacyHybridStrategy\|InMemoryVectorStore\|get_store\b" \
  --include="*.py" packages/ | sort
```

Write the full list into the ledger before deleting anything. Every line must end up either deleted or repointed, and the reviewer checks that list against the diff.

- [ ] **Step 2: Confirm the new path covers each use**

For each importer found, state in the ledger which of Tasks 8 to 12 replaced it. The known ones are: `search.py` uses `get_store().access_stats(...)` for the audit row's `sources_filtered`, and `answer.py` imports `content_tokens` from `ingest/embed.py`, which is a different module and stays. `access_stats` must move to the vector store interface as an optional capability, or the audit row must derive the filtered count another way. Decide, record the ruling, and implement it here.

Recommended ruling, to be confirmed by the implementer against the code: add `access_stats(filters, access) -> tuple[int, int]` to `PgVectorStore` and `ChromaVectorStore`, since both can count candidates before and after the predicate, and have the route call `store.access_stats(...)`. This keeps the audit row's `sources_filtered` a measured number rather than an estimate, which ADR 0004 requires.

- [ ] **Step 3: Delete the modules and their tests**

```bash
git rm -r packages/core/src/ragfabric_core/store packages/core/src/ragfabric_core/retrieve
git rm packages/core/src/ragfabric_core/strategies/legacy.py
```

Then remove the tests that exercise only the deleted modules, and remove `LegacyHybridStrategy` from any registry or `__init__` export.

- [ ] **Step 4: Verify nothing imports the deleted modules**

```bash
grep -rn "ragfabric_core.store\|retrieve\.retriever\|retrieve\.hybrid\|strategies\.legacy\|LegacyHybridStrategy\|InMemoryVectorStore" --include="*.py" packages/ || echo "clean"
uv run pytest -q -p no:warnings 2>&1 | tail -1
uv run lint-imports
```

Expected: `clean`, the suite passes with a lower count than before (the deleted tests are gone; record the exact number and the number deleted in the ledger), and the import contracts still pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "refactor: remove the v1 in memory index, retrievers and legacy strategy"
git log -1 --format='%(trailers)'
```

Expected: empty trailers.

---

### Task 14: `packages/sdk-python`, the typed HTTP client

**Files:**
- Create: `packages/sdk-python/pyproject.toml`, `packages/sdk-python/src/ragfabric_sdk/__init__.py`, `client.py`, `models.py`, `errors.py`, `packages/sdk-python/tests/test_client.py`
- Modify: root `pyproject.toml` (workspace members), `.importlinter` or the import contract configuration, `.github/workflows/ci.yml`

**Interfaces:**
- Produces: `Client(base_url, token=None, api_key=None, timeout=30.0)` with `ask(query, **params) -> Answer`, `ask_stream(query, **params) -> Iterator[AskEvent]`, `search(query, mode="semantic", **params) -> list[SearchResult]`, `ingest(path, collection=None) -> Document`, `documents() -> list[Document]`, `run(run_id) -> Run`; typed pydantic models in `models.py` mirroring the server schemas; `RagFabricError`, `AuthError`, `RateLimitError`, `NotFoundError` in `errors.py`.
- Package name `ragfabric-sdk`, import name `ragfabric_sdk`, version `0.1.0a1`.
- New import contract: `ragfabric_sdk` must not import `ragfabric_core`, `ragfabric_server` or `ragfabric_cli`. The contract count goes from 2 to 3.

Why the SDK talks HTTP only and never imports core: an adopter installs the SDK on a laptop or in a Lambda to call a RagFabric server they do not host. If the SDK imported core it would drag SQLAlchemy, pgvector, chromadb, tiktoken and the migrations into that environment, and a version skew between the client's core and the server's core would produce failures that look like API bugs. The contract is enforced by import linter, not by convention.

Why the client is generated by hand in this phase rather than from the OpenAPI spec: the roadmap generates the TypeScript SDK from OpenAPI in Phase 9, and an OpenAPI generator for Python would need a build step, a pinned generator version and a committed artefact in CI. Six endpoints written by hand are smaller than that toolchain, and the tests below pin the shapes to the server's real responses so drift is caught.

- [ ] **Step 1: Create the package skeleton**

`packages/sdk-python/pyproject.toml`:

```toml
[project]
name = "ragfabric-sdk"
version = "0.1.0a1"
description = "Python client for a RagFabric server."
readme = "README.md"
requires-python = ">=3.13,<3.14"
license = "Apache-2.0"
authors = [{ name = "Ranjan G", email = "ranjan.g@ispf.ngo" }]
dependencies = ["httpx>=0.28", "pydantic>=2.13"]

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ragfabric_sdk"]
```

Add `packages/sdk-python` to the workspace members in the root `pyproject.toml`, beside the existing packages, and add a `packages/sdk-python/README.md` with a five line usage example.

- [ ] **Step 2: Write the failing client tests**

Create `packages/sdk-python/tests/test_client.py`. Use `httpx.MockTransport` so the tests need no server:

```python
import json

import httpx
import pytest

from ragfabric_sdk import Client
from ragfabric_sdk.errors import AuthError, NotFoundError, RateLimitError


def transport(handler):
    return httpx.MockTransport(handler)


def client_with(handler, **kwargs):
    return Client("http://server", token="t", transport=transport(handler), **kwargs)


def test_ask_returns_a_typed_answer():
    def handler(request):
        assert request.url.path == "/api/ask"
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            json={
                "question": "q",
                "answer": "a [1]",
                "confidence": 0.8,
                "citations": [
                    {"marker": "[1]", "chunk_id": 3, "document_id": 1, "score": 0.9, "snippet": "s"}
                ],
                "highlights": [],
            },
        )

    answer = client_with(handler).ask("q")
    assert answer.answer == "a [1]"
    assert answer.citations[0].chunk_id == 3
    assert answer.confidence == pytest.approx(0.8)


def test_the_bearer_token_is_sent():
    def handler(request):
        assert request.headers["authorization"] == "Bearer t"
        return httpx.Response(200, json={"question": "q", "answer": "", "confidence": 0.0, "citations": [], "highlights": []})

    client_with(handler).ask("q")


def test_an_api_key_is_sent_in_the_x_api_key_header():
    def handler(request):
        assert request.headers["x-api-key"] == "rf_abc"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"question": "q", "answer": "", "confidence": 0.0, "citations": [], "highlights": []})

    Client("http://server", api_key="rf_abc", transport=transport(handler)).ask("q")


def test_ask_stream_yields_parsed_events():
    body = (
        'event: retrieval\ndata: {"chunks": 2}\n\n'
        'event: token\ndata: {"text": "hello "}\n\n'
        'event: token\ndata: {"text": "world"}\n\n'
        'event: citations\ndata: {"citations": []}\n\n'
        'event: done\ndata: {"run_id": 7, "latency_ms": 12}\n\n'
    )

    def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = list(client_with(handler).ask_stream("q"))
    assert [e.event for e in events] == ["retrieval", "token", "token", "citations", "done"]
    assert "".join(e.data["text"] for e in events if e.event == "token") == "hello world"
    assert events[-1].data["run_id"] == 7


def test_a_401_becomes_an_auth_error():
    def handler(request):
        return httpx.Response(401, json={"detail": "not authenticated"})

    with pytest.raises(AuthError):
        client_with(handler).ask("q")


def test_a_429_becomes_a_rate_limit_error():
    def handler(request):
        return httpx.Response(429, json={"detail": "too many"})

    with pytest.raises(RateLimitError):
        client_with(handler).ask("q")


def test_a_404_becomes_a_not_found_error():
    def handler(request):
        return httpx.Response(404, json={"detail": "no such run"})

    with pytest.raises(NotFoundError):
        client_with(handler).run(99)


def test_search_returns_typed_results():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "query": "q",
                "mode": "semantic",
                "results": [{"chunk_id": 1, "document_id": 2, "score": 0.5, "text": "t"}],
            },
        )

    results = client_with(handler).search("q")
    assert results[0].score == pytest.approx(0.5)
    assert results[0].text == "t"
```

The `transport` keyword on `Client` exists so the SDK is testable without a server. Document it in the class docstring as a testing seam, not as a public feature.

- [ ] **Step 3: Run to verify it fails, then implement**

Run: `uv run pytest packages/sdk-python/tests -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ragfabric_sdk'`.

Implement `models.py` (pydantic mirrors of `AnswerResponse`, `Citation`, `SearchResultItem`, `Document`, `Run`, and an `AskEvent` with `event: str` and `data: dict`), `errors.py` (base `RagFabricError` plus the three subclasses, raised by one `_raise_for_status` helper that maps 401 and 403 to `AuthError`, 404 to `NotFoundError`, 429 to `RateLimitError` and every other 4xx or 5xx to `RagFabricError`, carrying the response body's `detail` as the message), and `client.py`. `ask_stream` parses the SSE framing with `httpx.Client.stream` and yields `AskEvent` per blank line separated block.

- [ ] **Step 4: Add the import contract**

Add a third contract to the import linter configuration forbidding `ragfabric_sdk` from importing `ragfabric_core`, `ragfabric_server` and `ragfabric_cli`, following the format of the two existing contracts.

- [ ] **Step 5: Verify and commit**

```bash
uv sync -q && uv run pytest -q -p no:warnings 2>&1 | tail -1
uv run lint-imports
```

Expected: the suite passes and `lint-imports` reports 3 contracts kept.

```bash
git add packages/sdk-python pyproject.toml uv.lock .importlinter setup.cfg .github/workflows/ci.yml
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: add the ragfabric-sdk python client with a no core import contract"
git log -1 --format='%(trailers)'
```

Add only the import linter file that actually exists in this repository; check whether the contracts live in `setup.cfg`, `.importlinter` or `pyproject.toml` before staging.

---

### Task 15: `ragfabric ask`

**Files:**
- Create: `packages/cli/src/ragfabric_cli/commands/ask.py`
- Modify: `packages/cli/src/ragfabric_cli/main.py`, `packages/cli/pyproject.toml` (depend on `ragfabric-sdk`)
- Test: additions to `packages/cli/tests/test_cli_phase3.py`

**Interfaces:**
- Produces: `ragfabric ask QUESTION [--url URL] [--token TOKEN] [--api-key KEY] [--top-k N] [--threshold F] [--collection NAME] [--no-stream] [--json]`. Streams tokens to stdout as they arrive by default; `--json` prints the whole answer payload instead and implies `--no-stream`.
- Exit codes: 0 on an answer, 1 on a connection or HTTP error with the message on stderr, 2 when neither a token nor an API key is available.

Why the CLI goes through the SDK over HTTP rather than calling core directly: `ragfabric ask` must work against a remote server, which is the normal case for an operator, and a second in process code path would be a second thing to keep correct and would behave differently from what users of the API see. The cost is that `ask` needs a running server, which the help text states.

- [ ] **Step 1: Write the failing tests**

Add to `packages/cli/tests/test_cli_phase3.py`:

```python
def test_ask_streams_tokens_to_stdout(monkeypatch):
    from ragfabric_sdk.models import AskEvent

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def ask_stream(self, query, **params):
            yield AskEvent(event="retrieval", data={"chunks": 1})
            yield AskEvent(event="token", data={"text": "twenty four "})
            yield AskEvent(event="token", data={"text": "days"})
            yield AskEvent(event="citations", data={"citations": [{"marker": "[1]"}]})
            yield AskEvent(event="done", data={"run_id": 5, "latency_ms": 9})

    monkeypatch.setattr("ragfabric_cli.commands.ask.Client", FakeClient)
    result = runner.invoke(app, ["ask", "how much leave", "--token", "t"])
    assert result.exit_code == 0, result.output
    assert "twenty four days" in result.output
    assert "run 5" in result.output


def test_ask_without_any_credential_exits_2():
    result = runner.invoke(app, ["ask", "q", "--url", "http://server"])
    assert result.exit_code == 2
    assert "token" in result.output.lower()
```

The second test assumes no `RAGFABRIC_TOKEN` or `RAGFABRIC_API_KEY` is set in the environment; the `cli_env` fixture must clear both.

- [ ] **Step 2: Implement, run, commit**

Write `commands/ask.py` reading `--url` with a default of `RAGFABRIC_URL` then `http://localhost:8000`, and credentials from `--token` or `RAGFABRIC_TOKEN`, `--api-key` or `RAGFABRIC_API_KEY`. Stream with `typer.echo(text, nl=False)` per token, print a blank line, then the citation markers with their filenames, then `run <id> in <latency>ms`. Register `app.command("ask")(ask_command)` in `main.py` and add `ragfabric-sdk` to the CLI package dependencies as a workspace source.

Run: `uv sync -q && uv run pytest packages/cli/tests -q && uv run pytest -q -p no:warnings 2>&1 | tail -1`
Expected: all pass.

```bash
git add packages/cli packages/cli/tests uv.lock
git -c user.name="Ranjan G" -c user.email="ranjan.g@ispf.ngo" commit -m "feat: add ragfabric ask on top of the python sdk"
git log -1 --format='%(trailers)'
```

---

### Task 16: Clear the deferred items from the Phase 1 and Phase 2 reviews

**Files:** touched per item below.

**Interfaces:** no new public interface except where an item names one.

This task exists because the list has grown across two phases and two of its items already cause flaky test runs. It is one task so the reviewer sees the whole cleanup as one diff, and so nothing here can hide inside a feature commit. Two of the twenty items are voided by Task 13, which deleted the code they referred to; that is recorded rather than silently dropped.

Each item gets its own commit, using the message given. Work top to bottom; the ordering puts correctness before cosmetics.

| # | Item (source) | What to do | Commit message |
|---|---|---|---|
| 1 | Per process test database and uploads paths (Phase 2) | Move the server test database and uploads directory onto `tmp_path_factory` so two pytest processes cannot share `/tmp/rag_test.db`. This is first because it is the item that makes the rest of the suite untrustworthy under parallel runs. | `test: isolate the server test database and uploads dir per process` |
| 2 | Policy query optimisation, three unscoped reads per request (Phase 2) | `compute_access_filter` performs three unscoped table reads per call. Combine them into one query per axis scoped to the principal's groups, and add a test asserting the row counts do not grow with corpus size. | `perf: scope the access policy queries to the principal` |
| 3 | Atomic grant upsert (Phase 2) | Replace the read then write grant upsert with a single dialect appropriate upsert on the unique pair, and add a concurrent grant test. | `fix: make the collection grant upsert atomic` |
| 4 | `last_used_at` written on the read path (Phase 2) | `verify_api_key` writes `last_used_at` on every authenticated read, which puts a write in every request. Batch it to at most once per minute per key, or move it behind a config flag defaulting to on with the write coalesced. Record which was chosen and why in the ledger. | `perf: stop writing api key last_used_at on every request` |
| 5 | Two transaction fan out (Phase 2) | The fan out writes vectors and the lexical index in two transactions, so a crash between them leaves a document indexed in one store only. Make it one transaction, or make the worker idempotent and reconcile on retry. State the choice and the reason. | `fix: make the index fan out atomic across both stores` |
| 6 | `access_stats` double scan (Phase 2) | `access_stats` scans candidates twice to produce the before and after counts. Produce both in one query with a conditional aggregate. | `perf: compute access stats in a single scan` |
| 7 | Ingestion run record so ingestion spans are stored (Phase 2) | Ingestion spans are only stored when a trace context is active, which today means only the query route. Add an `IngestionRun` row, written by the pipeline and the worker, carrying the spans, and expose it at `GET /api/runs/ingestion/{id}`. This needs migration 0005. | `feat: record ingestion runs so ingestion spans are stored` |
| 8 | Re-index on collection move (Phase 2) | `chunk_embeddings` and `chunk_search` denormalise `collection_id`, so moving a document between collections leaves a stale access value in the index, which is an access control correctness bug, not a cosmetic one. Update both index tables inside the same transaction as the document move, and add a test that a moved document is invisible to a principal granted only its old collection. | `fix: update denormalised collection ids when a document moves` |
| 9 | Content type pinning on download (Phase 2) | The download route returns the stored file without pinning a content type, so an uploaded HTML file can render in the browser origin. Serve a pinned type from a small extension map, default `application/octet-stream`, and add `Content-Disposition: attachment` and `X-Content-Type-Options: nosniff`. Add a test. | `fix: pin the content type and force download on the originals route` |
| 10 | Graceful worker SIGTERM (Phase 2) | The worker does not handle SIGTERM, so a container stop can kill it mid job. Install a handler that sets the stop event, finishes the current job and exits 0. | `feat: handle SIGTERM in the worker for a clean shutdown` |
| 11 | Wire or drop the `limits.*` config keys (Phase 2) | `limits.max_upload_mb`, `limits.allowed_types` and `limits.rate_limit_per_minute` are validated and then ignored. Wire all three: enforce the upload size and type in the upload route, and use `rate_limit_per_minute` as the default for a key with no explicit limit. A config key that does nothing is a broken promise in a file users edit. | `feat: enforce the limits config keys on upload and rate limiting` |
| 12 | `Cache` protocol lacks `name` (Phase 1) | Add `name: str` to the `Cache` protocol and to both implementations, matching every other store interface. | `refactor: give the Cache protocol a name attribute` |
| 13 | `serve` default host (Phase 1) | `ragfabric serve` defaults to `0.0.0.0`, which exposes a development server on every interface. Default to `127.0.0.1` and require `--host 0.0.0.0` explicitly. Note it in the changelog as a behaviour change. | `fix: bind ragfabric serve to localhost by default` |
| 14 | Naive DateTime columns (Phase 1) | Several columns are naive `DateTime` while their docstrings claim timezone awareness. Make them `DateTime(timezone=True)` with a migration, or correct the docstrings. Choose awareness, since audit and run timestamps are compared across deployments, and record the reason. | `fix: store timestamps as timezone aware datetimes` |
| 15 | Provider SDKs as extras (Phase 1) | `openai` and `anthropic` are hard dependencies even for an Ollama only deployment. Move them to extras `ragfabric[openai]` and `ragfabric[anthropic]`, import them inside their provider modules, and raise a `ProviderError` naming the extra when one is selected without being installed, exactly as Task 6 does for the reranker. | `refactor: move the openai and anthropic sdks to extras` |
| 16 | `HTTPException` singleton in `deps.py` (Phase 1) | A module level `HTTPException` instance is shared across requests. Construct it per raise. While in the file, replace the Task 10 `lru_cache` singletons with objects built once on application startup and held on `app.state`, which is the same class of problem. | `fix: stop sharing mutable singletons across requests in deps` |
| 17 | Frontend dist path and package name (Phase 1) | `apps/assistant` builds to an unexpected output directory and its package name does not match the workspace convention. Align both and update the Dockerfile and CI paths that reference them. | `chore: align the assistant build output path and package name` |
| 18 | Digest pinned base images (Phase 1) | Compose and the Dockerfiles pin image tags, not digests, so a rebuild is not reproducible. Pin every base image by digest and add a comment giving the tag each digest corresponded to and the date. | `chore: pin base images by digest` |
| 19 | Adapter test against the real `HybridRetriever` with a restrictive filter (Phase 1) | **Voided by Task 13.** `HybridRetriever` no longer exists; the equivalent coverage is `test_traditional_strategy.py::test_it_passes_the_access_filter_into_the_store_rather_than_filtering_after` plus the restrictive filter tests on both stores. Record the void in the ledger with those test names. | no commit |
| 20 | `collection_ids[0]` in the legacy adapter (Phase 1) | **Voided by Task 13.** `strategies/legacy.py` is deleted. `TraditionalRAGStrategy` inherits the same single collection limitation, so open a new tracked item for multi collection filters in Phase 4, where the console exposes collection selection. | no commit |

- [ ] **Step 1: Work the table top to bottom**

For each numbered row: write the test first where the item is a behaviour, make the change, run `uv run pytest -q -p no:warnings 2>&1 | tail -1`, and commit with the given message. Record each item's outcome in the ledger as it completes, including the two voids with their justification.

- [ ] **Step 2: Verify the whole list is accounted for**

State in the ledger, for all 20 rows, one of: the commit hash, or the recorded void. A row with neither is an incomplete task.

- [ ] **Step 3: Full verification**

```bash
uv run pytest -q -p no:warnings 2>&1 | tail -1
uv run lint-imports
uv run ruff check . && uv run ruff format --check .
docker compose --profile lite config >/dev/null && docker compose --profile full config >/dev/null && docker compose --profile workers config >/dev/null
```

Expected: everything passes and all three compose profiles are valid.

---

### Task 17: End to end verification, documentation sync, PyPI publish, pull request

**Files:**
- Modify: `README.md`, `ROADMAP.md`, `CHANGELOG.md`, `docs/README.md`, `docs/getting-started.md`, `docs/configuration.md`, `docs/architecture.md`, `docs/providers.md`, `docs/traditional-rag.md`, `docs/troubleshooting.md`
- Create: `docs/concepts/reranking.md`, `docs/concepts/embeddings.md`
- Local only: `~/AI/ragfabric/MEMORY.md`, the ledger

Steps 1 to 6 are the implementer's. Steps 7 to 9 are the controller's, after the whole branch review.

- [ ] **Step 1: Prove it works end to end, offline, from an empty database**

Run this exactly, from the worktree, and paste the real output into the ledger. Nothing in this phase is complete until this passes.

```bash
docker compose --profile lite up -d postgres redis
export DATABASE_URL="postgresql+psycopg://ragfabric:ragfabric@localhost:5432/ragfabric"
uv run ragfabric db upgrade
uv run ragfabric users create --email admin@example.com --password 'ChangeMe123!' --role admin
mkdir -p /tmp/rf-demo && printf 'Employees receive 24 days of annual leave per year. Leave accrues monthly and unused days expire in March.\n' > /tmp/rf-demo/handbook.txt
uv run ragfabric ingest /tmp/rf-demo --collection handbook
uv run ragfabric serve --host 127.0.0.1 --port 8000 &
sleep 3
uv run ragfabric ask "how much annual leave do employees get" --token "$(
  curl -s -X POST http://127.0.0.1:8000/api/auth/login \
    -H 'content-type: application/json' \
    -d '{"email":"admin@example.com","password":"ChangeMe123!"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
)"
```

Expected: the answer names 24 days, carries at least one `[n]` marker, and prints a run id. If the login response field is not `access_token`, use the real name from `routes/auth.py`. Record the answer text verbatim in the ledger, because it is the first real answer this project has produced from a real vector index and it is the evidence the phase is done.

- [ ] **Step 2: Prove the queue path and the Chroma path**

```bash
# queue mode
sed -i '' 's/^  indexing: inline/  indexing: queue/' ragfabric.yaml
docker compose --profile workers up -d worker
printf 'Overtime is paid at 1.5x after 45 hours in a week.\n' > /tmp/rf-demo/overtime.txt
uv run ragfabric ingest /tmp/rf-demo/overtime.txt --collection handbook
sleep 5
docker compose logs worker --tail 20
sed -i '' 's/^  indexing: queue/  indexing: inline/' ragfabric.yaml

# chroma
docker compose --profile full up -d chroma
sed -i '' 's/^  kind: pgvector/  kind: chroma/' ragfabric.yaml
CHROMA_URL=http://localhost:8000 uv run ragfabric reindex --yes
sed -i '' 's/^  kind: chroma/  kind: pgvector/' ragfabric.yaml
```

Expected: the worker log shows the job succeeding and the document reaching `ready`; the Chroma reindex reports the same chunk count as the corpus. Record both. On Linux use `sed -i` without the empty string argument.

- [ ] **Step 3: Documentation sync, following the one change every place rule**

| Surface | What it must say after this phase |
|---|---|
| `README.md` | "What works today" gains: real embeddings through Ollama or OpenAI; a pgvector index with an HNSW cosine index, and Chroma as an alternative; similarity threshold, metadata filters, optional reranking and a context budget; LLM written answers with numbered citations verified against a mechanical contract; `POST /api/ask` with SSE; the Python SDK; `ragfabric ask` and `ragfabric reindex`. Remove every statement that the query path uses an in memory index. State plainly that the answer's faithfulness is not yet measured and that Phase 8 measures it |
| `ROADMAP.md` | Tick every Phase 3 line. Move "Embeddings to pgvector or Chroma" out of Phase 3's open items since both shipped. Add to Phase 4: `- [ ] Multi collection metadata filters (single collection only since Phase 3)`. Add to Phase 8: `- [ ] Cross encoder reranking measured against llm and none` |
| `CHANGELOG.md` | Unreleased: Added, Changed, Removed, Fixed and Notes sections. Removed lists the in memory index, the v1 retrievers and the legacy strategy. Changed lists the default provider switch to Ollama, `ragfabric serve` binding to localhost, and the openai and anthropic SDKs becoming extras, each of which is a behaviour change an upgrader must know about |
| `docs/getting-started.md` | The real flow from Step 1, verbatim, including `ollama pull nomic-embed-text` as the first step and the note that no API key is needed |
| `docs/configuration.md` | `embeddings.dim` must match migration 0004's 768; the runbook for changing the embedding model (change config, run `ragfabric reindex`; if the dimension differs, add a migration first); `reranker.kind` and the `ragfabric[rerank]` extra; `vector_store.kind: chroma` and `CHROMA_URL`; `strategies.traditional.generation`; the openai and anthropic extras; the `limits.*` keys now being enforced |
| `docs/architecture.md` | Add `rerank/`, `tokens.py`, `embeddings/normalise.py`, `strategies/traditional.py`, `generate/{cited,contract}.py`, `ingest/reindex.py`, `stores/chroma_store.py`, `packages/sdk-python`. Remove `store/`, `retrieve/`, `strategies/legacy.py`. The request flow becomes: authenticate, compute AccessFilter, embed, vector search with the filter inside the query, threshold, rerank, budget, generate, verify citations, write run, sources and audit |
| `docs/providers.md` | pgvector and Chroma become "Phase 3 (shipped, queried)". Reranker rows: none and llm "Phase 3 (shipped)", cross_encoder "Phase 3 (shipped, extra `ragfabric[rerank]`)". Ollama becomes the default for both llm and embeddings |
| `docs/traditional-rag.md` | Status becomes shipped in v0.1.0. Document the real pipeline order and say why the threshold runs before the rerank and why the budget drops whole chunks |
| `docs/concepts/embeddings.md` | New. text to tokeniser to model to vector to store; cosine against dot product and why normalisation makes them equal; why one model per deployment (ADR 0006); what reindexing costs |
| `docs/concepts/reranking.md` | New. Why a cheap ranking plus an expensive re-ranking beats one expensive ranking; cross encoder against bi encoder; when reranking does not pay |
| `docs/troubleshooting.md` | Add: dimension mismatch on query (run `ragfabric reindex`), `nomic-embed-text` not pulled, Chroma unreachable, `sentence-transformers` missing when `cross_encoder` is selected, and an answer falling back to extractive because the citation contract failed twice |
| `docs/adr/0006-...md` | Already written in Task 3. Verify it still matches what shipped |
| Issue #4 and Discussion #13 | Controller's Step 8 |
| `MEMORY.md` | Phase 3 status, the decision log's outcomes, the new deferred list, the learning log entries |

- [ ] **Step 4: Fill the learning log**

`MEMORY.md` has a "Learning log (fill per phase)" section that is still empty after two phases. Write the Phase 3 entries: what cosine distance against a normalised vector actually computes and why the normalisation removed a real dialect divergence; why an HNSW index needs a fixed dimension; what a cross encoder does that a bi encoder cannot; what the citation contract can and cannot prove; why the access filter had to stay inside the store query when the store changed underneath it.

- [ ] **Step 5: Verify the whole branch before review**

```bash
uv run pytest -q -p no:warnings 2>&1 | tail -1
RAGFABRIC_TEST_DATABASE_URL=... uv run pytest -q -m integration 2>&1 | tail -1
RAGFABRIC_TEST_CHROMA_URL=http://localhost:8000 uv run pytest -q -m integration packages/core/tests/test_stores_chroma.py 2>&1 | tail -1
uv run ruff check . && uv run ruff format --check .
uv run lint-imports
git log --format='%H %(trailers)' origin/main..HEAD | grep -v '^[0-9a-f]* $' || echo "no trailers anywhere"
grep -rn "Claude\|Anthropic assistant\|AI assistant\|Generated with" --include="*.py" --include="*.md" --include="*.yml" . | grep -v "anthropic_provider\|docs/providers.md\|CHANGELOG" || echo "no assistant references"
grep -rn "ispf\|ISPF" --include="*.py" --include="*.md" --include="*.yml" --include="*.toml" . | grep -v "ranjan.g@ispf.ngo" || echo "no organisation specific content"
```

The last check is new this phase and matters: this is a general purpose platform that anyone will fork, so nothing outside the maintainer's own commit email may name a specific organisation. Any hit other than the author email is a finding.

- [ ] **Step 6: Push the branch**

```bash
git push -u origin feat/phase-3-traditional-rag
```

Pre authorised. Opening the PR is not.

- [ ] **Step 7 (controller, after the whole branch review): publish the SDK and the umbrella to PyPI**

This needs a PyPI token from the owner and cannot be automated. Publish `ragfabric` and `ragfabric-sdk` at `0.1.0a1`:

```bash
uv build --package ragfabric && uv build --package ragfabric-sdk
uv publish --token "$PYPI_TOKEN"
```

Verify with `uv run --with ragfabric-sdk --no-project python -c "import ragfabric_sdk; print(ragfabric_sdk.__version__)"` in a clean directory. If the token is not available, stop here and report; do not open the PR claiming the publish happened.

- [ ] **Step 8 (controller): open the pull request**

Body sections: what shipped, the decision log with outcomes, the deletion list from Task 13, the 20 deferred items with their commit hashes or voids, the end to end output from Steps 1 and 2, test counts including the integration runs, and the new deferred list for Phase 4. Include `closes #4`. End the body with the pull request attribution line the session's instructions specify, and no other attribution anywhere.

- [ ] **Step 9 (controller): after merge**

Tick issue #4, comment on Discussion #13, remove the worktree with `git worktree remove ~/AI/ragfabric-wt/phase-3`, pull `main`, copy the ledger to `MEMORY.phase-3-ledger.md` and the whole branch review to `MEMORY.phase-3-final-review.md`, update the Ledge task `ragfabric`, and record the Phase 4 starting point.

---

## Verification summary for the whole phase

Nothing in this phase is complete until all of the following are true, and each is a command whose real output goes in the ledger rather than a claim.

| Check | Command | Expected |
|---|---|---|
| Unit suite | `uv run pytest -q -p no:warnings` | All pass. The count is recorded per task, not predicted, because Task 13 deletes tests |
| PostgreSQL integration | `RAGFABRIC_TEST_DATABASE_URL=... uv run pytest -q -m integration` | All pass, including the restrictive filter assertions on both stores |
| Chroma integration | `RAGFABRIC_TEST_CHROMA_URL=... uv run pytest -q -m integration packages/core/tests/test_stores_chroma.py` | 3 pass |
| Lint and format | `uv run ruff check . && uv run ruff format --check .` | Clean |
| Import contracts | `uv run lint-imports` | 3 contracts kept |
| Migrations | `uv run pytest packages/core/tests/test_migrations.py -q` | Pass, including the drift test |
| HNSW index exists | the psql query in Task 3 Step 7 | `ix_chunk_embeddings_hnsw` present |
| Compose profiles | `docker compose --profile lite\|full\|workers config` | All three valid |
| End to end answer | Task 17 Step 1 | An answer naming 24 days, with a marker and a run id |
| Queue path | Task 17 Step 2 | Worker log shows the job, document reaches `ready` |
| Chroma path | Task 17 Step 2 | Reindex count equals the corpus chunk count |
| No trailers | `git log --format='%H %(trailers)' origin/main..HEAD` | Every line ends with an empty trailer field |
| No assistant references | the grep in Task 17 Step 5 | No hits |
| No organisation specific content | the grep in Task 17 Step 5 | No hits beyond the author email |
| Strategy contract | `assert_strategy_contract` in `test_traditional_strategy.py` | Passes for `TraditionalRAGStrategy` |

## What this phase deliberately does not do

Recorded so a reviewer does not read an omission as an oversight.

| Not done | Why | Where it lands |
|---|---|---|
| Measure whether answers are correct | Retrieval and generation metrics need a question set with known answers and a scoring harness. Shipping a number without that would be a fabricated metric (ADR 0004) | Phase 8 |
| Verify that a paraphrase is faithful to its source | Not mechanically checkable. The citation contract proves a quote is real and a marker points at retrieved evidence, and claims nothing more | Phase 8 |
| BM25, phrase and identifier boosting | `/hybrid` uses simple normalised sum fusion this phase, with a comment naming its successor | Phase 4 |
| Multi collection metadata filters | `collection_ids[0]` only, inherited from Phase 1 and now tracked as a Phase 4 item | Phase 4 |
| Serve two embedding models at once | ADR 0006. Comparing embedding models belongs to evaluation, against separate indexes | Phase 8 |
| Update the Angular app to use `/api/ask` | `/api/search/query` keeps its shape precisely so no frontend work is needed this phase | Phase 9 |
| Reranker for the other three strategies | The interface is shared and ready; the strategies do not exist yet | Phases 4 to 6 |

---

## Appendix A: reference implementations

The three modules an implementer would otherwise have to invent. Referenced from Tasks 11, 14 and 15.

### A1. `packages/server/src/ragfabric_server/api/routes/ask.py` (Task 11 Step 8)

```python
"""POST /api/ask: retrieve, stream the answer, then record the run.

Event order is retrieval, token*, citations, done. Citations come last because a
citation's `used` flag is derived from the markers the answer contains, which is
unknown until the final token. The run row is written after the stream for the
same reason: latency, tokens and citations are only final at the end, and a row
written early would leave /api/runs/{id} returning a half recorded run.

The database session is deliberately NOT held across the stream. Retrieval and
generation happen first, then one short session writes every row. A session held
for as long as a client reads would pin a connection per in flight answer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.db.session import get_db
from ragfabric_core.generate.answer import build_answer
from ragfabric_core.generate.cited import (
    NO_EVIDENCE_ANSWER,
    build_prompt,
    generate_cited_answer,
)
from ragfabric_core.generate.contract import CitationViolation, assert_citation_contract
from ragfabric_core.models.access import AuditLog
from ragfabric_core.models.document import QueryLog
from ragfabric_core.models.runs import RetrievalRun, Source
from ragfabric_core.providers.base import Message
from ragfabric_core.strategies.base import (
    RetrievalContext,
    StrategyName,
    StrategyParams,
)
from ragfabric_core.telemetry.tracing import start_trace, trace
from ragfabric_server.deps import (
    get_access_filter,
    get_embedding_model,
    get_llm_provider,
    get_principal,
    get_strategy_registry,
)
from ragfabric_server.schemas.ask import AskRequest
from ragfabric_server.schemas.search import AnswerResponse

router = APIRouter()

_SYSTEM = (
    "You answer strictly from the numbered passages provided. "
    "Cite every claim with the passage number in square brackets, for example [1]. "
    "Quote verbatim only, inside double quotation marks; paraphrase everything else. "
    f"If the passages do not answer the question, reply exactly: {NO_EVIDENCE_ANSWER}"
)


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def _context(payload: AskRequest, principal: Principal, access: AccessFilter) -> RetrievalContext:
    filters: dict[str, str | int | float | bool] = {}
    if payload.document_id is not None:
        filters["document_id"] = payload.document_id
    if payload.format is not None:
        filters["format"] = payload.format
    return RetrievalContext(
        principal=principal,
        access_filter=access,
        collection_ids=[payload.collection_id] if payload.collection_id is not None else None,
        params=StrategyParams(
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
            metadata_filters=filters,
        ),
    )


def _rows(chunks) -> list[dict]:
    return [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "collection_id": c.collection_id,
            "text": c.text,
            "page": c.page,
            "score": c.score,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "filename": c.metadata.get("filename"),
            "format": c.metadata.get("format"),
        }
        for c in chunks
    ]


def _record(
    db: Session,
    *,
    payload: AskRequest,
    principal: Principal,
    access: AccessFilter,
    result,
    answer: dict,
    spans: list[dict],
    embedding_model: str,
    llm_calls: int,
    input_tokens: int,
    output_tokens: int,
    total_ms: int,
    retrieval_ms: int,
    store,
) -> int:
    used = {c["chunk_id"] for c in answer["citations"] if c["used"]}
    before, after = store.access_stats(
        {
            "collection_id": payload.collection_id,
            "document_id": payload.document_id,
            "format": payload.format,
        },
        access,
    )
    run = RetrievalRun(
        user_id=principal.user_id,
        api_key_id=principal.api_key_id,
        question=payload.query,
        mode="manual",
        requested_strategy=payload.strategy,
        selected_strategy=payload.strategy,
        answer=answer["answer"],
        latency_ms=total_ms,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=max(total_ms - retrieval_ms, 0),
        llm_calls=llm_calls,
        retrieval_calls=result.retrieval_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=0.0,
        embedding_model=embedding_model,
        trace=spans,
    )
    db.add(run)
    db.flush()
    for rank, row in enumerate(_rows(result.chunks), start=1):
        db.add(
            Source(
                retrieval_run_id=run.id,
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                rank=rank,
                score=row["score"],
                cited=row["chunk_id"] in used,
                page=row["page"],
            )
        )
    db.add(
        AuditLog(
            principal_user_id=principal.user_id,
            api_key_id=principal.api_key_id,
            action="query",
            question=payload.query,
            strategy=payload.strategy,
            retrieval_run_id=run.id,
            sources_returned=len(result.chunks),
            sources_filtered=max(before - after, 0),
            details={"collection_id": payload.collection_id, "endpoint": "ask"},
        )
    )
    db.add(
        QueryLog(
            user_id=principal.user_id,
            collection_id=payload.collection_id,
            question=payload.query,
            confidence=answer["confidence"],
            cited_document_ids=sorted(
                {
                    c["document_id"]
                    for c in answer["citations"]
                    if c["used"] and c["document_id"] is not None
                }
            ),
        )
    )
    db.commit()
    return run.id


@router.post("/ask")
def ask(
    payload: AskRequest,
    db: Session = Depends(get_db),
    principal: Principal = Depends(get_principal),
    access: AccessFilter = Depends(get_access_filter),
    registry=Depends(get_strategy_registry),
    llm=Depends(get_llm_provider),
    embedding_model: str = Depends(get_embedding_model),
) -> StreamingResponse | AnswerResponse:
    strategy = registry.get(StrategyName(payload.strategy))
    store = getattr(strategy, "_store", None)

    if not payload.stream:
        started = time.perf_counter()
        with start_trace() as tracing:
            result = strategy.retrieve(payload.query, _context(payload, principal, access))
            retrieval_ms = int((time.perf_counter() - started) * 1000)
            with trace("answer"):
                cited = generate_cited_answer(payload.query, result.chunks, llm)
            answer = build_answer(payload.query, _rows(result.chunks), answer_text=cited.text)
        total_ms = int((time.perf_counter() - started) * 1000)
        _record(
            db,
            payload=payload,
            principal=principal,
            access=access,
            result=result,
            answer=answer,
            spans=[s.model_dump() for s in result.trace] + [s.model_dump() for s in tracing.spans],
            embedding_model=embedding_model,
            llm_calls=result.llm_calls + (0 if cited.generator == "extractive" else 1),
            input_tokens=result.input_tokens + cited.input_tokens,
            output_tokens=result.output_tokens + cited.output_tokens,
            total_ms=total_ms,
            retrieval_ms=retrieval_ms,
            store=store,
        )
        return AnswerResponse(**answer)

    def events() -> Iterator[str]:
        started = time.perf_counter()
        with start_trace() as tracing:
            result = strategy.retrieve(payload.query, _context(payload, principal, access))
            retrieval_ms = int((time.perf_counter() - started) * 1000)
            yield _event(
                "retrieval",
                {
                    "chunks": len(result.chunks),
                    "strategy": str(result.strategy),
                    "trace": [s.model_dump() for s in result.trace],
                },
            )

            if not result.chunks:
                text = NO_EVIDENCE_ANSWER
                yield _event("token", {"text": text})
                generator = "extractive"
                in_tokens = out_tokens = 0
            else:
                pieces: list[str] = []
                with trace("answer_stream"):
                    for delta in llm.stream(
                        [
                            Message(role="system", content=_SYSTEM),
                            Message(role="user", content=build_prompt(payload.query, result.chunks)),
                        ],
                        max_tokens=800,
                    ):
                        pieces.append(delta)
                        yield _event("token", {"text": delta})
                text = "".join(pieces)
                generator = "llm"
                in_tokens = out_tokens = 0
                try:
                    assert_citation_contract(text, result.chunks)
                except CitationViolation:
                    # Streaming cannot be retried in place: the client has already
                    # read the tokens. Fall back for the RECORDED answer, and tell
                    # the client its stream was superseded so it can re-render.
                    cited = generate_cited_answer(payload.query, result.chunks, llm)
                    text = cited.text
                    generator = cited.generator
                    in_tokens, out_tokens = cited.input_tokens, cited.output_tokens
                    yield _event("superseded", {"text": text, "reason": "citation contract"})

            answer = build_answer(payload.query, _rows(result.chunks), answer_text=text)
            yield _event("citations", {"citations": answer["citations"]})
            total_ms = int((time.perf_counter() - started) * 1000)
            spans = [s.model_dump() for s in result.trace] + [
                s.model_dump() for s in tracing.spans
            ]

        run_id = _record(
            db,
            payload=payload,
            principal=principal,
            access=access,
            result=result,
            answer=answer,
            spans=spans,
            embedding_model=embedding_model,
            llm_calls=result.llm_calls + (0 if generator == "extractive" else 1),
            input_tokens=result.input_tokens + in_tokens,
            output_tokens=result.output_tokens + out_tokens,
            total_ms=total_ms,
            retrieval_ms=retrieval_ms,
            store=store,
        )
        yield _event("done", {"run_id": run_id, "latency_ms": total_ms})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Two notes the implementer must not lose.

First, the `superseded` event. A streamed answer cannot be retried in place, because the client has already rendered the tokens. So the contract is still enforced, but the repair is announced rather than hidden: the client receives the corrected text and knows to replace what it drew. Silently recording a different answer from the one streamed would make the run record disagree with what the user saw, which is worse than either failing or announcing. Add `superseded` to the documented event list in `docs/traditional-rag.md` and to the SDK's event handling in Task 14.

Second, `store = getattr(strategy, "_store", None)` is a placeholder for the public accessor. Task 10 adds a `model` property to both stores; add a `store` property to `TraditionalRAGStrategy` in the same step and use `strategy.store` here. Reaching into a private attribute must not survive into the merged branch, and the reviewer checks for it.

### A2. `packages/sdk-python/src/ragfabric_sdk/client.py` (Task 14 Step 3)

```python
"""HTTP client for a RagFabric server.

Talks HTTP only and never imports ragfabric_core: an adopter installs this
next to their own code, against a server someone else runs, and a shared core
would drag the whole server dependency tree into that environment and make
version skew look like an API bug. The rule is enforced by import linter.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx

from ragfabric_sdk.errors import raise_for_status
from ragfabric_sdk.models import Answer, AskEvent, Document, Run, SearchResult

DEFAULT_TIMEOUT = 30.0


class Client:
    """A RagFabric API client.

    Args:
        base_url: e.g. "http://localhost:8000".
        token: a JWT from POST /api/auth/login.
        api_key: an rf_ key; sent as X-API-Key. Use one or the other.
        timeout: seconds.
        transport: testing seam only. Pass an httpx transport to avoid a server.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def ask(self, query: str, **params) -> Answer:
        res = self._http.post("/api/ask", json={"query": query, "stream": False, **params})
        raise_for_status(res)
        return Answer.model_validate(res.json())

    def ask_stream(self, query: str, **params) -> Iterator[AskEvent]:
        with self._http.stream(
            "POST", "/api/ask", json={"query": query, "stream": True, **params}
        ) as res:
            if res.status_code >= 400:
                res.read()
                raise_for_status(res)
            name: str | None = None
            for line in res.iter_lines():
                if line.startswith("event:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and name is not None:
                    yield AskEvent(event=name, data=json.loads(line.split(":", 1)[1].strip()))
                    name = None

    def search(self, query: str, mode: str = "semantic", **params) -> list[SearchResult]:
        path = "/api/search/hybrid" if mode == "hybrid" else "/api/search/semantic"
        res = self._http.post(path, json={"query": query, "mode": mode, **params})
        raise_for_status(res)
        return [SearchResult.model_validate(r) for r in res.json()["results"]]

    def ingest(self, path: str | Path, collection: str | None = None) -> Document:
        file_path = Path(path)
        data = {"collection": collection} if collection else None
        with file_path.open("rb") as handle:
            res = self._http.post(
                "/api/documents/upload", files={"file": (file_path.name, handle)}, data=data
            )
        raise_for_status(res)
        return Document.model_validate(res.json())

    def documents(self) -> list[Document]:
        res = self._http.get("/api/documents")
        raise_for_status(res)
        body = res.json()
        rows = body["items"] if isinstance(body, dict) and "items" in body else body
        return [Document.model_validate(r) for r in rows]

    def run(self, run_id: int) -> Run:
        res = self._http.get(f"/api/runs/{run_id}")
        raise_for_status(res)
        return Run.model_validate(res.json())
```

`errors.py` holds one mapper so every method fails the same way:

```python
"""Exception types, mapped from HTTP status codes in one place."""

from __future__ import annotations

import httpx


class RagFabricError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(RagFabricError): ...


class NotFoundError(RagFabricError): ...


class RateLimitError(RagFabricError): ...


def raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    code = response.status_code
    if code in (401, 403):
        raise AuthError(str(detail), code)
    if code == 404:
        raise NotFoundError(str(detail), code)
    if code == 429:
        raise RateLimitError(str(detail), code)
    raise RagFabricError(str(detail), code)
```

The `documents()` method tolerating both a bare list and an `{"items": [...]}` envelope is not defensive padding: check `routes/documents.py` for the real shape and delete the branch that does not apply. A client that guesses at two shapes is a client that is wrong about one of them.

### A3. `packages/cli/src/ragfabric_cli/commands/ask.py` (Task 15 Step 2)

```python
"""ragfabric ask: ask a running RagFabric server a question.

Goes through the SDK over HTTP rather than calling core in process, because the
normal case is a remote server, and a second in process path would drift from
what API users experience.
"""

from __future__ import annotations

import json
import os

import typer

from ragfabric_sdk import Client
from ragfabric_sdk.errors import RagFabricError

DEFAULT_URL = "http://localhost:8000"


def ask(
    question: str = typer.Argument(..., help="The question to ask."),
    url: str = typer.Option(None, "--url", help=f"Server URL (env RAGFABRIC_URL, {DEFAULT_URL})."),
    token: str = typer.Option(None, "--token", help="JWT (env RAGFABRIC_TOKEN)."),
    api_key: str = typer.Option(None, "--api-key", help="rf_ key (env RAGFABRIC_API_KEY)."),
    top_k: int = typer.Option(8, "--top-k", min=1, max=50),
    threshold: float = typer.Option(0.0, "--threshold", min=0.0, max=1.0),
    collection: int = typer.Option(None, "--collection", help="Collection id to search."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Wait for the whole answer."),
    as_json: bool = typer.Option(False, "--json", help="Print the payload as JSON."),
) -> None:
    """Ask a question and print the cited answer. Needs a running server."""
    url = url or os.environ.get("RAGFABRIC_URL") or DEFAULT_URL
    token = token or os.environ.get("RAGFABRIC_TOKEN")
    api_key = api_key or os.environ.get("RAGFABRIC_API_KEY")
    if not token and not api_key:
        typer.echo(
            "No credentials. Pass --token or --api-key, or set RAGFABRIC_TOKEN "
            "or RAGFABRIC_API_KEY.",
            err=True,
        )
        raise typer.Exit(2)

    params = {"top_k": top_k, "similarity_threshold": threshold}
    if collection is not None:
        params["collection_id"] = collection

    client = Client(url, token=token, api_key=api_key)
    try:
        if as_json or no_stream:
            answer = client.ask(question, **params)
            if as_json:
                typer.echo(answer.model_dump_json(indent=2))
            else:
                _print_answer(answer.answer, [c.model_dump() for c in answer.citations])
            return

        citations: list[dict] = []
        run_id = None
        latency = None
        for event in client.ask_stream(question, **params):
            if event.event == "token":
                typer.echo(event.data.get("text", ""), nl=False)
            elif event.event == "superseded":
                typer.echo("\n\n[answer corrected: citation contract]\n", err=True)
                typer.echo(event.data.get("text", ""), nl=False)
            elif event.event == "citations":
                citations = event.data.get("citations", [])
            elif event.event == "done":
                run_id = event.data.get("run_id")
                latency = event.data.get("latency_ms")
        typer.echo("")
        _print_sources(citations)
        if run_id is not None:
            typer.echo(f"run {run_id} in {latency}ms")
    except RagFabricError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:  # connection refused, DNS, timeout
        typer.echo(f"could not reach {url}: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        client.close()


def _print_answer(answer: str, citations: list[dict]) -> None:
    typer.echo(answer)
    typer.echo("")
    _print_sources(citations)


def _print_sources(citations: list[dict]) -> None:
    used = [c for c in citations if c.get("used")]
    if not used:
        return
    typer.echo("sources:")
    for citation in used:
        name = citation.get("filename") or f"document {citation.get('document_id')}"
        page = f" p{citation['page']}" if citation.get("page") else ""
        typer.echo(f"  {citation['marker']} {name}{page}")
```

`json` is imported and unused in that module as written; either use it for `--json` output instead of `model_dump_json`, or remove the import. Ruff will catch it, and the implementer must resolve it rather than adding a noqa.

---

## Appendix B: self review of this plan

Run before dispatching Task 1, and recorded here so the controller can check it rather than repeat it.

**Spec coverage against issue #4.**

| Issue #4 checklist item | Task |
|---|---|
| TraditionalRAGStrategy: embeddings to pgvector or Chroma, top_k, similarity_threshold, metadata filter, optional reranker, context budget | 5, 6, 7, 8 |
| Per request parameter overrides: top_k, similarity_threshold, chunk_size, chunk_overlap | 12 |
| Numbered citations with a tested citation contract | 9 |
| POST /api/ask manual mode with SSE streaming | 11, and `mode="manual"` is written on the run in Appendix A1 |
| ragfabric ask; Python SDK first release | 14, 15, published in 17 |
| Unit tests with the offline embedder; integration tests against the Chroma and pgvector containers | Global Constraints, 2, 5 |
| Learning notes: text to tokeniser to model to vector to store; cosine against dot product; reranking | 17, in `docs/concepts/embeddings.md`, `docs/concepts/reranking.md` and the MEMORY.md learning log |
| Roadmap extra: retire the in memory index | 13 |

No issue #4 item is unassigned.

**Known soft spots, stated rather than hidden.**

| Spot | Why it is acceptable | What the implementer must do |
|---|---|---|
| `count_tokens("hello world", "text-embedding-3-small") == 2` | The exact tiktoken count for a two word string is almost certainly 2, but it is an external library's behaviour, not ours | Assert the real value if it differs and note it in the ledger. Do not change the implementation to match a guessed number |
| Fixture names `client`, `admin_token`, `db_session`, `restricted_token`, `restricted_setup`, `cli_env`, `session_factory`, `seed_chunks` | They follow the Phase 2 test suite's conventions, but the plan was written from the source, not from the conftest files | Check each conftest before writing tests. Reuse what exists, add only `ingested_doc` and `seed_chunks` if genuinely absent, and never create a parallel fixture with a new name |
| Upload response field `chunk_count` and the login field `access_token` | Used in Task 12 and Task 17 | Read `routes/documents.py` and `routes/auth.py` and use the real names |
| `extractive_answer(query, rows)` row shape | Task 9 passes `RetrievedChunk.model_dump()` | Check the real signature in `generate/llm.py` and adapt the call, not the function |
| `ScriptedLLMProvider` response attribute name | Task 11 writes `self._responses` | Use whatever the class really calls it |
| `store.access_stats` on both vector stores | Task 13 introduces it as a ruling to confirm | Confirm against the code, implement on both stores, keep the audit count measured |

**Type consistency, checked across tasks.** `build_vector_store(cfg, session_factory, embedding_model=None)` is used with that signature in Tasks 2, 4, 5, 8 and 10. `PgVectorStore(session_factory, model=...)` and `ChromaVectorStore(client, collection_name, model, session_factory)` are constructed consistently, and both gain the public `model` property in the tasks that create them, so Appendix A1's `getattr` placeholder is replaced by `strategy.store` and `store.model`. `Reranker.name` is one of `none`, `llm`, `cross_encoder` everywhere, and `TraditionalRAGStrategy` tests that name rather than the type. `fit_to_budget(chunks, max_tokens, model)` returns `(kept, tokens_used)` in Task 7 and is unpacked that way in Task 8. `generate_cited_answer` keeps one signature across Tasks 9, 10, 11 and Appendix A1. `build_answer(query, retrieved, answer_text=None)` is called with the keyword in Tasks 10, 11 and A1. `AskEvent(event, data)` is produced by the server, parsed by the SDK and consumed by the CLI with the same two fields.

**Placeholder scan.** Tasks 12, 13, 16 and 17 contain steps that direct work without a code block. Each is either a deletion, a configuration edit, a documentation edit, or a fix whose exact code depends on a file the implementer must read first, and each names the file, the change and the reason. The three novel modules that would otherwise be invented are written in full in Appendix A. There are no `TBD`, `TODO`, "implement later" or "similar to Task N" placeholders.
