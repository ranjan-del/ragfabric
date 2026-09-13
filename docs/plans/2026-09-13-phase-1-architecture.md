# Phase 1: Architecture, Monorepo, Interfaces, Docker, Database. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the v1 single strategy assistant into the RagFabric skeleton: a uv monorepo with `packages/core`, `packages/server`, `packages/cli`, `apps/assistant`, every core interface the four strategies and later phases plug into, four LLM and embedding providers, configuration and cost modules, the platform database tables, lite and full Docker Compose profiles, and the frontend on Angular 22 with Tailwind 4, with all 100 existing tests still passing from their new homes.

**Architecture:** `ragfabric_core` owns interfaces, models, ingestion, retrieval, generation, providers, config, pricing and migrations and has no HTTP dependency. `ragfabric_server` (FastAPI) depends on core. `ragfabric_cli` is the umbrella distribution published as `ragfabric` and depends on both. Apps talk only to the HTTP API. Import direction is enforced by import-linter in CI. Nothing about retrieval quality changes in this phase; the v1 pipeline is wrapped as the first `RetrieverStrategy` implementation so the interface is proven against real code.

**Tech Stack:** Python 3.13, uv workspaces, FastAPI 0.141, Pydantic 2.13, SQLAlchemy 2.0, Alembic 1.20, psycopg 3, openai 3.x, anthropic 1.x, PyYAML, Typer, pytest 9, ruff, import-linter 2.15, PostgreSQL 18 with pgvector 0.8.6, Redis 8.8, Chroma 1.5.9, Neo4j 2026.08.1 community, Angular 22.1, TypeScript 6.0, Tailwind 4.3, Node 24 LTS in Docker and CI, OrbStack locally.

**Spec:** `docs/design/2026-09-13-ragfabric-design.md` sections 3, 4, 5, 6, 7 (data model only), 11; ADR 0001, 0002, 0003, 0004. Roadmap issue: #2.

## Global Constraints

- Python `>=3.13,<3.14` everywhere (pyproject, CI, Dockerfile `python:3.13-slim`). Local interpreter: `uv python find 3.13` (already installed at `~/.local/share/uv/python/cpython-3.13-macos-aarch64-none`).
- No em dashes in any file written in this phase. No AI assistant references anywhere. Commits authored `Ranjan G <ranjan.g@ispf.ngo>`, conventional commit prefixes, no trailers.
- No secrets in the repository. Provider keys only from environment. `.env` ignored, `.env.example` committed.
- No fabricated numbers: `pricing.yaml` entries must carry `as_of` and `source` fields filled from the provider's public pricing page on the day of implementation. If a price cannot be verified, omit the entry; the cost module reports `known=false`.
- Every store, provider and strategy is behind a `typing.Protocol` marked `@runtime_checkable`. Every `RetrieverStrategy` returns `RetrievalResult` (ADR 0002). Store query methods accept an `AccessFilter` (ADR 0003).
- Import direction (ADR 0001): `ragfabric_core` may not import `ragfabric_server` or `ragfabric_cli`; `ragfabric_server` may not import `ragfabric_cli`. Enforced by `lint-imports`.
- All 100 existing test functions (107 including parametrised cases) must pass after the move, unchanged in behaviour. Only import paths and file locations change.
- Work on branch `feat/phase-1-architecture` in a git worktree at `~/AI/ragfabric-wt/phase-1` (created with `superpowers:using-git-worktrees` at execution time). Commit after every task. Push to the feature branch is pre-authorised; opening the PR is the last task; merging is the owner's action.
- Default model ids (verified against the account on 2026-09-13): OpenAI chat `gpt-5.4-mini`, OpenAI embeddings `text-embedding-3-small`, Anthropic `claude-sonnet-5`, Ollama chat `llama3.2`, Ollama embeddings `nomic-embed-text`. All overridable in `ragfabric.yaml`.

---

## File structure after this phase

```
pyproject.toml                       uv workspace root; ruff, import-linter, pytest config
uv.lock
ragfabric.example.yaml               documented configuration file
.env.example                         secrets and connection strings (moved from backend/)
docker-compose.yml                   lite by default, `--profile full` adds chroma + neo4j
deploy/docker/api.Dockerfile
deploy/docker/ui.Dockerfile
deploy/docker/nginx.conf
packages/core/pyproject.toml         distribution ragfabric-core
packages/core/src/ragfabric_core/
    __init__.py                      __version__
    config.py                        env Settings (moved from app/core/config.py)
    config_file.py                   ragfabric.yaml models + load_config()
    security.py                      JWT + bcrypt (moved from app/core/security.py)
    pricing.py, pricing.yaml         cost estimation from configured pricing
    models/base.py, user.py, document.py       (moved)
    models/access.py                 Group, GroupMember, CollectionGrant, DocumentOverride, ApiKey, AuditLog
    models/runs.py                   Conversation, Message, RetrievalRun, Source
    models/evaluation.py             EvaluationRun, EvaluationResult
    models/graph.py                  Entity, Relationship
    db/session.py                    (moved)
    db/migrate.py                    alembic_config(), upgrade(), downgrade()
    migrations/alembic.ini, env.py, script.py.mako, versions/0001_initial_schema.py, versions/0002_platform_tables.py
    ingest/parser.py, chunk.py, embed.py, pipeline.py        (moved)
    retrieve/retriever.py, hybrid.py                          (moved)
    store/vector_store.py                                      (moved, v1 in-memory index)
    generate/answer.py, llm.py                                 (moved)
    auth/principal.py                Principal, AccessFilter
    auth/base.py                     AuthProvider protocol
    strategies/base.py               StrategyName, RetrievedChunk, TraceSpan, Budget, StrategyParams, RetrievalContext, RetrievalResult, RetrieverStrategy, StrategyRegistry
    strategies/contract.py           assert_strategy_contract()
    strategies/legacy.py             LegacyHybridStrategy (v1 pipeline behind the interface)
    providers/base.py                Message, Completion, LLMProvider, EmbeddingResult, EmbeddingProvider, ProviderError
    providers/offline.py             ScriptedLLMProvider, HashingEmbeddingProvider
    providers/openai_compat.py       OpenAIProvider, OllamaProvider, OpenAIEmbeddingProvider, OllamaEmbeddingProvider
    providers/anthropic_provider.py  AnthropicProvider
    providers/registry.py            build_llm_provider(), build_embedding_provider()
    stores/base.py                   VectorStore, LexicalStore, GraphStore, Cache protocols
    connectors/base.py               SourceDocument, Connector protocol
    testing/fixtures.py              make_pdf/docx/pptx/txt/csv (moved from tests/fixtures.py)
packages/core/tests/                 test_pipeline, test_parsers, test_migrations, test_production_safety (moved) + new tests
packages/server/pyproject.toml       distribution ragfabric-server
packages/server/src/ragfabric_server/
    main.py, deps.py                 (moved)
    api/routes/{auth,documents,search,collections,analytics,admin}.py   (moved)
    schemas/{user,document,search}.py                                   (moved)
packages/server/tests/               conftest, test_api, test_index (moved)
packages/cli/pyproject.toml          distribution ragfabric (umbrella), console script `ragfabric`
packages/cli/src/ragfabric_cli/main.py
packages/cli/tests/test_cli.py
apps/assistant/                      Angular 22 + Tailwind 4 (moved from frontend/)
.github/workflows/ci.yml             lint, backend, migrations (PG18), frontend (Node 24)
.github/dependabot.yml               uv + npm at new paths, ignore Angular/TypeScript majors
```

---

### Task 1: Worktree, uv workspace root, and the move of `backend/app` into `packages/core` and `packages/server`

**Files:**
- Create: `pyproject.toml` (root), `packages/core/pyproject.toml`, `packages/server/pyproject.toml`, `packages/core/src/ragfabric_core/__init__.py`, `packages/server/src/ragfabric_server/__init__.py`, `packages/core/src/ragfabric_core/testing/__init__.py`
- Move (git mv): everything under `backend/app`, `backend/alembic`, `backend/alembic.ini`, `backend/tests`, `backend/.env.example`
- Delete: `backend/requirements.txt`, `backend/pytest.ini`, `backend/Dockerfile`, `backend/.dockerignore` (Dockerfile is recreated in Task 12)
- Test: all moved tests

**Interfaces:**
- Produces: import roots `ragfabric_core.*` and `ragfabric_server.*`; `ragfabric_core.testing.fixtures.make_pdf/make_docx/make_pptx/make_txt/make_csv`; `ragfabric_core.db.migrate.alembic_config(db_url) -> alembic.config.Config`, `upgrade(db_url, revision="head")`, `downgrade(db_url, revision="base")`.

- [ ] **Step 1: Create the worktree and branch**

```bash
cd ~/AI/ragfabric && git fetch -q origin && git checkout -q main && git pull -q --ff-only
mkdir -p ~/AI/ragfabric-wt
git worktree add -b feat/phase-1-architecture ~/AI/ragfabric-wt/phase-1 main
cd ~/AI/ragfabric-wt/phase-1 && git status --short | wc -l   # expect 0
```

- [ ] **Step 2: Create the package skeletons and move the code with git mv**

```bash
cd ~/AI/ragfabric-wt/phase-1
mkdir -p packages/core/src/ragfabric_core packages/core/tests packages/server/src/ragfabric_server/api packages/server/tests packages/cli/src/ragfabric_cli packages/cli/tests
C=packages/core/src/ragfabric_core; S=packages/server/src/ragfabric_server
# core
git mv backend/app/core/config.py   $C/config.py
git mv backend/app/core/security.py $C/security.py
git mv backend/app/models  $C/models
git mv backend/app/db      $C/db
git mv backend/app/ingest  $C/ingest
git mv backend/app/retrieve $C/retrieve
git mv backend/app/store   $C/store
git mv backend/app/generate $C/generate
mkdir -p $C/migrations && git mv backend/alembic/env.py $C/migrations/env.py && git mv backend/alembic/script.py.mako $C/migrations/script.py.mako && git mv backend/alembic/versions $C/migrations/versions && git mv backend/alembic.ini $C/migrations/alembic.ini
mkdir -p $C/testing && git mv backend/tests/fixtures.py $C/testing/fixtures.py
# server
git mv backend/app/main.py $S/main.py
git mv backend/app/deps.py $S/deps.py
git mv backend/app/api/routes $S/api/routes
git mv backend/app/schemas $S/schemas
# tests
git mv backend/tests/test_pipeline.py backend/tests/test_parsers.py backend/tests/test_migrations.py backend/tests/test_production_safety.py packages/core/tests/
git mv backend/tests/conftest.py backend/tests/test_api.py backend/tests/test_index.py packages/server/tests/
git mv backend/.env.example .env.example
# leftovers
git rm -rq backend
find packages -name "__pycache__" -prune -exec rm -rf {} +
touch $C/__init__.py $S/__init__.py $S/api/__init__.py $C/testing/__init__.py
rm -f $C/core/__init__.py 2>/dev/null; rmdir $C/core 2>/dev/null; true
ls backend 2>/dev/null && echo "backend still exists" || echo "backend removed"
```

- [ ] **Step 3: Rewrite imports**

```bash
cd ~/AI/ragfabric-wt/phase-1
files=$(grep -rlE "from app\.|import app\." packages --include="*.py" --include="*.ini" --include="*.mako")
for f in $files; do
  sed -i '' \
    -e 's/from app\.core\.config import/from ragfabric_core.config import/g' \
    -e 's/from app\.core\.security import/from ragfabric_core.security import/g' \
    -e 's/from app\.models/from ragfabric_core.models/g' \
    -e 's/from app\.db\.session import/from ragfabric_core.db.session import/g' \
    -e 's/from app\.ingest/from ragfabric_core.ingest/g' \
    -e 's/from app\.retrieve/from ragfabric_core.retrieve/g' \
    -e 's/from app\.store/from ragfabric_core.store/g' \
    -e 's/from app\.generate/from ragfabric_core.generate/g' \
    -e 's/from app\.api\.routes import/from ragfabric_server.api.routes import/g' \
    -e 's/from app\.deps import/from ragfabric_server.deps import/g' \
    -e 's/from app\.schemas/from ragfabric_server.schemas/g' \
    -e 's/from app\.main import/from ragfabric_server.main import/g' \
    -e 's/from tests\.fixtures import/from ragfabric_core.testing.fixtures import/g' \
    "$f"
done
grep -rnE "from app\.|import app\.|from tests\." packages || echo "no stale imports"
```

Then fix the three docstrings and comments that name old paths by hand (they are not imports, so sed left them):

```bash
grep -rn "app\.core\.config\|app\.main\.lifespan\|app/core/config\|tests/fixtures.py" packages --include="*.py" --include="*.ini"
```

Edit each hit to the new name (`ragfabric_core.config`, `ragfabric_server.main.lifespan`, `ragfabric_core/config.py`, `ragfabric_core/testing/fixtures.py`).

- [ ] **Step 4: Write the root workspace pyproject**

```toml
# pyproject.toml (repository root)
[project]
name = "ragfabric-workspace"
version = "0.0.0"
description = "Workspace root for RagFabric. Not published. Install `ragfabric` instead."
requires-python = ">=3.13,<3.14"

[tool.uv]
package = false

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
ragfabric-core = { workspace = true }
ragfabric-server = { workspace = true }
ragfabric = { workspace = true }

[dependency-groups]
dev = [
  "pytest>=9.1",
  "httpx>=0.28",
  "ruff>=0.16",
  "import-linter>=2.15",
]

[tool.pytest.ini_options]
testpaths = ["packages/core/tests", "packages/server/tests", "packages/cli/tests"]
addopts = "-q"
filterwarnings = ["ignore::DeprecationWarning"]
markers = ["integration: needs a live provider or a running service; skipped by default in CI"]

[tool.ruff]
line-length = 100
target-version = "py313"
src = ["packages/core/src", "packages/server/src", "packages/cli/src"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]

[tool.importlinter]
root_packages = ["ragfabric_core", "ragfabric_server", "ragfabric_cli"]

[[tool.importlinter.contracts]]
name = "core has no upward imports"
type = "forbidden"
source_modules = ["ragfabric_core"]
forbidden_modules = ["ragfabric_server", "ragfabric_cli"]

[[tool.importlinter.contracts]]
name = "server does not import the cli"
type = "forbidden"
source_modules = ["ragfabric_server"]
forbidden_modules = ["ragfabric_cli"]
```

- [ ] **Step 5: Write the core and server package pyprojects**

```toml
# packages/core/pyproject.toml
[project]
name = "ragfabric-core"
version = "0.1.0a0"
description = "RagFabric engine: retrieval strategies, providers, stores, ingestion, evaluation, access policy."
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.13,<3.14"
dependencies = [
  "pydantic>=2.13",
  "pydantic-settings>=2.15",
  "sqlalchemy>=2.0.52",
  "alembic>=1.20",
  "psycopg[binary]>=3.3",
  "numpy>=2.3",
  "pypdf>=6.18",
  "python-docx>=1.2",
  "python-pptx>=1.0",
  "python-jose[cryptography]>=3.5",
  "bcrypt>=5.0",
  "email-validator>=2.3",
  "pyyaml>=6.0",
  "openai>=3.13",
  "anthropic>=1.5",
]

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ragfabric_core"]
```

```toml
# packages/server/pyproject.toml
[project]
name = "ragfabric-server"
version = "0.1.0a0"
description = "RagFabric HTTP API (FastAPI)."
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.13,<3.14"
dependencies = [
  "ragfabric-core",
  "fastapi>=0.141",
  "uvicorn[standard]>=0.52",
  "python-multipart>=0.0.32",
]

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ragfabric_server"]
```

Create one line READMEs so hatch does not complain:

```bash
cd ~/AI/ragfabric-wt/phase-1
printf '# ragfabric-core\n\nThe RagFabric engine. See the repository README.\n' > packages/core/README.md
printf '# ragfabric-server\n\nThe RagFabric HTTP API. See the repository README.\n' > packages/server/README.md
printf '"""RagFabric engine."""\n\n__version__ = "0.1.0a0"\n' > packages/core/src/ragfabric_core/__init__.py
printf '"""RagFabric HTTP API."""\n' > packages/server/src/ragfabric_server/__init__.py
```

- [ ] **Step 6: Move Alembic inside the package and add the migrate helper**

Edit `packages/core/src/ragfabric_core/migrations/alembic.ini`: set `script_location = %(here)s` and delete the `prepend_sys_path = .` line. Everything else stays.

Edit `packages/core/src/ragfabric_core/migrations/env.py`: the import block becomes

```python
from ragfabric_core.config import get_settings
from ragfabric_core.models.base import Base
from ragfabric_core.models import document, user  # noqa: F401
```

Create `packages/core/src/ragfabric_core/db/migrate.py`:

```python
"""Programmatic Alembic entry points.

The migration scripts ship inside the package so `pip install ragfabric` can
migrate a database without a checkout. Both the CLI (`ragfabric db upgrade`)
and the tests go through these functions, so there is one way to run them.
"""

from __future__ import annotations

from importlib.resources import files

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = files("ragfabric_core") / "migrations"


def alembic_config(db_url: str) -> Config:
    """Alembic Config pointed at the packaged scripts and the given database."""
    config = Config(str(MIGRATIONS_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", db_url)
    return config


def upgrade(db_url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(db_url), revision)


def downgrade(db_url: str, revision: str = "base") -> None:
    command.downgrade(alembic_config(db_url), revision)
```

Edit `packages/core/tests/test_migrations.py`: replace the `BACKEND_DIR` line and `_alembic_config` function with

```python
from ragfabric_core.db.migrate import alembic_config as _alembic_config
```

and delete the now unused `Config` import. The three tests keep their bodies.

- [ ] **Step 7: Install with uv on Python 3.13 and run every test**

```bash
cd ~/AI/ragfabric-wt/phase-1
uv python pin 3.13
uv sync
uv run python -c "import ragfabric_core, ragfabric_server; print('imports ok')"
uv run pytest 2>&1 | tail -5
```

Expected: `100 passed` or `107 passed` (parametrised cases counted). Fix any failure caused by a missed import path before continuing; do not change test logic.

- [ ] **Step 8: Commit**

```bash
cd ~/AI/ragfabric-wt/phase-1 && git add -A && git commit -q -m "refactor: move the v1 backend into packages/core and packages/server as a uv workspace" && git log --oneline -1
```

---

### Task 2: Lint, import direction, and CI on Python 3.13

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/dependabot.yml`

**Interfaces:**
- Produces: CI jobs named exactly `Lint (ruff, import-linter)`, `Backend tests (pytest)`, `Migrations apply cleanly`, `Frontend build and tests`. The last three names are required by the `protect-main` ruleset and must not change.

- [ ] **Step 1: Run the linters locally and fix what they report**

```bash
cd ~/AI/ragfabric-wt/phase-1
uv run ruff check packages --fix && uv run ruff format packages
uv run lint-imports
uv run pytest 2>&1 | tail -2
```

Expected: ruff clean, `Contracts: 2 kept, 0 broken.`, tests still pass.

- [ ] **Step 2: Rewrite CI**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  UV_VERSION: "0.12.13"

jobs:
  lint:
    name: Lint (ruff, import-linter)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v7
        with:
          version: ${{ env.UV_VERSION }}
          enable-cache: true
      - run: uv sync --frozen
      - run: uv run ruff check packages
      - run: uv run ruff format --check packages
      - run: uv run lint-imports

  backend:
    name: Backend tests (pytest)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v7
        with:
          version: ${{ env.UV_VERSION }}
          enable-cache: true
      - run: uv sync --frozen
      - name: Run tests
        env:
          JWT_SECRET: ci-test-secret
          DATABASE_URL: sqlite:///./ci.db
        run: uv run pytest -m "not integration"

  migrations:
    name: Migrations apply cleanly
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:0.8.6-pg18
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: ragfabric
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U postgres"
          --health-interval 5s
          --health-timeout 5s
          --health-retries 10
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v7
        with:
          version: ${{ env.UV_VERSION }}
          enable-cache: true
      - run: uv sync --frozen
      - name: upgrade head, downgrade base, upgrade head on PostgreSQL 18
        env:
          JWT_SECRET: ci-test-secret
          DATABASE_URL: postgresql+psycopg://postgres:postgres@localhost:5432/ragfabric
        run: |
          uv run ragfabric db upgrade
          uv run ragfabric db downgrade
          uv run ragfabric db upgrade

  frontend:
    name: Frontend build and tests
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: apps/assistant
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-node@v7
        with:
          node-version: "24"
          cache: npm
          cache-dependency-path: apps/assistant/package-lock.json
      - run: npm ci
      - name: Unit tests (headless Chrome)
        env:
          CHROME_BIN: /usr/bin/google-chrome
        run: npm test
      - run: npm run build
```

Note: the `migrations` job calls `ragfabric db upgrade`, which Task 11 provides, and `frontend` uses `apps/assistant`, which Task 13 provides. CI will be red on this branch until those tasks land; that is expected and is why the PR opens only at the end.

- [ ] **Step 3: Update Dependabot paths**

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: uv
    directory: /
    schedule: { interval: weekly }
    groups: { python-deps: { patterns: ["*"] } }
  - package-ecosystem: npm
    directory: /apps/assistant
    schedule: { interval: weekly }
    groups: { frontend-deps: { patterns: ["*"] } }
    ignore:
      - dependency-name: "@angular/*"
        update-types: ["version-update:semver-major"]
      - dependency-name: "@angular-devkit/*"
        update-types: ["version-update:semver-major"]
      - dependency-name: "typescript"
        update-types: ["version-update:semver-major"]
  - package-ecosystem: github-actions
    directory: /
    schedule: { interval: weekly }
```

- [ ] **Step 4: Commit**

```bash
cd ~/AI/ragfabric-wt/phase-1 && git add -A && git commit -q -m "ci: run lint, tests and migrations with uv on Python 3.13 and PostgreSQL 18" && git log --oneline -1
```

---

### Task 3: Principal, AccessFilter, and the RetrieverStrategy interface with its contract test and the v1 adapter

**Files:**
- Create: `packages/core/src/ragfabric_core/auth/__init__.py`, `auth/principal.py`
- Create: `packages/core/src/ragfabric_core/strategies/__init__.py`, `strategies/base.py`, `strategies/contract.py`, `strategies/legacy.py`
- Test: `packages/core/tests/test_strategy_interface.py`

**Interfaces:**
- Produces (exact):
  - `Principal(user_id: int | None, email: str | None, role: str, group_ids: list[int], api_key_id: int | None)`
  - `AccessFilter(document_ids: frozenset[int] | None, collection_ids: frozenset[int] | None)` with `allows(document_id, collection_id) -> bool`; `AccessFilter.unrestricted()`
  - `StrategyName` StrEnum: `TRADITIONAL="traditional"`, `VECTORLESS="vectorless"`, `AGENTIC="agentic"`, `GRAPH="graph"`
  - `TraceSpan(name, started_ms, duration_ms, attributes: dict)`
  - `RetrievedChunk(chunk_id, document_id, collection_id, text, page, section, score, char_start, char_end, metadata)`
  - `StrategyParams(top_k=5, similarity_threshold=0.0, metadata_filters={})`
  - `Budget(max_llm_calls=8, max_latency_ms=30000, max_cost_usd=0.10)`
  - `RetrievalContext(principal, access_filter, collection_ids, params, budget)`
  - `RetrievalResult(strategy, chunks, retrieval_calls, llm_calls, input_tokens, output_tokens, latency_ms, trace, fallback_from=None)`
  - `RetrieverStrategy` Protocol: attribute `name: StrategyName`, method `retrieve(query: str, ctx: RetrievalContext) -> RetrievalResult`
  - `StrategyRegistry().register(strategy)`, `.get(name) -> RetrieverStrategy`, `.names() -> list[StrategyName]`
  - `assert_strategy_contract(strategy, query, ctx) -> RetrievalResult`
  - `LegacyHybridStrategy()` with `name = StrategyName.TRADITIONAL`

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_strategy_interface.py
"""The strategy contract is the product: four implementations, one result shape."""

import pytest

from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.strategies.base import (
    Budget,
    RetrievalContext,
    RetrievalResult,
    RetrievedChunk,
    RetrieverStrategy,
    StrategyName,
    StrategyParams,
    StrategyRegistry,
    TraceSpan,
)
from ragfabric_core.strategies.contract import assert_strategy_contract


def make_ctx(**overrides) -> RetrievalContext:
    values = dict(
        principal=Principal(user_id=1, email="u@example.com", role="user", group_ids=[], api_key_id=None),
        access_filter=AccessFilter.unrestricted(),
        collection_ids=None,
        params=StrategyParams(),
        budget=Budget(),
    )
    values.update(overrides)
    return RetrievalContext(**values)


class StaticStrategy:
    """Minimal conforming implementation used to test the contract itself."""

    name = StrategyName.VECTORLESS

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        chunk = RetrievedChunk(
            chunk_id=1, document_id=1, collection_id=None, text=f"about {query}",
            page=1, section=None, score=0.9, char_start=0, char_end=10, metadata={},
        )
        return RetrievalResult(
            strategy=self.name, chunks=[chunk], retrieval_calls=1, llm_calls=0,
            input_tokens=0, output_tokens=0, latency_ms=3,
            trace=[TraceSpan(name="lexical_search", started_ms=0, duration_ms=3, attributes={"k": 1})],
        )


def test_static_strategy_satisfies_the_runtime_protocol():
    assert isinstance(StaticStrategy(), RetrieverStrategy)


def test_contract_helper_returns_the_result_for_a_conforming_strategy():
    result = assert_strategy_contract(StaticStrategy(), "leave policy", make_ctx())
    assert result.strategy == StrategyName.VECTORLESS
    assert result.chunks[0].text == "about leave policy"


def test_contract_helper_rejects_a_result_whose_strategy_name_lies():
    class Liar(StaticStrategy):
        def retrieve(self, query, ctx):
            result = super().retrieve(query, ctx)
            return result.model_copy(update={"strategy": StrategyName.GRAPH})

    with pytest.raises(AssertionError, match="strategy"):
        assert_strategy_contract(Liar(), "q", make_ctx())


def test_contract_helper_rejects_more_chunks_than_top_k():
    class TooMany(StaticStrategy):
        def retrieve(self, query, ctx):
            result = super().retrieve(query, ctx)
            return result.model_copy(update={"chunks": result.chunks * 3})

    with pytest.raises(AssertionError, match="top_k"):
        assert_strategy_contract(TooMany(), "q", make_ctx(params=StrategyParams(top_k=2)))


def test_access_filter_unrestricted_allows_everything():
    f = AccessFilter.unrestricted()
    assert f.allows(document_id=42, collection_id=None)


def test_access_filter_restricts_by_document_and_collection():
    f = AccessFilter(document_ids=frozenset({1, 2}), collection_ids=frozenset({10}))
    assert f.allows(document_id=1, collection_id=None)
    assert f.allows(document_id=99, collection_id=10)
    assert not f.allows(document_id=99, collection_id=11)


def test_registry_round_trip_and_unknown_name():
    registry = StrategyRegistry()
    registry.register(StaticStrategy())
    assert registry.names() == [StrategyName.VECTORLESS]
    assert registry.get(StrategyName.VECTORLESS).name == StrategyName.VECTORLESS
    with pytest.raises(KeyError):
        registry.get(StrategyName.GRAPH)


def test_result_counts_cannot_be_negative():
    with pytest.raises(ValueError):
        RetrievalResult(
            strategy=StrategyName.TRADITIONAL, chunks=[], retrieval_calls=-1, llm_calls=0,
            input_tokens=0, output_tokens=0, latency_ms=0, trace=[],
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `cd ~/AI/ragfabric-wt/phase-1 && uv run pytest packages/core/tests/test_strategy_interface.py -q 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'ragfabric_core.auth'`

- [ ] **Step 3: Implement principal and access filter**

```python
# packages/core/src/ragfabric_core/auth/__init__.py
"""Who is asking, and what they may see."""
```

```python
# packages/core/src/ragfabric_core/auth/principal.py
"""Principal and AccessFilter.

A Principal is whoever made the request: a signed in user or an API key. An
AccessFilter is the set of documents that principal may read, computed once per
request and handed to every store query, so ranking only ever sees permitted
chunks (ADR 0003). Phase 1 defines the types; Phase 2 computes real filters
from groups and grants. Until then every caller uses ``AccessFilter.unrestricted()``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: int | None
    email: str | None
    role: str = "user"
    group_ids: list[int] = Field(default_factory=list)
    api_key_id: int | None = None


class AccessFilter(BaseModel):
    """``None`` for a field means "no restriction on that axis".

    A chunk is allowed when its document is explicitly permitted, or its
    collection is permitted, or neither axis is restricted.
    """

    model_config = ConfigDict(frozen=True)

    document_ids: frozenset[int] | None = None
    collection_ids: frozenset[int] | None = None

    @classmethod
    def unrestricted(cls) -> AccessFilter:
        return cls(document_ids=None, collection_ids=None)

    @property
    def is_unrestricted(self) -> bool:
        return self.document_ids is None and self.collection_ids is None

    def allows(self, document_id: int | None, collection_id: int | None) -> bool:
        if self.is_unrestricted:
            return True
        if self.document_ids is not None and document_id in self.document_ids:
            return True
        if self.collection_ids is not None and collection_id in self.collection_ids:
            return True
        return False
```

- [ ] **Step 4: Implement the strategy interface**

```python
# packages/core/src/ragfabric_core/strategies/__init__.py
"""Retrieval strategies. Four implementations, one interface, one result shape."""
```

```python
# packages/core/src/ragfabric_core/strategies/base.py
"""The RetrieverStrategy interface and the common RetrievalResult (ADR 0002).

Why one result shape: the product compares four retrieval architectures. The
comparison is only fair if generation, citation checking, metrics and
evaluation run identical code over identical data. So every strategy, however
different inside, returns a RetrievalResult, and nothing downstream is allowed
to ask which strategy produced it.

Why a Protocol rather than an abstract base class: adopters can implement a
strategy in their own package without inheriting from ours. ``runtime_checkable``
lets the registry and the tests verify conformance with ``isinstance``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ragfabric_core.auth.principal import AccessFilter, Principal


class StrategyName(StrEnum):
    TRADITIONAL = "traditional"
    VECTORLESS = "vectorless"
    AGENTIC = "agentic"
    GRAPH = "graph"


class TraceSpan(BaseModel):
    """One timed step inside a strategy, for the Trace page and OTel export."""

    name: str
    started_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: int
    document_id: int
    collection_id: int | None
    text: str
    page: int | None = None
    section: str | None = None
    score: float | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class StrategyParams(BaseModel):
    """Per request tuning. Strategies read what applies to them and ignore the rest."""

    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_filters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Budget(BaseModel):
    """Hard limits a strategy must respect. Agentic RAG stops when any is hit."""

    max_llm_calls: int = Field(default=8, ge=0)
    max_latency_ms: int = Field(default=30_000, ge=0)
    max_cost_usd: float = Field(default=0.10, ge=0.0)


class RetrievalContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    principal: Principal
    access_filter: AccessFilter
    collection_ids: list[int] | None = None
    params: StrategyParams = Field(default_factory=StrategyParams)
    budget: Budget = Field(default_factory=Budget)


class RetrievalResult(BaseModel):
    strategy: StrategyName
    chunks: list[RetrievedChunk]
    retrieval_calls: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    trace: list[TraceSpan] = Field(default_factory=list)
    fallback_from: StrategyName | None = None


@runtime_checkable
class RetrieverStrategy(Protocol):
    name: StrategyName

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult: ...


class StrategyRegistry:
    """Name to implementation. The router and the API resolve strategies here."""

    def __init__(self) -> None:
        self._by_name: dict[StrategyName, RetrieverStrategy] = {}

    def register(self, strategy: RetrieverStrategy) -> None:
        if not isinstance(strategy, RetrieverStrategy):
            raise TypeError(f"{type(strategy).__name__} does not implement RetrieverStrategy")
        self._by_name[StrategyName(strategy.name)] = strategy

    def get(self, name: StrategyName | str) -> RetrieverStrategy:
        key = StrategyName(name)
        if key not in self._by_name:
            raise KeyError(f"no strategy registered under {key!r}")
        return self._by_name[key]

    def names(self) -> list[StrategyName]:
        return list(self._by_name)
```

```python
# packages/core/src/ragfabric_core/strategies/contract.py
"""Executable contract for RetrieverStrategy implementations.

Plugin authors call this from their own tests. It runs the strategy once and
asserts the invariants every downstream component relies on.
"""

from __future__ import annotations

from ragfabric_core.strategies.base import RetrievalContext, RetrievalResult, RetrieverStrategy


def assert_strategy_contract(
    strategy: RetrieverStrategy, query: str, ctx: RetrievalContext
) -> RetrievalResult:
    assert isinstance(strategy, RetrieverStrategy), "object does not implement RetrieverStrategy"
    result = strategy.retrieve(query, ctx)
    assert isinstance(result, RetrievalResult), "retrieve() must return a RetrievalResult"
    assert result.strategy == strategy.name, (
        f"result.strategy is {result.strategy!r} but the strategy is named {strategy.name!r}"
    )
    assert len(result.chunks) <= ctx.params.top_k, (
        f"returned {len(result.chunks)} chunks, more than top_k={ctx.params.top_k}"
    )
    for chunk in result.chunks:
        assert ctx.access_filter.allows(chunk.document_id, chunk.collection_id), (
            f"chunk {chunk.chunk_id} is outside the caller's access filter"
        )
        assert chunk.text.strip(), f"chunk {chunk.chunk_id} has empty text"
    assert result.llm_calls <= ctx.budget.max_llm_calls, "strategy exceeded max_llm_calls"
    return result
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest packages/core/tests/test_strategy_interface.py -q 2>&1 | tail -3`
Expected: `8 passed`

- [ ] **Step 6: Write the failing test for the v1 adapter**

Append to `packages/core/tests/test_strategy_interface.py`:

```python
def test_legacy_hybrid_strategy_meets_the_contract(monkeypatch):
    """The v1 pipeline, wrapped, is the first real strategy behind the interface."""
    from ragfabric_core.strategies.legacy import LegacyHybridStrategy

    class FakeRetriever:
        def retrieve(self, query, top_k, collection_id=None, document_id=None, format=None):
            return [
                {"chunk_id": 7, "document_id": 3, "collection_id": None, "text": "leave is 12 days",
                 "page": 2, "score": 0.42, "char_start": 5, "char_end": 21, "filename": "hr.pdf"},
            ][:top_k]

    monkeypatch.setattr("ragfabric_core.strategies.legacy.HybridRetriever", lambda: FakeRetriever())
    strategy = LegacyHybridStrategy()
    assert strategy.name == StrategyName.TRADITIONAL
    result = assert_strategy_contract(strategy, "leave", make_ctx(params=StrategyParams(top_k=3)))
    assert result.retrieval_calls == 1 and result.llm_calls == 0
    assert result.chunks[0].metadata["filename"] == "hr.pdf"
    assert result.trace[0].name == "hybrid_search"
```

Run: `uv run pytest packages/core/tests/test_strategy_interface.py -q 2>&1 | tail -3`
Expected: FAIL with `No module named 'ragfabric_core.strategies.legacy'`

- [ ] **Step 7: Implement the adapter**

First confirm the v1 result dict keys by reading `packages/core/src/ragfabric_core/retrieve/hybrid.py` `retrieve()` return value. Then:

```python
# packages/core/src/ragfabric_core/strategies/legacy.py
"""The v1 hybrid pipeline behind the RetrieverStrategy interface.

This is deliberately thin. It exists so the interface is proven against real
retrieval code from day one and so the API can route through the registry
before Phase 3 replaces the internals with real embeddings and a real store.
The access filter is applied here, after the v1 retriever, because the v1
in-memory index predates the filter; Phase 3 stores take the filter natively.
"""

from __future__ import annotations

import time

from ragfabric_core.retrieve.hybrid import HybridRetriever
from ragfabric_core.strategies.base import (
    RetrievalContext,
    RetrievalResult,
    RetrievedChunk,
    StrategyName,
    TraceSpan,
)


class LegacyHybridStrategy:
    name = StrategyName.TRADITIONAL

    def retrieve(self, query: str, ctx: RetrievalContext) -> RetrievalResult:
        started = time.perf_counter()
        collection_id = ctx.collection_ids[0] if ctx.collection_ids else None
        rows = HybridRetriever().retrieve(query, top_k=ctx.params.top_k, collection_id=collection_id)
        chunks = [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                collection_id=row.get("collection_id"),
                text=row["text"],
                page=row.get("page"),
                section=None,
                score=row.get("score"),
                char_start=row.get("char_start"),
                char_end=row.get("char_end"),
                metadata={"filename": row.get("filename")},
            )
            for row in rows
            if ctx.access_filter.allows(row["document_id"], row.get("collection_id"))
        ][: ctx.params.top_k]
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RetrievalResult(
            strategy=self.name,
            chunks=chunks,
            retrieval_calls=1,
            llm_calls=0,
            input_tokens=0,
            output_tokens=0,
            latency_ms=elapsed_ms,
            trace=[TraceSpan(name="hybrid_search", started_ms=0, duration_ms=elapsed_ms,
                             attributes={"top_k": ctx.params.top_k, "returned": len(chunks)})],
        )
```

If the v1 dict uses different key names (for example `id` instead of `chunk_id`), adapt the mapping in this file, never the v1 retriever.

- [ ] **Step 8: Run the tests, then commit**

Run: `uv run pytest packages/core/tests/test_strategy_interface.py -q 2>&1 | tail -3`
Expected: `9 passed`

```bash
cd ~/AI/ragfabric-wt/phase-1 && uv run ruff check packages --fix -q && git add -A && git commit -q -m "feat(core): add Principal, AccessFilter, the RetrieverStrategy interface, its contract test and the v1 adapter" && git log --oneline -1
```

---

### Task 4: Provider interfaces and the offline providers

**Files:**
- Create: `packages/core/src/ragfabric_core/providers/__init__.py`, `providers/base.py`, `providers/offline.py`
- Test: `packages/core/tests/test_providers_offline.py`

**Interfaces:**
- Produces (exact):
  - `Message(role: Literal["system","user","assistant"], content: str)`
  - `Completion(text, model, provider, input_tokens, output_tokens, latency_ms, finish_reason: str | None)`
  - `LLMProvider` Protocol: `name: str`, `default_model: str`, `complete(messages: list[Message], *, model: str | None = None, max_tokens: int = 1024, temperature: float = 0.0, json_schema: dict | None = None) -> Completion`
  - `EmbeddingResult(vectors: list[list[float]], model, provider, input_tokens, latency_ms)`
  - `EmbeddingProvider` Protocol: `name: str`, `model: str`, `dim: int`, `embed(texts: list[str]) -> EmbeddingResult`
  - `ProviderError(Exception)` with `.provider: str`
  - `ScriptedLLMProvider(responses: list[str], model="scripted")` returning responses in order, raising `ProviderError` when exhausted; counts tokens as whitespace words
  - `HashingEmbeddingProvider(dim: int | None = None)` wrapping the v1 `HashingEmbedder`

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_providers_offline.py
import pytest

from ragfabric_core.providers.base import (
    Completion,
    EmbeddingProvider,
    LLMProvider,
    Message,
    ProviderError,
)
from ragfabric_core.providers.offline import HashingEmbeddingProvider, ScriptedLLMProvider


def test_scripted_provider_returns_responses_in_order_and_counts_tokens():
    llm = ScriptedLLMProvider(["first answer", "second"])
    assert isinstance(llm, LLMProvider)
    one = llm.complete([Message(role="user", content="hello there world")])
    two = llm.complete([Message(role="user", content="again")])
    assert isinstance(one, Completion)
    assert (one.text, two.text) == ("first answer", "second")
    assert one.input_tokens == 3 and one.output_tokens == 2
    assert one.provider == "scripted" and one.model == "scripted"
    assert llm.calls == 2


def test_scripted_provider_raises_when_exhausted():
    llm = ScriptedLLMProvider(["only"])
    llm.complete([Message(role="user", content="x")])
    with pytest.raises(ProviderError) as info:
        llm.complete([Message(role="user", content="y")])
    assert info.value.provider == "scripted"


def test_hashing_embedding_provider_is_deterministic_and_normalised():
    emb = HashingEmbeddingProvider(dim=64)
    assert isinstance(emb, EmbeddingProvider)
    result = emb.embed(["leave policy", "leave policy", "kubernetes"])
    assert result.provider == "offline" and result.model == "hashing-64"
    assert len(result.vectors) == 3 and len(result.vectors[0]) == 64
    assert result.vectors[0] == result.vectors[1]
    norm = sum(v * v for v in result.vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-5
    assert result.input_tokens == 5


def test_hashing_embedding_provider_handles_empty_input():
    result = HashingEmbeddingProvider(dim=8).embed([])
    assert result.vectors == [] and result.input_tokens == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/core/tests/test_providers_offline.py -q 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'ragfabric_core.providers'`

- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/providers/__init__.py
"""LLM and embedding providers behind two small interfaces."""
```

```python
# packages/core/src/ragfabric_core/providers/base.py
"""Provider interfaces.

Why so small: strategies need exactly two things from the outside world, a
completion and an embedding. Everything vendor specific (retries, auth, model
names, token accounting quirks) stays inside the provider module, so a strategy
written against these protocols runs unchanged on OpenAI, Anthropic, Ollama or
the offline doubles used in tests.

Token counts are copied from the provider's response when it reports them and
are never estimated silently; a provider that cannot report tokens documents
how it counts.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ProviderError(Exception):
    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Completion(BaseModel):
    text: str
    model: str
    provider: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    finish_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    default_model: str

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion: ...


class EmbeddingResult(BaseModel):
    vectors: list[list[float]]
    model: str
    provider: str
    input_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    model: str
    dim: int

    def embed(self, texts: list[str]) -> EmbeddingResult: ...
```

```python
# packages/core/src/ragfabric_core/providers/offline.py
"""Offline providers: the test doubles that keep CI free of keys and networks.

ScriptedLLMProvider replays canned responses so agent loops and routers can be
unit tested branch by branch. HashingEmbeddingProvider wraps the v1 hashing
embedder: deterministic, no download, useful for tests and for the "no key"
first run, not for production quality retrieval.
"""

from __future__ import annotations

import time

from ragfabric_core.ingest.embed import HashingEmbedder, tokenize
from ragfabric_core.providers.base import Completion, EmbeddingResult, Message, ProviderError


class ScriptedLLMProvider:
    name = "scripted"

    def __init__(self, responses: list[str], model: str = "scripted") -> None:
        self._responses = list(responses)
        self.default_model = model
        self.calls = 0

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        if not self._responses:
            raise ProviderError(self.name, "script exhausted: no more responses")
        started = time.perf_counter()
        text = self._responses.pop(0)
        self.calls += 1
        return Completion(
            text=text,
            model=model or self.default_model,
            provider=self.name,
            input_tokens=sum(len(m.content.split()) for m in messages),
            output_tokens=len(text.split()),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason="stop",
        )


class HashingEmbeddingProvider:
    name = "offline"

    def __init__(self, dim: int | None = None) -> None:
        self._embedder = HashingEmbedder(dim=dim)
        self.dim = self._embedder.dim
        self.model = f"hashing-{self.dim}"

    def embed(self, texts: list[str]) -> EmbeddingResult:
        started = time.perf_counter()
        vectors = [row.tolist() for row in self._embedder.embed(texts)] if texts else []
        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            provider=self.name,
            input_tokens=sum(len(tokenize(t)) for t in texts),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
```

- [ ] **Step 4: Run the tests, then commit**

Run: `uv run pytest packages/core/tests/test_providers_offline.py -q 2>&1 | tail -3`
Expected: `4 passed`

```bash
cd ~/AI/ragfabric-wt/phase-1 && uv run ruff check packages --fix -q && git add -A && git commit -q -m "feat(core): add LLMProvider and EmbeddingProvider interfaces with offline test doubles" && git log --oneline -1
```

---

### Task 5: OpenAI and Ollama providers (one OpenAI compatible implementation)

**Files:**
- Create: `packages/core/src/ragfabric_core/providers/openai_compat.py`
- Test: `packages/core/tests/test_providers_openai_compat.py`

**Interfaces:**
- Consumes: `Message`, `Completion`, `EmbeddingResult`, `ProviderError` from Task 4.
- Produces (exact):
  - `OpenAICompatibleLLM(name, api_key, default_model, base_url=None, client=None)` implementing `LLMProvider`
  - `OpenAIProvider(api_key, default_model="gpt-5.4-mini", client=None)` with `name="openai"`
  - `OllamaProvider(base_url="http://localhost:11434/v1", default_model="llama3.2", client=None)` with `name="ollama"`
  - `OpenAICompatibleEmbeddings(name, api_key, model, dim, base_url=None, client=None)` implementing `EmbeddingProvider`
  - `OpenAIEmbeddingProvider(api_key, model="text-embedding-3-small", dim=1536, client=None)`, `OllamaEmbeddingProvider(base_url=..., model="nomic-embed-text", dim=768, client=None)`
  - `REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")`: temperature is not sent for these models.

Ollama exposes an OpenAI compatible API at `/v1`, so one implementation serves both vendors and any other compatible endpoint. The `client` argument accepts a prebuilt `openai.OpenAI` instance; tests pass a fake with the same two call shapes.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_providers_openai_compat.py
from types import SimpleNamespace

import pytest

from ragfabric_core.providers.base import LLMProvider, Message, ProviderError
from ragfabric_core.providers.openai_compat import (
    OllamaEmbeddingProvider,
    OllamaProvider,
    OpenAIEmbeddingProvider,
    OpenAIProvider,
)


class FakeChat:
    def __init__(self, text="hi", prompt_tokens=11, completion_tokens=2, finish="stop"):
        self.calls = []
        self._resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            model="gpt-5.4-mini-2026-03-17",
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


class FakeEmbeddings:
    def __init__(self, dim=4):
        self.calls = []
        self._dim = dim

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [SimpleNamespace(embedding=[float(i)] * self._dim, index=i) for i, _ in enumerate(kwargs["input"])]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=7), model=kwargs["model"])


def fake_client(chat=None, embeddings=None):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=chat or FakeChat()),
        embeddings=embeddings or FakeEmbeddings(),
    )


def test_openai_provider_maps_response_and_usage():
    chat = FakeChat(text="12 days", prompt_tokens=20, completion_tokens=3)
    llm = OpenAIProvider(api_key="sk-test", client=fake_client(chat=chat))
    assert isinstance(llm, LLMProvider)
    out = llm.complete([Message(role="system", content="answer briefly"), Message(role="user", content="leave?")])
    assert out.text == "12 days"
    assert (out.input_tokens, out.output_tokens) == (20, 3)
    assert out.provider == "openai" and out.model == "gpt-5.4-mini-2026-03-17"
    assert out.finish_reason == "stop"
    sent = chat.calls[0]
    assert sent["model"] == "gpt-5.4-mini"
    assert sent["messages"][0] == {"role": "system", "content": "answer briefly"}
    assert sent["max_completion_tokens"] == 1024


def test_reasoning_models_do_not_receive_temperature_but_others_do():
    chat = FakeChat()
    OpenAIProvider(api_key="k", client=fake_client(chat=chat)).complete([Message(role="user", content="x")], model="gpt-5.4-mini")
    assert "temperature" not in chat.calls[0]
    OpenAIProvider(api_key="k", client=fake_client(chat=chat)).complete([Message(role="user", content="x")], model="gpt-4.1-mini", temperature=0.2)
    assert chat.calls[1]["temperature"] == 0.2


def test_json_schema_is_passed_as_response_format():
    chat = FakeChat(text='{"a": 1}')
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    OpenAIProvider(api_key="k", client=fake_client(chat=chat)).complete([Message(role="user", content="x")], json_schema=schema)
    rf = chat.calls[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == schema and rf["json_schema"]["strict"] is True


def test_ollama_provider_uses_local_defaults_and_sends_temperature():
    chat = FakeChat()
    llm = OllamaProvider(client=fake_client(chat=chat))
    assert llm.name == "ollama" and llm.default_model == "llama3.2"
    llm.complete([Message(role="user", content="x")])
    assert chat.calls[0]["model"] == "llama3.2" and chat.calls[0]["temperature"] == 0.0


def test_provider_errors_are_wrapped():
    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("connection refused")

    with pytest.raises(ProviderError, match="connection refused"):
        OpenAIProvider(api_key="k", client=fake_client(chat=Boom())).complete([Message(role="user", content="x")])


def test_empty_choices_is_a_provider_error():
    chat = FakeChat()
    chat._resp.choices = []
    with pytest.raises(ProviderError, match="no choices"):
        OpenAIProvider(api_key="k", client=fake_client(chat=chat)).complete([Message(role="user", content="x")])


def test_openai_embeddings_map_vectors_in_input_order():
    emb = OpenAIEmbeddingProvider(api_key="k", client=fake_client(embeddings=FakeEmbeddings(dim=4)))
    result = emb.embed(["a", "b"])
    assert result.provider == "openai" and result.model == "text-embedding-3-small"
    assert result.vectors == [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    assert result.input_tokens == 7 and emb.dim == 1536


def test_ollama_embeddings_defaults():
    emb = OllamaEmbeddingProvider(client=fake_client(embeddings=FakeEmbeddings(dim=3)))
    assert (emb.name, emb.model, emb.dim) == ("ollama", "nomic-embed-text", 768)
    assert emb.embed([]).vectors == []


def test_missing_api_key_is_rejected_at_construction():
    with pytest.raises(ProviderError, match="api_key"):
        OpenAIProvider(api_key="")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/core/tests/test_providers_openai_compat.py -q 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'ragfabric_core.providers.openai_compat'`

- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/providers/openai_compat.py
"""OpenAI and OpenAI compatible providers (Ollama, Azure, vLLM, LM Studio).

One class, several vendors: Ollama serves the OpenAI chat and embeddings API at
``/v1``, so the same code talks to a hosted model or a local one. Vendor
differences live in small subclasses that only set defaults.

Token accounting comes from ``usage`` on the response; both OpenAI and Ollama
populate it. Reasoning model families reject a ``temperature`` argument, so it
is only sent for models outside REASONING_MODEL_PREFIXES.

``max_completion_tokens`` is used rather than the deprecated ``max_tokens``.
"""

from __future__ import annotations

import time
from typing import Any

from ragfabric_core.providers.base import Completion, EmbeddingResult, Message, ProviderError

REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _build_client(api_key: str, base_url: str | None) -> Any:
    from openai import OpenAI  # imported lazily so the module loads without the SDK

    return OpenAI(api_key=api_key, base_url=base_url)


class OpenAICompatibleLLM:
    def __init__(
        self,
        name: str,
        api_key: str,
        default_model: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(name, "api_key is required (set the provider's key in the environment)")
        self.name = name
        self.default_model = default_model
        self._client = client or _build_client(api_key, base_url)

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        chosen = model or self.default_model
        kwargs: dict[str, Any] = {
            "model": chosen,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": max_tokens,
        }
        if not chosen.startswith(REASONING_MODEL_PREFIXES):
            kwargs["temperature"] = temperature
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # the SDK raises many types; callers get one
            raise ProviderError(self.name, str(exc)) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not response.choices:
            raise ProviderError(self.name, "no choices in response")
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return Completion(
            text=(choice.message.content or ""),
            model=getattr(response, "model", None) or chosen,
            provider=self.name,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency_ms,
            finish_reason=getattr(choice, "finish_reason", None),
        )


class OpenAIProvider(OpenAICompatibleLLM):
    def __init__(self, api_key: str, default_model: str = "gpt-5.4-mini", client: Any | None = None) -> None:
        super().__init__("openai", api_key, default_model, base_url=None, client=client)


class OllamaProvider(OpenAICompatibleLLM):
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        default_model: str = "llama3.2",
        client: Any | None = None,
    ) -> None:
        # Ollama ignores the key but the SDK requires a non empty string.
        super().__init__("ollama", "ollama", default_model, base_url=base_url, client=client)


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        dim: int,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(name, "api_key is required (set the provider's key in the environment)")
        self.name = name
        self.model = model
        self.dim = dim
        self._client = client or _build_client(api_key, base_url)

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], model=self.model, provider=self.name, input_tokens=0, latency_ms=0)
        started = time.perf_counter()
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        usage = getattr(response, "usage", None)
        return EmbeddingResult(
            vectors=[list(map(float, item.embedding)) for item in ordered],
            model=getattr(response, "model", None) or self.model,
            provider=self.name,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class OpenAIEmbeddingProvider(OpenAICompatibleEmbeddings):
    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small", dim: int = 1536, client: Any | None = None
    ) -> None:
        super().__init__("openai", api_key, model, dim, base_url=None, client=client)


class OllamaEmbeddingProvider(OpenAICompatibleEmbeddings):
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "nomic-embed-text",
        dim: int = 768,
        client: Any | None = None,
    ) -> None:
        super().__init__("ollama", "ollama", model, dim, base_url=base_url, client=client)
```

- [ ] **Step 4: Run tests, add one integration test, commit**

Run: `uv run pytest packages/core/tests/test_providers_openai_compat.py -q 2>&1 | tail -3`
Expected: `9 passed`

Append to the same test file an integration test that runs only with a real key:

```python
import os


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
def test_openai_live_roundtrip():
    llm = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
    out = llm.complete([Message(role="user", content="Reply with the single word: pong")], max_tokens=16)
    assert "pong" in out.text.lower()
    assert out.input_tokens > 0 and out.output_tokens > 0
```

Run it once locally to prove the real path works, without printing the key:

```bash
cd ~/AI/ragfabric-wt/phase-1 && source ~/.claude/lib/keyring.sh && OPENAI_API_KEY="$(kr_lookup openai type api-key)" uv run pytest packages/core/tests/test_providers_openai_compat.py -m integration -q 2>&1 | tail -2
```

Expected: `1 passed`. Then:

```bash
uv run ruff check packages --fix -q && git add -A && git commit -q -m "feat(core): add OpenAI and Ollama providers over the OpenAI compatible API" && git log --oneline -1
```

---

### Task 6: Anthropic provider

**Files:**
- Create: `packages/core/src/ragfabric_core/providers/anthropic_provider.py`
- Test: `packages/core/tests/test_providers_anthropic.py`

**Interfaces:**
- Produces: `AnthropicProvider(api_key, default_model="claude-sonnet-5", client=None)` with `name="anthropic"`, implementing `LLMProvider`.

Before writing this file the implementer loads the `claude-api` skill and confirms the current Sonnet model id and the `messages.create` parameter names. The default id above was verified from the environment on 2026-09-13.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_providers_anthropic.py
from types import SimpleNamespace

import pytest

from ragfabric_core.providers.anthropic_provider import AnthropicProvider
from ragfabric_core.providers.base import LLMProvider, Message, ProviderError


class FakeMessages:
    def __init__(self, text="ok", in_tok=9, out_tok=1, stop="end_turn"):
        self.calls = []
        self._resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
            model="claude-sonnet-5",
            stop_reason=stop,
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._resp


def client_with(messages):
    return SimpleNamespace(messages=messages)


def test_system_messages_become_the_system_parameter():
    fm = FakeMessages(text="12 days", in_tok=30, out_tok=3)
    llm = AnthropicProvider(api_key="k", client=client_with(fm))
    assert isinstance(llm, LLMProvider) and llm.name == "anthropic"
    out = llm.complete([
        Message(role="system", content="be brief"),
        Message(role="user", content="leave?"),
        Message(role="assistant", content="asking"),
    ], max_tokens=64, temperature=0.3)
    sent = fm.calls[0]
    assert sent["system"] == "be brief"
    assert sent["messages"] == [{"role": "user", "content": "leave?"}, {"role": "assistant", "content": "asking"}]
    assert sent["model"] == "claude-sonnet-5" and sent["max_tokens"] == 64 and sent["temperature"] == 0.3
    assert out.text == "12 days" and (out.input_tokens, out.output_tokens) == (30, 3)
    assert out.finish_reason == "end_turn" and out.provider == "anthropic"


def test_no_system_message_means_no_system_parameter():
    fm = FakeMessages()
    AnthropicProvider(api_key="k", client=client_with(fm)).complete([Message(role="user", content="x")])
    assert "system" not in fm.calls[0]


def test_json_schema_adds_a_json_only_instruction_to_system():
    fm = FakeMessages(text='{"a": 1}')
    AnthropicProvider(api_key="k", client=client_with(fm)).complete(
        [Message(role="user", content="x")], json_schema={"type": "object"}
    )
    assert "JSON" in fm.calls[0]["system"] and '"type": "object"' in fm.calls[0]["system"]


def test_text_blocks_are_joined_and_errors_wrapped():
    fm = FakeMessages()
    fm._resp.content = [SimpleNamespace(type="text", text="a"), SimpleNamespace(type="tool_use"), SimpleNamespace(type="text", text="b")]
    assert AnthropicProvider(api_key="k", client=client_with(fm)).complete([Message(role="user", content="x")]).text == "ab"

    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("overloaded")

    with pytest.raises(ProviderError, match="overloaded"):
        AnthropicProvider(api_key="k", client=client_with(Boom())).complete([Message(role="user", content="x")])


def test_missing_key_rejected():
    with pytest.raises(ProviderError, match="api_key"):
        AnthropicProvider(api_key="")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/core/tests/test_providers_anthropic.py -q 2>&1 | tail -3`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/providers/anthropic_provider.py
"""Anthropic provider.

Differences from the OpenAI shape that this module absorbs so strategies do not
have to: the system prompt is a separate parameter, ``max_tokens`` is required,
the response is a list of content blocks, and usage is reported as
``input_tokens`` and ``output_tokens``. Structured output in Phase 1 is a
system instruction to answer with JSON matching the schema; Phase 5 upgrades
this to tool use when the agent needs guaranteed shapes.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ragfabric_core.providers.base import Completion, Message, ProviderError


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, default_model: str = "claude-sonnet-5", client: Any | None = None) -> None:
        if not api_key:
            raise ProviderError(self.name, "api_key is required (set ANTHROPIC_API_KEY)")
        self.default_model = default_model
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
        self._client = client

    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict | None = None,
    ) -> Completion:
        system_parts = [m.content for m in messages if m.role == "system"]
        if json_schema is not None:
            system_parts.append(
                "Respond with JSON only, no prose, matching this JSON schema exactly:\n"
                + json.dumps(json_schema, indent=2)
            )
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages if m.role != "system"],
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, str(exc)) from exc
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=getattr(response, "model", None) or kwargs["model"],
            provider=self.name,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
            finish_reason=getattr(response, "stop_reason", None),
        )
```

- [ ] **Step 4: Run the tests, then commit**

Run: `uv run pytest packages/core/tests/test_providers_anthropic.py -q 2>&1 | tail -3`
Expected: `5 passed`

```bash
uv run ruff check packages --fix -q && git add -A && git commit -q -m "feat(core): add the Anthropic provider" && git log --oneline -1
```

---

### Task 7: Pricing configuration and cost estimation

**Files:**
- Create: `packages/core/src/ragfabric_core/pricing.py`, `packages/core/src/ragfabric_core/pricing.yaml`
- Modify: `packages/core/pyproject.toml` (include the yaml in the wheel; hatch includes non Python files under the package by default, verify with `uv build --package ragfabric-core` and `unzip -l`)
- Test: `packages/core/tests/test_pricing.py`

**Interfaces:**
- Produces (exact):
  - `ModelPrice(input: float | None, output: float | None, embedding: float | None, as_of: date, source: str, note: str | None)` prices per one million tokens in USD
  - `PricingTable(models: dict[str, ModelPrice], currency="USD")` with `PricingTable.load(path: Path | None = None)` (None loads the packaged file) and `.lookup(provider, model) -> ModelPrice | None` (exact `provider/model`, then longest prefix match on `provider/model-prefix*`, then `provider/*`)
  - `CostEstimate(provider, model, input_tokens, output_tokens, embedding_tokens, usd: float | None, known: bool, note: str, as_of: date | None, source: str | None)` with `note` always `"estimate from configured pricing"`
  - `estimate_cost(table, provider, model, input_tokens, output_tokens=0, embedding_tokens=0) -> CostEstimate`

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_pricing.py
from datetime import date
from pathlib import Path

from ragfabric_core.pricing import CostEstimate, PricingTable, estimate_cost

SAMPLE = """
currency: USD
unit: per_million_tokens
models:
  openai/test-chat:
    input: 1.00
    output: 4.00
    as_of: 2026-09-13
    source: https://example.test/pricing
  openai/test-embed:
    embedding: 0.02
    as_of: 2026-09-13
    source: https://example.test/pricing
  openai/test-chat-2026*:
    input: 2.00
    output: 8.00
    as_of: 2026-09-13
    source: https://example.test/pricing
  ollama/*:
    input: 0
    output: 0
    embedding: 0
    as_of: 2026-09-13
    source: local inference has no per token price
    note: local
"""


def table(tmp_path: Path) -> PricingTable:
    p = tmp_path / "pricing.yaml"
    p.write_text(SAMPLE)
    return PricingTable.load(p)


def test_exact_lookup_and_cost_arithmetic(tmp_path):
    t = table(tmp_path)
    est = estimate_cost(t, "openai", "test-chat", input_tokens=1_000_000, output_tokens=500_000)
    assert isinstance(est, CostEstimate)
    assert est.known is True
    assert est.usd == 1.00 + 2.00
    assert est.note == "estimate from configured pricing"
    assert est.as_of == date(2026, 9, 13) and est.source == "https://example.test/pricing"


def test_embedding_cost(tmp_path):
    est = estimate_cost(table(tmp_path), "openai", "test-embed", input_tokens=0, embedding_tokens=2_000_000)
    assert est.usd == 0.04


def test_dated_snapshot_ids_match_by_prefix_wildcard(tmp_path):
    est = estimate_cost(table(tmp_path), "openai", "test-chat-2026-03-17", input_tokens=1_000_000)
    assert est.usd == 2.00


def test_provider_wildcard_matches_any_local_model(tmp_path):
    est = estimate_cost(table(tmp_path), "ollama", "llama3.2", input_tokens=123456, output_tokens=999)
    assert est.usd == 0.0 and est.known is True


def test_unknown_model_is_reported_not_guessed(tmp_path):
    est = estimate_cost(table(tmp_path), "anthropic", "mystery", input_tokens=10)
    assert est.usd is None and est.known is False


def test_packaged_pricing_file_loads_and_every_entry_is_dated():
    t = PricingTable.load()
    assert t.currency == "USD"
    assert "openai/text-embedding-3-small" in t.models
    for key, price in t.models.items():
        assert price.as_of is not None and price.source, f"{key} lacks as_of or source"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/core/tests/test_pricing.py -q 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'ragfabric_core.pricing'`

- [ ] **Step 3: Implement the module**

```python
# packages/core/src/ragfabric_core/pricing.py
"""Cost estimation from configured pricing (ADR 0004).

Prices are data, not code. They live in pricing.yaml with a date and a source
so anyone can see how old a number is and where it came from. A model that is
not in the table yields ``known=False`` and ``usd=None``; the system never
guesses a price. Every estimate carries the same note so UIs cannot present
it as a measured bill.

Lookup order for ``provider/model``: exact key, then the longest matching
``provider/prefix*`` key, then ``provider/*``.
"""

from __future__ import annotations

from datetime import date
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ESTIMATE_NOTE = "estimate from configured pricing"
PER_TOKENS = 1_000_000


class ModelPrice(BaseModel):
    input: float | None = Field(default=None, ge=0)
    output: float | None = Field(default=None, ge=0)
    embedding: float | None = Field(default=None, ge=0)
    as_of: date
    source: str
    note: str | None = None


class PricingTable(BaseModel):
    currency: str = "USD"
    unit: str = "per_million_tokens"
    models: dict[str, ModelPrice]

    @classmethod
    def load(cls, path: Path | None = None) -> PricingTable:
        source = Path(path) if path else files("ragfabric_core") / "pricing.yaml"
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def lookup(self, provider: str, model: str) -> ModelPrice | None:
        exact = f"{provider}/{model}"
        if exact in self.models:
            return self.models[exact]
        best: tuple[int, ModelPrice] | None = None
        for key, price in self.models.items():
            if not key.startswith(f"{provider}/") or not key.endswith("*"):
                continue
            prefix = key[len(provider) + 1 : -1]
            if prefix and model.startswith(prefix) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), price)
        if best is not None:
            return best[1]
        return self.models.get(f"{provider}/*")


class CostEstimate(BaseModel):
    provider: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    embedding_tokens: int = Field(ge=0)
    usd: float | None
    known: bool
    note: str = ESTIMATE_NOTE
    as_of: date | None = None
    source: str | None = None


def estimate_cost(
    table: PricingTable,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
) -> CostEstimate:
    price = table.lookup(provider, model)
    base = dict(provider=provider, model=model, input_tokens=input_tokens,
                output_tokens=output_tokens, embedding_tokens=embedding_tokens)
    if price is None:
        return CostEstimate(usd=None, known=False, **base)
    usd = 0.0
    usd += (price.input or 0.0) * input_tokens / PER_TOKENS
    usd += (price.output or 0.0) * output_tokens / PER_TOKENS
    usd += (price.embedding or 0.0) * embedding_tokens / PER_TOKENS
    return CostEstimate(usd=round(usd, 8), known=True, as_of=price.as_of, source=price.source, **base)
```

- [ ] **Step 4: Fill the packaged pricing file from the public pricing pages**

Fetch the current prices on the day of implementation (WebFetch on `https://openai.com/api/pricing/` and `https://www.anthropic.com/pricing`, or the provider docs pages they redirect to) and write `packages/core/src/ragfabric_core/pricing.yaml` in this shape, replacing every `<verified>` with the number read from the page and `<date>` with that day. An entry whose price could not be read is left out entirely.

```yaml
# Prices in USD per one million tokens. Every entry carries the date it was read
# and the page it was read from. Cost shown anywhere in RagFabric is an estimate
# from this file. Update it when providers change prices; nothing is hard coded.
currency: USD
unit: per_million_tokens
models:
  openai/gpt-5.4-mini*:
    input: <verified>
    output: <verified>
    as_of: <date>
    source: https://openai.com/api/pricing/
  openai/gpt-5.4*:
    input: <verified>
    output: <verified>
    as_of: <date>
    source: https://openai.com/api/pricing/
  openai/gpt-4.1-mini*:
    input: <verified>
    output: <verified>
    as_of: <date>
    source: https://openai.com/api/pricing/
  openai/text-embedding-3-small:
    embedding: <verified>
    as_of: <date>
    source: https://openai.com/api/pricing/
  openai/text-embedding-3-large:
    embedding: <verified>
    as_of: <date>
    source: https://openai.com/api/pricing/
  anthropic/claude-sonnet-5*:
    input: <verified>
    output: <verified>
    as_of: <date>
    source: https://www.anthropic.com/pricing
  anthropic/claude-haiku-4-5*:
    input: <verified>
    output: <verified>
    as_of: <date>
    source: https://www.anthropic.com/pricing
  ollama/*:
    input: 0
    output: 0
    embedding: 0
    as_of: <date>
    source: local inference has no per token price; hardware cost is not modelled
    note: local
  offline/*:
    input: 0
    output: 0
    embedding: 0
    as_of: <date>
    source: offline test doubles
    note: test double
```

- [ ] **Step 5: Run the tests, verify the yaml ships in the wheel, commit**

Run: `uv run pytest packages/core/tests/test_pricing.py -q 2>&1 | tail -3`
Expected: `6 passed`

```bash
cd ~/AI/ragfabric-wt/phase-1 && uv build --package ragfabric-core -q && unzip -l dist/ragfabric_core-*.whl | grep -c "pricing.yaml" && rm -rf dist
```

Expected: `1`. Then:

```bash
uv run ruff check packages --fix -q && git add -A && git commit -q -m "feat(core): add pricing configuration and cost estimation" && git log --oneline -1
```

---

### Task 8: `ragfabric.yaml` configuration and the provider registry

**Files:**
- Create: `packages/core/src/ragfabric_core/config_file.py`, `packages/core/src/ragfabric_core/providers/registry.py`, `ragfabric.example.yaml` (repo root)
- Test: `packages/core/tests/test_config_file.py`, `packages/core/tests/test_provider_registry.py`

**Interfaces:**
- Produces (exact):
  - `LLMConfig(provider: Literal["openai","anthropic","ollama","offline"]="openai", model: str | None=None, base_url: str | None=None)`
  - `EmbeddingsConfig(provider: Literal["openai","ollama","offline"]="openai", model: str | None=None, dim: int | None=None, base_url: str | None=None)`
  - `RerankerConfig(kind: Literal["none","llm","cross_encoder"]="none")`
  - `VectorStoreConfig(kind: Literal["pgvector","chroma","memory"]="pgvector")`, `LexicalStoreConfig(kind: Literal["postgres_fts","bm25"]="postgres_fts")`, `GraphStoreConfig(kind: Literal["neo4j","none"]="neo4j", enabled: bool=False)`, `CacheConfig(kind: Literal["redis","memory"]="redis")`
  - `IngestionConfig(chunk_size=600, chunk_overlap=80, retain_originals=True)`
  - `StrategiesConfig(traditional: dict, vectorless: dict, agentic: dict, graph: dict)` with the defaults from docs/configuration.md
  - `RouterConfig(mode: Literal["auto","manual"]="auto", min_confidence=0.6, classifier_model: str | None=None)`
  - `LimitsConfig(max_upload_mb=50, allowed_types=[...], rate_limit_per_minute=60)`, `TelemetryConfig(otlp_endpoint: str | None=None)`
  - `RagFabricConfig(llm, embeddings, reranker, vector_store, lexical_store, graph_store, cache, ingestion, strategies, router, limits, telemetry)`
  - `load_config(path: Path | None = None) -> RagFabricConfig`: explicit path, else `RAGFABRIC_CONFIG`, else `./ragfabric.yaml`, else defaults. Unknown keys are an error.
  - `build_llm_provider(cfg: LLMConfig, env: Mapping[str, str] | None = None) -> LLMProvider`, `build_embedding_provider(cfg: EmbeddingsConfig, env=None) -> EmbeddingProvider`. Keys read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`; Ollama needs none; offline needs none.

- [ ] **Step 1: Write the failing tests**

```python
# packages/core/tests/test_config_file.py
import pytest
from pydantic import ValidationError

from ragfabric_core.config_file import RagFabricConfig, load_config


def test_defaults_when_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    cfg = load_config()
    assert isinstance(cfg, RagFabricConfig)
    assert cfg.llm.provider == "openai" and cfg.vector_store.kind == "pgvector"
    assert cfg.graph_store.enabled is False and cfg.ingestion.chunk_size == 600
    assert cfg.strategies.traditional["top_k"] == 8 and cfg.router.mode == "auto"


def test_explicit_path_and_partial_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  provider: ollama\n  model: qwen3:8b\nvector_store:\n  kind: chroma\n")
    cfg = load_config(p)
    assert cfg.llm.provider == "ollama" and cfg.llm.model == "qwen3:8b"
    assert cfg.vector_store.kind == "chroma" and cfg.embeddings.provider == "openai"


def test_env_var_locates_the_file(tmp_path, monkeypatch):
    p = tmp_path / "x.yaml"
    p.write_text("router:\n  mode: manual\n")
    monkeypatch.setenv("RAGFABRIC_CONFIG", str(p))
    assert load_config().router.mode == "manual"


def test_cwd_ragfabric_yaml_is_found(tmp_path, monkeypatch):
    (tmp_path / "ragfabric.yaml").write_text("cache:\n  kind: memory\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFABRIC_CONFIG", raising=False)
    assert load_config().cache.kind == "memory"


def test_unknown_keys_and_bad_values_fail_loudly(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("llm:\n  provider: openai\n  modle: typo\n")
    with pytest.raises(ValidationError):
        load_config(p)
    p.write_text("vector_store:\n  kind: pinecone\n")
    with pytest.raises(ValidationError):
        load_config(p)


def test_example_file_in_repo_root_is_valid():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    cfg = load_config(root / "ragfabric.example.yaml")
    assert cfg.llm.provider in {"openai", "anthropic", "ollama", "offline"}
```

```python
# packages/core/tests/test_provider_registry.py
import pytest

from ragfabric_core.config_file import EmbeddingsConfig, LLMConfig
from ragfabric_core.providers.base import ProviderError
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider


def test_offline_providers_need_no_keys():
    llm = build_llm_provider(LLMConfig(provider="offline"), env={})
    emb = build_embedding_provider(EmbeddingsConfig(provider="offline", dim=32), env={})
    assert llm.name == "scripted" and emb.name == "offline" and emb.dim == 32


def test_openai_requires_key_from_env():
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        build_llm_provider(LLMConfig(provider="openai"), env={})
    llm = build_llm_provider(LLMConfig(provider="openai", model="gpt-4.1-mini"), env={"OPENAI_API_KEY": "sk-x"})
    assert llm.name == "openai" and llm.default_model == "gpt-4.1-mini"


def test_anthropic_requires_key_and_uses_default_model():
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        build_llm_provider(LLMConfig(provider="anthropic"), env={})
    llm = build_llm_provider(LLMConfig(provider="anthropic"), env={"ANTHROPIC_API_KEY": "k"})
    assert llm.default_model == "claude-sonnet-5"


def test_ollama_uses_base_url_and_no_key():
    llm = build_llm_provider(LLMConfig(provider="ollama", base_url="http://ollama:11434/v1"), env={})
    assert llm.name == "ollama"
    emb = build_embedding_provider(EmbeddingsConfig(provider="ollama"), env={})
    assert (emb.model, emb.dim) == ("nomic-embed-text", 768)


def test_openai_embeddings_default_dim_matches_model():
    emb = build_embedding_provider(EmbeddingsConfig(provider="openai"), env={"OPENAI_API_KEY": "sk"})
    assert (emb.model, emb.dim) == ("text-embedding-3-small", 1536)
    large = build_embedding_provider(EmbeddingsConfig(provider="openai", model="text-embedding-3-large"), env={"OPENAI_API_KEY": "sk"})
    assert large.dim == 3072
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/core/tests/test_config_file.py packages/core/tests/test_provider_registry.py -q 2>&1 | tail -3`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement the config models and loader**

```python
# packages/core/src/ragfabric_core/config_file.py
"""ragfabric.yaml: everything that is not a secret.

Secrets stay in the environment (see config.py Settings). This file chooses
implementations and tunes them. Unknown keys are errors, because a typo that
silently falls back to a default is the worst kind of configuration bug.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ENV_VAR = "RAGFABRIC_CONFIG"
DEFAULT_FILENAME = "ragfabric.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Strict):
    provider: Literal["openai", "anthropic", "ollama", "offline"] = "openai"
    model: str | None = None
    base_url: str | None = None


class EmbeddingsConfig(_Strict):
    provider: Literal["openai", "ollama", "offline"] = "openai"
    model: str | None = None
    dim: int | None = Field(default=None, ge=1)
    base_url: str | None = None


class RerankerConfig(_Strict):
    kind: Literal["none", "llm", "cross_encoder"] = "none"


class VectorStoreConfig(_Strict):
    kind: Literal["pgvector", "chroma", "memory"] = "pgvector"


class LexicalStoreConfig(_Strict):
    kind: Literal["postgres_fts", "bm25"] = "postgres_fts"


class GraphStoreConfig(_Strict):
    kind: Literal["neo4j", "none"] = "neo4j"
    enabled: bool = False


class CacheConfig(_Strict):
    kind: Literal["redis", "memory"] = "redis"


class IngestionConfig(_Strict):
    chunk_size: int = Field(default=600, ge=50)
    chunk_overlap: int = Field(default=80, ge=0)
    retain_originals: bool = True


class StrategiesConfig(_Strict):
    traditional: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {"top_k": 8, "similarity_threshold": 0.25, "rerank": "none", "max_context_tokens": 6000}
    )
    vectorless: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {"top_k": 8, "phrase_boost": 2.0, "identifier_boost": 3.0}
    )
    agentic: dict[str, float | int | str | bool] = Field(
        default_factory=lambda: {"max_iterations": 4, "max_cost_usd": 0.10, "max_latency_ms": 30000}
    )
    graph: dict[str, float | int | str | bool] = Field(default_factory=lambda: {"max_hops": 2, "max_nodes": 200})


class RouterConfig(_Strict):
    mode: Literal["auto", "manual"] = "auto"
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    classifier_model: str | None = None


class LimitsConfig(_Strict):
    max_upload_mb: int = Field(default=50, ge=1)
    allowed_types: list[str] = Field(default_factory=lambda: ["pdf", "docx", "pptx", "txt", "csv", "md"])
    rate_limit_per_minute: int = Field(default=60, ge=1)


class TelemetryConfig(_Strict):
    otlp_endpoint: str | None = None


class RagFabricConfig(_Strict):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    lexical_store: LexicalStoreConfig = Field(default_factory=LexicalStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


def resolve_config_path(path: Path | None = None) -> Path | None:
    if path is not None:
        return Path(path)
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env)
    local = Path.cwd() / DEFAULT_FILENAME
    return local if local.exists() else None


def load_config(path: Path | None = None) -> RagFabricConfig:
    resolved = resolve_config_path(path)
    if resolved is None:
        return RagFabricConfig()
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return RagFabricConfig.model_validate(data)
```

- [ ] **Step 4: Implement the registry**

```python
# packages/core/src/ragfabric_core/providers/registry.py
"""Build providers from configuration. The only place vendor names are switched on."""

from __future__ import annotations

from collections.abc import Mapping
import os

from ragfabric_core.config_file import EmbeddingsConfig, LLMConfig
from ragfabric_core.providers.anthropic_provider import AnthropicProvider
from ragfabric_core.providers.base import EmbeddingProvider, LLMProvider, ProviderError
from ragfabric_core.providers.offline import HashingEmbeddingProvider, ScriptedLLMProvider
from ragfabric_core.providers.openai_compat import (
    OllamaEmbeddingProvider,
    OllamaProvider,
    OpenAIEmbeddingProvider,
    OpenAIProvider,
)

OPENAI_EMBEDDING_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


def _key(env: Mapping[str, str], name: str, provider: str) -> str:
    value = env.get(name, "")
    if not value:
        raise ProviderError(provider, f"{name} is not set")
    return value


def build_llm_provider(cfg: LLMConfig, env: Mapping[str, str] | None = None) -> LLMProvider:
    env = os.environ if env is None else env
    if cfg.provider == "openai":
        return OpenAIProvider(api_key=_key(env, "OPENAI_API_KEY", "openai"), default_model=cfg.model or "gpt-5.4-mini")
    if cfg.provider == "anthropic":
        return AnthropicProvider(api_key=_key(env, "ANTHROPIC_API_KEY", "anthropic"), default_model=cfg.model or "claude-sonnet-5")
    if cfg.provider == "ollama":
        return OllamaProvider(base_url=cfg.base_url or DEFAULT_OLLAMA_URL, default_model=cfg.model or "llama3.2")
    return ScriptedLLMProvider(responses=[], model=cfg.model or "scripted")


def build_embedding_provider(cfg: EmbeddingsConfig, env: Mapping[str, str] | None = None) -> EmbeddingProvider:
    env = os.environ if env is None else env
    if cfg.provider == "openai":
        model = cfg.model or "text-embedding-3-small"
        dim = cfg.dim or OPENAI_EMBEDDING_DIMS.get(model)
        if dim is None:
            raise ProviderError("openai", f"embeddings.dim must be set for unknown model {model!r}")
        return OpenAIEmbeddingProvider(api_key=_key(env, "OPENAI_API_KEY", "openai"), model=model, dim=dim)
    if cfg.provider == "ollama":
        return OllamaEmbeddingProvider(base_url=cfg.base_url or DEFAULT_OLLAMA_URL, model=cfg.model or "nomic-embed-text", dim=cfg.dim or 768)
    return HashingEmbeddingProvider(dim=cfg.dim)
```

- [ ] **Step 5: Write the example configuration at the repo root**

```yaml
# ragfabric.example.yaml
# Copy to ragfabric.yaml and edit. Every key is optional; defaults are shown.
# Secrets never go here: OPENAI_API_KEY, ANTHROPIC_API_KEY, DATABASE_URL, JWT_SECRET,
# NEO4J_PASSWORD come from the environment (.env.example lists them).

llm:
  provider: openai            # openai | anthropic | ollama | offline
  model: gpt-5.4-mini         # anthropic default: claude-sonnet-5, ollama default: llama3.2
  base_url: null              # ollama or any OpenAI compatible endpoint, e.g. http://localhost:11434/v1

embeddings:
  provider: openai            # openai | ollama | offline
  model: text-embedding-3-small
  dim: null                   # inferred for known models

reranker:
  kind: none                  # none | llm | cross_encoder

vector_store:
  kind: pgvector              # pgvector (lite profile) | chroma (full profile) | memory (tests)
lexical_store:
  kind: postgres_fts          # postgres_fts | bm25
graph_store:
  kind: neo4j
  enabled: false              # true needs the full compose profile
cache:
  kind: redis                 # redis | memory

ingestion:
  chunk_size: 600
  chunk_overlap: 80
  retain_originals: true

strategies:
  traditional: { top_k: 8, similarity_threshold: 0.25, rerank: none, max_context_tokens: 6000 }
  vectorless:  { top_k: 8, phrase_boost: 2.0, identifier_boost: 3.0 }
  agentic:     { max_iterations: 4, max_cost_usd: 0.10, max_latency_ms: 30000 }
  graph:       { max_hops: 2, max_nodes: 200 }

router:
  mode: auto                  # auto | manual
  min_confidence: 0.6
  classifier_model: null      # defaults to llm.model

limits:
  max_upload_mb: 50
  allowed_types: [pdf, docx, pptx, txt, csv, md]
  rate_limit_per_minute: 60

telemetry:
  otlp_endpoint: null         # off by default; e.g. http://otel-collector:4318
```

- [ ] **Step 6: Run the tests, then commit**

Run: `uv run pytest packages/core/tests/test_config_file.py packages/core/tests/test_provider_registry.py -q 2>&1 | tail -3`
Expected: `11 passed`

```bash
uv run ruff check packages --fix -q && git add -A && git commit -q -m "feat(core): add ragfabric.yaml configuration, example file and the provider registry" && git log --oneline -1
```

---

### Task 9: Store, auth and connector interfaces

**Files:**
- Create: `packages/core/src/ragfabric_core/stores/__init__.py`, `stores/base.py`, `auth/base.py`, `connectors/__init__.py`, `connectors/base.py`
- Test: `packages/core/tests/test_interfaces_shape.py`

**Interfaces:**
- Produces (exact, all `@runtime_checkable` Protocols):
  - `VectorStore`: `name: str`; `upsert(chunk_ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None`; `query(vector: list[float], top_k: int, access: AccessFilter, filters: dict | None = None) -> list[RetrievedChunk]`; `delete_document(document_id: int) -> None`; `count() -> int`
  - `LexicalStore`: `name: str`; `index(chunk_ids: list[int], texts: list[str], payloads: list[dict]) -> None`; `search(query: str, top_k: int, access: AccessFilter, filters: dict | None = None) -> list[RetrievedChunk]`; `delete_document(document_id: int) -> None`
  - `GraphStore`: `name: str`; `upsert_entities(entities: list[dict]) -> None`; `upsert_relationships(relationships: list[dict]) -> None`; `neighbours(entity_ids: list[int], hops: int, access: AccessFilter, max_nodes: int) -> dict`; `delete_document(document_id: int) -> None`
  - `Cache`: `get(key: str) -> bytes | None`; `set(key: str, value: bytes, ttl_seconds: int | None = None) -> None`; `incr(key: str, ttl_seconds: int | None = None) -> int`
  - `AuthProvider`: `name: str`; `authenticate(credential: str) -> Principal | None`
  - `SourceDocument(source_id: str, name: str, content_type: str, size_bytes: int | None, modified_at: datetime | None, metadata: dict)`; `Connector`: `name: str`; `list_documents() -> Iterator[SourceDocument]`; `fetch(source_id: str) -> bytes`

Phase 1 only defines these. Phase 2 and 3 implement them. The test proves a class shaped like each protocol is recognised at runtime, which is what the registry checks will rely on.

- [ ] **Step 1: Write the failing test**

```python
# packages/core/tests/test_interfaces_shape.py
from datetime import UTC, datetime

from ragfabric_core.auth.base import AuthProvider
from ragfabric_core.auth.principal import AccessFilter, Principal
from ragfabric_core.connectors.base import Connector, SourceDocument
from ragfabric_core.stores.base import Cache, GraphStore, LexicalStore, VectorStore
from ragfabric_core.strategies.base import RetrievedChunk


class MemVector:
    name = "memory"
    def upsert(self, chunk_ids, vectors, payloads): ...
    def query(self, vector, top_k, access, filters=None): return []
    def delete_document(self, document_id): ...
    def count(self): return 0


class MemLexical:
    name = "bm25"
    def index(self, chunk_ids, texts, payloads): ...
    def search(self, query, top_k, access, filters=None): return []
    def delete_document(self, document_id): ...


class MemGraph:
    name = "none"
    def upsert_entities(self, entities): ...
    def upsert_relationships(self, relationships): ...
    def neighbours(self, entity_ids, hops, access, max_nodes): return {"nodes": [], "edges": []}
    def delete_document(self, document_id): ...


class MemCache:
    def __init__(self): self.d = {}
    def get(self, key): return self.d.get(key)
    def set(self, key, value, ttl_seconds=None): self.d[key] = value
    def incr(self, key, ttl_seconds=None):
        self.d[key] = int(self.d.get(key, 0)) + 1
        return self.d[key]


class StaticAuth:
    name = "static"
    def authenticate(self, credential):
        return Principal(user_id=1, email="a@b.c", role="admin", group_ids=[], api_key_id=None) if credential == "ok" else None


class FolderConnector:
    name = "folder"
    def list_documents(self):
        yield SourceDocument(source_id="a.pdf", name="a.pdf", content_type="application/pdf",
                             size_bytes=10, modified_at=datetime.now(UTC), metadata={})
    def fetch(self, source_id): return b"%PDF"


def test_shapes_are_recognised_at_runtime():
    assert isinstance(MemVector(), VectorStore)
    assert isinstance(MemLexical(), LexicalStore)
    assert isinstance(MemGraph(), GraphStore)
    assert isinstance(MemCache(), Cache)
    assert isinstance(StaticAuth(), AuthProvider)
    assert isinstance(FolderConnector(), Connector)


def test_a_store_missing_a_method_is_rejected():
    class Broken:
        name = "x"
        def upsert(self, *a): ...
    assert not isinstance(Broken(), VectorStore)


def test_protocol_methods_run_with_access_filter():
    empty = MemVector().query([0.0], top_k=3, access=AccessFilter.unrestricted())
    assert empty == [] and MemCache().incr("k") == 1
    assert StaticAuth().authenticate("ok").role == "admin" and StaticAuth().authenticate("no") is None
    doc = next(FolderConnector().list_documents())
    assert isinstance(doc, SourceDocument) and FolderConnector().fetch(doc.source_id).startswith(b"%PDF")
    RetrievedChunk(chunk_id=1, document_id=1, collection_id=None, text="t")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest packages/core/tests/test_interfaces_shape.py -q 2>&1 | tail -3`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# packages/core/src/ragfabric_core/stores/__init__.py
"""Store interfaces. Implementations arrive with the strategies that need them."""
```

```python
# packages/core/src/ragfabric_core/stores/base.py
"""Store interfaces (ADR 0003: every query takes an AccessFilter).

The filter is an argument, not an afterthought, so an implementation cannot
forget it: a store that ranks first and filters later cannot satisfy these
signatures without lying, and the contract tests in later phases catch that.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragfabric_core.auth.principal import AccessFilter
from ragfabric_core.strategies.base import RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def upsert(self, chunk_ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None: ...
    def query(self, vector: list[float], top_k: int, access: AccessFilter, filters: dict | None = None) -> list[RetrievedChunk]: ...
    def delete_document(self, document_id: int) -> None: ...
    def count(self) -> int: ...


@runtime_checkable
class LexicalStore(Protocol):
    name: str

    def index(self, chunk_ids: list[int], texts: list[str], payloads: list[dict]) -> None: ...
    def search(self, query: str, top_k: int, access: AccessFilter, filters: dict | None = None) -> list[RetrievedChunk]: ...
    def delete_document(self, document_id: int) -> None: ...


@runtime_checkable
class GraphStore(Protocol):
    name: str

    def upsert_entities(self, entities: list[dict]) -> None: ...
    def upsert_relationships(self, relationships: list[dict]) -> None: ...
    def neighbours(self, entity_ids: list[int], hops: int, access: AccessFilter, max_nodes: int) -> dict: ...
    def delete_document(self, document_id: int) -> None: ...


@runtime_checkable
class Cache(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None: ...
    def incr(self, key: str, ttl_seconds: int | None = None) -> int: ...
```

```python
# packages/core/src/ragfabric_core/auth/base.py
"""AuthProvider: turns a credential (JWT, API key, OIDC token) into a Principal."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragfabric_core.auth.principal import Principal


@runtime_checkable
class AuthProvider(Protocol):
    name: str

    def authenticate(self, credential: str) -> Principal | None: ...
```

```python
# packages/core/src/ragfabric_core/connectors/__init__.py
"""Document sources: upload today, watched folder and Drive later."""
```

```python
# packages/core/src/ragfabric_core/connectors/base.py
"""Connector interface. A connector lists documents and fetches their bytes; ingestion does the rest."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    source_id: str
    name: str
    content_type: str
    size_bytes: int | None = None
    modified_at: datetime | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


@runtime_checkable
class Connector(Protocol):
    name: str

    def list_documents(self) -> Iterator[SourceDocument]: ...
    def fetch(self, source_id: str) -> bytes: ...
```

- [ ] **Step 4: Run the tests, then commit**

Run: `uv run pytest packages/core/tests/test_interfaces_shape.py -q 2>&1 | tail -3`
Expected: `3 passed`

```bash
uv run ruff check packages --fix -q && git add -A && git commit -q -m "feat(core): add VectorStore, LexicalStore, GraphStore, Cache, AuthProvider and Connector interfaces" && git log --oneline -1
```

---

### Task 10: Platform tables and migration 0002

**Files:**
- Create: `packages/core/src/ragfabric_core/models/access.py`, `models/runs.py`, `models/evaluation.py`, `models/graph.py`
- Modify: `packages/core/src/ragfabric_core/models/__init__.py` (import all model modules so `Base.metadata` is complete), `migrations/env.py`, `db/session.py` (the side effect import), `tests/test_migrations.py` (`EXPECTED_TABLES`)
- Create: `packages/core/src/ragfabric_core/migrations/versions/0002_platform_tables.py` (autogenerated then reviewed)

**Interfaces:**
- Produces the tables: `groups`, `group_members`, `collection_grants`, `document_overrides`, `api_keys`, `audit_log`, `conversations`, `messages`, `retrieval_runs`, `sources`, `evaluation_runs`, `evaluation_results`, `entities`, `relationships`. ORM classes: `Group`, `GroupMember`, `CollectionGrant`, `DocumentOverride`, `ApiKey`, `AuditLog`, `Conversation`, `ChatMessage`, `RetrievalRun`, `Source`, `EvaluationRun`, `EvaluationResult`, `Entity`, `Relationship`.

- [ ] **Step 1: Update the drift test first so it fails**

In `packages/core/tests/test_migrations.py` replace the `EXPECTED_TABLES` set and the model import:

```python
from ragfabric_core import models  # noqa: F401  (registers every table)

EXPECTED_TABLES = {
    "users", "collections", "documents", "chunks", "query_logs",
    "groups", "group_members", "collection_grants", "document_overrides", "api_keys", "audit_log",
    "conversations", "messages", "retrieval_runs", "sources",
    "evaluation_runs", "evaluation_results", "entities", "relationships",
}
```

Run: `uv run pytest packages/core/tests/test_migrations.py -q 2>&1 | tail -3`
Expected: `test_upgrade_head_creates_every_table` FAILS (tables missing).

- [ ] **Step 2: Make the models package import every module**

```python
# packages/core/src/ragfabric_core/models/__init__.py
"""ORM models. Importing this package registers every table on Base.metadata."""

from ragfabric_core.models import access, document, evaluation, graph, runs, user  # noqa: F401
from ragfabric_core.models.base import Base

__all__ = ["Base", "access", "document", "evaluation", "graph", "runs", "user"]
```

Change the two side effect imports to match: in `db/session.py` `init_db` and in `migrations/env.py` replace `from ragfabric_core.models import document, user  # noqa: F401` with `from ragfabric_core import models  # noqa: F401`.

- [ ] **Step 3: Write the models**

```python
# packages/core/src/ragfabric_core/models/access.py
"""Access control tables (ADR 0003).

Groups hold users. Grants give a group read or write on a collection.
Overrides restrict or allow one document below its collection. API keys are
principals with scopes. The audit log records what every run returned and
what the access filter removed.
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base, utcnow


class Group(Base):
    __tablename__ = "groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class GroupMember(Base):
    __tablename__ = "group_members"

    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class CollectionGrant(Base):
    __tablename__ = "collection_grants"
    __table_args__ = (UniqueConstraint("group_id", "collection_id", name="uq_grant_group_collection"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), nullable=False, index=True)
    collection_id: Mapped[int] = mapped_column(ForeignKey("collections.id"), nullable=False, index=True)
    # "read" | "write"
    permission: Mapped[str] = mapped_column(String, default="read", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class DocumentOverride(Base):
    __tablename__ = "document_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False, index=True)
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.id"), nullable=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    # "deny" | "read"
    permission: Mapped[str] = mapped_column(String, default="deny", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    principal_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    collection_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    strategies: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    principal_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True)
    action: Mapped[str] = mapped_column(String, nullable=False, index=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    retrieval_run_id: Mapped[int | None] = mapped_column(ForeignKey("retrieval_runs.id"), nullable=True)
    sources_returned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sources_filtered: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False, index=True)
```

```python
# packages/core/src/ragfabric_core/models/runs.py
"""Conversations, messages, retrieval runs and the sources each run returned.

RetrievalRun is the row every metric, trace and audit entry hangs off. One
row per ask, regardless of strategy, so the Compare page and the dashboards
read one table.
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base, utcnow


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class ChatMessage(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), nullable=False, index=True)
    # "user" | "assistant"
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_run_id: Mapped[int | None] = mapped_column(ForeignKey("retrieval_runs.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class RetrievalRun(Base):
    __tablename__ = "retrieval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"), nullable=True)
    conversation_id: Mapped[int | None] = mapped_column(ForeignKey("conversations.id"), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # "auto" | "manual"
    mode: Mapped[str] = mapped_column(String, default="manual", nullable=False)
    requested_strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    selected_strategy: Mapped[str] = mapped_column(String, nullable=False, index=True)
    fallback_from: Mapped[str | None] = mapped_column(String, nullable=True)
    router_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    router_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    query_type: Mapped[str | None] = mapped_column(String, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    retrieval_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    generation_latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    retrieval_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String, nullable=True)
    trace: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False, index=True)


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    retrieval_run_id: Mapped[int] = mapped_column(ForeignKey("retrieval_runs.id"), nullable=False, index=True)
    chunk_id: Mapped[int | None] = mapped_column(ForeignKey("chunks.id"), nullable=True)
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"), nullable=True, index=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cited: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
```

```python
# packages/core/src/ragfabric_core/models/evaluation.py
"""Benchmark runs and per question results (ADR 0004: numbers come from here, never from prose)."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base, utcnow


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    commit_sha: Mapped[str] = mapped_column(String, default="", nullable=False)
    strategy: Mapped[str] = mapped_column(String, nullable=False, index=True)
    llm_model: Mapped[str] = mapped_column(String, default="", nullable=False)
    embedding_model: Mapped[str] = mapped_column(String, default="", nullable=False)
    judge_model: Mapped[str | None] = mapped_column(String, nullable=True)
    question_set: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_run_id: Mapped[int] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False, index=True)
    question_id: Mapped[str] = mapped_column(String, nullable=False)
    question_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    difficulty: Mapped[str] = mapped_column(String, default="", nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, default="", nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_run_id: Mapped[int | None] = mapped_column(ForeignKey("retrieval_runs.id"), nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reciprocal_rank: Mapped[float | None] = mapped_column(Float, nullable=True)
    correctness: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
```

```python
# packages/core/src/ragfabric_core/models/graph.py
"""Relational mirror of the knowledge graph, for the console and audit. Neo4j answers queries."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ragfabric_core.models.base import Base, utcnow


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("normalized_name", "entity_type", name="uq_entity_name_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    normalized_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    aliases: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    source_chunk_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), nullable=False, index=True)
    target_entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), nullable=False, index=True)
    relation_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    source_chunk_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
```

- [ ] **Step 4: Generate migration 0002 against a database at revision 0001, then review it**

```bash
cd ~/AI/ragfabric-wt/phase-1
rm -f /tmp/rf_autogen.db
export DATABASE_URL=sqlite:////tmp/rf_autogen.db JWT_SECRET=dev-only-secret-not-for-production-use-1234567890
uv run python -c "from ragfabric_core.db.migrate import upgrade; import os; upgrade(os.environ['DATABASE_URL'], '0001_initial_schema')"
uv run alembic -c packages/core/src/ragfabric_core/migrations/alembic.ini revision --autogenerate -m "platform tables" --rev-id 0002_platform_tables
unset DATABASE_URL JWT_SECRET
ls packages/core/src/ragfabric_core/migrations/versions/
```

Open the generated `0002_platform_tables.py` and check: `down_revision = '0001_initial_schema'`; every table above is created in `upgrade()` with `retrieval_runs` created before `audit_log`, `messages` and `sources` (they reference it), `api_keys` before `audit_log` and `retrieval_runs`, `entities` before `relationships`; indexes use `batch_alter_table` like 0001; `downgrade()` drops in reverse order. Replace the generated docstring with:

```python
"""Platform tables for RagFabric: access control, runs and sources, evaluation, graph mirror.

Revision ID: 0002_platform_tables
Revises: 0001_initial_schema
"""
```

- [ ] **Step 5: Run the migration tests on SQLite, then on PostgreSQL 18 via OrbStack**

Run: `uv run pytest packages/core/tests/test_migrations.py -q 2>&1 | tail -3`
Expected: `3 passed` (create, drift, downgrade)

```bash
docker run -d --rm --name rf-pg-test -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragfabric -p 55432:5432 pgvector/pgvector:0.8.6-pg18 >/dev/null && sleep 6
export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:55432/ragfabric JWT_SECRET=dev-only-secret-not-for-production-use-1234567890
uv run python -c "
import os
from ragfabric_core.db.migrate import upgrade, downgrade
u=os.environ['DATABASE_URL']; upgrade(u); downgrade(u); upgrade(u); print('postgres round trip ok')"
unset DATABASE_URL JWT_SECRET; docker stop rf-pg-test >/dev/null
```

Expected: `postgres round trip ok`.

- [ ] **Step 6: Run everything, then commit**

Run: `uv run pytest -q 2>&1 | tail -2`
Expected: all green (the v1 count plus the new tests).

```bash
uv run ruff check packages --fix -q && git add -A && git commit -q -m "feat(core): add access control, run, evaluation and graph tables with migration 0002" && git log --oneline -1
```

---

### Task 11: The `ragfabric` CLI and umbrella package

**Files:**
- Create: `packages/cli/pyproject.toml`, `packages/cli/README.md`, `packages/cli/src/ragfabric_cli/__init__.py`, `packages/cli/src/ragfabric_cli/main.py`
- Test: `packages/cli/tests/test_cli.py`

**Interfaces:**
- Produces: console script `ragfabric` with commands `version`, `config validate [--path PATH] [--check-providers]`, `db upgrade [--revision head]`, `db downgrade [--revision base]`, `serve [--host 0.0.0.0] [--port 8000] [--reload]`. Exit code 0 on success, 1 on a reported problem.

- [ ] **Step 1: Write the failing tests**

```python
# packages/cli/tests/test_cli.py
import os

from sqlalchemy import create_engine, inspect
from typer.testing import CliRunner

from ragfabric_cli.main import app

runner = CliRunner()


def test_version_prints_the_core_version():
    from ragfabric_core import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0 and __version__ in result.stdout


def test_db_upgrade_and_downgrade_on_sqlite(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    assert runner.invoke(app, ["db", "upgrade"]).exit_code == 0
    assert "retrieval_runs" in inspect(create_engine(url)).get_table_names()
    assert runner.invoke(app, ["db", "downgrade"]).exit_code == 0
    assert inspect(create_engine(url)).get_table_names() == ["alembic_version"]


def test_config_validate_reports_active_implementations(tmp_path):
    p = tmp_path / "ragfabric.yaml"
    p.write_text("llm:\n  provider: offline\nembeddings:\n  provider: offline\n  dim: 16\ncache:\n  kind: memory\n")
    result = runner.invoke(app, ["config", "validate", "--path", str(p)])
    assert result.exit_code == 0, result.stdout
    assert "llm: offline" in result.stdout and "embeddings: offline (hashing-16)" in result.stdout
    assert "vector_store: pgvector" in result.stdout


def test_config_validate_fails_on_missing_provider_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = tmp_path / "ragfabric.yaml"
    p.write_text("llm:\n  provider: openai\n")
    result = runner.invoke(app, ["config", "validate", "--path", str(p)])
    assert result.exit_code == 1 and "OPENAI_API_KEY" in result.stdout


def test_config_validate_fails_on_typo(tmp_path):
    p = tmp_path / "ragfabric.yaml"
    p.write_text("llm:\n  provdier: openai\n")
    result = runner.invoke(app, ["config", "validate", "--path", str(p)])
    assert result.exit_code == 1 and "provdier" in result.stdout
```

- [ ] **Step 2: Create the package and run to verify failure**

```toml
# packages/cli/pyproject.toml
[project]
name = "ragfabric"
version = "0.1.0a0"
description = "RagFabric: self hosted, measurement first RAG platform. Installs the engine, the API and the ragfabric command."
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.13,<3.14"
dependencies = [
  "ragfabric-core",
  "ragfabric-server",
  "typer>=0.27",
]

[project.scripts]
ragfabric = "ragfabric_cli.main:app"

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ragfabric_cli"]
```

```bash
cd ~/AI/ragfabric-wt/phase-1
printf '# ragfabric\n\n`pip install ragfabric` installs the engine, the HTTP API and the `ragfabric` command. See the repository README.\n' > packages/cli/README.md
printf '"""RagFabric command line."""\n' > packages/cli/src/ragfabric_cli/__init__.py
uv sync
uv run pytest packages/cli/tests/test_cli.py -q 2>&1 | tail -3
```

Expected: `ModuleNotFoundError: No module named 'ragfabric_cli.main'`

- [ ] **Step 3: Implement**

```python
# packages/cli/src/ragfabric_cli/main.py
"""The ragfabric command.

Thin by design: every command calls a function in ragfabric_core or starts
ragfabric_server. No logic lives here that the API or the tests cannot reach
another way.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from pydantic import ValidationError

from ragfabric_core import __version__
from ragfabric_core.config_file import load_config, resolve_config_path
from ragfabric_core.db import migrate
from ragfabric_core.providers.base import ProviderError
from ragfabric_core.providers.registry import build_embedding_provider, build_llm_provider

app = typer.Typer(help="RagFabric: self hosted, measurement first RAG platform.", no_args_is_help=True)
db_app = typer.Typer(help="Database migrations.")
config_app = typer.Typer(help="Configuration.")
app.add_typer(db_app, name="db")
app.add_typer(config_app, name="config")


def _database_url() -> str:
    from ragfabric_core.config import get_settings

    return os.environ.get("DATABASE_URL") or get_settings().database_url


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(f"ragfabric {__version__}")


@db_app.command()
def upgrade(revision: str = typer.Option("head", help="Target revision.")) -> None:
    """Apply migrations up to REVISION (default head)."""
    migrate.upgrade(_database_url(), revision)
    typer.echo(f"database upgraded to {revision}")


@db_app.command()
def downgrade(revision: str = typer.Option("base", help="Target revision.")) -> None:
    """Roll migrations back to REVISION (default base, which removes every table)."""
    migrate.downgrade(_database_url(), revision)
    typer.echo(f"database downgraded to {revision}")


@config_app.command("validate")
def config_validate(
    path: Path | None = typer.Option(None, "--path", help="ragfabric.yaml to validate."),
    check_providers: bool = typer.Option(False, "--check-providers", help="Also make one live call per provider."),
) -> None:
    """Load the configuration and report the active implementation for each interface."""
    resolved = resolve_config_path(path)
    typer.echo(f"config file: {resolved if resolved else 'none (defaults)'}")
    try:
        cfg = load_config(path)
    except ValidationError as exc:
        typer.echo("configuration is invalid:")
        for err in exc.errors():
            typer.echo(f"  {'.'.join(str(p) for p in err['loc'])}: {err['msg']}")
        raise typer.Exit(code=1)

    problems: list[str] = []
    try:
        llm = build_llm_provider(cfg.llm)
        typer.echo(f"llm: {cfg.llm.provider} ({llm.default_model})")
    except ProviderError as exc:
        problems.append(str(exc))
    try:
        emb = build_embedding_provider(cfg.embeddings)
        typer.echo(f"embeddings: {cfg.embeddings.provider} ({emb.model})")
    except ProviderError as exc:
        problems.append(str(exc))
    typer.echo(f"reranker: {cfg.reranker.kind}")
    typer.echo(f"vector_store: {cfg.vector_store.kind}")
    typer.echo(f"lexical_store: {cfg.lexical_store.kind}")
    typer.echo(f"graph_store: {cfg.graph_store.kind} ({'enabled' if cfg.graph_store.enabled else 'disabled'})")
    typer.echo(f"cache: {cfg.cache.kind}")
    typer.echo(f"router: {cfg.router.mode} (min_confidence {cfg.router.min_confidence})")

    if check_providers and not problems:
        from ragfabric_core.providers.base import Message

        try:
            out = llm.complete([Message(role="user", content="Reply with the single word: pong")], max_tokens=8)
            typer.echo(f"llm live check: ok ({out.input_tokens} in, {out.output_tokens} out, {out.latency_ms} ms)")
            vec = emb.embed(["ping"])
            typer.echo(f"embeddings live check: ok (dim {len(vec.vectors[0])})")
        except ProviderError as exc:
            problems.append(str(exc))

    if problems:
        typer.echo("problems:")
        for p in problems:
            typer.echo(f"  {p}")
        raise typer.Exit(code=1)
    typer.echo("configuration ok")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
    reload: bool = typer.Option(False, help="Auto reload (development only)."),
) -> None:
    """Run the HTTP API with uvicorn."""
    import uvicorn

    uvicorn.run("ragfabric_server.main:app", host=host, port=port, reload=reload)
```

- [ ] **Step 4: Run the tests, verify the console script and the import contract, commit**

Run: `uv run pytest packages/cli/tests/test_cli.py -q 2>&1 | tail -3`
Expected: `5 passed`

```bash
cd ~/AI/ragfabric-wt/phase-1 && uv run ragfabric version && uv run ragfabric --help | head -5 && uv run lint-imports && uv run pytest -q 2>&1 | tail -2
uv run ruff check packages --fix -q && git add -A && git commit -q -m "feat(cli): add the ragfabric command with version, config validate, db upgrade/downgrade and serve" && git log --oneline -1
```

---

### Task 12: Docker Compose profiles and images

**Files:**
- Create: `deploy/docker/api.Dockerfile`, `deploy/docker/ui.Dockerfile`, `deploy/docker/nginx.conf` (moved from `frontend/nginx.conf` in Task 13; this task writes the api side and the compose file, and Task 13 completes the ui image)
- Rewrite: `docker-compose.yml`, `.env.example`
- Create: `.dockerignore` (repo root)

**Interfaces:**
- Produces: `docker compose up` starts `postgres`, `redis`, `api`, `ui` (lite). `docker compose --profile full up` adds `chroma` and `neo4j`. API health at `http://localhost:8000/health`, UI at `http://localhost:4200`, Chroma heartbeat at `http://localhost:8001/api/v2/heartbeat`, Neo4j browser at `http://localhost:7474`.

- [ ] **Step 1: Write the API image**

```dockerfile
# deploy/docker/api.Dockerfile
# RagFabric API image. Multi stage: resolve the uv workspace, then copy only the venv.
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/README.md packages/core/
COPY packages/server/pyproject.toml packages/server/README.md packages/server/
COPY packages/cli/pyproject.toml packages/cli/README.md packages/cli/
RUN uv sync --frozen --no-dev --no-install-workspace
COPY packages/ packages/
RUN uv sync --frozen --no-dev

FROM python:3.13-slim
RUN useradd --create-home --uid 10001 ragfabric
WORKDIR /app
COPY --from=builder --chown=ragfabric:ragfabric /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER ragfabric
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=12 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["sh", "-c", "ragfabric db upgrade && ragfabric serve --host 0.0.0.0 --port 8000"]
```

```
# .dockerignore (repo root)
.git
.venv
**/__pycache__
**/*.pyc
**/node_modules
**/dist
**/.angular
.env
MEMORY*.md
docs
```

- [ ] **Step 2: Write the compose file**

```yaml
# docker-compose.yml
# Lite profile (default):   docker compose up --build
#   postgres (pgvector), redis, api, ui. Enough for Traditional and Vectorless RAG.
# Full profile:             docker compose --profile full up --build
#   adds chroma and neo4j for all four strategies.
# Secrets and connection strings come from .env (copy .env.example).

name: ragfabric

services:
  postgres:
    image: pgvector/pgvector:0.8.6-pg18
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-ragfabric}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-ragfabric}
      POSTGRES_DB: ${POSTGRES_DB:-ragfabric}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-ragfabric} -d ${POSTGRES_DB:-ragfabric}"]
      interval: 5s
      timeout: 5s
      retries: 12

  redis:
    image: redis:8.8-alpine
    command: ["redis-server", "--save", "60", "1", "--loglevel", "warning"]
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 12

  api:
    build:
      context: .
      dockerfile: deploy/docker/api.Dockerfile
    env_file:
      - path: .env
        required: false
    environment:
      DATABASE_URL: postgresql+psycopg://${POSTGRES_USER:-ragfabric}:${POSTGRES_PASSWORD:-ragfabric}@postgres:5432/${POSTGRES_DB:-ragfabric}
      REDIS_URL: redis://redis:6379/0
      CHROMA_URL: http://chroma:8000
      NEO4J_URI: bolt://neo4j:7687
      CORS_ORIGINS: http://localhost:4200,http://localhost:80
      RAGFABRIC_CONFIG: /app/ragfabric.yaml
    volumes:
      - ./ragfabric.yaml:/app/ragfabric.yaml:ro
      - uploads:/app/data/uploads
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  ui:
    build:
      context: ./apps/assistant
      dockerfile: ../../deploy/docker/ui.Dockerfile
    ports:
      - "4200:80"
    depends_on:
      - api

  chroma:
    image: chromadb/chroma:1.5.9
    profiles: ["full"]
    environment:
      ANONYMIZED_TELEMETRY: "false"
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/data
    healthcheck:
      test: ["CMD-SHELL", "bash -c 'exec 3<>/dev/tcp/127.0.0.1/8000 && echo ok' || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 12

  neo4j:
    image: neo4j:2026.08.1-community
    profiles: ["full"]
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-please-change-me-12}
      NEO4J_server_memory_heap_max__size: 1G
      NEO4J_server_memory_pagecache_size: 512M
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://127.0.0.1:7474 >/dev/null || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 18

volumes:
  postgres_data:
  redis_data:
  chroma_data:
  neo4j_data:
  uploads:
```

The api service mounts `./ragfabric.yaml`; create it from the example before the first run (`cp ragfabric.example.yaml ragfabric.yaml`) and add `ragfabric.yaml` to `.gitignore` under a comment `# local configuration (copy of ragfabric.example.yaml)`.

- [ ] **Step 3: Rewrite `.env.example`**

```bash
# .env.example  (copy to .env; .env is ignored by git)
# Every value has a safe development default in code. Production refuses the
# placeholder JWT secret and the shipped bootstrap admin (see docs/configuration.md).

ENVIRONMENT=development

# Provider keys: set only the ones your ragfabric.yaml uses. Ollama needs none.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Relational database (compose sets this for the api container automatically)
DATABASE_URL=sqlite:///./ragfabric.db
POSTGRES_USER=ragfabric
POSTGRES_PASSWORD=ragfabric
POSTGRES_DB=ragfabric

# Cache and stores (compose sets these for the api container)
REDIS_URL=redis://localhost:6379/0
CHROMA_URL=http://localhost:8001
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=please-change-me-12

# Auth. Generate a real secret: python -c "import secrets; print(secrets.token_urlsafe(64))"
JWT_SECRET=change-me
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Bootstrap admin, seeded on first start when both are set. Refused in production.
FIRST_ADMIN_EMAIL=admin@example.com
FIRST_ADMIN_PASSWORD=adminpass123

# CORS for the Angular dev server and the nginx container
CORS_ORIGINS=http://localhost:4200,http://localhost:80

# Path to ragfabric.yaml (defaults to ./ragfabric.yaml when present)
# RAGFABRIC_CONFIG=./ragfabric.yaml
```

Keep the v1 variable names (`EMBEDDING_DIM`, `DEFAULT_TOP_K`, `MAX_CONTEXT_CHARS`, `HYBRID_ALPHA`) working: they still exist in `ragfabric_core/config.py` and stay documented in `docs/configuration.md` as v1 tuning that Phase 3 retires.

- [ ] **Step 4: Bring up the lite profile on OrbStack and verify (ui builds only after Task 13; use `--no-deps` for now)**

```bash
cd ~/AI/ragfabric-wt/phase-1 && cp -n ragfabric.example.yaml ragfabric.yaml; cp -n .env.example .env
docker compose up -d --build postgres redis api 2>&1 | tail -3
for i in $(seq 1 30); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 2; done
curl -s http://localhost:8000/health | python3 -m json.tool | head -8
docker compose ps --format 'table {{.Service}}\t{{.Status}}'
```

Expected: `status: ok`, `service: ragfabric`, postgres and redis `healthy`, api `healthy`. If `ragfabric db upgrade` failed inside the container, `docker compose logs api` shows the Alembic error; fix before continuing.

```bash
docker compose --profile full up -d chroma neo4j 2>&1 | tail -2
for i in $(seq 1 30); do curl -sf http://localhost:8001/api/v2/heartbeat >/dev/null && curl -sf http://localhost:7474 >/dev/null && break; sleep 3; done
curl -s http://localhost:8001/api/v2/heartbeat; echo; curl -s -o /dev/null -w "neo4j http %{http_code}\n" http://localhost:7474
docker compose --profile full down
```

Expected: a heartbeat JSON from Chroma and `neo4j http 200`.

- [ ] **Step 5: Commit**

```bash
cd ~/AI/ragfabric-wt/phase-1 && git add -A && git status --short | grep -E "^\?\? (\.env|ragfabric\.yaml)$" && echo "STOP: .env or ragfabric.yaml is unignored" || git commit -q -m "build: add lite and full compose profiles with PostgreSQL 18 pgvector, Redis, Chroma and Neo4j, and the API image" && git log --oneline -1
```

---

### Task 13: Move the frontend to `apps/assistant`, upgrade to Angular 22, add Tailwind 4

**Files:**
- Move: `frontend/` to `apps/assistant/`, `frontend/nginx.conf` to `deploy/docker/nginx.conf`, delete `frontend/Dockerfile`
- Create: `deploy/docker/ui.Dockerfile`, `apps/assistant/.postcssrc.json`, `apps/assistant/src/tailwind.css`
- Modify: `apps/assistant/package.json`, `angular.json`, `karma.conf.js` as the update chain requires

**Interfaces:**
- Produces: `npm test` and `npm run build` pass in `apps/assistant` on Angular 22.1, TypeScript 6.0, Tailwind 4.3; the built site contains Tailwind utilities; the ui image serves it behind nginx with `/api` proxied to `api:8000`.

The upgrade runs one major at a time, because Angular migrations are per major. The local Node is 26, which Angular 18 and 19 accept but Angular 20 does not list, so the chain runs inside a `node:24` container. Tests need Chrome, so they run on the host after the chain, on Angular 22 which supports Node 26.

- [ ] **Step 1: Move**

```bash
cd ~/AI/ragfabric-wt/phase-1
git mv frontend apps/assistant
git mv apps/assistant/nginx.conf deploy/docker/nginx.conf
git rm -q apps/assistant/Dockerfile
sed -i '' 's|proxy_pass http://backend:8000/api/;|proxy_pass http://api:8000/api/;|' deploy/docker/nginx.conf
git add -A && git commit -q -m "refactor: move the Angular app to apps/assistant" && git log --oneline -1
```

- [ ] **Step 2: Baseline on the host before touching versions**

```bash
cd ~/AI/ragfabric-wt/phase-1/apps/assistant && rm -rf node_modules .angular && npm ci 2>&1 | tail -1 && npm test 2>&1 | tail -3 && npm run build 2>&1 | tail -2
```

Expected: 20 specs pass, build succeeds. This is the number that must still hold at the end.

- [ ] **Step 3: Run the update chain inside Node 24**

```bash
cd ~/AI/ragfabric-wt/phase-1/apps/assistant
for v in 18 19 20 21 22; do
  echo "=== Angular $v ==="
  docker run --rm -v "$PWD":/w -w /w -e NG_CLI_ANALYTICS=false node:24 sh -c "
    npx --yes @angular/cli@$v update @angular/core@$v @angular/cli@$v --allow-dirty --force 2>&1 | tail -15" || break
  git add -A && git commit -q -m "build(assistant): update to Angular $v" && git log --oneline -1
done
```

After each major, if the update reports a migration that needs a manual decision, apply it in the working tree and re run that iteration. Known points on this chain:
- Angular 20 removes the `@angular-devkit/build-angular` karma plugin path. If `npm test` later fails on `require('@angular-devkit/build-angular/plugins/karma')`, switch the `test` target in `angular.json` to `"builder": "@angular/build:karma"`, delete the `frameworks` and `plugins` lines that name `@angular-devkit/build-angular` from `karma.conf.js`, and keep the `customLaunchers` block.
- Angular 20+ may move `zone.js` out of `polyfills` if the app opts into zoneless; this app keeps zone.js, so leave `polyfills: ["zone.js"]` and `["zone.js", "zone.js/testing"]` as they are.
- Angular 22 pins `typescript` to `>=6.0 <6.1`; the update writes it. Do not bump TypeScript to 7.

- [ ] **Step 4: Verify on the host**

```bash
cd ~/AI/ragfabric-wt/phase-1/apps/assistant && rm -rf node_modules .angular && npm ci 2>&1 | tail -1
node -e "const p=require('./package.json'); console.log('angular', p.dependencies['@angular/core'], 'typescript', p.devDependencies.typescript)"
npm test 2>&1 | tail -3 && npm run build 2>&1 | tail -3
```

Expected: `angular ^22.1.x typescript ~6.0.x`, `Executed 20 of 20 SUCCESS`, build succeeds. Fix compile errors introduced by removed APIs in the app code (typical: `HttpClientModule` to `provideHttpClient`, which the v1 app already uses) and commit as `fix(assistant): adapt to Angular 22 APIs`.

- [ ] **Step 5: Add Tailwind 4**

```bash
cd ~/AI/ragfabric-wt/phase-1/apps/assistant && npm install -D tailwindcss@4 @tailwindcss/postcss@4 postcss 2>&1 | tail -1
printf '{\n  "plugins": {\n    "@tailwindcss/postcss": {}\n  }\n}\n' > .postcssrc.json
printf '@import "tailwindcss";\n' > src/tailwind.css
```

Edit `angular.json`: in both `build.options.styles` and `test.options.styles` change `["src/styles.scss"]` to `["src/tailwind.css", "src/styles.scss"]`. In `src/app/app.component.html` add `class="min-h-screen"` to the outermost element so at least one utility is used.

```bash
npm run build 2>&1 | tail -2 && grep -rl "min-h-screen" dist/frontend/browser/*.css | head -1 && npm test 2>&1 | tail -2
```

Expected: build ok, one css file contains `min-h-screen`, 20 specs pass.

- [ ] **Step 6: Write the ui image and verify the whole lite profile**

```dockerfile
# deploy/docker/ui.Dockerfile  (build context: apps/assistant)
FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.29-alpine
COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html
COPY --from=build /app/../deploy/docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

The second COPY cannot reach outside the build context, so change it: in `docker-compose.yml` the ui service already has `context: ./apps/assistant`; instead copy nginx.conf into the context at build time is wrong too. Use the repository root as the context:

```yaml
  ui:
    build:
      context: .
      dockerfile: deploy/docker/ui.Dockerfile
```

and write the Dockerfile against the root:

```dockerfile
# deploy/docker/ui.Dockerfile  (build context: repository root)
FROM node:24-alpine AS build
WORKDIR /app
COPY apps/assistant/package.json apps/assistant/package-lock.json ./
RUN npm ci
COPY apps/assistant/ .
RUN npm run build

FROM nginx:1.29-alpine
COPY --from=build /app/dist/frontend/browser /usr/share/nginx/html
COPY deploy/docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

Check the exact nginx alpine tag exists before using it (`curl -s -o /dev/null -w '%{http_code}' https://hub.docker.com/v2/repositories/library/nginx/tags/1.29-alpine`); if it returns 404 use `nginx:alpine`.

```bash
cd ~/AI/ragfabric-wt/phase-1 && docker compose up -d --build 2>&1 | tail -2
for i in $(seq 1 30); do curl -sf http://localhost:4200 >/dev/null && curl -sf http://localhost:4200/api/../health >/dev/null; curl -sf http://localhost:8000/health >/dev/null && break; sleep 2; done
curl -s -o /dev/null -w "ui %{http_code}\n" http://localhost:4200
curl -s -o /dev/null -w "api via nginx %{http_code}\n" -X POST http://localhost:4200/api/auth/login -d "username=admin@example.com&password=adminpass123"
docker compose ps --format 'table {{.Service}}\t{{.Status}}'
docker compose down
```

Expected: `ui 200`, `api via nginx 200`, four services up, api healthy.

- [ ] **Step 7: Commit**

```bash
cd ~/AI/ragfabric-wt/phase-1 && git add -A && git commit -q -m "build(assistant): add Tailwind 4 and the nginx ui image on Node 24" && git log --oneline -1
```

---

### Task 14: Documentation sync, tracking, and the pull request

**Files:**
- Modify: `README.md`, `ROADMAP.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/architecture.md`, `docs/getting-started.md`, `docs/configuration.md`, `docs/providers.md`, `docs/README.md`
- Modify (local only): `~/AI/ragfabric/MEMORY.md`
- GitHub: issue #2 checklist, Discussion #13 comment, PR

This task applies the sync rule (memory: "one change, every place"). Nothing is merged here; the PR is opened for the owner.

- [ ] **Step 1: README**

In the Technology stack table set Python 3.13, PostgreSQL 18 with pgvector, Neo4j 2026.08 community, Angular 22, TypeScript 6, Tailwind 4, Node 24, uv. In Installation replace the two profile commands with:

```bash
cp .env.example .env && cp ragfabric.example.yaml ragfabric.yaml
docker compose up --build                  # lite: PostgreSQL with pgvector, Redis, API, UI
docker compose --profile full up --build   # adds Chroma and Neo4j
```

and remove the note that says the profiles land in Phase 1. In Repository layout remove the sentence that says the layout is introduced in Phase 1 and that `backend/` and `frontend/` remain. In "What works today (v1)" replace the local run block with:

```bash
uv sync                       # Python 3.13 workspace: core, server, cli
uv run ragfabric db upgrade
uv run ragfabric serve --reload            # API on :8000
cd apps/assistant && npm ci && npm start   # UI on :4200
```

Add one sentence after the strategy interface code block: "Phase 1 shipped this interface, the provider and store interfaces, the configuration and cost modules, the platform tables and the compose profiles; the four strategies arrive in the following phases."

- [ ] **Step 2: ROADMAP, CHANGELOG, CONTRIBUTING**

Tick every Phase 1 checkbox in `ROADMAP.md` and change the section heading to `### Phase 1: Architecture, monorepo, interfaces, Docker, database (merged)` once the PR merges; on the branch leave it unticked-to-ticked with `[x]` and no "(merged)" word. Under Phase 2 add `- [ ] CLI: extend with ingest, users, keys (init, version, config validate, db, serve exist since Phase 1)`.

In `CHANGELOG.md` under `## [Unreleased]` add:

```markdown
### Added
- Monorepo: `packages/core` (engine), `packages/server` (API), `packages/cli` (the `ragfabric` command, published as `ragfabric`), `apps/assistant` (UI). uv workspace on Python 3.13.
- Core interfaces: `RetrieverStrategy` and `RetrievalResult`, `LLMProvider`, `EmbeddingProvider`, `VectorStore`, `LexicalStore`, `GraphStore`, `Cache`, `AuthProvider`, `Connector`; `Principal` and `AccessFilter`.
- Providers: OpenAI, Anthropic, Ollama (OpenAI compatible), offline test doubles. Provider registry driven by `ragfabric.yaml`.
- `ragfabric.yaml` configuration with strict validation; `ragfabric config validate`.
- Pricing configuration (`pricing.yaml`, dated and sourced) and cost estimation that reports unknown models instead of guessing.
- Migration 0002: groups, grants, overrides, API keys, audit log, conversations, messages, retrieval runs, sources, evaluation runs and results, entities, relationships.
- Docker Compose profiles: lite (PostgreSQL 18 with pgvector, Redis, API, UI) and full (adds Chroma 1.5.9 and Neo4j 2026.08.1).
- CI: lint (ruff, import-linter), tests, migrations against PostgreSQL 18, frontend on Node 24.

### Changed
- Frontend upgraded from Angular 17 to Angular 22 with TypeScript 6 and Tailwind 4.
- PostgreSQL driver psycopg2 to psycopg 3; connection URLs use `postgresql+psycopg://`.
- The v1 hybrid pipeline is available as `LegacyHybridStrategy` behind the strategy interface until Phase 3 replaces it.
```

In `CONTRIBUTING.md` replace the Development setup paragraph with:

```markdown
## Development setup

```bash
uv sync                                   # installs core, server, cli into .venv on Python 3.13
uv run pytest                             # every package's tests
uv run ruff check packages && uv run lint-imports
cd apps/assistant && npm ci && npm test   # Angular specs (needs Chrome)
docker compose up --build                 # lite profile on Docker Desktop, OrbStack or Podman
```

## Keep every surface in sync

A change to scope, behaviour, naming or plan updates README, ROADMAP, CHANGELOG, the relevant `docs/*.md`, the phase issue and, for plan changes, the Roadmap discussion, in the same pull request. Reviewers check this before anything else.
```

- [ ] **Step 3: docs**

`docs/architecture.md`: replace the "Migration from the v1 layout" table with a sentence that the move happened in Phase 1 and list the actual `packages/core/src/ragfabric_core/` tree from this plan's "File structure" section. Change every "Python 3.12" to "Python 3.13".

`docs/getting-started.md`: replace the status line with "Status: the compose profiles, the CLI and `ragfabric.yaml` exist since Phase 1. Traditional and Vectorless RAG arrive in Phases 3 and 4." Replace the RagFabric run block with the compose commands from Step 1, and the CLI block with the commands that exist now: `ragfabric version`, `ragfabric config validate`, `ragfabric db upgrade`, `ragfabric serve`. Mark `ingest`, `users`, `ask` as Phase 2 and 3.

`docs/configuration.md`: replace the status line with "Status: implemented in Phase 1; strategy blocks are read by the phases that ship them." Paste the real `ragfabric.example.yaml`. Add a row to the environment table for `RAGFABRIC_CONFIG` and `REDIS_URL`. Add a sentence under Pricing: "Every entry in `pricing.yaml` carries `as_of` and `source`; a model without an entry reports `known=false` and no number."

`docs/providers.md`: change the status of OpenAI, Anthropic, Ollama LLM rows and OpenAI, Ollama, offline embedding rows to "Phase 1 (shipped)". Add a "Default models" table: OpenAI `gpt-5.4-mini` and `text-embedding-3-small`, Anthropic `claude-sonnet-5`, Ollama `llama3.2` and `nomic-embed-text`.

`docs/README.md`: update the status column for architecture, configuration and providers to "Implemented in Phase 1".

Grep the whole tree for em dashes and old paths before committing:

```bash
cd ~/AI/ragfabric-wt/phase-1 && grep -rn $'\xe2\x80\x94' --include="*.md" --include="*.py" --include="*.ts" --include="*.yml" --include="*.yaml" --include="*.toml" --exclude-dir=node_modules --exclude-dir=.angular --exclude-dir=.git . | grep -v "^./apps/assistant/src/app/pages/documents\|^./packages/core/src/ragfabric_core/retrieve/hybrid.py\|^./packages/server/src/ragfabric_server/api/routes/auth.py\|^./packages/server/tests/conftest.py" ; grep -rn "backend/app\|frontend/src\|from app\." --include="*.md" --include="*.py" --include="*.yml" --exclude-dir=node_modules --exclude-dir=.git . | grep -v CHANGELOG
```

Expected: no output from either grep (the four excluded files carry v1 em dashes and are out of scope).

- [ ] **Step 4: Run the complete verification, then commit and push**

```bash
cd ~/AI/ragfabric-wt/phase-1
uv run ruff check packages && uv run ruff format --check packages && uv run lint-imports
uv run pytest -q 2>&1 | tail -2
(cd apps/assistant && npm test 2>&1 | tail -2 && npm run build 2>&1 | tail -1)
docker compose --profile full up -d --build 2>&1 | tail -1 && sleep 25 && docker compose ps --format 'table {{.Service}}\t{{.Status}}' && docker compose --profile full down
git add -A && git commit -q -m "docs: sync README, roadmap, changelog and docs with the Phase 1 architecture" && git push -u origin feat/phase-1-architecture 2>&1 | tail -1
```

Expected: linters clean, all Python tests pass (v1 count plus about 45 new), 20 specs pass, six services healthy.

- [ ] **Step 5: Update issue #2, comment on Discussion #13, open the PR**

```bash
cd ~/AI/ragfabric-wt/phase-1 && R=ranjan-del/ragfabric
gh issue comment 2 -R $R --body "Implementation is in PR #<number>. Every checklist item in this issue is covered by a task in docs/plans/2026-09-13-phase-1-architecture.md; the checklist is ticked when the PR merges."
gh pr create -R $R --base main --head feat/phase-1-architecture --title "feat: Phase 1 architecture, monorepo, interfaces, providers, migrations and compose profiles" --body-file - <<'EOF'
## What

Phase 1 of the roadmap (closes #2 when merged). The v1 assistant becomes the RagFabric skeleton: a uv monorepo with `packages/core`, `packages/server`, `packages/cli` and `apps/assistant`; every core interface later phases plug into; OpenAI, Anthropic, Ollama and offline providers; `ragfabric.yaml` and `pricing.yaml`; migration 0002 with the platform tables; lite and full compose profiles; Angular 22 with Tailwind 4. Retrieval behaviour is unchanged; the v1 pipeline runs behind the new `RetrieverStrategy` interface.

## Phase / area

Phase 1. packages/core, packages/server, packages/cli, apps/assistant, deploy, CI, docs.

## How it was tested

- `uv run pytest`: all v1 tests pass from their new locations plus the new interface, provider, pricing, configuration, migration and CLI tests (counts in the CI log).
- `uv run lint-imports`: 2 contracts kept.
- Migrations: SQLite round trip in tests; PostgreSQL 18 round trip in CI and locally on OrbStack.
- `npm test` in apps/assistant: 20 specs on Angular 22; `npm run build` succeeds with Tailwind utilities in the output.
- `docker compose --profile full up --build`: six services healthy; `/health` and the login route via nginx return 200.
- Live provider check run once locally with a real OpenAI key (`pytest -m integration`), not in CI.

## Learning notes

Written in module docstrings: why one RetrievalResult, why Protocols over base classes, why the access filter is an argument, why pricing is dated data, why one OpenAI compatible class serves two vendors.

## Checklist

- [x] Tests added or updated
- [x] Docs / README updated (README, ROADMAP, CHANGELOG, CONTRIBUTING, docs/architecture, getting-started, configuration, providers)
- [x] No secrets, keys or real credentials in the diff
- [x] No benchmark numbers
EOF
NUM=$(gh pr list -R $R --head feat/phase-1-architecture --json number -q '.[0].number'); echo "PR #$NUM"
gh issue comment 2 -R $R --body "Implementation PR: #$NUM" >/dev/null
gh api graphql -f query='mutation($id:ID!,$body:String!){addDiscussionComment(input:{discussionId:$id,body:$body}){comment{url}}}' -f id="$(gh api graphql -f query='{repository(owner:"ranjan-del",name:"ragfabric"){discussion(number:13){id}}}' -q .data.repository.discussion.id)" -f body="Phase 1 is in review as pull request #$NUM: monorepo layout, core interfaces, four providers, configuration and pricing modules, migration 0002, compose profiles, Angular 22 with Tailwind 4. Retrieval quality is unchanged in this phase by design; Phases 3 and 4 bring the first two strategies. Details: https://github.com/ranjan-del/ragfabric/pull/$NUM" -q .data.addDiscussionComment.comment.url
```

Then update `~/AI/ragfabric/MEMORY.md`: Phase 1 status "PR #<n> open, awaiting owner review", the learning log with the five notes above, and any decision changed during implementation.

- [ ] **Step 6: Report to the owner**

Report in the agreed format: Completed, Architecture, Why, Testing, Next. Include the PR link, the exact test counts from the run, and the statement that merge and deploy wait for the owner.

---

## Self review

**Spec coverage.** Design section 3 (layers, dependency direction): Tasks 1, 2, 11. Section 4 (layout): Task 1, 11, 13; `packages/sdk-python`, `packages/sdk-typescript`, `apps/console`, `evaluation/`, `examples/` are Phase 3, 4, 8, 9 deliverables and are intentionally absent. Section 5 (interfaces): Tasks 3, 4, 9. Section 6 (configuration, pricing): Tasks 7, 8. Section 7 (access control data model): Task 10; enforcement is Phase 2. Section 11 (releases, GHCR): CI in Task 2; image publishing is Phase 4. ADR 0001: Task 2 import-linter. ADR 0002: Task 3. ADR 0003: Task 3 `AccessFilter` and Task 9 signatures. ADR 0004: Task 7 dated pricing, Task 14 no numbers in docs.

**Placeholder scan.** The only `<verified>` and `<date>` tokens are in Task 7 Step 4 and are instructions to read values from the pricing pages on the day, with the rule that unverifiable entries are omitted. Task 13 Step 3 names the three known manual points on the Angular chain concretely. No "TBD" or "handle edge cases".

**Type consistency.** `RetrievedChunk` fields are identical in Tasks 3, 9 and the legacy adapter. `AccessFilter.allows(document_id, collection_id)` is used with those keyword names in Tasks 3 and 9. `Completion` and `EmbeddingResult` field names match between Tasks 4, 5, 6, 8. `LLMConfig.provider` literal set in Task 8 matches the branches in `build_llm_provider` and the CLI messages in Task 11 (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). `ragfabric_core.db.migrate.upgrade/downgrade(db_url, revision)` signature is the same in Tasks 1, 10, 11 and the CI job. CI check names in Task 2 match the `protect-main` ruleset.
