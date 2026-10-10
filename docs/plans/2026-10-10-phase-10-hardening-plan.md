# Phase 10: hardening, connectors, docs site, deployment. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RagFabric can be run for other people: bounded abuse, traceable requests, validated
uploads, documents arriving from a folder or Google Drive, a docs site, guarded release automation
and deployment guides, with the whole upload to cited answer path proved offline in CI.

**Design:** `docs/plans/2026-10-10-phase-10-hardening.md` (decisions D1 to D33, each with its reason).

**Tech Stack:** Python 3.13 (uv), FastAPI, SQLAlchemy and Alembic, Redis, Typer.
New optional dependency: `google-api-python-client` (extra `drive`). New dependency group `docs`:
`mkdocs-material`. One migration: `0009_connector_items`.

**Issue:** https://github.com/ranjan-del/ragfabric/issues/11

**Predecessor on main:** `docs/plans/2026-10-01-phase-7b-terminal-experience.md`

**Release:** none. Nothing is tagged, published, enabled or deployed.

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.13**, line length 100, `ruff check packages` and `ruff format --check packages` clean.
  Run `ruff format packages/`, never `ruff format .`.
- **import-linter stays at 3 contracts kept, 0 broken.** Core never imports the server or the CLI.
- **Tests are offline and deterministic.** No model, no Google, no network. PostgreSQL tests use the
  `RAGFABRIC_TEST_DATABASE_URL` guard and the `integration` marker.
- **CI keeps exactly three required checks.** New tests go into "Backend tests" or "Migrations apply
  cleanly"; anything else is a new optional job.
- **No workflow publishes on a push to main.** Publishing jobs run only on manual dispatch or behind
  a repository variable the owner sets.
- **ADR 0004 holds for instructions too.** A guide for a platform the project has not deployed to
  says so.
- **Commits** authored `Ranjan G` with the GitHub noreply address; conventional prefixes; no
  trailers; no assistant references; no organisation names; no em dashes. Push after every task.
- **Tests must be able to fail.** Each new test is seen failing before the code that passes it.

---

## Review Focus

Inputs the design implies but a happy-path test would miss, most likely first. Each has a test in
the task that owns it.

1. **A route that resolves the caller twice** (upload depends on `get_current_user` and
   `get_access_filter`). Expected: charged once (Task 2).
2. **Redis down while counting.** Expected: request allowed, one ERROR log line with the request id
   (Task 2).
3. **A spoofed `X-Forwarded-For` on login.** Expected: ignored unless uvicorn trusts the peer
   (Task 2).
4. **A `.docx` that is a zip bomb, or a zip with no `word/document.xml`.** Expected: 413 or 415
   before any parser runs (Task 3).
5. **A chunked upload with no `Content-Length` over the limit.** Expected: 413 without reading the
   rest (Task 3).
6. **A file still being copied into the watched folder.** Expected: not ingested until stable
   (Task 5).
7. **A symlink in the watched folder pointing outside it.** Expected: skipped (Task 5).
8. **A file removed from the source.** Expected: its document deleted from every store, and only
   documents that connector created (Task 4).
9. **A touched but unchanged file.** Expected: no re-ingest, no version bump (Task 4).
10. **An `X-Request-ID` header with a newline or 500 characters.** Expected: replaced by a fresh id
    (Task 1).
11. **A streamed `/api/ask` response.** Expected: still streams, and its log lines carry the request
    id (Task 1).

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/core/src/ragfabric_core/telemetry/logs.py` | `request_id_var`, `JsonFormatter`, `configure_logging(fmt, level)`, `valid_request_id` |
| `packages/server/src/ragfabric_server/middleware.py` | `RequestContextMiddleware` (ids, access line), `UploadSizeLimitMiddleware` |
| `packages/core/src/ragfabric_core/auth/ratelimit.py` | `RateLimitResult`, `hit(cache, subject, limit, now)`; `check_rate_limit` kept as a wrapper |
| `packages/server/src/ragfabric_server/ratelimit.py` | `charge(request, subject, limit)`, `auth_attempt_limit` dependency, the 429 builder |
| `packages/core/src/ragfabric_core/ingest/validate.py` | `UploadRejected`, `validate_upload(filename, data, limits)` |
| `packages/core/src/ragfabric_core/ingest/delete.py` | `delete_document_everywhere(db, document_id)` (moved out of the route) |
| `packages/core/src/ragfabric_core/models/connector.py` | `ConnectorItem` |
| `packages/core/src/ragfabric_core/migrations/versions/0009_connector_items.py` | the table |
| `packages/core/src/ragfabric_core/connectors/sync.py` | `SyncReport`, `sync_connector(...)` |
| `packages/core/src/ragfabric_core/connectors/folder.py` | `FolderConnector` |
| `packages/core/src/ragfabric_core/connectors/google_drive.py` | `GoogleDriveConnector`, `build_drive_service` |
| `packages/core/src/ragfabric_core/connectors/registry.py` | `build_connector(cfg)` |
| `packages/cli/src/ragfabric_cli/commands/connectors.py` | `connectors list`, `connectors sync [NAME] [--watch] [--once]` |
| `packages/server/tests/test_e2e_flow.py` | SQLite end-to-end through the SDK |
| `packages/server/tests/test_e2e_postgres.py` | the same scenario on PostgreSQL (integration) |
| `deploy/compose/docker-compose.prod.yml`, `deploy/compose/Caddyfile`, `deploy/compose/.env.prod.example`, `deploy/compose/ragfabric.prod.yaml` | single-VM stack |
| `render.yaml` | Render blueprint |
| `scripts/bump_version.py` | lockstep version bump |
| `mkdocs.yml`, `docs/index.md`, `docs/contributing.md`, `docs/connectors.md`, `docs/deploy/*.md`, `docs/operations.md` | docs site |
| `.github/workflows/docs.yml` | build on PR (optional), deploy on manual dispatch |
| `.github/workflows/release.yml` | version guard, dists, gated PyPI job |

---

## Task 1: Request ids and structured JSON logging

**Files:** create `telemetry/logs.py`, `ragfabric_server/middleware.py`; modify `config_file.py`
(`LoggingConfig`), `ragfabric_server/main.py`, `cli/main.py` (`serve` turns uvicorn's access log off
under JSON), `cli/commands/worker.py`, `workers/runner.py` (job id in context),
`ragfabric.example.yaml` and the CLI's packaged copy.
Tests: `core/tests/test_logs.py`, `server/tests/test_request_context.py`.

- [ ] Tests first:
  - `valid_request_id`: accepts `abc-123_x.y`; rejects empty, 129 characters, newline, quote, space.
  - `JsonFormatter`: one line of valid JSON with `ts`, `level`, `logger`, `msg`; `request_id` only
    when set; `extra={"k": 1}` appears; an exception adds `exc`; a non-JSON-serialisable extra is
    stringified, never raises.
  - `configure_logging("json", "INFO")` installs one handler on the root logger, idempotent when
    called twice; `RAGFABRIC_LOG_FORMAT` and `RAGFABRIC_LOG_LEVEL` override the config.
  - Server: every response has `X-Request-ID`; a valid incoming id is echoed; an invalid one is
    replaced by a 32 character hex id; a log line written inside a route carries the id; one access
    line per request with method, path without query string, status and duration; a streamed
    `/api/ask` still yields SSE events and carries the header; an unhandled exception gives 500
    `{"detail": "Internal server error", "request_id": <id>}` and the traceback is logged under
    that id.
- [ ] Implement, run the files, then the full suite, ruff, lint-imports. Commit
  `feat: request ids and structured JSON logging`. Push. Update progress.md.

## Task 2: Rate limits for users, keys and sign-in attempts

**Files:** modify `auth/ratelimit.py`, `stores/redis_cache.py` (`incr` atomic), `stores/base.py`
docstring, `config_file.py` (`LimitsConfig.user_rate_limit_per_minute`,
`auth_attempts_per_minute`), `deps.py`, `api/routes/auth.py`; create `ragfabric_server/ratelimit.py`.
Tests: extend `core/tests/test_api_keys.py`, `core/tests/test_caches.py`; create
`server/tests/test_rate_limits.py`.

- [ ] Tests first:
  - `hit` returns `allowed`, `limit`, `remaining`, `retry_after` (seconds to the next minute, at
    least 1); `check_rate_limit` still returns a bool.
  - `RedisCache.incr` sends INCR and EXPIRE NX in one transaction (fake client records the pipeline).
  - A signed-in user over `user_rate_limit_per_minute` gets 429 with `Retry-After` and the JSON body.
  - Upload (two caller-resolving dependencies) charges once: with a limit of 2, two uploads succeed.
  - Login over `auth_attempts_per_minute` from one address gets 429; another address is not affected;
    an `X-Forwarded-For` header from an untrusted peer does not change the address.
  - An API key keeps its own per-key limit and is not also charged as a user.
  - A cache whose `incr` raises: request allowed, ERROR logged with the request id.
- [ ] Implement, test, lint. Commit `feat: rate limits for signed-in users and sign-in attempts`.
  Push. Update progress.md.

## Task 3: Upload validation in core, size enforced on the wire

**Files:** create `ingest/validate.py`; modify `config_file.py` (`max_uncompressed_mb`),
`api/routes/documents.py` (use the validator; read in bounded chunks), `ragfabric_server/middleware.py`
(`UploadSizeLimitMiddleware`), `main.py`, `cli/commands/ingest.py` (validator, skipped files
reported with the reason).
Tests: `core/tests/test_validate.py`, extend `server/tests/test_api.py` or a new
`server/tests/test_upload_validation.py`, extend `cli/tests/test_cli.py`.

- [ ] Tests first (each builds its bytes in the test):
  - Valid PDF, DOCX, PPTX, TXT, MD, CSV pass and return the extension.
  - Unknown extension: 400. Not in `allowed_types`: 400. Empty: 400. Over `max_upload_mb`: 413.
  - `.pdf` without `%PDF-`: 415. `.docx` that is not a zip, or a zip without `word/document.xml`:
    415. `.pptx` without `ppt/presentation.xml`: 415. `.txt` with a NUL byte or invalid UTF-8: 415.
  - Zip over `max_uncompressed_mb`, over 10,000 entries, or with a ratio over 100: 413, and no entry
    is decompressed (assert via a zip whose declared sizes are large).
  - Middleware: `Content-Length` over the limit on the upload path gives 413 before the route runs;
    a chunked body over the limit gives 413; other paths are not limited by it.
  - CLI `ingest` of a folder with one fake PDF reports it as skipped with the reason and exits 1.
- [ ] Implement, test, lint. Commit `feat: validate upload content and bound upload size on the wire`.
  Push. Update progress.md.

## Task 4: Connector state and the sync engine

**Files:** create `models/connector.py`, migration `0009_connector_items.py`, `ingest/delete.py`,
`connectors/sync.py`; modify `models/__init__.py`, `api/routes/documents.py` (delete through core),
`connectors/base.py` (`SourceDocument.version` optional).
Tests: `core/tests/test_connector_sync.py`, extend `core/tests/test_migrations.py` expectations if
they enumerate heads, `core/tests/test_migrations_postgres.py` covers upgrade and downgrade.

- [ ] Tests first, with an in-memory fake connector:
  - New source: ingested, row written, report `added=1`.
  - Same size, modified time and version: not fetched (fake counts fetches), `unchanged=1`.
  - Changed metadata, same bytes: fetched, not re-ingested, version unchanged, metadata updated.
  - Changed bytes: re-ingested in place, same document id, version 2.
  - Source gone with `on_delete: delete`: document deleted from every store and its row removed;
    with `keep`: document kept, row kept.
  - A document uploaded by hand, or created by another connector, is never deleted.
  - A rejected file (validator): not ingested, counted in `skipped` with the reason, retried only
    if its metadata changes.
  - A fetch that raises: counted in `failed`, other sources still processed, no row written.
  - Migration 0009 upgrades and downgrades on SQLite and PostgreSQL.
- [ ] Implement, test, lint. Commit `feat(core): connector sync engine with per-source state`.
  Push. Update progress.md.

## Task 5: Watched folder connector and the `connectors` command

**Files:** create `connectors/folder.py`, `connectors/registry.py`, `cli/commands/connectors.py`;
modify `config_file.py` (`ConnectorConfig` list with unique names, discriminated by `kind`),
`cli/main.py`, example configs.
Tests: `core/tests/test_connector_folder.py`, `core/tests/test_config_connectors.py`,
`cli/tests/test_connectors_cli.py`.

- [ ] Tests first:
  - Lists supported files recursively; skips hidden, `~$`, `.tmp`, `.part`, `.crdownload`;
    skips a symlink resolving outside the root; source id is the path relative to the root.
  - A file modified less than `settle_seconds` ago is not listed yet (clock injected).
  - Config: duplicate names rejected; unknown `kind` rejected; folder needs `path`; Drive needs
    `folder_id` and `credentials_file`.
  - CLI: `connectors list` prints each connector; `connectors sync inbox --once` ingests a file
    dropped into a temporary folder and prints the report; a second run reports it unchanged;
    `--watch` loops until interrupted (test with an injected stop after one cycle).
- [ ] Implement, test, lint. Commit `feat: watched folder connector and ragfabric connectors`.
  Push. Update progress.md.

## Task 6: Google Drive connector

**Files:** create `connectors/google_drive.py`; modify `packages/core/pyproject.toml` (extra
`drive = ["google-api-python-client>=2.190", "google-auth>=2.40"]`), `packages/cli/pyproject.toml`
(extra passthrough), root `pyproject.toml` dev group, `uv.lock`, `connectors/registry.py`; create
`docs/connectors.md` (folder and Drive setup, service account steps, sharing the folder, scopes,
export formats and limits).
Tests: `core/tests/test_connector_google_drive.py`.

- [ ] Tests first:
  - With `HttpMockSequence` and the library's static discovery document: `list_documents` sends
    `q="'<folder>' in parents and trashed = false"`, the expected `fields`, `supportsAllDrives` and
    `includeItemsFromAllDrives`, follows `nextPageToken`, and recurses into subfolders once each.
  - Mapping: a PDF keeps its name and md5 as version; a Google Doc becomes `<name>.docx` with export
    MIME type; a Slides file `.pptx`; a Sheet `.csv`; a shortcut, form or image is skipped.
  - `fetch` of a binary file calls `get_media`; of a Google Doc calls `export_media` with the DOCX
    type.
  - Missing extra: a clear error naming `pip install 'ragfabric[drive]'`.
  - Missing or unreadable credentials file: a clear error, no traceback through the CLI.
- [ ] Implement, test, lint. Commit `feat: Google Drive connector on the official API client`.
  Push. Update progress.md.

## Task 7: End-to-end tests

**Files:** create `server/tests/test_e2e_flow.py`, `server/tests/test_e2e_postgres.py`; modify
`.github/workflows/ci.yml` (the migrations job also runs the PostgreSQL E2E file).

- [ ] Write the scenario from the design's E2E table through `ragfabric_sdk.Client` with
  `transport=test_client._transport`. Make one assertion fail on purpose first to prove it runs.
- [ ] The PostgreSQL tier uses the same scenario function with `RAGFABRIC_TEST_DATABASE_URL`, a
  config selecting `pgvector` and `postgres_fts`, and is skipped without the variable.
- [ ] Run locally against a private PostgreSQL container on a free port. Commit
  `test: end-to-end upload to cited answer on SQLite and PostgreSQL`. Push. Update progress.md.

## Task 8: Single-VM deployment, Render, Railway and static UI guides

**Files:** create `deploy/compose/docker-compose.prod.yml`, `Caddyfile`, `.env.prod.example`,
`ragfabric.prod.yaml`, `render.yaml`, `docs/deploy/single-vm.md`, `docs/deploy/render.md`,
`docs/deploy/railway.md`, `docs/deploy/static-ui.md`, `docs/operations.md` (logs, request ids, rate
limits, validation, backups, upgrades); rewrite `docs/deployment.md` as the overview.
Tests: `core/tests/test_deploy_files.py` (the prod compose parses, publishes only Caddy's ports,
sets `ENVIRONMENT=production` and JSON logs; `render.yaml` parses and references the Dockerfile;
`ragfabric.prod.yaml` validates against `RagFabricConfig`).

- [ ] Tests first, then files. `docker compose -f deploy/compose/docker-compose.prod.yml config`
  must pass. Commit `docs(deploy): single VM, Render, Railway and static UI guides`. Push.

## Task 9: Release automation

**Files:** create `scripts/bump_version.py`; modify `.github/workflows/release.yml`; create
`docs/releasing.md`.
Tests: `core/tests/test_bump_version.py` (runs the script on a copy of the five files in a temporary
directory: all versions and pins move; a malformed version is refused; running twice is a no-op;
`--check vX.Y.Z` exits non-zero on a mismatch).

- [ ] Tests first. Workflow: a `verify` job runs `bump_version.py --check $GITHUB_REF_NAME`; the
  images job needs it; a `dists` job builds with `uv build --all-packages`, runs `twine check`
  through `uvx`, and uploads the files as a workflow artifact; a `pypi` job needs `dists`, runs only
  `if: vars.PYPI_PUBLISH == 'true'`, uses environment `pypi`, `id-token: write`, and
  `pypa/gh-action-pypi-publish` pinned to a release. Commit `ci: version guard, built dists and a
  gated PyPI publish on release tags`. Push.

## Task 10: Docs site

**Files:** create `mkdocs.yml`, `docs/index.md`, `docs/contributing.md`,
`.github/workflows/docs.yml`; modify root `pyproject.toml` (`docs` group), `uv.lock`, `docs/README.md`,
`.gitignore` (`site/`).

- [ ] `mkdocs build --strict` passes locally with nav: Home, Quick start, Concepts, Strategies
  (Traditional, Vectorless, Agentic, Graph, Routing), Connectors, Deployment, Operations, Reference
  (configuration, providers, troubleshooting, architecture, ADRs), Contributing, Licensing. Plans
  and design documents are excluded. Mermaid renders.
- [ ] Workflow: `build` on pull requests touching `docs/**` or `mkdocs.yml` (optional check);
  `deploy` only on `workflow_dispatch`, checks out `ragfabric/ragfabric.github.io` with the
  `DOCS_DEPLOY_KEY` secret and pushes the built site; fails with a clear message when the secret is
  missing. Commit `docs: documentation site with a manual deploy to ragfabric.github.io`. Push.

## Task 11: Verify for real

- [ ] Docs site builds locally (strict).
- [ ] Production compose under project `rf-p10` on non-default ports, Ollama on the host: comes up
  healthy, first admin signs in, a document is uploaded, a question gets a cited answer.
- [ ] Folder connector in the compose `connectors` profile ingests a dropped file.
- [ ] Rate limit returns 429 with `Retry-After`; a fake PDF returns 415; an oversize upload 413.
- [ ] Logs are JSON lines carrying the request id from the response header.
- [ ] Write `docs/learning/hardening-first-run.md` with what happened, including what went wrong.
  Commit, push. Tear down the `rf-p10` stack (never OrbStack itself).

## Task 12: Review, fix wave, documentation sync

- [ ] Whole-branch diff review against the design and the Review Focus list; fix what it finds.
- [ ] CHANGELOG `[Unreleased]`, ROADMAP Phase 10 ticks, `docs/README.md`, README docs table and
  Project documentation statuses, `docs/configuration.md` for every new key.
- [ ] Draft the issue #11 update at `~/AI/ragfabric-notes/phase-10-sdd/issue-11-draft.md`.
- [ ] Commit, push, update progress.md.
