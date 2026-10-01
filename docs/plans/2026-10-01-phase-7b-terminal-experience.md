# Phase 7b: the terminal experience. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Someone who runs `pip install ragfabric` reaches a first cited answer without reading the
docs, and every expected failure tells them the exact command that fixes it.

**Architecture:** A small `ragfabric_cli/ui/` module owns all terminal output and one error handler
at the CLI entry point turns known failures into a problem line plus a fix command. New commands
(`quickstart`, `doctor`, `strategies`) and a welcome screen sit on top of it. `ragfabric ask` and
`ragfabric ingest` gain rich output on a terminal and keep their plain and `--json` output
unchanged. The CLI stays thin: anything that is not presentation lives in core.

**Tech Stack:** Python 3.13 (uv), Typer, `rich` (already a declared dependency of Typer
`>=0.27`, `Requires-Dist: rich>=13.8.0`), SQLAlchemy, Alembic. **No new runtime dependencies. No
migration.**

**Issue:** https://github.com/ranjan-del/ragfabric/issues/8

**Predecessor:** `docs/plans/2026-09-30-phase-7a-query-router.md` (merged as #53, 6520612)

**Release:** v0.4.0 is tagged after this phase merges; tagging and the PyPI publish each need the
owner's explicit approval.

---

## Why this shape, and what was rejected

### Who it is for, and what was found

The target user has just run `pip install ragfabric`, often with no Docker running and no model
configured. Two things in the current CLI fail exactly that user:

| Finding | Effect today |
|---|---|
| `ragfabric init` copies `ragfabric.example.yaml` and `.env.example` from the current directory | A pip install has neither, so `init` fails with "not found in the current directory" |
| `ragfabric ask` goes through a running server over HTTP | A first-time user has no server, and the failure is a connection error |

And one that makes the zero-setup path possible: the default `database_url` is
`sqlite:///./rag.db`, `PgVectorStore` stores vectors as JSON and scores in Python on SQLite, and
the BM25 and full-text stores have Python fallbacks. The no-Docker path works with today's code,
provided the cache is `memory` (the default is Redis) and indexing stays `inline`.

### Decisions the owner made (2026-10-01)

| Question | Decision |
|---|---|
| Quickstart's setup path | Zero-setup first (SQLite), Docker and PostgreSQL optional behind `--docker` |
| No model available | Use offline mode and say so clearly, with the command to upgrade |
| The first answer | Start a temporary server, ask over HTTP through the SDK exactly like `ragfabric ask`, stop the server |
| Output and error layer | One shared `ui` module and one error handler (approach A) |

### Rejected

| Option | Why not |
|---|---|
| Docker-only quickstart | Fails for anyone without Docker, which is the user this phase is for |
| Answering in-process through core in quickstart | A second code path that can drift from `ragfabric ask`; the CLI's own docstring forbids it |
| Pulling an Ollama model for the user | A multi-gigabyte download nobody asked for; quickstart prints the `ollama pull` command instead |
| Per-command formatting and error handling | Drifts command by command, repeated in more than ten files |
| A richer CLI framework (for example rich-click) | A new dependency for what `rich` already does |

---

## Design

### Units

| Unit | File | What it does |
|---|---|---|
| Console | `ragfabric_cli/ui/console.py` | One shared console. Rich output when stdout is a terminal; plain text when piped; no colour under `--json` or `NO_COLOR` |
| Panels | `ragfabric_cli/ui/panels.py` | Answer panel, sources list, routing line, graph path, next-steps panel, check list, strategies table |
| Error handler | `ragfabric_cli/ui/errors.py` | Wraps the entry point. Maps known failures to one problem line plus the exact fix command, exit code 1. `--debug` prints the traceback |
| Welcome screen | `ragfabric_cli/main.py` callback | Bare `ragfabric`: version, `quickstart`, `doctor`, `strategies`, the docs link |
| `doctor` | `ragfabric_cli/commands/doctor.py` and `ragfabric_core/diagnostics.py` | Checks live in core (each returns `CheckResult(name, status, detail, fix)`); the command renders them |
| `quickstart` | `ragfabric_cli/commands/quickstart.py` | The guided path below |
| `strategies` | `ragfabric_cli/commands/strategies.py` | Five rows: auto, traditional, vectorless, agentic, graph |
| `ask` | `ragfabric_cli/commands/ask.py` | Rich answer on a terminal; plain and `--json` unchanged |
| `ingest` | `ragfabric_cli/commands/ingest.py` | Progress bar on a terminal; plain per-file lines when piped |
| `--help` examples | every command | An "Examples" block that works after `quickstart` |
| Packaged files | `ragfabric_cli/data/` | `ragfabric.example.yaml`, `.env.example`, `docker-compose.yml` (PostgreSQL with pgvector plus Redis), a small sample corpus |
| Migration helpers | `ragfabric_core/db/migrate.py` | `current_revision(db_url)` and `head_revision()` |

### Errors the handler knows

| Failure | Message and fix |
|---|---|
| Server not reachable | "No RagFabric server at <url>" and `ragfabric serve` |
| Database not reachable | The database URL with the password masked, and the start command for SQLite or Docker |
| Migrations behind | `ragfabric db upgrade` |
| Invalid configuration | Each bad key and `ragfabric config validate` |
| Model unreachable | The provider and `ollama serve`, `ollama pull <model>`, or the API key variable to set |
| `ProviderError` with 401 or 429 | The key was rejected, or the quota is used up |
| Anything else | A short message and "rerun with `--debug` for details" |

### `ragfabric quickstart`

Works in the current directory, or `--dir`.

| Step | What happens |
|---|---|
| 1. Config | Writes `ragfabric.yaml` and `.env` from the packaged templates when missing; never overwrites without `--force` |
| 2. Model | Probes Ollama at the configured URL: a chat model plus `nomic-embed-text` means Ollama; else `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`; else offline mode with a clear note and the upgrade command. Writes the choice and says why |
| 3. Database | Default SQLite at `./ragfabric.db` with `cache.kind: memory`. `--docker`: checks Docker is running, writes the packaged `docker-compose.yml`, starts PostgreSQL and Redis, waits for health, writes `DATABASE_URL` to `.env`. Docker not running: says how to start it and offers the SQLite path |
| 4. Migrate | Runs migrations, with a spinner |
| 5. Sample data | Ingests the packaged corpus with a progress bar |
| 6. Ask | Starts `ragfabric serve` in the background on a free port, waits for `/health`, asks the sample question through the SDK, prints the rich answer, stops the server. Ctrl-C or any failure always stops it |
| 7. Next steps | A panel: `ragfabric serve`, `ragfabric ask "..."`, `ragfabric ingest <folder>`, `ragfabric strategies`, `ragfabric doctor` |

Rules: every step is idempotent, so a rerun continues where it stopped; no secret is printed;
`--yes` skips prompts; no model is ever downloaded for the user.

### `ragfabric doctor`

| Check | Status | Fix on failure |
|---|---|---|
| Python 3.13 or later | fail | Install the right version |
| Config file found and valid | warn if none (defaults in use), fail if invalid | `ragfabric init`, or the bad keys |
| Database reachable | fail | The start command for SQLite or Docker |
| Migrations at head | fail | `ragfabric db upgrade` |
| LLM provider reachable (one tiny call; `--no-network` skips it) | fail, or warn in offline mode | `ollama serve`, `ollama pull <model>`, or the key variable |
| Embedding provider reachable and dimension matches | fail | `ollama pull nomic-embed-text`, or the dimension note |
| Graph extraction enabled with no model | warn | The config key |
| Server at `--url` reachable | warn | `ragfabric serve` |

Exit code 0 when nothing failed, 1 otherwise; `--json` for scripts. A check reports only what it
actually checked: a skipped check says skipped, never passed (ADR 0004).

### `ragfabric ask`

On a terminal: an answer panel, then numbered sources with document names and pages, the routing
line, a compact graph path for graph answers, and a dimmed "Removed (unsupported)" block when the
citation contract dropped claims. The streaming path shows the answer as it arrives, then the
panel. Piped output keeps today's plain format; `--json` is byte-for-byte unchanged.

### `ragfabric strategies`

Five rows with columns best at, model calls, and an example question. The example questions are
the same strings the router's signal tests use, so the table cannot drift from what the router
actually does.

### Testing

| Area | How |
|---|---|
| UI module | Rendered with a forced terminal and in plain mode; assert the content, not ANSI codes |
| Error handler | One test per known failure: message, fix, exit code 1, traceback only with `--debug` |
| Doctor | One test per check with fakes for pass, warn and fail; one CLI test for `--json` and the exit code |
| Quickstart | End to end in a temporary directory with SQLite and the offline provider: real migrations, real ingest, a real background server, a real cited answer, the server stopped afterwards. Plus: never overwrites without `--force`, masks secrets, stops the server after an error |
| `init` | Works from an empty directory with the packaged templates |
| Regression | Every existing CLI test passes; `--json` output byte-identical |

### Out of scope

An interactive terminal UI, shell completion setup, Windows-specific console work, pulling models
for the user, and the assistant UI (Phase 9).

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.13**, line length 100, `ruff` clean (`E`, `F`, `I`, `UP`, `B`), `ruff format` clean.
- **Run `ruff format packages/`, never `ruff format .`**
- **import-linter contracts stay at 3 kept, 0 broken.** The CLI may import core and the SDK; core
  never imports the CLI.
- **No new runtime dependencies.** `rich` is used only because Typer already declares it.
- **The CLI stays thin.** Checks, migration revisions and anything a test or the API could need
  live in core; the CLI renders.
- **`--json` output is unchanged** on every command that has it. Piped (non-terminal) output stays
  plain and parseable.
- **No secret is ever printed.** Database URLs are shown with the password masked; API keys never.
- **ADR 0004 holds.** `doctor` reports only what it checked; offline mode is labelled as such.
- **SQLite remains the test default.** PostgreSQL-only tests use the `RAGFABRIC_TEST_DATABASE_URL`
  guard, never silently passing.
- **No AI attribution anywhere.** Commits authored `Ranjan G <ranjan.g@ispf.ngo>`, no trailers, no
  assistant references in code, comments, docs or PR bodies.
- **No em dashes** anywhere.
- **Tests must be able to fail.**
- **One implementer at a time in a worktree.**

---

## Review Focus

Inputs and conditions the design implies but a happy-path test would miss, most likely first. Each
line has a test in the task that owns the code.

1. **Output piped to a file or another program** (`ragfabric ask ... | cat`, `> out.txt`). Expected:
   no ANSI codes, no boxes, today's plain lines (Tasks 1, 5, 6).
2. **`quickstart` run twice, or after a crash part-way.** Expected: it continues; it never
   overwrites `ragfabric.yaml` or `.env` without `--force`, and never leaves a server running
   (Task 4).
3. **A `DATABASE_URL` with a password**, in any message from `doctor`, `quickstart` or the error
   handler. Expected: the password is masked as `***` (Tasks 1, 3, 4).
4. **`NO_COLOR=1`, or a terminal narrower than 60 columns.** Expected: no colour; panels wrap and
   never crash (Task 1).
5. **The server port already in use when `quickstart` starts its temporary server.** Expected:
   it picks a free port itself rather than failing (Task 4).

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/cli/src/ragfabric_cli/ui/__init__.py` | Re-exports `console`, `is_rich`, `mask_url` |
| `packages/cli/src/ragfabric_cli/ui/console.py` | The shared console and the rich-or-plain decision |
| `packages/cli/src/ragfabric_cli/ui/panels.py` | Every renderer: answer, sources, routing, graph, dropped claims, checks, strategies, next steps |
| `packages/cli/src/ragfabric_cli/ui/errors.py` | `friendly_error(exc) -> FriendlyError | None` and `run_app(app)` |
| `packages/cli/src/ragfabric_cli/data/` | `ragfabric.example.yaml`, `env.example`, `docker-compose.yml`, `samples/*.md` |
| `packages/cli/src/ragfabric_cli/templates.py` | `template_path(name) -> Path` over the packaged data |
| `packages/core/src/ragfabric_core/diagnostics.py` | `CheckResult` and one function per doctor check |
| `packages/core/src/ragfabric_core/db/migrate.py` | Gains `current_revision(db_url)` and `head_revision()` |
| `packages/cli/src/ragfabric_cli/commands/doctor.py` | Renders the checks |
| `packages/cli/src/ragfabric_cli/commands/quickstart.py` | The guided path |
| `packages/cli/src/ragfabric_cli/commands/strategies.py` | The strategies table |
| `packages/cli/src/ragfabric_cli/main.py` | Welcome callback, `init` from packaged templates, entry point through `run_app` |
| `packages/cli/src/ragfabric_cli/commands/ask.py`, `ingest.py` | Rich output on a terminal |

Tests live in `packages/cli/tests/` and `packages/core/tests/`, using `typer.testing.CliRunner` as
the existing CLI tests do. Run one file with `uv run pytest packages/cli/tests/<file> -v`; the full
suite with `uv run pytest -q -o addopts=""` (add `RAGFABRIC_TEST_DATABASE_URL` for the PostgreSQL
suites).

---

## Task 1: The `ui` module and the error handler

**Files:**
- Create: `ragfabric_cli/ui/__init__.py`, `ui/console.py`, `ui/panels.py`, `ui/errors.py`
- Modify: `packages/cli/pyproject.toml` (`[project.scripts] ragfabric = "ragfabric_cli.main:main"`), `ragfabric_cli/main.py` (add `main()`)
- Test: `packages/cli/tests/test_ui_console.py`, `test_ui_panels.py`, `test_ui_errors.py`

Foundation. Both tracks render through it, so the controller lands it before either starts.

**Interfaces:**
- Produces:
  - `console.is_rich(stream=None) -> bool`: True only when the stream is a terminal, `NO_COLOR` is unset and `RAGFABRIC_PLAIN` is unset.
  - `console.get_console() -> rich.console.Console`: one per process, `highlight=False`, `soft_wrap=False`, `no_color` honoured.
  - `console.mask_url(url: str) -> str`: replaces the password in a URL with `***`; returns non-URLs unchanged.
  - `panels.render_answer(text, citations, *, strategy, router, fallback_from, subgraph, dropped_claims, dropped_relationship_claims) -> None`: rich panel when `is_rich()`, otherwise exactly today's plain lines (moved from `commands/ask.py`: `_print_sources`, `_print_graph`, `_print_routing`).
  - `panels.render_checks(results: list[CheckResult]) -> None`, `panels.render_strategies(rows: list[StrategyRow]) -> None`, `panels.render_next_steps(steps: list[tuple[str, str]]) -> None`.
  - `errors.FriendlyError(problem: str, fix: str | None)`; `errors.friendly_error(exc: BaseException) -> FriendlyError | None`; `errors.run_app(app: typer.Typer) -> None`.
  - `main.main() -> None`: calls `run_app(app)`; `--debug` is a global option on the app callback stored in `ragfabric_cli.ui.errors.DEBUG`.
- `CheckView` and `StrategyRow` are defined here in `panels.py` as plain dataclasses so the CLI owns its view models; Task 3's core `CheckResult` is converted to `CheckView` at the call site.

```python
# ui/panels.py
@dataclass(frozen=True)
class CheckView:
    name: str
    status: Literal["pass", "warn", "fail", "skip"]
    detail: str
    fix: str | None = None

@dataclass(frozen=True)
class StrategyRow:
    name: str
    best_at: str
    model_calls: str
    example: str
```

`render_checks` takes `list[CheckView]`.

**The error map** (in `friendly_error`, order matters, first match wins):

| Exception | Problem line | Fix |
|---|---|---|
| `httpx.ConnectError`, `httpx.ConnectTimeout` | `No RagFabric server at <url>` (url from the request) | `ragfabric serve` |
| `ragfabric_sdk.errors.AuthError` | `The server refused the credentials` | `pass --token or --api-key, or set RAGFABRIC_TOKEN` |
| `ragfabric_sdk.errors.RagFabricError` | `The server returned an error: <message>` | `None` |
| `sqlalchemy.exc.OperationalError` | `Cannot reach the database at <mask_url(url)>` | `ragfabric quickstart` for SQLite, or `docker compose up -d postgres` |
| `ragfabric_core.diagnostics.MigrationsBehind` (Task 3 defines it; import lazily and skip the row if absent) | `The database schema is behind` | `ragfabric db upgrade` |
| `pydantic.ValidationError` from `load_config` | `ragfabric.yaml is invalid: <first error loc and msg>` | `ragfabric config validate` |
| `ProviderError` whose message contains `401` or `invalid_api_key` | `<provider> rejected the API key` | `check the key in .env` |
| `ProviderError` whose message contains `429` or `insufficient_quota` | `<provider> quota is used up or rate limited` | `None` |
| `ProviderError` whose message contains `Connection` | `Cannot reach <provider>` | `ollama serve` for ollama, otherwise `check the provider URL` |
| any other `ProviderError` | `<provider>: <message>` | `None` |

`ProviderError` carries no status code, so the 401 and 429 rows match on the message text; say so in a comment. Anything not in the map returns `None`, and `run_app` prints `Unexpected error: <type>: <message>` plus `rerun with --debug for details`, exit 1. `typer.Exit` and `typer.Abort` pass through untouched. With `--debug`, `run_app` re-raises so the traceback prints.

- [ ] **Step 1: Write the failing tests.**

```python
# test_ui_console.py
import io
from ragfabric_cli.ui.console import is_rich, mask_url

def test_a_pipe_is_never_rich():
    assert is_rich(io.StringIO()) is False

def test_no_color_turns_rich_off(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self): return True
    monkeypatch.setenv("NO_COLOR", "1")
    assert is_rich(Tty()) is False

def test_a_password_is_masked():
    assert mask_url("postgresql+psycopg://rf:s3cret@db:5432/rf") == "postgresql+psycopg://rf:***@db:5432/rf"

def test_a_url_without_a_password_is_unchanged():
    assert mask_url("sqlite:///./ragfabric.db") == "sqlite:///./ragfabric.db"
```

```python
# test_ui_panels.py (plain mode, the default under CliRunner)
from ragfabric_cli.ui.panels import render_answer

CITATIONS = [{"marker": "[1]", "used": True, "filename": "leave.pdf", "page": 2, "document_id": 4}]

def test_plain_answer_keeps_todays_lines(capsys):
    render_answer("Ten days carry forward [1].", CITATIONS, strategy="traditional",
                  router={"source": "signals", "reasoning": "A short question about one concept."},
                  fallback_from=None, subgraph=None, dropped_claims=[], dropped_relationship_claims=[])
    out = capsys.readouterr().out
    assert "Ten days carry forward [1]." in out
    assert "sources:\n  [1] leave.pdf p2" in out
    assert "Strategy: traditional (signals). A short question about one concept." in out
    assert "\x1b[" not in out

def test_rich_answer_on_a_narrow_terminal_does_not_crash(monkeypatch):
    from ragfabric_cli.ui import console
    monkeypatch.setattr(console, "is_rich", lambda stream=None: True)
    monkeypatch.setenv("COLUMNS", "40")
    render_answer("x " * 200, CITATIONS, strategy="graph", router=None, fallback_from=None,
                  subgraph={"nodes": [{"id": 1, "name": "Ravi Sharma"}, {"id": 2, "name": "Asha Rao"}],
                            "edges": [{"source_id": 1, "target_id": 2, "walked_as": "REPORTS_TO"}]},
                  dropped_claims=[], dropped_relationship_claims=[])
```

```python
# test_ui_errors.py
import httpx, pytest
from ragfabric_cli.ui.errors import friendly_error
from ragfabric_core.providers.base import ProviderError

def test_no_server_names_the_url_and_the_fix():
    exc = httpx.ConnectError("refused", request=httpx.Request("POST", "http://localhost:8000/api/ask"))
    fe = friendly_error(exc)
    assert "http://localhost:8000" in fe.problem and fe.fix == "ragfabric serve"

def test_a_database_password_never_appears():
    from sqlalchemy.exc import OperationalError
    exc = OperationalError("connect", {}, Exception("could not connect to postgresql+psycopg://rf:s3cret@db/rf"))
    fe = friendly_error(exc)
    assert "s3cret" not in fe.problem + (fe.fix or "")

@pytest.mark.parametrize(("message", "needle"), [("HTTP 401 invalid_api_key", "rejected"),
                                                 ("429 insufficient_quota", "quota")])
def test_provider_errors_are_explained(message, needle):
    assert needle in friendly_error(ProviderError("openai", message)).problem

def test_an_unknown_error_is_not_claimed():
    assert friendly_error(KeyError("x")) is None
```

Add one `CliRunner` test that a command raising `httpx.ConnectError` exits 1 with the problem and fix and no `Traceback`, and that the same command with `--debug` shows `Traceback`. Since `CliRunner` invokes the Typer app, not `main()`, wire `run_app`'s handling into the app callback so both paths behave the same, or test `main()` through `subprocess` with `sys.executable -m ragfabric_cli.main`; state which in the report.

- [ ] **Step 2: Run them, confirm they fail** (`ModuleNotFoundError: ragfabric_cli.ui`).
- [ ] **Step 3: Implement.** Move `_print_sources`, `_print_graph` and `_print_routing` from `commands/ask.py` into `panels.py` unchanged as the plain branch, and leave `ask.py` calling `render_answer` so every existing `test_ask_*` and `test_cli_*` test passes untouched. The rich branch uses `rich.panel.Panel`, `rich.table.Table` and `rich.text.Text` only. The database URL in an `OperationalError` is masked by applying `mask_url` to every URL-shaped substring of the message (regex `\b[a-z+]+://\S+`).
- [ ] **Step 4: Run the new tests and every existing CLI test.**
- [ ] **Step 5: Commit** `feat(cli): add the shared console, renderers and the friendly error handler`

---

## Task 2: Packaged templates, sample corpus, and `init` from any directory

**Files:**
- Create: `ragfabric_cli/data/ragfabric.example.yaml` (copy of the repo root file), `data/env.example` (copy of `.env.example`), `data/docker-compose.yml`, `data/samples/handbook.md`, `data/samples/leave-policy.md`, `data/samples/team.md`, `ragfabric_cli/templates.py`
- Modify: `ragfabric_cli/main.py` (`init`)
- Test: `packages/cli/tests/test_templates.py`, additions to `test_cli.py`

**Interfaces:**
- Produces: `templates.template_path(name: str) -> Path` (raises `FileNotFoundError` naming the template); `templates.sample_paths() -> list[Path]`; `templates.SAMPLE_QUESTION: str`.

The sample corpus is three short markdown files about a fictional company, written for this task: a handbook page, a leave policy that states "up to 10 days of unused annual leave carry forward", and a team page that says who reports to whom. `SAMPLE_QUESTION = "How many days of unused annual leave carry forward?"`. `docker-compose.yml` is the repo's `postgres` and `redis` services only (same images and digests as the root `docker-compose.yml`), with no `api`, `worker` or `ui`.

- [ ] **Step 1: Write the failing tests:** every template exists via `template_path`; the samples parse with the core parser (`ragfabric_core.ingest.parser`); a test that the root `ragfabric.example.yaml` and the packaged copy are identical (so they cannot drift); `init` run in an empty `tmp_path` writes `.env` and `ragfabric.yaml`; a second `init` without `--force` leaves both untouched; with `--force` it rewrites them.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement** `templates.py` with `importlib.resources.files("ragfabric_cli") / "data" / name`, and change `init` to copy from `template_path` instead of the current directory. Keep `init`'s existing messages.
- [ ] **Step 4: Prove packaging.** Build the wheel (`uv build packages/cli --wheel -o /tmp/rf-wheel`) and assert with `python -m zipfile -l` that every file under `ragfabric_cli/data/` is in it. Record the listing in the report.
- [ ] **Step 5: Commit** `feat(cli): ship the config templates and a sample corpus, and let init run anywhere`

---

## Task 3: Diagnostics in core and `ragfabric doctor`

**Files:**
- Create: `packages/core/src/ragfabric_core/diagnostics.py`, `ragfabric_cli/commands/doctor.py`
- Modify: `packages/core/src/ragfabric_core/db/migrate.py`, `ragfabric_cli/main.py` (register `doctor`)
- Test: `packages/core/tests/test_diagnostics.py`, `packages/cli/tests/test_doctor.py`

**Interfaces:**
- Produces (core):
  - `CheckResult(name: str, status: Literal["pass","warn","fail","skip"], detail: str, fix: str | None = None)` (frozen dataclass)
  - `MigrationsBehind(Exception)` with `current: str | None`, `head: str`
  - `check_python() -> CheckResult`, `check_config(path: Path | None) -> CheckResult`, `check_database(db_url: str) -> CheckResult`, `check_migrations(db_url: str) -> CheckResult`, `check_llm(cfg, *, network: bool) -> CheckResult`, `check_embeddings(cfg, *, network: bool) -> CheckResult`, `check_graph(cfg) -> CheckResult`, `check_server(url: str) -> CheckResult`, `run_all(*, config_path, db_url, server_url, network) -> list[CheckResult]`
  - `migrate.current_revision(db_url: str) -> str | None`, `migrate.head_revision() -> str`
- Produces (CLI): `ragfabric doctor [--url URL] [--no-network] [--json]`, exit 0 when no result is `fail`, else 1.

Rules: every `detail` that contains a URL goes through a masking function in core (`diagnostics.mask_url`, same behaviour as the CLI's; the CLI's `ui.console.mask_url` then imports it from core so there is one definition). A check that does not run reports `skip` with the reason, never `pass` (ADR 0004). `check_llm` with `network=True` makes one completion of at most 1 token through `build_llm_provider(cfg.llm)`; with the `offline` provider it returns `warn` with detail `offline mode: answers are extractive` and fix `install Ollama, or set OPENAI_API_KEY or ANTHROPIC_API_KEY, then mv ragfabric.yaml ragfabric.yaml.bak && ragfabric quickstart && ragfabric reindex --yes` (the R13 upgrade steps; never `--force`, which would re-copy .env and lose the key and JWT_SECRET; prefixed with `pip install 'ragfabric[openai]', then` while the openai extra is missing). `check_embeddings` embeds the word `ping` and fails when the vector length differs from `cfg.embeddings.dim`.

- [ ] **Step 1: Write the failing tests** (core): Python check passes on this interpreter; config check is `warn` with no file, `fail` on an invalid file naming the key, `pass` on a valid one; database check `pass` on a fresh SQLite URL and `fail` on `postgresql+psycopg://rf:s3cret@127.0.0.1:1/rf` with `s3cret` absent from detail and fix; migrations check `fail` before `db upgrade` and `pass` after on SQLite, using `current_revision`/`head_revision`; llm and embeddings checks `skip` with `network=False`, `warn` for offline, `fail` with a scripted provider that raises `ProviderError`; server check `warn` when nothing listens on a free port.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement** the core module and helpers (`current_revision` reads `MigrationContext.configure(conn).get_current_revision()`; `head_revision` reads `ScriptDirectory.from_config(alembic_config(...)).get_current_head()`), then the command, which converts each `CheckResult` to `CheckView` and calls `render_checks`, or prints `json.dumps([asdict(r) for r in results])` under `--json`.
- [ ] **Step 4:** CLI tests: `--json` output parses and the exit code is 1 when any check fails; plain output has one line per check with the status word and the fix; `--no-network` makes no provider call (assert with a provider double that raises if called).
- [ ] **Step 5: Commit** `feat: add ragfabric doctor with checks that live in core`

---

## Task 4: `ragfabric quickstart`

**Files:**
- Create: `ragfabric_cli/commands/quickstart.py`
- Modify: `ragfabric_cli/main.py` (register `quickstart`)
- Test: `packages/cli/tests/test_quickstart.py` (unit), `packages/cli/tests/test_quickstart_e2e.py` (real server, marked slow by the existing marker convention if one exists, otherwise a plain test)

**Interfaces:**
- Consumes: `templates.template_path`, `templates.sample_paths`, `templates.SAMPLE_QUESTION` (Task 2); `diagnostics.check_database`, `migrate.upgrade` (Task 3 and existing); `panels.render_answer`, `panels.render_next_steps` (Task 1); `ragfabric_sdk.Client`.
- Produces: `ragfabric quickstart [--dir PATH] [--docker] [--force] [--yes] [--model-check/--no-model-check]`; internal functions `choose_model(cfg_path, env) -> ModelChoice`, `free_port() -> int`, `temporary_server(dir, port) -> ContextManager[str]` (yields the base URL, always terminates the process).

| Step | Implementation |
|---|---|
| Config | Copy templates into `--dir` when missing; `--force` overwrites |
| Model | `ModelChoice(kind, llm, embeddings, reason)`. Probe `http://localhost:11434/api/tags` with a 2 s timeout (httpx); Ollama with a chat model and `nomic-embed-text` gives `ollama`; else `OPENAI_API_KEY` gives `openai`; else `ANTHROPIC_API_KEY` gives `anthropic` for the LLM with offline embeddings at the configured dim; else `offline` for both. Patch `ragfabric.yaml` (llm, embeddings, `cache.kind: memory` on SQLite) with a YAML round trip that keeps comments where possible, and print the reason |
| Database | SQLite: `DATABASE_URL=sqlite:///<dir>/ragfabric.db` in `.env`. `--docker`: `docker info` must succeed, else print how to start Docker and ask (or with `--yes`, use SQLite); write `docker-compose.yml`, run `docker compose up -d postgres redis`, wait for `pg_isready`-equivalent (connect with SQLAlchemy, 60 s), set `DATABASE_URL` |
| Migrate | `migrate.upgrade(db_url)` under a spinner |
| Ingest | The packaged samples through the same code path as `ragfabric ingest` (call its function), with progress |
| Ask | `temporary_server`: `subprocess.Popen([sys.executable, "-m", "uvicorn", "ragfabric_server.main:app", "--port", str(port)], cwd=dir, env=...)`, poll `/health` (30 s), `Client(url).ask(SAMPLE_QUESTION)` with no strategy, `render_answer(...)`, then terminate and wait in `finally` (kill after 5 s) |
| Next steps | `render_next_steps` with the five commands |

Every step checks whether it is already done (file exists, migrations at head, sample documents present by filename) and skips with "already done". No step prints `.env` contents or a secret.

- [ ] **Step 1: Write the failing unit tests:** `choose_model` for each of the four branches (Ollama double via `httpx.MockTransport` or monkeypatching the probe, env vars via `monkeypatch`); quickstart in `tmp_path` never overwrites an existing `ragfabric.yaml` without `--force` (compare file bytes); `free_port` returns a bindable port; `temporary_server` terminates the process when the body raises (assert `poll()` is not None afterwards); `--docker` with `docker info` failing and `--yes` falls back to SQLite and says so.
- [ ] **Step 2: Write the failing end-to-end test:** in `tmp_path`, with `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` unset and the Ollama probe forced to fail, `runner.invoke(app, ["quickstart", "--dir", str(tmp_path), "--yes"])` exits 0, the output contains the sample answer text and a `[1]` source, `ragfabric.db` exists, no process from the test is left listening (the port refuses a connection afterwards), and a second run exits 0 and reports steps as already done.
- [ ] **Step 3: Run them, confirm they fail.**
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run the tests, then the full CLI suite. Commit** `feat(cli): add ragfabric quickstart, from nothing to a cited answer`

---

## Task 5: Rich `ask` output

**Files:**
- Modify: `ragfabric_cli/commands/ask.py`, `ragfabric_cli/ui/panels.py` (rich branch only)
- Test: `packages/cli/tests/test_ask_rich.py`

**Interfaces:**
- Consumes: `panels.render_answer` (Task 1).

On a terminal: the streamed tokens print as they arrive (unchanged), then the rich block replaces the plain sources and routing lines: a `Sources` table (marker, document name, page, a 60-character snippet from `citation["text"]` or `citation["snippet"]` when present), the routing line as `Strategy  <name> (<source>). <reasoning>`, a `Graph` block of `name ─RELATION→ name` lines, and a dimmed `Removed (unsupported)` block for dropped claims. The non-stream path renders the answer inside a `Panel` titled `Answer`. Piped output and `--json` are unchanged.

- [ ] **Step 1: Write the failing tests:** with `is_rich` forced True and a fake SDK client (the existing `test_ask_auto.py` fakes), the output contains `Answer`, `Sources`, the document name, `Strategy`, and for a graph answer `─REPORTS_TO→`; with `is_rich` False, the output is byte-identical to the output of the same invocation on `main` before this task (store the expected strings in the test); `--json` output parses and equals the SDK model dump.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement** the rich branch in `panels.py` and keep `ask.py` free of formatting.
- [ ] **Step 4: Run the new tests and every existing `test_ask_*` and `test_cli_*` test.**
- [ ] **Step 5: Commit** `feat(cli): render ask answers as a panel with sources and the routing decision`

---

## Task 6: Ingest progress

**Files:**
- Modify: `ragfabric_cli/commands/ingest.py`
- Test: `packages/cli/tests/test_ingest_progress.py`

**Interfaces:**
- Produces: `ingest_files(files, *, collection, owner, on_file: Callable[[Path, Document], None] | None = None) -> tuple[int, int]` (ingested, failed), extracted from `ingest` so Task 4 can call it.

On a terminal, a `rich.progress.Progress` bar with the file name and a count; one line per failed file printed under the bar with its error. Piped output keeps today's per-file lines and the summary line exactly.

- [ ] **Step 1: Write the failing tests:** plain output for a two-file directory is byte-identical to today's (`<name>: <status> (<n> chunks)` lines and `2 ingested, 0 failed`); with `is_rich` forced True the run completes and prints the summary; `ingest_files` returns the counts and calls `on_file` once per file; a failing file still exits 1.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the new tests and the existing ingest tests.**
- [ ] **Step 5: Commit** `feat(cli): show ingest progress on a terminal`

---

## Task 7: `ragfabric strategies`, the welcome screen, and examples in `--help`

**Files:**
- Create: `ragfabric_cli/commands/strategies.py`
- Modify: `ragfabric_cli/main.py` (callback and existing command docstrings), every module in `ragfabric_cli/commands/` that defines a command Track A does not own (`ask`, `ingest`, `users`, `access`, `graph`, `worker`, `reindex`, `reconcile`)
- Test: `packages/cli/tests/test_strategies.py`, `test_welcome.py`, `test_help_examples.py`

**Interfaces:**
- Consumes: `panels.render_strategies`, `StrategyRow` (Task 1).
- Produces: `strategies.ROWS: list[StrategyRow]`.

`ROWS` has five rows in this order: `auto`, `traditional`, `vectorless`, `agentic`, `graph`. The example questions are imported from the router's own signal test table where possible; where a module import from tests is not allowed, the strings are copied and a test asserts that `propose(extract_signals(example, relation_types=...))` picks that row's strategy (for `auto`, any). The welcome screen replaces `no_args_is_help=True` with an `invoke_without_command=True` callback that, when no subcommand is given, prints the version and three starting commands (`quickstart`, `doctor`, `strategies`) and the docs link, exit 0. Every command's help text gains an `Examples:` block in its docstring (Typer prints docstrings), with commands that work after `quickstart`.

- [ ] **Step 1: Write the failing tests:** `ragfabric strategies` prints all five names and their examples, plain and rich; each example routes to its row's strategy through the real signals; bare `ragfabric` exits 0 and mentions `quickstart`, `doctor` and `strategies`; for every registered command (walk `app.registered_commands` and sub-apps) `--help` contains `Examples:`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** For the help-examples test, commands added by Track A (`doctor`, `quickstart`, `init`) are covered when the tracks merge; until then the test skips names not yet registered rather than failing, and the merge step removes that skip.
- [ ] **Step 4: Run the new tests and the full CLI suite.**
- [ ] **Step 5: Commit** `feat(cli): add ragfabric strategies, a welcome screen and examples in every help`

---

## Task 8: Documentation

**Files:**
- Modify: `README.md` (a "Five minute start" section at the top of getting started: `pip install ragfabric`, `ragfabric quickstart`, `ragfabric doctor`), `docs/getting-started.md`, `docs/troubleshooting.md` (the error table from the design, each with its fix), `ROADMAP.md` (tick the Phase 7 terminal experience items, mark Phase 7 `[x]`), `CHANGELOG.md` (`[0.4.0] - unreleased` Added entries for 7b; `Changed`: `ragfabric init` no longer needs the example files in the current directory; bare `ragfabric` shows a welcome screen instead of the help)
- Test: none beyond a docs check: every command quoted in the new README and getting-started sections exists (a test that runs `--help` for each quoted command)

- [ ] **Step 1:** Write the docs-commands test, confirm it fails on a deliberately misspelt command, fix the spelling.
- [ ] **Step 2:** Write the docs. No em dashes; no quality claims.
- [ ] **Step 3: Commit** `docs: the five minute start, troubleshooting and the 7b changelog`

---

## Task 9: Verification and the pull request

- [ ] **Step 1:** Full suite with PostgreSQL; record passed and skipped against the base.
- [ ] **Step 2:** `ruff check packages/`, `ruff format --check packages/`, `lint-imports` (3 kept, 0 broken).
- [ ] **Step 3:** A real `pip install` check: build the CLI wheel and its local dependencies, install them into a fresh `uv venv` in `/tmp`, run `ragfabric quickstart --yes` in an empty directory with no model and no Docker, and record the output in the report. This is the user this phase is for.
- [ ] **Step 4:** Grep the branch's commit messages and diff for em dashes and assistant references.
- [ ] **Step 5:** Whole-branch review, one fix wave, push `feat/phase-7b-terminal`, open a PR into `main` that says v0.4.0 is tagged after it merges. **Stop there.**

---

## Parallel execution

Task 1 lands first, by the controller, because every renderer and both tracks depend on it.

| Track | Tasks | Area | Worktree and branch |
|---|---|---|---|
| A, setup | 2, 3, 4 | templates and `init`, doctor, quickstart | `~/AI/ragfabric-wt/phase-7b-setup`, `feat/phase-7b-track-setup` |
| B, output | 5, 6, 7 | ask, ingest progress, strategies, welcome, help examples | `~/AI/ragfabric-wt/phase-7b-output`, `feat/phase-7b-track-output` |

Task 4 consumes Task 6's `ingest_files`. Until Track B lands, Task 4 calls the existing `ingest_document` loop through a small local helper with the same signature, and the merge replaces it with `ingest_files`. Both tracks touch `main.py` in different functions (`init` and command registration in A; the callback and docstrings in B), so expect a small hand-merged conflict. Tasks 8 and 9 run after the merge, one implementer at a time. Track A uses the test database `ragfabric_test`, Track B `ragfabric_test_b`.

---

## Self-Review

Before the phase is called done:

- Does any piped or `--json` output contain an ANSI escape or differ from before?
- Can any message print a password or an API key?
- Can `quickstart` leave a server running, or overwrite a user's file without `--force`?
- Does `doctor` ever report `pass` for a check it did not run?
- Does a first-time user with no Docker and no model reach a cited answer from `pip install`?
- Is every command quoted in the docs real?
