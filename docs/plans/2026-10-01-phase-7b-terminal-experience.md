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

## Tasks

To be written once this design is approved.
