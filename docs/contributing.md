# Contributing

The full guide, with the pull request flow, what will and will not be merged, and the ground rules,
is [CONTRIBUTING.md](https://github.com/ranjan-del/ragfabric/blob/main/CONTRIBUTING.md) in the
repository. Everyone taking part agrees to the
[Code of Conduct](https://github.com/ranjan-del/ragfabric/blob/main/CODE_OF_CONDUCT.md), and a first
pull request asks for the [CLA](licensing.md#contributor-license-agreement) once.

## The short version

1. Pick an open issue, or open one before a large change.
2. Fork, branch (`feat/...`, `fix/...`, `docs/...`), and open a pull request against `main`.
3. Write the test first and see it fail. Keep CI green: backend tests, migrations on PostgreSQL,
   frontend tests and build.
4. Update every surface the change touches in the same pull request: README, ROADMAP, CHANGELOG
   and the pages on this site.

No fabricated numbers, no placeholder presented as finished, access control inside retrieval,
secrets never in the repository.

## Development setup

```bash
uv sync                                   # Python 3.13 workspace into .venv
uv run pytest                             # every package's tests
uv run ruff check packages && uv run lint-imports
cd apps/assistant && npm ci && npm test   # web app specs (needs Chrome)
docker compose up --build                 # the development stack
```

## Working on this site

The site is built from `docs/` with Material for MkDocs. Every page is plain Markdown, readable on
GitHub too; diagrams are Mermaid code blocks.

```bash
uv sync --group docs
uv run mkdocs serve             # http://127.0.0.1:8000, reloads on save
uv run mkdocs build --strict    # what the pull request check runs: a broken link fails it
```

Add a new page to `nav` in `mkdocs.yml`; strict mode fails on a page left out. Plans and design
documents (`docs/plans/`, `docs/design/`) stay in the repository and are not published.

Releases are described in [Releasing](releasing.md).
