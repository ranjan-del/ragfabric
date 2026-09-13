# Contributing to RagFabric

Thank you for considering a contribution. This project is built in the open, phase by phase, and every
phase is tracked as a GitHub issue under a milestone. The fastest way to help is to pick an open issue.

## How contributions flow

1. Fork the repository. Nobody pushes to `main` directly, including the maintainer.
2. Create a branch in your fork: `feat/<area>-<short-name>`, `fix/...`, `docs/...`.
3. Open a pull request against `main` using the PR template. Link the roadmap issue it advances.
4. CI must pass (backend tests, migration round trip against PostgreSQL, frontend tests and build).
5. On your first pull request, sign the [CLA](CLA.md) by posting the sentence the CLA bot asks for. It is a one time step.
6. A maintainer reviews. Small, focused PRs are reviewed fastest.

`main` is protected: pull request required, status checks required, no force pushes, linear history.

## What we will and will not merge

| Welcome | Please open an issue first |
|---|---|
| A new implementation of an existing interface (a `VectorStore`, an `LLMProvider`, a `Connector`) with tests | A new interface or a change to a shared result type |
| Bug fixes with a failing test that now passes | Anything that adds a required external service |
| Documentation, learning notes, diagrams, corrections | A new retrieval strategy (discuss the design in an ADR first) |
| Evaluation questions and corpus improvements | Changes to the access control model |

## Ground rules that keep the project honest

- **No fabricated numbers.** Any benchmark figure in docs or README must come from a real `make eval` run,
  be reproducible from the shipped corpus and questions, and say which commit and which models produced it.
- **No placeholder features presented as complete.** If something is partial, mark it `TODO` in code and
  say so in docs.
- **Access control is enforced inside retrieval, never after.** A change that filters sources after ranking
  will not be merged.
- **Secrets never enter the repo.** Use environment variables and `.env.example`.
- **Every strategy returns the common `RetrievalResult`.** The rest of the system must not need to know
  which strategy ran.

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

## Commit messages

Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`. One logical change per commit.

## Learning notes

This project is also a teaching codebase. When you add something non-obvious, add a short "why" in the
module docstring or in `docs/`. Explaining a trade-off you rejected is as valuable as the code.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
