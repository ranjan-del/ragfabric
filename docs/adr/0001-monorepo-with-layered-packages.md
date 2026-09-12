# ADR 0001: Monorepo with layered packages

Date: 2026-09-13. Status: accepted.

## Context
Adopters must be able to keep the engine and replace the UI, or use the SDK from their own system, or
run only the CLI. Four separate repositories would make the shared result types drift.

## Decision
One repository. `packages/core` has no HTTP or UI dependency. `packages/server` depends on core.
CLI and SDKs are separate packages. UIs live in `apps/` and talk only to the SDK. Dependency direction
is enforced by import linting in CI.

## Consequences
Bigger repository, one CI. Deleting `apps/assistant` breaks nothing else. A new store or provider is a
new module implementing one interface plus tests.
