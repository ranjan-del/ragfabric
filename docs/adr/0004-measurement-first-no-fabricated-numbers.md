# ADR 0004: Measurement first, no fabricated numbers

Date: 2026-09-13. Status: accepted.

## Context
RAG architecture advice is mostly anecdotal. The value of this project is that its claims are measured.

## Decision
Every benchmark figure in the repository is produced by `make eval` from the shipped corpus and questions,
persisted to the database, and written to `docs/benchmarks/latest.md` with commit hash, models and date.
Cost is always labelled an estimate from configured pricing. The complexity score is labelled an
engineering assessment. Hand typed numbers are rejected in review.

## Consequences
Docs lag code by one evaluation run. Contributors need provider keys to regenerate numbers; CI runs the
metric code on fixtures, not the paid benchmark.
