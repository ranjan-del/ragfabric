# ADR 0002: Every strategy returns the same RetrievalResult

Date: 2026-09-13. Status: accepted.

## Context
Comparison across strategies is the product. It is only fair if generation, citation, metrics and
evaluation run identical code over identical shapes.

## Decision
`RetrieverStrategy.retrieve(query, ctx) -> RetrievalResult`. The result carries chunks with source
metadata, counts of retrieval and LLM calls, tokens, latency, a trace and an optional `fallback_from`.
Generation never inspects which strategy produced the result.

## Consequences
Agentic and Graph strategies must fold their internal steps into the trace rather than returning bespoke
objects. Adding a strategy is one class plus registration.
