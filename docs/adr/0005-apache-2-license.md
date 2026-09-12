# ADR 0005: Apache License 2.0

Date: 2026-09-13. Status: accepted (replaces MIT from v1).

## Context
The project is meant to be adopted and modified by companies. MIT has no explicit patent grant, which
some legal teams flag.

## Decision
Relicense under Apache 2.0 before any external contribution lands. Add a NOTICE file. Contributors sign
an Individual CLA (modelled on the Apache Software Foundation ICLA) once, through a free GitHub Action
that blocks review until every author of a pull request has signed. The CLA grants the same Apache 2.0
terms; it does not transfer copyright.

## Consequences
Attribution and NOTICE requirements for redistributors. Compatible with the permissive dependencies used.
First time contributors post one sentence on their PR; signatures live in the `cla-signatures` branch.
Explained for adopters in docs/licensing.md.
