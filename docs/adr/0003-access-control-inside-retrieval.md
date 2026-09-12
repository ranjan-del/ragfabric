# ADR 0003: Access control is enforced inside retrieval

Date: 2026-09-13. Status: accepted.

## Context
Filtering sources after ranking leaks restricted text into the model context and can leak it into the
answer even if the citation is hidden.

## Decision
The permitted document set for the principal is computed once per request and passed as a filter into
every store query (vector, lexical, graph). Ranking only sees permitted chunks. The policy lives in core,
not in the server or UI, so replacing the UI cannot remove it. Filtered counts are recorded in the audit
log.

## Consequences
Every store interface takes an access filter argument. Stores that cannot filter natively must
pre-filter candidate ids. Slightly more work per store, no leak path.
