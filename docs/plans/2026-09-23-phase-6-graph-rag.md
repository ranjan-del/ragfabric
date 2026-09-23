# Phase 6: Graph RAG, release v0.3.0. Implementation Plan

**Goal:** Answer questions whose answer is not in any single passage but in the relationships between passages, and do it without ever asserting a relationship that no source sentence supports.

**Architecture:** A knowledge graph built from chunks at ingest and traversed with recursive CTEs in PostgreSQL. Every entity and edge carries a confidence and the chunk ids it came from. The access filter runs inside the traversal, on node matching as well as edge walking. Relationship claims in the answer are verified against their edge and that edge's source chunk, and dropped when unsupported.

**Tech Stack:** Python 3.13 (uv), SQLAlchemy 2, Alembic, PostgreSQL 18, FastAPI, Pydantic. **No new runtime dependencies, and no Neo4j.**

**Issue:** https://github.com/ranjan-del/ragfabric/issues/7

**Predecessor:** `docs/plans/2026-09-22-phase-5-agentic-rag.md` (Agentic RAG, merged as PR #39)

---

## Why this shape, and what was rejected

Recorded here because the reasoning is the deliverable, not just the code.

### No Neo4j, and the spike that decided it

The roadmap, issue #7, `docs/graph-rag.md` and the compose `full` profile all name Neo4j. A spike built the traversal against real PostgreSQL and measured the alternative.

| Criterion | PostgreSQL recursive CTE | Neo4j |
|---|---|---|
| Python dependencies | 0, already present | 2 (`neo4j`, `pytz`), which is light and was not the issue |
| Extra servers to run, back up and secure | 0 | 1 |
| **Access filter inside the traversal** | **Yes, proven by a real run** | **No, not without duplicating the access model** |
| Traversal code | about 30 lines of SQL | about 5 lines of Cypher |

The third row decided it. The spike seeded two collections, made one of them denied, and ran a 3-hop traversal twice:

```
sees BOTH collections           -> ravi sharma(0), platform team(1), cto(2), skunkworks(3)
sees ONLY the public collection -> ravi sharma(0), platform team(1), cto(2)
```

The restricted caller does not receive `skunkworks` filtered out of a result. The path to it never exists, because the access predicate sits inside the recursive term and the forbidden edge is never walked.

Neo4j cannot do that directly, because `chunks`, `documents` and `collections` live in PostgreSQL and Neo4j has no knowledge of them. That leaves three options and all three are bad: duplicate the chunk to document to collection mapping into the graph, which gives the access model two sources of truth that will drift; fetch every allowed chunk id from PostgreSQL and pass them into Cypher as a parameter, which has a scaling wall on a real corpus; or traverse unfiltered and filter afterwards, which is what `docs/graph-rag.md` currently describes and what ADR 0003 forbids.

**In a graph, filtering after traversal leaks more than it does in a search.** Knowing that `A REPORTS_TO B` is information even when the sentence that said so is never shown. A traversal that walks edges the caller cannot see hides the text and reveals the shape.

This is the third dependency rejected on the same principle: `pg_search` in ADR 0007, LangGraph in ADR 0009, Neo4j here. Recorded as **ADR 0011**. `ROADMAP.md`, issue #7, `docs/graph-rag.md` and the compose `full` profile are all corrected, because they currently promise it.

Honest about the evidence: the PostgreSQL side was run against a real database. The Neo4j side was analysed rather than run, because the conclusion follows from where the access data lives and not from Neo4j's performance.

### What was wrong with the committed concept design

`docs/graph-rag.md` describes the textbook shape. Six weaknesses, each closed here:

| Weakness | Consequence | Closed by |
|---|---|---|
| Extraction is trusted blindly | An LLM emits an edge and it becomes a fact. A hallucinated relationship is indistinguishable from a real one, forever | Task 3, confidence with a floor |
| Entity resolution is a permanent invisible guess | Merge "R. Sharma" with "Ravi Sharma" wrongly and two people are fused. The graph asserts falsehoods and nothing detects it | Task 5, merges recorded as reversible decisions with their evidence |
| No graph-level citation contract | Phase 3 verifies claims against retrieved chunks. A graph answer asserts relationships, and "A reports to B" can be a traversal artifact no sentence stated | Task 10, the graph citation contract |
| Blind k-hop traversal | Walking two hops from every matched node explodes combinatorially and returns noise | Task 7, relation-type filter and a node budget |
| Access filter applied after traversal | Step 4 of the concept doc's query flow. Leaks graph shape, as above | Task 6, the filter inside the traversal |
| Silence on coverage | A corpus with few named entities gains nothing from a graph and nothing says so | Task 9, the honest empty-graph path |

### Four more found during the spike, two of them bugs in the spike itself

**Traversal direction.** `REPORTS_TO` is directed. The spike walked edges in both directions, which is a correctness bug: traverse `B REPORTS_TO A` backwards, generate "A reports to B", and a fact the corpus never stated has been fabricated. Task 6 respects direction, and distinguishes relations that are meaningfully invertible (`BELONGS_TO` and `CONTAINS` are the same edge read two ways) from those that are not.

**The access filter must apply to entity matching, not only to edges.** The spike filtered edges alone. Matching a question entity to a node reveals the node exists, so if the only chunk mentioning an entity is denied, the match itself must fail. Same class of leak, and the spike missed it.

**Re-extraction cost.** One LLM call per chunk is the dominant cost of this whole strategy, and without a content hash every re-ingest pays for the entire corpus again. Task 4 stores a hash and skips unchanged chunks. This is not prompt-versioned re-extraction, which stays out of scope; it is only "do not pay twice for a chunk that did not change".

**The empty case.** When no question entity matches, the strategy must return nothing and say why rather than returning a traversal of whatever was nearest. Task 9.

### Deliberately out of scope

**Community summaries.** Corpus-wide questions ("what are the main themes across all project notes") need a different mechanism, and it cannot be tuned or validated without measurement, which is Phase 8. The concept doc already defers them and that stands.

**Prompt-versioned re-extraction.** Task 4 takes the cheap 80% with a content hash. Rebuilding the graph because the extraction prompt changed remains a manual re-ingest.

**A console UI for merge review.** Task 5 ships the data model and the API so a merge can be inspected and undone. The screen belongs to Phase 9.

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.13**, line length 100, `ruff` clean (`E`, `F`, `I`, `UP`, `B`), `ruff format` clean.
- **Run `ruff format packages/`, never `ruff format .`** The repo-root form rewrites Python code blocks inside markdown and silently mangled six documents in Phase 4.
- **import-linter contracts stay at 3 kept, 0 broken.**
- **No new runtime dependencies.** If a task appears to need one, stop and say so rather than adding it.
- **ADR 0003 holds, and it is the hardest constraint in this phase.** The access filter goes inside the traversal query, applied to node matching and to edge walking. A graph that filters after the walk leaks structure.
- **ADR 0004 holds: never fabricate a number.** An extraction confidence is what the model reported, never a default that looks plausible. Unknown is `None`.
- **ADR 0002 holds: one result shape.** The strategy returns `RetrievalResult`.
- **Citations reuse the Phase 3 contract** (`generate/contract.py`) for chunk-grounded claims, unchanged. The graph contract in Task 10 is additive and covers relationship claims, which the Phase 3 contract cannot see.
- **SQLite remains the test default.** Recursive CTEs work on both, but anything PostgreSQL-only is skipped via the `RAGFABRIC_TEST_DATABASE_URL` guard, never by silently passing. **Baseline on this branch is 746 passed and 8 skipped** with that variable set.
- **No AI attribution anywhere.** Commits authored `Ranjan G <ranjan.g@ispf.ngo>`, no trailers, no assistant references in code, comments, docs or PR bodies.
- **No em dashes** anywhere.
- **Tests must be able to fail.** A test that passes against a deliberately broken implementation is not a test.

---

## File Structure

| File | Responsibility |
|---|---|
| `packages/core/src/ragfabric_core/migrations/versions/0008_graph_confidence_and_resolution.py` | Confidence and content hash columns, `entity_merges`, direction metadata |
| `packages/core/src/ragfabric_core/graph/contracts.py` | JSON schemas for extracted entities and relationships, with confidence |
| `packages/core/src/ragfabric_core/graph/extract.py` | Per-chunk extraction, confidence floor, content-hash skipping |
| `packages/core/src/ragfabric_core/graph/resolve.py` | Normalisation, alias merge, embedding tie-break, reversible merge records |
| `packages/core/src/ragfabric_core/graph/traverse.py` | Directed recursive-CTE traversal with the access filter inside |
| `packages/core/src/ragfabric_core/graph/citations.py` | The graph citation contract |
| `packages/core/src/ragfabric_core/strategies/graph.py` | `GraphRAGStrategy` |
| `docs/concepts/knowledge-graphs.md` | Entities, relations, traversal, and where extraction goes wrong |
| `docs/adr/0011-postgres-recursive-cte-over-neo4j.md` | The graph store decision |
| `docs/adr/0012-extraction-confidence-and-graph-citations.md` | Why an unverified edge is not a fact |

---

## Task 1: Graph schema and migration 0008

**Files:** Create the migration; modify `models/graph.py`; test `packages/core/tests/test_models_graph_confidence.py`

Foundation. Both tracks build on it, so it lands before either starts.

`entities` and `relationships` gain `confidence: float | None` and `extraction_model: str | None`. `chunks` gains `extraction_hash: str | None`, the hash of the chunk text that was last extracted from, so Task 4 can skip unchanged work. A new `entity_merges` table records each resolution decision: the surviving entity, the merged-away name and type, the evidence that justified it, the model and method, and a timestamp, so a merge can be inspected and undone.

`confidence` is nullable on purpose. A row written before this phase has no confidence, and `None` says "not measured" where `0.0` would say "measured as worthless". ADR 0004.

- [ ] **Step 1: Write the failing tests.** `test_confidence_defaults_to_none_not_zero`, `test_a_merge_record_requires_its_evidence`, `test_extraction_hash_starts_null`, `test_confidence_outside_zero_to_one_is_rejected`.
- [ ] **Step 2: Run them, confirm they fail** with `ImportError` or a missing column.
- [ ] **Step 3: Implement** the models and the migration.
- [ ] **Step 4: Verify the migration upgrades and downgrades cleanly** against real PostgreSQL.
- [ ] **Step 5: Commit** `feat: add graph confidence, extraction hashes and merge records`

---

## Task 2: Extraction contracts

**Files:** Create `graph/contracts.py`; test `packages/core/tests/test_graph_contracts.py`

The same discipline as Phase 5's `agent/contracts.py`, and it reuses that module's JSON extraction helpers rather than duplicating them. `ExtractionResponse` carries entities (name, type, description, confidence) and relationships (source, target, relation type, description, confidence). Entity types and relation types are **fixed enums**, so a model cannot invent a relation the traversal has no rules for.

Forgiving about packaging, strict about content: a fenced code block or surrounding prose is tolerated, a missing confidence or an unknown relation type is a `ContractViolation`.

- [ ] **Step 1: Write the failing tests**, including `test_an_unknown_relation_type_is_a_violation`, `test_a_missing_confidence_is_a_violation_not_a_default`, `test_fenced_json_is_accepted`, `test_an_edge_naming_an_entity_not_in_the_list_is_a_violation`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**, reusing the Phase 5 JSON helpers.
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add validated extraction contracts for entities and relationships`

---

## Task 3: Extraction with a confidence floor

**Files:** Create `graph/extract.py`; test `packages/core/tests/test_graph_extract.py`

One LLM call reads a chunk with the type schema and returns structured JSON. Everything below the configured confidence floor is **discarded rather than stored**, and the count of discarded items is reported.

**The floor is the point of this task.** Without it the graph silently accumulates the model's guesses as facts, and a hallucinated edge becomes indistinguishable from a real one the moment it is written. Reporting how many were dropped is what lets an operator see that their corpus or model is producing junk.

Every stored entity and edge records its `source_chunk_ids`, its `confidence` and the `extraction_model` that produced it. A wrong edge must be traceable to the sentence and the model that produced it.

- [ ] **Step 1: Write the failing tests**, including `test_an_edge_below_the_floor_is_not_stored`, `test_the_number_discarded_is_reported`, `test_every_stored_item_records_its_source_chunk_and_model`, `test_a_contract_violation_stores_nothing_from_that_chunk`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Mutation check.** Change the floor comparison from `>=` to `>` on a fixture whose confidence equals the floor exactly, confirm the boundary test fails, revert.
- [ ] **Step 5: Commit** `feat: extract entities and relationships behind a confidence floor`

---

## Task 4: Skip re-extraction of unchanged chunks

**Files:** Modify `graph/extract.py`; test `packages/core/tests/test_graph_extract_incremental.py`

Before extracting, hash the chunk text and compare it to `chunks.extraction_hash`. Unchanged means skip, and the skip is counted and reported.

One LLM call per chunk is the dominant cost of this entire strategy. Without this, re-ingesting a corpus to fix one document pays for every chunk in it again.

A changed chunk must also **remove the entities and edges it previously contributed** before re-extracting, or stale assertions from the old text survive forever. This is the subtle half of the task.

- [ ] **Step 1: Write the failing tests**, including `test_an_unchanged_chunk_is_not_re_extracted`, `test_a_changed_chunk_is_re_extracted`, `test_re_extraction_removes_the_edges_the_old_text_produced`, `test_the_number_skipped_is_reported`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Mutation check.** Remove the stale-edge cleanup and confirm `test_re_extraction_removes_the_edges_the_old_text_produced` fails. Revert.
- [ ] **Step 5: Commit** `feat: skip extraction for chunks whose text has not changed`

---

## Task 5: Entity resolution, recorded and reversible

**Files:** Create `graph/resolve.py`; test `packages/core/tests/test_graph_resolve.py`

Three stages, in order of confidence: exact match on normalised name and type; alias match; then embedding similarity between names and descriptions above a configured threshold as a tie-break.

**Every merge writes an `entity_merges` row carrying the evidence that justified it**, and an unmerge restores the separated entity with the edges that came from its chunks.

Entity resolution is the main failure mode of graph RAG. Merge too aggressively and two people named Sharma become one node, and the graph now asserts things about a person who does not exist. Merge too little and the graph fragments, so a traversal that should connect two facts finds no path. Neither can be tuned without measurement, which is Phase 8, so what this phase owes is not a perfect threshold but a **reversible** decision with its reasoning attached.

- [ ] **Step 1: Write the failing tests**, including `test_an_exact_normalised_match_merges`, `test_an_alias_match_merges`, `test_similarity_below_the_threshold_does_not_merge`, `test_a_merge_records_the_evidence_that_justified_it`, `test_unmerging_restores_the_entity_and_its_edges`, `test_entities_of_different_types_never_merge`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Mutation check.** Drop the entity-type check from the merge predicate and confirm `test_entities_of_different_types_never_merge` fails. Revert.
- [ ] **Step 5: Commit** `feat: resolve entities with recorded, reversible merges`

---

## Task 6: Directed traversal with the access filter inside

**Files:** Create `graph/traverse.py`; test `packages/core/tests/test_graph_traverse.py`

A recursive CTE from matched nodes outward. Two things must be right, and both were wrong in the spike.

**Direction.** `REPORTS_TO` is directed and traversing it backwards asserts the opposite fact. A fixed table declares which relation types are invertible and under what inverse name (`BELONGS_TO` read backwards is `CONTAINS`); the rest are walked forwards only. Generating "A reports to B" from an edge that said the reverse is a fabricated fact, and no downstream contract can catch it because the edge really does exist.

**Access, on nodes as well as edges.** The predicate sits inside the recursive term, so a forbidden edge is never walked, and it also applies to the initial node match, because matching an entity reveals it exists. If the only chunk mentioning an entity is denied, the match must fail.

- [ ] **Step 1: Write the failing tests**, including `test_a_denied_edge_is_never_walked`, `test_a_denied_entity_does_not_match`, `test_a_directed_relation_is_not_walked_backwards`, `test_an_invertible_relation_is_walked_both_ways_under_its_inverse_name`, `test_a_cycle_terminates`, `test_the_hop_limit_is_respected`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Mutation check, twice.** Remove the access predicate from the recursive term and confirm `test_a_denied_edge_is_never_walked` fails. Then make the walk undirected and confirm `test_a_directed_relation_is_not_walked_backwards` fails. Revert both.
- [ ] **Step 5: Verify against real PostgreSQL** with `RAGFABRIC_TEST_DATABASE_URL` set, and confirm the SQLite path passes too.
- [ ] **Step 6: Commit** `feat: traverse the graph with direction and the access filter inside the query`

---

## Task 7: Question-guided traversal with a node budget

**Files:** Modify `graph/traverse.py`; test `packages/core/tests/test_graph_guided.py`

Two bounds on the walk. The **node budget** is a hard cap on the frontier, and it is the safety property: without it a dense graph makes a two-hop walk return most of the corpus. The **relation-type filter** narrows the walk to the types the question implies, and it is worth having only because the question's entities are already being extracted, so the implied types come from the same call rather than a new one.

When the budget truncates the walk, the result says so. A silently truncated traversal looks identical to a complete one.

- [ ] **Step 1: Write the failing tests**, including `test_the_node_budget_caps_the_frontier`, `test_truncation_is_reported`, `test_only_the_implied_relation_types_are_walked`, `test_no_implied_types_falls_back_to_all_types`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Mutation check.** Remove the truncation flag and confirm `test_truncation_is_reported` fails. Revert.
- [ ] **Step 5: Commit** `feat: bound traversal by relation type and a node budget`

---

## Task 8: GraphRAGStrategy

**Files:** Create `strategies/graph.py`; modify `strategies/registry_defaults.py`; test `packages/core/tests/test_strategy_graph.py`

Implements `RetrieverStrategy`: `retrieve(query, ctx) -> RetrievalResult`. Extracts the question's entities, matches them under the access filter, traverses, collects the source chunks of every node and edge on the path, and returns them as `chunks` with the sub-graph rendered into the trace. Registered as `graph`.

Counters report what happened: `llm_calls` is the question-extraction call, `embedding_calls` is whatever the similarity tie-break actually spent and zero when it was not needed.

- [ ] **Step 1: Write the failing tests**, including `test_counters_report_actual_calls`, `test_the_access_filter_reaches_the_traversal`, `test_chunks_come_from_nodes_and_edges_on_the_path`, `test_the_strategy_is_resolvable_from_the_registry`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement and register.**
- [ ] **Step 4: Run the full suite** and confirm no Phase 3, 4 or 5 strategy test regressed.
- [ ] **Step 5: Commit** `feat: add GraphRAGStrategy over the directed traversal`

---

## Task 9: The honest empty-graph path

**Files:** Modify `strategies/graph.py`; test `packages/core/tests/test_graph_empty.py`

Three distinct empty cases, each reported as itself rather than as a generic no-result: no entity in the question matched any node; entities matched but no edges were walkable; the graph holds nothing for this corpus at all.

A corpus with few named entities gains nothing from a graph. Returning the nearest traversal of whatever happened to be there is how a strategy manufactures false relevance, and it is worse than returning nothing with a reason.

- [ ] **Step 1: Write the failing tests**, including `test_no_matched_entity_reports_that_specifically`, `test_matched_but_isolated_entities_report_no_walkable_edges`, `test_an_empty_graph_reports_no_coverage`, `test_no_chunks_are_returned_when_nothing_matched`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Commit** `feat: report why a graph query found nothing`

---

## Task 10: The graph citation contract

**Files:** Create `graph/citations.py`; modify `generate/cited.py`; test `packages/server/tests/test_graph_citations.py`

The distinctive piece of this phase. The Phase 3 contract verifies a claim against retrieved chunk text and is reused unchanged for that. It cannot see relationship claims, because the sentence "the platform team reports to the CTO" may appear in no chunk while being a true reading of two edges.

So a relationship claim must cite **the edge** and the edge's **source chunk**. A claim that names a relationship absent from the traversed sub-graph is dropped, and so is one whose edge exists but whose source chunk does not support it. Both removals are recorded.

**The failure this prevents:** a traversal produces a path, the model narrates the path as prose, and the prose asserts a relationship no source sentence ever stated. The edge is real, the traversal is correct, and the claim is still fabricated. Nothing in Phase 3 or Phase 5 can catch that.

- [ ] **Step 1: Write the failing tests**, including `test_a_claim_about_an_edge_not_in_the_subgraph_is_dropped`, `test_a_claim_whose_edge_exists_but_whose_chunk_does_not_support_it_is_dropped`, `test_a_supported_relationship_claim_survives_with_both_citations`, `test_a_dropped_claim_is_recorded`, `test_the_phase_3_contract_is_still_applied_to_chunk_claims`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Mutation check.** Accept any claim whose relation type appears anywhere in the graph rather than in the traversed sub-graph, and confirm the first test fails. Revert.
- [ ] **Step 5: Commit** `feat: verify relationship claims against their edge and its source chunk`

---

## Task 11: Ingestion wiring and configuration

**Files:** Modify the ingestion path, `config_file.py`, `ragfabric.example.yaml`; tests

Extraction runs during ingest, behind a config switch so a deployment that does not want a graph pays nothing. Typed settings: the confidence floor, the similarity threshold for resolution, `max_hops`, the node budget, the enabled entity and relation types, and the extraction model. Strict validation, extra keys rejected.

**Every setting must be read by something.** Phase 5 shipped four typed limits that nothing enforced; a test asserts each key here reaches the code that uses it.

- [ ] **Step 1: Write the failing tests**, including `test_an_unknown_graph_key_is_rejected`, `test_a_confidence_floor_outside_zero_to_one_is_rejected`, `test_every_graph_setting_reaches_the_code_that_uses_it`, `test_graph_extraction_is_skipped_when_disabled`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**, documenting every key with a one-line comment in `ragfabric.example.yaml`.
- [ ] **Step 4: Verify `ragfabric config validate` reports the graph settings.**
- [ ] **Step 5: Commit** `feat: wire graph extraction into ingestion behind typed configuration`

---

## Task 12: API, CLI and SDK surface

**Files:** Server schemas and routes, `packages/cli`, the SDK; tests

Add `graph` to the strategy choices on `POST /api/ask` and `POST /api/search/query`, to `ragfabric ask --strategy graph`, and to the SDK. `POST /api/search/hybrid` continues to accept only `traditional`. The traversed sub-graph and any dropped relationship claims come back on the ask response. A CLI command to inspect and undo a merge, since Task 5's records are useless without a way to reach them.

- [ ] **Step 1: Read the current request and response schemas and write down their real fields.**
- [ ] **Step 2: Write the failing tests**, including `test_the_ask_response_carries_the_subgraph`, `test_hybrid_still_refuses_graph`, `test_the_cli_can_undo_a_merge`.
- [ ] **Step 3: Run them, confirm they fail.**
- [ ] **Step 4: Implement across route, CLI and SDK.**
- [ ] **Step 5: Run the full suite** and confirm no earlier route test regressed.
- [ ] **Step 6: Commit** `feat: expose the graph strategy over the API, CLI and SDK`

---

## Task 13: A real extraction run against Ollama

**Files:** `packages/core/tests/test_graph_integration.py`, marked `integration`

Skipped unless `RAGFABRIC_TEST_OLLAMA` is set. Ingest a small fixture corpus with known entities and relationships, run extraction, and **record what the model actually produced**: how many entities and edges, how many fell below the floor, which relationships it invented, which it missed, and whether resolution merged correctly.

**Per ADR 0004 this is what happened on that corpus with that model on that day. It is not a benchmark.** Extraction quality is the whole risk of this phase, and a local model inventing relationships is a finding to write down, not to tune the fixture around until it looks good. Phase 5's equivalent run found the planner routing a paraphrase question to lexical search, and that finding was worth more than a passing test.

- [ ] **Step 1: Write the test and the fixture corpus**, with the correct entities and relations written down first so misses and inventions are both visible.
- [ ] **Step 2: Run it against real Ollama and record the actual output.**
- [ ] **Step 3: Write the findings into `docs/learning/`**, including everything the model got wrong.
- [ ] **Step 4: Commit** `test: record one real extraction run against a local model`

---

## Task 14: Documentation and ADRs

**Files:** Create `docs/adr/0011-...`, `docs/adr/0012-...`, `docs/concepts/knowledge-graphs.md`; rewrite `docs/graph-rag.md`; update `docs/README.md`, `ROADMAP.md`, issue #7, `docker-compose.yml`

`docs/graph-rag.md` currently describes a Neo4j design that this phase did not build, and a query flow that filters access after traversal. It is rewritten to describe what shipped.

ADR 0011 records the graph store decision with the spike's real output. ADR 0012 records why an unverified edge is not a fact, covering both the confidence floor and the graph citation contract.

The compose `full` profile still starts Neo4j. Remove it, and say in the file why.

**One implementer at a time in the worktree.** Two agents sharing a tree means one runs `git add -A` over the other's half-finished work. This is recorded in the Phase 4 and Phase 5 notes and has nearly happened twice.

- [ ] **Step 1: Write both ADRs.**
- [ ] **Step 2: Rewrite `docs/graph-rag.md` to match what shipped.**
- [ ] **Step 3: Remove Neo4j from `docker-compose.yml` and `.env.example`.**
- [ ] **Step 4: Update `ROADMAP.md`, `docs/README.md` and issue #7**, ticking Phase 6 only for what actually merged.
- [ ] **Step 5: Re-read every changed doc against the code.**
- [ ] **Step 6: Commit** `docs: record the graph store and extraction trust decisions`

---

## Task 15: Release v0.3.0 preparation

**Files:** `CHANGELOG.md`, version fields across the packages

**Publishing is GATED.** Prepare in the working tree only. Tagging, pushing to main, publishing images and creating a release each need explicit per-action approval from the owner at the time. Do not read a standing permission out of this document.

- [ ] **Step 1: Write the `[0.3.0]` changelog section**, every entry traceable to a merged commit, heading dated `unreleased` until tag time, with a Notes block recording what is deliberately absent.
- [ ] **Step 2: Set the version to `0.3.0`** across core, server, cli and sdk-python, and confirm `uv.lock` agrees.
- [ ] **Step 3: Verify the image still builds and runs** with the graph strategy reachable.
- [ ] **Step 4: Stop.** Report and wait for approval on the release steps.

---

## Parallel execution

Tasks 1 and 2 land first, by the controller, because both tracks compile against them. Phase 5 proved the cost of handing a shared interface to one track: the other cannot import it until the merge, and both invent conflicting shapes.

| Track | Tasks | Area |
|---|---|---|
| A, the build side | 3, 4, 5 | extraction, confidence floor, incremental skipping, resolution |
| B, the query side | 6, 7, 8, 9 | directed traversal, access filter, guidance, strategy, empty cases |

Each track works in **its own worktree on its own branch**, merged when both land. Tasks 10 to 15 run sequentially afterwards, one implementer at a time.

**Expect the merge to break.** Phase 5's two tracks were individually green and the merged branch was 3 failed with 118 errors. The merged-branch test run is the only one that proves anything, and neither track can run it.

---

## Self-Review

Before the phase is called done:

- Can a caller reach a node or walk an edge sourced only from a document they cannot read? Prove by test, not by reading.
- Is any directed relation walked backwards anywhere?
- Does any stored entity or edge carry a confidence nobody measured?
- Can a relationship claim survive without citing both its edge and a supporting chunk?
- Does a re-ingest re-extract chunks that did not change?
- Does a truncated traversal look identical to a complete one?
- Is every configured graph setting read by something?
- Is any quality claim made anywhere without a measured run behind it?
