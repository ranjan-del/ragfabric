# ADR 0011: The graph lives in PostgreSQL, walked with recursive CTEs, not Neo4j

Status: accepted
Date: 2026-09-24
Supersedes: none
Related: ADR 0003 (access control inside retrieval), ADR 0007 (BM25 in SQL, over pg_search),
ADR 0009 (a plain Python state machine, over LangGraph)

## Context

The roadmap, issue #7, `docs/graph-rag.md` and the compose `full` profile all named Neo4j for
Phase 6. `stores/base.py` already carried a `GraphStore` protocol shaped for it
(`upsert_entities`, `upsert_relationships`, `neighbours`), written in Phase 1 before anything
about graph retrieval had been built.

Before writing to that promise, a spike built the traversal against real PostgreSQL and measured
the alternative, the same discipline ADR 0007 and ADR 0009 applied before rejecting `pg_search`
and LangGraph.

| Criterion | PostgreSQL recursive CTE | Neo4j |
|---|---|---|
| Python dependencies | 0, already present | 2 (`neo4j`, `pytz`), which is light and was not the issue |
| Extra servers to run, back up and secure | 0 | 1 |
| **Access filter inside the traversal** | **Yes, proven by a real run** | **No, not without duplicating the access model** |
| Traversal code | about 30 lines of SQL | about 5 lines of Cypher |

The third row decided it. The spike seeded two collections, made one of them denied, and ran a
3-hop traversal twice:

```
sees BOTH collections           -> ravi sharma(0), platform team(1), cto(2), skunkworks(3)
sees ONLY the public collection -> ravi sharma(0), platform team(1), cto(2)
```

The restricted caller does not receive `skunkworks` filtered out of a result. The path to it never
exists, because the access predicate sits inside the recursive term and the forbidden edge is
never walked.

Neo4j cannot do that directly, because `chunks`, `documents` and `collections` live in PostgreSQL
and Neo4j has no knowledge of them. That leaves three options and all three are bad: duplicate the
chunk to document to collection mapping into the graph, which gives the access model two sources
of truth that will drift; fetch every allowed chunk id from PostgreSQL and pass them into Cypher as
a parameter, which has a scaling wall on a real corpus; or traverse unfiltered and filter
afterwards, which is what `docs/graph-rag.md` used to describe and what ADR 0003 forbids.

**In a graph, filtering after traversal leaks more than it does in a search.** Knowing that
`A REPORTS_TO B` is information even when the sentence that said so is never shown. A traversal
that walks edges the caller cannot see hides the text and reveals the shape.

Honest about the evidence: the PostgreSQL side was run against a real database. The Neo4j side was
analysed rather than run, because the conclusion follows from where the access data lives, not
from Neo4j's performance. This is the third dependency rejected on the same principle: `pg_search`
in ADR 0007, LangGraph in ADR 0009, Neo4j here.

The spike itself was not free of bugs, and both are load bearing for what shipped, not footnotes:

**Traversal direction.** `REPORTS_TO` is directed. The spike walked edges in both directions,
which is a correctness bug: traverse `B REPORTS_TO A` backwards, generate "A reports to B", and a
fact the corpus never stated has been fabricated.

**The access filter must apply to entity matching, not only to edges.** The spike filtered edges
alone. Matching a question entity to a node reveals the node exists, so if the only chunk
mentioning an entity is denied, the match itself must fail. Same class of leak, and the spike
missed it.

## Decision

The graph is relational data, not a second store. `entities` and `relationships` (already present
since migration 0002) are the graph; `entity_sources` and `relationship_sources` (ruling R1,
migration 0008) are the single source of truth for which chunks justify which node or edge, and
therefore for what a caller may see of it. `graph/traverse.py` walks it with `WITH RECURSIVE`,
`UNION`, `CASE` and `EXISTS`, portable SQL that runs unchanged on SQLite and PostgreSQL.

- **Access sits inside the query, not after it (ADR 0003, ruling R10).** A node is visible iff at
  least one `entity_sources` row points at a chunk the caller's `AccessFilter` admits. An edge is
  walkable iff at least one `relationship_sources` chunk is admitted and the node it leads to is
  visible. Both predicates are correlated `EXISTS` subqueries placed in the match, in the anchor of
  the recursive term, in the recursive term itself, and in the edge fetch. Nothing is filtered
  after the walk.
- **Direction is a fixed table, not an assumption (ruling R2).** `graph/contracts.py` declares
  `INVERSES`, a fixed mapping from a relation type to the name it is read under when walked
  backwards, for relations that are genuinely the same fact seen from either side (`BELONGS_TO`
  backwards is `CONTAINS`). A relation absent from that table, `REPORTS_TO` among them, is walked
  forwards only. Reading it backwards would assert a fact the corpus never stated.
- **The recursive term produces `(node_id, depth)` rows, deduplicated by `UNION`, bounded by
  `MAX_HOPS_CEILING = 4` (ruling R12).** That is what keeps the work proportional to visible nodes
  times hops rather than to branching factor to the power of hops, which is what makes the node
  budget in `graph/traverse.py` (ruling R16) a meaningful safety property instead of a number that
  never actually caps anything.
- **`GraphStore` in `stores/base.py` is removed.** It was written in Phase 1 for a Neo4j
  implementation that was never built and nothing else ever implemented; the only thing that ever
  satisfied its shape was a test fixture in `test_interfaces_shape.py`. Keeping a protocol with one
  fictional conformer would document a pluggability the graph does not have: `graph/traverse.py`
  is written directly against the relational schema and the access predicate it shares with every
  other store, not behind a swappable interface, and a config that still names `kind: neo4j` is
  rejected at validation with a message pointing at this decision (`config_file.py`).
- **`graph_store.kind` in `ragfabric.yaml` has one value, `postgres`.** The Neo4j value the v0.2
  roadmap implied is rejected, loudly, rather than silently mapped onto whatever ships.

## Alternatives considered

| Option | Why it was rejected |
|---|---|
| Neo4j, as the roadmap promised | Measured in the spike against the same traversal. Every workable way to give it the access data PostgreSQL already holds either duplicates the access model, does not scale, or filters after the walk, which ADR 0003 forbids outright. |
| Duplicate the chunk to document to collection mapping into the graph store | Two sources of truth for the same fact drift the moment either one changes without the other, and the access model is exactly the place a silent drift is worst. |
| Fetch every allowed chunk id from PostgreSQL and pass them into Cypher as a parameter | Works at fixture scale. A real corpus's allowed-chunk set for a broad principal has a scaling wall this hits directly, and it was never measured because the first two rows of the criteria table already ruled it out. |
| Traverse unfiltered and filter the result afterwards | What the pre-Phase-6 `docs/graph-rag.md` described. This is exactly the leak ADR 0003 exists to forbid: a walk that crosses a denied edge has already used it to reach whatever lies beyond, before anything is dropped from the response. |
| Keep the `GraphStore` protocol for a future backend | Nothing implements it today and the traversal that shipped does not use it; keeping it would advertise a pluggability point that is fictional. A future store can be given its own interface when it exists, over data it can actually see the access predicate on. |

## Consequences

- **One relational database, not two systems to run, back up and keep consistent.** A deployment
  that turns on `graph_store.enabled` pays one extra LLM call per changed chunk at ingest and one
  extra query shape at read time; it adds no new service to the compose stack.
- **The access proof is a property of the query, not of application code remembering to filter.**
  The same class of guarantee ADR 0003 already requires of the vector and lexical stores now holds
  for the graph, checked the same way: by what the SQL itself can reach, not by a pass over what it
  returned.
- **Every known relation type needs a direction rule.** A relation type absent from both the forward
  set and `INVERSES` has no rule and is never walked; adding a new `RelationType` means deciding,
  explicitly, whether it is directed or invertible, not defaulting to either.
- **`MAX_HOPS_CEILING` is a hard ceiling on the walk's own work, and the node budget is a hard
  ceiling on what is returned; they bound different things and both are load bearing.** PostgreSQL
  forbids `LIMIT` inside a recursive term, so the budget is applied to the reached set after the
  CTE runs, not inside it (ruling R16).
- **No performance comparison is claimed.** The spike measured whether the access filter could sit
  inside the query on each side, not query latency or throughput on a real corpus at scale; that
  measurement, if it is ever needed, belongs to Phase 8 (ADR 0004).
- **A future graph backend is not precluded, only undocumented as a present capability.** If one is
  ever added, it earns its own interface over data it can prove the same access property on, rather
  than reviving a protocol nothing built against.
