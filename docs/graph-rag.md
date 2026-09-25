# Graph RAG

> Status: **core implementation merged for v0.3.0** (Phase 6, landed on the phase branch, release
> not yet cut). `GraphRAGStrategy` (`packages/core/src/ragfabric_core/strategies/graph.py`) is
> registered as `graph` in `strategies/registry_defaults.py` and reachable from
> `POST /api/ask`, `POST /api/search/query`, `ragfabric ask --strategy graph` and the Python SDK.
> `POST /api/search/hybrid` continues to refuse it with a 422, the same way it refuses every
> strategy but `traditional`. The graph lives in **PostgreSQL, walked with recursive CTEs, not
> Neo4j** ([ADR 0011](adr/0011-postgres-recursive-cte-over-neo4j.md)); the roadmap and this document
> promised Neo4j and that promise is withdrawn.
>
> Retrieval quality (whether extraction is accurate, whether resolution merges correctly, whether a
> generated answer is faithful) is **not measured**; Phase 8 measures it. This document describes
> the mechanism that shipped, not how well it works.

See [concepts/knowledge-graphs.md](concepts/knowledge-graphs.md) for the mental model (entities,
direction, why identity is a pair, where extraction goes wrong), [ADR 0011](adr/0011-postgres-recursive-cte-over-neo4j.md)
for the graph store decision and the spike behind it, and
[ADR 0012](adr/0012-extraction-confidence-and-graph-citations.md) for the confidence floor and the
graph citation contract.

## What it does

```
ingest: chunks -> extraction behind a confidence floor -> entity resolution (recorded, reversible)
query:  question's entities and implied relation types -> match visible nodes
        -> directed, access-checked traversal, bounded by hops and a node budget
        -> collect visible source chunks -> generate, citing chunks and edges together
```

A knowledge graph is built from a document's chunks as they are ingested, one LLM call per chunk
that has not already been extracted with the same text. Questions are answered by finding their
entities among the visible nodes, walking relationships outward under the caller's access filter,
and pulling the chunks that justify every node and edge the walk kept.

## Why it is needed

"Which teams report to the CTO and which security policies apply to them?" is three facts joined by
relationships, probably spread over three documents. A vector or lexical search returns passages
close to the whole question's wording and finds none of the three cleanly. A graph answers by
traversal: match "CTO", walk `REPORTS_TO` backwards under its rules, walk `APPLIES_TO` from what
that reaches, and collect the passages along the path.

## How it works internally

| Concept | Meaning in this system |
|---|---|
| Extraction | One LLM call per changed chunk, against `graph.contracts.ExtractionResponse`; everything below the confidence floor is discarded and counted |
| Entity, relationship | `EntityType` and `RelationType` are fixed enums; entity identity is `(normalise(name), entity_type)` |
| Provenance | `entity_sources` / `relationship_sources`: one row per (entity or relationship, chunk), each with that chunk's own confidence; an `entity_sources` row also keeps the spelling that chunk used (`surface_name`). The single source of truth for citation, access and the names a caller sees |
| Resolution | Three stages (exact, alias, embedding tie-break) merge duplicate entities; every merge between two stored entities is recorded, and unmerge is last-in-first-out and reports whatever it could not restore |
| Traversal | A recursive CTE from matched, visible nodes outward, access-checked on nodes and edges alike, bounded by a hop ceiling and a node budget |
| Subgraph | What a traversal returns: nodes with their depth, edges with their walked direction and visible source chunks, `truncated`, and `empty_reason` when there are no edges |
| Citation contract | An additive check over relationship claims: a claim naming a relationship must cite the edge and a chunk that backs it |

### Extraction behind a confidence floor (Task 3)

`graph/extract.py` sends one chunk's text to the configured model against the extraction contract
and gets back entities and relationships, each carrying the confidence the model reported.
Everything with `confidence < graph_store.confidence_floor` is discarded rather than stored, and the
count is reported (`ExtractionReport.entities_discarded`, `relationships_discarded`). An edge whose
endpoint entity itself fell below the floor is discarded too, even if the edge's own confidence
clears it (ruling R9): an edge cannot point at a row that was never stored. A response that fails
the extraction contract outright stores nothing from that chunk and is reported as a contract
violation, never raised.

Every stored entity and edge records the chunk it came from and the model that produced it
(`EntitySource`, `RelationshipSource`), and the parent row's own `confidence` is recomputed as the
maximum over its current sources every time a source is added, removed or moved, never left as a
number a current source no longer supports (ruling R20). See
[ADR 0012](adr/0012-extraction-confidence-and-graph-citations.md) for why this is a decision and not
a detail.

### Incremental hashing (Task 4)

One LLM call per chunk is the dominant cost of this strategy, so a chunk whose text has not changed
since its last extraction is skipped: `Chunk.extraction_hash` (sha256 of the chunk text) is compared
before any call is made, and the skip is counted (`ExtractionReport.chunks_skipped`) rather than
silent. `extraction_hash` is `None` until a chunk's first successful extraction, which always counts
as "changed".

A chunk whose text **has** changed has its previous `EntitySource` / `RelationshipSource` links
removed first, so re-extraction finds an entity or edge the new text still reports through the
ordinary upsert lookup and simply adds a fresh link, keeping its id, aliases and merge history,
rather than deleting and recreating it under a new id. Only once re-extraction has run is whatever
that removal left with zero sources at all garbage collected, and confidences on everything that
survives are recomputed. The whole pass, removal, extraction and cleanup, runs inside one savepoint,
so a contract violation on the new text rolls everything back and leaves the old graph intact, the
hash untouched, so the next pass retries the same chunk.

**Re-ingesting a document** (`reingest_document`) replaces every chunk row, so the hash alone would
not help: the new rows have no hash and no links. Before the old rows are deleted, each new chunk
whose text is identical to an old chunk's (paired in `chunk_index` order when the same text occurs
more than once) inherits that chunk's `extraction_hash` and has its `EntitySource` and
`RelationshipSource` rows moved to it unchanged, confidence and spelling included (ruling R41). The
extraction pass that follows skips it. An old chunk with no identical successor has its links
removed, whatever they alone supported is garbage collected, and the confidence of everything they
touched is recomputed from the sources left. What this covers precisely: a chunk whose text is
byte-identical to one in the previous revision costs no extraction call and no resolution embedding
call. A change that moves chunk boundaries changes the text of every chunk it touches, and those
chunks are extracted again. The search indexes are rebuilt for the new chunks either way, since that
is a different pipeline. Merge records are not rewritten: one that names a replaced chunk's id treats
that chunk as gone when unmerged, and reports what it could not restore.

**Deleting a document or a collection** removes its chunks' graph links, garbage collects what only
they sourced and recomputes every survivor's confidence before the rows are deleted (ruling R42), so
no entity or edge keeps a confidence that only a deleted chunk reported.

### Entity resolution: recorded, reversible merges (Task 5)

Three stages run in order of confidence, and entities of different `entity_type` never merge at any
stage:

1. **`exact`**: the two entities' names normalise to the same key.
2. **`alias`**: a normalised alias of one entity equals the other's normalised name or one of its
   own normalised aliases.
3. **`embedding`**: cosine similarity between `"name: description"` of two same-type entities is at
   or above `graph_store.similarity_threshold`. Each entity takes part in at most one stage-3 merge
   per resolution call (ruling R27), so no pass clusters transitively; a legitimate larger cluster
   needs a second pass.

Every merge writes an `EntityMerge` row carrying the evidence that justified it: which stage, the
survivor and the merged-away entity, the normalised key or the alias or the similarity that matched,
and a `restore` payload with everything an unmerge needs, down to each moved edge's original
endpoints and per-chunk reports. **Unmerge is globally last-in, first-out** (ruling R30, superseding
two earlier, narrower ordering rules that both turned out to be bypassable through ids that change
under unmerge): `unmerge(merge_id)` refuses, before writing anything, while any live merge has a
higher id, and names every one of them; the caller undoes those first, newest first. Where the graph
has genuinely changed since a merge, an unmerge is honest about what it can and cannot restore
(ruling R31): a recorded edge counts as still present only if its relation type, its non-merged
endpoint and at least one recorded source chunk still match; a chunk re-extracted since the merge
stays with whichever entity its current text names; anything not restored is reported, never
silently dropped.

**The cost this buys**: reversibility, not correctness. Neither merging too eagerly (two people
fused into one node) nor merging too little (a graph that fragments and finds no path) can be tuned
without measurement, which is Phase 8's job. What resolution guarantees today is that every decision
is inspectable and undoable (`ragfabric graph merges list|show|undo`), not that the threshold is
right.

### Directed traversal with access inside the query (Task 6, ADR 0011)

`graph/traverse.py` walks a recursive CTE outward from the matched, visible entities. Two properties
hold everywhere in the walk, not just at the edges of it:

- **Access.** A node is visible iff at least one of its source chunks is admitted by the caller's
  `AccessFilter`; an edge is walkable iff at least one of its source chunks is admitted **and** the
  node it leads to is visible. Both predicates are correlated `EXISTS` subqueries inside the match,
  the anchor of the recursive term, the recursive term itself, and the edge fetch. Nothing is
  filtered after the walk runs (ADR 0003).
- **Names and confidence follow access too.** An entity's spelling is text from a chunk, so a
  node's `name` is the `surface_name` of the caller's best admitted source (highest per-source
  confidence, then lowest chunk id), and a question matches an entity only through the normalised
  spellings of its admitted sources (ruling R40). The stored `Entity.name` and `Entity.aliases` are
  resolution's bookkeeping: after a merge they hold spellings from every source, including ones this
  caller cannot read, so they never reach a caller and are never matched against a question. An
  edge's `confidence` is likewise the max over its admitted sources (ruling R43), `None` when none
  of them measured one, never the stored aggregate that also counts denied chunks.
- **Direction.** Every `RelationType` walks forwards. It walks backwards only when it appears in
  `INVERSES`, under its inverse name, with `reversed=True`; a relation absent from that table
  (`REPORTS_TO`) is never walked backwards. See
  [concepts/knowledge-graphs.md](concepts/knowledge-graphs.md) for the full table and why this
  matters.

The recursive term produces deduplicated `(node_id, depth)` rows, capped at
`max_hops` (1 to `MAX_HOPS_CEILING = 4`), which bounds the walk's own work by visible nodes times
hops rather than branching factor to the power of hops. `relation_types` (the implied types from the
question, or every type when none is implied) narrows the walk from inside the recursive term and
the edge fetch alike, so a node reachable only through an unimplied type is never reached.

### Node budget and truncation (Task 7)

`node_budget` (default `DEFAULT_NODE_BUDGET = 50`) caps the **reached set**, not the CTE's own work:
nodes are ordered by (minimum depth, id), the seeds first, and only the first `node_budget` are
kept. Every edge touching a dropped node is dropped with it. `Subgraph.truncated` is `True` exactly
when more nodes were reached than kept; a truncated walk that kept no edge still reports
`no_walkable_edges`, and the trace records `truncated` separately so a caller can tell a genuinely
isolated node from one whose neighbours existed but were cut off by the budget. A `LIMIT` cannot sit
inside a recursive term on PostgreSQL, which is why the budget is a post-CTE cut rather than part of
the walk itself.

### The three empty cases (Task 9, ruling R22)

A graph request that finds nothing says why, checked in this order:

1. **`no_graph_coverage`**: nothing is visible to this caller at all, within the request's
   `collection_ids` when set. Checked **first**, with an access-checked `EXISTS` query, before any
   LLM call: the one query this costs is cheaper than the call it saves. `llm_calls=0` on this path.
2. **`no_entity_matched`**: the question named no entity that matched a visible node, including a
   question that named none at all.
3. **`no_walkable_edges`**: at least one entity matched, but the walk kept no edge (whether or not it
   was also truncated before it could keep one).

A corpus with few named entities gains nothing from a graph, and returning the nearest traversal of
whatever happened to be there would manufacture false relevance. Reporting which of the three
happened, and never conflating "nothing is visible to you" with "your question matched nothing",
is what keeps the empty case honest rather than merely quiet.

### Collection scoping (ruling R25)

`ctx.collection_ids` is a request's own scope, never a permission: it narrows every visibility
predicate the walk uses (`has_visible_entities`, `match_entities`, the traversal's anchor and
recursive term, the edge fetch, `visible_entity_chunks`) on top of the caller's access filter. It is
never folded into `AccessFilter.collection_ids`, which is an allow axis ORed with document ids and
would widen the result rather than narrow it. The coverage check runs in its own session, opened and
closed before the question-extraction LLM call, so a database connection is never held across that
call.

### The citation contract (Task 10)

Covered in full in [ADR 0012](adr/0012-extraction-confidence-and-graph-citations.md): a relationship
claim (naming two or more sub-graph entities, or carrying an `[E k]` edge marker) must cite an edge
in the traversed sub-graph that joins two of the entities it names, back that edge with a cited
chunk drawn from its own `source_chunk_ids`, have that chunk actually name both endpoints, and have
every entity the claim names covered by such an edge. A claim that fails is dropped and recorded
with its reason; the Phase 3 contract (`generate/contract.py`) still applies unchanged to every
chunk-grounded claim, this contract is additive.

### Chunk ordering (ruling R18, R21, R37)

A graph chunk carries `score=None`: a traversal is not a similarity search, and ADR 0004 forbids
inventing a number for one. Chunks are ordered by the minimum hop depth of whichever node or edge
sourced them (recorded in `metadata["graph_depth"]`), with an edge's source chunk ahead of a
node-only chunk at the same depth, so the `top_k` cut removes a node's own passage before it removes
the one passage a relationship claim about that edge could cite. When the cap still removes it, the
prompt renders the edge with "none provided" as its source passages, the honest fallback.

## Configuration

Two sections of `ragfabric.yaml`, read by `packages/core/src/ragfabric_core/config_file.py`:

```yaml
graph_store:                  # knowledge graph extraction during ingest
  enabled: false               # true runs one extraction call per changed chunk at ingest
  kind: postgres                # postgres only: the graph lives in the database (ADR 0011)
  extraction_model: null        # null uses the llm provider's default model
  confidence_floor: 0.5         # 0 to 1: items the model reports below this are discarded. Untuned until Phase 8
  similarity_threshold: 0.9     # 0 to 1: embedding similarity at which two entities merge. Untuned until Phase 8
  entity_types: [person, team, organisation, project, product, document, policy, location]
  relation_types: [REPORTS_TO, MEMBER_OF, BELONGS_TO, OWNS, WORKS_ON, LOCATED_IN, AUTHORED, MENTIONS, RELATED_TO]

strategies:
  graph:                       # graph traversal bounds at query time
    max_hops: 2                 # 1 to 4: edges walked out from the matched entities
    node_budget: 50              # most entities kept from the walk, nearest first (1 to 1000, untuned)
```

`graph_store.kind` accepts only `postgres`; a config naming `neo4j` is rejected at validation with a
message citing ADR 0011, rather than silently served by something else. `entity_types` and
`relation_types` narrow **extraction**: a disabled type is removed from the prompt, and anything the
model reports under it anyway is discarded and counted like a below-floor item. They do not narrow
what the query side can walk: a relation type that was disabled after ingestion still has its edges
in the database, and a traversal can still walk them; disabling a type stops the graph from
**growing** in that type going forward, it does not retroactively remove or hide what was already
extracted. `strategies.graph.max_hops` is capped by `MAX_HOPS_CEILING` at validation time, because
the walk itself raises above it and a config that validates but fails every query is worse than one
that fails at startup.

The `extract_graph` ingestion handler (`workers/handlers.py`) does nothing when `graph_store.enabled`
is false: no LLM call, no provider constructed. When enabled, it extracts the document's chunks,
then resolves only the entities sourced by chunks that were actually (re)extracted in this run
(ruling R36), skipping resolution entirely when every chunk was unchanged. Together with the
re-ingest carry-over above (ruling R41), re-ingesting a document whose chunks are all byte-identical
to the previous revision's makes no extraction call and no resolution embedding call.

`index_document` and `extract_graph` are separate jobs and a queue may run them in either order, so
`index_document` leaves an existing `extract_graph failed:` error and its `failed` status in place
instead of resetting them, and only a later successful `extract_graph` clears its own failure
(ruling R44): back to `ready` if the document's index job has finished since its last ingest, to
`indexing` otherwise.

## API, CLI and SDK surface

- **`POST /api/ask`** and **`POST /api/search/query`** accept `strategy: "graph"`. The response
  carries `subgraph` (nodes with the caller's own spelling and depth, edges with `walked_as`,
  `reversed` and `confidence` (`GraphEdgeOut.confidence`, the highest confidence the extraction
  model reported for that edge in a chunk this caller may read, or `None` when none of them was
  measured), `truncated`, `empty_reason`; never a stored description, ruling R6) and
  `dropped_relationship_claims` (each with its `RelationshipDropReason`), alongside the ordinary
  `dropped_claims` from the Phase 3 contract. `Citation.score` is `None` on the graph path, since
  nothing was ranked (ADR 0004).
- **Answer confidence on the graph path drops the relevance term.** `_confidence`
  (`packages/core/src/ragfabric_core/generate/answer.py`) blends coverage, relevance and support;
  a graph chunk carries no similarity score to compute relevance from (ruling R18), so relevance is
  left out rather than counted as zero, and the remaining measured weights (coverage and support)
  are renormalised to sum to one. This is why `Citation.score` is `None` for a graph citation:
  nothing was ranked, so there is no score to report (ruling R38(b), ADR 0004).
- **`POST /api/search/hybrid`** continues to refuse anything but `traditional`, graph included, with
  a 422: hybrid is one vector ranking fused with one lexical ranking, and a graph walk is neither.
- **A `document_id` or `format` filter with `strategy: "graph"` is a 422**, naming the filter,
  rather than silently ignored: the graph walk has no document or format predicate to apply, only
  access and `collection_id`.
- **`ragfabric ask --strategy graph`** prints the walked relationships and any dropped relationship
  claims alongside the answer.
- **The Python SDK** (`ragfabric_sdk`) accepts `strategy="graph"` on `ask` and `ask_stream`; the
  returned `Answer.subgraph` is `None` for every other strategy.
- **`ragfabric graph merges list|show|undo`** is an admin CLI, following the existing `users` /
  `groups` / `grants` pattern (database credentials, no separate principal check): `list` shows every
  recorded merge oldest first, `show` prints one merge's evidence and what an undo would restore,
  `undo` calls `graph.resolve.unmerge` and commits only on success, printing the blocking merge ids
  and exiting non-zero when a newer merge is in the way. There is no HTTP merge endpoint this phase;
  the Phase 9 console owns that.

## Alternatives and trade offs

| Alternative | Trade off |
|---|---|
| Agentic RAG with multiple retrievals | Can chain several searches, but has no notion that a relationship exists as a typed, directed fact; it can retrieve the passages that mention two entities and never know they are connected |
| Structured data via SQL tool | Better when the relationships already live in a normalised table; a knowledge graph exists because the relationships are stated in prose, not already structured |
| Hybrid: graph for entities, vector for prose | The usual production pattern; the router (Phase 7) is where this composition would live |

## Where it fails, and what is not measured

- **Extraction quality is the whole risk of this strategy and it is not measured.** The confidence
  floor discards the least confident guesses; it does not verify what clears it. Task 13's recorded
  run against a local model is a record of what happened on one corpus on one day, not a benchmark
  (ADR 0004).
- **A disabled entity or relation type narrows extraction, not the query side.** A type turned off
  after ingestion still has its previously extracted edges in the database, and the traversal will
  still walk them; see "Configuration" above.
- **Stage 3 of resolution costs one embedding call per batch of the whole type group of every
  entity touched by a changed document** (ruling R36): only pairs with an in-scope member can merge,
  but the comparison needs the other side's vector too, so a changed document naming one person
  re-embeds every person in the corpus. An approximate nearest neighbour or cached-vector prefilter
  is deferred to Phase 8, where it can be measured against the recall it would cost.
- **The citation contract has five stated loosenesses**, direction and paraphrase faithfulness among
  them: see [ADR 0012](adr/0012-extraction-confidence-and-graph-citations.md). It is deliberately
  over-strict rather than over-lenient, so a true claim can be dropped; it never lets a fabricated
  one through on a technicality it does not check for.
- **Entity resolution's thresholds are untuned.** `confidence_floor` (0.5) and `similarity_threshold`
  (0.9) are starting points, not measured optima.
- **No performance claim is made for the PostgreSQL traversal against a real corpus at scale.** The
  spike behind [ADR 0011](adr/0011-postgres-recursive-cte-over-neo4j.md) measured whether the access
  filter could sit inside the query, not query latency or throughput; that is Phase 8 work if it is
  ever needed.
- **A real extraction run against a local model**, recording what the model actually got right,
  missed and invented, is written up separately: see
  [learning/graph-extraction-first-run.md](learning/graph-extraction-first-run.md) once that record
  lands alongside this task.
