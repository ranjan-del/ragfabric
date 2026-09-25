# Knowledge graphs, concept notes

> Status: Phase 6 (shipped in core). This is a learning document: what an entity and a relationship
> actually are here, why direction is not optional, and where extraction goes wrong, not an API
> reference. For the code, see `packages/core/src/ragfabric_core/graph/contracts.py`, `extract.py`,
> `resolve.py`, `traverse.py` and `citations.py`. For the strategy as a whole, see
> [graph-rag.md](../graph-rag.md). The decisions behind the design are
> [ADR 0011](../adr/0011-postgres-recursive-cte-over-neo4j.md) (why PostgreSQL, not Neo4j) and
> [ADR 0012](../adr/0012-extraction-confidence-and-graph-citations.md) (why an unverified edge is
> not a fact).

## What a graph adds that a search does not

A vector or lexical search returns passages similar to the whole question. "Which teams report to
the CTO and which security policies apply to them?" is three facts joined by relationships, probably
spread over three documents, and no single passage states all three. A search returns the passages
closest to the question's wording; a graph answers by walking from an entity the question names,
along the relationships that connect it to others, collecting the passages along the way.

The cost is that a graph only helps when the corpus actually contains named entities in a
relationship structure. A policy document that never names a person or a team gains nothing from
one, and that is the honest empty case Task 9 exists to report rather than paper over.

## Entities and relationships

| Concept | Meaning here |
|---|---|
| **Entity** | A thing with an identity: `EntityType` is a fixed enum (person, team, organisation, project, product, document, policy, location). Identity is `(normalise(name), entity_type)`, exactly the pair the unique constraint enforces |
| **Relationship** | A typed, directed link between two entities: `RelationType` is a fixed enum (`REPORTS_TO`, `MEMBER_OF`, `BELONGS_TO`, `OWNS`, `WORKS_ON`, `LOCATED_IN`, `AUTHORED`, `MENTIONS`, `RELATED_TO`) |
| **Node** | The graph's representation of an entity in a traversal result (`GraphNode`): id, name, type and the minimum hop depth it was reached at. No description, and no alias: a stored description or alias may be paraphrased from a chunk the caller cannot see, which is exactly the leak ADR 0003 forbids |
| **Edge** | The graph's representation of a relationship in a traversal result (`GraphEdge`): the stored direction, the name it was walked under, whether that reading is reversed, and only the source chunks the caller may see |
| **Provenance** | `entity_sources` and `relationship_sources`: one row per (entity or relationship, chunk) that named it, each carrying that chunk's own reported confidence. This is the single source of truth for both citation and access; there is no second, denormalised copy to drift out of sync with it |

Both enums are **fixed**, not free text a model can extend. A model that reports a relation type
outside the list has nowhere for it to land: the extraction contract (`graph/contracts.py`) rejects
it as a `ContractViolation`, and the traversal has no direction rule to apply to a type it has never
heard of. The alternative, letting a model invent a relation type on the fly, would mean every new
type needs a direction decision made up on the spot at query time, which is exactly the kind of
judgement ADR 0004 will not let code assert without a rule behind it.

## Why identity is `(normalised name, type)`, not name alone

"Atlas" the project and "Atlas" the person are different entities that happen to share a surface
name. If identity were name alone, extracting "Atlas shipped in May" and "Atlas reports to the CTO"
from different chunks would merge two unrelated things into one node the moment both chunks were
read. Type is part of identity from the first write (the unique constraint is on the pair), and every
resolution stage (`graph/resolve.py`) refuses to merge across it: `_may_merge` requires equal
`entity_type` before anything else is even considered.

The corollary is that `normalise()` is deliberately conservative: NFKC and casefold, quote and
bracket stripping, trailing punctuation, nothing more. It folds `"Ravi Sharma"` and `"ravi sharma."`
together and leaves `"R. Sharma"` and `"Ravi Sharma"` apart, on purpose. Closing that second gap is
what entity resolution's alias and embedding stages exist for, not the identity function itself.

## Direction is not a detail, it is the fact

A relationship states something in one direction. "B reports to A" is not the same fact as "A
reports to B" read backwards, and a graph that cannot tell the two apart will eventually generate
the wrong one from a real edge.

`INVERSES` (`graph/contracts.py`) is the fixed table that says which relations are genuinely one
fact seen from either side, and what name they take when read the other way:

| Relation, stored | Forward reading | Backward reading | Why it is invertible |
|---|---|---|---|
| `BELONGS_TO` | "the team belongs to the department" | `CONTAINS`: "the department contains the team" | One containment fact, two directions to say it |
| `MEMBER_OF` | "the person is a member of the group" | `HAS_MEMBER`: "the group has the person as a member" | One membership fact |
| `OWNS` | "the owner owns it" | `OWNED_BY`: "it is owned by the owner" | One ownership fact |
| `WORKS_ON` | "the person works on the project" | `HAS_CONTRIBUTOR`: "the project has that contributor" | One contribution fact |
| `LOCATED_IN` | "the office is located in Pune" | `LOCATION_OF`: "Pune is the location of the office" | One location fact |
| `AUTHORED` | "the person authored it" | `AUTHORED_BY`: "it was authored by the person" | One authorship fact |
| `MENTIONS` | "the document mentions the entity" | `MENTIONED_IN`: "the entity is mentioned in the document" | One mention fact |
| `RELATED_TO` | symmetric | same name, either direction | The relation states no direction to begin with |
| `REPORTS_TO` | "A reports to B" | **not walked backwards** | A dotted line is not management; the reverse is a different, unstated fact |

A relation absent from `INVERSES`, `REPORTS_TO` is the one example above, is walked forwards only.
There is no default direction rule and no attempt to guess one: a type the table has no entry for
is simply never walked the other way. This is the bug the spike behind
[ADR 0011](../adr/0011-postgres-recursive-cte-over-neo4j.md) actually made: it walked every edge in
both directions, so a real `B REPORTS_TO A` edge could surface as a generated "A reports to B", a
fact the corpus never stated. Getting this table right, for every relation type the corpus can
produce, is the whole defence against that class of fabrication; adding a new `RelationType` without
deciding, explicitly, whether it belongs in `INVERSES` reopens exactly this hole.

One consequence worth being explicit about: a node's `depth` (the minimum hops from a seed) is
recorded on the node itself as the walk measures it, never re-derived from which edges point at it.
When both directions of an invertible relation are walkable, the forward reading is reported even
if the node was actually reached by walking backwards, which means an edge-direction reconstruction
of "how far is this node" would be silently wrong for exactly the nodes reached that way.

## Traversal: outward from matched entities, under access, with a budget

A traversal starts from the entities a question names, walks a bounded number of hops, and stops.
Three things bound it, and each answers a different failure:

| Bound | What it stops | Where |
|---|---|---|
| **Access, on nodes and on edges** | A caller reaching or even confirming the existence of an entity or a relationship sourced only by a chunk they cannot read | Every visibility check is an `EXISTS` correlated to the chunk's document and collection, applied inside the anchor, the recursive term and the edge fetch, never after |
| **Hop ceiling** (`MAX_HOPS_CEILING = 4`) | The walk's own work exploding as branching factor to the power of hops | Bounds the recursive CTE itself, which produces deduplicated `(node, depth)` rows rather than one row per path |
| **Node budget** | A dense neighbourhood flooding the result even though the walk itself stayed cheap | Applied to the reached set after the CTE runs: nodes ordered by (depth, id), the first `node_budget` kept, seeds always included, `truncated` set when more were reached than kept |

Matching a question's entity to a node is itself an access-checked operation, not just the walk that
follows it: an entity whose only source chunk is denied does not match, because a match reveals the
entity exists even before anything about it is shown. This is the second bug the ADR 0011 spike
made and the second thing this phase had to get right that a naive graph implementation would not:
filtering only the edges, and matching entities unconditionally, still leaks through the match
itself.

Truncation and isolation look different from the outside on purpose. A node with genuinely no
walkable neighbour and a node whose neighbours existed but were cut off by the budget are not the
same fact, and a caller who cannot tell them apart cannot tell "this corpus does not connect these
things" from "ask again with more room." See [graph-rag.md](../graph-rag.md) for the three named
empty cases this distinction feeds into.

## Where extraction goes wrong

Two different things can go wrong, and they are not caught the same way.

**The model reports something false.** A confidence floor (default 0.5, untuned) discards anything
the model was not confident enough about, and the discard is counted so an operator can see a model
or a corpus producing junk. What clears the floor is still only as good as the model: a floor stops
the least confident guesses, not every wrong one. See
[ADR 0012](../adr/0012-extraction-confidence-and-graph-citations.md) for the full write-time
discipline, including why a stored confidence is recomputed rather than fixed at write time.

**The model reports something true, and the answer still narrates a relationship no sentence
stated.** A traversal is correct and its edges are real, but generating prose over a walked path can
assert more than any single passage said: "the platform team reports to the CTO" is a fair reading
of two edges while appearing in no chunk verbatim. This is not an extraction failure, it is a
generation failure, and it is why the graph citation contract exists as a second, additive check
over relationship claims specifically. See ADR 0012 for the rules and their stated loosenesses.

**Entity resolution merges the wrong things, or fails to merge the right ones.** "R. Sharma" and
"Ravi Sharma" not merging fragments the graph: a traversal that should connect two facts finds no
path. "R. Sharma" and a different Sharma merging wrongly fuses two people, and the graph now asserts
things about someone who does not exist. Neither error can be tuned without measurement, which
belongs to Phase 8, so what this phase guarantees instead is that every merge is recorded with the
evidence that justified it and can be undone: see `graph/resolve.py` and `ragfabric graph merges` for
how, and [graph-rag.md](../graph-rag.md) for what a merge and an unmerge actually move.

None of the above is a quality claim. Whether the floor, the resolution stages or the citation
contract catch what they are meant to on a real corpus is a measurement, and per
[ADR 0004](../adr/0004-measurement-first-no-fabricated-numbers.md) no figure for any of them appears
anywhere in this repository until Phase 8 produces one.
