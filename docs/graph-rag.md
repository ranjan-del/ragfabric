# Graph RAG

> Status: concept complete. Implementation ships in **v0.3.0** using Neo4j.

## What it does

```
ingest: chunks -> entity extraction -> relationship extraction -> entity resolution -> Neo4j (nodes and edges keep chunk ids)
query:  entities in the question -> match nodes -> traverse k hops -> collect source chunks -> context -> LLM -> answer + citations
```

A knowledge graph is built from the documents once. Questions are answered by finding their entities in
the graph, following relationships, and pulling the source passages along the path.

## Why it is needed

"Which teams report to the CTO and which security policies apply to them?" is three facts joined by
relationships, probably spread over three documents. Vector search finds passages similar to the whole
question and returns none of the three cleanly. A graph answers by traversal.

## How it works internally

| Concept | Meaning |
|---|---|
| Entity | A thing with an identity: a person, organisation, department, product, policy, technology, project, location, date |
| Relationship | A typed, directed link between two entities: `REPORTS_TO`, `BELONGS_TO`, `APPLIES_TO`, `WORKS_ON`, `LOCATED_IN` |
| Node | The graph representation of an entity, with properties and a list of source chunk ids |
| Edge | The graph representation of a relationship, also carrying source chunk ids |
| Traversal | Walking edges from matched nodes, up to k hops, collecting nodes, edges and their source chunks |

### Extraction

An LLM reads each chunk with a schema of allowed entity types and relationship types and returns
structured JSON. Every extracted node and edge records the `chunk_id` it came from. This link is what
makes Graph RAG answers citable and debuggable: a wrong edge can be traced to the sentence that produced
it.

### Entity resolution

The same entity appears in many spellings: "R. Sharma", "Ravi Sharma", "Dr Sharma". Resolution
normalises names, merges known aliases, and uses embedding similarity between candidate names and
descriptions as a tie break. Merging too aggressively conflates people; too little fragments the graph.
Both are visible in the console, where merges can be corrected.

### Storage

Neo4j with uniqueness constraints per entity type, `MERGE` upserts so re-ingestion is idempotent, and
`entities` and `relationships` mirrored into PostgreSQL for the console and for auditing.

### Query

1. Extract the question's entities (same extractor, question mode).
2. Match them to nodes by normalised name and type, with fuzzy fallback.
3. Traverse relevant relationship types up to k hops (default 2), bounded by a node limit.
4. Collect the source chunks of every node and edge on the path, filtered by the caller's access filter.
5. Build the context from those chunks plus a textual rendering of the sub graph.
6. Generate with citations to the chunks.

Community summaries for corpus wide questions ("what are the main themes across all project notes") are
a later addition; v0.3.0 ships local, entity anchored retrieval.

## Alternatives and trade offs

| Alternative | Trade off |
|---|---|
| Agentic RAG with multiple retrievals | Can follow some chains, does not know the relationships exist |
| Structured data via SQL tool | Better when the relationships already live in a database |
| Hybrid: graph for entities, vector for prose | The usual production pattern; RagFabric's router enables it |

## Where it fails

Extraction misses or hallucinates a relationship and the graph is wrong until re-ingestion; corpora with
few named entities gain nothing; index time cost is one LLM call per chunk; operating Neo4j is another
service to run. Complexity is the highest of the four strategies, which is why it is optional in the
lite profile.
