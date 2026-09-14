# Ingestion, concept notes

> Status: Phase 2 (shipped). This is a learning document: why the pipeline is shaped the way it is,
> not an API reference. For the code, see `packages/core/src/ragfabric_core/ingest/`.

## Why clean before you chunk

A parser turns a PDF, DOCX, PPTX, TXT or MD file into plain text with page markers. That text is dirty
in three specific ways, and each one hurts retrieval quality if it reaches the chunker unfixed:

| Problem | Example | Why it hurts retrieval |
|---|---|---|
| Hyphenation at a line break | `employ-\nee` | The broken word never matches a query for "employee" |
| Irregular whitespace | Repeated spaces, tabs, blank line runs | Dilutes the embedding with noise instead of meaning |
| Repeated headers and footers | A page number or company name on every page | Becomes the "most similar" chunk to almost any question, because it is the one thing every chunk near it shares |

`ingest/clean.py` fixes all three before chunking: it repairs hyphenation, collapses whitespace, and
removes any line that repeats on three or more pages (`_MIN_REPEATS = 3`). Cleaning is deterministic
and conservative: it never reorders text and never crosses a page boundary, so the character spans
computed afterward stay exact for the cleaned text that ends up in the chunk. Clean once, at ingestion
time, rather than trying to compensate for dirty text at query time in every retrieval strategy.

The same pass also detects headings with three cheap heuristics: numbered lines (`1.`, `2.3`), short
ALL CAPS lines, and markdown `#` headings. Each chunk records the nearest preceding heading as its
`section`. Wrong guesses here are cheap, because nothing is ranked by section yet; it exists so later
phases and the UI have something better than "page 14" to show.

## Chunk size and overlap: the trade-off, and the two numbers we chose

Chunking splits cleaned text into overlapping windows so each one is small enough for an embedding
model to represent well, and large enough to still contain a complete thought.

| Direction | What goes wrong |
|---|---|
| Chunk too small | A passage loses the surrounding context needed to answer; a fact gets split across chunk boundaries and neither half is complete on its own |
| Chunk too large | One vector has to represent several topics at once, which blurs it; wastes context budget on irrelevant sentences alongside the relevant one |
| Overlap too small | An answer sitting across a chunk boundary is cut in half in every chunk that contains it |
| Overlap too large | More chunks per document, more storage, more embedding cost, for diminishing return |

RagFabric ships `chunk_size: 600` and `chunk_overlap: 80` characters in `ragfabric.yaml` (about 13
percent overlap). These are read from configuration, not hard coded, specifically so a corpus of short
technical snippets or one of long narrative prose can each be tuned without a code change. Raising
`chunk_size` trades precision for context: fewer, broader chunks, cheaper to embed and index, but each
one is a blunter match for a narrow question. Raising `chunk_overlap` reduces boundary loss at the cost
of more chunks for the same document.

## Why metadata has to survive every stage

A chunk is useless for a citation unless it can point back to somewhere specific. Four pieces of
metadata are attached at chunking time and must survive cleaning, storage, indexing and retrieval
unchanged: **page** number, character **span** within the page, **section** (the nearest heading), and
the document's **document_type** (`document`, `presentation`, `table`, `text`, or `other`, from
`document_type_for`, based on the file extension). Losing any of them at any stage turns "the answer is
in there somewhere" back into "the answer is in this document," which is the exact regression citations
exist to prevent. `document_type` and the retained file's `storage_path` live on the `documents` table;
`section` lives on the `chunks` table, added by migration 0003 alongside the two new index tables.

## Retained originals

Every uploaded file is kept, not just its extracted text, under a configurable `uploads_dir` (default
`data/uploads`) at `<uploads_dir>/<document id>/<safe name>`, with the filename sanitised and capped at
120 characters. `GET /api/documents/{id}/download` serves it back, subject to the same access filter as
everything else, and the file is removed when the document is deleted. This exists because extracted
text always loses something (a diagram, exact table formatting, the original layout) and because a
citation is more convincing next to the source document than next to a wall of extracted text.

## Inline versus queued indexing

After chunking, `ingest/indexing.schedule_indexing` decides how the chunks get embedded and indexed:

| Mode | What happens | When to use it |
|---|---|---|
| `inline` (default) | The same request that ingests the document also embeds it and writes both indexes before returning | Development, small corpora, no Redis running |
| `queue` | The document is marked `indexing` and two jobs (`index_document`, `extract_graph`) are pushed onto a `JobQueue`; a separate `ragfabric worker` process drains them | Larger corpora, or an API that should not block on embedding calls |

A document's `status` moves through `processing` (parsing and chunking) to `indexing` (embedding and
index writes, in `queue` mode) to `ready`, or `failed` if any step raises. In `inline` mode a document
goes straight from `processing` to `ready` inside the one request.

## Why both indexes are written in one job

`index_document` writes to the vector store and the lexical store in the same call, regardless of
whether that call runs inline or inside a worker. This keeps the two indexes consistent with each
other and with the source chunks: there is no window where the vector index has a chunk the lexical
index does not, or vice versa. It also means Phase 2 can build and populate both indexes now, ahead of
retrieval switching over to them in Phase 3, without a second migration to "catch up" the data later.
The trade-off is that today's queries still run against the older, v1 in memory index, so ingestion
does strictly more work per document than the query path currently uses; Phase 3 removes that
duplication by pointing retrieval at `chunk_embeddings` and `chunk_search` and retiring the in memory
index.
