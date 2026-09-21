# Lexical versus vector retrieval, concept notes

> Status: Phase 4 (shipped). This is a learning document: what BM25 actually computes, why RagFabric
> computes it inside PostgreSQL, and where lexical retrieval beats vector search and where it loses,
> not an API reference. For the code, see `packages/core/src/ragfabric_core/stores/bm25_sql.py`,
> `stores/term_stats.py`, `stores/fusion.py`, `stores/boosting.py` and
> `strategies/vectorless.py`. For the strategy as a whole, see
> [vectorless-rag.md](../vectorless-rag.md).

## Two different questions

Vector search and lexical search are not two implementations of one idea. They answer different
questions, and the difference is the whole reason both exist.

| | What it compares | What it is blind to |
|---|---|---|
| **Vector search** | the meaning of the question against the meaning of each chunk, as a model that read millions of documents represents meaning | any word the model was never trained on, and any distinction the model considers unimportant |
| **Lexical search** | the words of the question against the words of each chunk, weighted by how rare each word is in this corpus | anything said in different words from the ones the asker used |

A vector model turns "how many times will it retry" and "what is the retry limit" into nearby vectors
because it learned, during training, that those phrasings are about the same thing. It has no
corresponding knowledge of `ERR_QUOTA_4419`, which it has never seen, and which it will chop into
sub-word pieces that mean nothing.

Lexical search is the mirror image. It has never heard of a synonym, but it knows exactly how unusual
`ERR_QUOTA_4419` is in **your** corpus, because it counted, and it does not need to have been trained
on anything.

## BM25, term by term

BM25 is the scoring function the vectorless strategy ranks with. It is a sum over the query's terms:
each term the query and the chunk share contributes something, and the contributions add up.

```
score(D, Q) = sum over t in Q of  IDF(t) * ( tf(t,D) * (k1 + 1) )
                                 / ( tf(t,D) + k1 * (1 - b + b * |D| / avgdl) )

IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

That is exactly what `bm25_score` in `stores/bm25_sql.py` computes, and exactly what the SQL
expression in the same file's `_search_postgres` builds. The function was kept pure, with no database
access, so that you can read it next to this formula and check it.

| Symbol | Meaning | Where RagFabric gets it |
|---|---|---|
| `D` | one chunk | a row of `chunk_search` |
| `Q` | the question | the terms `to_tsvector('english', query)` produces, so the query is in the same vocabulary the index was written in |
| `t` | one term of the question | one lexeme |
| `tf(t,D)` | how many times `t` occurs in chunk `D` | read out of the `tsvector` that `chunk_search` has stored since Phase 2, via `array_length(positions, 1)`. No per-term-per-chunk table exists |
| `df(t)` | how many chunks in the whole corpus contain `t` at least once | `term_stats.df`, maintained on every index and every delete |
| `N` | total number of indexed chunks | `corpus_stats.n_chunks` |
| `\|D\|` | length of chunk `D`, counted in indexed terms | `chunk_search.doc_len` |
| `avgdl` | mean chunk length across the corpus | `corpus_stats.sum_len / corpus_stats.n_chunks` |
| `k1` | term frequency saturation, default 1.2 | `strategies.vectorless.k1` |
| `b` | length normalisation strength, default 0.75 | `strategies.vectorless.b` |

Two of those deserve a note.

`|D|` is measured in **indexed terms, not words**. On PostgreSQL the index is a `tsvector`, so stop
words are gone and the rest is stemmed before anything is counted. In the worked example below, a
chunk of fifteen written words has `doc_len = 10`.

`avgdl` is **not stored**. `corpus_stats` holds two running totals, `n_chunks` and `sum_len`, and the
average is divided out when it is needed. A mean cannot be updated incrementally without drifting:
add and subtract chunk lengths from a stored average for a year and it will be quietly wrong. A
running sum stays exact under any sequence of additions and subtractions. `corpus_stats(db)` returns
`avgdl = 1.0` when the corpus is empty, so the denominator can never be a division by zero.

### The three factors, in plain terms

Read the formula as three things multiplied together.

**Rarity (`IDF`).** A term in almost every chunk tells you almost nothing about which chunk you want.
A term in one chunk out of a million tells you nearly everything. IDF is that intuition as a number,
and it is the single reason an error code outranks the word "limit".

**Repetition, with diminishing returns (the `tf` part).** A chunk that mentions the term four times
is more likely to be about it than one that mentions it once. But the fourth mention is worth far
less than the first, and the fortieth is worth nearly nothing. The shape
`tf * (k1 + 1) / (tf + k1 * ...)` is what makes that happen: as `tf` grows the fraction rises towards
`k1 + 1` and stops there, so one term's contribution is capped at `IDF(t) * (k1 + 1)` no matter how
many times it repeats. Without saturation, a chunk could win by repeating a word.

**Length (`b * |D| / avgdl`).** A long chunk contains more of everything, so it matches more queries
by accident. Dividing by length corrects for that. Dividing by length completely would over-correct,
since a long chunk genuinely can be more relevant, so `b` sets how much of the correction to apply.

### What `k1` and `b` actually do

| Parameter | Controls | Raise it | Lower it | Bounds enforced |
|---|---|---|---|---|
| `k1` | how fast repetition stops paying | repetition keeps mattering for longer | the second occurrence of a term is worth almost nothing; the score becomes closer to "does this term appear at all" | `k1 >= 0` |
| `b` | how hard long chunks are penalised | long chunks are pushed down harder | length matters less. `b = 0` removes length normalisation entirely, so `\|D\|` no longer affects the score | `0 <= b <= 1` |

`b` is clamped to `[0, 1]` in `config_file.py`, and the reason is worth knowing: a negative `b`
rewards long chunks instead of penalising them, and a `b` above 1 can drive the denominator negative,
which flips the sign of the score so that a better match ranks worse. Neither is a tuning choice
anyone means to make, so the configuration refuses them rather than ranking strangely.

### Why the `+ 1` inside the logarithm

Robertson's original IDF is `ln((N - df + 0.5) / (df + 0.5))`, with no `+ 1`. When a term appears in
more than half the corpus, `N - df` is smaller than `df`, the fraction drops below 1, and the
logarithm of a number below 1 is **negative**.

A negative IDF means the term subtracts from the score. A chunk that contains a common query term
would then rank **below** a chunk that contains none of the query's terms at all, which is not a
defensible ranking under any reading.

Adding 1 inside the logarithm (the variant Lucene uses, and the one `bm25_score` implements) floors
IDF at zero. A term present in every chunk contributes almost nothing, which is correct, instead of
contributing a penalty, which is not. The test
`test_idf_is_never_negative_for_a_very_common_term` in `packages/core/tests/test_bm25_sql.py` pins
this: a term in 99 chunks out of 100 must still score above zero.

## Why `ts_rank_cd` is not BM25

PostgreSQL ships `ts_rank_cd`, and RagFabric has used it since Phase 2. It is a ranking function, but
it is not BM25 and it cannot stand in for one.

`ts_rank_cd` scores a single document against a single query using only what is inside that document:
how often the query's terms occur, what weights they carry, and how close together they are (the `cd`
is cover density, a proximity measure). **It has no corpus-wide term.** Nothing in it knows how many
other chunks contain a given word.

That is precisely the knowledge needed to tell `ERR_QUOTA_4419` from `limit`. To `ts_rank_cd`, one
occurrence of a unique error code and one occurrence of a word every chunk contains look the same.
IDF is the missing piece, and supplying it is what `term_stats` exists for.

There is a second, more practical difference. The `ts_rank_cd` path builds its query with
`plainto_tsquery`, which joins the terms with `AND`. On the fixture corpus,
`plainto_tsquery('english', 'what is the retry limit for ERR_QUOTA_4419')` is

```
'retri' & 'limit' & 'err' & 'quota' & '4419'
```

so only a chunk containing **every** one of those matches at all. BM25 sums over whichever terms a
chunk happens to share, so a chunk matching three of five still scores. The two therefore do not just
order results differently, they can return different result sets.

Neither is strictly better. BM25 is stronger on rare terms; `ts_rank_cd` rewards terms appearing near
each other, which BM25, a pure bag of words, cannot see at all. They fail differently, so the
vectorless strategy runs both and fuses them.

### A note for anyone running on SQLite

Everything above describes the PostgreSQL path, which is the deployed one. RagFabric's stores also run
on SQLite so that development and most of the test suite need no database server, and two things
differ there. BM25 itself is identical, computed by the same `bm25_score` function over the same
`term_stats` and `corpus_stats`, with only the arithmetic moved into Python while the access and
metadata filters stay inside the SQL query. But there is no `tsvector`, so tokens are neither stemmed
nor stop-word-stripped, and there is no `ts_rank_cd`, so the second ranking is a plain token-overlap
score instead. Orderings will therefore not match PostgreSQL exactly. That is a real difference
between the two dialects, not a defect in either.

## How the two rankings are combined

`stores/fusion.py` combines them with Reciprocal Rank Fusion:

```
rrf_score(d) = sum over rankings r of  weight_r / (k + rank_r(d))
```

`rank` is 1-based and `k` defaults to 60, from Cormack, Clarke and Buettcher's original paper.

Fusion works on **ranks, not scores**, and that is not an arbitrary choice. BM25 scores are unbounded
and corpus-dependent: `4.0094` in the worked example below means nothing without knowing that corpus
and that query. `ts_rank_cd` returns roughly 0 to 1. To add them you would have to normalise, and to
normalise you would have to know each one's distribution, which changes with every query. Min-max
normalising a single result set is worse, because it makes the best hit a 1.0 whether it was
excellent or merely the least bad. Second place is second place regardless of scale, so ranks need no
such assumption.

Fusion **reorders** chunks; it never introduces one. The same holds for the phrase and identifier
boosts in `stores/boosting.py`, which multiply an existing hit's score and can never add a chunk to
the result set. Both restrictions exist for the same reason: every ranking came out of a store query
that applied the access filter inside the SQL (ADR 0003), so a chunk that appears in no ranking is a
chunk this principal is not permitted to see. Introducing one later would route around the filter
entirely.

The boosts cover the two cases IDF alone gets wrong. A quoted phrase, because BM25 is a bag of words
and cannot tell `"retry limit"` adjacent from those two words four paragraphs apart. An identifier,
because `to_tsvector('english')` splits and stems `ERR_QUOTA_4419` into `err`, `quota` and `4419`, so
the exact form is gone by ranking time. The boosts look at the chunk's raw text, where it survives.

## Why BM25 is computed in SQL

`rank_bm25` is a correct, well known Python implementation. RagFabric does not depend on it for the
default path, for one reason: it holds the entire corpus in the memory of the process that built it.

Under an API server with four worker processes there would be four copies of the corpus. Nothing
synchronises them, so a document ingested through worker 1 stays invisible to workers 2, 3 and 4
until they restart, and the copies drift apart the moment anything changes. The ceiling on corpus
size becomes RAM rather than disk. Ranking therefore runs next to the data, where there is one copy
and it is the same copy every worker reads.

An in-process store still ships, as `stores/bm25_memory.py`, for small corpora where operational
simplicity is worth more than a shared index, and as a reference implementation the SQL store is
tested against. It is capped (`DEFAULT_MAX_CHUNKS = 50_000`) and raises `StoreCapacityError` rather
than silently truncating, because a store that quietly answers from half its documents is worse than
one that refuses.

### Why not `pg_search`

ParadeDB's `pg_search` provides native BM25 inside PostgreSQL with no ranking code from us, which is
genuinely attractive. It was rejected on portability. `pgvector` is available on essentially every
managed PostgreSQL (RDS, Cloud SQL, Supabase); `pg_search` is not. Depending on it would force
self-hosters onto a custom PostgreSQL image and lock managed-PostgreSQL users out of lexical search
entirely. RagFabric runs on stock PostgreSQL with `pgvector` and nothing else.

### Why there is no `chunk_terms` table

The obvious way to store term frequency is one row per term per chunk. At a million chunks of roughly
600 tokens, that is on the order of hundreds of millions of rows to write, index and keep consistent.

It is not needed. `chunk_search.tsv` has stored each chunk's lexemes **with their positions** since
Phase 2, and the number of positions is the term frequency. `_search_postgres` unnests the `tsvector`
laterally and reads `array_length(positions, 1)`. Only document frequency, which no single row can
answer, needed new storage, and that is one small table keyed by term.

## Where lexical retrieval beats vectors

These are the cases where BM25 is not merely competitive but structurally better, and where an
embedding model is working against you rather than for you.

**Exact identifiers and codes.** `ERR_QUOTA_4419`, `get_user_by_id`, `CVE-2024-3094`, part number
`RM-4419-B`, order `#88213`. An embedding model has almost certainly never seen these strings, so it
tokenises them into fragments and produces a vector that reflects the fragments rather than the
identifier. BM25 does not need to have seen them either: it counted them, found them in one chunk out
of `N`, and gave them the IDF that rarity earns. This is the case the boosts reinforce.

**Quoted phrases.** When a user puts quotation marks around something they are asserting that the
exact wording matters. Vector search cannot honour that assertion; it compares meanings, and the
nearest chunk by meaning may not contain the phrase at all.

**Rare proper nouns.** A person, a project, a customer, an internal codename. Rarity is exactly what
IDF measures, and a name occurring in three chunks out of a hundred thousand is about as strong a
signal as retrieval ever gets.

**Numbers and versions.** `v2.1.4`, a threshold of `429`, a date. Embedding models represent numbers
poorly and near-identical numbers nearly identically, which is precisely the wrong behaviour when the
difference between `2.1.4` and `2.14.0` is the whole question.

**Jargon-heavy or low-resource corpora.** Internal tooling, a niche legal or clinical vocabulary, a
language the embedding model saw little of. The model's representation of vocabulary it was never
trained on is not wrong in an interesting way; it is close to uninformative. BM25's statistics come
from your corpus, so its notion of "rare here" is correct by construction on any corpus at all.

**Anywhere reproducibility and explainability matter.** A BM25 score decomposes into per-term
contributions you can print, as the worked example below does. "This chunk ranked first because
`quota` has `df = 1`" is an explanation. A cosine similarity of 0.83 is not.

**When there is no budget, no GPU and no vendor.** The vectorless strategy makes zero embedding calls
and reports `embedding_calls = 0` because none happened, not because the number was omitted. There is
no model to download, no API key, no rate limit and no per-token cost on the retrieval path.

## Where lexical retrieval loses to vectors

A document that only sells BM25 would be useless. These are the cases where lexical search is the
wrong tool, and no amount of tuning `k1` and `b` will fix them.

**Paraphrase.** "How long before it gives up retrying" against a chunk that says "the retry limit is
five attempts". The answer is right there, and BM25 scores it near zero, because the two texts share
almost no terms. This is not a ranking failure that a better formula would solve; the evidence BM25
uses is simply absent.

**Synonymy.** "car" and "automobile", "postcode" and "ZIP code", "invoice" and "bill", "SSO" and
"single sign-on". Each pair is one concept and two disjoint sets of terms. Stemming does not help:
`to_tsvector('english')` reduces inflections, not meanings.

**Questions whose answer never repeats the question's words.** "Why is this slow?" answered by a
chunk about lock contention. "Is this safe to run in production?" answered by a chunk listing
preconditions. The overlap is zero and the answer is correct.

**Conceptual and comparative questions.** "What are the trade-offs here", "how does this differ from
the other approach". These describe a shape of answer rather than name its vocabulary.

**Cross-lingual retrieval.** A question in one language against a corpus in another shares no terms
at all. A multilingual embedding model handles this as a matter of course; lexical search cannot
begin to.

**Users who do not know the corpus's vocabulary.** A newcomer asks in their own words. An expert
corpus answers in its own. The gap between the two is exactly what vector search bridges and lexical
search does not.

The honest summary: **lexical retrieval finds what you named; vector retrieval finds what you meant.**
When the user knows the exact string, lexical wins outright. When the user knows only the idea,
lexical can return nothing at all, and returning nothing is the failure mode you should expect,
rather than returning something subtly wrong.

## A worked example on the fixture corpus

The numbers below are from a real run, not an estimate. They were produced by indexing the six-chunk
fixture corpus defined in `packages/core/tests/test_bm25_sql.py` into a scratch PostgreSQL 18.6
database through `Bm25Store.index`, then reading `term_stats` and `corpus_stats` back out and calling
`Bm25Store.search` and `PostgresLexicalStore.search`. It is a six-chunk toy corpus chosen to make the
mechanism visible. It demonstrates how the ranking behaves; it is not a benchmark, and nothing about
retrieval quality at scale can be concluded from it.

The corpus is one chunk containing an error code, and five containing the phrase "retry limit":

| Chunk | Text | `doc_len` |
|---|---|---|
| 1 | Error ERR_QUOTA_4419 is returned once the retry limit for the export endpoint has been exhausted. | 10 |
| 2 | The retry limit for the export endpoint is five attempts. | 6 |
| 3 | A retry limit protects downstream services from a thundering herd. | 7 |
| 4 | Raising the retry limit above ten is discouraged in production. | 6 |
| 5 | Each connector has its own retry limit and its own backoff curve. | 5 |
| 6 | The retry limit is configured per environment, not per request. | 7 |

`corpus_stats` after indexing: `n_chunks = 6`, `sum_len = 41`, so `avgdl = 41 / 6 = 6.8333`.

Chunk 1 is fifteen written words and `doc_len = 10`. Seven of them (`is`, `once`, `the`, `for`, `the`,
`has`, `been`) are stop words `to_tsvector` drops, leaving eight; and `ERR_QUOTA_4419` is split into
three lexemes (`err`, `quota`, `4419`), which brings the count back to ten.

Selected rows of `term_stats`:

| Term | `df` |
|---|---|
| `limit` | 6 |
| `retri` | 6 |
| `export` | 2 |
| `endpoint` | 2 |
| `err` | 1 |
| `quota` | 1 |
| `4419` | 1 |

### Query one, which lexical wins: `what is the retry limit for ERR_QUOTA_4419`

`to_tsvector` turns the query into five lexemes: `retri`, `limit`, `err`, `quota`, `4419`.

BM25 results, in order:

| Rank | Chunk | Score |
|---|---|---|
| 1 | 1 | 4.0094 |
| 2 | 5 | 0.1665 |
| 3 | 2 | 0.1560 |
| 4 | 4 | 0.1560 |
| 5 | 3 | 0.1468 |
| 6 | 6 | 0.1468 |

The winner is ahead by a factor of roughly 24, and the arithmetic says why. Chunk 1's score breaks
down per term like this, with `N = 6`, `avgdl = 6.8333`, `|D| = 10`, `k1 = 1.2`, `b = 0.75`:

| Term | `tf` | `df` | Contribution |
|---|---|---|---|
| `4419` | 1 | 1 | 1.2950 |
| `err` | 1 | 1 | 1.2950 |
| `quota` | 1 | 1 | 1.2950 |
| `limit` | 1 | 6 | 0.0623 |
| `retri` | 1 | 6 | 0.0623 |
| | | **sum** | **4.0094** |

Take `quota` and check it against the formula by hand:

```
IDF   = ln( (6 - 1 + 0.5) / (1 + 0.5) + 1 )   = ln(4.6667)  = 1.5404
denom = 1 + 1.2 * (1 - 0.75 + 0.75 * 10 / 6.8333)          = 2.6171
score = 1.5404 * (1 * (1.2 + 1)) / 2.6171                  = 1.2950
```

And `limit`, which every chunk contains:

```
IDF   = ln( (6 - 6 + 0.5) / (6 + 0.5) + 1 )   = ln(1.0769)  = 0.0741
score = 0.0741 * 2.2 / 2.6171                              = 0.0623
```

The denominator is identical for both, because it depends only on the chunk, not the term. The entire
difference is IDF: `1.5404 / 0.0741`, a ratio of about 20.8. That ratio is what "rare term beats
common term" means numerically, and it is the reason a query mentioning an error code finds the
chunk containing it rather than the five chunks containing "retry limit".

No embedding model was involved, and none could have done better here: a model that never saw
`ERR_QUOTA_4419` has nothing to represent.

For the same query, the `ts_rank_cd` store returned **only chunk 1**, with score `0.02`, because
`plainto_tsquery` requires all five lexemes and only chunk 1 has them. The two rankings agree on the
winner and disagree completely about the rest, which is the situation RRF exists to resolve.

### Query two, which lexical loses: `how often may a client try again before it is refused`

This asks the same thing chunk 2 answers. `to_tsvector` turns it into `often`, `may`, `client`,
`tri`, `refus`.

**Not one of those five lexemes has a row in `term_stats`.** BM25 returned an empty list. So did
`ts_rank_cd`.

Note especially that `try` stems to `tri` while `retry` stems to `retri`, so even the one word the
question and the answer arguably share does not match. Zero terms in common means zero evidence, and
the store correctly returns nothing rather than inventing a ranking. This is the case vector search
exists for: "how often may a client try again" and "the retry limit is five attempts" are close in
meaning and far apart in vocabulary, and meaning is the only one of the two that vector search
compares. The vector path was not run on this corpus, so that is stated as a mechanism, not as a
measured result.

This is the shape of the lexical failure mode, and it is worth internalising: it is usually an empty
or near-empty result set, not a plausible wrong answer. That makes it easy to detect and hard to
ignore.

To run the same kind of query against your own corpus through the shipped CLI, with a server
running:

```
ragfabric ask "what is the retry limit for ERR_QUOTA_4419" --strategy vectorless
```

## Choosing between them

| Situation | Use |
|---|---|
| Users search by identifier, code, version, part number or name | `vectorless` |
| Users ask in their own words about concepts | `traditional` |
| Corpus vocabulary is niche, internal or low-resource | `vectorless` |
| Questions and answers are phrased differently by nature | `traditional` |
| No embedding budget, no GPU, no external API allowed | `vectorless` |
| Cross-lingual | `traditional` |
| Both patterns in one deployment | select per request |

The strategy is selectable per request, so this is not a deployment-wide commitment:

- API: `"strategy": "vectorless"` in the body of `POST /api/search/query`, `POST /api/search/semantic`
  or `POST /api/ask`. `POST /api/search/hybrid` accepts only `traditional`
- CLI: `ragfabric ask "..." --strategy vectorless`
- Python SDK: `strategy="vectorless"` on `Client.ask`, `ask_stream` and `search`

Tuning lives under `strategies.vectorless` in `ragfabric.yaml`: `k1`, `b`, `phrase_boost`,
`identifier_boost`, `fusion_k`, `fusion_weights`, `top_k` and `max_context_tokens`. See
[configuration.md](../configuration.md).

One behaviour that is deliberately absent: the vectorless strategy **ignores
`similarity_threshold`**. BM25 scores are unbounded and corpus-dependent, as the worked example shows,
so a fixed floor means one thing on one corpus and something entirely different on the next.
Honouring it would be honouring a number that cannot mean anything, so it is ignored and said so
rather than silently dropped.

## One upgrade requirement

BM25 needs `chunk_search.doc_len`, and rows indexed before Phase 4 do not have it. They carry
`doc_len = 0`.

Those rows are **excluded from BM25 results, not scored**. A zero length would zero the
`b * |D| / avgdl` term, shrinking the denominator and ranking every legacy row above everything
correctly indexed. Excluding them is visible in the results, since documents you know exist do not
come back. Scoring them would have been silently and spectacularly wrong.

Bring them forward with:

```
ragfabric reindex --lexical-only
```

This rebuilds the lexical index and the term statistics without re-embedding anything, so it costs no
embedding calls and no provider spend.
