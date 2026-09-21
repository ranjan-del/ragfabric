# Phase 4: Vectorless RAG, Console v1, Release v0.1.0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrieval that finds the right chunk without embeddings, ranked by real BM25 computed inside PostgreSQL, plus an admin console that manages access without anyone touching the database, released as v0.1.0.

**Architecture:** BM25 is computed in SQL. Term frequency comes from the `tsvector` that `chunk_search` already stores (Phase 2), so no per-term-per-chunk table is needed. Document frequency comes from one new `term_stats` table, maintained incrementally on ingest and delete. BM25 and `ts_rank_cd` are then combined by Reciprocal Rank Fusion, because they fail differently. The console is an Angular build on the existing app shell, filling the API gaps that currently force database edits.

**Tech Stack:** Python 3.13 (uv), SQLAlchemy 2, Alembic, PostgreSQL 18, FastAPI, Angular 22, Tailwind, karma/jasmine.

**Issue:** https://github.com/ranjan-del/ragfabric/issues/5

**Predecessor:** `docs/plans/2026-09-18-phase-3-traditional-rag.md` (Traditional RAG, merged as PR #29)

---

## Why this shape, and what was rejected

Recorded here because the reasoning is the deliverable, not just the code.

**BM25 in SQL, not in Python.** `rank_bm25` is real BM25 but holds the whole corpus in process memory. Every API worker would hold its own copy, the copies drift apart as documents change, and the ceiling is RAM rather than disk. Ranking runs next to the data instead.

**`ts_rank_cd` is not BM25 and is not a substitute.** PostgreSQL's ranking functions weigh term frequency inside a document, term proximity, and optionally document length. They have no corpus-wide document-frequency term. That means they cannot tell a rare identifier from a common English word, which is precisely what this phase is measured on ("identifier questions on the fixture corpus rank correctly"). BM25's IDF supplies it.

**`pg_search` (ParadeDB) rejected.** It provides native BM25 with no code from us. It is rejected because `pgvector` is available on essentially every managed PostgreSQL (RDS, Cloud SQL, Supabase) and `pg_search` is not. Depending on it would force self-hosters onto a custom PostgreSQL image and lock managed-PostgreSQL users out entirely. RagFabric must run on stock PostgreSQL with only `pgvector`.

**No `chunk_terms` table.** The naive way to get term frequency is a row per term per chunk. At a million 600-token chunks that is on the order of hundreds of millions of rows. The existing `chunk_search.tsv` column already stores terms with positions, so term frequency is derivable from it and only document frequency needs new storage.

**Fusion kept.** BM25 and `ts_rank_cd` fail differently: BM25 is strong on rare terms, `ts_rank_cd` rewards proximity and phrase cohesion. Reciprocal Rank Fusion combines ranks rather than scores, so the two do not need a shared scale.

---

## Global Constraints

Every task's requirements implicitly include this section.

- **Python 3.13**, line length 100, `ruff` clean (`E`, `F`, `I`, `UP`, `B`), `ruff format` clean.
- **import-linter contracts must stay at 3 kept, 0 broken.** `ragfabric_core` imports nothing upward; `ragfabric_server` does not import `ragfabric_cli`; `ragfabric_sdk` is HTTP-only.
- **ADR 0003 holds: the access filter goes INSIDE the store query, never after it.** Filtering after retrieval computes `top_k` over rows the caller may not see.
- **ADR 0004 holds: never fabricate a number.** Unknown cost is `None`, not `0.0`. Counters report what actually happened.
- **ADR 0002 holds: one result shape.** Lexical retrieval returns `RetrievedChunk`, the same shape the vector path returns.
- **Citations reuse the Phase 3 contract unchanged** (`generate/contract.py`). Do not fork it, do not loosen it.
- **Every migration must downgrade cleanly.** CI runs upgrade head, downgrade base, upgrade head against real PostgreSQL 18.
- **SQLite remains the test default.** Anything PostgreSQL-only is skipped via the existing `RAGFABRIC_TEST_DATABASE_URL` guard, never by silently passing.
- **Config is strict.** `ragfabric.yaml` models forbid extra keys. A typo must be an error.
- **No AI attribution anywhere.** Commits are authored `Ranjan G <ranjan.g@ispf.ngo>`, conventional commit subjects, no `Co-Authored-By` trailers, no "Generated with" in PR bodies.
- **No em dashes** in code comments, docs, commit messages or UI copy.
- **Tests must be able to fail.** A test that passes against a deliberately broken implementation is not a test.

---
## File Structure

| File | Responsibility |
|---|---|
| `packages/core/src/ragfabric_core/migrations/versions/0007_bm25_term_stats.py` | Create `term_stats`, `corpus_stats`, add `chunk_search.doc_len` |
| `packages/core/src/ragfabric_core/models/index.py` | Add `TermStat`, `CorpusStat` models, add `doc_len` to `ChunkSearch` |
| `packages/core/src/ragfabric_core/stores/term_stats.py` | Maintain document frequency, corpus size and average length |
| `packages/core/src/ragfabric_core/stores/bm25_sql.py` | BM25 scoring expression and store |
| `packages/core/src/ragfabric_core/stores/fusion.py` | Reciprocal Rank Fusion over ranked lists |
| `packages/core/src/ragfabric_core/stores/boosting.py` | Exact phrase and identifier detection and boosts |
| `packages/core/src/ragfabric_core/stores/bm25_memory.py` | Optional in-process `rank_bm25` store with a corpus cap |
| `packages/core/src/ragfabric_core/strategies/vectorless.py` | `VectorlessRAGStrategy` |
| `packages/server/src/ragfabric_server/api/routes/admin.py` | User create and delete |
| `packages/server/src/ragfabric_server/api/routes/access.py` | Group update and delete |
| `packages/server/src/ragfabric_server/api/routes/collections.py` | Collection update |
| `packages/server/src/ragfabric_server/api/routes/providers.py` | Provider configuration read and write |
| `apps/assistant/src/app/**` | Console v1 screens |
| `docs/concepts/lexical-vs-vector.md` | BM25 explained, where lexical wins and fails |

---

## A decision the plan must make up front: term frequency outside PostgreSQL

`PostgresLexicalStore.index` currently writes, for non-PostgreSQL dialects:

```python
" ".join(sorted(_tokens(text)))
```

`_tokens` returns a `set`, so **term frequency is destroyed**. A term appearing nine times is indistinguishable from one appearing once.

BM25 without term frequency is not BM25. Since 390 of the suite's tests run on SQLite, leaving this as-is would mean the phase's central algorithm is only exercised behind a `RAGFABRIC_TEST_DATABASE_URL` guard that is skipped by default.

**Ruling:** change the non-PostgreSQL representation to preserve repetition, so `tf` is derivable and BM25 is unit-testable on SQLite. The PostgreSQL path keeps the real `tsvector` and is unaffected. Task 1 covers the migration; Task 3 covers the store change, including re-indexing existing rows.

---

## Task 1: Term statistics schema and migration 0007

**Files:**
- Create: `packages/core/src/ragfabric_core/migrations/versions/0007_bm25_term_stats.py`
- Modify: `packages/core/src/ragfabric_core/models/index.py`
- Test: `packages/core/tests/test_migrations_postgres.py`, `packages/core/tests/test_models_term_stats.py`

**Interfaces:**
- Produces: `TermStat(term: str, df: int)`, `CorpusStat(id: int, n_chunks: int, sum_len: int)`, `ChunkSearch.doc_len: int`

BM25 needs three quantities the schema does not currently hold:

| Quantity | Meaning | Where it lives |
|---|---|---|
| `tf(t, D)` | times term `t` occurs in chunk `D` | derivable from existing `chunk_search.tsv` |
| `df(t)` | number of chunks containing `t` | **new** `term_stats.df` |
| `N` | total indexed chunks | **new** `corpus_stats.n_chunks` |
| `avgdl` | mean chunk length in terms | **new**, `corpus_stats.sum_len / n_chunks` |
| `\|D\|` | length of chunk `D` in terms | **new** `chunk_search.doc_len` |

`avgdl` is stored as `sum_len` and `n_chunks` rather than a float average, because incrementally updating a mean is lossy and a running sum is exact.

- [x] **Step 1: Write the failing model test**

```python
def test_term_stat_and_corpus_stat_round_trip(session):
    session.add(TermStat(term="quota", df=3))
    session.add(CorpusStat(id=1, n_chunks=10, sum_len=6000))
    session.commit()
    assert session.get(TermStat, "quota").df == 3
    stats = session.get(CorpusStat, 1)
    assert stats.n_chunks == 10 and stats.sum_len == 6000
```

- [x] **Step 2: Run it, confirm it fails** with `ImportError` or `NameError` on `TermStat`.

- [x] **Step 3: Add the models**

In `packages/core/src/ragfabric_core/models/index.py`:

```python
class TermStat(Base):
    __tablename__ = "term_stats"

    term: Mapped[str] = mapped_column(String(255), primary_key=True)
    df: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class CorpusStat(Base):
    """One row, id=1. Running totals, not averages: a mean cannot be
    updated incrementally without drifting."""

    __tablename__ = "corpus_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    n_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sum_len: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
```

Add to `ChunkSearch`:

```python
    doc_len: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
```

- [x] **Step 4: Write the migration**

`down_revision = "0006"`. Upgrade creates both tables, adds `chunk_search.doc_len` with `server_default="0"`, and seeds `corpus_stats` with a single `id=1` row. Downgrade drops both tables and the column.

`term` is `String(255)`: PostgreSQL B-tree keys are bounded, and a lexeme longer than 255 characters is not a word. Truncation is applied at write time in Task 2, not left to the database to reject.

- [x] **Step 5: Verify against real PostgreSQL**

Run: `RAGFABRIC_TEST_DATABASE_URL=... uv run pytest packages/core/tests/test_migrations_postgres.py -v`
Expected: upgrade head, downgrade base, upgrade head all pass.

- [x] **Step 6: Commit** `feat: add term_stats, corpus_stats and chunk_search.doc_len for BM25`

---

## Task 2: Term statistics maintenance

**Files:**
- Create: `packages/core/src/ragfabric_core/stores/term_stats.py`
- Test: `packages/core/tests/test_term_stats.py`

**Interfaces:**
- Produces:
  - `record_indexed(db: Session, per_chunk_terms: list[dict[str, int]]) -> None`
  - `record_removed(db: Session, per_chunk_terms: list[dict[str, int]]) -> None`
  - `corpus_stats(db: Session) -> tuple[int, float]` returning `(n_chunks, avgdl)`

This is the task where correctness is easiest to lose, because statistics drift silently. A wrong `df` does not raise; it just ranks badly forever.

**Rules this task must enforce:**

| Rule | Why |
|---|---|
| `df` counts **chunks containing the term**, not occurrences | That is the definition. Counting occurrences inflates IDF for repeated terms |
| Re-indexing an existing chunk must decrement the old terms before incrementing the new | Otherwise editing a document permanently inflates `df` |
| A `df` reaching zero deletes the row | Unbounded growth of dead terms otherwise |
| `n_chunks` and `sum_len` move with the same transaction as `df` | A crash between them leaves IDF computed against the wrong `N` |
| Updates use atomic SQL (`df = df + :delta`), never read-modify-write in Python | Two concurrent ingests would otherwise lose an update |
| `avgdl` returns `1.0` when `n_chunks` is zero | Division by zero in the BM25 denominator |

- [x] **Step 1: Write the failing tests**

```python
def test_df_counts_chunks_not_occurrences(session):
    record_indexed(session, [{"quota": 9}])
    assert session.get(TermStat, "quota").df == 1


def test_reindexing_a_chunk_does_not_inflate_df(session):
    record_indexed(session, [{"quota": 1}])
    record_removed(session, [{"quota": 1}])
    record_indexed(session, [{"quota": 1}])
    assert session.get(TermStat, "quota").df == 1


def test_df_reaching_zero_removes_the_row(session):
    record_indexed(session, [{"gone": 1}])
    record_removed(session, [{"gone": 1}])
    assert session.get(TermStat, "gone") is None


def test_avgdl_is_one_on_an_empty_corpus(session):
    assert corpus_stats(session) == (0, 1.0)
```

- [x] **Step 2: Run them, confirm each fails.**

- [x] **Step 3: Implement.** Use `insert(...).on_conflict_do_update()` on PostgreSQL and a portable upsert elsewhere; the `delta` must be applied in SQL.

- [x] **Step 4: Run tests, confirm they pass.**

- [x] **Step 5: Mutation check.** Change `df = df + delta` to `df = delta` and confirm `test_reindexing_a_chunk_does_not_inflate_df` fails. Revert.

- [x] **Step 6: Commit** `feat: maintain BM25 term and corpus statistics`

---
## Task 3: BM25 scoring

**Files:**
- Create: `packages/core/src/ragfabric_core/stores/bm25_sql.py`
- Modify: `packages/core/src/ragfabric_core/stores/postgres_fts.py` (preserve term frequency off PostgreSQL, populate `doc_len`, call the Task 2 hooks)
- Test: `packages/core/tests/test_bm25_sql.py`

**Interfaces:**
- Consumes: `TermStat`, `CorpusStat`, `ChunkSearch.doc_len` (Task 1); `record_indexed`, `record_removed`, `corpus_stats` (Task 2)
- Produces: `Bm25Store` with `name = "bm25"` satisfying `LexicalStore`; `bm25_score(tf, df, n, doc_len, avgdl, k1, b) -> float` as a pure function

**The formula, exactly.** Implement this and nothing else:

```
score(D, Q) = sum over t in Q of  IDF(t) * ( tf(t,D) * (k1 + 1) )
                                 / ( tf(t,D) + k1 * (1 - b + b * |D| / avgdl) )

IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

The `+ 1` inside the logarithm is the Lucene variant. It exists so IDF can never go negative, which the original Robertson formulation allows for terms present in more than half the corpus. Without it, a common term actively subtracts from the score and a document can rank below one containing none of the query terms.

Defaults: `k1 = 1.2`, `b = 0.75`. Both configurable in Task 9.

| Parameter | Effect | Raising it |
|---|---|---|
| `k1` | term frequency saturation | more reward for repetition |
| `b` | length normalisation strength | penalises long chunks harder. `b = 0` disables it |

- [ ] **Step 1: Write the failing pure function tests**

```python
def test_idf_is_never_negative_for_a_very_common_term():
    # present in 99 of 100 chunks
    assert bm25_score(tf=1, df=99, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75) > 0


def test_a_rarer_term_scores_higher_than_a_common_one():
    rare = bm25_score(tf=1, df=1, n=1000, doc_len=10, avgdl=10, k1=1.2, b=0.75)
    common = bm25_score(tf=1, df=900, n=1000, doc_len=10, avgdl=10, k1=1.2, b=0.75)
    assert rare > common


def test_term_frequency_saturates():
    # going 1 -> 2 must help more than 9 -> 10
    first = bm25_score(tf=2, df=5, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75) - bm25_score(
        tf=1, df=5, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75
    )
    later = bm25_score(tf=10, df=5, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75) - bm25_score(
        tf=9, df=5, n=100, doc_len=10, avgdl=10, k1=1.2, b=0.75
    )
    assert first > later


def test_b_zero_disables_length_normalisation():
    short = bm25_score(tf=1, df=5, n=100, doc_len=5, avgdl=100, k1=1.2, b=0.0)
    long = bm25_score(tf=1, df=5, n=100, doc_len=500, avgdl=100, k1=1.2, b=0.0)
    assert short == long


def test_a_long_chunk_is_penalised_when_b_is_on():
    short = bm25_score(tf=1, df=5, n=100, doc_len=5, avgdl=100, k1=1.2, b=0.75)
    long = bm25_score(tf=1, df=5, n=100, doc_len=500, avgdl=100, k1=1.2, b=0.75)
    assert short > long
```

- [ ] **Step 2: Run them, confirm they fail.**

- [ ] **Step 3: Implement `bm25_score` as a pure function.** No database access. It is the piece most worth testing in isolation and the piece a reviewer must be able to check against the formula above by eye.

- [ ] **Step 4: Preserve term frequency off PostgreSQL.**

In `postgres_fts.py`, the non-PostgreSQL branch currently writes `" ".join(sorted(_tokens(text)))`, where `_tokens` returns a `set`. Replace with a representation that keeps repetition, so `tf` survives. Populate `ChunkSearch.doc_len` on both branches. Call `record_indexed` / `record_removed` so statistics stay consistent with what is indexed.

Existing rows written before this change carry `doc_len = 0` and a set-based `tsv`. Extend the existing `reindex_all` (it already takes a `lexical_store`) and the existing `ragfabric reindex` command with a lexical-only mode, and say plainly in the release notes that lexical search requires a re-index after upgrading. Do not silently treat `doc_len = 0` as valid: it would make the length-normalisation denominator collapse.

- [ ] **Step 5: Write the store integration tests**, including the phase's done-criteria:

```python
def test_an_identifier_outranks_a_common_word(store, corpus):
    # corpus: 'ERR_QUOTA_4419' in one chunk, 'limit' in most
    hits = store.search("what is the retry limit for ERR_QUOTA_4419", top_k=3, access=ALL)
    assert hits[0].chunk_id == corpus["chunk_with_error_code"]


def test_doc_len_zero_rows_are_excluded_not_scored(store, legacy_row):
    assert legacy_row.chunk_id not in {h.chunk_id for h in store.search("anything", 10, ALL)}
```

- [ ] **Step 6: Run against real PostgreSQL** with `RAGFABRIC_TEST_DATABASE_URL` set, and confirm the SQLite path passes too.

- [ ] **Step 7: Commit** `feat: compute BM25 in SQL over term and corpus statistics`

---

## Task 4: Reciprocal Rank Fusion

**Files:**
- Create: `packages/core/src/ragfabric_core/stores/fusion.py`
- Test: `packages/core/tests/test_fusion.py`

**Interfaces:**
- Produces: `rrf(rankings: list[list[RetrievedChunk]], k: int = 60, weights: list[float] | None = None) -> list[RetrievedChunk]`

RRF combines **ranks**, not scores:

```
rrf_score(d) = sum over rankings r of  weight_r / (k + rank_r(d))
```

`rank` is 1-based. `k = 60` is the value from the original Cormack paper and exists to stop the top rank from dominating.

**Why rank and not score.** BM25 is unbounded and corpus-dependent; `ts_rank_cd` is roughly 0 to 1. Normalising them onto a shared scale requires knowing each one's distribution, which changes per query. Ranks need no such assumption. This is the reason fusion is RRF and not a weighted score sum, and the reason must survive into the docstring.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_chunk_ranked_first_by_both_wins():
    a, b = chunk(1), chunk(2)
    assert rrf([[a, b], [a, b]])[0].chunk_id == 1


def test_fusion_promotes_a_chunk_both_rank_second_over_a_one_list_winner():
    # a is 1st in one list and absent from the other; b is 2nd in both
    a, b, c = chunk(1), chunk(2), chunk(3)
    out = rrf([[a, b], [c, b]])
    assert out[0].chunk_id == 2


def test_missing_from_a_ranking_contributes_nothing_rather_than_a_penalty():
    a, b = chunk(1), chunk(2)
    assert {h.chunk_id for h in rrf([[a], [b]])} == {1, 2}


def test_weights_shift_the_outcome():
    a, b = chunk(1), chunk(2)
    assert rrf([[a, b], [b, a]], weights=[10.0, 1.0])[0].chunk_id == 1


def test_the_result_carries_the_fused_score_not_an_input_score():
    a = chunk(1, score=999.0)
    assert rrf([[a]])[0].score != 999.0
```

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** Ties break on `chunk_id` so ordering is deterministic.
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add reciprocal rank fusion for lexical rankings`

---

## Task 5: Exact phrase and identifier boosting

**Files:**
- Create: `packages/core/src/ragfabric_core/stores/boosting.py`
- Test: `packages/core/tests/test_boosting.py`

**Interfaces:**
- Produces: `identifiers(query: str) -> set[str]`, `phrases(query: str) -> list[str]`, `apply_boosts(hits, query, phrase_boost, identifier_boost) -> list[RetrievedChunk]`

This is the other half of the phase's done-criteria. BM25's IDF already favours rare tokens; boosting handles the two cases IDF alone gets wrong.

| Case | Why IDF is not enough |
|---|---|
| Quoted phrase `"retry limit"` | BM25 is a bag of words. It cannot tell adjacent from scattered |
| Identifier `ERR_QUOTA_4419`, `get_user_by_id`, `v2.1.4` | `to_tsvector('english')` stems and splits these, so the exact form is lost |

**Identifier detection.** A token is an identifier if it matches any of: contains a digit **and** a letter; contains `_` between word characters; is `CamelCase` with an internal capital; matches a dotted or hyphenated version pattern. Pure English words never qualify. The rule must be a documented regex set, not a heuristic that "feels right", and each branch needs its own test including a negative case.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize(
    "token,expected",
    [
        ("ERR_QUOTA_4419", True),
        ("get_user_by_id", True),
        ("v2.1.4", True),
        ("CamelCase", True),
        ("retry", False),
        ("limit", False),
        ("The", False),
    ],
)
def test_identifier_detection(token, expected):
    assert (token in identifiers(f"what about {token} here")) is expected


def test_quoted_phrases_are_extracted_verbatim():
    assert phrases('find the "retry limit" now') == ["retry limit"]


def test_a_chunk_containing_the_exact_phrase_is_boosted():
    near = chunk(1, text="the retry limit is 5")
    far = chunk(2, text="retry ... limit")
    out = apply_boosts([far, near], 'the "retry limit"', phrase_boost=2.0, identifier_boost=3.0)
    assert out[0].chunk_id == 1


def test_boosting_never_invents_a_hit_that_was_not_retrieved():
    assert apply_boosts([], "ERR_QUOTA_4419", 2.0, 3.0) == []
```

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** Boosts multiply an existing hit's score. They never add a chunk to the result set: retrieval decides membership, boosting only reorders. A boost that could introduce a chunk would bypass the access filter, which ADR 0003 forbids.
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: boost exact phrases and identifiers in lexical ranking`

---

## Task 6: Metadata filters on the lexical path

**Files:**
- Modify: `packages/core/src/ragfabric_core/stores/bm25_sql.py`
- Test: `packages/core/tests/test_bm25_filters.py`

**Interfaces:**
- Consumes: the `filters` dict already honoured by `PostgresLexicalStore.search` (`document_id`, `collection_id`, `format`)

The BM25 store must honour the same filter keys as `postgres_fts`, applied **inside** the SQL query alongside `access_clause`, per ADR 0003.

- [ ] **Step 1: Write the failing tests**, including the one that matters most:

```python
def test_filters_are_applied_before_top_k_not_after(store, corpus):
    # 10 matching chunks, only 2 in collection 7
    hits = store.search("quota", top_k=5, access=ALL, filters={"collection_id": 7})
    assert len(hits) == 2
    assert all(h.collection_id == 7 for h in hits)


def test_an_unknown_filter_key_is_rejected_not_ignored(store):
    with pytest.raises(ValueError):
        store.search("quota", top_k=5, access=ALL, filters={"nonsense": 1})
```

The second test is deliberate. `postgres_fts.search` currently ignores unknown filter keys silently, which means a typo returns unfiltered results that look plausible. The BM25 store must reject them. Whether to also tighten `postgres_fts` is a judgement call for the implementer: tighten it and note the behaviour change, or leave it and record why.

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: apply metadata filters inside the BM25 query`

---
## Task 7: VectorlessRAGStrategy

**Files:**
- Create: `packages/core/src/ragfabric_core/strategies/vectorless.py`
- Modify: `packages/core/src/ragfabric_core/strategies/registry_defaults.py`
- Test: `packages/core/tests/test_strategy_vectorless.py`

**Interfaces:**
- Consumes: `Bm25Store` (Task 3), `rrf` (Task 4), `apply_boosts` (Task 5), the Phase 3 citation contract in `generate/contract.py`
- Produces: `VectorlessRAGStrategy(name="vectorless")` satisfying `RetrieverStrategy`

**Order of operations, which is not negotiable:**

```
query
  -> BM25 store query      (access filter INSIDE, filters INSIDE)
  -> ts_rank_cd query      (access filter INSIDE, filters INSIDE)
  -> RRF fuse the two ranked lists
  -> apply phrase and identifier boosts
  -> cut to top_k
  -> token budget
  -> generate with citations (Phase 3 contract, unchanged)
```

Read that against `strategies/traditional.py` before writing anything. The shape is deliberately the same, minus the embedding call. The differences that matter:

| | Traditional | Vectorless |
|---|---|---|
| Embedding call | yes | **none**. `llm_calls` must not count one |
| Similarity threshold | yes | no. BM25 scores are unbounded, so a fixed floor is meaningless |
| Store queries | one | two, fused |

**On `llm_calls` and cost.** Vectorless makes no embedding call at all. Report that honestly: the embedding count is zero, not omitted and not defaulted to one. Per ADR 0004, `estimated_cost_usd` stays `None` when unknown. Phase 3 shipped eight fabricated metrics before they were caught; do not add a ninth here.

- [ ] **Step 1: Write the failing tests**

```python
def test_vectorless_makes_no_embedding_call(strategy, spy_embeddings):
    strategy.retrieve("anything", access=ALL, top_k=5)
    assert spy_embeddings.calls == 0


def test_access_filtered_chunks_never_reach_the_ranker(strategy, restricted_access):
    hits = strategy.retrieve("quota", access=restricted_access, top_k=10)
    assert all(h.collection_id in restricted_access.collection_ids for h in hits)


def test_top_k_is_applied_after_fusion_not_per_store(strategy):
    # each store returns 5; fused and cut to 3
    assert len(strategy.retrieve("quota", access=ALL, top_k=3)) == 3


def test_an_answer_carries_citations_that_pass_the_phase_3_contract(strategy):
    answer = strategy.ask("what is the retry limit for ERR_QUOTA_4419", access=ALL)
    assert answer.citations
    assert_citation_contract(answer.text, answer.chunks)  # real API, do not reimplement
```

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**, registering `vectorless` in `registry_defaults.py`.
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Mutation check.** Move the `top_k` cut to before fusion and confirm `test_top_k_is_applied_after_fusion_not_per_store` fails. Revert.
- [ ] **Step 6: Commit** `feat: add VectorlessRAGStrategy over fused BM25 and ts_rank`

---

## Task 8: Optional in-process BM25 with an enforced cap

**Files:**
- Create: `packages/core/src/ragfabric_core/stores/bm25_memory.py`
- Modify: `packages/core/pyproject.toml` (add a `bm25` extra for `rank_bm25`)
- Test: `packages/core/tests/test_bm25_memory.py`

**Interfaces:**
- Produces: `InMemoryBm25Store` with `name = "bm25_memory"` satisfying `LexicalStore`

This exists for small corpora and for cross-checking the SQL implementation against a reference. It is **not** the production path and the code must say so.

**The cap is the point of this task.** Without it, this store silently becomes the thing that falls over in production. Requirements:

| Requirement | Behaviour |
|---|---|
| `max_chunks` config, default `50_000` | Exceeding it raises `StoreCapacityError`, never degrades quietly |
| Error message names the limit and the alternative | "in-memory BM25 holds N chunks, limit is M, use `lexical_store.kind: bm25` for larger corpora" |
| Docstring states the memory model plainly | Each worker holds its own copy; copies drift as documents change |
| `rank_bm25` is an optional extra | Not a hard dependency, mirroring `chroma` and `rerank` |

- [ ] **Step 1: Write the failing tests**

```python
def test_exceeding_the_cap_raises_rather_than_degrading(store_with_cap_of_2):
    with pytest.raises(StoreCapacityError) as exc:
        store_with_cap_of_2.index([1, 2, 3], ["a", "b", "c"], payloads)
    assert "2" in str(exc.value) and "limit" in str(exc.value).lower()


def test_it_agrees_with_the_sql_implementation_on_a_small_corpus(mem_store, sql_store, corpus):
    q = "ERR_QUOTA_4419 retry limit"
    assert [h.chunk_id for h in mem_store.search(q, 5, ALL)] == [
        h.chunk_id for h in sql_store.search(q, 5, ALL)
    ]
```

The second test is the valuable one: it is a differential test that catches a formula error in either implementation. If the two disagree, one of them is wrong, and the pure `bm25_score` tests from Task 3 say which.

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** Skip cleanly with `pytest.importorskip("rank_bm25")` so the suite still runs without the extra.
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: add capped in-process BM25 store for small corpora`

---

## Task 9: Configuration

**Files:**
- Modify: `packages/core/src/ragfabric_core/config_file.py`, `ragfabric.example.yaml`
- Test: `packages/core/tests/test_config_file.py`

**Interfaces:**
- Produces: a typed `vectorless` strategy config

`StrategiesConfig.vectorless` is currently an untyped `dict[str, float | int | str | bool]` defaulting to `{"top_k": 8, "phrase_boost": 2.0, "identifier_boost": 3.0}`. That dict accepts any key, so a typo in `k1` would be accepted and ignored, which contradicts the file's own stated principle that "a typo that silently falls back to a default is the worst kind of configuration bug".

Replace it with a strict model:

```python
class VectorlessConfig(_Strict):
    top_k: int = Field(default=8, ge=1)
    k1: float = Field(default=1.2, ge=0.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    phrase_boost: float = Field(default=2.0, ge=1.0)
    identifier_boost: float = Field(default=3.0, ge=1.0)
    fusion_k: int = Field(default=60, ge=1)
    fusion_weights: tuple[float, float] = (1.0, 1.0)
    max_context_tokens: int = Field(default=6000, ge=100)
```

`LexicalStoreConfig.kind` already allows `"postgres_fts" | "bm25"`. Add `"bm25_memory"` for Task 8 and a `max_chunks` field.

**Bounds are deliberate.** `b` is constrained to `[0, 1]` because outside that range the length-normalisation term is meaningless: negative `b` rewards long chunks, and `b > 1` can drive the denominator negative. `phrase_boost` and `identifier_boost` have a floor of `1.0` because a boost below one is a penalty, and if anyone wants that they should say so with a different key.

- [ ] **Step 1: Write the failing tests**

```python
def test_an_unknown_vectorless_key_is_rejected():
    with pytest.raises(ValidationError):
        VectorlessConfig(k_1=1.2)  # typo


def test_b_outside_zero_to_one_is_rejected():
    with pytest.raises(ValidationError):
        VectorlessConfig(b=1.5)


def test_a_boost_below_one_is_rejected():
    with pytest.raises(ValidationError):
        VectorlessConfig(phrase_boost=0.5)


def test_defaults_match_the_documented_bm25_defaults():
    c = VectorlessConfig()
    assert (c.k1, c.b, c.fusion_k) == (1.2, 0.75, 60)
```

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement**, and update `ragfabric.example.yaml` with every key and a one-line comment on what it does.
- [ ] **Step 4: Verify `ragfabric config validate` reports the vectorless settings.**
- [ ] **Step 5: Commit** `feat: type the vectorless strategy configuration`

---

## Task 10: API and CLI surface

**Files:**
- Modify: `packages/server/src/ragfabric_server/api/routes/search.py`, `ask.py`
- Modify: `packages/cli/src/ragfabric_cli/` (the `ask` and `search` commands)
- Modify: `packages/sdk-python/src/ragfabric_sdk/`
- Test: `packages/server/tests/test_vectorless_routes.py`, `packages/cli/tests/test_cli_vectorless.py`

`POST /v1/search` and `POST /v1/ask` must accept `strategy: "vectorless"` and route to the new strategy.

**A Phase 3 defect to avoid repeating.** Task 12 of Phase 3 was dispatched on the claim that `SearchRequest.similarity_threshold` existed. It did not, which meant a shipped route could not honour a documented parameter. **Before writing anything, read the actual request models and confirm every field this task references exists.** If a field is missing, add it in this task and say so in the report rather than assuming.

- [ ] **Step 1: Read `SearchRequest` and `AskRequest` and write down their real fields.**
- [ ] **Step 2: Write the failing tests**

```python
def test_search_accepts_the_vectorless_strategy(client, token):
    r = client.post("/v1/search", json={"query": "ERR_QUOTA_4419", "strategy": "vectorless"}, headers=auth(token))
    assert r.status_code == 200
    assert r.json()["strategy"] == "vectorless"


def test_vectorless_reports_zero_embedding_calls(client, token):
    r = client.post("/v1/ask", json={"query": "x", "strategy": "vectorless"}, headers=auth(token))
    assert r.json()["usage"]["embedding_calls"] == 0


def test_an_unknown_strategy_is_a_422_not_a_500(client, token):
    assert client.post("/v1/search", json={"query": "x", "strategy": "nope"}, headers=auth(token)).status_code == 422
```

- [ ] **Step 3: Run them, confirm they fail.**
- [ ] **Step 4: Implement** across route, CLI and SDK.
- [ ] **Step 5: Run the full suite** and confirm no Phase 3 route test regressed.
- [ ] **Step 6: Commit** `feat: expose the vectorless strategy over the API, CLI and SDK`

---
## Task 11: Close the API gaps the console needs

**Files:**
- Modify: `packages/server/src/ragfabric_server/api/routes/admin.py`, `access.py`, `collections.py`
- Create: `packages/server/src/ragfabric_server/api/routes/providers.py`
- Test: `packages/server/tests/test_admin_user_crud.py`, `test_group_crud.py`, `test_providers_route.py`

The console cannot "manage access without DB edits" until these exist. Current state, verified by reading the routers:

| Area | Exists | Missing |
|---|---|---|
| Users | `GET /users`, `PUT /users/{id}/permissions` | create, delete |
| Groups | `POST`, `GET`, member add, member remove | update, delete |
| Grants | `POST`, `GET`, `DELETE` | nothing |
| API keys | `POST`, `GET`, `DELETE` | nothing |
| Collections | `GET`, `POST`, `GET {id}`, `DELETE` | update |
| Providers | nothing | read, write |

**Provider configuration is the sensitive one.** It touches API keys for OpenAI and Anthropic. Rules:

- The read endpoint returns provider kind, model, base URL and **whether a secret is set**. It never returns the secret, not even masked in a way that leaks length.
- Secrets continue to live in the environment, per `config.py`. This endpoint reports and selects; it does not become a secret store.
- Writing provider config requires admin, and every write goes to the audit log that Phase 2 built.

- [ ] **Step 1: Write the failing tests**, including these:

```python
def test_deleting_a_user_revokes_their_api_keys(client, admin):
    ...
    assert client.get("/v1/keys", headers=auth(victim_key)).status_code == 401


def test_provider_config_never_returns_the_secret(client, admin):
    body = client.get("/v1/providers", headers=auth(admin)).json()
    assert "sk-" not in json.dumps(body)
    assert body["llm"]["has_key"] is True


def test_a_non_admin_cannot_write_provider_config(client, member):
    assert client.put("/v1/providers", json={...}, headers=auth(member)).status_code == 403


def test_deleting_a_group_removes_its_grants_not_its_users(client, admin):
    ...
```

- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** Deleting a user or group must not orphan rows: decide cascade versus restrict per relationship and state the choice in the report.
- [ ] **Step 4: Run tests, confirm they pass.**
- [ ] **Step 5: Commit** `feat: complete the admin API for console v1`

---

## Task 12: Console design foundation

**Files:**
- Create: `apps/assistant/src/app/ui/` (tokens, primitives)
- Modify: `apps/assistant/src/styles.scss`, the app shell
- Test: `apps/assistant/src/app/ui/*.spec.ts`

The stack is **Tailwind 4 with no component library**. That is a deliberate advantage here: the look is ours to set rather than something to override.

**The brief is a clean, modern, genuinely good-looking console.** Concretely, and testably:

| Concern | Requirement |
|---|---|
| Design tokens | Colour, spacing, radius and shadow as CSS custom properties on `:root`. No hard-coded hex values in components |
| Dark mode | Full support via `prefers-color-scheme`, overridable by an explicit toggle. Every surface defines a background |
| Primitives | Button, Input, Select, Table, Modal, Toast, Badge, EmptyState, ConfirmDialog. Each standalone, each with a spec |
| Density | Tables are the main surface. Comfortable and compact modes |
| Responsiveness | Usable from 360px up. No horizontal page scroll |
| Accessibility | Focus visible everywhere, labels tied to inputs, modals trap focus and restore it on close, colour is never the only signal |
| Loading and error | Every async surface has a skeleton and an error state. No spinner-only screens, no silent failures |

**Do not** introduce Angular Material or PrimeNG. Mixing a component library into an established Tailwind build means fighting two styling systems, and the visual result is usually worse than either alone.

- [ ] **Step 1: Write failing specs for the primitives** (render, disabled state, keyboard interaction, focus trap on Modal).
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement tokens and primitives.**
- [ ] **Step 4: Run `npm test`, confirm all pass and the existing 20 tests still pass.**
- [ ] **Step 5: Run `npm run build` and confirm the bundle budget is not exceeded.** If it is, raise the budget deliberately and say why, or split. Do not silently disable the budget.
- [ ] **Step 6: Commit** `feat: add console design tokens and UI primitives`

---

## Task 13: Users and Groups screens

**Files:**
- Create: `apps/assistant/src/app/pages/console/users/`, `groups/`
- Test: co-located specs

Full CRUD on both, built on Task 12's primitives, against Task 11's API.

| Screen | Must support |
|---|---|
| Users | list with search and pagination, create, edit permissions, delete with confirmation, empty state |
| Groups | list, create, rename, delete with confirmation, add and remove members inline |

Destructive actions require a typed confirmation, not a bare "are you sure". Deleting a user is not undoable and the dialog must say what else it removes.

- [ ] **Step 1: Write failing specs** including `test_delete_requires_confirmation` and `test_a_failed_request_shows_an_error_not_a_blank_table`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `npm test` and `npm run build`.**
- [ ] **Step 5: Commit** `feat: add console users and groups screens`

---

## Task 14: Collections and Grants screens

**Files:**
- Create: `apps/assistant/src/app/pages/console/grants/`
- Modify: `apps/assistant/src/app/pages/collections/` (add update, bring onto the new primitives)
- Test: co-located specs

Grants are the screen that decides who can read what. It must make the effective permission obvious rather than leaving the operator to infer it:

- Show grants by subject (user or group) **and** by collection. Both directions.
- Show the effective access for a chosen user, resolved through group membership, because that is the question an operator actually has.
- Creating a grant that already exists is reported clearly, not duplicated silently.

- [ ] **Step 1: Write failing specs** including `test_effective_access_resolves_through_group_membership`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `npm test` and `npm run build`.**
- [ ] **Step 5: Commit** `feat: add console collections and grants screens`

---

## Task 15: API keys screen

**Files:**
- Create: `apps/assistant/src/app/pages/console/keys/`
- Test: co-located specs

Security-sensitive, so it gets its own task and its own review.

| Rule | Reason |
|---|---|
| The secret is shown **once**, at creation, and never again | It is stored hashed. There is nothing to show later, and pretending otherwise teaches operators a false expectation |
| The one-time panel has an explicit copy action and a "I have saved this" acknowledgement before it closes | Closing by accident loses the key |
| The list shows prefix, scopes, creation date, last used and expiry. Never the secret | |
| Revocation is immediate and confirmed | |
| The secret is never written to `localStorage`, a URL, or a log | |

- [ ] **Step 1: Write failing specs** including `test_the_secret_is_not_present_in_the_dom_after_the_panel_closes` and `test_the_secret_is_never_written_to_local_storage`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `npm test` and `npm run build`.**
- [ ] **Step 5: Commit** `feat: add console API keys screen`

---

## Task 16: Provider configuration screen

**Files:**
- Create: `apps/assistant/src/app/pages/console/providers/`
- Test: co-located specs

Selects LLM and embedding providers, models and base URLs, against Task 11's endpoint.

- Shows whether a secret is present, never the secret itself.
- Changing the embedding provider or model must warn, loudly, that the embedding dimension is pinned (ADR 0006) and that changing it requires a migration and a re-index. This is the single most destructive action in the console and the UI must treat it that way.
- A "test connection" action reports the real result. It never reports success without a successful call.

- [ ] **Step 1: Write failing specs** including `test_changing_the_embedding_model_warns_about_reindexing`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `npm test` and `npm run build`.**
- [ ] **Step 5: Commit** `feat: add console provider configuration screen`

---

## Task 17: docs/concepts/lexical-vs-vector.md and learning notes

**Files:**
- Create: `docs/concepts/lexical-vs-vector.md`, `docs/learning/lexical-vs-semantic.md`

The issue asks for the BM25 formula with `tf`, `idf`, `k1`, `b`, and an honest account of where lexical wins and fails. Required content:

- The formula as written in Task 3, with each symbol defined in a table.
- What `k1` and `b` do, with the effect of moving each.
- Why the `+ 1` inside the IDF logarithm exists.
- **Where lexical wins:** identifiers, error codes, exact phrases, rare proper nouns, numbers, and any corpus where the user's vocabulary matches the document's.
- **Where lexical fails:** paraphrase, synonymy, cross-lingual, conceptual questions, and questions whose answer never repeats the question's words.
- Why `ts_rank_cd` is not BM25, stated plainly.
- Why RagFabric implements BM25 in SQL rather than depending on `pg_search`.
- A worked example on the fixture corpus showing one query each way.

Write it for someone who has not read the code. No unexplained jargon.

- [ ] **Step 1: Write both documents.**
- [ ] **Step 2: Verify every claim against the implementation.** Any number quoted must come from a real run. Per ADR 0004, do not invent a benchmark.
- [ ] **Step 3: Commit** `docs: explain BM25 and when lexical retrieval beats vectors`

---

## Task 18: ADRs and documentation sync

**Files:**
- Create: `docs/adr/0007-bm25-in-sql.md`, `docs/adr/0008-rrf-over-score-fusion.md`
- Modify: `README.md`, `ROADMAP.md`, `ragfabric.example.yaml`, `docs/` as needed

ADR 0007 records: BM25 computed in SQL, `pg_search` rejected for managed-PostgreSQL portability, no `chunk_terms` table, `tf` reused from `tsvector`, and the re-index requirement for corpora indexed before this phase.

ADR 0008 records: fusion combines ranks not scores, and why normalising BM25 against `ts_rank_cd` is not viable.

**A Phase 3 process failure to avoid.** Two implementer agents were once dispatched in parallel into the same worktree; the result was documentation that described an already-fixed bug as unfixed. **One implementer at a time.** Documentation is written after the code it describes is final, and every claim in it is checked against the tree as it then stands.

- [ ] **Step 1: Write both ADRs.**
- [ ] **Step 2: Update `ROADMAP.md`, ticking Phase 4 only for what actually merged.**
- [ ] **Step 3: Re-read every changed doc against the code.**
- [ ] **Step 4: Commit** `docs: record the BM25 and fusion decisions`

---

## Task 19: Release v0.1.0

**Files:**
- Create: `CHANGELOG.md`, `.github/workflows/release.yml`
- Modify: version fields across the packages

**The engineer has authorised tagging and publishing automatically once CI is green.** That authorisation is recorded here so the executing agent does not re-ask. It covers this release only.

- [ ] **Step 1: Write `CHANGELOG.md`** covering Phases 1 to 4, in Keep a Changelog form. Every entry traceable to a merged commit. No entry for anything not merged.
- [ ] **Step 2: Set the version to `0.1.0`** across `packages/core`, `packages/server`, `packages/cli`, `packages/sdk-python`, and confirm the lock agrees.
- [ ] **Step 3: Add the GHCR workflow.** Build and push the API and worker images on a `v*` tag. Multi-arch `linux/amd64,linux/arm64`. Images tagged with both the version and the commit SHA.
- [ ] **Step 4: Verify the images build and run locally before tagging.** The Phase 3 Docker defect, where the default Ollama config could not run because the OpenAI SDK had moved to an extra, was found only because someone ran the image. Run it.
- [ ] **Step 5: Confirm all CI is green on `main`.**
- [ ] **Step 6: Tag `v0.1.0`, push it, and confirm the workflow published the images.** Report the real digests.
- [ ] **Step 7: Create the GitHub release** from the changelog.
- [ ] **Step 8: Commit and close** `chore: release v0.1.0`

---

## Self-Review

**Spec coverage against issue #5:**

| Issue requirement | Task |
|---|---|
| `VectorlessRAGStrategy`, BM25 plus tsvector, fusion | 3, 4, 7 |
| Exact phrase and identifier boosting | 5 |
| Metadata filter | 6 |
| `docs/concepts/lexical-vs-vector.md` with the BM25 formula | 17 |
| Console v1: users, groups, collections, grants, keys, providers | 11 to 16 |
| Tag v0.1.0, changelog, GHCR images | 19 |
| Learning notes | 17 |
| Done: identifier questions rank correctly | 3 (test), 5 (boosting), 17 (worked example) |
| Done: console manages access without DB edits | 11 (API gaps), 13, 14 |
| Done: v0.1.0 tagged | 19 |

**Interface consistency:** `bm25_score` is defined in Task 3 and consumed in Task 8's differential test. `rrf` is defined in Task 4 and consumed in Task 7. `apply_boosts` is defined in Task 5 and consumed in Task 7. `record_indexed` and `record_removed` are defined in Task 2 and consumed in Task 3. `VectorlessConfig` is defined in Task 9 and consumed in Tasks 7 and 10. No name is used before it is defined.

**Known ordering constraint:** Task 3 modifies `postgres_fts.py`, and Task 6 modifies `bm25_sql.py`, which Task 3 creates. Tasks 1, 2, 3 must run in order. Tasks 4 and 5 are independent of each other and of 1 to 3, but Task 7 needs all of them. Tasks 11 to 16 depend only on Task 11 and each other's primitives from Task 12.

**Deliberate omissions:** no hybrid lexical-plus-vector retrieval (that is the Phase 7 router's job), no query rewriting (Phase 5), no evaluation harness (Phase 8).
