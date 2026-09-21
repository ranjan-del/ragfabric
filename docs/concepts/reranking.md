# Reranking, concept notes

> Status: Phase 3 (shipped). This is a learning document: why reranking is a separate, optional stage
> and how the two implementations differ, not an API reference. For the code, see
> `packages/core/src/ragfabric_core/rerank/`.

## Why a cheap ranking plus an expensive re-ranking beats one expensive ranking

Vector search compares the question's embedding against every chunk's embedding independently: each
side is encoded once, in isolation, and the two vectors are compared with one cheap operation (a
normalised dot product). That is what makes it fast enough to run over a whole corpus, but it is also
exactly what limits its accuracy: the model never actually reads the question and the passage
together, so it cannot notice a subtlety that only shows up when both are considered at once, such as a
passage that shares the question's vocabulary but answers a different question entirely.

Running a more accurate, more expensive comparison over the *whole* corpus for every query would be
correct but far too slow to serve. Reranking is the standard answer to that trade-off: use the cheap,
independent comparison to narrow a huge corpus down to a small candidate set (`top_k * 3` candidates by
default in this codebase, see [traditional-rag.md](../traditional-rag.md)), then spend the expensive,
joint comparison only on that small set. You get most of the accuracy of "compare everything expensively"
for a cost proportional to the candidate count, not the corpus size.

## Cross encoder versus bi encoder

The embedding model used for vector search is a **bi-encoder**: the question and a chunk are each
encoded into a vector independently ("bi", two separate passes), and compared afterward with cheap
arithmetic. Nothing about encoding the question knows anything about the chunk, or vice versa.

A **cross encoder** (`rerank/cross_encoder.py`, `CrossEncoderReranker`) reads the question and one
candidate passage together, in a single forward pass, and outputs a single relevance score for that
pair directly. Because the model attends across both texts at once, it can pick up on interactions a
bi-encoder structurally cannot: whether a specific number in the passage actually answers the specific
quantity the question asks about, for example, rather than the two texts merely sharing a topic. The
cost is that a cross encoder cannot be indexed the way embeddings can: there is no way to pre-compute a
"chunk vector" that will later be compared cheaply against an arbitrary future question, because the
comparison only exists once both texts are known. Every rerank is a fresh forward pass per candidate,
which is why it only ever runs over a narrowed-down set, never the whole corpus.

This codebase also ships `LlmReranker` (`rerank/llm_reranker.py`), which reranks the same way
conceptually, question and passage read together, but by asking the configured chat LLM to score each
candidate with a JSON response (`{"scores": [0.0, 1.0, ...]}`) instead of running a dedicated scoring
model. It is more flexible (any configured LLM, no separate model to install) and typically slower and
more expensive per candidate than a small local cross encoder.

## When reranking does not pay

Reranking is a pure cost, in latency and (for `llm`) money, on top of retrieval; it is worth that cost
only when it changes the outcome often enough to matter. It tends not to pay when:

- **The corpus or the question set is easy.** If the top vector-search result is already almost always
  the right one, for example a small, topically narrow corpus, reranking mostly re-confirms an order
  that was already correct and adds latency for no measurable gain.
- **Latency budget is tight.** A cross encoder forward pass, or a whole extra LLM call, is not free; for
  a use case where sub-second response matters more than squeezing out the last few points of ranking
  quality, it is the wrong trade.
- **The candidate set handed to it is already bad.** Reranking reorders what vector search retrieved;
  it cannot recover a genuinely relevant chunk that never made it into the candidate set in the first
  place (a retrieval miss, not a ranking miss). A low `similarity_threshold` or a small `top_k` upstream
  will not be fixed by reranking downstream.

Whether reranking actually helps *this* deployment's questions and corpus is exactly the kind of claim
this project will not make without measuring it (ADR 0004): Phase 8 tracks measuring `cross_encoder`
reranking against `llm` and `none` with real numbers, rather than asserting it here.

## The candidate cap, and the real reason it exists

`LlmReranker` refuses to send an unbounded number of candidates in one prompt; it caps how many
candidates it will ever score in a single call (`max_candidates`, 12 by default), independent of how
many the caller hands it. This is not an arbitrary safety limit dressed up as a design decision: some
model runtimes, Ollama's included, have a context window (`num_ctx`) that commonly defaults to 2048 or
4096 tokens, and when a prompt exceeds it, the runtime **silently truncates the prompt** rather than
rejecting the request. The model then returns a syntactically valid, correctly sized JSON score array,
one score per expected candidate, computed against a prompt that, past the truncation point, no longer
contained the actual text of some of those candidates. Nothing in the response shape reveals this: the
array has the right length, the right types, scores in range. It looks exactly like a successful
rerank and is actually a rerank partly computed against passages the model never saw. Capping how many
candidates are ever built into the prompt in the first place is the only way to rule this failure mode
out, because it cannot be detected after the fact from the response alone.
