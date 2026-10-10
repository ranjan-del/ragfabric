# The assistant: Ask, Compare, Trace, Evaluation

> Status: Phase 9 (issue #10), on the phase branch for v1.0.0. The Evaluation page reads the
> evaluation API that Phase 8 adds; until a server has it, the page says so.

The reference UI in `apps/assistant` is an Angular 22 app that talks to the server only through
[`@ragfabric/sdk`](../packages/sdk-typescript/README.md). It has the console pages from earlier
releases (documents, collections, analytics, users, groups, grants, keys, providers) and four pages
for asking questions and reading the results.

## Ask

One question, one streamed answer, and why that strategy answered it.

| Part | What it shows |
|---|---|
| Mode | **AUTO** lets the router choose (`strategy: "auto"`). **MANUAL** sends the strategy you pick |
| Router card | The strategy that answered; whether a rule or the classifier chose it; the classifier's confidence as reported (uncalibrated), or "rule based, no confidence measured"; the reasoning; query type, complexity, expected cost and latency. A fallback is always called out: the router chose one strategy, it found nothing, traditional answered |
| Answer | Streams token by token. If the server corrects it (a `superseded` event: the streamed text failed the citation contract), the text is replaced by the recorded answer and a notice says why, with any removed claims |
| Citations | Every `[n]` in the answer and every source chip is a button. Cited sources come first; sources retrieved but not cited are marked |
| Source viewer | File, page, score (or "not measured" for graph traversal), the passage with your question's words highlighted and the quoted sentence underlined, and a download of the original |
| After the answer | Latency, LLM, retrieval and embedding calls, tokens, and a link to the trace |

## Compare

One question sent to traditional, vectorless, agentic and graph, side by side. Each column is an
ordinary `/api/ask` with the strategy named, so each is recorded as its own run.

| Column field | Source | Notes |
|---|---|---|
| Answer and sources | The stream | Same rendering as Ask |
| Latency, calls, tokens | The `done` event | Measured by the server |
| Cost (estimate) | `GET /api/runs/{id}` | `n/a` until the server prices the run's model (Phase 8 writes the estimate) |
| Benchmark score (latest batch) | `GET /api/eval/runs` | Mean correctness of that strategy in the latest evaluation batch. **Not a score for this question**: an ad hoc question has no expected answer. Admins only, because the evaluation API is admin only |

The strategies run one after another by default. A local model serves one request at a time, so four
at once would measure the queue as much as the strategy. "Run all four at once" is there for hosted
providers, and the page then labels the latencies as measured under contention. A strategy that cannot
run (for example agentic with no model configured) fails its own column with the server's reason; the
other three still answer. The summary badges mark the fastest, the cheapest (only among priced runs)
and the best benchmark score.

## Trace

`/trace/{run id}`, linked from Ask and Compare. It reads the stored run:

- the question, mode, requested and selected strategy, fallback, router confidence and reasoning,
  models;
- total, retrieval and generation latency, calls, tokens and the estimated cost;
- every stored span as a waterfall in start order with its attributes (for agentic runs, these are
  the agent's steps);
- the sources by rank, cited or not, and the recorded answer.

The walked sub graph and the agent's sub questions are **not stored with a run**. Trace shows them for
runs asked in the same browser session and says so for any other run. Storing them needs a migration
and is out of scope for Phase 9.

## Evaluation

Admin only. Reads Phase 8's evaluation API.

- **Batch table**: one row per target with precision, recall, hit rate, MRR, correctness,
  faithfulness, context relevance, citation correctness, p50 and p95 latency, estimated cost (with
  the number of unpriced questions) and errors. Skipped targets are listed with their reason. The
  judge kind, model and rubric version are always shown, with the commit and models.
- **Per category**: the same metrics for one target, one row per question category.
- **Live system** (last 7, 30 or 90 days of real queries): latency p50 and p95 per strategy, estimated
  cost per day with unpriced runs counted separately, fallback rate, calls per strategy, and the
  quality trend per strategy across batches.

Every value the API returns as null is shown as `n/a`, and a null point breaks a trend line rather
than drawing a zero.

## For developers

```bash
cd packages/sdk-typescript && npm ci && npm test   # the SDK, offline
cd apps/assistant && npm ci
npm run check:sdk-only    # fails if any app file bypasses the SDK
npm test                  # Karma specs (headless Chrome)
npm start                 # http://localhost:4200, proxying /api to the server
```

See ADR 0016 for why the SDK is built the way it is.
