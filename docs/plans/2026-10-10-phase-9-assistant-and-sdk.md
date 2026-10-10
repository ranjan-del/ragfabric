# Phase 9: the assistant UI and the TypeScript SDK. Design

> Status: design for Phase 9 (issue #10, milestone v1.0.0). The task plan follows in the second half
> of this file once it is written. Written 2026-10-10 under the owner's pre-approval to design,
> record every decision with its reason, plan and build on a feature branch. Every decision below is
> open to the owner's review before the branch is merged.

**Goal:** A person can ask a question in the browser and watch the answer stream in with the router's
reasoning and clickable citations, put the same question to all four strategies side by side with
their latency, calls, tokens, cost and benchmark score, open the full trace of any run, and read the
latest evaluation results, all through `@ragfabric/sdk`, a TypeScript SDK whose types are generated
from the server's OpenAPI specification.

**Issue:** https://github.com/ranjan-del/ragfabric/issues/10

**Predecessor:** the v0.4.0 release (#61) and the frontend dependency bumps (#57, #63).

**Runs alongside:** Phase 8 (issue #9, branch `feat/phase-8-evaluation`), which is being built at the
same time. Phase 8 ships the evaluation API and the dashboard data; this phase owns the pages that
draw them (Phase 8 design, decision D17). See "Dependency on Phase 8" below.

---

## What exists today

| Piece | State on main (c9d5035) |
|---|---|
| `apps/assistant` | Angular 22 console v1: login, dashboard, documents, collections, analytics, admin, console (users, groups, grants, keys, providers), and a v1 Search page that calls `POST /api/search/query`. 95 Karma specs |
| HTTP in the app | Eight hand written services over `HttpClient`, hand written types in `models.ts`, one auth interceptor (bearer token, 401 redirect) |
| `POST /api/ask` | Server sent events: `retrieval` (strategy, trace, sub_questions, subgraph, router, fallback_from), `token`*, optional `superseded`, `citations`, `done` (run_id, latency_ms, usage). `stream: false` returns `AnswerResponse`. Declared `response_model=None`, so the OpenAPI specification does not describe the response |
| `GET /api/runs/{id}` | The stored run: latency split, calls, tokens, `estimated_cost_usd` (always null on main, written by Phase 8 Task 10), trace spans, sources. The router's confidence and reasoning are stored but not returned |
| OpenAPI | Generated at runtime by FastAPI. No snapshot is checked in |
| `packages/sdk-python` | Hand written `httpx` client (`ask`, `ask_stream`, `search`, `ingest`, `documents`, `run`) |
| `@ragfabric/sdk` | Name reserved in the plan, nothing published, no package in the repository |

---

## Design

### Units

| Unit | Path | What it does |
|---|---|---|
| Spec export | `packages/server/src/ragfabric_server/openapi.py` | `python -m ragfabric_server.openapi [PATH]` writes the server's OpenAPI document deterministically (sorted keys, two space indent, trailing newline) |
| Spec snapshot | `packages/sdk-typescript/openapi.json` | The checked in specification the SDK is generated from |
| Generator | `packages/sdk-typescript/scripts/generate.mjs` | Zero dependency script: `openapi.json` to `src/generated/schema.ts` (one exported type per component schema, plus a `paths` map of operations) |
| Generated types | `packages/sdk-typescript/src/generated/schema.ts` | Checked in, never edited by hand |
| Public types | `src/types.ts` | Friendly names over the generated types (`Answer`, `Citation`, `Run`, `RouterDecision` ...) and the hand typed `AskEvent` union |
| Request builders | `src/requests.ts` | One function per endpoint returning an `ApiRequest<T>` (method, path with query, body, headers, response kind). The only place in the repository's TypeScript that spells an API path |
| SSE decoder | `src/sse.ts` | Incremental server sent events decoder and `toAskEvent` |
| Errors | `src/errors.ts` | `RagFabricError` (status, detail, body) and `describeDetail` (FastAPI `detail` to one readable line) |
| Fetch client | `src/client.ts` | `RagFabricClient`: executes builders with `fetch`, `ask`, `askStream` (an async iterator), `run`, and the rest, for scripts, Node and other front ends |
| Evaluation types | `src/evaluation.ts` | Phase 8's API shapes, hand typed and marked provisional until Phase 8 merges and the spec carries them |
| Angular transport | `apps/assistant/src/app/sdk/sdk-http.service.ts` | Executes `ApiRequest`s through Angular's `HttpClient`; streams `/api/ask` from XHR download progress through the SDK decoder |
| Ask | `pages/ask/` and `services/ask-stream.ts` | AUTO and MANUAL, the router decision card, the streamed answer, clickable citations, the source viewer |
| Compare | `pages/compare/` and `compare/compare-state.ts` | One question across the four strategies, one column each, state in a plain class with its own specs |
| Trace | `pages/trace/` | One run: span waterfall, latency split, calls, tokens, cost, routing, sources, agent steps, graph path when known |
| Evaluation | `pages/evaluation/` | Latest batch tables, per category breakdown, the four dashboards (latency percentiles, cost per day, fallback rate, quality trend) |
| Run details cache | `services/run-details.service.ts` | Keeps, for this browser session only, what a streamed run reported that the server does not store (sub questions, subgraph, router decision), keyed by run id, so Trace can show it |

### Request flow

```
page  ->  feature service  ->  requests.ask(...)  ->  ApiRequest<T>
                                                    |
                         Angular: SdkHttp (HttpClient, interceptor, XHR progress)
                         Node or other apps: RagFabricClient (fetch)
                                                    |
                                                 server
```

Every path, body and response type comes from the SDK. The app owns only how a request travels.

### Ask

- A mode switch: AUTO (sends `strategy: "auto"`) or MANUAL (a strategy picker: traditional,
  vectorless, agentic, graph). Collection filter and `top_k` as optional settings.
- The answer streams token by token from `POST /api/ask`. A `superseded` event replaces the drawn
  text and shows why, with the dropped claims when the server sent them.
- The router decision card renders from the `retrieval` event: the strategy that ran, whether the
  router picked it by rule or by classifier, its confidence (or "rule based, no confidence"), its
  reasoning, the query type, estimated complexity, expected cost and latency levels, and a fallback
  notice when `fallback_from` is set. In MANUAL it says "chosen by you".
- Citations arrive last. Each `[n]` marker in the answer and each citation chip is a button that
  opens the source viewer: file name, page, score, the snippet with the query terms and the quoted
  sentence marked, and a download of the original document.
- After `done`: latency, calls and tokens, and a link to the run's trace.

### Compare

- One question, the same `top_k` and collection, sent once per strategy with the strategy named
  explicitly: traditional, vectorless, agentic, graph.
- Each column streams its own answer and then shows: sources (the cited ones first), latency,
  LLM, retrieval and embedding calls, input and output tokens, estimated cost, and the benchmark
  score, plus a link to its trace.
- The state machine (idle, running, done, failed per column; the run order; the summary row that
  marks the fastest, the cheapest and the highest scoring column) is a plain TypeScript class with
  its own specs, separate from the component.

### Trace

- `/trace/:runId` loads `GET /api/runs/{id}`.
- A waterfall of the stored spans, each bar placed by `started_ms` and sized by `duration_ms`, with
  its attributes. Agent steps are the agent's spans, shown in order with their attributes.
- Latency split (retrieval, generation), calls, tokens, estimated cost, models, mode, requested and
  selected strategy, fallback, router confidence and reasoning.
- Sources by rank, marked cited or not.
- Graph path: the walked nodes and edges when this browser session streamed the run; otherwise a
  plain note that the walked sub graph is not stored with a run.

### Evaluation

- `/evaluation`, admin only, like the API it reads.
- Latest batch: one row per target (strategy) with precision, recall, hit rate, MRR, correctness,
  faithfulness, context relevance, citation correctness, p50 and p95 latency and estimated cost; skipped
  targets listed with their reason; the judge kind and model always shown.
- Per category breakdown for the chosen target.
- Dashboards from `GET /api/eval/dashboard`: latency p50 and p95 per strategy, cost per day (with
  the unpriced count shown separately), fallback rate, quality trend per strategy.
- `n/a` for every value the API returns as null. Nothing is drawn as zero that was not measured.

---

## Decisions and reasons

| # | Decision | Reason |
|---|---|---|
| D1 | The SDK lives at `packages/sdk-typescript`, npm name `@ragfabric/sdk`, ESM only, no runtime dependencies, `fetch` based, Node 20 or later and every current browser | The layout the plan fixed in Phase 1. Zero runtime dependencies keeps an adopter's bundle and audit surface to this package alone; `fetch` and `ReadableStream` are built into every supported runtime |
| D2 | Types are generated from the OpenAPI specification, and both the specification snapshot (`openapi.json`) and the generated file (`schema.ts`) are checked in | A reviewer sees an API change as a diff in the pull request that made it. The SDK builds and tests without Python or a running server. Generating at build time would hide API changes from review and make an npm build depend on the Python environment |
| D3 | The generator is a zero dependency script in the repository, not `openapi-typescript` | `openapi-typescript` 7.13 declares a peer dependency on TypeScript 5 while the repository is on TypeScript 6.0, and the Angular packages and TypeScript must move in lockstep (#57, #63); forcing the peer would be the kind of drift those pull requests had to clean up. It also brings `@redocly/openapi-core` for a job that, for the subset of JSON Schema FastAPI emits, is a few hundred lines. The generator throws on any construct it does not know, so it can never silently emit `any` |
| D4 | Drift is enforced by tests inside CI jobs that already exist: a pytest test proves the live server's specification equals `openapi.json`; an SDK test proves the generator's output equals `schema.ts`; the frontend job runs the SDK tests | The ruleset requires exactly three named checks. Tests inside those jobs block a merge today, while a new job would not be required until the owner edits the ruleset |
| D5 | Request builders plus transports: the SDK describes every call as an `ApiRequest<T>`; `RagFabricClient` executes them with `fetch`; the Angular app executes them through `HttpClient` | One definition of every path, body and response type, shared by every client. The Angular app keeps its interceptor (bearer token, 401 redirect) and all 95 existing specs keep `HttpTestingController` and their synchronous flow. A promise only SDK would have forced every existing spec to be rewritten for asynchronous flushing and moved auth handling into a second place |
| D6 | Streaming uses `POST` with a streamed response body, decoded by the SDK's incremental SSE decoder; never `EventSource` | `EventSource` can only `GET` and cannot send an `Authorization` header, and `/api/ask` is a `POST` with a JSON body. One decoder serves the fetch client (`ReadableStream`) and the Angular transport (XHR download progress), so both parse the wire format identically |
| D7 | `superseded` replaces the drawn answer and shows the reason and the dropped claims | The server's contract (ask.py): the streamed text failed the citation contract and the recorded answer is the repaired one. Showing the streamed text after that would display an answer the server did not record |
| D8 | The `AskEvent` union is typed by hand in the SDK, with every payload built from generated component types | OpenAPI 3.1 has no standard way to describe a server sent events stream. Typing the envelope by hand while reusing the generated `Citation`, `RouterDecisionOut`, `SubgraphOut` and the rest keeps the drift surface to the event names, which a test pins |
| D9 | `POST /api/ask` documents its responses in OpenAPI (JSON `AnswerResponse` and `text/event-stream`) without changing behaviour | Today the specification says nothing about the answer shape, so the generated SDK could not type `ask()`. A documentation only change to the route decorator fixes that |
| D10 | `RunOut` gains `router_confidence` and `router_reasoning` | Both columns are already written on every auto run and the Trace page needs them for runs it did not stream. Two read only fields, no migration |
| D11 | The app uses only the SDK, enforced by `npm run check:sdk-only` in CI: no `HttpClient` import and no `/api/` literal outside `sdk/`. `models.ts` re-exports SDK types | The issue asks for it, and a rule nobody checks does not hold. Re-exporting from `models.ts` moves every type onto the generated ones without touching every component |
| D12 | Compare runs the four fixed strategies, not `auto` | `auto` always serves one of the four, so its column would duplicate another. The Ask page is where the router's choice is explained |
| D13 | Compare runs the strategies one after another by default, in a fixed order (traditional, vectorless, agentic, graph); a switch runs all four at once and labels the latencies as measured under contention | A local Ollama serves one model at a time, so four parallel requests measure the queue rather than the strategy (the same reason as Phase 8's D13). On a hosted provider parallel is faster, so it is offered, labelled |
| D14 | One strategy failing (for example agentic with the offline provider, a 422) fails its column with the server's message and never stops the other three | A comparison that hides the other three answers because one strategy cannot run here would answer nothing |
| D15 | Compare's cost comes from the stored run (`GET /api/runs/{id}` after `done`), shown as an estimate, `n/a` while null | The stream carries calls and tokens but not cost. The run row is where Phase 8 Task 10 writes the estimate, so the column fills itself in once Phase 8 merges, with no change here |
| D16 | Compare's "eval score" is each strategy's mean correctness from the latest evaluation batch, labelled "benchmark score (latest batch)", shown only to admins | An ad hoc question has no expected answer, so a per question score cannot be measured; inventing one would break ADR 0004. The benchmark score is a measured number that says how the strategy did on the shared question set. The evaluation API is admin only (Phase 8), so a non admin sees the column explain that |
| D17 | Trace shows what is stored for every run, and the walked sub graph, sub questions and router decision only for runs this browser session streamed, from an in memory cache; otherwise it says they are not stored | Persisting the sub graph and sub questions needs a migration and a storage decision that belongs to the run model, not to a UI phase. Saying plainly what is not stored is better than an empty panel that looks like an empty walk |
| D18 | The v1 Search page is replaced by Ask; `/search` redirects to `/ask` | The plan said "the Search page becomes Ask" (Phase 1). Two pages that both ask questions, one through the old route, would split the product |
| D19 | No new runtime dependency in the app: charts are plain SVG and CSS bars | The initial bundle budget is 500 kB; the four dashboards are bar and line charts with a handful of points. A chart library would cost more than the pages it draws |
| D20 | The Evaluation page and the dashboards are built against Phase 8's planned API with hand typed shapes (`src/evaluation.ts`) and fakes in specs; a 404 from the evaluation API shows "this server has no evaluation API yet" | Phase 8 is not merged. Building the pages now and re-verifying the shapes when it merges keeps the two phases from serialising |
| D21 | SDK tests use Node's built in test runner over the compiled output with a fake `fetch`; no test touches the network | Deterministic and offline, as the repository requires. `node:test` adds no dependency |
| D22 | The app compiles the SDK from its source through a TypeScript path mapping (`@ragfabric/sdk` to `packages/sdk-typescript/src`), and the UI image copies that folder | No build step or `file:` link has to run before `npm ci` and `ng build`, and the app always uses the SDK at the same commit. The npm package is built separately with `tsc` |
| D23 | The SDK version follows the server it was generated from (0.4.0 on this branch) | The SDK is only correct for the specification it was generated from; a matching version says which server it speaks to |
| D24 | npm packaging is prepared (metadata, `exports`, `files`, build, `npm pack --dry-run`) and not published | Publishing needs the owner's npm login and approval |

### Rejected

| Option | Why not |
|---|---|
| `openapi-typescript` or `openapi-generator` | See D3. `openapi-generator` also needs Java and emits a client this repository would then have to maintain |
| Generating the SDK at build time | See D2 |
| A promise only SDK used directly by Angular services | See D5 |
| `EventSource` for streaming | See D6 |
| A chart library (Chart.js, ECharts, ngx-charts) | See D19 |
| Running Compare's four strategies in one server call | There is no such endpoint, and four ordinary `/api/ask` calls are what a user would make: each is recorded as its own run with its own trace |
| A per question "eval score" from a judge call on Compare | No expected answer exists for an ad hoc question; see D16 |

---

## Dependency on Phase 8

| Piece | Built against | Must be re-verified when Phase 8 merges |
|---|---|---|
| Evaluation page | `GET /api/eval/runs`, `GET /api/eval/runs/{id}` as described in Phase 8's design (run rows from `evaluation_runs` with `summary`, result rows from `evaluation_results`) | Field names and nesting of `summary` (means, per category, percentiles, skipped, judge, prompt version) |
| Dashboards | `GET /api/eval/dashboard?days=` (latency percentiles per strategy, cost per day with unknown cost counted separately, calls per strategy, fallback rate, quality trend) | The response shape: Phase 8's design names the contents but not the JSON |
| Compare cost | `RunOut.estimated_cost_usd`, null on main | That Phase 8 Task 10 fills it for `/api/ask` runs |
| Compare benchmark score | `GET /api/eval/runs` summary `correctness` mean per strategy | The summary key name |
| SDK | `src/evaluation.ts` hand typed | Replace with aliases of generated types once Phase 8's routes are in `openapi.json`; the snapshot test will fail on whichever branch merges second until `openapi.json` and `schema.ts` are regenerated |

---

## Out of scope

- Publishing `@ragfabric/sdk` to npm (owner action) and any PyPI publish.
- Persisting sub graphs, sub questions or router decisions beyond the two router fields in D10
  (needs a migration).
- Starting an evaluation run from the UI (Phase 8 D16: the API is read only).
- Changes to the Python SDK.
- A browser end to end suite in CI and rate limiting (Phase 10).
- Restyling the console v1 pages beyond what moving them onto the SDK requires.
