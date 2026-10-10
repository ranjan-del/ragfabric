# What the first real run of the assistant looked like, learning notes

> Status: Phase 9 (issue #10), on the phase branch. A record of **one** session on one laptop on
> 2026-10-10, driving Ask, Compare and Trace across all four strategies against a real server, once
> through `@ragfabric/sdk` from Node and once through the built UI in a headless browser. It is not a
> benchmark and nothing in it is a quality or latency claim. Nothing was tuned in response to it.

## Setup

| Piece | What ran |
|---|---|
| Server | `ragfabric serve` from the phase branch, PostgreSQL 18 with pgvector in a throwaway container, `cache: memory`, `indexing: inline` |
| Models | Ollama: `llama3.1:8b` for answers, routing, the agent and graph extraction; `nomic-embed-text` for embeddings |
| Graph | `graph_store.enabled: true`, so every chunk went through one extraction call at ingest |
| Data | The three packaged quickstart samples (`leave-policy.md`, `handbook.md`, `team.md`), uploaded through the SDK's `upload` |
| UI | `ng serve --configuration production` with `/api` proxied to the server, driven by Playwright |
| Contention | Another evaluation job was using the same Ollama at the same time for part of the session. Latencies below include that, which is one more reason they mean nothing as numbers |

Two questions were asked of every strategy and of `auto` through the SDK; then, through the UI, one
question on Ask in AUTO, one question on Compare across the four strategies, and the traces.

## What worked end to end

- **Streaming through both transports.** The SDK's `askStream` (fetch and `ReadableStream`) and the
  app's `SdkHttp` (XHR download progress) received the same event order on every run: `retrieval`,
  `token`..., `citations`, `done`. No `superseded` event occurred in this session, so the replace in
  place path was exercised only by the specs.
- **The router card said what happened, including the fallback.** "Who does Asha Rao report to?" in
  AUTO: the signals chose graph by rule ("The question asks how named things are related"), graph
  walked nothing, traditional answered, and the card showed all three facts with "Rule based, no
  confidence measured" rather than a number. The same question through the SDK recorded the same
  route, and `GET /api/runs/{id}` now returns `router_reasoning` so the cold Trace page showed it too.
- **Citations opened the right source.** Clicking `[1]` opened `team.md` with score 0.615 and the
  quoted sentence underlined.
- **Compare completed all four columns one after another**, each with its own run and trace link, and
  marked traditional as the fastest. With no priced model, every cost showed `n/a` and no column was
  marked cheapest, which is the intended behaviour, not a gap.
- **Trace drew every stored span.** The AUTO run's trace showed nine spans in order: `router`, the
  graph attempt (`check_coverage`, `extract_question`, `match_entities`, `traverse`), then
  traditional's `embed_query`, `vector_search`, `context_budget` and `answer_stream`. That is the
  fallback, visible as time.
- **Session only details behaved as designed.** Opened from the Compare column in the same session,
  the graph trace showed "Nothing walked: no_entity_matched". Opened cold, the same kind of page said
  the sub graph is not stored with a run.
- **Evaluation said why it was empty.** Against a server without the v0.5.0 evaluation API both
  panels said "This server has no evaluation API yet"; Compare's benchmark column said the same.

## What went wrong, and what was learnt

| Finding | Detail | What it means |
|---|---|---|
| Graph found nothing for the relationship question | `team.md` says "Asha Rao reports to Ravi Sharma" in plain words, yet extraction with `llama3.1:8b` produced 15 entities across the three files, none of them Asha Rao, Ravi Sharma or the Platform Team, and 11 relationships. The graph strategy answered "I could not find an answer" every time, with `empty_reason: no_entity_matched` | The same extraction weakness `docs/learning/graph-extraction-first-run.md` recorded. The UI made it legible (fallback notice, empty reason, the waterfall) but cannot fix it; Phase 8's measurements are where extraction quality gets a number |
| The agent answered the same question once and not the other time | Through the SDK, "How many days of annual leave do employees get, and how many carry forward?" with `agentic` ended after `plan`, `tool_check`, `finalize` with both sub questions `open`, no retrieval call and the no evidence answer. Through Compare, in a later session, the same question answered correctly in 37.6 s | One run each, so no rate. The first looks like a budget stop before any retrieval (the default `max_latency_ms` is 30000 and planning alone on a contended 8B model is slow); recorded, not investigated further here |
| Cost is `n/a` everywhere | `estimated_cost_usd` and `llm_model` are null on every run on main | Expected: Phase 8 Task 10 starts writing them. The column needs no change when it does; to be re-verified |
| Streamed answers report 0 output tokens | Traditional and vectorless streamed runs show `Tokens in / out 0 / 0` (vectorless) or `18 / 0` (traditional) | The server's stream path counts no generation tokens because the stream protocol returns no usage (`ask.py` says so). The UI shows what the server reports; a reader could take 0 for a measurement. Worth a server follow up: report generation tokens as null on the stream path |
| Opening an admin page directly sent the admin away | The first scripted visit to `/evaluation` landed on the dashboard: `adminGuard` ran before the profile had loaded and saw no admin | A real bug in the v1 guard that every console page had too. Fixed on this branch: with a token and no profile the guard now waits for the profile, with specs |
| The first scripted Ask timed out | Under contention the first AUTO answer did not finish within the script's ten minute wait; a rerun with a longer wait finished in 15.8 s | A local model shared with another job is the whole explanation; recorded so nobody mistakes it for a UI hang. The SSE proxy path was checked separately with `curl -N` and streamed immediately |
| Ingest was slow | 127 to 162 s per two chunk file | One extraction call per chunk with an 8B model, as configured. Not a UI concern |
| Uncited sources crowded each column | Six chips per column, five of them "retrieved, not cited" | Changed: uncited sources are now folded under "N more retrieved, not cited", still one click away |

## Not exercised

- The `superseded` event against a real model (it did not occur; covered by specs).
- Compare in parallel mode against a real model.
- The Evaluation page with data: the evaluation API does not exist on main yet.
- An API key client, or a non admin user's view of Compare.
