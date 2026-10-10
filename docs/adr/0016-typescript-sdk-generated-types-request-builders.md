# ADR 0016: The TypeScript SDK: generated types, requests as data, pluggable transports

Status: accepted
Date: 2026-10-10
Supersedes: none
Related: ADR 0001 (monorepo with layered packages), ADR 0004 (measurement first, no fabricated
numbers), the Phase 9 design `docs/plans/2026-10-10-phase-9-assistant-and-sdk.md`

## Context

The plan fixed `@ragfabric/sdk` as the TypeScript client and said the apps would use only the SDK.
Three questions had to be answered to build it:

1. How do the SDK's types stay true to the server?
2. How can one SDK serve both a plain `fetch` caller and the Angular app, whose HTTP stack carries
   the auth interceptor and whose 95 existing specs drive it synchronously through
   `HttpTestingController`?
3. How is `POST /api/ask`, a server sent events stream, typed and read?

## Decision

| Question | Decision |
|---|---|
| Types | Generated from the server's OpenAPI specification by a zero dependency script in the package. The specification snapshot (`openapi.json`) and the output (`src/generated/schema.ts`) are both checked in. A server test fails when the live specification and the snapshot differ; an SDK test fails when `schema.ts` is not what the generator makes of the snapshot |
| One SDK, two HTTP stacks | Every endpoint has a builder that returns an `ApiRequest<T>` (method, path with query, body, headers, response kind) and sends nothing. `RagFabricClient` executes requests with `fetch`; the Angular app executes the same requests through `HttpClient` (`SdkHttp`). Only the SDK spells API paths, enforced in CI by `npm run check:sdk-only` |
| The stream | `POST` with a streamed body, never `EventSource` (which cannot `POST` or send a header). One incremental decoder (`SseDecoder`) serves both transports. The event envelope is typed by hand and every payload reuses generated types; a test pins the event names to the ones the route emits |

| Option | Verdict |
|---|---|
| `openapi-typescript` | Rejected. 7.13 peers on TypeScript 5 while the repository is on TypeScript 6, with Angular and TypeScript moving in lockstep; it also brings `@redocly/openapi-core` |
| `openapi-generator` | Rejected. Needs Java and emits a whole client to maintain |
| Generating at build time, nothing checked in | Rejected. Hides API changes from review and makes an npm build depend on Python |
| A promise only SDK called directly by Angular services | Rejected. Every existing spec would need rewriting for asynchronous flushing, and auth handling would live in two places |
| **Generated types, requests as data, pluggable transports** | **Chosen** |

## Consequences

- An API change is a three file diff a reviewer can read: the route, `openapi.json`, `schema.ts`.
  Forgetting to regenerate fails CI in jobs that already block a merge.
- The generator is ours to maintain. It is strict on purpose: an unknown JSON Schema keyword throws
  with its pointer, so a new construct stops the build instead of becoming `unknown`.
- Hand typed shapes remain where the specification is silent: the event stream, the two analytics
  routes with no response model, and the evaluation API until its routes are in the specification.
- The Angular app compiles the SDK from source through a path mapping, so its Karma build uses the
  esbuild based `builderMode: application` (webpack does not resolve the SDK's `.js` import
  specifiers to `.ts` files). The UI image copies the SDK source.
- The SDK's version follows the server it was generated from.
