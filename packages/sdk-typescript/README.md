# @ragfabric/sdk

TypeScript client for a [RagFabric](https://github.com/ranjan-del/ragfabric) server. Ask a question
and stream the cited answer, compare the four retrieval strategies, read a run's trace, and read the
evaluation results. The request and response types are generated from the server's OpenAPI
specification.

- No runtime dependencies. Uses the platform's `fetch`, `ReadableStream` and `TextDecoder`.
- ESM only. Node 20 or later, or any current browser.
- Version follows the server it was generated from: `0.4.0` speaks to a RagFabric 0.4.0 server.

> Not yet published to npm. Until it is, build it from this folder (`npm ci && npm run build`).

## Quick start

```ts
import { RagFabricClient } from '@ragfabric/sdk';

const client = new RagFabricClient({ baseUrl: 'http://localhost:8000' });
await client.login('admin@example.com', 'your-password');
// or: new RagFabricClient({ baseUrl, apiKey: 'rf_...' })

// The finished answer.
const answer = await client.ask('How many days of annual leave do I get?', { strategy: 'auto' });
console.log(answer.answer, answer.strategy, answer.router?.reasoning);

// The answer as it streams.
let text = '';
for await (const event of client.askStream('Who does the Platform Team report to?', { strategy: 'graph' })) {
  switch (event.event) {
    case 'retrieval':
      console.log('strategy', event.data.strategy, 'walked', event.data.subgraph?.edges.length ?? 0, 'edges');
      break;
    case 'token':
      text += event.data.text;
      break;
    case 'superseded':
      // The streamed text failed the citation contract and was repaired.
      // Replace what you drew: this is the answer the server recorded.
      text = event.data.text;
      break;
    case 'citations':
      console.log(event.data.citations.filter((c) => c.used).map((c) => c.filename));
      break;
    case 'done':
      console.log(await client.run(event.data.run_id)); // the stored run and its trace
      break;
  }
}
```

Errors are `RagFabricError` with the HTTP `status` (0 when the server was not reached) and the
server's own `detail`.

## Strategies

`'traditional'`, `'vectorless'`, `'agentic'`, `'graph'`, or `'auto'` to let the server's router
choose. Leave `strategy` unset to use the server's configured `router.mode`.

## Requests as data

Every endpoint has a builder in `requests` that describes the call without sending it:

```ts
import { requests } from '@ragfabric/sdk';

const req = requests.documents.list({ collection_id: 3 });
// { method: 'GET', path: '/api/documents?collection_id=3', responseType: 'json' }
const docs = await client.send(req);
```

This is how the bundled Angular app uses the SDK: it executes the same requests through its own
HTTP stack. `SseDecoder` and `toAskEvent` are exported for transports that stream another way.

## Evaluation (provisional)

`client.evaluationRuns()`, `evaluationRun(id)` and `evaluationDashboard(days)` read the evaluation
API added in RagFabric 0.5.0. Their types are typed by hand until that API is in the specification
this package is generated from, and may change.

## Development

```bash
npm ci
npm test            # compile, then node --test; offline
npm run generate    # regenerate src/generated/schema.ts from openapi.json
npm run build       # dist/ for publishing
```

`openapi.json` is exported from the server with
`uv run python -m ragfabric_server.openapi packages/sdk-typescript/openapi.json` (run from the
repository root). A server test fails when the server and `openapi.json` disagree, and an SDK test
fails when `schema.ts` was not regenerated, so the two cannot drift apart unnoticed.

Licensed under Apache 2.0.
