// The fetch client against a fake fetch: nothing touches the network.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { RagFabricClient } from '../src/client.js';
import { RagFabricError } from '../src/errors.js';
import type { AskEvent } from '../src/types.js';

interface Call {
  url: string;
  init: RequestInit;
}

function fake(respond: (call: Call) => Response): { fetch: typeof fetch; calls: Call[] } {
  const calls: Call[] = [];
  const impl = (async (url: string | URL | Request, init: RequestInit = {}) => {
    const call = { url: String(url), init };
    calls.push(call);
    return respond(call);
  }) as typeof fetch;
  return { fetch: impl, calls };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function streamOf(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
}

test('send joins the base URL, sends JSON and the bearer token', async () => {
  const f = fake(() => json({ id: 3 }));
  const client = new RagFabricClient({ baseUrl: 'http://h:8000/', token: 't', fetch: f.fetch });

  const run = await client.run(3);

  assert.deepEqual(run, { id: 3 });
  assert.equal(f.calls[0]!.url, 'http://h:8000/api/runs/3');
  assert.equal((f.calls[0]!.init.headers as Record<string, string>)['Authorization'], 'Bearer t');
});

test('an API key wins over a token, as in the Python SDK', async () => {
  const f = fake(() => json({}));
  await new RagFabricClient({ token: 't', apiKey: 'rf_k', fetch: f.fetch }).ask('q');
  const headers = f.calls[0]!.init.headers as Record<string, string>;
  assert.equal(headers['X-API-Key'], 'rf_k');
  assert.equal(headers['Authorization'], undefined);
  assert.equal(headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(String(f.calls[0]!.init.body)), { query: 'q', stream: false });
});

test('an error status raises RagFabricError with the server detail', async () => {
  const f = fake(() => json({ detail: 'Agentic needs a model.' }, 422));
  const client = new RagFabricClient({ fetch: f.fetch });
  await assert.rejects(client.ask('q', { strategy: 'agentic' }), (error: unknown) => {
    assert.ok(error instanceof RagFabricError);
    assert.equal(error.status, 422);
    assert.equal(error.detail, 'Agentic needs a model.');
    return true;
  });
});

test('validation errors join their messages; an unreachable server is status 0', async () => {
  const f = fake(() => json({ detail: [{ msg: 'too short' }, { msg: 'bad format' }] }, 422));
  await assert.rejects(new RagFabricClient({ fetch: f.fetch }).run(1), /too short\. bad format/);

  const down = (async () => {
    throw new TypeError('fetch failed');
  }) as typeof fetch;
  await assert.rejects(new RagFabricClient({ fetch: down }).run(1), (error: unknown) => {
    assert.ok(error instanceof RagFabricError);
    assert.equal(error.status, 0);
    assert.equal(error.detail, 'Could not reach the server.');
    return true;
  });
});

test('askStream yields typed events in order across split chunks', async () => {
  const f = fake(() =>
    streamOf([
      'event: retrieval\ndata: {"chunks":2,"strategy":"traditional","trace":[],',
      '"sub_questions":[],"subgraph":null,"router":null,"fallback_from":null}\n\n',
      'event: token\ndata: {"text":"Leave is "}\n\nevent: token\ndata: {"text":"24 days [1]."}\n\n',
      'event: superseded\ndata: {"text":"Leave is 24 days [1].","reason":"citation contract"}\n\n',
      'event: citations\ndata: {"citations":[]}\n\n',
      'event: done\ndata: {"run_id":9,"latency_ms":120,"usage":{"llm_calls":1}}\n\n',
    ]),
  );
  const events: AskEvent[] = [];
  for await (const event of new RagFabricClient({ fetch: f.fetch }).askStream('q', { strategy: 'auto' })) {
    events.push(event);
  }

  assert.deepEqual(
    events.map((e) => e.event),
    ['retrieval', 'token', 'token', 'superseded', 'citations', 'done'],
  );
  const done = events[5]!;
  assert.ok(done.event === 'done' && done.data.run_id === 9);
  assert.deepEqual(JSON.parse(String(f.calls[0]!.init.body)), {
    strategy: 'auto',
    query: 'q',
    stream: true,
  });
});

test('a stream refused before it starts raises, it does not yield', async () => {
  const f = fake(() => json({ detail: 'Not authenticated' }, 401));
  const client = new RagFabricClient({ fetch: f.fetch });
  await assert.rejects(async () => {
    for await (const _ of client.askStream('q')) {
      assert.fail('no event expected');
    }
  }, /Not authenticated/);
});

test('login keeps the token for later calls', async () => {
  const f = fake((call) =>
    call.url.endsWith('/login') ? json({ access_token: 'jwt', token_type: 'bearer' }) : json({ id: 1 }),
  );
  const client = new RagFabricClient({ fetch: f.fetch });
  await client.login('a@example.com', 'pw');
  assert.equal(f.calls[0]!.init.body, 'username=a%40example.com&password=pw');
  assert.equal((f.calls[1]!.init.headers as Record<string, string>)['Authorization'], 'Bearer jwt');
});

test('a download is returned as a Blob', async () => {
  const f = fake(() => new Response('file bytes'));
  const client = new RagFabricClient({ fetch: f.fetch });
  const { requests } = await import('../src/requests.js');
  const blob = await client.send(requests.documents.download(4));
  assert.equal(await blob.text(), 'file bytes');
});
