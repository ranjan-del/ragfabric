// The hand typed event names must be the names the server sends (decision D8).
//
// OpenAPI cannot describe the /api/ask stream, so this test reads the route's
// source and compares. A new event on the server fails here until the SDK
// learns it.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { ASK_EVENT_NAMES } from '../src/types.js';

test('ASK_EVENT_NAMES matches every _event(...) the /api/ask route emits', () => {
  const source = readFileSync(
    new URL('../../../server/src/ragfabric_server/api/routes/ask.py', import.meta.url),
    'utf8',
  );
  const sent = new Set([...source.matchAll(/_event\(\s*"([a-z_]+)"/g)].map((m) => m[1]));
  assert.deepEqual([...sent].sort(), [...ASK_EVENT_NAMES].sort());
});
