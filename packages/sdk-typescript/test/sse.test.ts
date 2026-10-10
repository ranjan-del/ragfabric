// The incremental SSE decoder (Phase 9, Task 3; review focus 2).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { SseDecoder, toAskEvent } from '../src/sse.js';

function all(chunks: (string | Uint8Array)[]) {
  const decoder = new SseDecoder();
  const out = chunks.flatMap((chunk) => decoder.push(chunk));
  return [...out, ...decoder.end()];
}

test('one event per blank line, with its name and data', () => {
  assert.deepEqual(all(['event: token\ndata: {"text":"Hi"}\n\n']), [
    { event: 'token', data: '{"text":"Hi"}' },
  ]);
});

test('CRLF and CR line endings, and a CRLF split across chunks', () => {
  assert.deepEqual(all(['event: a\r\ndata: 1\r', '\n\r\nevent: b\rdata: 2\r\r']), [
    { event: 'a', data: '1' },
    { event: 'b', data: '2' },
  ]);
});

test('comments are ignored and multi line data is joined with a newline', () => {
  assert.deepEqual(all([': keep alive\nevent: x\ndata: one\ndata: two\n\n']), [
    { event: 'x', data: 'one\ntwo' },
  ]);
});

test('a chunk boundary inside an event yields the event once, intact', () => {
  const decoder = new SseDecoder();
  assert.deepEqual(decoder.push('event: tok'), []);
  assert.deepEqual(decoder.push('en\ndata: {"te'), []);
  assert.deepEqual(decoder.push('xt":"a"}\n'), []);
  assert.deepEqual(decoder.push('\n'), [{ event: 'token', data: '{"text":"a"}' }]);
});

test('a chunk boundary inside a multi byte character', () => {
  const bytes = new TextEncoder().encode('event: token\ndata: {"text":"café ₹"}\n\n');
  const cut = bytes.indexOf(0xe2) + 1; // inside the three byte rupee sign
  const events = all([bytes.slice(0, cut), bytes.slice(cut)]);
  assert.equal(JSON.parse(events[0]!.data).text, 'café ₹');
});

test('a data block with no event line is a "message", per the specification', () => {
  assert.deepEqual(all(['data: 1\n\n']), [{ event: 'message', data: '1' }]);
});

test('a final event without its trailing blank line still arrives at end()', () => {
  assert.deepEqual(all(['event: done\ndata: {}']), [{ event: 'done', data: '{}' }]);
});

test('toAskEvent types the known names and passes unknown ones through', () => {
  assert.deepEqual(toAskEvent({ event: 'token', data: '{"text":"x"}' }), {
    event: 'token',
    data: { text: 'x' },
  });
  assert.deepEqual(toAskEvent({ event: 'progress', data: '1' }), {
    event: 'unknown',
    name: 'progress',
    data: 1,
  });
});
