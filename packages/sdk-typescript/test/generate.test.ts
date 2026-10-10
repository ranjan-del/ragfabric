// The OpenAPI to TypeScript generator (Phase 9, Task 2).
//
// The generator is small on purpose (decision D3): it covers the JSON Schema
// that FastAPI emits and throws on anything else, so it can never quietly
// type a field as `any`. The last test is the drift check: the checked in
// schema.ts must be exactly what the generator makes of the checked in
// openapi.json.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { generate, typeOf } from '../scripts/generate.js';

const root = new URL('../../', import.meta.url);

function spec(schemas: Record<string, unknown>, paths: Record<string, unknown> = {}): unknown {
  return { openapi: '3.1.0', info: { title: 't', version: '1' }, paths, components: { schemas } };
}

test('an object with required and optional fields', () => {
  const out = generate(
    spec({
      User: {
        type: 'object',
        properties: { id: { type: 'integer' }, email: { type: 'string' }, note: { type: 'string' } },
        required: ['id', 'email'],
      },
    }),
  );
  assert.match(out, /export interface User \{\n {2}id: number;\n {2}email: string;\n {2}note\?: string;\n\}/);
});

test('anyOf with null becomes a union with null', () => {
  assert.equal(typeOf({ anyOf: [{ type: 'integer' }, { type: 'null' }] }, '#'), 'number | null');
});

test('a $ref names the component', () => {
  assert.equal(typeOf({ $ref: '#/components/schemas/Citation' }, '#'), 'Citation');
});

test('arrays, including arrays of unions', () => {
  assert.equal(typeOf({ type: 'array', items: { type: 'string' } }, '#'), 'string[]');
  assert.equal(
    typeOf({ type: 'array', items: { anyOf: [{ type: 'string' }, { type: 'null' }] } }, '#'),
    '(string | null)[]',
  );
});

test('enum and const become literal unions', () => {
  assert.equal(typeOf({ type: 'string', enum: ['read', 'write'] }, '#'), "'read' | 'write'");
  assert.equal(typeOf({ const: 'llm' }, '#'), "'llm'");
});

test('additionalProperties becomes a record', () => {
  assert.equal(
    typeOf({ type: 'object', additionalProperties: { type: 'integer' } }, '#'),
    'Record<string, number>',
  );
  assert.equal(typeOf({ type: 'object' }, '#'), 'Record<string, unknown>');
  assert.equal(typeOf({ type: 'object', additionalProperties: true }, '#'), 'Record<string, unknown>');
});

test('a schema with no type is unknown, and a list of types is a union', () => {
  assert.equal(typeOf({}, '#'), 'unknown');
  assert.equal(typeOf({ type: ['string', 'null'] }, '#'), 'string | null');
});

test('a binary upload field is a Blob', () => {
  assert.equal(typeOf({ type: 'string', contentMediaType: 'application/octet-stream' }, '#'), 'Blob');
  assert.equal(typeOf({ type: 'string', format: 'binary' }, '#'), 'Blob');
});

test('an unknown construct throws and names where it is', () => {
  assert.throws(
    () => generate(spec({ Bad: { type: 'object', properties: { x: { allOf: [] } } } })),
    /#\/components\/schemas\/Bad\/properties\/x: unsupported keyword "allOf"/,
  );
  assert.throws(() => typeOf({ type: 'tuple' }, '#/x'), /#\/x: unsupported type "tuple"/);
});

test('property names that are not identifiers are quoted', () => {
  const out = generate(
    spec({ Odd: { type: 'object', properties: { 'two words': { type: 'string' } } } }),
  );
  assert.match(out, /'two words'\?: string;/);
});

test('descriptions become doc comments and cannot close the comment early', () => {
  const out = generate(
    spec({ Doc: { type: 'object', description: 'Ends */ here', properties: {} } }),
  );
  assert.match(out, /\/\*\* Ends \*\\\/ here \*\//);
});

test('operations are typed by path and method: parameters, body and the success response', () => {
  const out = generate(
    spec(
      { RunOut: { type: 'object', properties: { id: { type: 'integer' } }, required: ['id'] } },
      {
        '/api/runs/{run_id}': {
          get: {
            operationId: 'get_run',
            parameters: [
              { in: 'path', name: 'run_id', required: true, schema: { type: 'integer' } },
              { in: 'query', name: 'verbose', required: false, schema: { type: 'boolean' } },
            ],
            responses: {
              '200': { content: { 'application/json': { schema: { $ref: '#/components/schemas/RunOut' } } } },
              '422': { content: { 'application/json': { schema: {} } } },
            },
          },
        },
      },
    ),
  );
  assert.match(out, /'\/api\/runs\/\{run_id\}': \{\n {4}get: \{/);
  assert.match(out, /path: \{ run_id: number \};/);
  assert.match(out, /query: \{ verbose\?: boolean \};/);
  assert.match(out, /body: never;/);
  assert.match(out, /response: RunOut;/);
});

test('the output is stable: schemas and paths sorted, same input same bytes', () => {
  const s = spec({ B: { type: 'string' }, A: { type: 'integer' } });
  const out = generate(s);
  assert.equal(out, generate(s));
  assert.ok(out.indexOf('export type A') < out.indexOf('export type B'));
});

test('the checked in schema.ts is exactly the generator output for openapi.json', () => {
  const openapi = JSON.parse(readFileSync(new URL('openapi.json', root), 'utf8'));
  const checkedIn = readFileSync(new URL('src/generated/schema.ts', root), 'utf8');
  assert.equal(
    checkedIn,
    generate(openapi),
    'src/generated/schema.ts is out of date or was edited by hand. Run: npm run generate',
  );
});
