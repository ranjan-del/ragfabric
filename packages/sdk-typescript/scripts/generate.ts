// Generate TypeScript types from the server's OpenAPI specification.
//
//   node build/scripts/generate.js openapi.json src/generated/schema.ts
//
// Why a script in this repository and not openapi-typescript: see decision D3
// in docs/plans/2026-10-10-phase-9-assistant-and-sdk.md. In short, the
// established generators peer on TypeScript 5 while this repository is on
// TypeScript 6, and the subset of JSON Schema that FastAPI emits is small.
// The cost of owning the generator is that it must be strict: every keyword it
// does not understand throws with its JSON pointer, so a new construct in the
// specification stops the build instead of silently becoming `unknown`.

import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

type Schema = Record<string, unknown>;

// Keywords that constrain or describe a value but do not change its type.
const ANNOTATIONS = new Set([
  'title',
  'description',
  'default',
  'examples',
  'example',
  'deprecated',
  'readOnly',
  'writeOnly',
  'minLength',
  'maxLength',
  'pattern',
  'minimum',
  'maximum',
  'exclusiveMinimum',
  'exclusiveMaximum',
  'minItems',
  'maxItems',
  'uniqueItems',
  'format',
  'contentMediaType',
]);

const TYPE_KEYWORDS = new Set([
  'type',
  'anyOf',
  'oneOf',
  '$ref',
  'enum',
  'const',
  'items',
  'properties',
  'required',
  'additionalProperties',
]);

const PRIMITIVES: Record<string, string> = {
  string: 'string',
  integer: 'number',
  number: 'number',
  boolean: 'boolean',
  null: 'null',
};

const BODY_TYPES = ['application/json', 'multipart/form-data', 'application/x-www-form-urlencoded'];
const METHODS = ['delete', 'get', 'head', 'options', 'patch', 'post', 'put'];

function fail(pointer: string, message: string): never {
  throw new Error(`${pointer}: ${message}`);
}

function isObject(value: unknown): value is Schema {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function literal(value: unknown, pointer: string): string {
  if (typeof value === 'string') {
    return `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
  }
  if (typeof value === 'number' || typeof value === 'boolean' || value === null) {
    return String(value);
  }
  return fail(pointer, `unsupported literal ${JSON.stringify(value)}`);
}

function key(name: string): string {
  return /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name) ? name : literal(name, '#');
}

function union(parts: string[]): string {
  const unique = [...new Set(parts)];
  return unique.join(' | ');
}

function wrap(type: string): string {
  return /[|&]/.test(type) ? `(${type})` : type;
}

function comment(text: unknown, indent: string): string {
  if (typeof text !== 'string' || text.trim() === '') {
    return '';
  }
  const safe = text.trim().replace(/\*\//g, '*\\/');
  const lines = safe.split('\n');
  if (lines.length === 1) {
    return `${indent}/** ${lines[0]} */\n`;
  }
  const body = lines.map((line) => `${indent} *${line.trim() === '' ? '' : ` ${line.trimEnd()}`}`);
  return `${indent}/**\n${body.join('\n')}\n${indent} */\n`;
}

function checkKeywords(schema: Schema, pointer: string): void {
  for (const name of Object.keys(schema)) {
    if (!ANNOTATIONS.has(name) && !TYPE_KEYWORDS.has(name)) {
      fail(`${pointer}`, `unsupported keyword "${name}"`);
    }
  }
}

function objectBody(schema: Schema, pointer: string, indent: string): string {
  const properties = schema['properties'];
  if (!isObject(properties)) {
    return fail(pointer, 'properties must be an object');
  }
  if (schema['additionalProperties'] !== undefined && schema['additionalProperties'] !== false) {
    fail(pointer, 'properties together with additionalProperties are not supported');
  }
  const required = new Set(Array.isArray(schema['required']) ? (schema['required'] as string[]) : []);
  const lines: string[] = [];
  for (const [name, child] of Object.entries(properties)) {
    const childPointer = `${pointer}/properties/${name}`;
    if (!isObject(child)) {
      fail(childPointer, 'a property schema must be an object');
    }
    const optional = required.has(name) ? '' : '?';
    lines.push(
      `${comment(child['description'], `${indent}  `)}${indent}  ${key(name)}${optional}: ${typeOf(
        child,
        childPointer,
        `${indent}  `,
      )};`,
    );
  }
  return lines.length === 0 ? '{}' : `{\n${lines.join('\n')}\n${indent}}`;
}

/** The TypeScript type for one schema. Exported for the tests. */
export function typeOf(schema: unknown, pointer: string, indent = ''): string {
  if (!isObject(schema)) {
    return fail(pointer, 'a schema must be an object');
  }
  checkKeywords(schema, pointer);

  const ref = schema['$ref'];
  if (typeof ref === 'string') {
    const prefix = '#/components/schemas/';
    if (!ref.startsWith(prefix)) {
      fail(pointer, `unsupported $ref "${ref}"`);
    }
    return ref.slice(prefix.length);
  }
  if ('const' in schema) {
    return literal(schema['const'], pointer);
  }
  if (Array.isArray(schema['enum'])) {
    return union(schema['enum'].map((value, i) => literal(value, `${pointer}/enum/${i}`)));
  }
  for (const combinator of ['anyOf', 'oneOf']) {
    const members = schema[combinator];
    if (Array.isArray(members)) {
      return union(members.map((m, i) => typeOf(m, `${pointer}/${combinator}/${i}`, indent)));
    }
  }

  const type = schema['type'];
  if (type === undefined) {
    return 'unknown';
  }
  if (Array.isArray(type)) {
    return union(type.map((t) => typeOf({ ...schema, type: t }, pointer, indent)));
  }
  if (typeof type !== 'string') {
    return fail(pointer, 'type must be a string or a list of strings');
  }
  if (type === 'string' && (schema['format'] === 'binary' || schema['contentMediaType'] !== undefined)) {
    return 'Blob';
  }
  if (type in PRIMITIVES) {
    return PRIMITIVES[type] as string;
  }
  if (type === 'array') {
    const items = schema['items'];
    return items === undefined ? 'unknown[]' : `${wrap(typeOf(items, `${pointer}/items`, indent))}[]`;
  }
  if (type === 'object') {
    if (schema['properties'] !== undefined) {
      return objectBody(schema, pointer, indent);
    }
    const extra = schema['additionalProperties'];
    if (isObject(extra)) {
      return `Record<string, ${typeOf(extra, `${pointer}/additionalProperties`, indent)}>`;
    }
    return 'Record<string, unknown>';
  }
  return fail(pointer, `unsupported type "${type}"`);
}

function component(name: string, schema: unknown): string {
  const pointer = `#/components/schemas/${name}`;
  if (!isObject(schema)) {
    return fail(pointer, 'a schema must be an object');
  }
  const doc = comment(schema['description'], '');
  if (schema['type'] === 'object' && schema['properties'] !== undefined) {
    checkKeywords(schema, pointer);
    return `${doc}export interface ${name} ${objectBody(schema, pointer, '')}\n`;
  }
  return `${doc}export type ${name} = ${typeOf(schema, pointer)};\n`;
}

function parameters(operation: Schema, where: string, pointer: string): string {
  const list = Array.isArray(operation['parameters']) ? operation['parameters'] : [];
  const parts: string[] = [];
  list.forEach((param: unknown, i: number) => {
    const p = `${pointer}/parameters/${i}`;
    if (!isObject(param)) {
      fail(p, 'a parameter must be an object');
    }
    if (param['in'] !== where) {
      return;
    }
    const optional = param['required'] === true ? '' : '?';
    parts.push(`${key(String(param['name']))}${optional}: ${typeOf(param['schema'], `${p}/schema`)}`);
  });
  return parts.length === 0 ? 'Record<string, never>' : `{ ${parts.join('; ')} }`;
}

function body(operation: Schema, pointer: string): string {
  const request = operation['requestBody'];
  if (request === undefined) {
    return 'never';
  }
  if (!isObject(request) || !isObject(request['content'])) {
    return fail(`${pointer}/requestBody`, 'requestBody must have content');
  }
  const content = request['content'];
  for (const media of BODY_TYPES) {
    const entry = content[media];
    if (isObject(entry)) {
      return typeOf(entry['schema'], `${pointer}/requestBody/content/${media}/schema`);
    }
  }
  return fail(`${pointer}/requestBody`, `unsupported media types ${Object.keys(content).join(', ')}`);
}

function response(operation: Schema, pointer: string): string {
  const responses = operation['responses'];
  if (!isObject(responses)) {
    return fail(pointer, 'responses must be an object');
  }
  const success = Object.keys(responses)
    .filter((code) => /^2\d\d$/.test(code))
    .sort();
  for (const code of success) {
    const entry = responses[code];
    const content = isObject(entry) ? entry['content'] : undefined;
    if (isObject(content) && isObject(content['application/json'])) {
      const json = content['application/json'];
      return typeOf(json['schema'] ?? {}, `${pointer}/responses/${code}/content/application~1json/schema`);
    }
  }
  return success.length > 0 ? 'unknown' : 'never';
}

function pathsBlock(paths: Schema): string {
  const lines: string[] = ['export interface paths {'];
  for (const path of Object.keys(paths).sort()) {
    const item = paths[path];
    const pointer = `#/paths/${path.replace(/~/g, '~0').replace(/\//g, '~1')}`;
    if (!isObject(item)) {
      fail(pointer, 'a path item must be an object');
    }
    lines.push(`  ${literal(path, pointer)}: {`);
    for (const method of METHODS) {
      const operation = item[method];
      if (!isObject(operation)) {
        continue;
      }
      const p = `${pointer}/${method}`;
      lines.push(`${comment(operation['summary'], '    ')}    ${method}: {`);
      lines.push(`      path: ${parameters(operation, 'path', p)};`);
      lines.push(`      query: ${parameters(operation, 'query', p)};`);
      lines.push(`      body: ${body(operation, p)};`);
      lines.push(`      response: ${response(operation, p)};`);
      lines.push('    };');
    }
    lines.push('  };');
  }
  lines.push('}');
  return `${lines.join('\n')}\n`;
}

/** Render the whole schema.ts for one OpenAPI document. */
export function generate(spec: unknown): string {
  if (!isObject(spec)) {
    return fail('#', 'the specification must be an object');
  }
  const info = isObject(spec['info']) ? spec['info'] : {};
  const components = isObject(spec['components']) ? spec['components'] : {};
  const schemas = isObject(components['schemas']) ? components['schemas'] : {};
  const paths = isObject(spec['paths']) ? spec['paths'] : {};

  const header = [
    `// Generated by scripts/generate.ts from openapi.json (${String(info['title'])} ${String(
      info['version'],
    )}).`,
    '// Do not edit by hand: change the server, then run `npm run generate` in packages/sdk-typescript.',
    '/* eslint-disable */',
    '',
  ].join('\n');

  const blocks = Object.keys(schemas)
    .sort()
    .map((name) => component(name, schemas[name]));
  return `${header}\n${blocks.join('\n')}\n${pathsBlock(paths)}`;
}

function main(argv: string[]): void {
  const [input, output] = argv;
  if (input === undefined || output === undefined) {
    throw new Error('usage: generate <openapi.json> <schema.ts>');
  }
  const spec: unknown = JSON.parse(readFileSync(input, 'utf8'));
  writeFileSync(output, generate(spec), 'utf8');
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2));
}
