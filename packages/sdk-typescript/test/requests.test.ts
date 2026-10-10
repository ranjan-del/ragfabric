// Request builders describe calls without sending them (decision D5).

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { requests } from '../src/requests.js';

test('ask sends stream false and omits an unset strategy', () => {
  const request = requests.ask('how much leave', { top_k: 5 });
  assert.equal(request.method, 'POST');
  assert.equal(request.path, '/api/ask');
  assert.deepEqual(request.body, { top_k: 5, query: 'how much leave', stream: false });
  assert.ok(!('strategy' in (request.body as object)));
});

test('askStream sends stream true and the named strategy', () => {
  const request = requests.askStream('q', { strategy: 'auto', collection_id: undefined });
  assert.deepEqual(request.body, { strategy: 'auto', query: 'q', stream: true });
});

test('query strings skip unset values and encode the rest', () => {
  assert.equal(requests.documents.list().path, '/api/documents');
  assert.equal(requests.documents.list({ collection_id: 3 }).path, '/api/documents?collection_id=3');
  assert.equal(requests.evaluation.runs().path, '/api/eval/runs');
  assert.equal(requests.evaluation.dashboard(30).path, '/api/eval/dashboard?days=30');
});

test('login is the urlencoded OAuth2 password form', () => {
  const request = requests.auth.login('a+b@example.com', 'p&w');
  assert.equal(request.body, 'username=a%2Bb%40example.com&password=p%26w');
  assert.deepEqual(request.headers, { 'Content-Type': 'application/x-www-form-urlencoded' });
});

test('upload is multipart with the optional fields only when set', () => {
  const request = requests.documents.upload(new Blob(['x']), { filename: 'a.md', collection_id: 2 });
  const form = request.body as FormData;
  assert.ok(form instanceof FormData);
  assert.equal((form.get('file') as File).name, 'a.md');
  assert.equal(form.get('collection_id'), '2');
  assert.equal(form.get('chunk_size'), null);
});

test('paths, methods and bodies for the console calls', () => {
  assert.deepEqual(requests.admin.grants.create(1, 2, 'read'), {
    method: 'POST',
    path: '/api/admin/grants',
    body: { group_id: 1, collection_id: 2, permission: 'read' },
    responseType: 'json',
  });
  assert.deepEqual(requests.admin.groups.removeMember(7, 2), {
    method: 'DELETE',
    path: '/api/admin/groups/7/members/2',
    responseType: 'json',
  });
  assert.equal(requests.runs.get(12).path, '/api/runs/12');
  assert.equal(requests.documents.download(4).responseType, 'blob');
});
