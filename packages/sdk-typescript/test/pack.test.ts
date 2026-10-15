// What `npm pack` would publish (decision D24: prepared, not published).

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

test('the package contains only dist, the README, the licence and package.json', () => {
  const cwd = fileURLToPath(new URL('../../', import.meta.url));
  const out = execFileSync('npm', ['pack', '--dry-run', '--json'], { cwd, encoding: 'utf8' });
  const start = out.indexOf('[');
  const [manifest] = JSON.parse(out.slice(start)) as { name: string; files: { path: string }[] }[];
  assert.equal(manifest!.name, '@ragfabric/sdk');
  const paths = manifest!.files.map((f) => f.path);
  for (const path of paths) {
    assert.ok(
      path.startsWith('dist/') || ['README.md', 'LICENSE', 'package.json'].includes(path),
      `unexpected file in the package: ${path}`,
    );
  }
  assert.ok(paths.includes('dist/index.js'));
  assert.ok(paths.includes('dist/index.d.ts'));
  assert.ok(paths.includes('LICENSE'));
});
