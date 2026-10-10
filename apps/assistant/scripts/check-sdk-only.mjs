// The app talks to the server only through @ragfabric/sdk (Phase 9, decision D11).
//
// Fails when any application file outside src/app/sdk/ imports HttpClient or
// spells an API path. Specs are exempt: they assert the URLs the SDK builds.
// Run with `npm run check:sdk-only`; CI runs it in the frontend job.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const app = fileURLToPath(new URL('../src/app', import.meta.url));
const allowed = join(app, 'sdk') + sep;
const rules = [
  { pattern: /\bHttpClient\b/, why: 'imports HttpClient; send requests through SdkHttp instead' },
  { pattern: /['"`]\/api(\/|['"`?])/, why: 'spells an API path; use a builder from requests' },
];

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  });
}

const problems = [];
for (const file of files(app)) {
  if (!file.endsWith('.ts') || file.endsWith('.spec.ts') || file.startsWith(allowed)) {
    continue;
  }
  readFileSync(file, 'utf8')
    .split('\n')
    .forEach((line, i) => {
      if (line.trim().startsWith('//') || line.trim().startsWith('*')) {
        return;
      }
      for (const rule of rules) {
        if (rule.pattern.test(line)) {
          problems.push(`src/app/${relative(app, file)}:${i + 1}: ${rule.why}`);
        }
      }
    });
}

if (problems.length > 0) {
  console.error(problems.join('\n'));
  console.error(`\n${problems.length} place(s) bypass @ragfabric/sdk.`);
  process.exit(1);
}
console.log('ok: the app reaches the server only through @ragfabric/sdk');
