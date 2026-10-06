import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createJsonResource, fetchJson } from '../src/lib/http.js';
import { parseProjects, parseValidation } from '../src/lib/manifests.js';
import fs from 'node:fs';

test('JSON requests reject HTTP errors and malformed bodies', async () => {
  await assert.rejects(
    fetchJson('missing.json', { fetcher: async () => ({ ok: false, status: 404 }) }),
    /404/,
  );
  await assert.rejects(
    fetchJson('invalid.json', {
      fetcher: async () => ({
        ok: true,
        json: async () => {
          throw new SyntaxError('invalid JSON');
        },
      }),
    }),
    /invalid JSON/,
  );
});

test('timeouts bound response-body reads and abort the request', async () => {
  let signal;
  await assert.rejects(
    fetchJson('slow.json', {
      timeoutMs: 5,
      fetcher: async (_, options) => {
        signal = options.signal;
        return { ok: true, json: () => new Promise(() => {}) };
      },
    }),
    /timed out/,
  );
  assert.equal(signal.aborted, true);
});

test('resource cache shares concurrent requests, retries failures and isolates clients', async () => {
  const load = createJsonResource('receipt.json');
  let calls = 0;
  const client = async () => {
    calls++;
    if (calls === 1) throw new Error('offline');
    return { ok: true, json: async () => ({ revision: 2 }) };
  };
  await assert.rejects(load(client), /offline/);
  const [first, second] = await Promise.all([load(client), load(client)]);
  assert.deepEqual(first, { revision: 2 });
  assert.equal(first, second);
  assert.equal(calls, 2);
  await load(client);
  assert.equal(calls, 2);
  assert.deepEqual(await load(async () => ({ ok: true, json: async () => ({ revision: 3 }) })), {
    revision: 3,
  });
});

test('invalid manifest responses are rejected before rendering and do not poison the cache', async () => {
  const projects = JSON.parse(fs.readFileSync('public/api/v1/projects.json', 'utf8'));
  assert.equal(parseProjects(projects).length, projects.projects.length);
  const invalid = structuredClone(projects);
  invalid.projects[0].stages[0].command = null;
  assert.throws(() => parseProjects(invalid), /registry/);
  const receipt = JSON.parse(fs.readFileSync('public/validation.json', 'utf8'));
  assert.equal(parseValidation(receipt).lessons.length, receipt.lessons.length);
  assert.throws(() => parseValidation({ lessons: [] }), /receipt/);
  const load = createJsonResource('receipt.json', { parse: parseValidation });
  let calls = 0;
  const client = async () => ({ ok: true, json: async () => (++calls === 1 ? {} : receipt) });
  await assert.rejects(load(client), /receipt/);
  assert.equal((await load(client)).jax, receipt.jax);
  assert.equal(calls, 2);
});
