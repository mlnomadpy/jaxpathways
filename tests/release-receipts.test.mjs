import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

test('rerunning execution receipts preserves a release while canonical source changes require a new one', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'jax-release-receipts-'));
  try {
    for (const name of ['curriculum', 'learning-paths', 'projects']) mkdirSync(path.join(directory, name));
    const write = (name, value) => writeFileSync(path.join(directory, name), JSON.stringify(value));
    writeFileSync(path.join(directory, 'release.mjs'), readFileSync('scripts/release.mjs'));
    write('package.json', { version: '1.0.0' });
    write('curriculum/releases.json', [{ version: '1.0.0', date: '2026-10-13', title: 'Fixture' }]);
    write('curriculum/course.json', { phases: [] });
    write('curriculum/project-validation.json', { executedAt: 'first run', projects: [] });
    const run = (...args) => spawnSync(process.execPath, ['release.mjs', ...args], { cwd: directory, encoding: 'utf8' });
    assert.equal(run('--prepare').status, 0);
    write('curriculum/project-validation.json', { executedAt: 'later run', projects: [{ status: 'passed' }] });
    write('curriculum/validation.json', { executedAt: 'later lesson run' });
    assert.equal(run().status, 0, 'Execution timestamps are not canonical course revisions');
    write('learning-paths/fixture.json', { title: 'New learning material' });
    const changed = run();
    assert.notEqual(changed.status, 0);
    assert.match(changed.stderr, /Course sources changed/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
