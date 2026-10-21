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
    writeFileSync(path.join(directory, 'portable-path.mjs'), readFileSync('scripts/portable-path.mjs'));
    write('package.json', { version: '1.0.0' });
    write('curriculum/releases.json', [{ version: '1.0.0', date: '2026-10-13', title: 'Fixture' }]);
    write('curriculum/course.json', { phases: [] });
    write('curriculum/projects.json', ['projects/fixture/project.json']);
    mkdirSync(path.join(directory, 'projects/fixture'));
    write('projects/fixture/project.json', {id:'fixture'});
    writeFileSync(path.join(directory, 'projects/fixture/model.py'), 'value = 1\n');
    write('curriculum/project-plans.json', [{id:'planned', source:'projects/planned/README.md'}]);
    mkdirSync(path.join(directory, 'projects/planned'));
    mkdirSync(path.join(directory, 'assessments'));
    writeFileSync(path.join(directory, 'projects/planned/README.md'), 'Original planned recipe\n');
    writeFileSync(path.join(directory, 'assessments/fixture.md'), 'Original assessment\n');
    write('curriculum/project-validation.json', { executedAt: 'first run', projects: [] });
    const run = (...args) => spawnSync(process.execPath, ['release.mjs', ...args], { cwd: directory, encoding: 'utf8' });
    assert.equal(run('--prepare').status, 0);
    write('curriculum/project-validation.json', { executedAt: 'later run', projects: [{ status: 'passed' }] });
    write('curriculum/validation.json', { executedAt: 'later lesson run' });
    assert.equal(run().status, 0, 'Execution timestamps are not canonical course revisions');
    for (const source of ['projects/planned/README.md', 'assessments/fixture.md']) {
      const filename=path.join(directory, source);
      const original=readFileSync(filename,'utf8');
      writeFileSync(filename,original+'A revised teaching contract\n');
      const revision=run();
      assert.notEqual(revision.status,0,source+' must invalidate the release snapshot');
      assert.match(revision.stderr,/Course sources changed/);
      writeFileSync(filename,original);
      assert.equal(run().status,0);
    }
    writeFileSync(path.join(directory, 'projects/fixture/model.py'), 'value = 2\n');
    const changed = run();
    assert.notEqual(changed.status, 0);
    assert.match(changed.stderr, /Course sources changed/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
