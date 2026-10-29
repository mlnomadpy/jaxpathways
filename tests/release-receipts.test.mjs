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
    write('curriculum/releases.json', [{ version: '1.0.0', date: '2026-10-13', title: 'Fixture',
      editionKey: 'jaxpathways-release-receipts-fixture', modifiedAt: '2026-10-13T12:00:00Z' }]);
    write('curriculum/course.json', { phases: [] });
    write('curriculum/projects.json', ['projects/fixture/project.json']);
    mkdirSync(path.join(directory, 'projects/fixture'));
    write('projects/fixture/project.json', {id:'fixture'});
    writeFileSync(path.join(directory, 'projects/fixture/model.py'), 'value = 1\n');
    mkdirSync(path.join(directory, 'projects/fixture/foundations/reference/corrupt-package'), {recursive:true});
    writeFileSync(path.join(directory, 'projects/fixture/foundations/reference/corrupt-package/tokenizer.json'), '{"intentional":"invalid"}corrupt');
    mkdirSync(path.join(directory, 'projects/fixture/foundations/data'));
    write('projects/fixture/foundations/data/manifest.json', {split:'train'});
    write('curriculum/project-plans.json', [{id:'planned', source:'projects/planned/README.md'}]);
    mkdirSync(path.join(directory, 'projects/planned'));
    mkdirSync(path.join(directory, 'assessments'));
    writeFileSync(path.join(directory, 'projects/planned/README.md'), 'Original planned recipe\n');
    writeFileSync(path.join(directory, 'projects/planned/task.yaml'), 'task: frozen-task\n');
    writeFileSync(path.join(directory, 'assessments/fixture.md'), 'Original assessment\n');
    write('curriculum/project-validation.json', { executedAt: 'first run', projects: [] });
    const run = (...args) => spawnSync(process.execPath, ['release.mjs', ...args], { cwd: directory, encoding: 'utf8' });
    assert.equal(run('--prepare').status, 0);
    write('curriculum/project-validation.json', { executedAt: 'later run', projects: [{ status: 'passed' }] });
    write('curriculum/validation.json', { executedAt: 'later lesson run' });
    assert.equal(run().status, 0, 'Execution timestamps are not canonical course revisions');
    writeFileSync(path.join(directory, 'projects/fixture/foundations/reference/corrupt-package/tokenizer.json'), 'another deliberately invalid corruption fixture');
    assert.equal(run().status, 0, 'Nested retained failure fixtures are evidence, not canonical JSON');
    write('projects/fixture/foundations/data/manifest.json', {split:'changed'});
    assert.match(run().stderr, /Course sources changed/, 'Canonical nested data recipes still invalidate the snapshot');
    write('projects/fixture/foundations/data/manifest.json', {split:'train'});
    assert.equal(run().status, 0);
    for (const source of ['projects/planned/README.md', 'projects/planned/task.yaml', 'assessments/fixture.md']) {
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

test('the declared corrupt tokenizer canary is tracked by bytes while other JSON stays canonical and strict', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'jax-release-negative-json-'));
  const fixture = 'projects/text-harness/foundations/reference/run/corrupt-package/tokenizer.json';
  const neighbor = 'projects/text-harness/foundations/data/tokenizer-info.json';
  const write = (name, content) => {
    const filename = path.join(directory, name);
    mkdirSync(path.dirname(filename), { recursive: true });
    writeFileSync(filename, content);
  };
  const run = (...args) => spawnSync(process.execPath, [path.resolve('scripts/release.mjs'), ...args], {
    cwd: directory,
    encoding: 'utf8',
  });
  try {
    for (const name of ['curriculum', 'learning-paths', 'projects']) mkdirSync(path.join(directory, name));
    write('package.json', JSON.stringify({ version: '1.0.0' }));
    write('curriculum/releases.json', JSON.stringify([{ version: '1.0.0', date: '2026-10-29', title: 'Fixture',
      editionKey: 'jaxpathways-negative-json-fixture', modifiedAt: '2026-10-29T12:00:00Z' }]));
    write('curriculum/course.json', JSON.stringify({ phases: [] }));
    write('curriculum/projects.json', '[]');
    write(fixture, '{"tokenizer":1}corrupt');
    write(neighbor, '{"b":2,"a":1}');
    const prepared = run('--prepare');
    assert.equal(prepared.status, 0, prepared.stderr);
    const published = readFileSync(path.join(directory, 'curriculum/release-snapshots.json'), 'utf8');
    assert.equal(run().status, 0);

    write(neighbor, '{\n  "a": 1,\n  "b": 2\n}\n');
    assert.equal(run().status, 0, 'ordinary valid JSON retains canonical key-order/whitespace hashing');
    for (const changed of ['{"tokenizer":1}corrupt\n', '{"tokenizer":1}', '{"tokenizer":2}corrupt']) {
      write(fixture, changed);
      for (const args of [[], ['--prepare']]) {
        const result = run(...args);
        assert.notEqual(result.status, 0, 'every negative fixture byte change requires a new release');
        assert.match(result.stderr, /Course sources changed/);
        assert.equal(readFileSync(path.join(directory, 'curriculum/release-snapshots.json'), 'utf8'), published);
      }
    }
    write(fixture, '{"tokenizer":1}corrupt');
    assert.equal(run().status, 0);

    for (const invalidPath of [neighbor, 'projects/fixture/tokenizer.json']) {
      write(invalidPath, '{"unexpected":true}corrupt');
      const result = run('--prepare');
      assert.notEqual(result.status, 0, 'malformed JSON outside the exact canary must fail parsing');
      assert(result.stderr.includes(`Invalid canonical JSON in ${invalidPath}:`), result.stderr);
      assert.equal(readFileSync(path.join(directory, 'curriculum/release-snapshots.json'), 'utf8'), published);
      if (invalidPath === neighbor) write(neighbor, '{"a":1,"b":2}');
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
