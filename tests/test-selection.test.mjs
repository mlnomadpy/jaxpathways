import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFileSync, mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import {
  dependencyReader,
  preparationFor,
  selectTests,
  suites,
  validateInventory,
} from '../scripts/test-suites.mjs';

test('every discovered Node test belongs to exactly one suite', () => {
  validateInventory();
  const root = mkdtempSync(path.join(os.tmpdir(), 'test-inventory-'));
  try {
    for (const file of Object.values(suites).flat()) {
      mkdirSync(path.join(root, path.dirname(file)), { recursive: true });
      writeFileSync(path.join(root, file), '');
    }
    validateInventory(root);
    writeFileSync(path.join(root, 'tests/unassigned.test.mjs'), '');
    assert.throws(() => validateInventory(root), /unassigned.*tests\/unassigned.test.mjs/);
    rmSync(path.join(root, 'tests/unassigned.test.mjs'));
    mkdirSync(path.join(root, 'tests/nested'));
    writeFileSync(path.join(root, 'tests/nested/unassigned.test.mjs'), '');
    assert.throws(() => validateInventory(root), /tests\/nested\/unassigned.test.mjs/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('transitive source edits select their consumers and built-page checks', () => {
  const selected = selectTests({ changedFiles: ['src/lib/storage.js'] });
  assert(selected.includes('tests/browser-actions.test.mjs'));
  assert(selected.includes('scripts/platform-form.test.cjs'));
  assert(selected.includes('tests/browser-migration.test.mjs'));
  assert(!selected.includes('tests/hardware-evidence.test.mjs'));
  assert(!selected.includes('tests/portable-path.test.mjs'));
  assert(dependencyReader()('scripts/platform-form.test.cjs').has('src/lib/storage.js'));
});

test('course and tooling inputs conservatively select generated suites or every test', () => {
  const course = selectTests({ changedFiles: ['phases/00-welcome/01-first/lesson.json'] });
  for (const suite of ['app', 'content', 'site']) {
    for (const file of suites[suite]) assert(course.includes(file));
  }
  assert.deepEqual(selectTests({ changedFiles: ['package-lock.json'] }), selectTests());
  assert.deepEqual(selectTests({ changedFiles: ['unknown-input.txt'] }), selectTests());
  assert.deepEqual(selectTests({ changedFiles: [] }), []);
  assert.deepEqual(selectTests({ changedFiles: ['browser-tests/reader-print.spec.ts'] }), []);
});

test('direct test selection and suite restrictions keep focused runs bounded', () => {
  assert.deepEqual(selectTests({ changedFiles: ['tests/choices.test.mjs'] }), [
    'tests/choices.test.mjs',
  ]);
  assert.deepEqual(selectTests({ suite: 'unit', changedFiles: ['tests/http.test.mjs'] }), []);
  assert.deepEqual(selectTests({ changedFiles: ['tests/deleted.test.mjs'] }), []);
  assert.throws(() => selectTests({ suite: 'typo' }), /Unknown suite/);
  const portable = selectTests({ changedFiles: ['scripts/portable-path.mjs'] });
  assert(portable.includes('tests/releases.test.mjs'));
  assert(portable.includes('tests/release-receipts.test.mjs'));
  assert.equal(preparationFor(suites.unit), 'none');
  assert.equal(preparationFor(suites.app), 'content');
  assert.equal(preparationFor(suites.site), 'site');
});

test('listing a changed test never builds and invalid runner arguments fail', () => {
  const listed = spawnSync(
    process.execPath,
    ['scripts/run-tests.mjs', '--changed', '--list', 'tests/choices.test.mjs'],
    { encoding: 'utf8' },
  );
  assert.equal(listed.status, 0, listed.stderr);
  assert.equal(
    listed.stdout,
    'Selected 1 Node test files; preparation: none.\ntests/choices.test.mjs\n',
  );
  for (const args of [
    ['--suite', 'typo'],
    ['--test-concurrency', '0'],
    ['tests/missing.test.mjs'],
    ['--unknown'],
  ]) {
    const result = spawnSync(process.execPath, ['scripts/run-tests.mjs', ...args], {
      encoding: 'utf8',
    });
    assert.equal(result.status, 1, args.join(' '));
  }
});

test('unit imports do not require generated public manifests', () => {
  const dependencies = dependencyReader();
  for (const file of suites.unit) {
    assert.deepEqual(
      [...dependencies(file)].filter((input) => input.startsWith('public/')),
      [],
      file,
    );
  }
});

test('Git selection includes staged, unstaged, untracked and committed branch changes', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'test-changes-'));
  const git = (...args) => {
    const result = spawnSync('git', args, { cwd: root, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.trim();
  };
  const list = (...args) => {
    const result = spawnSync(
      process.execPath,
      ['scripts/run-tests.mjs', '--changed', '--list', ...args],
      { cwd: root, encoding: 'utf8' },
    );
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.split('\n').slice(1).filter(Boolean);
  };
  const commit = () =>
    git(
      '-c',
      'user.name=Test fixture',
      '-c',
      'user.email=fixture@example.invalid',
      '-c',
      'commit.gpgsign=false',
      'commit',
      '-m',
      'fixture',
    );
  try {
    for (const file of Object.values(suites).flat()) {
      mkdirSync(path.join(root, path.dirname(file)), { recursive: true });
      writeFileSync(path.join(root, file), '');
    }
    for (const name of ['run-tests.mjs', 'test-suites.mjs']) {
      copyFileSync(
        new URL(`../scripts/${name}`, import.meta.url),
        path.join(root, 'scripts', name),
      );
    }
    git('init', '--quiet');
    git('add', '.');
    commit();
    const base = git('rev-parse', 'HEAD');
    assert.deepEqual(list(), []);
    writeFileSync(path.join(root, 'tests/choices.test.mjs'), '// staged');
    git('add', 'tests/choices.test.mjs');
    writeFileSync(path.join(root, 'tests/browser-actions.test.mjs'), '// unstaged');
    assert.deepEqual(list(), ['tests/browser-actions.test.mjs', 'tests/choices.test.mjs']);
    git('add', '.');
    commit();
    assert.deepEqual(list(), []);
    assert.deepEqual(list('--since', base), [
      'tests/browser-actions.test.mjs',
      'tests/choices.test.mjs',
    ]);
    writeFileSync(path.join(root, 'new-tooling.json'), '{}');
    assert.deepEqual(list(), selectTests());
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
