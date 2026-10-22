import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function makeFixture() {
  const directory = mkdtempSync(path.join(tmpdir(), 'jax-release-edition-metadata-'));
  for (const name of ['curriculum', 'learning-paths', 'projects', 'assessments']) {
    mkdirSync(path.join(directory, name), { recursive: true });
  }
  const writeJson = (name, value) => writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + '\n');
  writeFileSync(path.join(directory, 'release.mjs'), readFileSync('scripts/release.mjs'));
  writeFileSync(path.join(directory, 'portable-path.mjs'), readFileSync('scripts/portable-path.mjs'));
  writeJson('curriculum/course.json', { phases: [] });
  writeJson('curriculum/projects.json', ['projects/fixture/project.json']);
  mkdirSync(path.join(directory, 'projects/fixture'), { recursive: true });
  writeJson('projects/fixture/project.json', { id: 'fixture' });
  writeJson('package.json', { version: '0.14.1' });
  const setRelease = release => {
    writeJson('package.json', { version: release.version });
    writeJson('curriculum/releases.json', [release]);
  };
  const run = (...args) => spawnSync(process.execPath, ['release.mjs', ...args], { cwd: directory, encoding: 'utf8' });
  return { directory, setRelease, run, snapshotPath: path.join(directory, 'curriculum/release-snapshots.json') };
}

const baseRelease = version => ({ version, date: '2026-10-13', title: `Fixture ${version}` });
const validMetadata = version => ({
  ...baseRelease(version),
  editionKey: `jaxpathways-fixture-${version}`,
  modifiedAt: '2026-10-13T12:00:00Z',
});

test('invalid edition metadata is rejected before any release snapshot is written', () => {
  const fixture = makeFixture();
  try {
    const invalidRecords = [
      [baseRelease('0.15.0'), /requires editionKey and modifiedAt/],
      [{ ...baseRelease('0.15.0'), editionKey: 'fixture-key' }, /provided together/],
      [{ ...baseRelease('0.15.0'), modifiedAt: '2026-10-13T12:00:00Z' }, /provided together/],
      [{ ...baseRelease('0.14.1'), editionKey: 'fixture-key' }, /provided together/],
      [{ ...baseRelease('0.14.1'), modifiedAt: '2026-10-13T12:00:00Z' }, /provided together/],
      [{ ...validMetadata('0.15.0'), editionKey: '' }, /non-empty trimmed/],
      [{ ...validMetadata('0.15.0'), editionKey: ' padded ' }, /non-empty trimmed/],
      [{ ...validMetadata('0.15.0'), modifiedAt: '2026-10-13' }, /UTC YYYY-MM-DDThh:mm:ssZ/],
      [{ ...validMetadata('0.15.0'), modifiedAt: '2026-10-13T12:00:00+00:00' }, /UTC YYYY-MM-DDThh:mm:ssZ/],
      [{ ...validMetadata('0.15.0'), modifiedAt: '2026-02-30T12:00:00Z' }, /valid UTC calendar date/],
      [{ ...validMetadata('01.15.0') }, /plain semantic major\.minor\.patch/],
      [{ ...validMetadata('0.015.0') }, /plain semantic major\.minor\.patch/],
      [{ ...validMetadata('0.15.00') }, /plain semantic major\.minor\.patch/],
      [{ ...validMetadata('0.15.0-rc.1') }, /plain semantic major\.minor\.patch/],
      [{ ...validMetadata('0.15.0+meta') }, /plain semantic major\.minor\.patch/],
      [{ ...validMetadata('0.15.0-..') }, /plain semantic major\.minor\.patch/],
    ];
    for (const [release, errorPattern] of invalidRecords) {
      fixture.setRelease(release);
      const result = fixture.run('--prepare');
      assert.notEqual(result.status, 0, release.version);
      assert.match(result.stderr, errorPattern, release.version);
      assert.equal(existsSync(fixture.snapshotPath), false, 'invalid metadata must not create a snapshot');
    }
  } finally {
    rmSync(fixture.directory, { recursive: true, force: true });
  }
});

test('legacy and explicit edition records prepare once and rerun with an unchanged snapshot', () => {
  const fixture = makeFixture();
  try {
    const releases = [baseRelease('0.14.1'), validMetadata('0.15.0'), validMetadata('1.0.0')];
    for (const release of releases) {
      fixture.setRelease(release);
      const prepare = fixture.run('--prepare');
      assert.equal(prepare.status, 0, prepare.stderr);
      const snapshot = readFileSync(fixture.snapshotPath);
      const repeat = fixture.run();
      assert.equal(repeat.status, 0, repeat.stderr);
      assert.deepEqual(readFileSync(fixture.snapshotPath), snapshot);
    }

    const acceptedSnapshot = readFileSync(fixture.snapshotPath);
    fixture.setRelease(baseRelease('1.0.1'));
    const invalid = fixture.run('--prepare');
    assert.notEqual(invalid.status, 0);
    assert.match(invalid.stderr, /requires editionKey and modifiedAt/);
    assert.deepEqual(readFileSync(fixture.snapshotPath), acceptedSnapshot, 'invalid metadata must not rewrite an existing snapshot');
  } finally {
    rmSync(fixture.directory, { recursive: true, force: true });
  }
});
