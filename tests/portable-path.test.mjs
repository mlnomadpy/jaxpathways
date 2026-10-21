import test from 'node:test';
import assert from 'node:assert/strict';
import { portablePath } from '../scripts/portable-path.mjs';

test('Windows recursive asset paths identify the same nested output folder and release key', () => {
  const windows = portablePath('reference\\suite\\report\\INTERPRETATION.md');
  const unix = 'reference/suite/report/INTERPRETATION.md';
  assert.equal(windows, unix);
  const output = `public/project-assets/reproducible-experiments/${windows}`;
  assert.equal(output.slice(0, output.lastIndexOf('/') + 1), 'public/project-assets/reproducible-experiments/reference/suite/report/');
});
