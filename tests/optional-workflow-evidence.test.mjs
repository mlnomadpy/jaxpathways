import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { Window } from 'happy-dom';

const root = process.cwd();
const read = (file) => JSON.parse(readFileSync(file, 'utf8'));
const sha256 = (file) => createHash('sha256').update(readFileSync(file)).digest('hex');

function retainedPath(relative) {
  assert.equal(typeof relative, 'string');
  assert(!path.isAbsolute(relative), 'retained evidence uses portable repository paths');
  const resolved = path.resolve(root, relative);
  assert(resolved.startsWith(root + path.sep), 'retained evidence stays inside the repository');
  return resolved;
}

test('Diffrax reference figures retain their executed source and raw-data identities', () => {
  const directory = 'projects/scientific-inverse/reference/diffrax';
  const receipt = read(`${directory}/receipt.json`);
  assert.equal(receipt.run.backend, 'cpu');
  assert.equal(receipt.run.dtype, 'float64');
  assert.equal(receipt.training_profile_reuses_fit_data, true);
  assert.equal(Object.keys(receipt.sources).length, 3);
  assert.equal(Object.keys(receipt.retainedAssets).length, 9);
  for (const [file, digest] of Object.entries(receipt.sources)) {
    assert.equal(sha256(retainedPath(file)), digest, `current Diffrax source: ${file}`);
  }
  for (const [file, digest] of Object.entries(receipt.retainedAssets)) {
    assert.equal(sha256(retainedPath(`${directory}/${file}`)), digest, `retained Diffrax asset: ${file}`);
  }
});

test('decoder pilot evidence preserves the original receipt and all mapped source/package bytes', () => {
  const directory = 'projects/text-harness/foundations/reference';
  const index = read(`${directory}/evidence-index.json`);
  const receiptPath = retainedPath(index.raw_receipt.repository_path);
  assert.equal(sha256(receiptPath), index.raw_receipt.sha256);
  const receipt = read(receiptPath);
  assert.deepEqual(Object.keys(index.sourcebound_artifacts).sort(), Object.keys(receipt.artifacts).sort());
  assert.equal(Object.keys(receipt.artifacts).length, 17);
  for (const [name, artifact] of Object.entries(index.sourcebound_artifacts)) {
    assert.equal(artifact.original_path, receipt.artifacts[name].path, `${name}: original run identity`);
    assert.equal(artifact.sha256, receipt.artifacts[name].sha256, `${name}: original digest`);
    assert.equal(sha256(retainedPath(artifact.repository_path)), artifact.sha256, `${name}: current retained bytes`);
  }
});

test('the rendered decoder workflow is discoverable and retains readable evidence links', async () => {
  const window = new Window({ settings: { disableJavaScriptEvaluation: true } });
  try {
    window.document.write(readFileSync('dist/project-guides/text-harness-foundations.html', 'utf8'));
    assert.match(window.document.querySelector('h1').textContent, /Decoder foundations/);
    assert.match(window.document.querySelector('.page-intro').textContent, /larger-model training.*remain open/s);
    assert(window.document.querySelector('img[src$="foundations/reference/pilot-metrics.svg"]'));
    assert(window.document.querySelector('a[href$="foundations/reference/evidence-index.json"]'));
    assert(window.document.querySelector('a[href$="foundations/reference/run/pilot-results.json"]'));
    assert(window.document.querySelector('a[href$="downloads/text-harness.zip"]'));
    assert(!window.document.querySelector('a[href$="reference/run/"]'));
    const catalog = readFileSync('dist/projects.html', 'utf8');
    assert.match(catalog, /href="[^"]*project-guides\/text-harness-foundations\.html"/);
    const parent = readFileSync('dist/project-guides/text-harness.html', 'utf8');
    assert.match(parent, /href="[^"]*project-guides\/text-harness-foundations\.html"/);
    for (const file of [
      'foundations/data/source/alice-pg11.txt',
      'foundations/reference/evidence-index.json',
      'foundations/reference/run/pilot-results.json',
      'foundations/reference/pilot-figure-data.json',
    ]) {
      assert.equal(sha256(`dist/project-assets/text-harness/${file}`), sha256(`projects/text-harness/${file}`));
    }
  } finally {
    await window.happyDOM.abort();
  }
});
