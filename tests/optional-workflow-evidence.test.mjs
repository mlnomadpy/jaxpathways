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
    assert.equal(
      sha256(retainedPath(`${directory}/${file}`)),
      digest,
      `retained Diffrax asset: ${file}`,
    );
  }
});

test('decoder pilot evidence preserves the original receipt and all mapped source/package bytes', () => {
  const directory = 'projects/text-harness/foundations/reference';
  const index = read(`${directory}/evidence-index.json`);
  const receiptPath = retainedPath(index.raw_receipt.repository_path);
  assert.equal(sha256(receiptPath), index.raw_receipt.sha256);
  const receipt = read(receiptPath);
  assert.deepEqual(
    Object.keys(index.sourcebound_artifacts).sort(),
    Object.keys(receipt.artifacts).sort(),
  );
  assert.equal(Object.keys(receipt.artifacts).length, 17);
  for (const [name, artifact] of Object.entries(index.sourcebound_artifacts)) {
    assert.equal(
      artifact.original_path,
      receipt.artifacts[name].path,
      `${name}: original run identity`,
    );
    assert.equal(artifact.sha256, receipt.artifacts[name].sha256, `${name}: original digest`);
    assert.equal(
      sha256(retainedPath(artifact.repository_path)),
      artifact.sha256,
      `${name}: current retained bytes`,
    );
  }
});

test('the rendered decoder workflow is discoverable and retains readable evidence links', async () => {
  const window = new Window({ settings: { disableJavaScriptEvaluation: true } });
  try {
    window.document.write(
      readFileSync('dist/project-guides/text-harness-foundations.html', 'utf8'),
    );
    assert.match(window.document.querySelector('h1').textContent, /Decoder foundations/);
    assert.match(
      window.document.querySelector('.page-intro').textContent,
      /larger-model training.*remain open/s,
    );
    assert(window.document.querySelector('img[src$="foundations/reference/pilot-metrics.svg"]'));
    assert(window.document.querySelector('a[href$="foundations/reference/evidence-index.json"]'));
    assert(
      window.document.querySelector('a[href$="foundations/reference/run/pilot-results.json"]'),
    );
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
      assert.equal(
        sha256(`dist/project-assets/text-harness/${file}`),
        sha256(`projects/text-harness/${file}`),
      );
    }
  } finally {
    await window.happyDOM.abort();
  }
});

test('two-controller recovery keeps its figure, local scope and optional entry discoverable', async () => {
  const window = new Window({ settings: { disableJavaScriptEvaluation: true } });
  try {
    window.document.write(
      readFileSync('dist/project-guides/sharded-training-multiprocess.html', 'utf8'),
    );
    assert.match(window.document.querySelector('h1').textContent, /two-controller CPU job/);
    assert.match(
      window.document.querySelector('.page-intro').textContent,
      /local.*remote networking.*open/s,
    );
    const figure = window.document.querySelector(
      'img[src$="multiprocess/figures/recovery-sequence.svg"]',
    );
    assert(figure, 'the Markdown figure must render as an image');
    assert.match(figure.alt, /whole cluster/);
    assert(window.document.querySelector('a[href$="downloads/sharded-training.zip"]'));
    for (const file of ['dist/projects.html', 'dist/project-guides/sharded-training.html']) {
      assert.match(
        readFileSync(file, 'utf8'),
        /href="[^"]*project-guides\/sharded-training-multiprocess\.html"/,
      );
    }
    assert.equal(
      sha256('dist/project-assets/sharded-training/multiprocess/figures/recovery-sequence.svg'),
      sha256('projects/sharded-training/multiprocess/figures/recovery-sequence.svg'),
    );
    const receipt = read('docs/evidence/multiprocess-recovery-2026-10-21/root-all.json');
    for (const [file, digest] of Object.entries(receipt.multiprocess.source_hashes)) {
      assert.equal(sha256(`projects/sharded-training/multiprocess/${file}`), digest);
    }
  } finally {
    await window.happyDOM.abort();
  }
});

test('C03 pilot status stays distinct from its image-harness preparation project and planned capstone', async () => {
  const window = new Window({ settings: { disableJavaScriptEvaluation: true } });
  try {
    window.document.write(readFileSync('dist/project-guides/image-model-300m-pilot.html', 'utf8'));
    const guideStatus = window.document.querySelector(
      '[data-material-key="pilot:image-model-300m"]',
    );
    assert(guideStatus, 'the guide must show its own optional pilot status key');
    assert.match(guideStatus.textContent, /In development/);
    assert.match(guideStatus.textContent, /Source and numeric checks passed/);
    assert.match(guideStatus.textContent, /classification quality gate failed/);
    assert.match(guideStatus.textContent, /Teaching review pending/);
    const evidenceDownload = window.document.querySelector(
      'a[download][href$="/downloads/image-model-300m-pilot.zip"]',
    );
    assert.match(evidenceDownload.textContent, /Download source and numeric evidence/);
    assert.equal(
      window.document.querySelector('[data-material-key="projects:image-harness"]'),
      null,
      'the optional pilot must not display the unrelated preparation-project status',
    );

    const planWindow = new Window({ settings: { disableJavaScriptEvaluation: true } });
    planWindow.document.write(readFileSync('dist/project-plans/image-model-300m.html', 'utf8'));
    assert.match(
      planWindow.document.querySelector('.page-intro').textContent,
      /CPU pilot is available/,
    );
    assert.doesNotMatch(
      planWindow.document.querySelector('.page-intro').textContent,
      /no runnable workspace/,
    );
    const capstoneStatus = planWindow.document.querySelector(
      'section[aria-labelledby="capstone-status-heading"]',
    );
    assert.match(capstoneStatus.querySelector('h2').textContent, /Full capstone status/);
    assert(capstoneStatus.querySelector('[data-material-key="plans:image-model-300m"]'));
    await planWindow.happyDOM.abort();
  } finally {
    await window.happyDOM.abort();
  }
});

test('the local recovery extension status is separate from its sharded-training preparation status', async () => {
  const window = new Window({ settings: { disableJavaScriptEvaluation: true } });
  try {
    window.document.write(
      readFileSync('dist/project-guides/sharded-training-multiprocess.html', 'utf8'),
    );
    const preparation = window.document.querySelector(
      'section[aria-labelledby="preparation-project-status-heading"]',
    );
    assert.match(preparation.querySelector('h2').textContent, /Preparation project status/);
    assert(preparation.querySelector('[data-material-key="projects:sharded-training"]'));
    assert(
      preparation.querySelector('a[href$="project-guides/sharded-training.html"]'),
      'the preparation status names its canonical project guide',
    );
    const extension = window.document.querySelector(
      '[data-material-key="extension:sharded-training-multiprocess"]',
    );
    assert(extension, 'the optional extension needs its own status key');
    assert.match(extension.textContent, /In development/);
    assert.match(extension.textContent, /Current local all-stage checks passed/);
    assert.match(extension.textContent, /Teaching and remote Linux qualification pending/);
  } finally {
    await window.happyDOM.abort();
  }
});
