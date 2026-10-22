import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import test from 'node:test';
import { Window } from 'happy-dom';

const project = 'projects/image-model-300m';
const packet = `${project}/reference/evidence/v14`;
const read = (file) => JSON.parse(readFileSync(file, 'utf8'));
const sha256 = (file) => createHash('sha256').update(readFileSync(file)).digest('hex');

test('canonical C03 pilot metadata drives the parent route and machine discovery', async () => {
  const plan = read('curriculum/project-plans.json').find((item) => item.id === 'image-model-300m');
  assert.equal(plan.status, 'planned');
  assert.match(plan.pilot.scope, /0\.318522.*0\.40.*0\.363836/);
  assert.equal(plan.pilot.evidenceSha256, sha256(`${packet}/packet-manifest.json`));

  const window = new Window({ settings: { disableJavaScriptEvaluation: true } });
  try {
    window.document.write(readFileSync('dist/project-plans/image-model-300m.html', 'utf8'));
    const page = window.document;
    const pilotSections = page.querySelectorAll('section#cpu-pilot');
    assert.equal(pilotSections.length, 1, 'the plan must render one canonical pilot section');
    const pilot = pilotSections[0];
    assert(pilot.querySelector('a[href$="/downloads/image-model-300m-pilot.zip"][download]'));
    assert(
      pilot.querySelector(
        'a[href$="/project-assets/image-model-300m/reference/evidence/v14/packet-manifest.json"]',
      ),
    );
    assert(pilot.querySelector('a[href$="/project-guides/image-model-300m-pilot.html"]'));
    assert.equal(page.querySelectorAll('#image-cpu-pilot-heading').length, 0);
  } finally {
    await window.happyDOM.abort();
  }

  const llms = readFileSync('public/llms.txt', 'utf8');
  assert.match(
    llms,
    /C03 flower-model CPU pilot.*\[Source and numeric evidence archive\]\(downloads\/image-model-300m-pilot\.zip\)/,
  );

  const catalogWindow = new Window({ settings: { disableJavaScriptEvaluation: true } });
  try {
    catalogWindow.document.write(readFileSync('dist/projects.html', 'utf8'));
    const imageHarness = catalogWindow.document.querySelector(
      'article[data-project-id="image-harness"]',
    );
    assert(
      imageHarness.querySelector('a[href$="project-guides/image-model-300m-pilot.html"]'),
      'the optional guide link must be discovered from the plan pilot descriptor',
    );
  } finally {
    await catalogWindow.happyDOM.abort();
  }
});

test('C03 loose project assets contain exactly the approved 26 safe public files', () => {
  const manifest = read(`${packet}/packet-manifest.json`);
  const expected = new Set([
    'README.md',
    'data/LICENSE.txt',
    ...Object.keys(manifest.files).filter((file) => !file.startsWith('tools/')),
    'reference/evidence/v14/packet-manifest.json',
  ]);
  assert.equal(expected.size, 26);
  const assetRoot = 'public/project-assets/image-model-300m';
  const actual = new Set(
    readdirSync(assetRoot, { recursive: true })
      .map((relative) => relative.toString().split(path.sep).join('/'))
      .filter((relative) => statSync(path.join(assetRoot, relative)).isFile()),
  );
  assert.deepEqual(actual, expected);
  assert([...actual].every((file) => !file.startsWith('tools/')));
  for (const relative of actual) {
    const sourceHash =
      manifest.files[relative] ??
      read(`${packet}/source-and-run-identities.json`).as_run_source_inventory[relative]?.sha256;
    const expectedHash =
      sourceHash ??
      (relative === 'README.md'
        ? read(`${packet}/approved-v14-recipe.json`).source_sha256[relative]
        : relative === 'data/LICENSE.txt'
          ? read(`${packet}/approved-v14-recipe.json`).source_sha256[relative]
          : sha256(`${project}/${relative}`));
    assert.equal(sha256(`${assetRoot}/${relative}`), expectedHash, relative);
  }
});
