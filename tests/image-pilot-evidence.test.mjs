import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { Window } from 'happy-dom';

const root = process.cwd();
const project = 'projects/image-model-300m';
const packet = `${project}/reference/evidence/v14`;
const read = (file) => JSON.parse(readFileSync(file, 'utf8'));
const sha256 = (file) => createHash('sha256').update(readFileSync(file)).digest('hex');

test('the accepted image pilot source and public packet retain their exact inventories', () => {
  const recipe = read(`${packet}/approved-v14-recipe.json`);
  const approval = read(`${packet}/approved-v14-approval.json`);
  const identity = read(`${packet}/source-and-run-identities.json`);
  const manifest = read(`${packet}/packet-manifest.json`);
  assert.equal(
    sha256(`${packet}/approved-v14-recipe.json`),
    'df0a1a138f0635a39fcaa1b03f154403ed272560df9181fb5ad37722614d3c4c',
  );
  assert.equal(
    sha256(`${packet}/approved-v14-approval.json`),
    'f4fb56a70ec453ffd995c36c56c9b80cbd7cd9f739deb59f4db1fdd95c91f82b',
  );
  assert.equal(
    sha256(`${packet}/packet-manifest.json`),
    '3b1dcfe589fffc25a43da336beb07d82334b6f8971a73084e2b989b6e8f2a794',
  );
  assert.equal(approval.recipe_sha256, sha256(`${packet}/approved-v14-recipe.json`));
  assert.deepEqual(approval.source_files_sha256, recipe.source_sha256);
  assert.equal(Object.keys(recipe.source_sha256).length, 28);
  assert.equal(Object.keys(identity.as_run_source_inventory).length, 28);
  assert.equal(Object.keys(identity.as_run_candidate_python_inventory).length, 11);
  for (const [relative, digest] of Object.entries(recipe.source_sha256)) {
    assert.equal(sha256(`${project}/${relative}`), digest, `as-run source ${relative}`);
    assert.equal(
      identity.as_run_source_inventory[relative].sha256,
      digest,
      `source receipt ${relative}`,
    );
  }
  assert.equal(Object.keys(manifest.files).length, 29);
  for (const [relative, digest] of Object.entries(manifest.files)) {
    assert.equal(sha256(`${project}/${relative}`), digest, `public packet ${relative}`);
  }
  const packetFiles = readdirSync(project, { recursive: true })
    .map((relative) => path.posix.join(project, relative.toString()))
    .filter(
      (file) =>
        statSync(file).isFile() && !file.includes('/__pycache__/') && !file.endsWith('.pyc'),
    );
  assert.equal(packetFiles.length, 58, 'project contains only 28 source files and 30 packet files');
});

test('the built guide exposes all seven public figures and the full planned capstone', async () => {
  const window = new Window({ settings: { disableJavaScriptEvaluation: true } });
  try {
    window.document.write(readFileSync('dist/project-guides/image-model-300m-pilot.html', 'utf8'));
    const document = window.document;
    assert.match(document.querySelector('h1').textContent, /C03 flower-model teaching guide/);
    assert.match(document.querySelector('.page-intro').textContent, /classification gate failed/i);
    assert.match(
      document.querySelector('.page-intro').textContent,
      /full 300M capstone remains Planned/i,
    );
    assert.match(
      document.querySelector('.page-intro').textContent,
      /README records the source-freeze pre-run state/i,
    );
    assert(document.querySelector('a[href$="project-plans/image-model-300m.html"]'));
    const figures = [
      ...document.querySelectorAll('section[aria-labelledby="pilot-figures"] figure img'),
    ];
    assert.deepEqual(
      figures.map((image) => image.getAttribute('src').split('/').pop()),
      [
        'loss-mae.svg',
        'loss-simclr.svg',
        'loss-supervised.svg',
        'quality-classification.svg',
        'quality-brightness.svg',
        'quality-calibration.svg',
        'quality-retrieval.svg',
      ],
    );
    assert.equal(
      document.querySelectorAll('section[aria-labelledby="pilot-figures"] figure figcaption')
        .length,
      7,
    );
    for (const image of figures) {
      const relative = image.getAttribute('src').replace(/^\//, 'dist/');
      assert.equal(
        sha256(relative),
        sha256(
          `${project}/reference/evidence/v14/figures/${image.getAttribute('src').split('/').pop()}`,
        ),
      );
    }
    const catalog = readFileSync('dist/projects.html', 'utf8');
    assert.match(catalog, /project-guides\/image-model-300m-pilot\.html/);
    const parent = readFileSync('dist/project-plans/image-model-300m.html', 'utf8');
    assert.match(parent, /project-guides\/image-model-300m-pilot\.html/);
    assert.match(parent, /downloads\/image-model-300m-pilot\.zip/);
  } finally {
    await window.happyDOM.abort();
  }
});

test('the C03 pilot archive and course workspace contain the exact reviewed source and public packet', () => {
  const script = String.raw`import hashlib,json,sys,zipfile
project,archive,workspace=sys.argv[1:]
packet=project+'/reference/evidence/v14/'
recipe=json.load(open(packet+'approved-v14-recipe.json'))
manifest=json.load(open(packet+'packet-manifest.json'))
expected={'projects/image-model-300m/'+p for p in recipe['source_sha256']}
expected|={'projects/image-model-300m/'+p for p in manifest['files']}
expected.add('projects/image-model-300m/reference/evidence/v14/packet-manifest.json')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert set(z.namelist())==expected, set(z.namelist())^expected
 for rel,digest in recipe['source_sha256'].items():
  assert hashlib.sha256(z.read('projects/image-model-300m/'+rel)).hexdigest()==digest,rel
 for rel,digest in manifest['files'].items():
  assert hashlib.sha256(z.read('projects/image-model-300m/'+rel)).hexdigest()==digest,rel
with zipfile.ZipFile(workspace) as z:
 names=set(z.namelist())
 assert {'jaxpathways/'+p for p in expected}<=names
 actual={p.removeprefix('jaxpathways/') for p in names if p.startswith('jaxpathways/projects/image-model-300m/')}
 assert actual==expected, actual^expected
`;
  const result = spawnSync(
    'python3',
    [
      '-c',
      script,
      project,
      'public/downloads/image-model-300m-pilot.zip',
      'public/downloads/jax-course-workspace.zip',
    ],
    { encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
});
