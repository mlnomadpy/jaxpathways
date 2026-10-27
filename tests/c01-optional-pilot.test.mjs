import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';

const projectId = 'embedding-model-300m';
const projectRoot = `projects/${projectId}`;
const indexPath = `${projectRoot}/reference/evidence-index.json`;

test('C01, C02 and C03 are scoped pilots while all ten full capstones stay planned', () => {
  const plans = JSON.parse(readFileSync('curriculum/project-plans.json', 'utf8'));
  const projects = JSON.parse(readFileSync('curriculum/projects.json', 'utf8')).map((path) =>
    JSON.parse(readFileSync(path, 'utf8')),
  );
  assert.equal(plans.length, 10);
  assert(plans.every((plan) => plan.status === 'planned'));
  const pilotPlans = plans.filter((plan) => plan.pilot);
  assert.deepEqual(
    pilotPlans.map((plan) => plan.id),
    [projectId, 'causal-model-300m', 'image-model-300m'],
  );
  assert.equal(
    projects.some((project) => project.id === projectId),
    false,
  );
  assert.deepEqual(pilotPlans[0].pilot, {
    title: 'C01 SciFact CPU pilot',
    scope: pilotPlans[0].pilot.scope,
    anchor: 'cpu-pilot',
    download: 'embedding-model-300m-pilot.zip',
    evidence: 'reference/evidence-index.json',
  });
  assert.match(pilotPlans[0].pilot.scope, /floors failed/i);
  assert.match(pilotPlans[0].pilot.scope, /does not qualify/i);
  const c03 = pilotPlans[2];
  assert.match(c03.pilot.scope, /0\.318522.*0\.40.*0\.363836/);
  assert.equal(c03.pilot.anchor, 'cpu-pilot');
  assert.equal(c03.pilot.download, 'image-model-300m-pilot.zip');
  assert.equal(c03.pilot.evidence, 'reference/evidence/v14/packet-manifest.json');
  assert.equal(
    c03.pilot.evidenceSha256,
    '3b1dcfe589fffc25a43da336beb07d82334b6f8971a73084e2b989b6e8f2a794',
  );
  assert.deepEqual(c03.pilot.guide, {
    path: 'project-guides/image-model-300m-pilot.html',
    label: 'Read the flower-model pilot guide and figures',
  });
});

test('audit reports optional pilot evidence separately from full-capstone status', () => {
  const script = `import importlib.util,json; s=importlib.util.spec_from_file_location('audit','scripts/audit-learning-material.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.build()))`;
  const result = spawnSync('python3', ['-c', script], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const audit = JSON.parse(result.stdout);
  const c01 = audit.items.find((item) => item.kind === 'plans' && item.id === projectId);
  const c03 = audit.items.find((item) => item.kind === 'plans' && item.id === 'image-model-300m');
  assert(c01);
  assert(c03);
  assert.equal(c01.level, 'planned');
  assert.equal(audit.summary.fullCapstonesPlanned, 10);
  assert.equal(audit.summary.optionalPilots, 3);
  const finding = audit.findings.find((item) => item.id === 'production-capstones');
  assert.equal(finding.status, 'open');
  assert.match(finding.title, /all ten full model-lifecycle capstones remain planned/i);
  assert.match(finding.action, /retrieval-quality floors failed/i);
  assert.match(finding.action, /all six branches failed their frozen QA quality floors/i);
  assert.match(finding.action, /C03.*0\.318522.*quality gate failed/);
  assert.equal(typeof c01.pilotAvailable, 'boolean');
  assert.equal(c01.pilotAvailable, existsSync(indexPath));
  if (!c01.pilotAvailable) {
    assert.match(c01.validation, /pilot packet pending/i);
  } else {
    assert.match(c01.validation, /evidence index/);
    assert.match(c01.pilotEvidenceSha256, /^[a-f0-9]{64}$/);
    assert(Object.keys(c01.pilotSourceHashes).length > 0);
    assert(Object.keys(c01.pilotPublicFiles).length > 0);
  }
  assert.equal(c03.level, 'planned');
  assert.equal(c03.pilotAvailable, true);
  assert.equal(
    c03.pilotEvidenceSha256,
    '3b1dcfe589fffc25a43da336beb07d82334b6f8971a73084e2b989b6e8f2a794',
  );
  assert.equal(Object.keys(c03.pilotSourceHashes).length, 28);
  assert.equal(Object.keys(c03.pilotPublicFiles).length, 26);
  assert.match(c03.validation, /evidence index.*3b1dcfe/);
});

test('built C01 route and source archive expose only hash-verified public evidence', () => {
  const index = JSON.parse(readFileSync(indexPath, 'utf8'));
  for (const [path, expectedHash] of Object.entries(index.public_files)) {
    const published = `public/project-assets/${projectId}/${path}`;
    assert(existsSync(published), `public pilot evidence asset is missing: ${path}`);
    const actualHash = createHash('sha256').update(readFileSync(published)).digest('hex');
    assert.equal(actualHash, expectedHash, path);
  }
  const archive = 'public/downloads/embedding-model-300m-pilot.zip';
  assert(existsSync(archive), 'source-only pilot ZIP is missing');
  const script = String.raw`import hashlib,json,sys,zipfile
archive,index_path=sys.argv[1:]
index=json.load(open(index_path))
with zipfile.ZipFile(archive) as z:
 names=set(z.namelist())
 assert z.testzip() is None
 expected={'projects/embedding-model-300m/reference/evidence-index.json'}
 expected.update('projects/embedding-model-300m/'+p for p in index['public_files'])
 expected.update(index['source_files_sha256'])
 expected.update({'projects/embedding-model-300m/README.md','projects/embedding-model-300m/starter/README.md','projects/embedding-model-300m/starter/model.py','projects/embedding-model-300m/tests/check.py','projects/embedding-model-300m/data/LICENSES.md','projects/embedding-model-300m/requirements-embedding-lock.txt','projects/embedding-model-300m/tools/build_evidence_index.py','projects/embedding-model-300m/tools/pack_reference_evidence.py'})
 assert expected <= names, (expected-names)
 reference_members={name for name in names if name.startswith('projects/embedding-model-300m/reference/')}
 reference_expected={name for name in expected if name.startswith('projects/embedding-model-300m/reference/')}
 assert reference_members==reference_expected, {'missing':reference_expected-reference_members,'unexpected':reference_members-reference_expected}
 for path,digest in index['public_files'].items():
  name='projects/embedding-model-300m/'+path
  assert hashlib.sha256(z.read(name)).hexdigest()==digest, name
 for path,digest in index['source_files_sha256'].items():
  assert hashlib.sha256(z.read(path)).hexdigest()==digest, path
 forbidden={"model.safetensors","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt"}
 suffixes={".safetensors",".ckpt",".pt",".pth",".onnx",".gguf",".msgpack",".bin"}
 unsafe=[name for name in names if name.rsplit('/',1)[-1].lower() in forbidden or name.lower().endswith(tuple(suffixes)) or 'baseline_snapshot' in name.lower().split('/') or ('/checkpoints/' in name.lower() and not name.lower().endswith('.json'))]
 assert not unsafe, unsafe
`;
  const checked = spawnSync('python3', ['-c', script, archive, indexPath], { encoding: 'utf8' });
  assert.equal(checked.status, 0, checked.stderr);
  const assetsCheck = String.raw`import json,os,sys
root,index_path=sys.argv[1:]
index=json.load(open(index_path))
expected=set(index['public_files'])|{'reference/evidence-index.json'}
actual=set()
for directory,_,files in os.walk(root):
 for name in files:
  relative=os.path.relpath(os.path.join(directory,name),root).replace(os.sep,'/')
  if relative.startswith('reference/'): actual.add(relative)
assert actual==expected, {'missing':expected-actual,'unexpected':actual-expected}
`;
  const assetsResult = spawnSync(
    'python3',
    ['-c', assetsCheck, `public/project-assets/${projectId}`, indexPath],
    { encoding: 'utf8' },
  );
  assert.equal(assetsResult.status, 0, assetsResult.stderr);
  const workspace = 'public/downloads/jax-course-workspace.zip';
  assert(existsSync(workspace), 'portable course workspace is missing');
  const numericalAssets = Object.keys(index.public_files).filter((path) => path.endsWith('.npz'));
  assert(
    numericalAssets.length > 0,
    'public evidence index must include required numeric NPZ assets',
  );
  const workspaceCheck = String.raw`import json,sys,zipfile
archive,index_path=sys.argv[1:]
index=json.load(open(index_path))
with zipfile.ZipFile(archive) as z:
 names=set(z.namelist())
 expected={'jaxpathways/projects/embedding-model-300m/'+p for p in index['public_files']}
 assert expected <= names, expected-names
 reference_expected=expected|{'jaxpathways/projects/embedding-model-300m/reference/evidence-index.json'}
 reference_actual={name for name in names if name.startswith('jaxpathways/projects/embedding-model-300m/reference/')}
 assert reference_actual==reference_expected, {'missing':reference_expected-reference_actual,'unexpected':reference_actual-reference_expected}
`;
  const workspaceResult = spawnSync('python3', ['-c', workspaceCheck, workspace, indexPath], {
    encoding: 'utf8',
  });
  assert.equal(workspaceResult.status, 0, workspaceResult.stderr);
  const c01Page = readFileSync('dist/project-plans/embedding-model-300m.html', 'utf8');
  assert(c01Page.includes('Build a staged embedding model: CPU pilot and project plan'));
  assert(c01Page.includes('id="cpu-pilot"'));
  assert(c01Page.includes('retrieval floors failed'));
  assert(c01Page.includes('/downloads/embedding-model-300m-pilot.zip'));
  assert(c01Page.includes('/project-assets/embedding-model-300m/reference/evidence-index.json'));
  const statusPage = readFileSync('dist/content-status.html', 'utf8');
  assert(statusPage.includes('separately scoped CPU pilot may be available'));
  const catalog = readFileSync('dist/projects.html', 'utf8');
  assert(catalog.includes('Full capstone planned · bounded CPU pilot available'));
  assert(catalog.includes('/downloads/embedding-model-300m-pilot.zip'));
  for (const plan of JSON.parse(readFileSync('curriculum/project-plans.json', 'utf8'))) {
    const route = readFileSync(`dist/project-plans/${plan.id}.html`, 'utf8');
    if (plan.id === projectId) continue;
    if (!plan.pilot) {
      assert(!route.includes('bounded CPU pilot available'), `${plan.id}: pilot badge leaked`);
    }
    assert(!route.includes('embedding-model-300m-pilot.zip'), `${plan.id}: pilot download leaked`);
  }
});
