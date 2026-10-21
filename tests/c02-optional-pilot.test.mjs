import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const root = 'projects/causal-model-300m';
const index = JSON.parse(readFileSync(`${root}/reference/evidence-index.json`, 'utf8'));

test('C02 source and public evidence retain their exact identities', () => {
  for (const [path, expected] of Object.entries(index.source_files_sha256)) {
    assert.equal(createHash('sha256').update(readFileSync(path)).digest('hex'), expected, path);
  }
  for (const [path, expected] of Object.entries(index.public_files)) {
    assert.equal(
      createHash('sha256')
        .update(readFileSync(`public/project-assets/causal-model-300m/${path}`))
        .digest('hex'),
      expected,
      path,
    );
  }
  for (const name of [
    'hellaswag/hellaswag.yaml',
    'arc_easy/arc_easy.yaml',
    'LICENSE.md',
    'NOTICE.md',
  ]) {
    assert(`${root}/data/lm_eval_tasks/${name}` in index.source_files_sha256);
  }
});

test('C02 archive is complete and excludes fitted model and tokenizer payloads', () => {
  const script = String.raw`import hashlib,json,sys,zipfile
root,index_file,archive,workspace=sys.argv[1:]
index=json.load(open(index_file))
expected=set(index['source_files_sha256'])|{root+'/'+p for p in index['public_files']}|{index_file}
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert set(z.namelist())==expected, (set(z.namelist())^expected)
 for path,digest in index['source_files_sha256'].items():
  assert hashlib.sha256(z.read(path)).hexdigest()==digest,path
 for path,digest in index['public_files'].items():
  assert hashlib.sha256(z.read(root+'/'+path)).hexdigest()==digest,path
 assert not any(p.endswith(('.npz','.safetensors','.ckpt','.pt','.pth','.bin')) or p.rsplit('/',1)[-1] in {'tokenizer.json','tokenizer_config.json','weights.json'} for p in z.namelist())
with zipfile.ZipFile(workspace) as z:
 assert {'jaxpathways/'+p for p in expected} <= set(z.namelist())
`;
  const result = spawnSync(
    'python3',
    [
      '-c',
      script,
      root,
      `${root}/reference/evidence-index.json`,
      'public/downloads/causal-model-300m-pilot.zip',
      'public/downloads/jax-course-workspace.zip',
    ],
    { encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
});

test('QA projections preserve all branches and scores without corpus inputs', () => {
  const packet = `${root}/reference/evidence/c02-pilot`;
  const rows = readFileSync(`${packet}/qa-generations-projection.jsonl`, 'utf8')
    .trim()
    .split('\n')
    .map(JSON.parse);
  const metrics = JSON.parse(readFileSync(`${packet}/metrics.json`, 'utf8'));
  assert.equal(rows.length, 768);
  assert.equal(new Set(rows.map((r) => `${r.branch}/${r.question_id}`)).size, 768);
  for (const row of rows) {
    for (const field of ['prompt', 'context', 'human_answers', 'prompt_token_ids'])
      assert(!(field in row));
  }
  for (const [branch, report] of Object.entries(metrics.branches)) {
    const selected = rows.filter((row) => row.branch === branch);
    assert.equal(selected.length, 128);
    for (const field of ['exact_match', 'token_f1']) {
      const mean = selected.reduce((total, row) => total + row[field], 0) / selected.length;
      assert(Math.abs(mean - report.custom_squad_qa[field]) < 1e-12);
    }
    assert.equal(report.custom_squad_qa.exact_match, 0);
    assert(report.custom_squad_qa.token_f1 < 0.002);
  }
});

test('C02 route explains its own task and failures and keeps full capstone planned', () => {
  const page = readFileSync('dist/project-plans/causal-model-300m.html', 'utf8');
  assert(page.includes('Full capstone planned'));
  assert(page.includes('causal-model-300m-pilot.zip'));
  assert(page.includes('All six branches failed the frozen QA quality floors'));
  assert(!page.includes('bounded SciFact CPU workflow'));
  assert(page.includes('Read the measured figures'));
  assert(page.includes('equal horizontal position is not equal cumulative training exposure'));
  assert(page.includes('c02-byte-rates.svg'));
  const catalog = readFileSync('dist/projects.html', 'utf8');
  assert(catalog.includes('causal-model-300m-pilot.zip'));
});
