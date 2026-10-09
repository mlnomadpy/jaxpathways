import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { assessmentContent } from '../src/lib/server/assessment-content.js';
import { renderPathway } from '../src/lib/views/pathway.js';

test('TPU guide is reachable from its route and resolves links under the deployment base', () => {
  const course = JSON.parse(fs.readFileSync('public/curriculum.json', 'utf8'));
  const route = course.pathways.find(p => p.id === 'tpu');
  assert(route.phaseIds.includes('operations'));
  assert.match(renderPathway(route, course, true), /href="tpu-gcp.html"/);
  const content = assessmentContent('content/guides/tpu-gcp.md', href => /^(https?:|#)/.test(href) ? href : '/jaxpathways/' + href);
  assert.match(content.body, /href="\/jaxpathways\/downloads\/jax-tpu-gcp.zip"/);
  assert.match(content.body, /single-device worker/);
});

test('portable TPU workspace contains the exact shared worker and a standalone CLI help entry', () => {
  const result = spawnSync('python3', ['-c', `
from pathlib import Path
from zipfile import ZipFile
import subprocess, sys, tempfile
with ZipFile('public/downloads/jax-tpu-gcp.zip') as bundle, tempfile.TemporaryDirectory() as tmp:
    worker = 'projects/workload-operations/solution/model.py'
    assert bundle.read('jax-tpu-gcp/' + worker) == Path(worker).read_bytes()
    bundle.extractall(tmp)
    root = Path(tmp) / 'jax-tpu-gcp'
    result = subprocess.run([sys.executable, str(root/'resources/tpu-gcp/launch.py'), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--platform' in result.stdout and '--resume' in result.stdout
with ZipFile('public/downloads/jax-course-workspace.zip') as bundle:
    assert 'jaxpathways/resources/tpu-gcp/launch.py' in bundle.namelist()
    assert 'jaxpathways/content/guides/tpu-gcp.md' in bundle.namelist()
`], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
});

test('precision guide exposes measured CPU results, typeset math and deployment-safe artifact links', () => {
  const course = JSON.parse(fs.readFileSync('public/curriculum.json', 'utf8'));
  const route = course.pathways.find(p => p.id === 'tpu');
  assert.match(renderPathway(route, course, true), /href="tpu-performance.html"/);
  const content = assessmentContent('content/guides/tpu-performance.md', href => /^(https?:|#)/.test(href) ? href : '/jaxpathways/' + href);
  assert.equal(content.sections.length, 7);
  assert.match(content.body, /src="\/jaxpathways\/downloads\/tpu-performance\/precision.png"/);
  assert.match(content.body, /class="katex/);
  assert.doesNotMatch(content.body, /katex-error/);
  assert.match(content.body, /not a TPU benchmark/);
  assert.match(content.body, /not runtime peak memory/);
  const report = JSON.parse(fs.readFileSync('resources/tpu-gcp/reference/report.json', 'utf8'));
  assert.equal(report.platform, 'cpu');
  assert.equal(report.results.length, 6);
  const byMode = report.results.filter(row => row.case === 'balanced');
  assert.deepEqual(byMode.map(row => row.stored_operand_bytes), [65536, 32768, 16896]);
});

test('portable precision experiment and figure retain source integrity without importing JAX for help', () => {
  const result = spawnSync('python3', ['-c', `
from pathlib import Path
from zipfile import ZipFile
import hashlib, json, subprocess, sys, tempfile
root = Path('resources/tpu-gcp')
provenance = json.loads((root/'reference/provenance.json').read_text())
for relative, expected in provenance['sha256'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected
with ZipFile('public/downloads/jax-tpu-gcp.zip') as bundle, tempfile.TemporaryDirectory() as tmp:
    prefix = 'jax-tpu-gcp/resources/tpu-gcp/'
    for name in ('precision_profile.py', 'render_precision.py', 'check_precision.py', 'reference/report.json', 'reference/precision.png'):
        assert bundle.read(prefix+name) == (root/name).read_bytes()
    assert 'jax-tpu-gcp/PERFORMANCE.md' in bundle.namelist()
    bundle.extractall(tmp)
    result = subprocess.run([sys.executable, str(Path(tmp)/prefix/'precision_profile.py'), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--platform' in result.stdout and '--trace' in result.stdout
`], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
});
