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
  assert.match(content.body, /does not distribute/);
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
