import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { assessmentContent } from '../src/lib/server/assessment-content.js';

test('project workspace preserves old runs, records changed inputs and rejects invalid configuration', () => {
  const result = spawnSync('python3', ['-m', 'unittest', 'discover', '-s', 'resources/project-workspace/tests', '-v'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
});

test('workflow guide links resolve beneath a deployment base and contain the runnable exercise', () => {
  const content = assessmentContent('content/guides/project-workflow.md', (href) => /^(https?:|#)/.test(href) ? href : '/jaxpathways/' + href);
  assert.equal(content.sections.length, 8);
  assert.match(content.body, /href="\/jaxpathways\/downloads\/jax-project-workspace.zip"/);
  assert.match(content.body, /href="\/jaxpathways\/lesson.html\?lesson=recovery-06"/);
  assert.match(content.body, /prepare --config configs\/baseline.json --run-id baseline-01/);
});
