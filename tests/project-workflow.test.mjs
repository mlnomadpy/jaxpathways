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
  assert(content.sections.some(section => /Review a comparison/.test(section.label)));
  for (const section of content.sections) assert(content.body.includes(`id="${section.id}"`));
  assert.match(content.body, /href="\/jaxpathways\/downloads\/jax-project-workspace.zip"/);
  assert.match(content.body, /href="\/jaxpathways\/lesson-recovery-06\.html"/);
  assert.match(content.body, /prepare --config configs\/baseline.json --run-id baseline-01/);
});
