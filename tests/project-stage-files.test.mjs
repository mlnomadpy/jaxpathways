import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { Window } from 'happy-dom';
import { parseProjects } from '../src/lib/manifests.js';
import { projectFileLayout, renderProject } from '../src/lib/views/project.js';

function fixture() {
  return structuredClone(JSON.parse(fs.readFileSync('public/api/v1/projects.json')).projects[0]);
}

test('a later project stage names its own edit file and verifier', async () => {
  const project = fixture();
  const file = `projects/${project.id}/starter/sequence_ppo.py`;
  const stage = {
    ...project.stages[0],
    id: 'extension',
    workspaceFile: file,
    command: `python3 projects/${project.id}/tests/check_sequence_ppo.py --implementation starter`,
    starterSnippet: 'def sequence_loss():\n    raise NotImplementedError',
  };
  project.stages.push(stage);
  const window = new Window();
  try {
    window.document.body.innerHTML = renderProject(project);
    const extension = window.document.querySelector('#stage-extension');
    assert.equal(extension.querySelector('p strong code').textContent, file);
    assert(extension.querySelector('.stage-code-header').textContent.includes(file));
    assert.equal(extension.querySelector('.project-command pre').textContent, stage.command);
    assert.equal(
      window.document.querySelector(`#stage-${project.stages[0].id} p strong code`).textContent,
      project.workspaceFile,
    );
    assert(window.document.querySelector('.code-target-callout').textContent.includes(file));
    assert(
      window.document
        .querySelector('.project-tree')
        .textContent.includes('starter/sequence_ppo.py'),
    );
    assert(!window.document.querySelector('.project-tree').textContent.includes('--stage 1 ..'));
    assert.deepEqual(projectFileLayout(project).additionalFiles, [file]);
  } finally {
    await window.happyDOM.abort();
  }
});

test('stage edit files reject malformed values while old manifests remain valid', () => {
  const project = fixture();
  assert.equal(parseProjects({ projects: [project] }).length, 1);
  for (const invalid of [null, false, 12, '', '   ', []]) {
    const changed = structuredClone(project);
    changed.stages[0].workspaceFile = invalid;
    assert.throws(() => parseProjects({ projects: [changed] }), /registry/);
  }
  project.stages[0].workspaceFile = `projects/${project.id}/starter/extra.py`;
  assert.equal(
    parseProjects({ projects: [project] })[0].stages[0].workspaceFile,
    project.stages[0].workspaceFile,
  );
});

test('the GAE teaching figure retains its independent EOS and timeout calculation', () => {
  const result = spawnSync(
    'python3',
    ['projects/posttraining-audit/figures/render_gae.py', '--check'],
    { encoding: 'utf8' },
  );
  assert.equal(result.status, 0, result.stderr);
  const { results } = JSON.parse(result.stdout);
  assert.deepEqual(
    results.map((row) => row.bootstrap),
    [0, 0.8],
  );
  assert.deepEqual(
    results[0].advantages.map((value) => Number(value.toFixed(6))),
    [1.148, 1.4],
  );
  assert.deepEqual(
    results[1].advantages.map((value) => Number(value.toFixed(6))),
    [0.810448, 1.0284, 1.22],
  );
});

test('the guide-linked plotting source is published byte-for-byte', () => {
  assert.deepEqual(
    fs.readFileSync('public/project-assets/posttraining-audit/figures/render_gae.py'),
    fs.readFileSync('projects/posttraining-audit/figures/render_gae.py'),
  );
});
