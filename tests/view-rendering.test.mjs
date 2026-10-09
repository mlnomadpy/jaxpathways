import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Window } from 'happy-dom';
import { projectCommand, renderProject } from '../src/lib/views/project.js';
import { renderCareer } from '../src/lib/views/career.js';
import { renderPathway } from '../src/lib/views/pathway.js';

test('project views preserve exact code and truly empty feedback regions', async () => {
  const window = new Window();
  try {
    const code = 'python check.py --input "a b"\n# <script> stays literal';
    window.document.body.innerHTML = projectCommand(code, 'Run <example>');
    assert.equal(window.document.querySelector('pre').textContent, code);
    assert.equal(window.document.querySelector('.copy-status').textContent, '');
    assert.equal(window.document.querySelector('script'), null);
    const project = structuredClone(
      JSON.parse(fs.readFileSync('public/api/v1/projects.json', 'utf8')).projects[0],
    );
    project.title = '<img src=x onerror=alert(1)>';
    window.document.body.innerHTML = renderProject(project);
    assert.equal(window.document.querySelector('h1').textContent, project.title);
    assert.equal(window.document.querySelector('img'), null);
    assert(window.document.querySelector('.project-blueprint'));
    assert(
      window.document
        .querySelector('.code-target-callout')
        .textContent.includes(project.workspaceFile),
    );
    assert(
      window.document.querySelector('.project-tree').textContent.includes('PUT YOUR CODE HERE'),
    );
    assert.equal(window.document.querySelectorAll('.project-workflow-strip > li').length, 4);
    assert.equal(window.document.querySelectorAll('[data-stage]').length, project.stages.length);
    assert.equal(
      window.document.querySelectorAll('.stage-code-walkthrough').length,
      project.stages.length,
    );
    assert.equal(
      window.document.querySelectorAll('.stage-starter-code').length,
      project.stages.length,
    );
  } finally {
    await window.happyDOM.abort();
  }
});

test('career and pathway views use explicit inputs without browser state', async () => {
  const window = new Window();
  const course = JSON.parse(fs.readFileSync('public/curriculum.json', 'utf8'));
  try {
    const role = structuredClone(course.roles[0]);
    role.title = '<script>role</script>';
    window.document.body.innerHTML = renderCareer(role, course);
    assert.equal(window.document.querySelector('.career-heading h3').textContent, role.title);
    assert.equal(window.document.querySelector('script'), null);
    assert.equal(
      window.document.querySelectorAll('.career-milestones > li').length,
      role.milestones.length,
    );
    const route = structuredClone(course.pathways[0]);
    route.title = '<script>route</script>';
    window.document.body.innerHTML = renderPathway(route, course, false);
    assert.equal(window.document.querySelector('h1').textContent, route.title);
    assert.equal(window.document.querySelector('#use-path'), null);
    assert.equal(
      window.document.querySelectorAll('.path-journey > li').length,
      route.phaseIds.length,
    );
    assert.equal(window.document.querySelector('script'), null);
  } finally {
    await window.happyDOM.abort();
  }
});
