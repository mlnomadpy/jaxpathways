import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { projectGuide } from '../src/lib/server/project-guide.js';
import { renderPathway } from '../src/lib/views/pathway.js';

const project = JSON.parse(fs.readFileSync('projects/reproducible-experiments/project.json', 'utf8'));

test('workshop joins existing lessons, projects, pathway guides and typeset teaching', () => {
  const course = JSON.parse(fs.readFileSync('public/curriculum.json', 'utf8'));
  const all = course.phases.flatMap(phase => phase.lessons);
  for (const id of project.requiredLessonIds) assert(all.some(lesson => lesson.id === id && lesson.status === 'authored'));
  for (const id of ['networks', 'recovery']) assert(course.phases.find(p => p.id === id).additionalProjectIds.includes(project.id));
  for (const id of ['models', 'ops']) assert.match(renderPathway(course.pathways.find(p => p.id === id), course, true), /project-guides\/reproducible-experiments.html/);
  const content = projectGuide(project, course);
  assert(content.sections.some(section => /Review handoff/.test(section.label)));
  for (const section of content.sections) assert(content.body.includes(`id="${section.id}"`));
  assert.match(content.body, /class="katex/);
  assert.doesNotMatch(content.body, /katex-error/);
  assert.match(content.body, /project-workflow.html/);
  assert.match(content.body, /project-assets\/reproducible-experiments\/reference\/suite\/report\/loss_curves.png/);
  assert.match(content.body, /fresh retry/);
  assert.match(content.body, /not a confidence interval/);
  assert.match(content.body, /Markdown/);
});

test('portable workshop keeps recorded evidence, starter, numerical source and CLI help intact', () => {
  const result = spawnSync('python3', ['-c', `
from pathlib import Path
from zipfile import ZipFile
import subprocess, sys, tempfile
root=Path('projects/reproducible-experiments')
with ZipFile('public/downloads/reproducible-experiments.zip') as bundle, tempfile.TemporaryDirectory() as tmp:
    for name in ['model.py','loggers.py','solution/workflow.py','starter/workflow.py','requirements.txt','reference/suite/report/RESEARCH_NOTEBOOK.md','reference/suite/report/loss_curves.png','reference/suite/data.npz']:
        assert bundle.read(str(root/name)) == (root/name).read_bytes()
    bundle.extractall(tmp)
    result=subprocess.run([sys.executable,str(Path(tmp)/root/'experiment.py'),'--help'],capture_output=True,text=True)
    assert result.returncode == 0 and 'analyze' in result.stdout
result=subprocess.run([sys.executable,'scripts/build-experiment-workshop.py'],capture_output=True,text=True)
assert result.returncode == 0,result.stdout+result.stderr
`], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stdout + result.stderr);
});
