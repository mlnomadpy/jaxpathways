import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pathwayCoverage, careerPlanMarkdown } from '../src/lib/careers.js';
import { assessmentContent } from '../src/lib/server/assessment-content.js';
import { projectGuide } from '../src/lib/server/project-guide.js';
const course = JSON.parse(readFileSync(new URL('../public/curriculum.json', import.meta.url)));

test('shared preparation cannot inflate specialist availability', () => {
  const fixture = {phases: [
    {id:'foundation',number:'01',lessons:[{status:'authored'},{status:'authored'}]},
    {id:'focus',number:'10',lessons:[{status:'planned'}]},
  ]};
  const route = {phaseIds:['foundation','focus'],focusPhaseIds:['focus']};
  const coverage = pathwayCoverage(route,fixture);
  assert.equal(coverage.focus.available,0);
  assert.equal(coverage.focus.planned,1);
  assert.equal(coverage.preparation.available,2);
  fixture.phases[1].lessons[0].status='authored';
  const updated=pathwayCoverage(route,fixture);
  assert.equal(updated.focus.available,1);
  assert.equal(updated.focus.planned,0);
  assert.equal(updated.preparation.available,2);
});

test('career downloads retain work scenarios and distinguish focus from preparation', () => {
  for (const role of course.roles) {
    const plan = careerPlanMarkdown(role, course, 'welcome');
    assert(plan.includes(role.workExample));
    for (const question of role.reviewQuestions) assert(plan.includes(question));
    const route=course.pathways.find(route=>route.id===role.defaultPathwayId);
    for (const lessonId of route.engineeringLessonIds || []) assert(plan.includes(lessonId));
    for (const item of route.capstone.evidence) assert(plan.includes(item));
    assert(plan.includes('Focus phases:'));
    assert(plan.includes('Preparation:'));
    for(const id of role.modalityTrackIds){
      const track=course.modalityTracks.find(track=>track.id===id);
      assert(plan.includes(track.harnessProjectId ? 'runnable harness: '+track.harnessProjectId : 'connected harness planned'));
    }
  }
});

test('math assessment renders explicit formulas as KaTeX and preserves code literals', () => {
  const content = assessmentContent('assessments/math.md');
  assert(content.body.includes('class="katex"'));
  assert(content.body.includes('annotation encoding="application/x-tex"'));
  assert(content.body.includes('X = [[1., 0.]'));
  assert(!content.body.includes('\\('));
  assert.equal(content.sections.filter(section=>section.label.startsWith('Task')).length,4);
});


test('assessment display equations retain TeX in accessible MathML', () => {
  const content = assessmentContent('assessments/probability.md');
  assert(content.body.includes('class="math-display"'));
  assert(content.body.includes('annotation encoding="application/x-tex"'));
  assert(!content.body.includes(String.raw`\[`));
  assert(!content.body.includes(String.raw`\]`));
});

test('project guides render real figures, math and course links outside the download', () => {
  const projects=JSON.parse(readFileSync('public/api/v1/projects.json')).projects;
  for(const project of projects){
    const content=projectGuide(project,course);
    assert(content.heading.includes('<h1'));
    assert(content.sections.length,project.id+': no teaching sections');
  }
  const optimizer=projectGuide(projects.find(p=>p.id==='optimizer-audit'),course);
  assert(optimizer.body.includes('class="math-display"'));
  assert(optimizer.body.includes('/project-assets/optimizer-audit/outputs/conditioning.png'));
  assert(optimizer.body.includes('/project-guides/regression-audit.html'));
  assert(optimizer.body.includes('/course.html#phase-optimization'));
  assert(optimizer.body.includes('/assessments/foundations.html'));
  const audio=projectGuide(projects.find(p=>p.id==='audio-harness'),course);
  assert(audio.body.includes('/project-assets/audio-harness/outputs/error-example.wav'));
});

test('planned capstones have valid preparation and stay outside runnable availability', () => {
  const plans = JSON.parse(readFileSync('curriculum/project-plans.json'));
  const projects = JSON.parse(readFileSync('public/api/v1/projects.json')).projects;
  const lessonIds = new Set(course.phases.flatMap(phase => phase.lessons.map(lesson => lesson.id)));
  const trackIds = new Set(course.modalityTracks.map(track => track.id));
  const ids = plans.map(plan => plan.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const plan of plans) {
    assert.equal(plan.status, 'planned');
    assert(!projects.some(project => project.id === plan.id), `${plan.id}: plan counted as runnable`);
    assert(projects.some(project => project.id === plan.harnessId));
    for (const id of plan.lessonIds) assert(lessonIds.has(id), `${plan.id}: unknown lesson ${id}`);
    for (const id of plan.trackIds) assert(trackIds.has(id));
    const content = projectGuide(plan, course, ids);
    assert(content.body.includes('class="katex"'), `${plan.id}: math not rendered`);
    assert(content.body.includes('project-plans/'), `${plan.id}: missing sibling plan navigation`);
    assert(!content.body.includes('href="../'), `${plan.id}: unresolved source link`);
  }
});
