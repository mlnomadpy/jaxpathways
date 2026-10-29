import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { lessonPreparation } from '../src/lib/views/routes.js';

const read = (file) => JSON.parse(readFileSync(file, 'utf8'));
const source = read('curriculum/course.json');
const routes = source.pathwayIds.map((id) => read(`learning-paths/${id}.json`));
const course = { ...source, pathways: routes };

test('the classifier outcome is reachable without cloud provisioning or RL enrollment', () => {
  const route = routes.find((item) => item.id === 'models');
  assert(!route.phaseIds.some((id) => ['tpu', 'tpu-workflows', 'tpu-systems', 'rl', 'posttraining'].includes(id)));
  const covered = new Set(source.phases.filter((phase) => route.phaseIds.includes(phase.id)).flatMap((phase) => phase.lessons.map((lesson) => lesson.id)));
  for (const id of read(`projects/${route.capstone.projectId}/project.json`).requiredLessonIds) assert(covered.has(id), `missing capstone prerequisite ${id}`);
  const seen = new Set();
  for (const phase of source.phases) {
    for (const id of phase.prerequisitePhaseIds) assert(seen.has(id), `${phase.id}: prerequisite must precede it in the book`);
    seen.add(phase.id);
  }
});

test('optional engineering work exposes its preparation without changing the selected route', () => {
  const before = JSON.stringify(course);
  const prep = lessonPreparation('operations-06', course);
  assert(prep.includes('distributed-02'));
  assert(prep.includes('operations-05'));
  assert(!prep.includes('operations-06'));
  assert(!prep.includes('tpu-01'));
  assert.equal(new Set(prep).size, prep.length);
  assert.equal(JSON.stringify(course), before);
  assert.throws(() => lessonPreparation('missing-lesson', course), /Unknown lesson/);
});

test('every advertised career skill has named lesson, project and review evidence', () => {
  for (const role of source.roles) {
    assert.deepEqual(role.skillEvidence.map((item) => item.skill), role.skills);
    for (const item of role.skillEvidence) {
      assert(item.lessonIds.length && item.projectId && item.reviewCriterion && item.evidence);
      assert(source.projectIds.includes(item.projectId));
    }
  }
  assert.equal(source.roles.find((role) => role.id === 'training-engineer').defaultPathwayId, 'models');
});
