import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  normalizeLearnerBackup,
  validateCareerPlan,
  validateLearnerNotebook,
  validateReadingState,
} from '../src/lib/learner-records.js';
import { readStoredJson } from '../src/lib/storage.js';

const course = JSON.parse(readFileSync('public/curriculum.json', 'utf8'));
const context = {
  routeIds: course.pathways.map((route) => route.id),
  lessonIds: course.phases.flatMap((phase) => phase.lessons.map((lesson) => lesson.id)),
  roleIds: course.roles.map((role) => role.id),
  phaseIds: course.phases.map((phase) => phase.id),
};
const validNotebook = { version: 1, route: context.routeIds[0], evidence: [] };
const validBackup = {
  version: 2,
  notebook: validNotebook,
  sampleCompleted: false,
  lessonProgress: { version: 1, lessons: {} },
  projects: {},
  careerPlan: null,
  reading: null,
};

test('malformed portfolio JSON and null evidence records are rejected without changing optional null records', () => {
  const malformedStorageValue = readStoredJson('portfolio', null, {
    getItem: () => 'null',
  });
  assert.equal(validateLearnerNotebook(malformedStorageValue, context), false);
  assert.equal(validateLearnerNotebook({ ...validNotebook, evidence: [null] }, context), false);
  assert.equal(validateReadingState(null, context), true);
  assert.equal(validateCareerPlan(null, context), true);
});

test('backup normalization refuses null portfolio records before import can mutate storage', () => {
  assert.throws(
    () => normalizeLearnerBackup({ ...validBackup, notebook: null }, context),
    /Unsupported learner backup/,
  );
  assert.throws(
    () =>
      normalizeLearnerBackup(
        { ...validBackup, notebook: { ...validNotebook, evidence: [null] } },
        context,
      ),
    /Unsupported learner backup/,
  );
  assert.deepEqual(normalizeLearnerBackup(validBackup, context).notebook, validNotebook);
});
