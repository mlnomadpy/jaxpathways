import test from 'node:test';
import assert from 'node:assert/strict';
import { validateLessonPrerequisites } from '../scripts/lesson-prerequisites.mjs';

const lesson = (id, prerequisites = []) => ({ id, prerequisites });
test('lesson dependencies permit a real cross-phase readiness gate and an independent branch', () => {
  const phases = [
    { lessons: [lesson('rollout'), lesson('policy', ['rollout'])] },
    { lessons: [lesson('sft'), lesson('reward', ['sft']),
      lesson('ppo', ['reward', 'policy']), lesson('dpo', ['reward'])] },
  ];
  const map = validateLessonPrerequisites(phases);
  assert.deepEqual(map.get('ppo').prerequisites, ['reward', 'policy']);
  assert.deepEqual(map.get('dpo').prerequisites, ['reward']);
});
test('unknown, duplicate and cyclic lesson prerequisites fail', () => {
  assert.throws(() => validateLessonPrerequisites([{ lessons: [lesson('a', ['missing'])] }]), /Unknown/);
  assert.throws(() => validateLessonPrerequisites([{ lessons: [lesson('a'), lesson('b', ['a', 'a'])] }]), /duplicate/);
  assert.throws(() => validateLessonPrerequisites([{ lessons: [lesson('a', ['b']), lesson('b', ['a'])] }]), /cycle/);
  assert.throws(() => validateLessonPrerequisites([{ lessons: [lesson('a'), lesson('b')] }]), /earlier preparation/);
});
