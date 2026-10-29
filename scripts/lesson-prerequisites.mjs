import assert from 'node:assert/strict';

/** Validate declared dependencies, including branches and cross-phase lessons. */
export function validateLessonPrerequisites(phases) {
  const lessons = new Map();
  for (const phase of phases) {
    for (const [index, lesson] of phase.lessons.entries()) {
      assert(!lessons.has(lesson.id), `Duplicate lesson ${lesson.id}`);
      assert(Array.isArray(lesson.prerequisites), `${lesson.id}: prerequisites must be an array`);
      assert.equal(new Set(lesson.prerequisites).size, lesson.prerequisites.length,
        `${lesson.id}: duplicate prerequisite`);
      const earlier = phase.lessons.slice(0, index).map((item) => item.id);
      if (index > 0) assert(lesson.prerequisites.some((id) => earlier.includes(id)),
        `${lesson.id}: declare an earlier preparation lesson in this phase`);
      lessons.set(lesson.id, lesson);
    }
  }
  const visiting = new Set();
  const visited = new Set();
  function visit(id) {
    assert(lessons.has(id), `Unknown lesson prerequisite ${id}`);
    assert(!visiting.has(id), `Lesson prerequisite cycle at ${id}`);
    if (visited.has(id)) return;
    visiting.add(id);
    for (const dependency of lessons.get(id).prerequisites) visit(dependency);
    visiting.delete(id);
    visited.add(id);
  }
  for (const id of lessons.keys()) visit(id);
  return lessons;
}
