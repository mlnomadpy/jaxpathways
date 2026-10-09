import courseJson from '../../public/curriculum.json';
const course = /** @type {import('../types/course').Course} */ (
  /** @type {unknown} */ (courseJson)
);
export { course };
export const allLessons = course.phases.flatMap((phase) => phase.lessons);
export const authoredLessons = allLessons.filter((lesson) => lesson.status === 'authored');
export const inventory = {
  authored: authoredLessons.length,
  planned: allLessons.filter((lesson) => lesson.status === 'planned').length,
  projects: course.projectIds.length,
};
