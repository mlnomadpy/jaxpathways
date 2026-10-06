import courseJson from '../../public/curriculum.json';
/** @type {import('../types/course').Course} */
const course = courseJson;
export { course };
export const allLessons = course.phases.flatMap((phase) => phase.lessons);
export const authoredLessons = allLessons.filter((lesson) => lesson.status === 'authored');
export const inventory = {
  authored: authoredLessons.length,
  planned: allLessons.filter((lesson) => lesson.status === 'planned').length,
  projects: course.projectIds.length,
};
