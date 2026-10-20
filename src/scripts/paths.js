import { courseState, phaseById } from '../lib/course-state.js';
import { $ } from '../lib/dom.js';
import { lessonLink } from '../lib/urls.js';
import { renderDomains } from '../lib/views/domains.js';
import { initCareers } from './careers.js';
import { initPathways } from './pathways.js';

/** Render the domain cards when course data and their page container are available. */
export function setupChoices() {
  const course = courseState.course;
  const domainCards = $('#domain-cards');
  if (course && domainCards) domainCards.innerHTML = renderDomains(course);
  initCareers();
  initPathways();
}

/** @param {string} id */
function startPath(id) {
  const course = courseState.course;
  if (!course) return;
  const route = course.pathways.find((r) => r.id === id);
  if (!route) return;
  const lesson = route.phaseIds
    .flatMap((phaseId) => phaseById(phaseId)?.lessons ?? [])
    .find((l) => l.status === 'authored');
  location.href = lesson
    ? lessonLink(lesson.id, id)
    : `course.html?path=${encodeURIComponent(id)}&roadmap=1`;
}

export { startPath };
