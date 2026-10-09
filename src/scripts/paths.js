import { courseState, phaseById } from '../lib/course-state.js';
import { $ } from '../lib/dom.js';
import { lessonLink } from '../lib/urls.js';
import { renderDomains } from '../lib/views/domains.js';
import { initCareers } from './careers.js';
import { initPathways } from './pathways.js';

export function setupChoices() {
  if ($('#domain-cards')) $('#domain-cards').innerHTML = renderDomains(courseState.course);
  initCareers();
  initPathways();
}

function startPath(id) {
  const route = courseState.course.pathways.find((r) => r.id === id);
  if (!route) return;
  const lesson = route.phaseIds
    .flatMap((id) => phaseById(id).lessons)
    .find((l) => l.status === 'authored');
  location.href = lesson
    ? lessonLink(lesson.id, id)
    : `course.html?path=${encodeURIComponent(id)}&roadmap=1`;
}

export { startPath };
