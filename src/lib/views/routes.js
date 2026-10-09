import { careerCoverage, pathwayCoverage } from '../careers.js';
import { escapeHtml } from '../html.js';
import { lessonLink } from '../urls.js';

/** @param {import('../../types/course').Course} course */
export function routeViewContext(course) {
  const phaseById = (/** @type {string} */ id) =>
    course.phases.find((phase) => phase.id === id) || course.phases[0];
  const routeName = (/** @type {string} */ id) =>
    course.pathways.find((r) => r.id === id)?.title || id;
  const sources = (/** @type {{ id: string }} */ route) =>
    `https://github.com/mlnomadpy/jaxpathways/blob/main/learning-paths/${route.id}.json`;
  const availability = (/** @type {string[]} */ ids) => careerCoverage(ids, course);
  const availabilityText = (/** @type {string[]} */ ids) => {
    const a = availability(ids);
    return `${a.available} available lesson${a.available === 1 ? '' : 's'}${a.planned ? ` · ${a.planned} unwritten` : ''} · ${a.projects.length} staged project${a.projects.length === 1 ? '' : 's'}`;
  };
  const trackLinks = (/** @type {string[]} */ ids = []) =>
    ids.length
      ? `<p>Apply this work in a guided project plan:</p>
    <div class="actions">${ids.map((id) => `<a class="button" href="tracks/${encodeURIComponent(id)}.html">${escapeHtml(id)} guide</a>`).join('')}</div>`
      : '';
  const engineeringLinks = (/** @type {import('../../types/course').Pathway} */ route) => {
    const ids = route.engineeringLessonIds || [];
    if (!ids.length) return '';
    const lessons = course.phases.flatMap((phase) => phase.lessons);
    return `<h4>Engineering skills to connect</h4>
    <p>Follow the listed lesson prerequisites when extending your core route.</p>
    <ul>${ids
      .map(
        (id) => `<li>
    <a href="${lessonLink(id, route.id)}">${escapeHtml(lessons.find((lesson) => lesson.id === id)?.title || id)}</a>
    </li>`,
      )
      .join('')}</ul>
    <p>
    <a href="project-guides/engineering-release.html">Build a tracked and containerized release</a>
    </p>`;
  };
  const focusSummary = (/** @type {import('../../types/course').Pathway} */ route) => {
    const { focus, preparation } = pathwayCoverage(route, course);
    return `<p class="route-availability">Focus phases: ${focus.available} available lessons${focus.planned ? ` · ${focus.planned} planned` : ''}. Preparation: ${preparation.available} available${preparation.planned ? ` · ${preparation.planned} planned` : ''}.</p>`;
  };
  return {
    phaseById,
    routeName,
    sources,
    availability,
    availabilityText,
    trackLinks,
    engineeringLinks,
    focusSummary,
  };
}
