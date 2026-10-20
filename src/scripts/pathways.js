import { courseState, phaseById } from '../lib/course-state.js';
import { $ } from '../lib/dom.js';
import { lessonLink } from '../lib/urls.js';
import { renderPathway } from '../lib/views/pathway.js';

const initializedDocuments = new WeakSet();

export function initPathways() {
  const routeList = $('#routes');
  const detail = $('#path-detail');
  const course = courseState.course;
  if (!routeList || !detail || !course) return;
  const panel = /** @type {HTMLElement} */ (detail);
  const data = /** @type {import('../types/course').Course} */ (course);
  if (initializedDocuments.has(document)) return;
  initializedDocuments.add(document);
  function showPath() {
    const id = new URLSearchParams(location.search).get('path') || '';
    const route = data.pathways.find((item) => item.id === id);
    if (!route) {
      panel.hidden = true;
      document.body.classList.remove('path-open');
      return;
    }
    document.body.classList.add('path-open');
    panel.hidden = false;
    const available = route.phaseIds
      .map((phaseId) => phaseById(phaseId))
      .filter((phase) => phase !== undefined)
      .flatMap((phase) => phase.lessons)
      .filter((l) => l.status === 'authored');
    const next =
      available.find((l) => !courseState.lessonProgress.lessons[l.id]?.evidenceSaved) ||
      available[0];
    panel.innerHTML = renderPathway(route, data, Boolean(next));
    const usePath = $('#use-path');
    if (next && usePath)
      usePath.onclick = () => {
        try {
          const raw = localStorage.getItem('jaxpathways-notebook-v1');
          const state = raw ? JSON.parse(raw) : { version: 1, evidence: [] };
          state.route = route.id;
          localStorage.setItem('jaxpathways-notebook-v1', JSON.stringify(state));
          location.href = lessonLink(next.id, route.id);
        } catch {
          const status = $('#path-save-status');
          if (status)
            status.innerHTML = `This path could not be saved on this browser. <a href="${lessonLink(next.id, route.id)}">Continue without saving</a>`;
        }
      };
    panel.querySelector('h1')?.focus({ preventScroll: true });
  }
  showPath();
  window.addEventListener('popstate', showPath);
}
