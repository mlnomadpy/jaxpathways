import { courseState, phaseById } from '../lib/course-state.js';
import { $ } from '../lib/dom.js';
import { renderPathway } from '../lib/views/pathway.js';

export function initPathways() {
  if (!$('#routes') || !$('#path-detail')) return;
  function showPath() {
    const id = new URLSearchParams(location.search).get('path'),
      route = courseState.course.pathways.find((r) => r.id === id),
      detail = $('#path-detail');
    if (!route) {
      detail.hidden = true;
      document.body.classList.remove('path-open');
      return;
    }
    document.body.classList.add('path-open');
    detail.hidden = false;
    const available = route.phaseIds
      .flatMap((id) => phaseById(id).lessons)
      .filter((l) => l.status === 'authored');
    const next =
      available.find((l) => !courseState.lessonProgress.lessons[l.id]?.evidenceSaved) ||
      available[0];
    detail.innerHTML = renderPathway(route, courseState.course, Boolean(next));
    if (next)
      $('#use-path').onclick = () => {
        try {
          const raw = localStorage.getItem('jaxpathways-notebook-v1');
          const state = raw ? JSON.parse(raw) : { version: 1, evidence: [] };
          state.route = id;
          localStorage.setItem('jaxpathways-notebook-v1', JSON.stringify(state));
          location.href = `lesson.html?path=${encodeURIComponent(id)}&lesson=${encodeURIComponent(next.id)}`;
        } catch {
          $('#path-save-status').innerHTML =
            `This path could not be saved on this browser. <a href="lesson.html?path=${encodeURIComponent(id)}&lesson=${encodeURIComponent(next.id)}">Continue without saving</a>`;
        }
      };
    detail.querySelector('h1').focus({ preventScroll: true });
  }
  showPath();
  window.addEventListener('popstate', showPath);
}
