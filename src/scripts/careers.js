import { courseState, phaseById } from '../lib/course-state.js';
import { careerStartingPhase, careerPlanMarkdown } from '../lib/careers.js';
import { $ } from '../lib/dom.js';
import { escapeHtml } from '../lib/html.js';
import { renderCareer } from '../lib/views/career.js';
import { downloadText } from './browser-actions.js';
import { navigateToCourse } from './course-navigation.js';

export function initCareers() {
  if (!$('#role-detail')) return;
  if ($('#career-links'))
    $('#career-links').innerHTML = courseState.course.roles
      .map((r) => `<button data-role="${r.id}">${escapeHtml(r.title)}</button>`)
      .join('');
  function showRole(id) {
    const role = courseState.course.roles.find((r) => r.id === id);
    if (!role) return;
    const detail = $('#role-detail');
    detail.hidden = false;
    if ($('#role-cards')) document.body.classList.add('career-open');
    document
      .querySelectorAll('#career-links [data-role],#role-cards [data-role]')
      .forEach((button) =>
        button.setAttribute('aria-expanded', String(button.dataset.role === id)),
      );
    detail.innerHTML = renderCareer(role, courseState.course);
    if ($('#domain-detail')) $('#domain-detail').hidden = true;
    const roleUrl = new URL(location.href);
    roleUrl.searchParams.set('role', role.id);
    if (roleUrl.href !== location.href) history.pushState(null, '', roleUrl);
    const heading = detail.querySelector('h3');
    heading.tabIndex = -1;
    heading.focus({ preventScroll: true });
    let known = [];
    try {
      const saved = JSON.parse(localStorage.getItem('jaxpathways-career-plan-v1'));
      if (saved?.roleId === role.id && Array.isArray(saved.knownPhaseIds))
        known = saved.knownPhaseIds;
    } catch {}
    detail
      .querySelectorAll('.starting-checks input')
      .forEach((input) => (input.checked = known.includes(input.value)));
    const selectedPhase = () =>
      careerStartingPhase(
        role,
        [...detail.querySelectorAll('.starting-checks input:checked')].map((input) => input.value),
        courseState.course,
      );
    const updateStart = () => {
      const phase = phaseById(selectedPhase()),
        lesson = phase.lessons.find((l) => l.status === 'authored');
      $('#career-start-note').textContent = lesson
        ? 'Suggested first lesson: ' +
          lesson.title +
          '. Earlier prerequisites remain available for review.'
        : 'This starting phase is unwritten. View its roadmap and choose available preparation.';
      $('#career-start').textContent = lesson
        ? 'Start: ' + lesson.title
        : 'View this planned milestone';
    };
    detail
      .querySelectorAll('.starting-checks input')
      .forEach((input) => (input.onchange = updateStart));
    updateStart();
    $('#close-career').onclick = () => {
      detail.hidden = true;
      document.body.classList.remove('career-open');
      const url = new URL(location.href);
      url.searchParams.delete('role');
      history.pushState(null, '', url);
      if ($('#career-browser')?.tagName === 'DETAILS') $('#career-browser').open = true;
      document
        .querySelectorAll('[data-role]')
        .forEach((button) => button.setAttribute('aria-expanded', 'false'));
      const origin = document.querySelector(`[data-role="${role.id}"]`);
      if (origin) origin.focus();
    };
    $('#career-start').onclick = () => {
      const phase = phaseById(selectedPhase());
      const lesson = phase.lessons.find((l) => l.status === 'authored');
      if (lesson) {
        location.href = `lesson.html?path=${encodeURIComponent(role.defaultPathwayId)}&lesson=${encodeURIComponent(lesson.id)}`;
      } else {
        location.href = `course.html?path=${encodeURIComponent(role.defaultPathwayId)}&phase=${encodeURIComponent(phase.id)}&roadmap=1#curriculum`;
      }
    };
    $('#career-save').onclick = () => {
      try {
        localStorage.setItem(
          'jaxpathways-career-plan-v1',
          JSON.stringify({
            version: 1,
            roleId: role.id,
            pathwayId: role.defaultPathwayId,
            startPhaseId: selectedPhase(),
            knownPhaseIds: [...detail.querySelectorAll('.starting-checks input:checked')].map(
              (input) => input.value,
            ),
          }),
        );
        const raw = localStorage.getItem('jaxpathways-notebook-v1');
        const notebook = raw ? JSON.parse(raw) : { version: 1, evidence: [] };
        notebook.route = role.defaultPathwayId;
        localStorage.setItem('jaxpathways-notebook-v1', JSON.stringify(notebook));
        $('#career-status').textContent =
          'Plan saved on this device. Find it in My learning notebook.';
      } catch {
        $('#career-status').textContent =
          'Device storage is unavailable. Download your plan to keep it.';
      }
    };
    $('#career-download').onclick = () => {
      downloadText(
        role.id + '-learning-plan.md',
        careerPlanMarkdown(role, courseState.course, selectedPhase()),
        'text/markdown',
      );
    };
    $('#career-share').onclick = async () => {
      const url = new URL('careers.html', location.href);
      url.searchParams.set('role', role.id);
      url.hash = 'roles';
      try {
        await navigator.clipboard.writeText(url.href);
        $('#career-status').textContent = 'Career route link copied.';
      } catch {
        $('#career-status').textContent = url.href;
      }
    };
    detail
      .querySelectorAll('[data-choice]')
      .forEach((b) => (b.onclick = () => navigateToCourse({ path: b.dataset.choice })));
    detail.querySelectorAll('[data-phase]').forEach(
      (b) =>
        (b.onclick = () => {
          location.href = `course.html?path=${encodeURIComponent(role.defaultPathwayId)}&phase=${encodeURIComponent(b.dataset.phase)}#curriculum`;
        }),
    );
    detail.scrollIntoView({
      behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
      block: 'start',
    });
  }
  document
    .querySelectorAll('[data-role]')
    .forEach((b) => (b.onclick = () => showRole(b.dataset.role)));
  window.addEventListener('popstate', () => {
    const id = new URLSearchParams(location.search).get('role');
    if (id) showRole(id);
    else {
      $('#role-detail').hidden = true;
      document.body.classList.remove('career-open');
    }
  });
  const selected = new URLSearchParams(location.search).get('role');
  if (selected && !new URLSearchParams(location.search).has('phase')) showRole(selected);
}
