import { courseState, phaseById } from '../lib/course-state.js';
import { careerStartingPhase, careerPlanMarkdown } from '../lib/careers.js';
import { $ } from '../lib/dom.js';
import { escapeHtml } from '../lib/html.js';
import { lessonLink } from '../lib/urls.js';
import { renderCareer } from '../lib/views/career.js';
import { downloadText } from './browser-actions.js';
import { navigateToCourse } from './course-navigation.js';

const initializedDocuments = new WeakSet();

export function initCareers() {
  const detail = /** @type {HTMLElement | null} */ (document.querySelector('#role-detail'));
  const course = courseState.course;
  if (!detail || !course) return;
  const panel = /** @type {HTMLElement} */ (detail);
  const data = /** @type {import('../types/course').Course} */ (course);
  if (initializedDocuments.has(document)) return;
  initializedDocuments.add(document);
  const links = /** @type {HTMLElement | null} */ (document.querySelector('#career-links'));
  if (links)
    links.innerHTML = data.roles
      .map((r) => `<button data-role="${r.id}">${escapeHtml(r.title)}</button>`)
      .join('');
  /** @param {string} selector */
  const required = (selector) => {
    const element = /** @type {HTMLElement | null} */ (document.querySelector(selector));
    if (!element) throw new Error(`Missing careers control: ${selector}`);
    return element;
  };
  /** @param {string} selector @param {string} value */
  const setText = (selector, value) => {
    required(selector).textContent = value;
  };
  /** @param {string} selector @param {(event: MouseEvent) => void} handler */
  const onClick = (selector, handler) => {
    required(selector).onclick = handler;
  };
  /** @param {string} id */
  function showRole(id) {
    const role = data.roles.find((r) => r.id === id);
    if (!role) return;
    panel.hidden = false;
    if ($('#role-cards')) document.body.classList.add('career-open');
    document
      .querySelectorAll('#career-links [data-role],#role-cards [data-role]')
      .forEach((button) =>
        button.setAttribute('aria-expanded', String(button.getAttribute('data-role') === id)),
      );
    panel.innerHTML = renderCareer(role, data);
    const domainDetail = /** @type {HTMLElement | null} */ (
      document.querySelector('#domain-detail')
    );
    if (domainDetail) domainDetail.hidden = true;
    const roleUrl = new URL(location.href);
    roleUrl.searchParams.set('role', role.id);
    if (roleUrl.href !== location.href) history.pushState(null, '', roleUrl);
    const heading = panel.querySelector('h3');
    if (heading) {
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
    }
    let known = [];
    try {
      const saved = JSON.parse(localStorage.getItem('jaxpathways-career-plan-v1') || 'null');
      if (
        saved?.roleId === role.id &&
        Array.isArray(saved.knownPhaseIds) &&
        saved.knownPhaseIds.every((/** @type {unknown} */ id) => typeof id === 'string')
      ) {
        known = saved.knownPhaseIds;
      }
    } catch {}
    /** @type {NodeListOf<HTMLInputElement>} */ (
      panel.querySelectorAll('.starting-checks input')
    ).forEach((input) => (input.checked = known.includes(input.value)));
    const selectedPhase = () =>
      careerStartingPhase(
        role,
        [
          .../** @type {NodeListOf<HTMLInputElement>} */ (
            panel.querySelectorAll('.starting-checks input:checked')
          ),
        ].map((input) => input.value),
        data,
      );
    const updateStart = () => {
      const phase = phaseById(selectedPhase());
      if (!phase) return;
      const lesson = phase.lessons.find((l) => l.status === 'authored');
      setText(
        '#career-start-note',
        lesson
          ? 'Suggested first lesson: ' +
              lesson.title +
              '. Earlier prerequisites remain available for review.'
          : 'This starting phase is unwritten. View its roadmap and choose available preparation.',
      );
      setText('#career-start', lesson ? 'Start: ' + lesson.title : 'View this planned milestone');
    };
    /** @type {NodeListOf<HTMLInputElement>} */ (
      panel.querySelectorAll('.starting-checks input')
    ).forEach((input) => (input.onchange = updateStart));
    updateStart();
    onClick('#close-career', () => {
      panel.hidden = true;
      document.body.classList.remove('career-open');
      const url = new URL(location.href);
      url.searchParams.delete('role');
      history.pushState(null, '', url);
      const browser = /** @type {HTMLDetailsElement | null} */ (
        document.querySelector('#career-browser')
      );
      if (browser) browser.open = true;
      document
        .querySelectorAll('[data-role]')
        .forEach((button) => button.setAttribute('aria-expanded', 'false'));
      const origin = /** @type {HTMLElement | null} */ (
        document.querySelector(`[data-role="${role.id}"]`)
      );
      if (origin) origin.focus();
    });
    onClick('#career-start', () => {
      const phase = phaseById(selectedPhase());
      if (!phase) return;
      const lesson = phase.lessons.find((l) => l.status === 'authored');
      if (lesson) {
        location.href = lessonLink(lesson.id, role.defaultPathwayId);
      } else {
        location.href = `course.html?path=${encodeURIComponent(role.defaultPathwayId)}&phase=${encodeURIComponent(phase.id)}&roadmap=1#curriculum`;
      }
    });
    onClick('#career-save', () => {
      try {
        localStorage.setItem(
          'jaxpathways-career-plan-v1',
          JSON.stringify({
            version: 1,
            roleId: role.id,
            pathwayId: role.defaultPathwayId,
            startPhaseId: selectedPhase(),
            knownPhaseIds: [
              .../** @type {NodeListOf<HTMLInputElement>} */ (
                panel.querySelectorAll('.starting-checks input:checked')
              ),
            ].map((input) => input.value),
          }),
        );
        const raw = localStorage.getItem('jaxpathways-notebook-v1');
        const notebook = raw ? JSON.parse(raw) : { version: 1, evidence: [] };
        notebook.route = role.defaultPathwayId;
        localStorage.setItem('jaxpathways-notebook-v1', JSON.stringify(notebook));
        setText('#career-status', 'Plan saved on this device. Find it in My learning notebook.');
      } catch {
        setText('#career-status', 'Device storage is unavailable. Download your plan to keep it.');
      }
    });
    onClick('#career-download', () => {
      downloadText(
        role.id + '-learning-plan.md',
        careerPlanMarkdown(role, data, selectedPhase()),
        'text/markdown',
      );
    });
    onClick('#career-share', async () => {
      const url = new URL('careers.html', location.href);
      url.searchParams.set('role', role.id);
      url.hash = 'roles';
      try {
        await navigator.clipboard.writeText(url.href);
        setText('#career-status', 'Career route link copied.');
      } catch {
        setText('#career-status', url.href);
      }
    });
    /** @type {NodeListOf<HTMLElement>} */ (panel.querySelectorAll('[data-choice]')).forEach(
      (button) => {
        button.onclick = () =>
          navigateToCourse({ path: button.getAttribute('data-choice') || undefined });
      },
    );
    /** @type {NodeListOf<HTMLElement>} */ (panel.querySelectorAll('[data-phase]')).forEach(
      (button) => {
        button.onclick = () => {
          const phase = button.getAttribute('data-phase');
          if (phase)
            location.href = `course.html?path=${encodeURIComponent(role.defaultPathwayId)}&phase=${encodeURIComponent(phase)}#curriculum`;
        };
      },
    );
    panel.scrollIntoView({
      behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
      block: 'start',
    });
  }
  document.querySelectorAll('[data-role]').forEach((b) => {
    const roleId = b.getAttribute('data-role');
    if (roleId) b.addEventListener('click', () => showRole(roleId));
  });
  window.addEventListener('popstate', () => {
    const id = new URLSearchParams(location.search).get('role');
    if (id) showRole(id);
    else {
      panel.hidden = true;
      document.body.classList.remove('career-open');
    }
  });
  const selected = new URLSearchParams(location.search).get('role');
  if (selected && !new URLSearchParams(location.search).has('phase')) showRole(selected);
}
