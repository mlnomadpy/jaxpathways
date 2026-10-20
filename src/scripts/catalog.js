// @ts-check

import { pathwayCoverage } from '../lib/careers.js';
import { $ } from '../lib/dom.js';
import { authoredIds, courseState, phaseById, lessonById } from '../lib/course-state.js';
import { inlineMath } from '../lib/math.js';
import { practiceKind, courseTypeCue } from '../lib/runtime.js';
import { searchLesson } from '../lib/search.js';
import { escapeHtml } from '../lib/html.js';
import { lessonLink } from '../lib/urls.js';

/** @param {string} selector @returns {HTMLInputElement | null} */
function inputField(selector) {
  const element = $(selector);
  return element?.tagName === 'INPUT' ? /** @type {HTMLInputElement} */ (element) : null;
}

/** @param {Element | null} element @returns {HTMLDetailsElement | null} */
function detailsElement(element) {
  return element?.tagName === 'DETAILS' ? /** @type {HTMLDetailsElement} */ (element) : null;
}

/** @param {Element | null} element @returns {HTMLAnchorElement | null} */
function anchorElement(element) {
  return element?.tagName === 'A' ? /** @type {HTMLAnchorElement} */ (element) : null;
}

function progress() {
  const status = $('#progress');
  if (!status) return;
  const ids = authoredIds(),
    entries = ids.map((id) => courseState.lessonProgress.lessons[id] || {});
  status.hidden = !entries.some((e) => e.checkpointPassed || e.evidenceSaved);
  status.textContent = `${entries.filter((e) => e.checkpointPassed).length}/${ids.length} checkpoints · ${entries.filter((e) => e.evidenceSaved).length}/${ids.length} exercises completed`;
}

function syncCourseFilterButtons(activeGroup = 'all') {
  /** @type {NodeListOf<HTMLButtonElement>} */ (
    document.querySelectorAll('[data-course-filter]')
  ).forEach((btn) => {
    btn.setAttribute('aria-pressed', String(btn.dataset.courseFilter === activeGroup));
  });
}

function updateCatalogScrollEra() {
  const scrollBar = $('#course-scroll-bar');
  const indicator = $('#active-era-indicator');
  if (!scrollBar || !indicator) return;
  const visibleCards = Array.from(
    /** @type {NodeListOf<HTMLDetailsElement>} */ (
      document.querySelectorAll('.phase-card:not([hidden])')
    ),
  );
  if (!visibleCards.length) return;
  const currentCard =
    visibleCards.filter((card) => card.getBoundingClientRect().top <= 220).at(-1) ||
    visibleCards[0];
  const eraId = currentCard.dataset.courseEra || 'era-onboarding';
  const eraTitle = currentCard.dataset.eraTitle || 'Start Here · Setup & Cloud TPU Track';
  scrollBar.setAttribute('data-active-era', eraId);
  indicator.textContent = `${eraTitle} · ${currentCard.querySelector('.phase-title')?.textContent || ''}`;
}

/** @param {string} id */
function openPhase(id) {
  if (!$('#curriculum')) {
    location.href = `course.html?phase=${encodeURIComponent(id)}#curriculum`;
    return;
  }
  const phase = phaseById(id);
  if (!phase) return;
  const route = courseState.curriculum.find((r) => r.id === courseState.activePath);
  if (route && !route.phaseIds.includes(id)) courseState.activePath = 'all';
  courseState.courseQuery = '';
  courseState.courseHardware = 'all';
  courseState.courseTypeFilter = 'all';
  syncCourseFilterButtons('all');
  const roadmap = inputField('#show-roadmap');
  if (!phase.lessons.some((l) => l.status === 'authored') && roadmap) roadmap.checked = true;
  filter('', false);
  const card = detailsElement(document.getElementById('phase-' + id));
  if (!card) return;
  const optionalTrack = detailsElement(card.closest('#optional-tpu-track'));
  if (optionalTrack) optionalTrack.open = true;
  card.open = true;
  const url = new URL(location.href);
  url.searchParams.set('phase', id);
  for (const key of ['q', 'hardware']) url.searchParams.delete(key);
  if (courseState.activePath === 'all') url.searchParams.delete('path');
  if (roadmap?.checked) url.searchParams.set('roadmap', '1');
  if (url.href !== location.href) history.pushState(null, '', url);
  card.scrollIntoView({
    block: 'start',
    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
  });
}

/** @param {string} id */
function openRoute(id) {
  if (!courseState.curriculum.some((r) => r.id === id)) return;
  if (!$('#curriculum')) {
    location.href = `course.html?path=${encodeURIComponent(id)}#curriculum`;
    return;
  }
  courseState.activePath = id;
  courseState.courseQuery = '';
  courseState.courseHardware = 'all';
  courseState.courseTypeFilter = 'all';
  syncCourseFilterButtons('all');
  filter('');
}

function filter(query = '', updateUrl = true) {
  const course = courseState.course;
  if (!$('#phases') || !course) return;
  query = query.trim().toLowerCase();
  courseState.courseQuery = query;
  const activeTypeFilter = courseState.courseTypeFilter || 'all';
  const route = courseState.curriculum.find((r) => r.id === courseState.activePath);
  const roadmap = inputField('#show-roadmap');
  const showRoadmap = Boolean(roadmap?.checked);
  let visible = 0,
    visibleLessons = 0;
  const visibleByEra = new Map();
  for (const phase of course.phases) {
    const card = detailsElement(document.getElementById('phase-' + phase.id));
    if (!card) continue;
    const cue = courseTypeCue(phase);
    const routeMatch = !route || route.phaseIds.includes(phase.id);
    const typeMatch = activeTypeFilter === 'all' || cue.filterGroup === activeTypeFilter;
    const phaseMatch = `${phase.title} ${phase.project} ${phase.part} ${cue.badge}`
      .toLowerCase()
      .includes(query);
    let matches = 0;
    for (const lesson of phase.lessons) {
      const match =
        routeMatch &&
        typeMatch &&
        (showRoadmap || lesson.status === 'authored') &&
        (courseState.courseHardware === 'all' ||
          (lesson.status === 'authored' && practiceKind(lesson) === courseState.courseHardware)) &&
        (phaseMatch || searchLesson(lesson, query).matched);
      const row = document.getElementById(lesson.id);
      if (!row) continue;
      row.hidden = !match;
      const excerpt = /** @type {HTMLElement | null} */ (row.querySelector('.search-excerpt'));
      if (excerpt) {
        const text = query && match ? searchLesson(lesson, query).excerpt : '';
        const link = anchorElement(excerpt.querySelector('a'));
        if (!link) continue;
        link.textContent = text;
        const baseHref = lessonLink(lesson.id, courseState.activePath);
        link.href = `${baseHref}${baseHref.includes('?') ? '&' : '?'}find=${encodeURIComponent(query)}`;
        excerpt.hidden = !text;
      }
      matches += Number(match);
      visibleLessons += Number(match);
    }
    card.hidden = matches === 0;
    if (matches > 0) {
      visibleByEra.set(cue.eraId, (visibleByEra.get(cue.eraId) || 0) + 1);
    }
    if (query && matches) card.open = true;
    visible += Number(matches > 0);
  }
  /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll('[data-era-banner]')).forEach(
    (banner) => {
      const eraId = banner.getAttribute('data-era-banner');
      banner.hidden = !visibleByEra.get(eraId);
    },
  );
  const empty = $('#empty');
  if (!empty) return;
  empty.hidden = visible !== 0;
  empty.innerHTML = query
    ? `No lessons match “${escapeHtml(query)}” in this view. Try a shorter term or <button type="button" data-clear-course-search>clear the search</button>.`
    : 'No available lessons in this view. <a href="course.html">Browse all available lessons</a> or show planned topics to inspect this path’s roadmap.';
  const optionalTrack = $('#optional-tpu-track');
  if (optionalTrack) {
    const track = /** @type {HTMLDetailsElement} */ (optionalTrack);
    if (activeTypeFilter === 'onboarding-tpu') track.open = true;
    const hasVisibleTpuPhase = [
      .../** @type {NodeListOf<HTMLDetailsElement>} */ (track.querySelectorAll('.phase-card')),
    ].some((card) => !card.hidden);
    track.hidden = !hasVisibleTpuPhase;
  }
  const searchInput = inputField('#course-search-input');
  if (searchInput && document.activeElement !== searchInput) searchInput.value = query;
  const clearSearch = $('#course-search-clear');
  if (clearSearch) clearSearch.hidden = !query;
  const searchButton = $('#course-search-open');
  if (searchButton)
    searchButton.setAttribute(
      'aria-label',
      query ? `Search lessons; current query: ${query}` : 'Search lessons',
    );
  const clearResults = $('[data-clear-course-search]');
  if (clearResults)
    clearResults.onclick = () => {
      courseState.courseQuery = '';
      const input = inputField('#course-search-input');
      if (input) input.value = '';
      filter('', true);
    };
  const summary = $('#path-summary');
  if (!summary) return;
  summary.hidden = !route;
  if (route) {
    const coverage = pathwayCoverage(route, course);
    summary.innerHTML = `<p class="edition">Your selected view</p><h2>${escapeHtml(route.title)}</h2><p>${inlineMath(route.description)}</p><p>Focus phases: ${coverage.focus.available} available${coverage.focus.planned ? ` · ${coverage.focus.planned} planned` : ''}. Preparation: ${coverage.preparation.available} available${coverage.preparation.planned ? ` · ${coverage.preparation.planned} planned` : ''}.</p><div class="actions"><a class="button" href="course.html">All course steps</a><a href="pathways.html?path=${encodeURIComponent(route.id)}">View path details</a></div>`;
  }
  const visiblePhases = course.phases.filter(
    (p) => !document.getElementById('phase-' + p.id)?.hidden,
  );
  const jumps = $('#course-jumps');
  if (!jumps) return;
  jumps.innerHTML = visiblePhases
    .map((p) => `<button type="button" data-jump-phase="${p.id}">${escapeHtml(p.title)}</button>`)
    .join('');
  /** @type {NodeListOf<HTMLButtonElement>} */ (jumps.querySelectorAll('button')).forEach((b) => {
    const phaseId = b.dataset.jumpPhase;
    if (phaseId) b.onclick = () => openPhase(phaseId);
  });
  if (updateUrl) serializeCourseView();
  const viewStatus = $('#course-view-status');
  if (!viewStatus) return;
  viewStatus.textContent = `${visibleLessons} ${showRoadmap ? 'topics' : 'available lesson drafts'} in ${visible} course steps${route ? ' on this path' : ''}.`;
  if (query || courseState.courseHardware !== 'all')
    viewStatus.insertAdjacentHTML(
      'beforeend',
      ' Legacy linked view. <a href="course.html">See the full course</a>',
    );
  updateCatalogScrollEra();
}

/** @param {'push' | 'replace'} [method] */
function serializeCourseView(method = 'push') {
  const url = new URL(location.href);
  const state = {
    path: courseState.activePath === 'all' ? '' : courseState.activePath,
    q: courseState.courseQuery,
    hardware: courseState.courseHardware === 'all' ? '' : courseState.courseHardware,
    roadmap: inputField('#show-roadmap')?.checked ? '1' : '',
  };
  for (const [key, value] of Object.entries(state)) {
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  }
  if (url.href !== location.href) {
    if (method === 'replace') history.replaceState(null, '', url);
    else history.pushState(null, '', url);
  }
}

function restoreCourseView() {
  const course = courseState.course;
  if (!$('#phases') || !course) return;
  const params = new URLSearchParams(location.search);
  const requestedPath = params.get('path');
  courseState.activePath = courseState.curriculum.some((r) => r.id === requestedPath)
    ? requestedPath || 'all'
    : 'all';
  courseState.courseQuery = params.get('q') || '';
  const hardware = params.get('hardware');
  courseState.courseHardware = ['cpu', 'virtual-cpu'].includes(hardware || '')
    ? hardware || 'all'
    : 'all';
  courseState.courseTypeFilter = 'all';
  syncCourseFilterButtons('all');
  const phaseId = params.get('phase');
  const requestedPhase = phaseId ? phaseById(phaseId) : undefined;
  const roadmap = inputField('#show-roadmap');
  if (roadmap)
    roadmap.checked =
      params.get('roadmap') === '1' ||
      Boolean(requestedPhase && !requestedPhase.lessons.some((l) => l.status === 'authored'));
  filter(courseState.courseQuery, false);
  const searchInput = inputField('#course-search-input');
  if (searchInput) searchInput.value = courseState.courseQuery;
  const phase = params.get('phase'),
    card = phase ? document.getElementById('phase-' + phase) : null;
  const details = detailsElement(card);
  if (details && !details.hidden) {
    const optionalTrack = detailsElement(details.closest('#optional-tpu-track'));
    if (optionalTrack) optionalTrack.open = true;
    details.open = true;
  }
}

function initCatalog() {
  const course = courseState.course;
  if ($('#phases') && course) {
    const total = course.phases.reduce((n, p) => n + p.lessons.length, 0);
    const plannedCount = total - authoredIds().length;
    const counts = $('#counts');
    if (counts)
      counts.textContent = `${course.phases.length} phases · ${authoredIds().length} lessons available${plannedCount ? ` · ${plannedCount} planned` : ''}`;
    /** @type {NodeListOf<HTMLButtonElement>} */ (
      document.querySelectorAll('[data-phase]')
    ).forEach(
      (b) =>
        (b.onclick = (event) => {
          event.preventDefault();
          const phaseId = b.dataset.phase;
          if (phaseId) openPhase(phaseId);
        }),
    );
    /** @type {NodeListOf<HTMLButtonElement>} */ (
      document.querySelectorAll('[data-course-filter]')
    ).forEach(
      (btn) =>
        (btn.onclick = () => {
          courseState.courseTypeFilter = btn.dataset.courseFilter || 'all';
          syncCourseFilterButtons(courseState.courseTypeFilter);
          filter(courseState.courseQuery, false);
        }),
    );
    /** @type {NodeListOf<HTMLAnchorElement>} */ (
      document.querySelectorAll('[data-lesson]')
    ).forEach(
      (b) =>
        (b.onclick = (event) => {
          event.preventDefault();
          const lessonId = b.dataset.lesson;
          const found = lessonId ? lessonById(lessonId) : undefined;
          if (found) location.href = lessonLink(found.lesson.id, courseState.activePath);
        }),
    );
    const roadmap = $('#show-roadmap');
    if (roadmap) roadmap.onchange = () => filter(courseState.courseQuery);
    const searchDialog = $('#course-search-dialog');
    const searchInput = inputField('#course-search-input');
    if (searchDialog?.tagName !== 'DIALOG' || !searchInput) return;
    const dialog = /** @type {HTMLDialogElement} */ (searchDialog);
    let searchApplied = false;
    let searchStartQuery = '';
    const applySearch = () => {
      searchApplied = true;
      filter(searchInput.value);
      dialog.close();
    };
    const searchOpen = $('#course-search-open');
    searchOpen?.addEventListener('click', () => {
      searchApplied = false;
      searchStartQuery = courseState.courseQuery;
      searchInput.value = courseState.courseQuery;
      dialog.showModal();
      searchInput.focus();
    });
    $('#course-search-apply')?.addEventListener('click', applySearch);
    searchInput?.addEventListener('input', () => filter(searchInput.value, false));
    searchInput?.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        applySearch();
      }
    });
    $('#course-search-cancel')?.addEventListener('click', () => dialog.close());
    $('#course-search-clear')?.addEventListener('click', () => {
      searchInput.value = '';
      filter('', true);
    });
    dialog.addEventListener('close', () => {
      if (!searchApplied) {
        courseState.courseQuery = searchStartQuery;
        searchInput.value = searchStartQuery;
        filter(searchStartQuery, false);
      }
      searchApplied = false;
    });
    document.addEventListener('keydown', (event) => {
      if (
        ((event.key === '/' && !event.metaKey && !event.ctrlKey) ||
          (event.key.toLowerCase() === 'k' && (event.metaKey || event.ctrlKey))) &&
        !event.altKey &&
        !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName || '') &&
        !dialog.open
      ) {
        event.preventDefault();
        searchOpen?.click();
      }
    });
    /** @type {NodeListOf<HTMLDetailsElement>} */ (
      document.querySelectorAll('.phase-card')
    ).forEach((card) =>
      card.addEventListener('toggle', () => {
        if (!card.open) return;
        const url = new URL(location.href);
        url.searchParams.set('phase', card.id.slice(6));
        history.replaceState(null, '', url);
      }),
    );
    window.addEventListener(
      'scroll',
      () => {
        requestAnimationFrame(updateCatalogScrollEra);
      },
      { passive: true },
    );
    const params = new URLSearchParams(location.search);
    const requestedPath = params.get('path');
    if (courseState.curriculum.some((r) => r.id === requestedPath)) {
      courseState.activePath = requestedPath || 'all';
    }
    restoreCourseView();
    progress();
    const last = lessonById(courseState.readingState.lastLessonId || ''),
      selected = courseState.curriculum.find((r) => r.id === courseState.activePath);
    const canResume =
      last?.lesson.status === 'authored' &&
      (!selected || selected.phaseIds.includes(last.phase.id));
    const first = (selected ? selected.lessons : course.phases.flatMap((p) => p.lessons)).find(
      (l) => l.status === 'authored',
    );
    const courseStart = $('#course-start');
    if (canResume && last && courseStart) {
      const path = courseState.curriculum.some((r) => r.id === courseState.readingState.path)
          ? courseState.readingState.path
          : 'all',
        section = courseState.readingState.lessons[last.lesson.id]?.sectionId || '';
      courseStart.textContent = 'Resume lesson';
      courseStart.setAttribute('aria-label', 'Resume ' + last.lesson.title);
      const resumeLink = anchorElement(courseStart);
      if (resumeLink)
        resumeLink.href = `${lessonLink(last.lesson.id, path)}${/^section-\d+$/.test(section) ? '#' + section : ''}`;
    } else if (first) {
      const firstLink = anchorElement(courseStart);
      if (firstLink) firstLink.href = lessonLink(first.id, courseState.activePath);
    } else if (courseStart) courseStart.hidden = true;
    const initialPhaseId = params.get('phase');
    const initialPhase = initialPhaseId ? document.getElementById('phase-' + initialPhaseId) : null;
    if (!initialPhase) {
      const first = /** @type {HTMLDetailsElement | null} */ (
        document.querySelector('.phase-card:not([hidden])')
      );
      if (first) first.open = true;
    }
    if (initialPhase && !initialPhase.hidden)
      initialPhase.scrollIntoView({ block: 'start', behavior: 'instant' });
    const requestedLesson = params.get('lesson');
    const found = requestedLesson ? lessonById(requestedLesson) : undefined;
    if (found) {
      location.href = lessonLink(found.lesson.id, courseState.activePath);
    }
  }
}

function initProjectCatalog() {
  const catalog = $('#available');
  if (!catalog) return;
  const buttons = [
    .../** @type {NodeListOf<HTMLButtonElement>} */ (
      catalog.querySelectorAll('[data-project-filter]')
    ),
  ];
  const cards = [
    .../** @type {NodeListOf<HTMLElement>} */ (catalog.querySelectorAll('[data-project-id]')),
  ];
  const sections = [
    .../** @type {NodeListOf<HTMLElement>} */ (
      catalog.querySelectorAll('[data-project-group-section]')
    ),
  ];
  /** @param {string | null} value @returns {string} */
  const selectedGroup = (value) => value || 'all';
  /** @param {string} group @param {boolean} [writeHistory] */
  const apply = (group, writeHistory = false) => {
    const active = sections.some((section) => section.dataset.projectGroupSection === group)
      ? group
      : 'all';
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(button.dataset.projectFilter === active));
    }
    for (const card of cards)
      card.hidden = active !== 'all' && card.dataset.projectGroup !== active;
    for (const section of sections) {
      section.hidden = !section.querySelector('[data-project-id]:not([hidden])');
    }
    const url = new URL(location.href);
    if (active === 'all') url.searchParams.delete('group');
    else url.searchParams.set('group', active);
    if (writeHistory && url.href !== location.href) history.pushState(null, '', url);
  };
  buttons.forEach((button) =>
    button.addEventListener('click', () => apply(button.dataset.projectFilter || 'all', true)),
  );
  window.addEventListener('popstate', () =>
    apply(selectedGroup(new URLSearchParams(location.search).get('group'))),
  );
  apply(selectedGroup(new URLSearchParams(location.search).get('group')));
}

export {
  progress,
  openPhase,
  openRoute,
  filter,
  serializeCourseView,
  restoreCourseView,
  initCatalog,
  initProjectCatalog,
};
