import { pathwayCoverage } from '../lib/careers.js';
import { $ } from '../lib/dom.js';
import { authoredIds, courseState, phaseById, lessonById } from '../lib/course-state.js';
import { inlineMath } from '../lib/math.js';
import { practiceKind, courseTypeCue } from '../lib/runtime.js';
import { searchLesson } from '../lib/search.js';
import { escapeHtml } from '../lib/html.js';
import { lessonLink } from '../lib/urls.js';

function progress() {
  if (!$('#progress')) return;
  const ids = authoredIds(),
    entries = ids.map((id) => courseState.lessonProgress.lessons[id] || {});
  $('#progress').hidden = !entries.some((e) => e.checkpointPassed || e.evidenceSaved);
  $('#progress').textContent =
    `${entries.filter((e) => e.checkpointPassed).length}/${ids.length} checkpoints · ${entries.filter((e) => e.evidenceSaved).length}/${ids.length} exercises self-reported`;
}

function syncCourseFilterButtons(activeGroup = 'all') {
  document.querySelectorAll('[data-course-filter]').forEach((btn) => {
    btn.setAttribute('aria-pressed', String(btn.dataset.courseFilter === activeGroup));
  });
}

function updateCatalogScrollEra() {
  const scrollBar = $('#course-scroll-bar');
  const indicator = $('#active-era-indicator');
  if (!scrollBar || !indicator) return;
  const visibleCards = Array.from(document.querySelectorAll('.phase-card:not([hidden])'));
  if (!visibleCards.length) return;
  const currentCard =
    visibleCards.filter((card) => card.getBoundingClientRect().top <= 220).at(-1) ||
    visibleCards[0];
  const eraId = currentCard.dataset.courseEra || 'era-onboarding';
  const eraTitle = currentCard.dataset.eraTitle || 'Start Here · Setup & Cloud TPU Track';
  scrollBar.setAttribute('data-active-era', eraId);
  indicator.textContent = `${eraTitle} · ${currentCard.querySelector('.phase-title')?.textContent || ''}`;
}

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
  if (!phase.lessons.some((l) => l.status === 'authored')) $('#show-roadmap').checked = true;
  filter('', false);
  const card = document.getElementById('phase-' + id);
  card.open = true;
  const url = new URL(location.href);
  url.searchParams.set('phase', id);
  for (const key of ['q', 'hardware']) url.searchParams.delete(key);
  if (courseState.activePath === 'all') url.searchParams.delete('path');
  if ($('#show-roadmap').checked) url.searchParams.set('roadmap', '1');
  if (url.href !== location.href) history.pushState(null, '', url);
  card.scrollIntoView({
    block: 'start',
    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
  });
}

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
  if (!$('#phases')) return;
  courseState.courseQuery = query;
  const activeTypeFilter = courseState.courseTypeFilter || 'all';
  const route = courseState.curriculum.find((r) => r.id === courseState.activePath);
  const showRoadmap = $('#show-roadmap').checked;
  let visible = 0,
    visibleLessons = 0;
  const visibleByEra = new Map();
  for (const phase of courseState.course.phases) {
    const card = document.getElementById('phase-' + phase.id);
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
      row.hidden = !match;
      const excerpt = row.querySelector('.search-excerpt');
      if (excerpt) {
        const text = query && match ? searchLesson(lesson, query).excerpt : '';
        const link = excerpt.querySelector('a');
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
  document.querySelectorAll('[data-era-banner]').forEach((banner) => {
    const eraId = banner.getAttribute('data-era-banner');
    banner.hidden = !visibleByEra.get(eraId);
  });
  $('#empty').hidden = visible !== 0;
  $('#empty').innerHTML =
    'No available lessons in this view. <a href="course.html">Browse all available lessons</a> or show planned topics to inspect this path’s roadmap.';
  const summary = $('#path-summary');
  summary.hidden = !route;
  if (route) {
    const coverage = pathwayCoverage(route, courseState.course);
    summary.innerHTML = `<p class="edition">Your selected view</p><h2>${escapeHtml(route.title)}</h2><p>${inlineMath(route.description)}</p><p>Focus phases: ${coverage.focus.available} available${coverage.focus.planned ? ` · ${coverage.focus.planned} planned` : ''}. Preparation: ${coverage.preparation.available} available${coverage.preparation.planned ? ` · ${coverage.preparation.planned} planned` : ''}.</p><div class="actions"><a class="button" href="course.html">All course steps</a><a href="pathways.html?path=${encodeURIComponent(route.id)}">View path details</a></div>`;
  }
  const visiblePhases = courseState.course.phases.filter(
    (p) => !document.getElementById('phase-' + p.id).hidden,
  );
  $('#course-jumps').innerHTML = visiblePhases
    .map((p) => `<button type="button" data-jump-phase="${p.id}">${escapeHtml(p.title)}</button>`)
    .join('');
  $('#course-jumps')
    .querySelectorAll('button')
    .forEach((b) => (b.onclick = () => openPhase(b.dataset.jumpPhase)));
  if (updateUrl) serializeCourseView();
  $('#course-view-status').textContent =
    `${visibleLessons} ${showRoadmap ? 'topics' : 'available lesson drafts'} in ${visible} course steps${route ? ' on this path' : ''}.`;
  if (query || courseState.courseHardware !== 'all')
    $('#course-view-status').insertAdjacentHTML(
      'beforeend',
      ' Legacy linked view. <a href="course.html">See the full course</a>',
    );
  updateCatalogScrollEra();
}

function serializeCourseView() {
  const url = new URL(location.href);
  const state = {
    path: courseState.activePath === 'all' ? '' : courseState.activePath,
    q: courseState.courseQuery,
    hardware: courseState.courseHardware === 'all' ? '' : courseState.courseHardware,
    roadmap: $('#show-roadmap').checked ? '1' : '',
  };
  for (const [key, value] of Object.entries(state)) {
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  }
  if (url.href !== location.href) history.pushState(null, '', url);
}

function restoreCourseView() {
  if (!$('#phases') || !courseState.course) return;
  const params = new URLSearchParams(location.search);
  courseState.activePath = courseState.curriculum.some((r) => r.id === params.get('path'))
    ? params.get('path')
    : 'all';
  courseState.courseQuery = params.get('q') || '';
  courseState.courseHardware = ['cpu', 'virtual-cpu'].includes(params.get('hardware'))
    ? params.get('hardware')
    : 'all';
  courseState.courseTypeFilter = 'all';
  syncCourseFilterButtons('all');
  const requestedPhase = phaseById(params.get('phase'));
  $('#show-roadmap').checked =
    params.get('roadmap') === '1' ||
    Boolean(requestedPhase && !requestedPhase.lessons.some((l) => l.status === 'authored'));
  filter(courseState.courseQuery, false);
  const phase = params.get('phase'),
    card = phase ? document.getElementById('phase-' + phase) : null;
  if (card && !card.hidden) card.open = true;
}

function initCatalog() {
  if ($('#phases')) {
    const total = courseState.course.phases.reduce((n, p) => n + p.lessons.length, 0);
    const plannedCount = total - authoredIds().length;
    $('#counts').textContent =
      `${courseState.course.phases.length} phases · ${authoredIds().length} lessons available${plannedCount ? ` · ${plannedCount} planned` : ''}`;
    document.querySelectorAll('[data-phase]').forEach(
      (b) =>
        (b.onclick = (event) => {
          event.preventDefault();
          openPhase(b.dataset.phase);
        }),
    );
    document.querySelectorAll('[data-course-filter]').forEach(
      (btn) =>
        (btn.onclick = () => {
          courseState.courseTypeFilter = btn.dataset.courseFilter || 'all';
          syncCourseFilterButtons(courseState.courseTypeFilter);
          filter(courseState.courseQuery, false);
        }),
    );
    document.querySelectorAll('[data-lesson]').forEach(
      (b) =>
        (b.onclick = (event) => {
          event.preventDefault();
          const found = lessonById(b.dataset.lesson);
          location.href = lessonLink(found.lesson.id, courseState.activePath);
        }),
    );
    $('#show-roadmap').onchange = () => filter(courseState.courseQuery);
    document.querySelectorAll('.phase-card').forEach((card) =>
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
    if (courseState.curriculum.some((r) => r.id === params.get('path'))) {
      courseState.activePath = params.get('path');
    }
    restoreCourseView();
    progress();
    const last = lessonById(courseState.readingState.lastLessonId),
      selected = courseState.curriculum.find((r) => r.id === courseState.activePath);
    const canResume =
      last?.lesson.status === 'authored' &&
      (!selected || selected.phaseIds.includes(last.phase.id));
    const first = (
      selected ? selected.lessons : courseState.course.phases.flatMap((p) => p.lessons)
    ).find((l) => l.status === 'authored');
    if (canResume) {
      const path = courseState.curriculum.some((r) => r.id === courseState.readingState.path)
          ? courseState.readingState.path
          : 'all',
        section = courseState.readingState.lessons[last.lesson.id]?.sectionId || '';
      $('#course-start').textContent = 'Resume lesson';
      $('#course-start').setAttribute('aria-label', 'Resume ' + last.lesson.title);
      $('#course-start').href =
        `${lessonLink(last.lesson.id, path)}${/^section-\d+$/.test(section) ? '#' + section : ''}`;
    } else if (first) $('#course-start').href = lessonLink(first.id, courseState.activePath);
    else $('#course-start').hidden = true;
    const initialPhase = params.get('phase')
      ? document.getElementById('phase-' + params.get('phase'))
      : null;
    if (!initialPhase) {
      const first = document.querySelector('.phase-card:not([hidden])');
      if (first) first.open = true;
    }
    if (initialPhase && !initialPhase.hidden)
      initialPhase.scrollIntoView({ block: 'start', behavior: 'instant' });
    if (lessonById(params.get('lesson'))) {
      const found = lessonById(params.get('lesson'));
      location.href = lessonLink(found.lesson.id, courseState.activePath);
    }
    if ($('#content-status'))
      $('#content-status').textContent =
        `${authoredIds().length} lesson drafts with CPU companions${plannedCount ? `; ${plannedCount} topics have no lesson material` : ''}. ${courseState.course.projectIds?.length || 1} staged projects are available. Project synthesis and math assessment review drafts are linked from their phases; remaining route assessments are planned.`;
  }
}

export {
  progress,
  openPhase,
  openRoute,
  filter,
  serializeCourseView,
  restoreCourseView,
  initCatalog,
};
