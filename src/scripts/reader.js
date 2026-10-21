import { executionScope } from '../lib/runtime.js';
import { parseValidation } from '../lib/manifests.js';
import { createJsonResource } from '../lib/http.js';

const loadValidation = createJsonResource('validation.json', {
  cache: 'no-store',
  parse: parseValidation,
});

import { initRevisionNotices } from './release-notices.js';
import { inlineMath } from '../lib/math.js';
import {
  courseState,
  readingKey,
  progressKey,
  phaseById,
  lessonById,
} from '../lib/course-state.js';
import { $, requireElement } from '../lib/dom.js';
import { escapeHtml } from '../lib/html.js';
import { lessonLink, legacyLessonUrl } from '../lib/urls.js';

import { bindGradientFigure, bindAttentionMaskFigure } from './lesson-diagrams.js';

import { bindCodeCopy } from './copy.js';

/** @typedef {'Understand' | 'Build' | 'Experiment' | 'Practice' | 'Evidence'} ActivityGroup */

/** @returns {import('../types/course').Course} */
function currentCourse() {
  if (!courseState.course) throw new Error('Course data has not loaded');
  return courseState.course;
}

/** @param {string} id @returns {import('../types/course').Phase} */
function requiredPhase(id) {
  const phase = phaseById(id);
  if (!phase) throw new Error(`Unknown course phase: ${id}`);
  return phase;
}

/** @param {string} id @returns {{ phase: import('../types/course').Phase; lesson: import('../types/course').Lesson }} */
function requiredLesson(id) {
  const found = lessonById(id);
  if (!found) throw new Error(`Unknown course lesson: ${id}`);
  return found;
}

/** @param {string | null | undefined} value @returns {value is ActivityGroup} */
function isActivityGroup(value) {
  return ['Understand', 'Build', 'Experiment', 'Practice', 'Evidence'].includes(value || '');
}

function saveReading(sectionId = '') {
  if (!courseState.standalone) return;
  const lessonId = requireElement('#lesson-body').dataset.lesson;
  if (!lessonId) return;
  if (
    courseState.readingState.lastLessonId === lessonId &&
    courseState.readingState.path === courseState.activePath &&
    courseState.readingState.sectionId === sectionId
  )
    return;
  const updatedAt = new Date().toISOString();
  courseState.readingState = {
    ...courseState.readingState,
    version: 1,
    lastLessonId: lessonId,
    path: courseState.activePath,
    sectionId,
    updatedAt,
    lessons: { ...courseState.readingState.lessons, [lessonId]: { sectionId, updatedAt } },
  };
  try {
    localStorage.setItem(readingKey, JSON.stringify(courseState.readingState));
  } catch {}
}

function saveLessonProgress() {
  try {
    localStorage.setItem(progressKey, JSON.stringify(courseState.lessonProgress));
    return true;
  } catch {
    const status = $('#lesson-save-status');
    if (status)
      status.textContent =
        'Browser storage is unavailable. Keep your downloaded exercise and export a backup before leaving.';
    return false;
  }
}

function closeLesson() {
  location.href = `course.html${courseState.activePath === 'all' ? '' : '?path=' + encodeURIComponent(courseState.activePath)}#curriculum`;
}

/** @param {import('../types/course').Phase} phase @param {import('../types/course').Lesson} lesson @param {{history?: boolean}} [options] */
function showLesson(phase, lesson, options = {}) {
  if (!courseState.standalone) {
    location.replace(legacyLessonUrl(lesson.id, location.href));
    return;
  }
  const staticId = $('#lesson-page')?.dataset.staticLesson;
  const body = requireElement('#lesson-body');
  const renderedTitle = $('#lesson-title');
  const hasMatchingServerRenderedLesson =
    staticId === lesson.id && body.dataset.lesson === lesson.id && Boolean(renderedTitle);
  if (!hasMatchingServerRenderedLesson) {
    const destination = staticId
      ? new URL(lessonLink(lesson.id, courseState.activePath), location.href)
      : new URL(legacyLessonUrl(lesson.id, location.href));
    location.replace(destination.href);
    return;
  }
  const authored = lesson.status === 'authored';
  const allLessons = (
    courseState.curriculum.find((r) => r.id === courseState.activePath)?.phaseIds ||
    currentCourse().phases.map((p) => p.id)
  )
    .flatMap((id) => requiredPhase(id).lessons)
    .filter((l) => (authored ? l.status === 'authored' : true));
  const index = allLessons.findIndex((l) => l.id === lesson.id);
  const lessonUrl = new URL(location.href),
    sameLesson = (staticId || lessonUrl.searchParams.get('lesson')) === lesson.id;
  const requestedHash = sameLesson ? lessonUrl.hash : '';
  const priorSection = courseState.readingState.lessons[lesson.id]?.sectionId;
  const prerequisite =
    lesson.id === 'welcome-01'
      ? 'No terminal or Git experience required'
      : lesson.prerequisites.map((id) => requiredLesson(id).lesson.title).join('; ') ||
        phase.prerequisitePhaseIds.map((id) => requiredPhase(id).title).join('; ') ||
        'Basic Python; begin with the setup phase';
  const c = lesson.content;
  if (authored && !c) throw new Error(`Authored lesson has no content: ${lesson.id}`);
  const exerciseScript = authored ? lesson.artifacts?.scriptSource : undefined;
  const prerequisiteLinks =
    lesson.id === 'welcome-01'
      ? escapeHtml(prerequisite)
      : lesson.prerequisites.length
        ? lesson.prerequisites
            .map((id) => {
              const l = requiredLesson(id).lesson;
              return `<a href="${lessonLink(id, courseState.activePath)}">${escapeHtml(l.title)}</a>`;
            })
            .join('; ')
        : phase.prerequisitePhaseIds.length
          ? phase.prerequisitePhaseIds
              .map(
                (id) =>
                  `<a href="course.html?phase=${encodeURIComponent(id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}">${escapeHtml(requiredPhase(id).title)}</a>`,
              )
              .join('; ')
          : escapeHtml(prerequisite);
  const before = body.querySelector('.before-start p.soft');
  if (before) {
    before.innerHTML = `<strong>Before this:</strong> ${prerequisiteLinks}<br><strong>Hardware:</strong> ${inlineMath(lesson.hardware || phase.hardware)}`;
  }
  const portfolioLink = body.querySelector('a[href^="notebook.html?lesson="]');
  if (portfolioLink) {
    portfolioLink.setAttribute(
      'href',
      `notebook.html?lesson=${encodeURIComponent(lesson.id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}#portfolio`,
    );
  }
  const nav = body.querySelector('.lesson-nav');
  if (nav) {
    nav.innerHTML = `${index > 0 ? `<button id="previous">Previous: ${escapeHtml(allLessons[index - 1].title)}</button>` : '<span>First lesson in this route</span>'}${index >= 0 && index < allLessons.length - 1 ? `<button id="next">Next: ${escapeHtml(allLessons[index + 1].title)}</button>` : '<button id="phase-project">View the phase project</button>'}`;
  }
  if (authored) {
    const content = /** @type {import('../types/course').LessonContent} */ (c);
    bindCodeCopy();
    bindPracticeDrawer();
    if (content.figure === 'gradient') bindGradientFigure();
    if (content.figure === 'attention-mask') bindAttentionMaskFigure();
    const copyRun = $('#copy-run');
    if (copyRun)
      copyRun.onclick = async () => {
        try {
          if (!exerciseScript) throw new Error('Exercise script is unavailable');
          await navigator.clipboard.writeText(`python3 ${exerciseScript}`);
          copyRun.textContent = 'Copied';
        } catch {
          copyRun.textContent = 'Select the command to copy';
        }
      };
    const entry = courseState.lessonProgress.lessons[lesson.id] || {
      checkpointPassed: false,
      evidenceSaved: false,
    };
    const button = /** @type {HTMLButtonElement} */ (requireElement('#exercise-complete'));
    const label = () => {
      button.textContent = entry.evidenceSaved
        ? 'Mark exercise unfinished'
        : 'I ran my exercise and saved the evidence';
    };
    label();
    if (entry.checkpointPassed)
      requireElement('#checkpoint-result').textContent =
        'Checkpoint previously passed in this browser. You can try it again.';
    const checkpointForm = /** @type {HTMLFormElement} */ (requireElement('#lesson-checkpoint'));
    checkpointForm.onsubmit = (e) => {
      e.preventDefault();
      const answer = new FormData(checkpointForm).get('answer');
      if (typeof answer !== 'string' || !/^(0|[1-9]\d*)$/.test(answer)) return;
      const selected = Number(answer);
      if (!Number.isInteger(selected) || selected < 0 || selected >= content.options.length) return;
      const passed = selected === content.answer;
      requireElement('#checkpoint-result').innerHTML = inlineMath(
        (passed ? 'You’ve got it. ' : 'Let’s work through it. ') + content.optionFeedback[selected],
      );
      if (passed) {
        entry.checkpointPassed = true;
        courseState.lessonProgress.lessons[lesson.id] = entry;
        saveLessonProgress();
      }
    };
    /** @type {HTMLButtonElement} */ (requireElement('#reset-lesson')).onclick = () => {
      entry.checkpointPassed = false;
      entry.evidenceSaved = false;
      courseState.lessonProgress.lessons[lesson.id] = entry;
      saveLessonProgress();
      label();
      requireElement('#checkpoint-result').textContent =
        'Lesson progress reset. Your downloaded work is unchanged.';
    };
    button.onclick = () => {
      entry.evidenceSaved = !entry.evidenceSaved;
      courseState.lessonProgress.lessons[lesson.id] = entry;
      saveLessonProgress();
      label();
    };
    loadValidation()
      .then((receipt) => {
        const record = receipt.lessons.find(
          (l) => l.lessonId === lesson.id && l.contentHash === lesson.contentHash,
        );
        const runtimeStatus = $('#runtime-status');
        if (
          runtimeStatus &&
          requireElement('#lesson-body').dataset.lesson === lesson.id &&
          record
        ) {
          const deviceCount = record.runtime?.deviceCount || 0;
          const runtime = deviceCount > 1 ? `${deviceCount} logical CPU devices` : 'CPU';
          runtimeStatus.textContent = `Worked examples and reference solutions verified on ${runtime} (JAX ${receipt.jax}, NumPy ${receipt.numpy}, Python ${receipt.python}). ${executionScope(lesson).description} ${executionScope(lesson).target}`;
        }
      })
      .catch(() => {});
  }
  body.dataset.lesson = lesson.id;
  initRevisionNotices(body);
  const previousButton = /** @type {HTMLButtonElement | null} */ ($('#previous'));
  if (previousButton) {
    previousButton.onclick = () => {
      const previousLesson = allLessons[index - 1];
      if (!previousLesson) return;
      const found = requiredLesson(previousLesson.id);
      showLesson(found.phase, found.lesson);
    };
  }
  const nextButton = /** @type {HTMLButtonElement | null} */ ($('#next'));
  const followingLesson = allLessons[index + 1];
  if (nextButton && followingLesson)
    nextButton.dataset.href = lessonLink(followingLesson.id, courseState.activePath);
  if (nextButton && followingLesson)
    nextButton.onclick = () => {
      const found = requiredLesson(followingLesson.id);
      showLesson(found.phase, found.lesson);
    };
  const phaseProjectButton = /** @type {HTMLButtonElement | null} */ ($('#phase-project'));
  if (phaseProjectButton)
    phaseProjectButton.onclick = () => {
      location.href = phase.projectId
        ? `project.html?id=${encodeURIComponent(phase.projectId)}`
        : `course.html?phase=${encodeURIComponent(phase.id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}#curriculum`;
    };
  const url = new URL(location.href),
    changingLesson = (staticId || url.searchParams.get('lesson')) !== lesson.id;
  url.hash = requestedHash;
  if (staticId) url.searchParams.delete('lesson');
  else url.searchParams.set('lesson', lesson.id);
  if (changingLesson) url.searchParams.delete('find');
  if (courseState.activePath === 'all') url.searchParams.delete('path');
  if (options.history !== false) {
    if (changingLesson) history.pushState(null, '', url);
    else history.replaceState(null, '', url);
  }
  renderReaderNavigation(phase, lesson);
  observeStageBar();
  document.title = lesson.title + ' — JAX Pathways';
  const savedSection = priorSection;
  const findTerm = sameLesson ? lessonUrl.searchParams.get('find') : null;
  const matchSection = findTerm ? findLessonSection(findTerm) : '';
  const destination =
    requestedHash ||
    (matchSection ? '#' + matchSection : '') ||
    (!changingLesson && savedSection ? '#' + savedSection : '');
  if (matchSection && !requestedHash) {
    const foundUrl = new URL(location.href);
    foundUrl.hash = matchSection;
    history.replaceState(null, '', foundUrl);
  }
  if (destination) {
    requestAnimationFrame(() => scrollToLessonSection(destination));
  } else {
    window.scrollTo({ top: 0, behavior: 'instant' });
    requireElement('#lesson-title').focus({ preventScroll: true });
  }
  updateReadingPosition();
  saveReading(destination.replace(/^#/, ''));
}

/** @param {import('../types/course').Phase} phase @param {import('../types/course').Lesson} lesson */
function renderReaderNavigation(phase, lesson) {
  const route = courseState.curriculum.find((r) => r.id === courseState.activePath);
  requireElement('#reader-route').textContent = route ? route.title : 'Shared JAX course';
  const phaseIds = route?.phaseIds || currentCourse().phases.map((p) => p.id);
  requireElement('#reader-phase-title').textContent = phase.title;
  requireElement('#reader-phases').innerHTML = phaseIds
    .map(
      (id) =>
        `<a href="course.html?phase=${encodeURIComponent(id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}">${escapeHtml(requiredPhase(id).title)}</a>`,
    )
    .join('');
  /** @param {import('../types/course').Lesson} l */
  const href = (l) => lessonLink(l.id, courseState.activePath);
  requireElement('#reader-lessons').innerHTML = phase.lessons
    .filter((l) => l.status === 'authored' || l.id === lesson.id)
    .map(
      (l, i) =>
        `<a href="${href(l)}" ${l.id === lesson.id ? 'aria-current="page"' : ''}><span class="edition">${phase.number}.${String(i + 1).padStart(2, '0')} · ${l.status === 'authored' ? 'Draft' : 'Unwritten'}</span>${escapeHtml(l.title)}</a>`,
    )
    .join('');
  const outline = /** @type {HTMLDetailsElement | null} */ ($('#reader-outline-disclosure'));
  if (outline) outline.open = matchMedia('(min-width: 1101px)').matches;
  requireElement('#reader-sidebar').classList.remove('is-open');
  requireElement('#reader-menu').setAttribute('aria-expanded', 'false');
  $('#activity-bar')?.remove();
  const headings = Array.from(requireElement('#lesson-body').querySelectorAll('h2'));
  /** @type {Record<ActivityGroup, string[]>} */
  const groups = { Understand: [], Build: [], Experiment: [], Practice: [], Evidence: [] };
  /** @type {Partial<Record<ActivityGroup, string>>} */
  const firstHeadingByGroup = {};
  headings.forEach((h, i) => {
    h.id = `section-${i}`;
    const title = h.textContent.trim();
    const stageEl = /** @type {HTMLElement | null} */ (h.closest('[data-lesson-stage]'));
    const requestedGroup = stageEl?.dataset.lessonStage;
    const group = isActivityGroup(requestedGroup)
      ? requestedGroup
      : h.closest('.build-step') || /^\d+[.)] |Run the example|Build a numerical/.test(title)
        ? 'Build'
        : h.closest('.worked-experiment')
          ? 'Experiment'
          : h.closest('.lesson-practice,.lesson-checkpoint,.exercise-workbench') ||
              /Make it yours|Diagnose the result|Check your/.test(title)
            ? 'Practice'
            : /Keep your evidence|Primary references|Carry forward/.test(title)
              ? 'Evidence'
              : 'Understand';
    h.dataset.activityGroup = group;
    if (!firstHeadingByGroup[group]) firstHeadingByGroup[group] = h.id;
    groups[group].push(`<a href="#${h.id}">${escapeHtml(title)}</a>`);
  });
  $('#lesson-stage-bar')
    ?.querySelectorAll('a')
    .forEach((pill) => {
      if (!pill.hasAttribute('data-stage-pill')) return;
      const stageName = pill.getAttribute('data-stage-pill');
      if (!isActivityGroup(stageName)) {
        pill.hidden = true;
        return;
      }
      const targetId = firstHeadingByGroup[stageName];
      if (targetId) {
        pill.setAttribute('href', '#' + targetId);
        pill.hidden = false;
      } else {
        pill.hidden = true;
      }
    });
  if (headings.length) {
    const bar = document.createElement('nav');
    bar.id = 'activity-bar';
    bar.setAttribute('aria-label', 'Lesson activities');
    bar.innerHTML =
      '<a class="button" id="previous-activity">Previous activity</a><span id="activity-position"></span><a class="button primary" id="next-activity">Next activity</a>';
    requireElement('#lesson-page').after(bar);
  }
  requireElement('#reader-toc').innerHTML = Object.entries(groups)
    .filter(([, links]) => links.length)
    .map(([name, links]) => `<div class="toc-group"><p>${name}</p>${links.join('')}</div>`)
    .join('');
}

/** @param {string} hash */
function scrollToLessonSection(hash) {
  const id = String(hash || '').replace(/^#/, ''),
    target = id ? document.getElementById(id) : requireElement('#lesson-title');
  if (!target) return;
  target.scrollIntoView({ block: 'start', behavior: 'instant' });
  if (id) saveReading(id);
  updateReadingPosition();
}

/** @type {ResizeObserver | undefined} */
let stageBarObserver;
function observeStageBar() {
  stageBarObserver?.disconnect();
  const bar = $('#lesson-stage-bar');
  const sync = () => {
    $('#lesson-body')?.style.setProperty(
      '--reader-anchor-offset',
      `${(bar?.getBoundingClientRect().height || 0) + 30}px`,
    );
  };
  sync();
  if (bar && typeof ResizeObserver !== 'undefined') {
    stageBarObserver = new ResizeObserver(sync);
    stageBarObserver.observe(bar);
  }
}

const stageDisplayNames = {
  Understand: 'Understand the Mechanism',
  Build: 'Build & Run Reference',
  Experiment: 'Controlled Experiments',
  Practice: 'Hands-On Exercise & Check',
  Evidence: 'Keep Your Evidence',
};

function updateReadingPosition() {
  if (!courseState.standalone) return;
  const headings = Array.from(requireElement('#lesson-body').querySelectorAll('h2'));
  if (!headings.length) return;
  const lessonBody = requireElement('#lesson-body');
  const stageBar = $('#lesson-stage-bar');
  const anchorOffset =
    Number.parseFloat(getComputedStyle(lessonBody).getPropertyValue('--reader-anchor-offset')) || 0;
  const scrollPadding =
    Number.parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) ||
    Number.parseFloat(getComputedStyle(document.body).scrollPaddingTop) ||
    0;
  const activationLine = Math.max(
    (stageBar?.getBoundingClientRect().bottom || 0) + 16,
    anchorOffset + scrollPadding + 1,
  );
  const current =
    headings.filter((h) => h.getBoundingClientRect().top <= activationLine).at(-1) || headings[0];
  if (!current) return;
  requireElement('#reader-toc')
    .querySelectorAll('a')
    .forEach((a) => {
      if (a.getAttribute('href') === '#' + current.id) a.setAttribute('aria-current', 'location');
      else a.removeAttribute('aria-current');
    });
  const groupValue = current.getAttribute('data-activity-group');
  if (!isActivityGroup(groupValue)) return;
  const group = groupValue,
    steps = [
      ...new Set(
        headings.map((h) => h.getAttribute('data-activity-group')).filter(isActivityGroup),
      ),
    ],
    index = steps.indexOf(group),
    previous = /** @type {HTMLAnchorElement | null} */ ($('#previous-activity')),
    next = /** @type {HTMLAnchorElement | null} */ ($('#next-activity')),
    stageLabel = `Stage ${index + 1} of ${steps.length} · ${stageDisplayNames[group] || group}`;
  $('#lesson-body')?.setAttribute('data-active-stage', group);
  if (stageBar) {
    stageBar.setAttribute('data-active-stage', group);
    const activeLabel = $('#stage-bar-active-label');
    if (activeLabel) activeLabel.textContent = stageLabel;
    stageBar.querySelectorAll('a').forEach((pill) => {
      if (!pill.hasAttribute('data-stage-pill')) return;
      const stage = pill.getAttribute('data-stage-pill');
      const pillIndex = isActivityGroup(stage) ? steps.indexOf(stage) : -1;
      if (stage === group) {
        pill.setAttribute('aria-current', 'step');
        pill.removeAttribute('data-stage-passed');
      } else {
        pill.removeAttribute('aria-current');
        if (pillIndex >= 0 && pillIndex < index) pill.setAttribute('data-stage-passed', 'true');
        else pill.removeAttribute('data-stage-passed');
      }
    });
  }
  /** @type {NodeListOf<HTMLElement>} */ (
    requireElement('#lesson-body').querySelectorAll('.lesson-stage')
  ).forEach((stageEl) => {
    if (stageEl.getAttribute('data-lesson-stage') === group)
      stageEl.setAttribute('data-stage-active', 'true');
    else stageEl.removeAttribute('data-stage-active');
  });
  if (previous && next) {
    $('#activity-bar')?.setAttribute('data-active-stage', group);
    previous.hidden = index === 0;
    previous.textContent = index ? 'Previous: ' + steps[index - 1] : 'Previous';
    const previousHeading = headings.find(
      (h) => h.getAttribute('data-activity-group') === steps[index - 1],
    );
    previous.href = index && previousHeading ? '#' + previousHeading.id : '#lesson-title';
    requireElement('#activity-position').textContent = stageLabel;
    const nextLesson = $('#next');
    next.textContent =
      index < steps.length - 1
        ? 'Next: ' + steps[index + 1]
        : nextLesson
          ? 'Next lesson'
          : 'Course map';
    const nextHeading = headings.find(
      (h) => h.getAttribute('data-activity-group') === steps[index + 1],
    );
    next.href =
      index < steps.length - 1 && nextHeading
        ? '#' + nextHeading.id
        : nextLesson
          ? nextLesson.dataset.href || 'course.html'
          : 'course.html';
  }
  saveReading(current.id);
}

function bindPracticeDrawer() {
  document.querySelectorAll('button').forEach((button) => {
    if (!button.hasAttribute('data-practice-mode')) return;
    button.onclick = () => {
      const mode = button.dataset.practiceMode;
      document.querySelectorAll('button').forEach((control) => {
        if (control.hasAttribute('data-practice-mode'))
          control.setAttribute('aria-pressed', String(control === button));
      });
      document.querySelectorAll('section').forEach((panel) => {
        if (panel.hasAttribute('data-practice-panel'))
          panel.hidden = panel.dataset.practicePanel !== mode;
      });
    };
  });
}

/** @param {string} query */
function findLessonSection(query) {
  const term = String(query).trim().toLowerCase();
  if (!term) return '';
  const section = Array.from(
    requireElement('#lesson-body').querySelectorAll(
      '.concept-section,.worked-experiment,.build-step,.lesson-practice,.exercise-workbench',
    ),
  ).find((node) => (node.textContent || '').toLowerCase().includes(term));
  if (section) return section.querySelector('h2')?.id || '';
  const paragraph = Array.from(
    requireElement('#lesson-body').querySelectorAll(
      '.lesson-stage > p, .lesson-stage > ul, .lesson-stage > ol, :scope > p, :scope > ul, :scope > ol',
    ),
  ).find((node) => (node.textContent || '').toLowerCase().includes(term));
  if (paragraph) {
    let previous = paragraph.previousElementSibling;
    while (previous && previous.tagName !== 'H2') previous = previous.previousElementSibling;
    return previous?.id || '';
  }
  return '';
}

function initReaderEvents() {
  const closeButton = /** @type {HTMLButtonElement | null} */ ($('.close'));
  if (closeButton) closeButton.onclick = closeLesson;
  if (courseState.standalone)
    matchMedia('(min-width: 1101px)').addEventListener('change', (event) => {
      const outline = /** @type {HTMLDetailsElement | null} */ ($('#reader-outline-disclosure'));
      if (outline) outline.open = event.matches;
    });
  window.addEventListener('popstate', () => {
    if (!courseState.course) return;
    const params = new URLSearchParams(location.search);
    const lessonId = $('#lesson-page')?.dataset.staticLesson || params.get('lesson');
    if (!lessonId) return;
    const found = lessonById(lessonId);
    if (!found) return;
    const requestedPath = params.get('path');
    courseState.activePath = courseState.curriculum.some((r) => r.id === requestedPath)
      ? requestedPath || 'all'
      : 'all';
    if (requireElement('#lesson-body').dataset.lesson !== found.lesson.id)
      showLesson(found.phase, found.lesson, { history: false });
    else scrollToLessonSection(location.hash);
  });
  if (courseState.standalone) {
    window.addEventListener('hashchange', () => scrollToLessonSection(location.hash));
    window.addEventListener(
      'scroll',
      () => {
        if (courseState.readingFrame) return;
        courseState.readingFrame = requestAnimationFrame(() => {
          courseState.readingFrame = 0;
          updateReadingPosition();
        });
      },
      { passive: true },
    );
  }
}

export {
  saveReading,
  saveLessonProgress,
  closeLesson,
  showLesson,
  renderReaderNavigation,
  scrollToLessonSection,
  updateReadingPosition,
  bindPracticeDrawer,
  findLessonSection,
  initReaderEvents,
};
