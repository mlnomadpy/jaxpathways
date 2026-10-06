import { inlineMath } from '../lib/math.js';
import { renderLesson } from '../lib/lesson-template.js';
import {
  courseState,
  readingKey,
  progressKey,
  phaseById,
  lessonById,
} from '../lib/course-state.js';
import { $ } from '../lib/dom.js';
import { escapeHtml } from '../lib/html.js';

import { bindGradientFigure, bindAttentionMaskFigure } from './lesson-diagrams.js';

import { bindCodeCopy } from './copy.js';

function saveReading(sectionId = '') {
  if (!courseState.standalone || !$('#lesson-body')?.dataset.lesson) return;
  const lessonId = $('#lesson-body').dataset.lesson;
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

function showLesson(phase, lesson, options = {}) {
  if (!courseState.standalone) {
    location.href = `lesson.html?lesson=${encodeURIComponent(lesson.id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}`;
    return;
  }
  const authored = lesson.status === 'authored';
  const allLessons = (
    courseState.curriculum.find((r) => r.id === courseState.activePath)?.phaseIds ||
    courseState.course.phases.map((p) => p.id)
  )
    .flatMap((id) => phaseById(id).lessons)
    .filter((l) => (authored ? l.status === 'authored' : true));
  const index = allLessons.findIndex((l) => l.id === lesson.id);
  const lessonUrl = new URL(location.href),
    sameLesson = lessonUrl.searchParams.get('lesson') === lesson.id;
  const requestedHash = sameLesson ? lessonUrl.hash : '';
  const priorSection = courseState.readingState.lessons[lesson.id]?.sectionId;
  const prerequisite =
    lesson.id === 'welcome-01'
      ? 'No terminal or Git experience required'
      : lesson.prerequisites.map((id) => lessonById(id).lesson.title).join('; ') ||
        phase.prerequisitePhaseIds.map((id) => phaseById(id).title).join('; ') ||
        'Basic Python; begin with the setup phase';
  const c = lesson.content;
  const prerequisiteLinks =
    lesson.id === 'welcome-01'
      ? escapeHtml(prerequisite)
      : lesson.prerequisites.length
        ? lesson.prerequisites
            .map((id) => {
              const l = lessonById(id).lesson;
              return `<a href="lesson.html?lesson=${encodeURIComponent(id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}">${escapeHtml(l.title)}</a>`;
            })
            .join('; ')
        : phase.prerequisitePhaseIds.length
          ? phase.prerequisitePhaseIds
              .map(
                (id) =>
                  `<a href="course.html?phase=${encodeURIComponent(id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}">${escapeHtml(phaseById(id).title)}</a>`,
              )
              .join('; ')
          : escapeHtml(prerequisite);
  $('#lesson-body').innerHTML = renderLesson({
    phase,
    lesson,
    prerequisiteLinks,
    allLessons,
    index,
    activePath: courseState.activePath,
  });
  if (authored) {
    bindCodeCopy();
    bindPracticeDrawer();
    if (c.figure === 'gradient') bindGradientFigure();
    if (c.figure === 'attention-mask') bindAttentionMaskFigure();
    if ($('#copy-run'))
      $('#copy-run').onclick = async () => {
        try {
          await navigator.clipboard.writeText(`python3 ${lesson.artifacts.scriptSource}`);
          $('#copy-run').textContent = 'Copied';
        } catch {
          $('#copy-run').textContent = 'Select the command to copy';
        }
      };
    const entry = courseState.lessonProgress.lessons[lesson.id] || {
      checkpointPassed: false,
      evidenceSaved: false,
    };
    const button = $('#exercise-complete');
    const label = () => {
      button.textContent = entry.evidenceSaved
        ? 'Mark exercise unfinished'
        : 'I ran my exercise and saved the evidence';
    };
    label();
    if (entry.checkpointPassed)
      $('#checkpoint-result').textContent =
        'Checkpoint previously passed in this browser. You can try it again.';
    $('#lesson-checkpoint').onsubmit = (e) => {
      e.preventDefault();
      const selected = Number(new FormData(e.target).get('answer'));
      const passed = selected === c.answer;
      $('#checkpoint-result').innerHTML = inlineMath(
        (passed ? 'You’ve got it. ' : 'Let’s work through it. ') + c.explanation,
      );
      if (passed) {
        entry.checkpointPassed = true;
        courseState.lessonProgress.lessons[lesson.id] = entry;
        saveLessonProgress();
      }
    };
    $('#reset-lesson').onclick = () => {
      entry.checkpointPassed = false;
      entry.evidenceSaved = false;
      courseState.lessonProgress.lessons[lesson.id] = entry;
      saveLessonProgress();
      label();
      $('#checkpoint-result').textContent =
        'Lesson progress reset. Your downloaded work is unchanged.';
    };
    button.onclick = () => {
      entry.evidenceSaved = !entry.evidenceSaved;
      courseState.lessonProgress.lessons[lesson.id] = entry;
      saveLessonProgress();
      label();
    };
    fetch('validation.json', { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((receipt) => {
        const record = receipt?.lessons.find(
          (l) => l.lessonId === lesson.id && l.contentHash === lesson.contentHash,
        );
        if ($('#runtime-status') && $('#lesson-body').dataset.lesson === lesson.id && record) {
          const runtime =
            record.runtime?.deviceCount > 1
              ? `${record.runtime.deviceCount} logical CPU devices`
              : 'CPU';
          $('#runtime-status').textContent =
            `Worked examples and reference solutions passed on ${runtime} with JAX ${receipt.jax}, NumPy ${receipt.numpy}, Python ${receipt.python}. TPU execution has not been validated.`;
        }
      })
      .catch(() => {});
  }
  $('#lesson-body').dataset.lesson = lesson.id;
  if ($('#previous'))
    $('#previous').onclick = () => {
      const found = lessonById(allLessons[index - 1].id);
      showLesson(found.phase, found.lesson);
    };
  if ($('#next'))
    $('#next').dataset.href =
      `lesson.html?lesson=${encodeURIComponent(allLessons[index + 1].id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}`;
  if ($('#next'))
    $('#next').onclick = () => {
      const found = lessonById(allLessons[index + 1].id);
      showLesson(found.phase, found.lesson);
    };
  if ($('#phase-project'))
    $('#phase-project').onclick = () => {
      location.href = phase.projectId
        ? `project.html?id=${encodeURIComponent(phase.projectId)}`
        : `course.html?phase=${encodeURIComponent(phase.id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}#curriculum`;
    };
  const url = new URL(location.href),
    changingLesson = url.searchParams.get('lesson') !== lesson.id;
  url.hash = requestedHash;
  url.searchParams.set('lesson', lesson.id);
  if (changingLesson) url.searchParams.delete('find');
  if (courseState.activePath === 'all') url.searchParams.delete('path');
  if (options.history !== false) {
    if (changingLesson) history.pushState(null, '', url);
    else history.replaceState(null, '', url);
  }
  renderReaderNavigation(phase, lesson);
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
    $('#lesson-title').focus({ preventScroll: true });
  }
  updateReadingPosition();
  saveReading(destination.replace(/^#/, ''));
}

function renderReaderNavigation(phase, lesson) {
  const route = courseState.curriculum.find((r) => r.id === courseState.activePath);
  $('#reader-route').textContent = route ? route.title : 'Shared JAX course';
  const phaseIds = route?.phaseIds || courseState.course.phases.map((p) => p.id);
  $('#reader-phase-title').textContent = phase.title;
  $('#reader-phases').innerHTML = phaseIds
    .map(
      (id) =>
        `<a href="course.html?phase=${encodeURIComponent(id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}">${escapeHtml(phaseById(id).title)}</a>`,
    )
    .join('');
  const href = (l) =>
    `lesson.html?lesson=${encodeURIComponent(l.id)}${courseState.activePath === 'all' ? '' : '&path=' + encodeURIComponent(courseState.activePath)}`;
  $('#reader-lessons').innerHTML = phase.lessons
    .filter((l) => l.status === 'authored' || l.id === lesson.id)
    .map(
      (l, i) =>
        `<a href="${href(l)}" ${l.id === lesson.id ? 'aria-current="page"' : ''}><span class="edition">${phase.number}.${String(i + 1).padStart(2, '0')} · ${l.status === 'authored' ? 'Draft' : 'Unwritten'}</span>${escapeHtml(l.title)}</a>`,
    )
    .join('');
  const outline = $('#reader-outline-disclosure');
  if (outline) outline.open = matchMedia('(min-width: 1101px)').matches;
  $('#reader-sidebar').classList.remove('is-open');
  $('#reader-menu').setAttribute('aria-expanded', 'false');
  $('#activity-bar')?.remove();
  const headings = Array.from($('#lesson-body').querySelectorAll('h2'));
  const groups = { Understand: [], Build: [], Experiment: [], Practice: [], Evidence: [] };
  headings.forEach((h, i) => {
    h.id = `section-${i}`;
    const title = h.textContent.trim();
    const group =
      h.closest('.build-step') || /^\d+[.)] |Run the example|Build a numerical/.test(title)
        ? 'Build'
        : h.closest('.worked-experiment')
          ? 'Experiment'
          : h.closest('.lesson-practice,.lesson-checkpoint') ||
              /Make it yours|Diagnose the result|Check your/.test(title)
            ? 'Practice'
            : /Keep your evidence|Primary references|Carry forward/.test(title)
              ? 'Evidence'
              : 'Understand';
    h.dataset.activityGroup = group;
    groups[group].push(`<a href="#${h.id}">${escapeHtml(title)}</a>`);
  });
  if (headings.length) {
    const bar = document.createElement('nav');
    bar.id = 'activity-bar';
    bar.setAttribute('aria-label', 'Lesson activities');
    bar.innerHTML =
      '<a class="button" id="previous-activity">Previous activity</a><span id="activity-position"></span><a class="button primary" id="next-activity">Next activity</a>';
    $('#lesson-page').after(bar);
  }
  $('#reader-toc').innerHTML = Object.entries(groups)
    .filter(([, links]) => links.length)
    .map(([name, links]) => `<div class="toc-group"><p>${name}</p>${links.join('')}</div>`)
    .join('');
}

function scrollToLessonSection(hash) {
  const id = String(hash || '').replace(/^#/, ''),
    target = id ? document.getElementById(id) : $('#lesson-title');
  if (!target) return;
  target.scrollIntoView({ block: 'start', behavior: 'instant' });
  if (id) saveReading(id);
  updateReadingPosition();
}

function updateReadingPosition() {
  if (!courseState.standalone) return;
  const headings = Array.from($('#lesson-body').querySelectorAll('h2[id]'));
  if (!headings.length) return;
  const current =
    headings.filter((h) => h.getBoundingClientRect().top <= 180).at(-1) || headings[0];
  $('#reader-toc')
    .querySelectorAll('a')
    .forEach((a) => {
      if (a.getAttribute('href') === '#' + current.id) a.setAttribute('aria-current', 'location');
      else a.removeAttribute('aria-current');
    });
  const group = current.dataset.activityGroup,
    steps = [...new Set(headings.map((h) => h.dataset.activityGroup))],
    index = steps.indexOf(group),
    previous = $('#previous-activity'),
    next = $('#next-activity');
  if (previous) {
    previous.hidden = index === 0;
    previous.textContent = index ? 'Previous: ' + steps[index - 1] : 'Previous';
    previous.href = index
      ? '#' + headings.find((h) => h.dataset.activityGroup === steps[index - 1]).id
      : '#lesson-title';
    $('#activity-position').textContent = group;
    const nextLesson = $('#next');
    next.textContent =
      index < steps.length - 1
        ? 'Next: ' + steps[index + 1]
        : nextLesson
          ? 'Next lesson'
          : 'Course map';
    next.href =
      index < steps.length - 1
        ? '#' + headings.find((h) => h.dataset.activityGroup === steps[index + 1]).id
        : nextLesson
          ? nextLesson.dataset.href || 'course.html'
          : 'course.html';
  }
  saveReading(current.id);
}

function bindPracticeDrawer() {
  document.querySelectorAll('[data-practice-mode]').forEach(
    (button) =>
      (button.onclick = () => {
        document
          .querySelectorAll('[data-practice-mode]')
          .forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
        document
          .querySelectorAll('[data-practice-panel]')
          .forEach(
            (panel) => (panel.hidden = panel.dataset.practicePanel !== button.dataset.practiceMode),
          );
      }),
  );
}

function findLessonSection(query) {
  const term = String(query).trim().toLowerCase();
  if (!term) return '';
  const section = Array.from(
    $('#lesson-body').querySelectorAll(
      '.concept-section,.worked-experiment,.build-step,.lesson-practice',
    ),
  ).find((node) => node.textContent.toLowerCase().includes(term));
  if (section) return section.querySelector('h2')?.id || '';
  const paragraph = Array.from($('#lesson-body').querySelectorAll(':scope > p')).find((node) =>
    node.textContent.toLowerCase().includes(term),
  );
  if (paragraph) {
    let previous = paragraph.previousElementSibling;
    while (previous && previous.tagName !== 'H2') previous = previous.previousElementSibling;
    return previous?.id || '';
  }
  return '';
}

function initReaderEvents() {
  if ($('.close')) $('.close').onclick = closeLesson;
  if (courseState.standalone)
    matchMedia('(min-width: 1101px)').addEventListener('change', (event) => {
      const outline = $('#reader-outline-disclosure');
      if (outline) outline.open = event.matches;
    });
  window.addEventListener('popstate', () => {
    if (!courseState.course) return;
    const params = new URLSearchParams(location.search),
      found = lessonById(params.get('lesson'));
    if (!found) return;
    courseState.activePath = courseState.curriculum.some((r) => r.id === params.get('path'))
      ? params.get('path')
      : 'all';
    if ($('#lesson-body').dataset.lesson !== found.lesson.id)
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
