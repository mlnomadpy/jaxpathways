import { $ } from '../lib/dom.js';
import { courseState, lessonById } from '../lib/course-state.js';
import { escapeHtml } from '../lib/html.js';

function renderHomeContinue() {
  const panel = /** @type {HTMLElement | null} */ ($('#home-continue'));
  if (!panel || !courseState.course) return;
  const lastId = courseState.readingState.lastLessonId;
  const found = lastId ? lessonById(lastId) : undefined;
  if (!found || found.lesson.status !== 'authored') {
    panel.hidden = true;
    return;
  }
  const savedPath = courseState.readingState.path;
  const path =
    savedPath && courseState.curriculum.some((r) => r.id === savedPath) ? savedPath : 'all';
  const routeTitle =
    path === 'all'
      ? 'Shared JAX course'
      : courseState.curriculum.find((r) => r.id === path)?.title || 'Shared JAX course';
  const section =
    courseState.readingState.lessons[found.lesson.id]?.sectionId ||
    courseState.readingState.sectionId ||
    '';
  const url = `lesson.html?lesson=${encodeURIComponent(found.lesson.id)}${path === 'all' ? '' : '&path=' + encodeURIComponent(path)}${/^section-\d+$/.test(section) ? '#' + section : ''}`;
  document.body.classList.add('has-reading');
  panel.hidden = false;
  panel.innerHTML = `<div><p class="edition">Welcome back</p><h2>${escapeHtml(found.lesson.title)}</h2><p>${escapeHtml(routeTitle)}. Return to your reading position; evidence and checkpoints are tracked separately.</p></div><a class="button primary" href="${url}">Resume lesson</a><a href="course.html${path === 'all' ? '' : '?path=' + encodeURIComponent(path)}">View the course</a>`;
}

export { renderHomeContinue };
