import { bootCourse } from '../boot.js';
import { courseState, lessonById } from '../../lib/course-state.js';
import { initReaderEvents, showLesson } from '../reader.js';
import { legacyLessonUrl } from '../../lib/urls.js';
bootCourse(() => {
  const params = new URLSearchParams(location.search);
  const page = document.querySelector('#lesson-page');
  const body = document.querySelector('#lesson-body');
  if (!(page instanceof HTMLElement) || !(body instanceof HTMLElement)) return;
  const staticId = page.dataset.staticLesson;
  const requestedId = staticId || params.get('lesson');
  const found = requestedId ? lessonById(requestedId) : undefined;
  if (!staticId && found?.lesson.status === 'authored') {
    location.replace(legacyLessonUrl(found.lesson.id, location.href));
    return;
  }
  if (!found) {
    body.innerHTML = '<h1>Lesson not found</h1><p><a href="course.html">Browse the course</a>.</p>';
    return;
  }
  const route = courseState.curriculum.find(
    (route) => route.id === params.get('path') && route.phaseIds.includes(found.phase.id),
  );
  courseState.activePath = route?.id || 'all';
  initReaderEvents();
  showLesson(found.phase, found.lesson);
  const menu = document.querySelector('#reader-menu');
  const sidebar = document.querySelector('#reader-sidebar');
  if (menu instanceof HTMLButtonElement && sidebar instanceof HTMLElement)
    menu.onclick = () => {
      sidebar.classList.toggle('is-open');
      menu.setAttribute('aria-expanded', String(sidebar.classList.contains('is-open')));
    };
}, '#lesson-body');
