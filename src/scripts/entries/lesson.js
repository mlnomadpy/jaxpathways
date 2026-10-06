import { bootCourse } from '../boot.js';
import { courseState, lessonById } from '../../lib/course-state.js';
import { initReaderEvents, showLesson } from '../reader.js';
bootCourse(() => {
  const params = new URLSearchParams(location.search);
  const found = lessonById(params.get('lesson'));
  if (!found) {
    document.querySelector('#lesson-body').innerHTML =
      '<h1>Lesson not found</h1><p><a href="course.html">Browse the course</a>.</p>';
    return;
  }
  const route = courseState.curriculum.find(
    (route) => route.id === params.get('path') && route.phaseIds.includes(found.phase.id),
  );
  courseState.activePath = route?.id || 'all';
  initReaderEvents();
  showLesson(found.phase, found.lesson);
  document.querySelector('#reader-menu').onclick = () => {
    const sidebar = document.querySelector('#reader-sidebar');
    sidebar.classList.toggle('is-open');
    document
      .querySelector('#reader-menu')
      .setAttribute('aria-expanded', String(sidebar.classList.contains('is-open')));
  };
}, '#lesson-body');
