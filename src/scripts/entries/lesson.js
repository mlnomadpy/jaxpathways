import { bootCourse } from '../boot.js';
import { courseState, lessonById } from '../../lib/course-state.js';
import { initReaderEvents, showLesson } from '../reader.js';
import { legacyLessonUrl } from '../../lib/urls.js';
bootCourse(() => {
  const params = new URLSearchParams(location.search);
  const staticId = document.querySelector('#lesson-page')?.dataset.staticLesson;
  const found = lessonById(staticId || params.get('lesson'));
  if (!staticId && found?.lesson.status === 'authored') {
    location.replace(legacyLessonUrl(found.lesson.id, location.href));
    return;
  }
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
