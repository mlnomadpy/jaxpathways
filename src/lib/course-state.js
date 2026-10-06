import { learnerStorageKeys, validateLessonProgress } from './learner-records.js';

function phaseById(id) {
  return courseState.course.phases.find((p) => p.id === id);
}

function lessonById(id) {
  for (const p of courseState.course.phases) {
    const lesson = p.lessons.find((l) => l.id === id);
    if (lesson) return { phase: p, lesson };
  }
}

function authoredIds() {
  return courseState.course.phases
    .flatMap((p) => p.lessons)
    .filter((l) => l.status === 'authored')
    .map((l) => l.id);
}

const courseState = {
  standalone: false,
  course: null,
  curriculum: [],
  activePath: 'all',
  courseQuery: '',
  courseHardware: 'all',
  completed: false,
  readingState: { version: 1, lessons: {} },
  readingFrame: 0,
  lessonProgress: { version: 1, lessons: {} },
};

const readingKey = 'jaxpathways-reading-v1';

const progressKey = 'jaxpathways-lessons-v1';

async function loadCourse() {
  courseState.standalone = Boolean(document.querySelector('#lesson-page'));
  const response = await fetch('curriculum.json');
  if (!response.ok) throw Error('Curriculum unavailable');
  courseState.course = await response.json();
  courseState.curriculum = courseState.course.pathways.map((route) => ({
    ...route,
    lessons: route.phaseIds.flatMap((id) => phaseById(id).lessons),
  }));
  try {
    courseState.completed = localStorage.getItem(learnerStorageKeys.sampleCompleted) === 'true';
  } catch {}
  try {
    const saved = JSON.parse(localStorage.getItem(readingKey));
    if (saved?.version === 1 && saved.lessons && typeof saved.lessons === 'object')
      courseState.readingState = saved;
  } catch {}
  try {
    const saved = JSON.parse(localStorage.getItem(progressKey));
    if (validateLessonProgress(saved, authoredIds())) courseState.lessonProgress = saved;
  } catch {}
  return courseState.course;
}

export { phaseById, lessonById, authoredIds, courseState, readingKey, progressKey, loadCourse };
