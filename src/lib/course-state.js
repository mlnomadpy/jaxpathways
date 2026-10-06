import { fetchJson } from './http.js';
import { readStoredJson } from './storage.js';
import {
  learnerStorageKeys,
  validateLessonProgress,
  validateReadingState,
} from './learner-records.js';

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

const readingKey = learnerStorageKeys.reading;

const progressKey = learnerStorageKeys.lessonProgress;

async function loadCourse() {
  courseState.standalone = Boolean(document.querySelector('#lesson-page'));
  courseState.course = await fetchJson('curriculum.json');
  courseState.curriculum = courseState.course.pathways.map((route) => ({
    ...route,
    lessons: route.phaseIds.flatMap((id) => phaseById(id).lessons),
  }));
  courseState.completed = false;
  try {
    courseState.completed = localStorage.getItem(learnerStorageKeys.sampleCompleted) === 'true';
  } catch {}
  const savedReading = readStoredJson(readingKey);
  const context = {
    lessonIds: courseState.course.phases.flatMap((phase) =>
      phase.lessons.map((lesson) => lesson.id),
    ),
    routeIds: courseState.curriculum.map((route) => route.id),
  };
  courseState.readingState =
    savedReading && validateReadingState(savedReading, context)
      ? savedReading
      : { version: 1, lessons: {} };
  const savedProgress = readStoredJson(progressKey);
  courseState.lessonProgress = validateLessonProgress(savedProgress, authoredIds())
    ? savedProgress
    : { version: 1, lessons: {} };
  return courseState.course;
}

export { phaseById, lessonById, authoredIds, courseState, readingKey, progressKey, loadCourse };
