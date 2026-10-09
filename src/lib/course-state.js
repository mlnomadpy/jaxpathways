import { fetchJson } from './http.js';
import { readStoredJson } from './storage.js';
import {
  learnerStorageKeys,
  validateLessonProgress,
  validateReadingState,
} from './learner-records.js';

/** @param {string} id */
function phaseById(id) {
  return courseState.course?.phases.find((p) => p.id === id);
}

/** @param {string} id */
function lessonById(id) {
  for (const p of courseState.course?.phases || []) {
    const lesson = p.lessons.find((l) => l.id === id);
    if (lesson) return { phase: p, lesson };
  }
}

function authoredIds() {
  return (courseState.course?.phases || [])
    .flatMap((p) => p.lessons)
    .filter((l) => l.status === 'authored')
    .map((l) => l.id);
}

/**
 * @typedef {import('../types/course').Pathway & { lessons: import('../types/course').Lesson[] }} CurriculumRoute
 * @typedef {{
 *   standalone: boolean;
 *   course: import('../types/course').Course | null;
 *   curriculum: CurriculumRoute[];
 *   activePath: string;
 *   courseQuery: string;
 *   courseHardware: string;
 *   completed: boolean;
 *   readingState: {
 *     version: number;
 *     lastLessonId?: string;
 *     path?: string;
 *     sectionId?: string;
 *     updatedAt?: string;
 *     lessons: Record<string, { sectionId: string; updatedAt: string }>;
 *   };
 *   readingFrame: number;
 *   lessonProgress: {
 *     version: number;
 *     lessons: Record<string, { checkpointPassed: boolean; evidenceSaved: boolean }>;
 *   };
 * }} CourseState
 */

/** @type {CourseState} */
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
  const loadedCourse = /** @type {import('../types/course').Course} */ (
    await fetchJson('curriculum.json')
  );
  courseState.course = loadedCourse;
  courseState.curriculum = loadedCourse.pathways.map((route) => ({
    ...route,
    lessons: route.phaseIds.flatMap((id) => phaseById(id)?.lessons || []),
  }));
  courseState.completed = false;
  try {
    courseState.completed = localStorage.getItem(learnerStorageKeys.sampleCompleted) === 'true';
  } catch {}
  const savedReading = /** @type {CourseState['readingState'] | null} */ (
    readStoredJson(readingKey)
  );
  const context = {
    lessonIds: loadedCourse.phases.flatMap((phase) => phase.lessons.map((lesson) => lesson.id)),
    routeIds: courseState.curriculum.map((route) => route.id),
  };
  courseState.readingState =
    savedReading && validateReadingState(savedReading, context)
      ? savedReading
      : { version: 1, lessons: {} };
  const savedProgress = /** @type {CourseState['lessonProgress'] | null} */ (
    readStoredJson(progressKey)
  );
  courseState.lessonProgress =
    savedProgress && validateLessonProgress(savedProgress, authoredIds())
      ? savedProgress
      : { version: 1, lessons: {} };
  return loadedCourse;
}

export { phaseById, lessonById, authoredIds, courseState, readingKey, progressKey, loadCourse };
