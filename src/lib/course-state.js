import { fetchJson } from './http.js';
import { readStoredJson } from './storage.js';
import {
  learnerStorageKeys,
  validateLessonProgress,
  validateReadingState,
} from './learner-records.js';

/** @param {unknown} value */
function courseRecord(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Unsupported course data');
  }
  const course = /** @type {Record<string, unknown>} */ (value);
  if (
    typeof course.title !== 'string' ||
    !Array.isArray(course.phases) ||
    !Array.isArray(course.pathways) ||
    !Array.isArray(course.projectIds) ||
    !Array.isArray(course.roles) ||
    !Array.isArray(course.domains)
  ) {
    throw new Error('Unsupported course data');
  }
  for (const phase of course.phases) {
    if (
      !phase ||
      typeof phase !== 'object' ||
      !('id' in phase) ||
      typeof phase.id !== 'string' ||
      !('lessons' in phase) ||
      !Array.isArray(phase.lessons) ||
      phase.lessons.some(
        /** @param {unknown} lesson */
        (lesson) =>
          !lesson ||
          typeof lesson !== 'object' ||
          !('id' in lesson) ||
          typeof lesson.id !== 'string' ||
          !('title' in lesson) ||
          typeof lesson.title !== 'string' ||
          !('status' in lesson) ||
          !['authored', 'planned'].includes(String(lesson.status)),
      )
    ) {
      throw new Error('Unsupported course phase data');
    }
  }
  return /** @type {import('../types/course').Course} */ (value);
}

/** @param {unknown} value */
function currentLessonRecord(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Current lesson data is missing');
  }
  const payload = /** @type {Record<string, unknown>} */ (value);
  if (
    typeof payload.phaseId !== 'string' ||
    !payload.lesson ||
    typeof payload.lesson !== 'object' ||
    Array.isArray(payload.lesson)
  ) {
    throw new Error('Unsupported current lesson data');
  }
  const lesson = /** @type {Record<string, unknown>} */ (payload.lesson);
  if (
    typeof lesson.id !== 'string' ||
    typeof lesson.title !== 'string' ||
    lesson.status !== 'authored' ||
    !lesson.content ||
    typeof lesson.content !== 'object'
  ) {
    throw new Error('Unsupported current lesson data');
  }
  return /** @type {{ phaseId: string; lesson: import('../types/course').Lesson }} */ (value);
}

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
 *   courseTypeFilter: string;
 *   completed: boolean;
 *   readingState: {
 *     version: 1;
 *     lastLessonId?: string;
 *     path?: string;
 *     sectionId?: string;
 *     updatedAt?: string;
 *     lessons: Record<string, { sectionId: string; updatedAt: string }>;
 *   };
 *   readingFrame: number;
 *   lessonProgress: {
 *     version: 1;
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
  courseTypeFilter: 'all',
  completed: false,
  readingState: { version: 1, lessons: {} },
  readingFrame: 0,
  lessonProgress: { version: 1, lessons: {} },
};

const readingKey = learnerStorageKeys.reading;

const progressKey = learnerStorageKeys.lessonProgress;

async function loadCourse() {
  courseState.standalone = Boolean(document.querySelector('#lesson-page'));
  let loadedCourse;
  if (courseState.standalone) {
    loadedCourse = courseRecord(await fetchJson('api/v1/course-navigation.json'));
    const payloadNode = document.querySelector('#current-lesson-data');
    if (payloadNode?.textContent) {
      const payload = currentLessonRecord(JSON.parse(payloadNode.textContent));
      const phase = loadedCourse.phases.find((item) => item.id === payload.phaseId);
      if (!phase || !phase.lessons.some((lesson) => lesson.id === payload.lesson.id)) {
        throw new Error('Current lesson does not match course navigation');
      }
      phase.lessons = phase.lessons.map((lesson) =>
        lesson.id === payload.lesson.id ? payload.lesson : lesson,
      );
    } else {
      throw new Error('The current lesson was not embedded in this page');
    }
  } else {
    loadedCourse = courseRecord(await fetchJson('curriculum.json'));
  }
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
