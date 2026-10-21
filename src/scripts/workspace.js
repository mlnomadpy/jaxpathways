// @ts-check

import { parseProjects } from '../lib/manifests.js';
import { setSelectValue } from './browser-actions.js';
import { fetchJson } from '../lib/http.js';
import { setupBackup } from './notebook/backup.js';
import { setupEvidence } from './notebook/evidence.js';
import { renderNotebookSummary } from './notebook/summary.js';
import {
  learnerStorageKeys,
  validateLearnerNotebook,
  validateCareerPlan,
  validateReadingState,
  learnerId,
} from '../lib/learner-records.js';
import { courseState } from '../lib/course-state.js';
import { $, requireElement } from '../lib/dom.js';
import { setupClubPlanner } from './club-planner.js';
import { setStatus, populateSelect } from './browser-actions.js';
import { readStoredJson } from '../lib/storage.js';

/** @param {string} selector @returns {HTMLSelectElement} */
function requiredSelect(selector) {
  return /** @type {HTMLSelectElement} */ (requireElement(selector));
}

/** @param {unknown} value @returns {value is string[]} */
function stringArray(value) {
  return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

/** @param {import('../types/learner-records').NotebookRoute[]} routes */
async function setupPlatform(routes) {
  const key = learnerStorageKeys.notebook;
  const course = courseState.course;
  if (!course) throw new Error('Course data must load before the learner workspace');
  const allLessons = course.phases.flatMap((p) => p.lessons);
  const context = {
    routeIds: routes.map((r) => r.id),
    lessonIds: allLessons.map((l) => l.id),
    roleIds: course.roles.map((r) => r.id),
    phaseIds: course.phases.map((p) => p.id),
  };
  const read = readStoredJson;
  /** @type {import('../types/learner-records').LearnerNotebook} */
  let state = { version: 1, route: 'foundations', evidence: [] };
  const saved = read(key);
  if (validateLearnerNotebook(saved, context)) state = saved;
  if ($('#club-route')) {
    populateSelect($('#club-route'), routes);
    setSelectValue($('#club-route'), state.route);
    setupClubPlanner(routes);
  }
  if (!$('#active-route')) return;

  /** @type {import('../types/manifests').ProjectManifest[]} */
  let projects = [];
  try {
    projects = parseProjects(await fetchJson('api/v1/projects.json', { cache: 'no-store' }));
  } catch {
    setStatus(
      '#evidence-status',
      'Project associations could not load. You can still save notes; refresh to attach a project.',
    );
  }
  /** @type {import('../types/learner-records').LearnerBackup | null} */
  let pendingExtras = null;
  const status = setStatus;
  function storedProjects() {
    /** @type {Record<string, string[]>} */
    const values = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const storageKey = localStorage.key(i);
        if (storageKey?.startsWith('jaxpathways-project-')) {
          const id = storageKey.slice('jaxpathways-project-'.length);
          const stages = read(storageKey, []);
          if (learnerId(id) && stringArray(stages)) values[id] = stages;
        }
      }
    } catch {}
    return values;
  }
  /** @returns {import('../types/learner-records').LearnerBackup} */
  function backup() {
    const careerCandidate = pendingExtras
      ? pendingExtras.careerPlan
      : read(learnerStorageKeys.careerPlan);
    const readingCandidate = pendingExtras
      ? pendingExtras.reading
      : read(learnerStorageKeys.reading);
    return {
      version: 2,
      notebook: state,
      sampleCompleted: courseState.completed,
      // course-state validates persisted lesson progress against version 1 before loading it.
      lessonProgress: /** @type {import('../types/learner-records').LessonProgressState} */ (
        courseState.lessonProgress
      ),
      projects: pendingExtras?.projects || storedProjects(),
      careerPlan: validateCareerPlan(careerCandidate, context) ? careerCandidate || null : null,
      reading: validateReadingState(readingCandidate, context) ? readingCandidate || null : null,
    };
  }
  function save() {
    try {
      localStorage.setItem(key, JSON.stringify(state));
      status('#backup-status', '');
      return true;
    } catch {
      status(
        '#backup-status',
        'Device storage could not save these changes. They remain in this page only. Export a backup before closing.',
      );
      return false;
    }
  }
  const populate = populateSelect;
  for (const selector of ['#active-route', '#evidence-route', '#evidence-filter-route'])
    populate(
      requiredSelect(selector),
      routes,
      selector === '#evidence-filter-route' ? 'All pathways' : null,
    );
  const activeRoute = requiredSelect('#active-route');
  setSelectValue(activeRoute, state.route);
  requireElement('#continue').onclick = () => {
    location.href = `course.html?path=${encodeURIComponent(state.route)}`;
  };
  activeRoute.onchange = () => {
    state.route = activeRoute.value;
    save();
    routeSummary();
  };
  const routeSummary = () =>
    renderNotebookSummary({ routes, projects, allLessons, context, state, backup });
  const evidence = setupEvidence({
    routes,
    projects,
    allLessons,
    context,
    getState: () => state,
    save,
    backup,
  });
  setupBackup({
    context,
    getBackup: backup,
    applyBackup(merged) {
      state = merged.notebook;
      courseState.lessonProgress = merged.lessonProgress;
      courseState.completed = merged.sampleCompleted;
      courseState.readingState = merged.reading || { version: 1, lessons: {} };
      pendingExtras = merged;
      setSelectValue(activeRoute, state.route);
      evidence.clearForm();
      evidence.renderEvidence();
      routeSummary();
    },
  });
  routeSummary();
}

export { setupPlatform };
