import { parseProjects } from '../lib/manifests.js';
import { setSelectValue } from './browser-actions.js';
import { fetchJson } from '../lib/http.js';
import { setupBackup } from './notebook/backup.js';
import { setupEvidence } from './notebook/evidence.js';
import { renderNotebookSummary } from './notebook/summary.js';
import { learnerStorageKeys, validateLearnerNotebook, learnerId } from '../lib/learner-records.js';
import { courseState } from '../lib/course-state.js';
import { $ } from '../lib/dom.js';
import { setupClubPlanner } from './club-planner.js';
import { setStatus, populateSelect } from './browser-actions.js';
import { readStoredJson } from '../lib/storage.js';

async function setupPlatform(routes) {
  const key = learnerStorageKeys.notebook;
  const allLessons = courseState.course.phases.flatMap((p) => p.lessons);
  const context = {
    routeIds: routes.map((r) => r.id),
    lessonIds: allLessons.map((l) => l.id),
    roleIds: courseState.course.roles.map((r) => r.id),
    phaseIds: courseState.course.phases.map((p) => p.id),
  };
  const read = readStoredJson;
  let state = { version: 1, route: 'foundations', evidence: [] };
  const saved = read(key);
  if (validateLearnerNotebook(saved, context)) state = saved;
  if ($('#club-route')) {
    populateSelect($('#club-route'), routes);
    setSelectValue($('#club-route'), state.route);
    setupClubPlanner(routes);
  }
  if (!$('#active-route')) return;

  let projects = [];
  try {
    projects = parseProjects(await fetchJson('api/v1/projects.json', { cache: 'no-store' }));
  } catch {
    setStatus(
      '#evidence-status',
      'Project associations could not load. You can still save notes; refresh to attach a project.',
    );
  }
  let pendingExtras = null;
  const status = setStatus;
  function storedProjects() {
    const values = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const storageKey = localStorage.key(i);
        if (storageKey?.startsWith('jaxpathways-project-')) {
          const id = storageKey.slice('jaxpathways-project-'.length);
          const stages = read(storageKey, []);
          if (learnerId(id) && Array.isArray(stages) && stages.every((s) => typeof s === 'string'))
            values[id] = stages;
        }
      }
    } catch {}
    return values;
  }
  function backup() {
    return {
      version: 2,
      notebook: state,
      sampleCompleted: courseState.completed,
      lessonProgress: courseState.lessonProgress,
      projects: pendingExtras?.projects || storedProjects(),
      careerPlan: pendingExtras?.careerPlan || read(learnerStorageKeys.careerPlan),
      reading: pendingExtras?.reading || read(learnerStorageKeys.reading),
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
    populate($(selector), routes, selector === '#evidence-filter-route' ? 'All pathways' : null);
  setSelectValue($('#active-route'), state.route);
  $('#continue').onclick = () => {
    location.href = `course.html?path=${encodeURIComponent(state.route)}`;
  };
  $('#active-route').onchange = (e) => {
    state.route = e.target.value;
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
      setSelectValue($('#active-route'), state.route);
      evidence.clearForm();
      evidence.renderEvidence();
      routeSummary();
    },
  });
  routeSummary();
}

export { setupPlatform };
