function validateLessonProgress(value, lessonIds) {
  return (
    value &&
    value.version === 1 &&
    value.lessons &&
    typeof value.lessons === 'object' &&
    !Array.isArray(value.lessons) &&
    Object.entries(value.lessons).every(
      ([id, entry]) =>
        lessonIds.includes(id) &&
        entry &&
        typeof entry.checkpointPassed === 'boolean' &&
        typeof entry.evidenceSaved === 'boolean',
    )
  );
}

function mergeLessonProgress(current, incoming, lessonIds) {
  if (!validateLessonProgress(incoming, lessonIds)) throw Error('Unsupported lesson progress');
  const merged = { version: 1, lessons: { ...current.lessons } };
  for (const [id, entry] of Object.entries(incoming.lessons)) {
    const previous = merged.lessons[id] || {};
    merged.lessons[id] = {
      checkpointPassed: Boolean(previous.checkpointPassed || entry.checkpointPassed),
      evidenceSaved: Boolean(previous.evidenceSaved || entry.evidenceSaved),
    };
  }
  return merged;
}

const learnerStorageKeys = {
  notebook: 'jaxpathways-notebook-v1',
  lessonProgress: 'jaxpathways-lessons-v1',
  sampleCompleted: 'jaxpathways-first-gradient',
  careerPlan: 'jaxpathways-career-plan-v1',
  reading: 'jaxpathways-reading-v1',
};

const learnerRecord = (value) => value && typeof value === 'object' && !Array.isArray(value);

const learnerId = (value) => typeof value === 'string' && /^[a-z0-9][a-z0-9-]{0,99}$/.test(value);

const learnerDate = (value) =>
  typeof value === 'string' && value.length < 60 && Number.isFinite(Date.parse(value));

function validateLearnerNotebook(value, context) {
  return (
    value &&
    value.version === 1 &&
    context.routeIds.includes(value.route) &&
    Array.isArray(value.evidence) &&
    value.evidence.length <= 500 &&
    value.evidence.every(
      (e) =>
        learnerRecord(e) &&
        typeof e.id === 'string' &&
        e.id.length > 0 &&
        e.id.length < 100 &&
        context.routeIds.includes(e.route) &&
        typeof e.title === 'string' &&
        e.title.length > 0 &&
        e.title.length <= 120 &&
        typeof e.note === 'string' &&
        e.note.length <= 4000 &&
        typeof e.url === 'string' &&
        e.url.length <= 2000 &&
        (!e.url || /^https?:\/\//i.test(e.url)) &&
        learnerDate(e.date) &&
        (!e.updatedAt || learnerDate(e.updatedAt)) &&
        (!e.lessonId || context.lessonIds.includes(e.lessonId)) &&
        (!e.projectId || learnerId(e.projectId)) &&
        (!e.stageId || (e.projectId && typeof e.stageId === 'string' && e.stageId.length <= 100)),
    )
  );
}

function validateReadingState(value, context) {
  if (value === null || value === undefined) return true;
  const position = (v) =>
    learnerRecord(v) &&
    typeof v.sectionId === 'string' &&
    /^[\w-]{0,120}$/.test(v.sectionId) &&
    learnerDate(v.updatedAt);
  return (
    value.version === 1 &&
    context.lessonIds.includes(value.lastLessonId) &&
    (value.path === 'all' || context.routeIds.includes(value.path)) &&
    position(value) &&
    learnerRecord(value.lessons) &&
    Object.entries(value.lessons).length <= 1000 &&
    Object.entries(value.lessons).every(([id, v]) => context.lessonIds.includes(id) && position(v))
  );
}

function validateCareerPlan(value, context) {
  return (
    value === null ||
    value === undefined ||
    (value.version === 1 &&
      context.roleIds.includes(value.roleId) &&
      context.routeIds.includes(value.pathwayId) &&
      context.phaseIds.includes(value.startPhaseId) &&
      Array.isArray(value.knownPhaseIds) &&
      value.knownPhaseIds.every((id) => context.phaseIds.includes(id)))
  );
}

function normalizeLearnerBackup(value, context) {
  // Version 1 is the original notebook export, not a different lesson-progress schema.
  const result =
    value?.version === 1
      ? {
          version: 2,
          notebook: { version: 1, route: value.route, evidence: value.evidence },
          sampleCompleted: value.sampleCompleted,
          lessonProgress: value.lessonProgress || { version: 1, lessons: {} },
          projects: {},
          careerPlan: null,
          reading: null,
        }
      : value;
  if (
    !result ||
    result.version !== 2 ||
    !validateLearnerNotebook(result.notebook, context) ||
    typeof result.sampleCompleted !== 'boolean' ||
    !validateLessonProgress(result.lessonProgress, context.lessonIds) ||
    !learnerRecord(result.projects) ||
    Object.entries(result.projects).length > 100 ||
    !Object.entries(result.projects).every(
      ([id, stages]) =>
        learnerId(id) &&
        Array.isArray(stages) &&
        stages.length <= 100 &&
        stages.every(
          (stage) => typeof stage === 'string' && stage.length > 0 && stage.length <= 100,
        ),
    ) ||
    !validateCareerPlan(result.careerPlan, context) ||
    !validateReadingState(result.reading, context)
  )
    throw Error('Unsupported learner backup');
  return result;
}

function mergeReadingState(current, incoming) {
  if (!incoming) return current || null;
  if (!current) return incoming;
  const latest =
    Date.parse(incoming.updatedAt) >= Date.parse(current.updatedAt) ? incoming : current;
  const lessons = { ...current.lessons };
  for (const [id, entry] of Object.entries(incoming.lessons))
    if (!lessons[id] || Date.parse(entry.updatedAt) >= Date.parse(lessons[id].updatedAt))
      lessons[id] = entry;
  return { ...latest, lessons };
}

function mergeLearnerBackup(current, rawIncoming, context) {
  const incoming = normalizeLearnerBackup(rawIncoming, context);
  const evidence = new Map(current.notebook.evidence.map((e) => [e.id, e]));
  for (const entry of incoming.notebook.evidence) {
    const old = evidence.get(entry.id);
    if (!old || Date.parse(entry.updatedAt || entry.date) > Date.parse(old.updatedAt || old.date))
      evidence.set(entry.id, entry);
  }
  if (evidence.size > 500) throw Error('Evidence limit exceeded');
  const projects = { ...current.projects };
  for (const [id, stages] of Object.entries(incoming.projects))
    projects[id] = [...new Set([...(projects[id] || []), ...stages])];
  return {
    version: 2,
    notebook: { version: 1, route: incoming.notebook.route, evidence: [...evidence.values()] },
    sampleCompleted: current.sampleCompleted || incoming.sampleCompleted,
    lessonProgress: mergeLessonProgress(
      current.lessonProgress,
      incoming.lessonProgress,
      context.lessonIds,
    ),
    projects,
    careerPlan: incoming.careerPlan || current.careerPlan || null,
    reading: mergeReadingState(current.reading, incoming.reading),
  };
}

function persistLearnerBackup(storage, backup) {
  const writes = Object.entries(learnerStorageKeys)
    .filter(([field]) => backup[field] !== null)
    .map(([field, key]) => [
      key,
      field === 'sampleCompleted' ? String(backup[field]) : JSON.stringify(backup[field]),
    ]);
  for (const [id, stages] of Object.entries(backup.projects))
    writes.push(['jaxpathways-project-' + id, JSON.stringify(stages)]);
  const before = new Map();
  try {
    for (const [key] of writes) before.set(key, storage.getItem(key));
    for (const [key, value] of writes) storage.setItem(key, value);
    return true;
  } catch {
    // Best-effort rollback; a disabled storage backend may reject rollback too.
    for (const [key, value] of before) {
      try {
        if (value === null) storage.removeItem(key);
        else storage.setItem(key, value);
      } catch {}
    }
    return false;
  }
}

export {
  validateLessonProgress,
  mergeLessonProgress,
  learnerStorageKeys,
  learnerRecord,
  learnerId,
  learnerDate,
  validateLearnerNotebook,
  validateReadingState,
  validateCareerPlan,
  normalizeLearnerBackup,
  mergeReadingState,
  mergeLearnerBackup,
  persistLearnerBackup,
};
