// @ts-check

/** @param {unknown} value
 * @returns {value is Record<string, unknown>}
 */
const record = (value) => typeof value === 'object' && value !== null && !Array.isArray(value);

/** @param {unknown} value
 * @returns {value is string[]}
 */
const strings = (value) => Array.isArray(value) && value.every((item) => typeof item === 'string');

/** @param {unknown} value
 * @param {string[]} keys
 */
const textFields = (value, keys) =>
  record(value) && keys.every((key) => typeof value[key] === 'string');

/** @param {unknown} value
 * @returns {value is import('../types/manifests').ProjectManifest}
 */
function project(value) {
  const environment = record(value) ? value.environment : undefined;
  const environmentIsValid = validEnvironment(environment);
  return (
    record(value) &&
    typeof value.schemaVersion === 'number' &&
    textFields(value, [
      'id',
      'title',
      'pathwayId',
      'hardware',
      'source',
      'setupNote',
      'workspaceFile',
    ]) &&
    strings(value.requiredLessonIds) &&
    value.requiredLessonIds.length > 0 &&
    strings(value.rubric) &&
    Array.isArray(value.stages) &&
    value.stages.length > 0 &&
    value.stages.every(
      (stage) =>
        textFields(stage, ['id', 'title', 'command', 'evidence', 'expected', 'diagnosis']) &&
        (stage.workspaceFile === undefined ||
          (typeof stage.workspaceFile === 'string' && stage.workspaceFile.trim().length > 0)) &&
        (stage.codeGuide === undefined || typeof stage.codeGuide === 'string') &&
        (stage.starterSnippet === undefined || typeof stage.starterSnippet === 'string') &&
        (stage.solutionSnippet === undefined || typeof stage.solutionSnippet === 'string'),
    ) &&
    (value.referenceCommand === undefined || typeof value.referenceCommand === 'string') &&
    (value.assessmentId === undefined || typeof value.assessmentId === 'string') &&
    (value.prepare === undefined ||
      (Array.isArray(value.prepare) &&
        value.prepare.every((item) => textFields(item, ['command', 'label'])))) &&
    environmentIsValid
  );
}

/** @param {unknown} value */
function validEnvironment(value) {
  if (
    !record(value) ||
    typeof value.requirementsPath !== 'string' ||
    typeof value.pythonCompatibility !== 'string'
  ) {
    return false;
  }
  const requirementsPath = value.requirementsPath;
  if (
    !/^(?:requirements-[a-z0-9-]+\.txt|projects\/[a-z0-9-]+\/requirements\.txt)$/.test(
      requirementsPath,
    ) ||
    value.pythonCompatibility !== '>=3.11,<3.15' ||
    !record(value.setupCommands) ||
    typeof value.setupCommands.unix !== 'string' ||
    typeof value.setupCommands.windows !== 'string'
  ) {
    return false;
  }
  const { unix, windows } = value.setupCommands;
  return (
    unix.includes(requirementsPath) &&
    windows.includes(requirementsPath) &&
    unix.includes('source .venv/bin/activate') &&
    windows.includes('.venv\\Scripts\\python.exe')
  );
}

/** @param {unknown} value
 * @returns {import('../types/manifests').ProjectManifest[]}
 */
export function parseProjects(value) {
  if (!record(value) || !Array.isArray(value.projects) || !value.projects.every(project)) {
    throw new Error('Unsupported project registry');
  }
  return value.projects;
}

/** @param {unknown} value
 * @returns {value is import('../types/manifests').ValidationReceipt}
 */
function receipt(value) {
  return (
    record(value) &&
    textFields(value, ['python', 'jax', 'numpy']) &&
    Array.isArray(value.lessons) &&
    value.lessons.every(
      (lesson) =>
        record(lesson) &&
        textFields(lesson, ['lessonId', 'contentHash']) &&
        (lesson.runtime === undefined ||
          (record(lesson.runtime) && typeof lesson.runtime.deviceCount === 'number')),
    )
  );
}

/** @param {unknown} value @returns {import('../types/manifests').ValidationReceipt} */
export function parseValidation(value) {
  if (!receipt(value)) throw new Error('Unsupported validation receipt');
  return value;
}
