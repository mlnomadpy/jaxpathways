import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { portablePath } from './portable-path.mjs';

const read = path => {
  const source = readFileSync(path, 'utf8');
  try {
    return JSON.parse(source);
  } catch (error) {
    throw new SyntaxError(`Invalid canonical JSON in ${path}: ${error.message}`);
  }
};
const stable = value => Array.isArray(value) ? value.map(stable) : value && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])])) : value;
const hash = value => createHash('sha256').update(JSON.stringify(stable(value))).digest('hex');
// This published canary intentionally appends corrupt bytes to a tokenizer file.
// Track its exact bytes: treating other malformed JSON as a canary would hide source errors.
const negativeJsonFixtures = new Set([
  'projects/text-harness/foundations/reference/run/corrupt-package/tokenizer.json',
]);

const releaseVersionPattern = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
const modifiedAtPattern = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$/;
function validateReleaseIdentity(release) {
  const match = typeof release.version === 'string' ? releaseVersionPattern.exec(release.version) : null;
  assert(match, 'Release version must be plain semantic major.minor.patch without leading zeroes or suffixes');
  const [major, minor] = match.slice(1, 3).map(Number);
  const hasEditionKey = Object.hasOwn(release, 'editionKey');
  const hasModifiedAt = Object.hasOwn(release, 'modifiedAt');
  assert.equal(hasEditionKey, hasModifiedAt, 'Release editionKey and modifiedAt must be provided together');
  if (!hasEditionKey) {
    assert(!(major > 0 || minor >= 15), 'Release version 0.15.0 and later requires editionKey and modifiedAt');
    return;
  }
  assert(typeof release.editionKey === 'string' && release.editionKey.length > 0
    && release.editionKey.trim() === release.editionKey, 'Release editionKey must be a non-empty trimmed string');
  const dateMatch = typeof release.modifiedAt === 'string' ? modifiedAtPattern.exec(release.modifiedAt) : null;
  assert(dateMatch, 'Release modifiedAt must use UTC YYYY-MM-DDThh:mm:ssZ');
  const [year, month, day, hour, minute, second] = dateMatch.slice(1).map(Number);
  const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const daysInMonth = [31, leapYear ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  assert(year > 0 && month >= 1 && month <= 12 && day >= 1 && day <= daysInMonth[month - 1]
    && hour <= 23 && minute <= 59 && second <= 59,
  'Release modifiedAt must be a valid UTC calendar date and time');
}


const notes = read('curriculum/releases.json');
const release = notes[0];
validateReleaseIdentity(release);
assert.match(release.date, /^\d{4}-\d{2}-\d{2}$/);
assert.equal(new Set(notes.map(note => note.version)).size, notes.length, 'Release versions must be unique');
assert.equal(read('package.json').version, release.version, 'Package and release versions must agree');
const filename = 'curriculum/release-snapshots.json';
const snapshots = existsSync(filename) ? read(filename) : [];
const previous = snapshots[0];
const course = read('curriculum/course.json');
const lessons = {};
const phases = {};
const revision = (title, digest, old) => ({ title, hash: digest,
  version: old?.hash === digest ? old.version : release.version,
  date: old?.hash === digest ? old.date : release.date });
for (const phase of course.phases) {
  const phaseLessons = {};
  for (const lesson of phase.lessons) {
    const content = lesson.status === 'authored' ? read(`${lesson.path}/lesson.json`) : null;
    lessons[lesson.id] = revision(lesson.title, hash({ lesson, content }), previous?.lessons[lesson.id]);
    phaseLessons[lesson.id] = lessons[lesson.id].hash;
  }
  phases[phase.id] = revision(phase.title, hash({ phase, phaseLessons }), previous?.phases[phase.id]);
}
// Include project and pathway sources: edits outside lesson prose also require a release note.
const sources = {};
for (const directory of ['curriculum', 'learning-paths', 'projects']) {
  for (const file of readdirSync(directory, { recursive: true }).map(portablePath).sort()) {
    if (!file.endsWith('.json') || /(?:^|\/)(?:releases|release-snapshots|(?:project-)?validation)\.json$/.test(file) || /(?:^|\/)(?:reference|outputs|runs|evidence)\//.test(file)) continue;
    sources[`${directory}/${file}`] = hash(read(`${directory}/${file}`));
  }
}
// Retain the release-bound identity of the one explicitly declared negative canary,
// without changing the policy that generated reference/run/evidence JSON is excluded.
for (const file of negativeJsonFixtures) {
  if (existsSync(file)) sources[file] = createHash('sha256').update(readFileSync(file)).digest('hex');
}
// Practical guides and their launchers are canonical course material too.
const projectDirectories = read('curriculum/projects.json').map(file => file.replace(/\/project\.json$/, ''));
const planDirectories = existsSync('curriculum/project-plans.json')
  ? read('curriculum/project-plans.json').map(plan => plan.source.slice(0, plan.source.lastIndexOf('/'))) : [];
for (const directory of new Set(['content/guides', 'resources/tpu-gcp', 'assessments', ...projectDirectories, ...planDirectories])) {
  if (!existsSync(directory)) continue;
  for (const file of readdirSync(directory, { recursive: true }).map(portablePath).sort()) {
    if (!/\.(md|py|txt|jsonl|csv|yaml|yml)$/.test(file) || file.includes('__pycache__') || file.startsWith('reference/') || /^my_.*\.py$/.test(file) || /^(?:\.venv|runs|evidence|outputs)\//.test(file)) continue;
    sources[`${directory}/${file}`] = hash(readFileSync(`${directory}/${file}`, 'utf8'));
  }
}
const digest = hash({ sources, lessons: Object.fromEntries(Object.entries(lessons).map(([id, item]) => [id, item.hash])) });
if (previous?.version === release.version) {
  assert.equal(previous.hash, digest, 'Course sources changed. Add a new version to curriculum/releases.json, update package version, then run npm run release:prepare. Do not overwrite a published snapshot.');
  assert.equal(hash(previous.notes), hash(release), 'Published release notes are immutable; add a new release.');
} else {
  assert(process.argv.includes('--prepare'), 'New release needs a snapshot: npm run release:prepare');
  if (previous) assert(release.date >= previous.date, 'Release dates must be chronological');
  snapshots.unshift({ version: release.version, date: release.date, hash: digest, notes: release,
    lessons, phases,
    changedLessons: Object.keys(lessons).filter(id => previous?.lessons[id]?.hash !== lessons[id].hash),
    changedPhases: Object.keys(phases).filter(id => previous?.phases[id]?.hash !== phases[id].hash),
    removedLessons: Object.keys(previous?.lessons || {}).filter(id => !lessons[id]) });
  writeFileSync(filename, JSON.stringify(snapshots, null, 2) + '\n');
}
console.log(`Release ${release.version}: ${Object.keys(lessons).length} lessons and ${Object.keys(phases).length} phases match the recorded source snapshot.`);
