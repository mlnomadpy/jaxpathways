import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';

const read = path => JSON.parse(readFileSync(path, 'utf8'));
const stable = value => Array.isArray(value) ? value.map(stable) : value && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])])) : value;
const hash = value => createHash('sha256').update(JSON.stringify(stable(value))).digest('hex');
const notes = read('curriculum/releases.json');
const release = notes[0];
assert.match(release.version, /^\d+\.\d+\.\d+$/);
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
  for (const file of readdirSync(directory, { recursive: true }).sort()) {
    if (!file.endsWith('.json') || /(?:^|\/)(?:releases|release-snapshots|validation)\.json$/.test(file) || file.includes('/outputs/')) continue;
    sources[`${directory}/${file}`] = hash(read(`${directory}/${file}`));
  }
}
// Practical guides and their launchers are canonical course material too.
for (const directory of ['content/guides', 'resources/tpu-gcp']) {
  if (!existsSync(directory)) continue;
  for (const file of readdirSync(directory, { recursive: true }).sort()) {
    if (!/\.(md|py)$/.test(file) || file.includes('__pycache__')) continue;
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
