import { existsSync, readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';

// Keep every Node test in exactly one suite. Selection fails closed if a new file is unassigned.
export const suites = {
  unit: [
    'scripts/code-highlight.test.cjs',
    'scripts/evidence-review.test.cjs',
    'scripts/learner-backup.test.cjs',
    'scripts/lesson-progress.test.cjs',
    'scripts/organizer-schedule.test.cjs',
    'scripts/platform-form.test.cjs',
    'tests/architecture.test.mjs',
    'tests/assessment-content-images.test.mjs',
    'tests/book-prose-links.test.mjs',
    'tests/browser-actions.test.mjs',
    'tests/choices.test.mjs',
    'tests/code-teaching-api.test.mjs',
    'tests/lesson-prerequisites.test.mjs',
    'tests/native-book-contracts.test.mjs',
    'tests/portable-path.test.mjs',
    'tests/radio-group.test.mjs',
    'tests/release-edition-metadata.test.mjs',
    'tests/release-receipts.test.mjs',
    'tests/releases.test.mjs',
    'tests/style-architecture.test.mjs',
    'tests/test-selection.test.mjs',
  ],
  app: [
    'scripts/career.test.cjs',
    'tests/book-repairs.test.mjs',
    'tests/discovery-fixes.test.mjs',
    'tests/http.test.mjs',
    'tests/math.test.mjs',
    'tests/notebook-records.test.mjs',
    'tests/view-rendering.test.mjs',
  ],
  content: [
    'scripts/course-resources.test.cjs',
    'tests/distribution-posttraining-audit.test.mjs',
    'tests/experiments-workshop.test.mjs',
    'tests/guided-reasoning.test.mjs',
    'tests/hardware-evidence.test.mjs',
    'tests/learner-starters.test.mjs',
    'tests/lesson-visuals.test.mjs',
    'tests/material-status.test.mjs',
    'tests/pathway-guides.test.mjs',
    'tests/plot-interpretation.test.mjs',
    'tests/posttraining-numeric-packet.test.mjs',
    'tests/project-stage-files.test.mjs',
    'tests/project-workflow.test.mjs',
    'tests/tpu-guide.test.mjs',
  ],
  site: [
    'tests/book-print-qualification.test.mjs',
    'tests/browser-migration.test.mjs',
    'tests/c01-optional-pilot.test.mjs',
    'tests/c02-optional-pilot.test.mjs',
    'tests/c03-canonical-pilot-discovery.test.mjs',
    'tests/checkpoint-feedback.test.mjs',
    'tests/course-reader.test.mjs',
    'tests/image-pilot-evidence.test.mjs',
    'tests/optional-workflow-evidence.test.mjs',
    'tests/posttraining-transformer-lineage-guide.test.mjs',
    'tests/reader-guidance.test.mjs',
  ],
};

const allTests = () => Object.values(suites).flat().sort();
const generatedSuites = ['app', 'content', 'site'];
const inDirectory = (file, directory) => file.startsWith(`${directory}/`);

export function validateInventory(root = process.cwd()) {
  const assigned = allTests();
  const discover = (directory) =>
    readdirSync(path.join(root, directory), { withFileTypes: true }).flatMap((entry) => {
      const file = `${directory}/${entry.name}`;
      return entry.isDirectory() ? discover(file) : /\.test\.(?:cjs|mjs)$/.test(file) ? [file] : [];
    });
  const discovered = ['scripts', 'tests'].flatMap(discover).sort();
  const missing = discovered.filter((file) => !assigned.includes(file));
  const stale = assigned.filter((file) => !discovered.includes(file));
  if (new Set(assigned).size !== assigned.length || missing.length || stale.length) {
    throw new Error(
      `Update scripts/test-suites.mjs: unassigned [${missing}], missing [${stale}], or duplicate tests.`,
    );
  }
}

// Follow literal local imports/requires (including dynamic imports) through the source graph.
// Non-module inputs such as generated HTML, Python subprocesses and registries are handled below.
export function dependencyReader(root = process.cwd()) {
  const cache = new Map();
  return function dependencies(file, seen = new Set()) {
    if (seen.has(file)) return seen;
    seen.add(file);
    if (!cache.has(file)) {
      const target = path.join(root, file);
      const imports = [];
      if (existsSync(target) && /\.(?:[cm]?js|json)$/.test(file)) {
        const source = readFileSync(target, 'utf8');
        const pattern = /(?:\bfrom\s*|\bimport\s*(?:\(\s*)?|\brequire\s*\(\s*)['"]([^'"]+)['"]/g;
        for (const [, specifier] of source.matchAll(pattern)) {
          if (specifier.startsWith('.')) {
            const resolved = path.posix.normalize(
              path.posix.join(path.posix.dirname(file), specifier),
            );
            imports.push(resolved);
          }
        }
      }
      cache.set(file, imports);
    }
    for (const imported of cache.get(file)) dependencies(imported, seen);
    return seen;
  };
}

export function selectTests({ suite, changedFiles, root = process.cwd() } = {}) {
  if (suite !== undefined && suite !== 'all' && !Object.hasOwn(suites, suite)) {
    throw new Error(
      `Unknown suite "${suite}". Choose ${[...Object.keys(suites), 'all'].join(', ')}.`,
    );
  }
  const candidates = suite && suite !== 'all' ? suites[suite] : allTests();
  if (changedFiles === undefined) return [...candidates].sort();
  const selected = new Set();
  const add = (files) => files.forEach((file) => selected.add(file));
  const addSuites = (names) => names.forEach((name) => add(suites[name]));
  const dependencies = dependencyReader(root);
  const dependencyGraph = new Map(allTests().map((test) => [test, dependencies(test)]));
  for (const input of changedFiles) {
    const file = input.replaceAll('\\', '/').replace(/^\.\//, '');
    const dependents = allTests().filter((test) => dependencyGraph.get(test).has(file));
    add(dependents);
    if (/\.test\.(?:cjs|mjs)$/.test(file)) continue;
    if (file === 'scripts/release.mjs' || file === 'scripts/portable-path.mjs') {
      // These checks copy or execute the release script instead of importing it.
      add([
        'tests/releases.test.mjs',
        'tests/release-receipts.test.mjs',
        'tests/release-edition-metadata.test.mjs',
      ]);
      continue;
    }
    if (inDirectory(file, 'src')) {
      // Static pages, CSS and browser entry wiring are read by build/output tests, not imported.
      add(suites.site);
      add(['tests/architecture.test.mjs', 'tests/style-architecture.test.mjs']);
      if (!dependents.length || inDirectory(file, 'src/data')) addSuites(generatedSuites);
      continue;
    }
    if (
      [
        'curriculum',
        'phases',
        'learning-paths',
        'projects',
        'resources',
        'assessments',
        'content',
        'skills',
        'public',
        'dist',
      ].some((directory) => inDirectory(file, directory))
    ) {
      addSuites(generatedSuites);
      continue;
    }
    if (file === 'README.md' || inDirectory(file, 'docs')) {
      add(suites.content);
      add(['tests/discovery-fixes.test.mjs']);
      // Book/project guides can embed documentation and evidence links.
      add(suites.site);
      continue;
    }
    if (inDirectory(file, 'browser-tests')) continue; // Separate Playwright suite.
    if (file === 'tests/test_material_audit.py') {
      add(['tests/material-status.test.mjs']);
      continue;
    }
    if (file === 'tests/test_python_runners.py') continue; // npm run test:tools
    // Tooling, unknown paths and dependency/config changes conservatively select everything.
    add(allTests());
  }
  return candidates.filter((file) => selected.has(file)).sort();
}

export function preparationFor(files) {
  if (files.some((file) => suites.site.includes(file))) return 'site';
  if (files.some((file) => [...suites.app, ...suites.content].includes(file))) return 'content';
  return 'none';
}
