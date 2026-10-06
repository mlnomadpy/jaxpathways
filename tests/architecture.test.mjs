import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

test('browser modules have an acyclic dependency graph and domain libraries do not import controllers', () => {
  const files = fs
    .readdirSync('src', { recursive: true })
    .filter((file) => file.endsWith('.js'))
    .map((file) => path.resolve('src', file));
  const graph = new Map();
  for (const file of files) {
    const source = fs.readFileSync(file, 'utf8');
    const edges = [...source.matchAll(/from\s+['"](\.[^'"]+)['"]/g)]
      .map((match) => path.resolve(path.dirname(file), match[1]))
      .filter((dependency) => dependency.endsWith('.js'));
    if (file.includes('/src/lib/'))
      for (const edge of edges)
        assert(
          !edge.includes('/src/scripts/'),
          `domain library imports a browser controller: ${file} → ${edge}`,
        );
    graph.set(file, edges);
  }
  const complete = new Set(),
    active = [];
  function visit(file) {
    assert(
      !active.includes(file),
      `circular module dependency: ${[...active, file].map((file) => path.relative(process.cwd(), file)).join(' → ')}`,
    );
    if (complete.has(file)) return;
    active.push(file);
    for (const dependency of graph.get(file) || []) visit(dependency);
    active.pop();
    complete.add(file);
  }
  files.forEach(visit);
});

test('browser entry points cannot reach server-only renderers or Node built-ins', () => {
  const visited = new Set();
  function visit(file) {
    if (visited.has(file)) return;
    visited.add(file);
    assert(!file.includes('/lib/server/'), `server-only module reached from browser: ${file}`);
    const source = fs.readFileSync(file, 'utf8');
    for (const [, specifier] of source.matchAll(/(?:from\s+|import\s*)['"]([^'"]+)['"]/g)) {
      assert(
        !specifier.startsWith('node:'),
        `Node built-in reached from browser: ${file}: ${specifier}`,
      );
      if (specifier.startsWith('.') && specifier.endsWith('.js'))
        visit(path.resolve(path.dirname(file), specifier));
    }
  }
  for (const file of fs.readdirSync('src/scripts/entries'))
    visit(path.resolve('src/scripts/entries', file));
});
