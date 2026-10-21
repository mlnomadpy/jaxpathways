import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Window } from 'happy-dom';
import { initReadingTools } from '../src/scripts/reading-tools.js';
import { initCatalog, initProjectCatalog, restoreCourseView } from '../src/scripts/catalog.js';
import { courseState } from '../src/lib/course-state.js';

function bindWindow(window) {
  for (const key of [
    'document',
    'location',
    'history',
    'navigator',
    'HTMLElement',
    'Event',
    'KeyboardEvent',
  ]) {
    Object.defineProperty(globalThis, key, {
      value: window[key],
      configurable: true,
      writable: true,
    });
  }
  globalThis.window = window;
}

test('reading tools leave project catalog stage summaries alone and name workspace stage links', async () => {
  const window = new Window();
  bindWindow(window);
  try {
    window.document.body.innerHTML = `
      <article class="project-entry"><div class="project-stages"><h4>Your stages</h4><ol><li>Audit setup</li></ol></div></article>
      <main id="project"><ol class="project-stages"><li id="stage-1"><h2>Check the baseline</h2></li><li id="stage-2"><h2>Verify a changed condition</h2></li></ol></main>`;
    initReadingTools();
    assert.equal(window.document.querySelectorAll('.project-stage-links').length, 1);
    const nav = window.document.querySelector('#project .project-stage-links');
    assert.deepEqual(
      [...nav.querySelectorAll('a')].map((link) => [link.textContent, link.getAttribute('href')]),
      [
        ['Check the baseline', '#stage-1'],
        ['Verify a changed condition', '#stage-2'],
      ],
    );
    assert.equal(window.document.querySelector('.project-entry .project-stage-links'), null);
    initReadingTools();
    assert.equal(window.document.querySelectorAll('#project .project-stage-links').length, 1);
  } finally {
    await window.happyDOM.abort();
  }
});

test('project stage chips filter all registered project cards and preserve a shareable URL', async () => {
  const window = new Window({ url: 'https://example.org/projects.html' });
  bindWindow(window);
  try {
    window.document.body.innerHTML = `
      <section id="available">
        <nav><button data-project-filter="all" aria-pressed="true">All</button><button data-project-filter="foundations" aria-pressed="false">Foundations</button></nav>
        <section data-project-group-section="foundations"><article data-project-id="regression-audit" data-project-group="foundations"></article></section>
        <section data-project-group-section="models"><article data-project-id="mlp-classifier" data-project-group="models"></article></section>
      </section>`;
    initProjectCatalog();
    window.document.querySelector('[data-project-filter="foundations"]').click();
    assert.equal(
      window.document.querySelector('[data-project-id="regression-audit"]').hidden,
      false,
    );
    assert.equal(window.document.querySelector('[data-project-id="mlp-classifier"]').hidden, true);
    assert.equal(
      window.document
        .querySelector('[data-project-filter="foundations"]')
        .getAttribute('aria-pressed'),
      'true',
    );
    assert.equal(window.location.search, '?group=foundations');
    assert.equal(window.document.querySelectorAll('[data-project-id]').length, 2);
    window.history.replaceState(null, '', 'projects.html');
    window.dispatchEvent(new window.PopStateEvent('popstate'));
    assert.equal(window.document.querySelector('[data-project-id="mlp-classifier"]').hidden, false);
  } finally {
    await window.happyDOM.abort();
  }
});

test('career milestones cover each default route and README counts match current registries', () => {
  const course = JSON.parse(fs.readFileSync('curriculum/course.json', 'utf8'));
  for (const role of course.roles) {
    const route = JSON.parse(
      fs.readFileSync(`learning-paths/${role.defaultPathwayId}.json`, 'utf8'),
    );
    const covered = new Set(role.milestones.flatMap((milestone) => milestone.phaseIds));
    assert.deepEqual(
      route.phaseIds.filter((phaseId) => !covered.has(phaseId)),
      [],
      `${role.id} milestones`,
    );
  }
  const assessments = JSON.parse(fs.readFileSync('curriculum/assessments.json', 'utf8'));
  const plans = JSON.parse(fs.readFileSync('curriculum/project-plans.json', 'utf8'));
  const readme = fs.readFileSync('README.md', 'utf8');
  assert.match(
    readme,
    new RegExp(
      `Current registry: \\*\\*${course.phases.length} phases, ${course.phases.flatMap((phase) => phase.lessons).filter((lesson) => lesson.status === 'authored').length} authored lessons, ${course.pathwayIds.length} pathways, ${course.roles.length} career routes, ${course.projectIds.length} registered practice projects, ${assessments.length} synthesis assessments, and ${plans.length} planned model-lifecycle capstones\\.\\*\\*`,
    ),
  );
  const renderScript = fs.readFileSync('scripts/render-curriculum.mjs', 'utf8');
  assert.match(renderScript, /COURSE-STATUS:START/);
  assert.match(renderScript, /assessments\.json/);
});

test('optional lesson search opens on demand, applies URL queries and restores direct links', async () => {
  const window = new Window({ url: 'https://example.org/course.html?q=checkpoint' });
  bindWindow(window);
  try {
    window.document.body.innerHTML = `
      <main id="curriculum">
        <p id="counts"></p><p id="progress"></p><a id="course-start"></a>
        <div id="course-scroll-bar"><strong id="active-era-indicator"></strong></div>
        <div id="path-summary"></div><div id="course-view-status"></div>
        <div id="phases">
          <div data-era-banner="era-onboarding"></div>
          <details id="phase-welcome" class="phase-card" data-course-era="era-onboarding" data-era-title="Start Here">
            <summary><h2 class="phase-title">Setup</h2></summary>
            <li id="welcome-01"><a class="lesson-link">Setup lesson</a><p class="search-excerpt" hidden><a></a></p></li>
          </details>
        </div>
        <details id="optional-tpu-track"><details id="phase-tpu" class="phase-card"><li id="tpu-01"></li></details></details>
        <div id="course-jumps"></div><input id="show-roadmap" type="checkbox">
        <p id="empty" hidden></p><p id="content-status"></p>
        <button id="course-search-open" type="button">Search lessons</button><button id="course-search-clear" type="button" hidden>Clear search</button>
        <dialog id="course-search-dialog">
          <form method="dialog"><button id="course-search-close" type="submit">Close</button>
            <label for="course-search-input">Search lesson explanations, code and practice</label><input id="course-search-input" type="search">
            <button id="course-search-apply" type="button">Show matches</button><button id="course-search-cancel" type="button">Cancel</button>
          </form>
        </dialog>
        <button data-course-filter="all"></button>
      </main>`;
    const lesson = {
      id: 'welcome-01',
      title: 'Set up your learning workspace',
      status: 'authored',
      minutes: 20,
      objective: 'Identify the local device.',
      content: { problem: 'Keep a checkpoint and replay it after a restart.' },
    };
    const phase = {
      id: 'welcome',
      title: 'Setup & first steps',
      part: 'Onboarding',
      project: 'Set up JAX',
      lessons: [lesson],
    };
    Object.assign(courseState, {
      course: { phases: [phase], projectIds: [] },
      curriculum: [],
      activePath: 'all',
      courseQuery: '',
      courseHardware: 'all',
      courseTypeFilter: 'all',
      readingState: { version: 1, lessons: {} },
      lessonProgress: { version: 1, lessons: {} },
    });
    initCatalog();
    assert.equal(window.document.querySelector('#welcome-01 .search-excerpt').hidden, false);
    assert.match(
      window.document.querySelector('#welcome-01 .search-excerpt').textContent,
      /checkpoint/,
    );

    window.document.querySelector('#course-search-open').click();
    const dialog = window.document.querySelector('#course-search-dialog');
    assert.equal(dialog.open, true);
    const input = window.document.querySelector('#course-search-input');
    input.value = 'restart';
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
    window.document.querySelector('#course-search-apply').click();
    assert.equal(new URLSearchParams(window.location.search).get('q'), 'restart');
    assert.equal(dialog.open, false);
    assert.match(
      window.document.querySelector('#welcome-01 .search-excerpt').textContent,
      /restart/,
    );

    window.history.replaceState(null, '', 'course.html?q=checkpoint');
    restoreCourseView();
    assert.equal(input.value, 'checkpoint');
    assert.equal(window.document.querySelector('#welcome-01').hidden, false);

    window.document.body.focus();
    window.document.dispatchEvent(new window.KeyboardEvent('keydown', { key: '/', bubbles: true }));
    assert.equal(dialog.open, true);
    window.document.querySelector('#course-search-cancel').click();
    assert.equal(dialog.open, false);
    assert.equal(input.value, 'checkpoint');
  } finally {
    await window.happyDOM.abort();
  }
});
